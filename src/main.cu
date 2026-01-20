#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "flash_attention.cuh"

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                         \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)        \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

template <typename T>
T to_device(float v);

template <>
half to_device<half>(float v) {
    return __float2half_rn(v);
}

template <>
nv_bfloat16 to_device<nv_bfloat16>(float v) {
    return __float2bfloat16_rn(v);
}

template <typename T>
float to_float(T v);

template <>
float to_float<half>(half v) {
    return __half2float(v);
}

template <>
float to_float<nv_bfloat16>(nv_bfloat16 v) {
    return __bfloat162float(v);
}

struct ErrorStats {
    float max_abs_err = 0.0f;
    float mean_abs_err = 0.0f;
    float rmse = 0.0f;
};

template <typename value_t>
std::vector<float> to_float_vec(const std::vector<value_t> &src) {
    std::vector<float> out(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        out[i] = to_float<value_t>(src[i]);
    }
    return out;
}

std::vector<float> attention_cpu_ref(const std::vector<float> &Q,
                                     const std::vector<float> &K,
                                     const std::vector<float> &V, int seq_len,
                                     int d_head) {
    std::vector<float> O(Q.size(), 0.0f);
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d_head));

    for (int i = 0; i < seq_len; ++i) {
        const int64_t q_base = static_cast<int64_t>(i) * d_head;

        std::vector<float> scores(seq_len);
        float row_max = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < seq_len; ++j) {
            const int64_t k_base = static_cast<int64_t>(j) * d_head;
            float dot = 0.0f;
            for (int d = 0; d < d_head; ++d) {
                dot += Q[q_base + d] * K[k_base + d];
            }
            dot *= inv_sqrt_d;
            scores[j] = dot;
            row_max = std::max(row_max, dot);
        }

        float denom = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            scores[j] = std::exp(scores[j] - row_max);
            denom += scores[j];
        }

        const int64_t o_base = static_cast<int64_t>(i) * d_head;
        for (int d = 0; d < d_head; ++d) {
            float acc = 0.0f;
            for (int j = 0; j < seq_len; ++j) {
                const int64_t v_base = static_cast<int64_t>(j) * d_head;
                const float p = scores[j] / denom;
                acc += p * V[v_base + d];
            }
            O[o_base + d] = acc;
        }
    }

    return O;
}

ErrorStats compare_outputs(const std::vector<float> &ref,
                           const std::vector<float> &gpu) {
    ErrorStats stats;
    double sum_abs = 0.0;
    double sum_sq = 0.0;

    for (size_t i = 0; i < ref.size(); ++i) {
        const float diff = std::abs(ref[i] - gpu[i]);
        stats.max_abs_err = std::max(stats.max_abs_err, diff);
        sum_abs += diff;
        sum_sq += static_cast<double>(diff) * diff;
    }

    const double n = static_cast<double>(ref.size());
    stats.mean_abs_err = static_cast<float>(sum_abs / n);
    stats.rmse = static_cast<float>(std::sqrt(sum_sq / n));
    return stats;
}

float checksum_output(const std::vector<float> &gpu) {
    double sum = 0.0;
    for (float v : gpu) {
        sum += v;
    }
    return static_cast<float>(sum);
}

template <typename value_t>
ErrorStats run_attention(int batch, int seq_len, int n_heads) {
    constexpr int d_head = Kernel1<value_t>::d_head;
    constexpr int B_r = Kernel1<value_t>::B_r;
    constexpr int B_c = Kernel1<value_t>::B_c;

    if (seq_len % B_r != 0 || seq_len % B_c != 0) {
        throw std::runtime_error("seq_len must be divisible by B_r and B_c");
    }

    const int64_t head_stride = d_head;
    const int64_t seq_stride = static_cast<int64_t>(n_heads) * d_head;
    const int64_t batch_stride = static_cast<int64_t>(seq_len) * seq_stride;

    const size_t total_elems =
        static_cast<size_t>(batch) * seq_len * n_heads * d_head;

    std::vector<value_t> h_Q(total_elems);
    std::vector<value_t> h_K(total_elems);
    std::vector<value_t> h_V(total_elems);
    std::vector<value_t> h_O(total_elems);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < total_elems; ++i) {
        h_Q[i] = to_device<value_t>(dist(rng));
        h_K[i] = to_device<value_t>(dist(rng));
        h_V[i] = to_device<value_t>(dist(rng));
    }

    value_t *d_Q = nullptr;
    value_t *d_K = nullptr;
    value_t *d_V = nullptr;
    value_t *d_O = nullptr;

    CUDA_CHECK(cudaMalloc(&d_Q, total_elems * sizeof(value_t)));
    CUDA_CHECK(cudaMalloc(&d_K, total_elems * sizeof(value_t)));
    CUDA_CHECK(cudaMalloc(&d_V, total_elems * sizeof(value_t)));
    CUDA_CHECK(cudaMalloc(&d_O, total_elems * sizeof(value_t)));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), total_elems * sizeof(value_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), total_elems * sizeof(value_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), total_elems * sizeof(value_t),
                          cudaMemcpyHostToDevice));

    ForwardKernelArgs args;
    args.Q = d_Q;
    args.K = d_K;
    args.V = d_V;
    args.O = d_O;
    args.batch_stride = batch_stride;
    args.seq_stride = seq_stride;
    args.head_stride = head_stride;
    args.seq_len = seq_len;
    args.n_heads = n_heads;
    args.n_Q_blocks = seq_len / B_r;
    args.n_KV_blocks = seq_len / B_c;

    if constexpr (std::is_same_v<value_t, half>) {
        launch_flash_attention_forward(args, ScalarType::kF16, batch);
    } else {
        launch_flash_attention_forward(args, ScalarType::kBF16, batch);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_O.data(), d_O, total_elems * sizeof(value_t),
                          cudaMemcpyDeviceToHost));

    const auto h_Qf = to_float_vec(h_Q);
    const auto h_Kf = to_float_vec(h_K);
    const auto h_Vf = to_float_vec(h_V);
    const auto h_Of = to_float_vec(h_O);
    const float checksum = checksum_output(h_Of);

    if constexpr (std::is_same_v<value_t, half>) {
        std::cout << "FP16 checksum (seq_len=" << seq_len
                  << "): " << checksum << "\n";
    } else {
        std::cout << "BF16 checksum (seq_len=" << seq_len
                  << "): " << checksum << "\n";
    }

    ErrorStats stats;
    if (seq_len == 64) {
        if (batch != 1 || n_heads != 1) {
            throw std::runtime_error(
                "CPU reference only supports batch=1 and n_heads=1");
        }
        const auto ref = attention_cpu_ref(h_Qf, h_Kf, h_Vf, seq_len, d_head);
        stats = compare_outputs(ref, h_Of);

        if constexpr (std::is_same_v<value_t, half>) {
            std::cout << "FP16 stats (seq_len=" << seq_len
                      << "): max_abs_err=" << stats.max_abs_err
                      << " mean_abs_err=" << stats.mean_abs_err
                      << " rmse=" << stats.rmse << "\n";
        } else {
            std::cout << "BF16 stats (seq_len=" << seq_len
                      << "): max_abs_err=" << stats.max_abs_err
                      << " mean_abs_err=" << stats.mean_abs_err
                      << " rmse=" << stats.rmse << "\n";
        }
    }

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));

    return stats;
}

int main(int argc, char **argv) {
    try {
        const int batch = 1;
        int seq_len = 0;
        const int n_heads = 1;
        std::string dtype = "all";

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--seq_len" && i + 1 < argc) {
                seq_len = std::atoi(argv[++i]);
            } else if (arg == "--dtype" && i + 1 < argc) {
                dtype = argv[++i];
            }
        }

        auto check_threshold = [](float max_abs_err, float threshold) {
            if (max_abs_err > threshold) {
                std::cerr << "FAILED: max_abs_err " << max_abs_err
                          << " exceeds threshold " << threshold << "\n";
                std::exit(2);
            }
        };

        auto run_case = [&](int sl) {
            if (dtype == "fp16" || dtype == "f16" || dtype == "all") {
                auto stats = run_attention<half>(batch, sl, n_heads);
                if (sl == 64) {
                    check_threshold(stats.max_abs_err, 0.15f);
                }
            }
            if (dtype == "bf16" || dtype == "b16" || dtype == "all") {
                auto stats = run_attention<nv_bfloat16>(batch, sl, n_heads);
                if (sl == 64) {
                    check_threshold(stats.max_abs_err, 0.20f);
                }
            }
        };

        if (seq_len == 0) {
            run_case(64);
            run_case(128);
        } else {
            run_case(seq_len);
        }
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
