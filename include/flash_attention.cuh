#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>

#include "constants.cuh"
#include "forward_kernel.cuh"
#include "gemm.cuh"
#include "tensor.cuh"

enum class ScalarType { kF16, kBF16 };

struct FlashForwardKernelConfig {
    const ScalarType dtype;
    const int d_head;
    const int B_r;
    const int B_c;
    const int n_warps;
};

template <typename T>
struct Kernel1 {
    using value_t = T;

    static constexpr int d_head = 128;
    static constexpr int B_r = 64;
    static constexpr int B_c = 64;
    static constexpr int n_warps = 4;

    static constexpr int QO_rows_per_warp = B_r / n_warps;

    struct N {
        static constexpr int QO_fragments_per_warp = QO_rows_per_warp / 8;
        static constexpr int KV_accum_fragments = B_c / 8;
        static constexpr int d_head_fragments = d_head / 8;
    };

    static constexpr int smem_elements =
        (B_r * d_head) + (B_c * d_head) + (B_c * d_head);
    static constexpr int smem_bytes = smem_elements * sizeof(value_t);

    static constexpr TensorLDSTConfig Q_LDST{
        TileLayout{QO_rows_per_warp / 8, d_head / 8},
        TileLayout{QO_rows_per_warp / 8, d_head / 8},
        false,
        B_r,
        d_head,
        QO_rows_per_warp,
        false,
        false,
        1,
        true};

    static constexpr TensorLDSTConfig K_LDST{
        TileLayout{(B_c / n_warps) / 8, d_head / 8},
        TileLayout{B_c / 8, d_head / 8},
        true,
        B_c,
        d_head,
        B_c / n_warps,
        true,
        true,
        1,
        true};

    static constexpr TensorLDSTConfig V_LDST{
        TileLayout{(B_c / n_warps) / 8, d_head / 8},
        TileLayout{B_c / 8, d_head / 8},
        false,
        B_c,
        d_head,
        B_c / n_warps,
        true,
        true,
        1,
        true};

    static constexpr TensorLDSTConfig O_LDST{
        TileLayout{QO_rows_per_warp / 8, d_head / 8},
        TileLayout{QO_rows_per_warp / 8, d_head / 8},
        false,
        B_r,
        d_head,
        QO_rows_per_warp,
        false,
        false,
        1,
        true};

    using Q_t = MatrixLDST<Q_LDST, value_t>;
    using K_t = MatrixLDST<K_LDST, value_t>;
    using V_t = MatrixLDST<V_LDST, value_t>;
    using O_value_t = MatrixLDST<O_LDST, value_t>;

    using S_accum_t = RFMatrix<float, 1, N::QO_fragments_per_warp,
                               N::KV_accum_fragments>;
    using P_value_t = RFMatrix<value_t, 1, N::QO_fragments_per_warp,
                               N::KV_accum_fragments>;
    using O_accum_t = RFMatrix<float, 1, N::QO_fragments_per_warp,
                               N::d_head_fragments>;

    using S_QK_GEMM =
        GemmConfig<value_t, N::QO_fragments_per_warp, N::KV_accum_fragments,
                   N::d_head_fragments>;
    using O_PV_GEMM =
        GemmConfig<value_t, N::QO_fragments_per_warp, N::d_head_fragments,
                   N::KV_accum_fragments>;
};

template <typename value_t>
inline void launch_flash_attention_forward(const ForwardKernelArgs &args,
                                           int batch_size) {
    using Kernel = Kernel1<value_t>;

    dim3 grid(args.n_Q_blocks, static_cast<unsigned int>(args.n_heads),
              static_cast<unsigned int>(batch_size));
    dim3 block(Kernel::n_warps * WARP_SIZE);

    flash_forward_kernel<Kernel><<<grid, block, Kernel::smem_bytes>>>(args);
}

inline void launch_flash_attention_forward(const ForwardKernelArgs &args,
                                           ScalarType dtype, int batch_size) {
    if (dtype == ScalarType::kF16) {
        launch_flash_attention_forward<half>(args, batch_size);
    } else if (dtype == ScalarType::kBF16) {
        launch_flash_attention_forward<nv_bfloat16>(args, batch_size);
    } else {
        throw std::runtime_error("Unsupported dtype");
    }
}
