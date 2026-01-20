#pragma once

#include <cuda/std/limits>
#include <cuda_runtime.h>

#include "constants.cuh"
#include "gemm.cuh"
#include "softmax.cuh"
#include "tensor.cuh"

struct ForwardKernelArgs {
    using index_t = int64_t;

    void *__restrict__ Q;
    void *__restrict__ K;
    void *__restrict__ V;
    void *__restrict__ O;

    const index_t batch_stride;
    const index_t seq_stride;
    const index_t head_stride;

    const index_t seq_len;
    const index_t n_heads;

    const int n_Q_blocks;
    const int n_KV_blocks;
};

template <typename Kernel>
__global__ void flash_forward_kernel(__grid_constant__ const ForwardKernelArgs args) {
    using accum_t = float;
    using index_t = int64_t;
    using value_t = typename Kernel::value_t;
    using CFG = Kernel;
    using N = typename Kernel::N;

    using Q_t = typename Kernel::Q_t;
    using K_t = typename Kernel::K_t;
    using V_t = typename Kernel::V_t;
    using S_accum_t = typename Kernel::S_accum_t;
    using P_value_t = typename Kernel::P_value_t;
    using O_accum_t = typename Kernel::O_accum_t;
    using O_value_t = typename Kernel::O_value_t;

    const int sample = blockIdx.z;
    const int head = blockIdx.y;
    const int q_seq_block = blockIdx.x;

    const index_t gmem_seq_stride = args.seq_stride;

    const index_t sample_head_offset =
        sample * args.batch_stride + head * args.head_stride;
    const index_t QO_gmem_block_offset =
        sample_head_offset + q_seq_block * CFG::B_r * gmem_seq_stride;
    const index_t KV_gmem_block_offset = sample_head_offset;

    value_t *gmem_Q = &static_cast<value_t *>(args.Q)[QO_gmem_block_offset];
    value_t *gmem_O = &static_cast<value_t *>(args.O)[QO_gmem_block_offset];
    value_t *gmem_K = &static_cast<value_t *>(args.K)[KV_gmem_block_offset];
    value_t *gmem_V = &static_cast<value_t *>(args.V)[KV_gmem_block_offset];

    extern __shared__ __align__(16) char ch_smem[];
    value_t *smem_Q = reinterpret_cast<value_t *>(ch_smem);
    value_t *smem_O = smem_Q;
    value_t *smem_K = smem_Q + CFG::B_r * CFG::d_head;
    value_t *smem_V = smem_K + CFG::B_c * CFG::d_head;

    Q_t Q(gmem_Q, gmem_seq_stride, smem_Q);
    K_t K(gmem_K, gmem_seq_stride, smem_K);
    V_t V(gmem_V, gmem_seq_stride, smem_V);

    S_accum_t S_accum;
    P_value_t P_b16;
    O_accum_t O_accum;
    O_value_t O_b16(gmem_O, gmem_seq_stride, smem_O);

    Q.copy_GM2SM();
    cp_async_commit();
    O_accum.zero();

    const accum_t softmax_scale = rsqrtf(static_cast<accum_t>(CFG::d_head));
    constexpr accum_t neg_inf =
        -cuda::std::numeric_limits<float>::infinity();
    accum_t m[N::QO_fragments_per_warp];
    accum_t l[N::QO_fragments_per_warp];
    #pragma unroll
    for (int q = 0; q < N::QO_fragments_per_warp; ++q) {
        m[q] = neg_inf;
        l[q] = 0.0f;
    }

    cp_async_wait<0>();
    __syncwarp();
    Q.copy_SM2RF();

    for (int j = 0; j < args.n_KV_blocks; ++j) {
        K.copy_GM2SM();
        K.advance_gmem_block();
        cp_async_commit();

        S_accum.zero();
        cp_async_wait<0>();
        __syncthreads();

        K.copy_SM2RF();

        matmul<typename Kernel::S_QK_GEMM>(Q, K, S_accum);

        accum_t m_next[N::QO_fragments_per_warp];

        scale_S_accum(S_accum.data(), softmax_scale);
        calc_row_max(S_accum.data(), m_next, m);
        scale_l_O(m_next, m, l, O_accum.data());
        exponentiate_tensor(S_accum.data(), m_next);
        update_row_exp_sum(S_accum.data(), l);

        convert_to_16_bit_dtype<value_t>(S_accum.data(), P_b16.data());

        V.copy_GM2SM();
        V.advance_gmem_block();
        cp_async_commit();
        cp_async_wait<0>();
        __syncthreads();
        V.copy_SM2RF();

        matmul<typename Kernel::O_PV_GEMM>(P_b16, V, O_accum);
    }

    final_softmax_normalization(O_accum.data(), l);
    convert_to_16_bit_dtype<value_t>(O_accum.data(), O_b16.data());

    O_b16.copy_RF2SM();
    __syncwarp();
    O_b16.copy_SM2GM();
}
