#pragma once

#include <cuda_runtime.h>

#include "constants.cuh"

#define SHFL_ENTIRE_WARP_MASK 0xffffffff

template <int QO_fragments, int KV_accum_fragments, typename accum_t = float>
__forceinline__ __device__ constexpr void scale_S_accum(
    accum_t (&S_accum)[QO_fragments][KV_accum_fragments],
    const accum_t &softmax_scale) {
    #pragma unroll
    for (int q = 0; q < QO_fragments; ++q) {
        #pragma unroll
        for (int k = 0; k < KV_accum_fragments; ++k) {
            S_accum[q][k] *= softmax_scale;
        }
    }
}

template <int QO_fragments, int KV_accum_fragments, typename accum_t = float>
__forceinline__ __device__ constexpr void calc_row_max(
    accum_t (&S_accum)[QO_fragments][KV_accum_fragments],
    accum_t (&m_next)[QO_fragments], accum_t (&m_cur)[QO_fragments]) {
    #pragma unroll
    for (int q = 0; q < QO_fragments; ++q) {
        m_next[q] = m_cur[q];

        #pragma unroll
        for (int k = 0; k < KV_accum_fragments; ++k) {
            m_next[q] = fmaxf(m_next[q], S_accum[q][k]);
        }

        m_next[q] = fmaxf(__shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, m_next[q], 2),
                         m_next[q]);
        m_next[q] = fmaxf(__shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, m_next[q], 1),
                         m_next[q]);
    }
}

template <int QO_fragments, int d_head_accum_fragments,
          typename accum_t = float>
__forceinline__ __device__ constexpr void scale_l_O(
    accum_t (&m_next)[QO_fragments], accum_t (&m_cur)[QO_fragments],
    accum_t (&l)[QO_fragments],
    accum_t (&O_accum)[QO_fragments][d_head_accum_fragments]) {
    #pragma unroll
    for (int q = 0; q < QO_fragments; ++q) {
        const accum_t scale = expf(m_cur[q] - m_next[q]);
        m_cur[q] = m_next[q];
        l[q] *= scale;
        #pragma unroll
        for (int d_head = 0; d_head < d_head_accum_fragments; ++d_head) {
            O_accum[q][d_head] *= scale;
        }
    }
}

template <int QO_fragments, int KV_accum_fragments, typename accum_t = float>
__forceinline__ __device__ constexpr void exponentiate_tensor(
    accum_t (&S_accum)[QO_fragments][KV_accum_fragments],
    accum_t (&m)[QO_fragments]) {
    #pragma unroll
    for (int q = 0; q < QO_fragments; ++q) {
        #pragma unroll
        for (int k = 0; k < KV_accum_fragments; ++k) {
            S_accum[q][k] = expf(S_accum[q][k] - m[q]);
        }
    }
}

template <int QO_fragments, int d_head_accum_fragments,
          typename accum_t = float>
__forceinline__ __device__ constexpr void update_row_exp_sum(
    accum_t (&P_accum)[QO_fragments][d_head_accum_fragments],
    accum_t (&l)[QO_fragments]) {
    #pragma unroll
    for (int q = 0; q < QO_fragments; ++q) {
        #pragma unroll
        for (int d_head = 0; d_head < d_head_accum_fragments; ++d_head) {
            l[q] += P_accum[q][d_head];
        }
    }
}

template <int QO_fragments, int d_head_accum_fragments,
          typename accum_t = float>
__forceinline__ __device__ constexpr void final_softmax_normalization(
    accum_t (&O_accum)[QO_fragments][d_head_accum_fragments],
    accum_t (&l)[QO_fragments]) {
    #pragma unroll
    for (int q = 0; q < QO_fragments; ++q) {
        l[q] += __shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, l[q], 2);
        l[q] += __shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, l[q], 1);
    }

    #pragma unroll
    for (int q = 0; q < QO_fragments; ++q) {
        #pragma unroll
        for (int d_head = 0; d_head < d_head_accum_fragments; ++d_head) {
            O_accum[q][d_head] /= l[q];
        }
    }
}
