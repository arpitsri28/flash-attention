#pragma once

#include <stdint.h>

#include "constants.cuh"
#include "ptx_functions.cuh"

enum : int {
    MMA_M_FRAGMENTS_PER_ITER = 2,
    MMA_N_FRAGMENTS_PER_ITER = 1,
    MMA_K_FRAGMENTS_PER_ITER = 2,
    N_REGS_PER_F32_ACCUM_FRAGMENT = 2
};

template <typename value_t, int M_fragments, int N_fragments, int K_fragments,
          typename accum_t = float>
__forceinline__ __device__ constexpr void warp_fragment_mma_f32_accum(
    uint32_t (&regs_A)[M_fragments][K_fragments],
    uint32_t (&regs_B)[N_fragments][K_fragments],
    accum_t (&regs_C)[M_fragments][N_fragments * N_REGS_PER_F32_ACCUM_FRAGMENT]) {
    #pragma unroll
    for (int k = 0; k < K_fragments; k += MMA_K_FRAGMENTS_PER_ITER) {
        #pragma unroll
        for (int m = 0; m < M_fragments; m += MMA_M_FRAGMENTS_PER_ITER) {
            #pragma unroll
            for (int n = 0; n < N_fragments; n += MMA_N_FRAGMENTS_PER_ITER) {
                mma_m16n8k16_f32_accum<value_t>(
                    regs_C[m][n * 2],
                    regs_C[m][n * 2 + 1],
                    regs_C[m + 1][n * 2],
                    regs_C[m + 1][n * 2 + 1],
                    regs_A[m][k],
                    regs_A[m + 1][k],
                    regs_A[m][k + 1],
                    regs_A[m + 1][k + 1],
                    regs_B[n][k],
                    regs_B[n][k + 1],
                    regs_C[m][n * 2],
                    regs_C[m][n * 2 + 1],
                    regs_C[m + 1][n * 2],
                    regs_C[m + 1][n * 2 + 1]);
            }
        }
    }
}

template <typename value_t, int M_fragments, int N_fragments, int K_fragments>
struct GemmConfig {
    using value_type = value_t;
    static constexpr int M = M_fragments;
    static constexpr int N = N_fragments;
    static constexpr int K = K_fragments;
};

template <typename GemmCfg, typename A, typename B, typename C>
__forceinline__ __device__ void matmul(A &A_mat, B &B_mat, C &C_mat) {
    auto &regs_A = A_mat.data();
    auto &regs_B = B_mat.data();
    auto &regs_C = C_mat.data();

    warp_fragment_mma_f32_accum<typename GemmCfg::value_type, GemmCfg::M,
                                GemmCfg::N, GemmCfg::K>(regs_A, regs_B,
                                                       regs_C);
}
