#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "constants.cuh"
#include "ptx_functions.cuh"

struct TileLayout {
    int row_fragments;
    int col_fragments;

    constexpr TileLayout(int rows = 0, int cols = 0)
        : row_fragments(rows), col_fragments(cols) {}
};

// constexpr non-type template parameter containing parameters for LD/ST for a
// block (Q, K, V, or O) from GMEM to SMEM and vice versa, and also loading from
// SMEM to the RF.
struct TensorLDSTConfig {
    TileLayout GSM;
    TileLayout RF;

    bool transposed;
    int block_size;
    int smem_cols;
    int warp_ldst_rows;
    bool compute_over_entire_block;
    bool load_entire_block_into_rf;
    int mma_load_stages;
    bool async_copy;

    constexpr TensorLDSTConfig(TileLayout gsm, TileLayout rf, bool transpose,
                               int block, int smem_columns, int warp_rows,
                               bool compute_entire, bool load_entire,
                               int mma_stages, bool async)
        : GSM(gsm), RF(rf), transposed(transpose), block_size(block),
          smem_cols(smem_columns), warp_ldst_rows(warp_rows),
          compute_over_entire_block(compute_entire),
          load_entire_block_into_rf(load_entire), mma_load_stages(mma_stages),
          async_copy(async) {}
};

constexpr int ROWS_PER_FRAGMENT = 8;
constexpr int COLS_PER_FRAGMENT = 8;
constexpr int GSM_LDST_ROWS_PER_ITER = 4;
constexpr int BYTES_PER_VEC4_ACCESS = 16;
constexpr int ELEMS_PER_VEC4_ACCESS = 8;

template <typename T>
struct GM2SM_async {
    __device__ constexpr void operator()(T *gmem, T *smem) {
        cp_async<BYTES_PER_VEC4_ACCESS>(smem, gmem);
    }
};

template <typename T>
struct SM2GM {
    __device__ constexpr void operator()(T *gmem, T *smem) {
        reinterpret_cast<uint4 *>(gmem)[0] =
            reinterpret_cast<uint4 *>(smem)[0];
    }
};

template <typename op, TensorLDSTConfig CFG, typename value_t,
          typename index_t = int64_t>
__forceinline__ __device__ constexpr void copy_block_GSM(value_t *gmem,
                                                         value_t *smem,
                                                         index_t gmem_seq_stride,
                                                         const int lane_id) {
    constexpr int n_row_iters =
        CFG.GSM.row_fragments * ROWS_PER_FRAGMENT / GSM_LDST_ROWS_PER_ITER;

    constexpr int col_fragments_per_iter = WARP_SIZE / GSM_LDST_ROWS_PER_ITER;
    constexpr int col_fragments_per_row = CFG.smem_cols / COLS_PER_FRAGMENT;

    const int thread_row = lane_id / col_fragments_per_iter;
    const int thread_col_fragment = lane_id % col_fragments_per_iter;

    #pragma unroll
    for (int r = 0; r < n_row_iters; ++r) {
        const int cur_row = r * GSM_LDST_ROWS_PER_ITER + thread_row;
        #pragma unroll
        for (int c = 0; c < col_fragments_per_row;
             c += col_fragments_per_iter) {
            const int col_fragment = c + thread_col_fragment;

            op{}(&gmem[cur_row * gmem_seq_stride +
                       col_fragment * COLS_PER_FRAGMENT],
                 &smem[cur_row * CFG.smem_cols +
                       col_fragment * COLS_PER_FRAGMENT]);
        }
    }
}

template <TensorLDSTConfig CFG, typename value_t>
__forceinline__ __device__ constexpr void copy_warp_fragment_SM2RF(
    uint32_t (&regs)[CFG.RF.row_fragments][CFG.RF.col_fragments], value_t *smem,
    const int lane_id, const int col_fragment_offset = 0) {
    constexpr int row_fragments_per_iter = 2;
    constexpr int rows_per_iter = ROWS_PER_FRAGMENT * row_fragments_per_iter;

    constexpr int col_fragments = CFG.smem_cols / ELEMS_PER_VEC4_ACCESS;
    constexpr int col_fragments_per_iter = WARP_SIZE / rows_per_iter;

    const int thread_row = lane_id % rows_per_iter;
    const int thread_col_fragment = lane_id / rows_per_iter;

    #pragma unroll
    for (int r = 0; r < CFG.RF.row_fragments; r += row_fragments_per_iter) {
        const int cur_row = thread_row + r * ROWS_PER_FRAGMENT;
        #pragma unroll
        for (int c = 0; c < CFG.RF.col_fragments; c += col_fragments_per_iter) {
            const int smem_col_fragment =
                thread_col_fragment + c + col_fragment_offset;

            ldmatrix_x4(&smem[cur_row * CFG.smem_cols +
                              smem_col_fragment * ELEMS_PER_VEC4_ACCESS],
                        regs[r][c], regs[r + 1][c], regs[r][c + 1],
                        regs[r + 1][c + 1]);
        }
    }
}

template <TensorLDSTConfig CFG, typename value_t>
__forceinline__ __device__ constexpr void copy_warp_fragment_transposed_SM2RF(
    uint32_t (&regs)[CFG.RF.row_fragments][CFG.RF.col_fragments], value_t *smem,
    const int lane_id, const int row_fragment_offset = 0) {
    constexpr int row_fragments_per_iter = 2;
    constexpr int rows_per_iter = ROWS_PER_FRAGMENT * row_fragments_per_iter;

    constexpr int col_fragments = CFG.smem_cols / ELEMS_PER_VEC4_ACCESS;
    constexpr int col_fragments_per_iter = WARP_SIZE / rows_per_iter;

    const int thread_row = lane_id % rows_per_iter;
    const int thread_col_fragment = lane_id / rows_per_iter;

    #pragma unroll
    for (int r = 0; r < CFG.RF.col_fragments; r += row_fragments_per_iter) {
        const int cur_row =
            thread_row + (r + row_fragment_offset) * ROWS_PER_FRAGMENT;
        #pragma unroll
        for (int c = 0; c < CFG.RF.row_fragments; c += col_fragments_per_iter) {
            const int smem_col_fragment = thread_col_fragment + c;

            ldmatrix_x4_transpose(
                &smem[cur_row * CFG.smem_cols +
                      smem_col_fragment * ELEMS_PER_VEC4_ACCESS],
                regs[c][r], regs[c][r + 1], regs[c + 1][r], regs[c + 1][r + 1]);
        }
    }
}

template <TensorLDSTConfig CFG, typename value_t>
__forceinline__ __device__ constexpr void copy_warp_fragment_RF2SM(
    uint32_t (&regs)[CFG.RF.row_fragments][CFG.RF.col_fragments], value_t *smem,
    const int lane_id) {
    constexpr int rows_per_iter = ROWS_PER_FRAGMENT;
    constexpr int col_fragments_per_iter = 1;
    constexpr int col_fragments = CFG.smem_cols / ELEMS_PER_VEC4_ACCESS;

    constexpr int elems_per_store = 2;
    const int thread_row = lane_id / 4;
    const int thread_inner_col = (lane_id % 4) * elems_per_store;

    #pragma unroll
    for (int r = 0; r < CFG.RF.row_fragments; ++r) {
        const int cur_row = thread_row + r * rows_per_iter;
        #pragma unroll
        for (int c = 0; c < CFG.RF.col_fragments; c += col_fragments_per_iter) {
            const int smem_col_fragment = c;

            reinterpret_cast<uint32_t *>(
                &smem[cur_row * CFG.smem_cols +
                      (smem_col_fragment * ELEMS_PER_VEC4_ACCESS +
                       thread_inner_col)])[0] = regs[r][c];
        }
    }
}
