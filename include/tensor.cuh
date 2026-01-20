#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

#include "load_store.cuh"

template <typename value_t, int N>
struct RFVector {
    static constexpr int size = N;
    value_t regs[N];

    __forceinline__ __device__ constexpr value_t &operator[](int idx) {
        return regs[idx];
    }
};

template <typename value_t, int n_stages, int row_fragments, int col_fragments>
struct RFMatrix {
    using storage_t = std::conditional_t<sizeof(value_t) == 4, float, uint32_t>;
    static constexpr int regs_per_fragment = sizeof(value_t) / 2;
    static constexpr int rows = row_fragments;
    static constexpr int cols = col_fragments * regs_per_fragment;

    storage_t regs[n_stages][rows][cols];

    __forceinline__ __device__ constexpr storage_t (&data(const int stage = 0))[rows][cols] {
        return regs[stage];
    }

    __forceinline__ __device__ constexpr void zero() {
        #pragma unroll
        for (int s = 0; s < n_stages; ++s) {
            #pragma unroll
            for (int j = 0; j < rows; ++j) {
                #pragma unroll
                for (int k = 0; k < cols; ++k) {
                    regs[s][j][k] = 0;
                }
            }
        }
    }
};

template <TensorLDSTConfig ldst, typename value_t, typename index_t = int64_t>
struct MatrixLDST {
    using matrix_storage_t = RFMatrix<value_t, ldst.mma_load_stages,
                                      ldst.RF.row_fragments,
                                      ldst.RF.col_fragments>;

    using GM2SM_op = GM2SM_async<value_t>;

    using SM2GM_op = SM2GM<value_t>;
    static constexpr int mma_load_stages = ldst.mma_load_stages;
    static constexpr bool load_entire_block_into_rf =
        ldst.load_entire_block_into_rf;
    static constexpr bool transposed = ldst.transposed;

    value_t *gmem_ptr;
    index_t gmem_seq_stride;
    value_t *smem_srm_ptr;
    value_t *smem_gsm_ptr;

    const int lane_id;

    matrix_storage_t storage;

    __forceinline__ __device__ MatrixLDST(value_t *gmem_block_ptr,
                                          index_t _gmem_seq_stride,
                                          value_t *_smem_ptr)
        : lane_id(threadIdx.x % WARP_SIZE) {
        const int warp_rank = threadIdx.x / WARP_SIZE;

        const index_t warp_seq = ldst.warp_ldst_rows * warp_rank;

        gmem_seq_stride = _gmem_seq_stride;
        gmem_ptr = gmem_block_ptr + warp_seq * gmem_seq_stride;

        smem_gsm_ptr = _smem_ptr + warp_seq * ldst.smem_cols;
        smem_srm_ptr = ldst.compute_over_entire_block ? _smem_ptr : smem_gsm_ptr;
    }

    __forceinline__ __device__ constexpr void zero() { storage.zero(); }

    __forceinline__ __device__ constexpr
        typename matrix_storage_t::storage_t (&data(const int stage = 0))
            [matrix_storage_t::rows][matrix_storage_t::cols] {
        return storage.data(stage);
    }

    __forceinline__ __device__ constexpr void advance_gmem_block() {
        gmem_ptr += ldst.block_size * gmem_seq_stride;
    }

    __forceinline__ __device__ constexpr void copy_GM2SM() {
        copy_block_GSM<GM2SM_op, ldst>(gmem_ptr, smem_gsm_ptr, gmem_seq_stride,
                                       lane_id);
    }

    __forceinline__ __device__ constexpr void copy_SM2GM() {
        copy_block_GSM<SM2GM_op, ldst>(gmem_ptr, smem_gsm_ptr, gmem_seq_stride,
                                       lane_id);
    }

    __forceinline__ __device__ constexpr void copy_SM2RF(int stage = 0,
                                                         int tile_offset = 0) {
        if constexpr (!transposed) {
            copy_warp_fragment_SM2RF<ldst, value_t>(storage.data(stage),
                                                    smem_srm_ptr, lane_id,
                                                    tile_offset);
        } else {
            copy_warp_fragment_transposed_SM2RF<ldst, value_t>(
                storage.data(stage), smem_srm_ptr, lane_id, tile_offset);
        }
    }

    __forceinline__ __device__ constexpr void copy_RF2SM() {
        copy_warp_fragment_RF2SM<ldst, value_t>(data(), smem_srm_ptr, lane_id);
    }
};

template <typename value_t, int M_fragments, int N_fragments>
__forceinline__ __device__ constexpr void convert_to_16_bit_dtype(
    float (&src_float)[M_fragments][N_fragments * 2],
    uint32_t (&dest_uint)[M_fragments][N_fragments]) {
    using value2_t =
        std::conditional_t<std::is_same_v<value_t, half>, half2, nv_bfloat162>;

    float2(&src)[M_fragments][N_fragments] =
        reinterpret_cast<float2(&)[M_fragments][N_fragments]>(src_float);
    value2_t(&dest)[M_fragments][N_fragments] =
        reinterpret_cast<value2_t(&)[M_fragments][N_fragments]>(dest_uint);

    #pragma unroll
    for (int m = 0; m < M_fragments; ++m) {
        #pragma unroll
        for (int n = 0; n < N_fragments; ++n) {
            if constexpr (std::is_same_v<value_t, half>) {
                dest[m][n] = __float22half2_rn(src[m][n]);
            } else {
                dest[m][n] = __float22bfloat162_rn(src[m][n]);
            }
        }
    }
}
