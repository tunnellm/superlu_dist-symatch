#pragma once

#include "xlupanels.hpp"

#ifdef HAVE_CUDA

static __device__ int_t symldl_v2_lfrag_nblocks(const int_t *frag_index)
{
    return frag_index[0];
}

static __device__ int_t symldl_v2_lfrag_gid(const int_t *frag_index,
                                            int_t k)
{
    return frag_index[LPANEL_HEADER_SIZE + k];
}

static __device__ int_t symldl_v2_lfrag_st_row(const int_t *frag_index,
                                               int_t k)
{
    int_t nblocks = symldl_v2_lfrag_nblocks(frag_index);
    return frag_index[LPANEL_HEADER_SIZE + nblocks + k];
}

static __device__ int_t symldl_v2_lfrag_nbrow(const int_t *frag_index,
                                              int_t k)
{
    return symldl_v2_lfrag_st_row(frag_index, k + 1) -
           symldl_v2_lfrag_st_row(frag_index, k);
}

static __device__ int_t *symldl_v2_lfrag_row_list(int_t *frag_index,
                                                  int_t k)
{
    int_t nblocks = symldl_v2_lfrag_nblocks(frag_index);
    return &frag_index[LPANEL_HEADER_SIZE + 2 * nblocks + 1 +
                       frag_index[LPANEL_HEADER_SIZE + nblocks + k]];
}

static __device__ void symldl_v2_lfrag_compute_indirect_map(
    int *src_to_dst, int_t src_len, int_t *src_vec,
    int_t dst_len, int_t *dst_vec, int *dst_idx)
{
    int thread_id = threadIdx.x;
    if (dst_vec == NULL)
    {
        if (thread_id < src_len)
            src_to_dst[thread_id] = static_cast<int>(src_vec[thread_id]);
        __syncthreads();
        return;
    }

    if (thread_id < src_len)
        dst_idx[src_vec[thread_id]] = GLOBAL_BLOCK_NOT_FOUND;
    __syncthreads();

    if (thread_id < dst_len)
        dst_idx[dst_vec[thread_id]] = thread_id;
    __syncthreads();

    if (thread_id < src_len)
        src_to_dst[thread_id] = dst_idx[src_vec[thread_id]];
    __syncthreads();
}

template <typename Ftype>
__device__ void symldl_v2_scatter_lpanel_fragment_dev(
    int_t iSt, int_t jSt,
    Ftype *gemmBuff, int LDgemmBuff,
    xlpanelGPU_t<Ftype> rowPanel,
    int_t *fragIndex,
    xLUstructGPU_t<Ftype> *dA)
{
    int_t ii = iSt + blockIdx.x;
    int_t jj = jSt + blockIdx.y;
    int thread_id = threadIdx.x;

    int_t gi = rowPanel.gid(ii);
    int_t gj = symldl_v2_lfrag_gid(fragIndex, jj);
    if (gi < gj)
        return;

    int_t lj = dA->lPanelIndex(gj);
    if (lj < 0)
        return;
    int_t li = dA->lPanelVec[lj].find(gi);
    if (li == GLOBAL_BLOCK_NOT_FOUND)
        return;

    Ftype *dst = dA->lPanelVec[lj].blkPtr(li);
    int_t lddst = dA->lPanelVec[lj].LDA();
    int_t dst_row_len = dA->lPanelVec[lj].nbrow(li);
    int_t *dst_row_list = dA->lPanelVec[lj].rowList(li);
    int_t dst_col_len = dA->supersize(gj);

    int nrows = static_cast<int>(rowPanel.nbrow(ii));
    int ncols = static_cast<int>(symldl_v2_lfrag_nbrow(fragIndex, jj));
    if (nrows <= 0 || ncols <= 0)
        return;

    extern __shared__ int baseSharedPtr[];
    int *row_src_to_dst = baseSharedPtr;
    int *col_src_to_dst = row_src_to_dst + dA->maxSuperSize;
    int *dst_idx = col_src_to_dst + dA->maxSuperSize;

    symldl_v2_lfrag_compute_indirect_map(
        row_src_to_dst, nrows, rowPanel.rowList(ii),
        dst_row_len, dst_row_list, dst_idx);
    symldl_v2_lfrag_compute_indirect_map(
        col_src_to_dst, ncols, symldl_v2_lfrag_row_list(fragIndex, jj),
        dst_col_len, NULL, dst_idx);

    int cols_per_thread_block = blockDim.x / nrows;
    if (cols_per_thread_block < 1)
        cols_per_thread_block = 1;

    int row_off = static_cast<int>(rowPanel.stRow(ii) -
                                   rowPanel.stRow(iSt));
    int col_off = static_cast<int>(
        symldl_v2_lfrag_st_row(fragIndex, jj) -
        symldl_v2_lfrag_st_row(fragIndex, jSt));
    Ftype *src = &gemmBuff[row_off + col_off * LDgemmBuff];

    if (thread_id < nrows * cols_per_thread_block)
    {
        int i = thread_id % nrows;
        int j = thread_id / nrows;
        while (j < ncols)
        {
            int di = row_src_to_dst[i];
            int dj = col_src_to_dst[j];
            if (di >= 0 && di < dst_row_len &&
                dj >= 0 && dj < dst_col_len)
                atomicAddT<Ftype>(&dst[di + lddst * dj],
                                  -src[i + LDgemmBuff * j]);
            j += cols_per_thread_block;
        }
    }
}

template <typename Ftype>
__global__ void symldl_v2_scatter_lpanel_fragment_kernel(
    int_t iSt, int_t jSt,
    Ftype *gemmBuff, int LDgemmBuff,
    xlpanelGPU_t<Ftype> rowPanel,
    int_t *fragIndex,
    xLUstructGPU_t<Ftype> *dA)
{
    symldl_v2_scatter_lpanel_fragment_dev(
        iSt, jSt, gemmBuff, LDgemmBuff, rowPanel, fragIndex, dA);
}

template <typename Ftype>
static void symldl_v2_scatter_lpanel_fragment(
    int_t iSt, int_t iEnd, int_t jSt, int_t jEnd,
    Ftype *gemmBuff, int LDgemmBuff,
    int maxSuperSize, int ldt,
    xlpanelGPU_t<Ftype> rowPanel,
    int_t *fragIndex,
    xLUstructGPU_t<Ftype> *dA,
    cudaStream_t cuStream)
{
    dim3 dimBlock(ldt);
    dim3 dimGrid(iEnd - iSt, jEnd - jSt);
    size_t sharedMemorySize = 3 * maxSuperSize * sizeof(int_t);
    symldl_v2_scatter_lpanel_fragment_kernel<Ftype>
        <<<dimGrid, dimBlock, sharedMemorySize, cuStream>>>(
            iSt, jSt, gemmBuff, LDgemmBuff, rowPanel, fragIndex, dA);
    gpuErrchk(cudaGetLastError());
}

#endif
