#pragma once

#include <algorithm>
#include <vector>

#include "xlupanels.hpp"

#ifdef HAVE_CUDA

static __device__ int_t symldl_v2_frag_nblocks(const int_t *frag_index)
{
    return frag_index[0];
}

static __device__ int_t symldl_v2_frag_gid(const int_t *frag_index,
                                           int_t k)
{
    return frag_index[LPANEL_HEADER_SIZE + k];
}

static __device__ int_t symldl_v2_frag_st_row(const int_t *frag_index,
                                              int_t k)
{
    int_t nblocks = symldl_v2_frag_nblocks(frag_index);
    return frag_index[LPANEL_HEADER_SIZE + nblocks + k];
}

static __device__ int_t symldl_v2_frag_nbrow(const int_t *frag_index,
                                             int_t k)
{
    return symldl_v2_frag_st_row(frag_index, k + 1) -
           symldl_v2_frag_st_row(frag_index, k);
}

static __device__ int_t *symldl_v2_frag_row_list(int_t *frag_index,
                                                 int_t k)
{
    int_t nblocks = symldl_v2_frag_nblocks(frag_index);
    return &frag_index[LPANEL_HEADER_SIZE + 2 * nblocks + 1 +
                       frag_index[LPANEL_HEADER_SIZE + nblocks + k]];
}

static __device__ void symldl_v2_compute_indirect_map(
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
__device__ void symldl_v2_scatter_two_fragments_dev(
    int_t iSt, int_t jSt,
    Ftype *gemmBuff, int LDgemmBuff,
    int_t *rowFragIndex, int_t *colFragIndex,
    xLUstructGPU_t<Ftype> *dA)
{
    int_t ii = iSt + blockIdx.x;
    int_t jj = jSt + blockIdx.y;
    int thread_id = threadIdx.x;

    int_t gi = symldl_v2_frag_gid(rowFragIndex, ii);
    int_t gj = symldl_v2_frag_gid(colFragIndex, jj);
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

    extern __shared__ int baseSharedPtr[];
    int *row_src_to_dst = baseSharedPtr;
    int *col_src_to_dst = row_src_to_dst + dA->maxSuperSize;
    int *dst_idx = col_src_to_dst + dA->maxSuperSize;

    int nrows = static_cast<int>(symldl_v2_frag_nbrow(rowFragIndex, ii));
    int ncols = static_cast<int>(symldl_v2_frag_nbrow(colFragIndex, jj));

    symldl_v2_compute_indirect_map(
        row_src_to_dst, nrows, symldl_v2_frag_row_list(rowFragIndex, ii),
        dst_row_len, dst_row_list, dst_idx);
    symldl_v2_compute_indirect_map(
        col_src_to_dst, ncols, symldl_v2_frag_row_list(colFragIndex, jj),
        dst_col_len, NULL, dst_idx);

    int nthreads = blockDim.x;
    int cols_per_thread_block = nrows > 0 ? nthreads / nrows : 1;
    if (cols_per_thread_block < 1)
        cols_per_thread_block = 1;

    int row_off = static_cast<int>(
        symldl_v2_frag_st_row(rowFragIndex, ii) -
        symldl_v2_frag_st_row(rowFragIndex, iSt));
    int col_off = static_cast<int>(
        symldl_v2_frag_st_row(colFragIndex, jj) -
        symldl_v2_frag_st_row(colFragIndex, jSt));
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
__global__ void symldl_v2_scatter_two_fragments_kernel(
    int_t iSt, int_t jSt,
    Ftype *gemmBuff, int LDgemmBuff,
    int_t *rowFragIndex, int_t *colFragIndex,
    xLUstructGPU_t<Ftype> *dA)
{
    symldl_v2_scatter_two_fragments_dev(
        iSt, jSt, gemmBuff, LDgemmBuff,
        rowFragIndex, colFragIndex, dA);
}

template <typename Ftype>
static void symldl_v2_scatter_two_fragments(
    int_t iSt, int_t iEnd, int_t jSt, int_t jEnd,
    Ftype *gemmBuff, int LDgemmBuff,
    int maxSuperSize, int ldt,
    int_t *rowFragIndex, int_t *colFragIndex,
    xLUstructGPU_t<Ftype> *dA,
    cudaStream_t cuStream)
{
    dim3 dimBlock(ldt);
    dim3 dimGrid(iEnd - iSt, jEnd - jSt);
    size_t sharedMemorySize = 3 * maxSuperSize * sizeof(int_t);
    symldl_v2_scatter_two_fragments_kernel<Ftype>
        <<<dimGrid, dimBlock, sharedMemorySize, cuStream>>>(
            iSt, jSt, gemmBuff, LDgemmBuff,
            rowFragIndex, colFragIndex, dA);
    gpuErrchk(cudaGetLastError());
}

static inline int_t symldl_v2_frag_host_nblocks(
    const std::vector<int_t> &frag)
{
    return frag.empty() ? 0 : frag[0];
}

static inline int_t symldl_v2_frag_host_gid(
    const std::vector<int_t> &frag, int_t k)
{
    return frag[LPANEL_HEADER_SIZE + k];
}

static inline int_t symldl_v2_frag_host_st_row(
    const std::vector<int_t> &frag, int_t k)
{
    int_t nblocks = symldl_v2_frag_host_nblocks(frag);
    return frag[LPANEL_HEADER_SIZE + nblocks + k];
}

static inline int_t symldl_v2_frag_host_nbrow(
    const std::vector<int_t> &frag, int_t k)
{
    return symldl_v2_frag_host_st_row(frag, k + 1) -
           symldl_v2_frag_host_st_row(frag, k);
}

static inline int_t symldl_v2_frag_host_find(
    const std::vector<int_t> &frag, int_t gid)
{
    int_t nblocks = symldl_v2_frag_host_nblocks(frag);
    for (int_t i = 0; i < nblocks; ++i)
        if (symldl_v2_frag_host_gid(frag, i) == gid)
            return i;
    return GLOBAL_BLOCK_NOT_FOUND;
}

static inline int_t symldl_v2_frag_host_end_block(
    const std::vector<int_t> &frag, int_t iSt, int maxRows)
{
    int_t nblocks = symldl_v2_frag_host_nblocks(frag);
    int_t base = symldl_v2_frag_host_st_row(frag, iSt);
    int_t iEnd = iSt + 1;
    while (iEnd < nblocks &&
           symldl_v2_frag_host_st_row(frag, iEnd + 1) - base <= maxRows)
        ++iEnd;
    return iEnd;
}

static inline int symldl_v2_frag_host_ranges_excluding(
    int_t begin, int_t end, int_t skip0, int_t skip1,
    int_t ranges[3][2])
{
    if (begin >= end)
        return 0;
    if (skip0 == skip1)
        skip1 = GLOBAL_BLOCK_NOT_FOUND;
    if (skip1 != GLOBAL_BLOCK_NOT_FOUND &&
        (skip0 == GLOBAL_BLOCK_NOT_FOUND || skip1 < skip0))
        std::swap(skip0, skip1);

    int nranges = 0;
    int_t cur = begin;
    int_t skips[2] = {skip0, skip1};
    for (int i = 0; i < 2; ++i)
    {
        int_t skip = skips[i];
        if (skip == GLOBAL_BLOCK_NOT_FOUND || skip < begin || skip >= end)
            continue;
        if (cur < skip)
        {
            ranges[nranges][0] = cur;
            ranges[nranges][1] = skip;
            ++nranges;
        }
        cur = skip + 1;
    }
    if (cur < end)
    {
        ranges[nranges][0] = cur;
        ranges[nranges][1] = end;
        ++nranges;
    }
    return nranges;
}

#endif
