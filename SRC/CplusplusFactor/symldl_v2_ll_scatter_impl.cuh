#pragma once

#include "xlupanels.hpp"

#ifdef HAVE_CUDA

template <typename Ftype>
__device__ inline Ftype symldl_v2_ll_zero();

template <>
__device__ inline double symldl_v2_ll_zero<double>()
{
    return 0.0;
}

template <>
__device__ inline float symldl_v2_ll_zero<float>()
{
    return 0.0f;
}

template <>
__device__ inline doublecomplex symldl_v2_ll_zero<doublecomplex>()
{
    doublecomplex z = {0.0, 0.0};
    return z;
}

template <typename Ftype>
__device__ inline Ftype symldl_v2_ll_mul(Ftype a, Ftype b)
{
    return a * b;
}

template <>
__device__ inline doublecomplex symldl_v2_ll_mul<doublecomplex>(
    doublecomplex a, doublecomplex b)
{
    doublecomplex z = {
        a.r * b.r - a.i * b.i,
        a.r * b.i + a.i * b.r
    };
    return z;
}

template <typename Ftype>
__device__ inline Ftype symldl_v2_ll_add(Ftype a, Ftype b)
{
    return a + b;
}

template <>
__device__ inline doublecomplex symldl_v2_ll_add<doublecomplex>(
    doublecomplex a, doublecomplex b)
{
    doublecomplex z = {a.r + b.r, a.i + b.i};
    return z;
}

template <typename Ftype>
__global__ void symldl_v2_build_raw_l_range_kernel(
    Ftype *rawBlock, int_t ldraw,
    xlpanelGPU_t<Ftype> lpanel,
    int_t jSt, int_t jEnd,
    const Ftype *diag, int_t diag_ld)
{
    int_t ncols = lpanel.stRow(jEnd) - lpanel.stRow(jSt);
    int_t ksupc = diag_ld;
    int_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int_t total = ksupc * ncols;
    if (idx >= total)
        return;

    int_t lrow_local = idx % ncols;
    int_t diag_col = idx / ncols;
    int_t lrow = lpanel.stRow(jSt) + lrow_local;
    Ftype sum = symldl_v2_ll_zero<Ftype>();
    for (int_t p = 0; p < ksupc; ++p)
        sum = symldl_v2_ll_add(
            sum,
            symldl_v2_ll_mul(lpanel.val[lrow + p * lpanel.LDA()],
                             diag[p + diag_col * diag_ld]));
    rawBlock[lrow_local + diag_col * ldraw] = sum;
}

static __device__ void symldl_v2_ll_compute_indirect_map(
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
__device__ void symldl_v2_scatter_ll_range_dev(
    int_t iSt, int_t jSt,
    Ftype *gemmBuff, int LDgemmBuff,
    xlpanelGPU_t<Ftype> lpanel,
    xLUstructGPU_t<Ftype> *dA)
{
    int_t ii = iSt + blockIdx.x;
    int_t jj = jSt + blockIdx.y;
    int thread_id = threadIdx.x;

    int_t gi = lpanel.gid(ii);
    int_t gj = lpanel.gid(jj);
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

    int nrows = static_cast<int>(lpanel.nbrow(ii));
    int ncols = static_cast<int>(lpanel.nbrow(jj));
    if (nrows <= 0 || ncols <= 0)
        return;

    extern __shared__ int baseSharedPtr[];
    int *row_src_to_dst = baseSharedPtr;
    int *col_src_to_dst = row_src_to_dst + dA->maxSuperSize;
    int *dst_idx = col_src_to_dst + dA->maxSuperSize;

    symldl_v2_ll_compute_indirect_map(
        row_src_to_dst, nrows, lpanel.rowList(ii),
        dst_row_len, dst_row_list, dst_idx);
    symldl_v2_ll_compute_indirect_map(
        col_src_to_dst, ncols, lpanel.rowList(jj),
        dst_col_len, NULL, dst_idx);

    int cols_per_thread_block = blockDim.x / nrows;
    if (cols_per_thread_block < 1)
        cols_per_thread_block = 1;

    int row_off = static_cast<int>(lpanel.stRow(ii) -
                                   lpanel.stRow(iSt));
    int col_off = static_cast<int>(lpanel.stRow(jj) -
                                   lpanel.stRow(jSt));
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
__global__ void symldl_v2_scatter_ll_range_kernel(
    int_t iSt, int_t jSt,
    Ftype *gemmBuff, int LDgemmBuff,
    xlpanelGPU_t<Ftype> lpanel,
    xLUstructGPU_t<Ftype> *dA)
{
    symldl_v2_scatter_ll_range_dev(
        iSt, jSt, gemmBuff, LDgemmBuff, lpanel, dA);
}

template <typename Ftype>
static void symldl_v2_scatter_ll_range(
    int_t iSt, int_t iEnd, int_t jSt, int_t jEnd,
    Ftype *gemmBuff, int LDgemmBuff,
    int maxSuperSize, int ldt,
    xlpanelGPU_t<Ftype> lpanel,
    xLUstructGPU_t<Ftype> *dA,
    cudaStream_t cuStream)
{
    dim3 dimBlock(ldt);
    dim3 dimGrid(iEnd - iSt, jEnd - jSt);
    size_t sharedMemorySize = 3 * maxSuperSize * sizeof(int_t);
    symldl_v2_scatter_ll_range_kernel<Ftype>
        <<<dimGrid, dimBlock, sharedMemorySize, cuStream>>>(
            iSt, jSt, gemmBuff, LDgemmBuff, lpanel, dA);
    gpuErrchk(cudaGetLastError());
}

#endif
