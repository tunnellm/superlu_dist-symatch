#pragma once
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/logical.h>
#include <thrust/extrema.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <cstdio>
#include <cstdlib>
#include "superlu_ddefs.h"
#include "lupanels_GPU.cuh"
#include "lupanels.hpp"
#include "gpuCommon.hpp"
#include "gpu_mpi_utils.hpp"
#include "cublas_cusolver_wrappers.hpp"

#define USABLE_GPU_MEM_FRACTION 0.9

#ifdef SLU_SYM_GPU3D_DEBUG_TRACE
static inline void xlu_sym_gpu3d_trace_gpu_setup(gridinfo3d_t *grid3d,
                                                 const char *msg)
{
    std::printf("[sym-gpu3d-trace] rank %d: %s\n",
                (grid3d != NULL) ? grid3d->iam : -1, msg);
    std::fflush(stdout);
}
#else
static inline void xlu_sym_gpu3d_trace_gpu_setup(gridinfo3d_t *grid3d,
                                                 const char *msg)
{
    (void)grid3d;
    (void)msg;
}
#endif

static inline int superlu_gpu3d_contract_for_setup()
{
    const char *env = std::getenv("GPU3DCONTRACT");
    if (env == NULL || env[0] == '\0')
        return 0;

    char *end = NULL;
    long value = std::strtol(env, &end, 10);
    if (end == env || *end != '\0' || value < 0 || value > 3)
        ABORT("GPU3DCONTRACT must be one of 0, 1, 2, or 3.");
    return (int)value;
}

static inline bool superlu_sym_v2_batch_schur_enabled()
{
    static int cached = -1;
    if (cached >= 0)
        return cached != 0;

    const char *env = std::getenv("GPU3DV2_BATCH_SCHUR");
    cached = (env == NULL || env[0] == '\0') ? 1 : (std::atoi(env) != 0);
    return cached != 0;
}

static inline bool superlu_sym_v2_cta_scatter_enabled()
{
    static int cached = -1;
    if (cached >= 0)
        return cached != 0;

    const char *env = std::getenv("GPU3DV2_CTA_SCATTER");
    cached = (env == NULL || env[0] == '\0') ? 0 : (std::atoi(env) != 0);
    return cached != 0;
}

static inline bool superlu_sym_v2_lower_envelope_enabled()
{
    static int cached = -1;
    if (cached >= 0)
        return cached != 0;

    const char *env = std::getenv("GPU3DV2_LOWER_ENVELOPE");
    cached = (env == NULL || env[0] == '\0') ? 1 : (std::atoi(env) != 0);
    return cached != 0;
}

static inline int superlu_sym_v2_batch_schur_col_limit(
    int nrows, int64_t gemm_capacity)
{
    static int cached_override = -1;
    if (cached_override < 0)
    {
        const char *env = std::getenv("GPU3DV2_BATCH_SCHUR_COLS");
        if (env == NULL || env[0] == '\0')
        {
            cached_override = 0;
        }
        else
        {
            char *end = NULL;
            long value = std::strtol(env, &end, 10);
            if (end == env || *end != '\0' || value <= 0 ||
                value > 2147483647L)
                ABORT("GPU3DV2_BATCH_SCHUR_COLS must be a positive integer.");
            cached_override = static_cast<int>(value);
        }
    }

    if (cached_override > 0)
        return cached_override;

    int limit = static_cast<int>(std::sqrt(
        static_cast<double>(SUPERLU_MAX((int64_t)1, gemm_capacity))));
    if (limit < 1)
        limit = 1;
    if (nrows > 0 && nrows < limit)
    {
        int64_t wide_limit = gemm_capacity / static_cast<int64_t>(nrows);
        if (wide_limit > limit)
            limit = static_cast<int>(SUPERLU_MIN(
                wide_limit, static_cast<int64_t>(2147483647L)));
    }
    return limit;
}

static inline bool superlu_sym_v2_front_probe_enabled()
{
    const char *env = std::getenv("GPU3DV2_FRONT_PROBE");
    return env != NULL && env[0] != '\0' && std::atoi(env) != 0;
}

template <typename Ftype>
void xLUstruct_t<Ftype>::symV2ProbeLLRange(
    int_t k, xlpanel_t<Ftype> &lpanel,
    int_t iSt, int_t iEnd, int_t jSt, int_t jEnd,
    const std::vector<int_t> *frag)
{
    if (!superlu_sym_v2_front_probe_enabled() || lpanel.isEmpty())
        return;
    const int_t nfrag = (frag != NULL && !frag->empty()) ? (*frag)[0] : 0;
    for (int_t jj = jSt; jj < jEnd; ++jj)
    {
        int_t gj = 0;
        int_t ncols = 0;
        const int_t *cols = NULL;
        if (frag == NULL)
        {
            gj = lpanel.gid(jj);
            ncols = lpanel.nbrow(jj);
            cols = lpanel.rowList(jj);
        }
        else
        {
            if (jj < 0 || jj >= nfrag)
                continue;
            gj = (*frag)[LPANEL_HEADER_SIZE + jj];
            const int_t starts = LPANEL_HEADER_SIZE + nfrag;
            const int_t rows = LPANEL_HEADER_SIZE + 2 * nfrag + 1;
            const int_t begin = (*frag)[starts + jj];
            const int_t end = (*frag)[starts + jj + 1];
            ncols = end - begin;
            cols = frag->data() + rows + begin;
        }
        bool contiguous_cols = ncols > 0;
        for (int_t c = 1; c < ncols; ++c)
            if (cols[c] != cols[0] + c)
                contiguous_cols = false;

        for (int_t ii = iSt; ii < iEnd; ++ii)
        {
            const int_t gi = lpanel.gid(ii);
            if (gi < gj)
                continue;
            ++symV2ProbePairs;
            const int_t lj = symV2PanelIndex(gj);
            if (lj < 0 || lPanelVec[lj].isEmpty())
                continue;
            const int_t li = lPanelVec[lj].find(gi);
            if (li == GLOBAL_BLOCK_NOT_FOUND)
                continue;
            ++symV2ProbePresentPairs;

            const int_t nrows = lpanel.nbrow(ii);
            const long long flops =
                2LL * static_cast<long long>(nrows) *
                static_cast<long long>(ncols) *
                static_cast<long long>(supersize(k));
            symV2ProbeFlops += flops;

            bool identical_rows =
                nrows == lPanelVec[lj].nbrow(li);
            if (identical_rows)
            {
                const int_t *src = lpanel.rowList(ii);
                const int_t *dst = lPanelVec[lj].rowList(li);
                for (int_t r = 0; r < nrows; ++r)
                    if (src[r] != dst[r])
                    {
                        identical_rows = false;
                        break;
                    }
            }
            if (identical_rows && contiguous_cols)
            {
                ++symV2ProbeDirectPairs;
                symV2ProbeDirectFlops += flops;
            }
        }
    }
}

template <typename Ftype>
void xLUstruct_t<Ftype>::printSymV2FrontProbe()
{
    if (!superlu_sym_v2_front_probe_enabled())
        return;
    long long local[5] = {
        symV2ProbePairs,
        symV2ProbePresentPairs,
        symV2ProbeDirectPairs,
        symV2ProbeFlops,
        symV2ProbeDirectFlops
    };
    long long global[5] = {0, 0, 0, 0, 0};
    MPI_Reduce(local, global, 5, MPI_LONG_LONG, MPI_SUM, 0, grid3d->comm);
    if (grid3d->iam == 0)
    {
        const double pair_pct = global[1] > 0
            ? 100.0 * static_cast<double>(global[2]) /
                  static_cast<double>(global[1])
            : 0.0;
        const double flop_pct = global[3] > 0
            ? 100.0 * static_cast<double>(global[4]) /
                  static_cast<double>(global[3])
            : 0.0;
        std::printf(
            "SymFact V2 front probe: lower_pairs=%lld present=%lld "
            "direct_pairs=%lld (%.2f%%) flops=%lld direct_flops=%lld (%.2f%%)\n",
            global[0], global[1], global[2], pair_pct,
            global[3], global[4], flop_pct);
        std::fflush(stdout);
    }
}

size_t getGPUMemPerProcs(MPI_Comm baseCommunicator);

template <typename Ftype>
static inline int_t *xlu_sym_v2_panel_index_table(
    trf3dpartitionType<Ftype> *trf3Dpartition)
{
    (void)trf3Dpartition;
    return NULL;
}

template <>
inline int_t *xlu_sym_v2_panel_index_table<double>(
    trf3dpartitionType<double> *trf3Dpartition)
{
    return trf3Dpartition ? trf3Dpartition->symV2PanelLocalIndex : NULL;
}

template <typename Ftype>
__global__ void indirectCopy(Ftype *dest, Ftype *src, int_t *idx, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dest[idx[i]] = src[i];
}

/**
 * @file schurCompUpdate.cu
 * @brief This function copies the packed buffers to GPU and performs the sparse
   initialization on GPU call indirectCopy, this is the kernel
 * @param gpuValBasePtr is the base pointer of the GPU matrix
 * @param valBufferPacked is the packed buffer of the matrix
 * @param valIdx is the index of the packed buffer
 */
 template <typename Ftype>
void copyToGPU(Ftype *gpuValBasePtr, std::vector<Ftype> &valBufferPacked,
               std::vector<int_t> &valIdx)
{
    int nnzCount = valBufferPacked.size();
    // calculate the size of the packed buffers
    int_t gpuLvalSizePacked = nnzCount * sizeof(Ftype);
    int_t gpuLidxSizePacked = nnzCount * sizeof(int_t);
    // allocate the memory for the packed buffers on GPU
    Ftype *dlvalPacked;
    int_t *dlidxPacked;
    gpuErrchk(cudaMalloc(&dlvalPacked, gpuLvalSizePacked));
    gpuErrchk(cudaMalloc(&dlidxPacked, gpuLidxSizePacked));
    // copy the packed buffers from CPU to GPU
    gpuErrchk(cudaMemcpy(dlvalPacked, valBufferPacked.data(), gpuLvalSizePacked, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dlidxPacked, valIdx.data(), gpuLidxSizePacked, cudaMemcpyHostToDevice));
    // perform the sparse initialization on GPU call indirectCopy
    const int ThreadblockSize = 256;
    int nThreadBlocks = (nnzCount + ThreadblockSize - 1) / ThreadblockSize;
    indirectCopy<<<nThreadBlocks, ThreadblockSize>>>(
        gpuValBasePtr, dlvalPacked, dlidxPacked, nnzCount);
    // wait for it to finish and free dlvalPacked and dlidxPacked
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaFree(dlvalPacked));
    gpuErrchk(cudaFree(dlidxPacked));
}

// copy the panel to GPU
template <typename Ftype>
void copyToGPU_Sparse(Ftype *gpuValBasePtr, Ftype *valBuffer, int_t gpuLvalSize)
{
    // sparse Initialization for GPU, this is the experimental code
    // find non-zero elements in the panel, their location and values  and copy to GPU
    int numFtypes = gpuLvalSize / sizeof(Ftype);
    std::vector<Ftype> valBufferPacked;
    std::vector<int_t> valIdx;
    for (int_t i = 0; i < numFtypes; i++)
    {
        if (valBuffer[i] != 0)
        {
            valBufferPacked.push_back(valBuffer[i]);
            valIdx.push_back(i);
        }
    }
    printf("%d non-zero elements in the panel, wrt original=%d\n", valBufferPacked.size(), numFtypes);
    // get the size of the packed buffers and allocate memory on GPU
    copyToGPU(gpuValBasePtr, valBufferPacked, valIdx);
}

//#define NDEBUG
template <typename Ftype>
__device__ int_t xlpanelGPU_t<Ftype>::find(int_t k)
{
    int threadId = threadIdx.x;
    __shared__ int idx;
    __shared__ int found;
    if (!threadId)
    {
        idx = -1;
        found = 0;
    }

    int nThreads = blockDim.x;
    int blocksPerThreads = CEILING(nblocks(), nThreads);
    __syncthreads();
    for (int blk = blocksPerThreads * threadIdx.x;
         blk < blocksPerThreads * (threadIdx.x + 1);
         blk++)
    {
        if(found) break;

        if (blk < nblocks())
        {
            if (k == gid(blk))
            {
                idx = blk;
                found = 1;
            }
        }
    }
    __syncthreads();
    return idx;
}

template <typename Ftype>
__device__ int_t xlpanelGPU_t<Ftype>::findSerial(int_t k)
{
    const int_t n = nblocks();
    int_t lo = 0;
    int_t hi = n;
    while (lo < hi)
    {
        const int_t mid = lo + (hi - lo) / 2;
        const int_t g = gid(mid);
        if (g < k)
            lo = mid + 1;
        else
            hi = mid;
    }
    if (lo < n && gid(lo) == k)
        return lo;

    /* Preserve correctness if a panel was not stored in sorted GID order. */
    for (int_t i = 0; i < n; ++i)
        if (gid(i) == k)
            return i;
    return GLOBAL_BLOCK_NOT_FOUND;
}

template <typename Ftype>
__device__ int_t xupanelGPU_t<Ftype>::find(int_t k)
{
    int threadId = threadIdx.x;
    __shared__ int idx;
    __shared__ int found;
    if (!threadId)
    {
        idx = -1;
        found = 0;
    }
    __syncthreads();

    int nThreads = blockDim.x;
    int blocksPerThreads = CEILING(nblocks(), nThreads);

    for (int blk = blocksPerThreads * threadIdx.x;
         blk < blocksPerThreads * (threadIdx.x + 1);
         blk++)
    {
        if(found) break;

        if (blk < nblocks())
        {
            if (k == gid(blk))
            {
                idx = blk;
                found = 1;
            }
        }
    }
    __syncthreads();
    return idx;
}

__device__ int computeIndirectMapGPU(int *rcS2D, int_t srcLen, int_t *srcVec,
                                     int_t dstLen, int_t *dstVec,
                                     int *dstIdx)
{
    int threadId = threadIdx.x;
    if (dstVec == NULL) /*uncompressed dimension*/
    {
        if (threadId < srcLen)
            rcS2D[threadId] = srcVec[threadId];
        __syncthreads();
        return 0;
    }

    if (threadId < srcLen)
        dstIdx[srcVec[threadId]] = GLOBAL_BLOCK_NOT_FOUND;
    __syncthreads();

    if (threadId < dstLen)
        dstIdx[dstVec[threadId]] = threadId;
    __syncthreads();

    if (threadId < srcLen)
        rcS2D[threadId] = dstIdx[srcVec[threadId]];
    __syncthreads();

    return 0;
}

template <typename Ftype>
__device__ void scatterGPU_dev(
    int iSt, int jSt,
    Ftype *gemmBuff, int LDgemmBuff,
    xlpanelGPU_t<Ftype>& lpanel, xupanelGPU_t<Ftype>& upanel,
    xLUstructGPU_t<Ftype> *dA
)
{
    // calculate gi,gj
    int ii = iSt + blockIdx.x;
    int jj = jSt + blockIdx.y;
    int threadId = threadIdx.x;

    int gi = lpanel.gid(ii);
    int gj = upanel.gid(jj);
#ifndef NDEBUG
    // if (!threadId)
    //     printf("Scattering to (%d, %d) \n", gi, gj);
#endif
    Ftype *Dst;
    int_t lddst;
    int_t dstRowLen, dstColLen;
    int_t *dstRowList;
    int_t *dstColList;
    int li, lj;
    if (gj > gi) // its in upanel
    {
        li = dA->g2lRow(gi);
        lj = dA->uPanelVec[li].find(gj);
        Dst = dA->uPanelVec[li].blkPtr(lj);
        lddst = dA->supersize(gi);
        dstRowLen = dA->supersize(gi);
        dstRowList = NULL;
        dstColLen = dA->uPanelVec[li].nbcol(lj);
        dstColList = dA->uPanelVec[li].colList(lj);
    }
    else
    {
        lj = dA->lPanelIndex(gj);
        if (lj < 0)
            return;
        li = dA->lPanelVec[lj].find(gi);
        Dst = dA->lPanelVec[lj].blkPtr(li);
        lddst = dA->lPanelVec[lj].LDA();
        dstRowLen = dA->lPanelVec[lj].nbrow(li);
        dstRowList = dA->lPanelVec[lj].rowList(li);
        // if(!threadId )
        // printf("Scattering to (%d, %d) by %d li=%d\n",gi, gj,threadId,li);
        dstColLen = dA->supersize(gj);
        dstColList = NULL;
    }

    // compute source row to dest row mapping
    int maxSuperSize = dA->maxSuperSize;
    extern __shared__ int baseSharedPtr[];
    int *rowS2D = baseSharedPtr;
    int *colS2D = &rowS2D[maxSuperSize];
    int *dstIdx = &colS2D[maxSuperSize];

    int nrows = lpanel.nbrow(ii);
    int ncols = upanel.nbcol(jj);
    // lpanel.rowList(ii), upanel.colList(jj)

    computeIndirectMapGPU(rowS2D, nrows, lpanel.rowList(ii),
                          dstRowLen, dstRowList, dstIdx);

    // compute source col to dest col mapping
    computeIndirectMapGPU(colS2D, ncols, upanel.colList(jj),
                          dstColLen, dstColList, dstIdx);

    int nThreads = blockDim.x;
    int colsPerThreadBlock = nThreads / nrows;

    int rowOff = lpanel.stRow(ii) - lpanel.stRow(iSt);
    int colOff = upanel.stCol(jj) - upanel.stCol(jSt);
    Ftype *Src = &gemmBuff[rowOff + colOff * LDgemmBuff];
    int ldsrc = LDgemmBuff;
    // TODO: this seems inefficient
    if (threadId < nrows * colsPerThreadBlock)
    {
        /* 1D threads are logically arranged in 2D shape. */
        int i = threadId % nrows;
        int j = threadId / nrows;

#pragma unroll 4
        while (j < ncols)
        {
            int di = rowS2D[i];
            int dj = colS2D[j];
            if (di < 0 || di >= dstRowLen || dj < 0 || dj >= dstColLen)
            {
                j += colsPerThreadBlock;
                continue;
            }

#define ATOMIC_SCATTER
// Atomic Scatter is need if I want to perform multiple Schur Complement
//  update concurrently
#ifdef ATOMIC_SCATTER
            atomicAddT<Ftype>(&Dst[di + lddst * dj], -Src[i + ldsrc * j]);
#else
            Dst[di + lddst * dj] -= Src[i + ldsrc * j];
#endif
            j += colsPerThreadBlock;
        }
    }

    __syncthreads();
}
template <typename Ftype>
__global__ void scatterGPU(
    int iSt, int jSt,
    Ftype *gemmBuff, int LDgemmBuff,
    xlpanelGPU_t<Ftype> lpanel, xupanelGPU_t<Ftype> upanel,
    xLUstructGPU_t<Ftype> *dA)
{
    scatterGPU_dev(iSt, jSt, gemmBuff, LDgemmBuff, lpanel, upanel, dA);
}

template <typename Ftype>
__global__ void scatterGPU_batch(
    int* iSt_batch, int *iEnd_batch, int *jSt_batch, int *jEnd_batch,
    Ftype **gemmBuff_ptrs, int *LDgemmBuff_batch, xlpanelGPU_t<Ftype> *lpanels,
    xupanelGPU_t<Ftype> *upanels, xLUstructGPU_t<Ftype> *dA
)
{
    int batch_index = blockIdx.z;
    int iSt = iSt_batch[batch_index], iEnd = iEnd_batch[batch_index];
    int jSt = jSt_batch[batch_index], jEnd = jEnd_batch[batch_index];

    int ii = iSt + blockIdx.x;
    int jj = jSt + blockIdx.y;
    if(ii >= iEnd || jj >= jEnd)
        return;

    Ftype* gemmBuff = gemmBuff_ptrs[batch_index];
    if(gemmBuff == NULL)
        return;

    int LDgemmBuff = LDgemmBuff_batch[batch_index];
    lpanelGPU_t& lpanel = lpanels[batch_index];
    upanelGPU_t& upanel = upanels[batch_index];
    scatterGPU_dev(iSt, jSt, gemmBuff, LDgemmBuff, lpanel, upanel, dA);
}

template <typename Ftype>
void scatterGPU_driver(
    int iSt, int iEnd, int jSt, int jEnd, Ftype *gemmBuff, int LDgemmBuff,
    int maxSuperSize, int ldt, xlpanelGPU_t<Ftype> lpanel, xupanelGPU_t<Ftype> upanel,
    xLUstructGPU_t<Ftype> *dA, cudaStream_t cuStream
)
{
    dim3 dimBlock(ldt); // 1d thread
    dim3 dimGrid(iEnd - iSt, jEnd - jSt);
    size_t sharedMemorySize = 3 * maxSuperSize * sizeof(int_t);

    scatterGPU<Ftype><<<dimGrid, dimBlock, sharedMemorySize, cuStream>>>(
        iSt, jSt, gemmBuff, LDgemmBuff, lpanel, upanel, dA
    );

    gpuErrchk(cudaGetLastError());
}

template <typename Ftype>
__global__ void scatterSymLowerGPU(
    int_t ii, int_t jj,
    Ftype *gemmBuff, int LDgemmBuff,
    xlpanelGPU_t<Ftype> lpanel,
    xLUstructGPU_t<Ftype> *dA)
{
    int threadId = threadIdx.x;
    int gi = lpanel.gid(ii);
    int gj = lpanel.gid(jj);
    if (gi < gj)
        return;

    int lj = dA->lPanelIndex(gj);
    if (lj < 0)
        return;
    int li = dA->lPanelVec[lj].find(gi);
    if (li == GLOBAL_BLOCK_NOT_FOUND)
        return;

    Ftype *Dst = dA->lPanelVec[lj].blkPtr(li);
    int_t lddst = dA->lPanelVec[lj].LDA();
    int_t dstRowLen = dA->lPanelVec[lj].nbrow(li);
    int_t *dstRowList = dA->lPanelVec[lj].rowList(li);
    int_t dstColLen = dA->supersize(gj);

    extern __shared__ int baseSharedPtr[];
    int *rowS2D = baseSharedPtr;
    int *colS2D = &rowS2D[dA->maxSuperSize];
    int *dstIdx = &colS2D[dA->maxSuperSize];

    int nrows = lpanel.nbrow(ii);
    int ncols = lpanel.nbrow(jj);

    computeIndirectMapGPU(rowS2D, nrows, lpanel.rowList(ii),
                          dstRowLen, dstRowList, dstIdx);
    computeIndirectMapGPU(colS2D, ncols, lpanel.rowList(jj),
                          dstColLen, NULL, dstIdx);

    int nThreads = blockDim.x;
    int colsPerThreadBlock = nThreads / nrows;
    if (colsPerThreadBlock < 1)
        colsPerThreadBlock = 1;

    if (threadId < nrows * colsPerThreadBlock)
    {
        int i = threadId % nrows;
        int j = threadId / nrows;
        while (j < ncols)
        {
            int di = rowS2D[i];
            int dj = colS2D[j];
            if (di >= 0 && di < dstRowLen && dj >= 0 && dj < dstColLen)
                atomicAddT<Ftype>(&Dst[di + lddst * dj],
                              -gemmBuff[i + LDgemmBuff * j]);
            j += colsPerThreadBlock;
        }
    }
}

template <typename Ftype>
void scatterSymLowerGPU_driver(
    int_t ii, int_t jj,
    Ftype *gemmBuff, int LDgemmBuff,
    int maxSuperSize, int ldt,
    xlpanelGPU_t<Ftype> lpanel,
    xLUstructGPU_t<Ftype> *dA,
    cudaStream_t cuStream)
{
    dim3 dimBlock(ldt);
    dim3 dimGrid(1);
    size_t sharedMemorySize = 3 * maxSuperSize * sizeof(int_t);

    scatterSymLowerGPU<Ftype><<<dimGrid, dimBlock, sharedMemorySize, cuStream>>>(
        ii, jj, gemmBuff, LDgemmBuff, lpanel, dA);

    gpuErrchk(cudaGetLastError());
}

template <typename Ftype>
__device__ inline Ftype symDeviceZero();

template <typename Ftype>
__global__ void buildSymRawLRangeGPU(
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
    Ftype sum = symDeviceZero<Ftype>();
    for (int_t p = 0; p < ksupc; ++p)
        sum += lpanel.val[lrow + p * lpanel.LDA()] *
               diag[p + diag_col * diag_ld];
    rawBlock[lrow_local + diag_col * ldraw] = sum;
}

template <typename Ftype>
__device__ void scatterSymLowerRangeGPU_dev(
    int_t iSt, int_t jSt,
    Ftype *gemmBuff, int LDgemmBuff,
    xlpanelGPU_t<Ftype> &lpanel,
    xLUstructGPU_t<Ftype> *dA)
{
    int_t ii = iSt + blockIdx.x;
    int_t jj = jSt + blockIdx.y;
    int threadId = threadIdx.x;

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

    Ftype *Dst = dA->lPanelVec[lj].blkPtr(li);
    int_t lddst = dA->lPanelVec[lj].LDA();
    int_t dstRowLen = dA->lPanelVec[lj].nbrow(li);
    int_t *dstRowList = dA->lPanelVec[lj].rowList(li);
    int_t dstColLen = dA->supersize(gj);
    int_t *dstColList = NULL;

    extern __shared__ int baseSharedPtr[];
    int *rowS2D = baseSharedPtr;
    int *colS2D = &rowS2D[dA->maxSuperSize];
    int *dstIdx = &colS2D[dA->maxSuperSize];

    int nrows = lpanel.nbrow(ii);
    int ncols = lpanel.nbrow(jj);

    computeIndirectMapGPU(rowS2D, nrows, lpanel.rowList(ii),
                          dstRowLen, dstRowList, dstIdx);
    computeIndirectMapGPU(colS2D, ncols, lpanel.rowList(jj),
                          dstColLen, dstColList, dstIdx);

    int nThreads = blockDim.x;
    int colsPerThreadBlock = nThreads / nrows;
    if (colsPerThreadBlock < 1)
        colsPerThreadBlock = 1;

    int rowOff = lpanel.stRow(ii) - lpanel.stRow(iSt);
    int colOff = lpanel.stRow(jj) - lpanel.stRow(jSt);
    Ftype *Src = &gemmBuff[rowOff + colOff * LDgemmBuff];

    if (threadId < nrows * colsPerThreadBlock)
    {
        int i = threadId % nrows;
        int j = threadId / nrows;
        while (j < ncols)
        {
            int di = rowS2D[i];
            int dj = colS2D[j];
            if (di >= 0 && di < dstRowLen && dj >= 0 && dj < dstColLen)
                atomicAddT<Ftype>(&Dst[di + lddst * dj],
                              -Src[i + LDgemmBuff * j]);
            j += colsPerThreadBlock;
        }
    }
}

template <typename Ftype>
__global__ void scatterSymLowerRangeGPU(
    int_t iSt, int_t jSt,
    Ftype *gemmBuff, int LDgemmBuff,
    xlpanelGPU_t<Ftype> lpanel,
    xLUstructGPU_t<Ftype> *dA)
{
    scatterSymLowerRangeGPU_dev(iSt, jSt, gemmBuff, LDgemmBuff,
                                lpanel, dA);
}

/*
 * CTA-metadata scatter path for native symmetric updates.
 *
 * The legacy kernel has every thread repeat the destination-panel lookup,
 * block find, pointer extraction, and column-map copy.  This variant performs
 * invariant metadata work once per CTA and broadcasts it through shared
 * variables.  It also bypasses row-map construction when source and
 * destination row lists are identical, which is common in dense-fill panels.
 */
template <typename Ftype>
__global__ void scatterSymLowerRangeGPU_cta(
    int_t iSt, int_t jSt,
    Ftype *gemmBuff, int LDgemmBuff,
    xlpanelGPU_t<Ftype> lpanel,
    xLUstructGPU_t<Ftype> *dA)
{
    const int threadId = threadIdx.x;
    const int_t ii = iSt + blockIdx.x;
    const int_t jj = jSt + blockIdx.y;

    __shared__ int s_valid;
    __shared__ int s_nrows;
    __shared__ int s_ncols;
    __shared__ int s_row_identity;
    __shared__ int_t s_lddst;
    __shared__ int_t s_dst_row_len;
    __shared__ int_t s_dst_col_len;
    __shared__ Ftype *s_dst;
    __shared__ Ftype *s_src;
    __shared__ int_t *s_src_rows;
    __shared__ int_t *s_dst_rows;
    __shared__ int_t *s_src_cols;

    const int_t gi = lpanel.gid(ii);
    const int_t gj = lpanel.gid(jj);
    const int_t lj = (gi >= gj) ? dA->lPanelIndex(gj) : -1;
    __shared__ int_t s_lookup_li;
    if (threadId == 0)
        s_lookup_li = (lj >= 0)
                          ? dA->lPanelVec[lj].findSerial(gi)
                          : GLOBAL_BLOCK_NOT_FOUND;
    __syncthreads();
    const int_t li = s_lookup_li;

    if (threadId == 0)
    {
        s_valid = 0;
        if (gi >= gj && lj >= 0 && li != GLOBAL_BLOCK_NOT_FOUND)
        {
            s_nrows = static_cast<int>(lpanel.nbrow(ii));
            s_ncols = static_cast<int>(lpanel.nbrow(jj));
            s_dst = dA->lPanelVec[lj].blkPtr(li);
            s_lddst = dA->lPanelVec[lj].LDA();
            s_dst_row_len = dA->lPanelVec[lj].nbrow(li);
            s_dst_rows = dA->lPanelVec[lj].rowList(li);
            s_dst_col_len = dA->supersize(gj);
            s_src_rows = lpanel.rowList(ii);
            s_src_cols = lpanel.rowList(jj);
            const int_t row_off = lpanel.stRow(ii) - lpanel.stRow(iSt);
            const int_t col_off = lpanel.stRow(jj) - lpanel.stRow(jSt);
            s_src = gemmBuff + row_off + col_off * LDgemmBuff;
            s_row_identity =
                (s_nrows == static_cast<int>(s_dst_row_len)) ? 1 : 0;
            s_valid = (s_nrows > 0 && s_ncols > 0) ? 1 : 0;
        }
    }
    __syncthreads();
    if (!s_valid)
        return;

    if (s_row_identity && threadId < s_nrows &&
        s_src_rows[threadId] != s_dst_rows[threadId])
        atomicExch(&s_row_identity, 0);
    __syncthreads();

    extern __shared__ int baseSharedPtr[];
    int *rowS2D = baseSharedPtr;
    int *dstIdx = rowS2D + dA->maxSuperSize;
    if (!s_row_identity)
        computeIndirectMapGPU(rowS2D, s_nrows, s_src_rows,
                              s_dst_row_len, s_dst_rows, dstIdx);

    int colsPerThreadBlock = blockDim.x / s_nrows;
    if (colsPerThreadBlock < 1)
        colsPerThreadBlock = 1;

    if (threadId < s_nrows * colsPerThreadBlock)
    {
        const int i = threadId % s_nrows;
        int j = threadId / s_nrows;
        while (j < s_ncols)
        {
            const int di = s_row_identity ? i : rowS2D[i];
            const int dj = static_cast<int>(s_src_cols[j]);
            if (di >= 0 && di < s_dst_row_len &&
                dj >= 0 && dj < s_dst_col_len)
                atomicAddT<Ftype>(&s_dst[di + s_lddst * dj],
                                  -s_src[i + LDgemmBuff * j]);
            j += colsPerThreadBlock;
        }
    }
}

template <typename Ftype>
void scatterSymLowerRangeGPU_driver(
    int_t iSt, int_t iEnd, int_t jSt, int_t jEnd,
    Ftype *gemmBuff, int LDgemmBuff,
    int maxSuperSize, int ldt,
    xlpanelGPU_t<Ftype> lpanel,
    xLUstructGPU_t<Ftype> *dA,
    cudaStream_t cuStream)
{
    dim3 dimBlock(ldt);
    dim3 dimGrid(iEnd - iSt, jEnd - jSt);
    const bool cta_scatter = superlu_sym_v2_cta_scatter_enabled();
    size_t sharedMemorySize =
        (cta_scatter ? 2 : 3) * maxSuperSize * sizeof(int_t);

    if (cta_scatter)
    {
        scatterSymLowerRangeGPU_cta<Ftype>
            <<<dimGrid, dimBlock, sharedMemorySize, cuStream>>>(
                iSt, jSt, gemmBuff, LDgemmBuff, lpanel, dA);
    }
    else
    {
        scatterSymLowerRangeGPU<Ftype>
            <<<dimGrid, dimBlock, sharedMemorySize, cuStream>>>(
                iSt, jSt, gemmBuff, LDgemmBuff, lpanel, dA);
    }

    gpuErrchk(cudaGetLastError());
}

template <typename Ftype>
__device__ void scatterSymLowerTwoLRangeGPU_dev(
    int_t iSt, int_t jSt,
    Ftype *gemmBuff, int LDgemmBuff,
    xlpanelGPU_t<Ftype> &rowPanel,
    xlpanelGPU_t<Ftype> &colPanel,
    xLUstructGPU_t<Ftype> *dA)
{
    int_t ii = iSt + blockIdx.x;
    int_t jj = jSt + blockIdx.y;
    int threadId = threadIdx.x;

    int_t gi = rowPanel.gid(ii);
    int_t gj = colPanel.gid(jj);
    if (gi < gj)
        return;

    int_t lj = dA->lPanelIndex(gj);
    if (lj < 0)
        return;
    int_t li = dA->lPanelVec[lj].find(gi);
    if (li == GLOBAL_BLOCK_NOT_FOUND)
        return;

    Ftype *Dst = dA->lPanelVec[lj].blkPtr(li);
    int_t lddst = dA->lPanelVec[lj].LDA();
    int_t dstRowLen = dA->lPanelVec[lj].nbrow(li);
    int_t *dstRowList = dA->lPanelVec[lj].rowList(li);
    int_t dstColLen = dA->supersize(gj);

    extern __shared__ int baseSharedPtr[];
    int *rowS2D = baseSharedPtr;
    int *colS2D = &rowS2D[dA->maxSuperSize];
    int *dstIdx = &colS2D[dA->maxSuperSize];

    int nrows = rowPanel.nbrow(ii);
    int ncols = colPanel.nbrow(jj);

    computeIndirectMapGPU(rowS2D, nrows, rowPanel.rowList(ii),
                          dstRowLen, dstRowList, dstIdx);
    computeIndirectMapGPU(colS2D, ncols, colPanel.rowList(jj),
                          dstColLen, NULL, dstIdx);

    int nThreads = blockDim.x;
    int colsPerThreadBlock = nThreads / nrows;
    if (colsPerThreadBlock < 1)
        colsPerThreadBlock = 1;

    int rowOff = rowPanel.stRow(ii) - rowPanel.stRow(iSt);
    int colOff = colPanel.stRow(jj) - colPanel.stRow(jSt);
    Ftype *Src = &gemmBuff[rowOff + colOff * LDgemmBuff];

    if (threadId < nrows * colsPerThreadBlock)
    {
        int i = threadId % nrows;
        int j = threadId / nrows;
        while (j < ncols)
        {
            int di = rowS2D[i];
            int dj = colS2D[j];
            if (di >= 0 && di < dstRowLen && dj >= 0 && dj < dstColLen)
                atomicAddT<Ftype>(&Dst[di + lddst * dj],
                              -Src[i + LDgemmBuff * j]);
            j += colsPerThreadBlock;
        }
    }
}

template <typename Ftype>
__global__ void scatterSymLowerTwoLRangeGPU(
    int_t iSt, int_t jSt,
    Ftype *gemmBuff, int LDgemmBuff,
    xlpanelGPU_t<Ftype> rowPanel,
    xlpanelGPU_t<Ftype> colPanel,
    xLUstructGPU_t<Ftype> *dA)
{
    scatterSymLowerTwoLRangeGPU_dev(iSt, jSt, gemmBuff, LDgemmBuff,
                                    rowPanel, colPanel, dA);
}

template <typename Ftype>
void scatterSymLowerTwoLRangeGPU_driver(
    int_t iSt, int_t iEnd, int_t jSt, int_t jEnd,
    Ftype *gemmBuff, int LDgemmBuff,
    int maxSuperSize, int ldt,
    xlpanelGPU_t<Ftype> rowPanel,
    xlpanelGPU_t<Ftype> colPanel,
    xLUstructGPU_t<Ftype> *dA,
    cudaStream_t cuStream)
{
    dim3 dimBlock(ldt);
    dim3 dimGrid(iEnd - iSt, jEnd - jSt);
    size_t sharedMemorySize = 3 * maxSuperSize * sizeof(int_t);

    scatterSymLowerTwoLRangeGPU<Ftype>
        <<<dimGrid, dimBlock, sharedMemorySize, cuStream>>>(
            iSt, jSt, gemmBuff, LDgemmBuff, rowPanel, colPanel, dA);

    gpuErrchk(cudaGetLastError());
}

static __device__ int_t symFragNBlocks(const int_t *frag_index)
{
    return frag_index[0];
}

static __device__ int_t symFragGid(const int_t *frag_index, int_t k)
{
    return frag_index[LPANEL_HEADER_SIZE + k];
}

static __device__ int_t symFragStRow(const int_t *frag_index, int_t k)
{
    int_t nblocks = symFragNBlocks(frag_index);
    return frag_index[LPANEL_HEADER_SIZE + nblocks + k];
}

static __device__ int_t symFragNbrow(const int_t *frag_index, int_t k)
{
    return symFragStRow(frag_index, k + 1) - symFragStRow(frag_index, k);
}

static __device__ int_t *symFragRowList(int_t *frag_index, int_t k)
{
    int_t nblocks = symFragNBlocks(frag_index);
    return &frag_index[LPANEL_HEADER_SIZE + 2 * nblocks + 1 +
                       frag_index[LPANEL_HEADER_SIZE + nblocks + k]];
}

template <typename Ftype>
__device__ void scatterSymLowerLFragmentRangeGPU_dev(
    int_t iSt, int_t jSt,
    Ftype *gemmBuff, int LDgemmBuff,
    xlpanelGPU_t<Ftype> &rowPanel,
    int_t *fragIndex, Ftype *fragVal,
    xLUstructGPU_t<Ftype> *dA)
{
    int_t ii = iSt + blockIdx.x;
    int_t jj = jSt + blockIdx.y;
    int threadId = threadIdx.x;

    int_t gi = rowPanel.gid(ii);
    int_t gj = symFragGid(fragIndex, jj);
    if (gi < gj)
        return;

    int_t lj = dA->lPanelIndex(gj);
    if (lj < 0)
        return;
    int_t li = dA->lPanelVec[lj].find(gi);
    if (li == GLOBAL_BLOCK_NOT_FOUND)
        return;

    Ftype *Dst = dA->lPanelVec[lj].blkPtr(li);
    int_t lddst = dA->lPanelVec[lj].LDA();
    int_t dstRowLen = dA->lPanelVec[lj].nbrow(li);
    int_t *dstRowList = dA->lPanelVec[lj].rowList(li);
    int_t dstColLen = dA->supersize(gj);

    extern __shared__ int baseSharedPtr[];
    int *rowS2D = baseSharedPtr;
    int *colS2D = &rowS2D[dA->maxSuperSize];
    int *dstIdx = &colS2D[dA->maxSuperSize];

    int nrows = rowPanel.nbrow(ii);
    int ncols = symFragNbrow(fragIndex, jj);

    computeIndirectMapGPU(rowS2D, nrows, rowPanel.rowList(ii),
                          dstRowLen, dstRowList, dstIdx);
    computeIndirectMapGPU(colS2D, ncols, symFragRowList(fragIndex, jj),
                          dstColLen, NULL, dstIdx);

    int nThreads = blockDim.x;
    int colsPerThreadBlock = nThreads / nrows;
    if (colsPerThreadBlock < 1)
        colsPerThreadBlock = 1;

    int rowOff = rowPanel.stRow(ii) - rowPanel.stRow(iSt);
    int colOff = symFragStRow(fragIndex, jj) - symFragStRow(fragIndex, jSt);
    Ftype *Src = &gemmBuff[rowOff + colOff * LDgemmBuff];

    if (threadId < nrows * colsPerThreadBlock)
    {
        int i = threadId % nrows;
        int j = threadId / nrows;
        while (j < ncols)
        {
            int di = rowS2D[i];
            int dj = colS2D[j];
            if (di >= 0 && di < dstRowLen && dj >= 0 && dj < dstColLen)
                atomicAddT<Ftype>(&Dst[di + lddst * dj],
                                  -Src[i + LDgemmBuff * j]);
            j += colsPerThreadBlock;
        }
    }
}

template <typename Ftype>
__global__ void scatterSymLowerLFragmentRangeGPU(
    int_t iSt, int_t jSt,
    Ftype *gemmBuff, int LDgemmBuff,
    xlpanelGPU_t<Ftype> rowPanel,
    int_t *fragIndex, Ftype *fragVal,
    xLUstructGPU_t<Ftype> *dA)
{
    scatterSymLowerLFragmentRangeGPU_dev(iSt, jSt, gemmBuff, LDgemmBuff,
                                         rowPanel, fragIndex, fragVal, dA);
}

/* CTA-invariant metadata path for received raw-L fragments. */
template <typename Ftype>
__global__ void scatterSymLowerLFragmentRangeGPU_cta(
    int_t iSt, int_t jSt,
    Ftype *gemmBuff, int LDgemmBuff,
    xlpanelGPU_t<Ftype> rowPanel,
    int_t *fragIndex, Ftype *fragVal,
    xLUstructGPU_t<Ftype> *dA)
{
    (void)fragVal;
    const int threadId = threadIdx.x;
    const int_t ii = iSt + blockIdx.x;
    const int_t jj = jSt + blockIdx.y;

    __shared__ int s_valid;
    __shared__ int s_nrows;
    __shared__ int s_ncols;
    __shared__ int s_row_identity;
    __shared__ int_t s_lddst;
    __shared__ int_t s_dst_row_len;
    __shared__ int_t s_dst_col_len;
    __shared__ Ftype *s_dst;
    __shared__ Ftype *s_src;
    __shared__ int_t *s_src_rows;
    __shared__ int_t *s_dst_rows;
    __shared__ int_t *s_src_cols;

    const int_t gi = rowPanel.gid(ii);
    const int_t gj = symFragGid(fragIndex, jj);
    const int_t lj = (gi >= gj) ? dA->lPanelIndex(gj) : -1;
    __shared__ int_t s_lookup_li;
    if (threadId == 0)
        s_lookup_li = (lj >= 0)
                          ? dA->lPanelVec[lj].findSerial(gi)
                          : GLOBAL_BLOCK_NOT_FOUND;
    __syncthreads();
    const int_t li = s_lookup_li;

    if (threadId == 0)
    {
        s_valid = 0;
        if (gi >= gj && lj >= 0 && li != GLOBAL_BLOCK_NOT_FOUND)
        {
            s_nrows = static_cast<int>(rowPanel.nbrow(ii));
            s_ncols = static_cast<int>(symFragNbrow(fragIndex, jj));
            s_dst = dA->lPanelVec[lj].blkPtr(li);
            s_lddst = dA->lPanelVec[lj].LDA();
            s_dst_row_len = dA->lPanelVec[lj].nbrow(li);
            s_dst_rows = dA->lPanelVec[lj].rowList(li);
            s_dst_col_len = dA->supersize(gj);
            s_src_rows = rowPanel.rowList(ii);
            s_src_cols = symFragRowList(fragIndex, jj);
            const int_t row_off =
                rowPanel.stRow(ii) - rowPanel.stRow(iSt);
            const int_t col_off =
                symFragStRow(fragIndex, jj) -
                symFragStRow(fragIndex, jSt);
            s_src = gemmBuff + row_off + col_off * LDgemmBuff;
            s_row_identity =
                (s_nrows == static_cast<int>(s_dst_row_len)) ? 1 : 0;
            s_valid = (s_nrows > 0 && s_ncols > 0) ? 1 : 0;
        }
    }
    __syncthreads();
    if (!s_valid)
        return;

    if (s_row_identity && threadId < s_nrows &&
        s_src_rows[threadId] != s_dst_rows[threadId])
        atomicExch(&s_row_identity, 0);
    __syncthreads();

    extern __shared__ int baseSharedPtr[];
    int *rowS2D = baseSharedPtr;
    int *dstIdx = rowS2D + dA->maxSuperSize;
    if (!s_row_identity)
        computeIndirectMapGPU(rowS2D, s_nrows, s_src_rows,
                              s_dst_row_len, s_dst_rows, dstIdx);

    int colsPerThreadBlock = blockDim.x / s_nrows;
    if (colsPerThreadBlock < 1)
        colsPerThreadBlock = 1;

    if (threadId < s_nrows * colsPerThreadBlock)
    {
        const int i = threadId % s_nrows;
        int j = threadId / s_nrows;
        while (j < s_ncols)
        {
            const int di = s_row_identity ? i : rowS2D[i];
            const int dj = static_cast<int>(s_src_cols[j]);
            if (di >= 0 && di < s_dst_row_len &&
                dj >= 0 && dj < s_dst_col_len)
                atomicAddT<Ftype>(&s_dst[di + s_lddst * dj],
                                  -s_src[i + LDgemmBuff * j]);
            j += colsPerThreadBlock;
        }
    }
}

template <typename Ftype>
void scatterSymLowerLFragmentRangeGPU_driver(
    int_t iSt, int_t iEnd, int_t jSt, int_t jEnd,
    Ftype *gemmBuff, int LDgemmBuff,
    int maxSuperSize, int ldt,
    xlpanelGPU_t<Ftype> rowPanel,
    int_t *fragIndex, Ftype *fragVal,
    xLUstructGPU_t<Ftype> *dA,
    cudaStream_t cuStream)
{
    dim3 dimBlock(ldt);
    dim3 dimGrid(iEnd - iSt, jEnd - jSt);
    const bool cta_scatter = superlu_sym_v2_cta_scatter_enabled();
    size_t sharedMemorySize =
        (cta_scatter ? 2 : 3) * maxSuperSize * sizeof(int_t);

    if (cta_scatter)
    {
        scatterSymLowerLFragmentRangeGPU_cta<Ftype>
            <<<dimGrid, dimBlock, sharedMemorySize, cuStream>>>(
                iSt, jSt, gemmBuff, LDgemmBuff,
                rowPanel, fragIndex, fragVal, dA);
    }
    else
    {
        scatterSymLowerLFragmentRangeGPU<Ftype>
            <<<dimGrid, dimBlock, sharedMemorySize, cuStream>>>(
                iSt, jSt, gemmBuff, LDgemmBuff,
                rowPanel, fragIndex, fragVal, dA);
    }

    gpuErrchk(cudaGetLastError());
}

template <>
__device__ inline double symDeviceZero<double>()
{
    return 0.0;
}

template <>
__device__ inline float symDeviceZero<float>()
{
    return 0.0f;
}

template <>
__device__ inline doublecomplex symDeviceZero<doublecomplex>()
{
    doublecomplex z = {0.0, 0.0};
    return z;
}

template <typename Ftype>
void scatterGPU_batchDriver(
    int* iSt_batch, int *iEnd_batch, int *jSt_batch, int *jEnd_batch,
    int max_ilen, int max_jlen, Ftype **gemmBuff_ptrs, int *LDgemmBuff_batch,
    int maxSuperSize, int ldt, xlpanelGPU_t<Ftype> *lpanels, xupanelGPU_t<Ftype> *upanels,
    xLUstructGPU_t<Ftype> *dA, int batchCount, cudaStream_t cuStream
)
{
    const int op_increment = 65535;

    for(int op_start = 0; op_start < batchCount; op_start += op_increment)
	{
		int batch_size = std::min(op_increment, batchCount - op_start);

        dim3 dimBlock(ldt); // 1d thread
        dim3 dimGrid(max_ilen, max_jlen, batch_size);
        size_t sharedMemorySize = 3 * maxSuperSize * sizeof(int_t);

        scatterGPU_batch<Ftype><<<dimGrid, dimBlock, sharedMemorySize, cuStream>>>(
            iSt_batch + op_start, iEnd_batch + op_start, jSt_batch + op_start,
            jEnd_batch + op_start, gemmBuff_ptrs + op_start, LDgemmBuff_batch + op_start,
            lpanels + op_start, upanels + op_start, dA
        );
    }
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSchurComplementUpdateGPU(
    int streamId,
    int_t k, // the k-th panel or supernode
    xlpanel_t<Ftype> &lpanel, xupanel_t<Ftype> &upanel)
{
    if (lpanel.isEmpty() || upanel.isEmpty())
        return 0;

#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
    typename xLUstruct_t<Ftype>::SymTimingScope sym_timer(
        (options != NULL && options->SymFact == YES) ? this : NULL,
        xLUstruct_t<Ftype>::SYM_GPU3D_T_SCHUR_UPDATE);
#endif
#ifdef HAVE_CUDA
    if (options != NULL && options->SymFact == YES &&
        Pr == 1 && Pc == 1 &&
        grid3d->cscp.Np <= 1 && grid3d->rscp.Np <= 1)
    {
        int event_id = 0;
        if (k >= 0 && static_cast<size_t>(k) < symPanelReadyEventIds.size() &&
            symPanelReadyEventIds[k] >= 0)
            event_id = symPanelReadyEventIds[k];
        gpuErrchk(cudaStreamWaitEvent(A_gpu.cuStreams[streamId],
                                      A_gpu.panelReadyEvents[event_id], 0));
    }
#endif

    int_t st_lb = 0;
    if (myrow == krow(k))
        st_lb = 1;

    int_t nlb = lpanel.nblocks();
    int_t nub = upanel.nblocks();

    int iSt = st_lb;
    int iEnd = iSt;

    int nrows = lpanel.stRow(nlb) - lpanel.stRow(st_lb);
    int ncols = upanel.nzcols();

    int maxGemmRows = nrows;
    int maxGemmCols = ncols;
    // entire gemm doesn't fit in gemm buffer
    if (nrows * ncols > A_gpu.gemmBufferSize)
    {
        int maxGemmOpSize = (int)sqrt(A_gpu.gemmBufferSize);
        int numberofRowChunks = (nrows + maxGemmOpSize - 1) / maxGemmOpSize;
        maxGemmRows = nrows / numberofRowChunks;
        maxGemmCols = A_gpu.gemmBufferSize / maxGemmRows;
        /* printf("buffer exceeded! k = %d, st_lb = %d, nlb = %d, nrowsXncols %d, maxGemRows %d, maxGemmCols %d\n",
	   k, st_lb, nlb, nrows*ncols, maxGemmRows, maxGemmCols);*/
    }

    while (iEnd < nlb)
    {
        iSt = iEnd;
        iEnd = lpanel.getEndBlock(iSt, maxGemmRows);

        assert(iEnd > iSt);
        int jSt = 0;
        int jEnd = 0;
        while (jEnd < nub)
        {
            jSt = jEnd;
            jEnd = upanel.getEndBlock(jSt, maxGemmCols);
            assert(jEnd > jSt);
            cublasHandle_t handle = A_gpu.cuHandles[streamId];
            cudaStream_t cuStream = A_gpu.cuStreams[streamId];
            cublasSetStream(handle, cuStream);
            int gemm_m = lpanel.stRow(iEnd) - lpanel.stRow(iSt);
            int gemm_n = upanel.stCol(jEnd) - upanel.stCol(jSt);
            int gemm_k = supersize(k);
#if 0
            printf("k = %d, iSt = %d, iEnd = %d, jst = %d, jend = %d\n", k, iSt, iEnd, jSt, jEnd);
            printf("m=%d, n=%d, k=%d\n", gemm_m, gemm_n, gemm_k);
	    fflush(stdout);
#endif

            Ftype alpha = one<Ftype>();
            Ftype beta = zeroT<Ftype>();
            // cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            //             gemm_m, gemm_n, gemm_k, &alpha,
            //             lpanel.blkPtrGPU(iSt), lpanel.LDA(),
            //             upanel.blkPtrGPU(jSt), upanel.LDA(), &beta,
            //             A_gpu.gpuGemmBuffs[streamId], gemm_m);
            myCublasGemm<Ftype>(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                gemm_m, gemm_n, gemm_k, &alpha,
                                lpanel.blkPtrGPU(iSt), lpanel.LDA(),
                                upanel.blkPtrGPU(jSt), upanel.LDA(), &beta,
                                A_gpu.gpuGemmBuffs[streamId], gemm_m);

            scatterGPU_driver<Ftype>(
                iSt, iEnd, jSt, jEnd, A_gpu.gpuGemmBuffs[streamId], gemm_m,
                A_gpu.maxSuperSize, ldt, lpanel.gpuPanel, upanel.gpuPanel,
                dA_gpu, cuStream
            );
        }
    }
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
    double sym_schur_sync_t = SuperLU_timer_();
#endif
    gpuErrchk(cudaStreamSynchronize(A_gpu.cuStreams[streamId]));
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
    if (options != NULL && options->SymFact == YES)
        symTimingAdd(SYM_GPU3D_T_SCHUR_SYNC,
                     SuperLU_timer_() - sym_schur_sync_t);
#endif
    return 0;
} /* end dSchurComplementUpdateGPU */

template <typename Ftype>
int_t xLUstruct_t<Ftype>::lookAheadUpdateGPU(
    int streamId,
    int_t k, int_t laIdx, xlpanel_t<Ftype> &lpanel, xupanel_t<Ftype> &upanel)
{
    if (lpanel.isEmpty() || upanel.isEmpty())
        return 0;

#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
    typename xLUstruct_t<Ftype>::SymTimingScope sym_timer(
        (options != NULL && options->SymFact == YES) ? this : NULL,
        xLUstruct_t<Ftype>::SYM_GPU3D_T_LOOKAHEAD_UPDATE);
#endif
#ifdef HAVE_CUDA
    if (options != NULL && options->SymFact == YES &&
        Pr == 1 && Pc == 1 &&
        grid3d->cscp.Np <= 1 && grid3d->rscp.Np <= 1)
    {
        int event_id = 0;
        if (k >= 0 && static_cast<size_t>(k) < symPanelReadyEventIds.size() &&
            symPanelReadyEventIds[k] >= 0)
            event_id = symPanelReadyEventIds[k];
        gpuErrchk(cudaStreamWaitEvent(A_gpu.lookAheadLStream[streamId],
                                      A_gpu.panelReadyEvents[event_id], 0));
        gpuErrchk(cudaStreamWaitEvent(A_gpu.lookAheadUStream[streamId],
                                      A_gpu.panelReadyEvents[event_id], 0));
    }
#endif

    int_t st_lb = 0;
    if (myrow == krow(k))
        st_lb = 1;

    int_t nlb = lpanel.nblocks();
    int_t nub = upanel.nblocks();

    int_t laILoc = lpanel.find(laIdx);
    int_t laJLoc = upanel.find(laIdx);

    int iSt = st_lb;
    int jSt = 0;

    /* call look ahead update on Lpanel*/
    if (laJLoc != GLOBAL_BLOCK_NOT_FOUND)
        dSchurCompUpdatePartGPU(
            iSt, nlb, laJLoc, laJLoc + 1,
            k, lpanel, upanel,
            A_gpu.lookAheadLHandle[streamId], A_gpu.lookAheadLStream[streamId],
            A_gpu.lookAheadLGemmBuffer[streamId]);

    /* call look ahead update on Upanel*/
    if (laILoc != GLOBAL_BLOCK_NOT_FOUND)
    {
        dSchurCompUpdatePartGPU(
            laILoc, laILoc + 1, jSt, laJLoc,
            k, lpanel, upanel,
            A_gpu.lookAheadUHandle[streamId], A_gpu.lookAheadUStream[streamId],
            A_gpu.lookAheadUGemmBuffer[streamId]);
        dSchurCompUpdatePartGPU(
            laILoc, laILoc + 1, laJLoc + 1, nub,
            k, lpanel, upanel,
            A_gpu.lookAheadUHandle[streamId], A_gpu.lookAheadUStream[streamId],
            A_gpu.lookAheadUGemmBuffer[streamId]);
    }

    // checkCudaLocal(cudaStreamSynchronize(A_gpu.lookAheadLStream[streamId]));
    // checkCudaLocal(cudaStreamSynchronize(A_gpu.lookAheadUStream[streamId]));

    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::SyncLookAheadUpdate(int streamId)
{
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
    double sym_lookahead_sync_t = SuperLU_timer_();
#endif
    gpuErrchk(cudaStreamSynchronize(A_gpu.lookAheadLStream[streamId]));
    gpuErrchk(cudaStreamSynchronize(A_gpu.lookAheadUStream[streamId]));
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
    if (options != NULL && options->SymFact == YES)
        symTimingAdd(SYM_GPU3D_T_LOOKAHEAD_SYNC,
                     SuperLU_timer_() - sym_lookahead_sync_t);
#endif

    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSchurCompUpdateExcludeOneGPU(
    int streamId,
    int_t k, int_t ex, // suypernodes to be excluded
    xlpanel_t<Ftype> &lpanel, xupanel_t<Ftype> &upanel)
{
    if (lpanel.isEmpty() || upanel.isEmpty())
        return 0;

#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
    typename xLUstruct_t<Ftype>::SymTimingScope sym_timer(
        (options != NULL && options->SymFact == YES) ? this : NULL,
        xLUstruct_t<Ftype>::SYM_GPU3D_T_EXCLUDE_UPDATE);
#endif
#ifdef HAVE_CUDA
    if (options != NULL && options->SymFact == YES &&
        Pr == 1 && Pc == 1 &&
        grid3d->cscp.Np <= 1 && grid3d->rscp.Np <= 1)
    {
        int event_id = 0;
        if (k >= 0 && static_cast<size_t>(k) < symPanelReadyEventIds.size() &&
            symPanelReadyEventIds[k] >= 0)
            event_id = symPanelReadyEventIds[k];
        gpuErrchk(cudaStreamWaitEvent(A_gpu.cuStreams[streamId],
                                      A_gpu.panelReadyEvents[event_id], 0));
    }
#endif

    int_t st_lb = 0;
    if (myrow == krow(k))
        st_lb = 1;

    int_t nlb = lpanel.nblocks();
    int_t nub = upanel.nblocks();

    int_t exILoc = lpanel.find(ex);
    int_t exJLoc = upanel.find(ex);

    dSchurCompUpLimitedMem(
        streamId,
        st_lb, exILoc, 0, exJLoc,
        k, lpanel, upanel);

    dSchurCompUpLimitedMem(
        streamId,
        st_lb, exILoc, exJLoc + 1, nub,
        k, lpanel, upanel);

    int_t nextStI = exILoc + 1;
    if (exILoc == GLOBAL_BLOCK_NOT_FOUND)
        nextStI = st_lb;
    /*
    for j we don't need to change since, if exJLoc == GLOBAL_BLOCK_NOT_FOUND =-1
    then exJLoc+1 =0 will work out correctly as starting j
    */
    dSchurCompUpLimitedMem(
        streamId,
        nextStI, nlb, 0, exJLoc,
        k, lpanel, upanel);

    dSchurCompUpLimitedMem(
        streamId,
        nextStI, nlb, exJLoc + 1, nub,
        k, lpanel, upanel);

    // checkCudaLocal(cudaStreamSynchronize(A_gpu.cuStreams[streamId]));
    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSchurCompUpdatePartGPU(
    int_t iSt, int_t iEnd, int_t jSt, int_t jEnd,
    int_t k, xlpanel_t<Ftype> &lpanel, xupanel_t<Ftype> &upanel,
    cublasHandle_t handle, cudaStream_t cuStream,
    Ftype *gemmBuff)
{
    if (iSt >= iEnd || jSt >= jEnd)
        return 0;

    cublasSetStream(handle, cuStream);
    int gemm_m = lpanel.stRow(iEnd) - lpanel.stRow(iSt);
    int gemm_n = upanel.stCol(jEnd) - upanel.stCol(jSt);
    int gemm_k = supersize(k);
    Ftype alpha = one<Ftype>();
    Ftype beta = zeroT<Ftype>();
#ifndef NDEBUG
   // printf("m=%d, n=%d, k=%d\n", gemm_m, gemm_n, gemm_k);
#endif
    // cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
    //             gemm_m, gemm_n, gemm_k, &alpha,
    //             lpanel.blkPtrGPU(iSt), lpanel.LDA(),
    //             upanel.blkPtrGPU(jSt), upanel.LDA(), &beta,
    //             gemmBuff, gemm_m);

    myCublasGemm<Ftype>(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        gemm_m, gemm_n, gemm_k, &alpha,
                        lpanel.blkPtrGPU(iSt), lpanel.LDA(),
                        upanel.blkPtrGPU(jSt), upanel.LDA(), &beta,
                        gemmBuff, gemm_m);

    // setting up scatter
    dim3 dimBlock(ldt); // 1d thread
    dim3 dimGrid(iEnd - iSt, jEnd - jSt);
    size_t sharedMemorySize = 3 * A_gpu.maxSuperSize * sizeof(int_t);

    scatterGPU<Ftype><<<dimGrid, dimBlock, sharedMemorySize, cuStream>>>(
        iSt, jSt,
        gemmBuff, gemm_m,
        lpanel.gpuPanel, upanel.gpuPanel, dA_gpu);

    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSchurCompUpLimitedMem(
    int streamId,
    int_t lStart, int_t lEnd,
    int_t uStart, int_t uEnd,
    int_t k, xlpanel_t<Ftype> &lpanel, xupanel_t<Ftype> &upanel)
{

    if (lStart >= lEnd || uStart >= uEnd)
        return 0;
    int iSt = lStart;
    int iEnd = iSt;
    int nrows = lpanel.stRow(lEnd) - lpanel.stRow(lStart);
    int ncols = upanel.stCol(uEnd) - upanel.stCol(uStart);

    int maxGemmRows = nrows;
    int maxGemmCols = ncols;
    // entire gemm doesn't fit in gemm buffer
    if (nrows * ncols > A_gpu.gemmBufferSize)
    {
        int maxGemmOpSize = (int)sqrt(A_gpu.gemmBufferSize);
        int numberofRowChunks = (nrows + maxGemmOpSize - 1) / maxGemmOpSize;
        maxGemmRows = nrows / numberofRowChunks;
        maxGemmCols = A_gpu.gemmBufferSize / maxGemmRows;
    }

    while (iEnd < lEnd)
    {
        iSt = iEnd;
        iEnd = lpanel.getEndBlock(iSt, maxGemmRows);
        if (iEnd > lEnd)
            iEnd = lEnd;

        assert(iEnd > iSt);
        int jSt = uStart;
        int jEnd = uStart;
        while (jEnd < uEnd)
        {
            jSt = jEnd;
            jEnd = upanel.getEndBlock(jSt, maxGemmCols);
            if (jEnd > uEnd)
                jEnd = uEnd;

            cublasHandle_t handle = A_gpu.cuHandles[streamId];
            cudaStream_t cuStream = A_gpu.cuStreams[streamId];
            dSchurCompUpdatePartGPU(iSt, iEnd, jSt, jEnd,
                                    k, lpanel, upanel, handle, cuStream, A_gpu.gpuGemmBuffs[streamId]);
        }
    }

    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymSchurCompUpdatePartLLGPU(
    int_t iSt, int_t iEnd, int_t jSt, int_t jEnd,
    int_t k, xlpanel_t<Ftype> &lpanel,
    cublasHandle_t handle, cudaStream_t cuStream,
    Ftype *rawBlock, Ftype *gemmBuff)
{
    if (iSt >= iEnd || jSt >= jEnd || lpanel.isEmpty())
        return 0;
    if (symV2DiagBlocksGPU.size() != static_cast<size_t>(nsupers) ||
        symV2DiagBlocksGPU[k] == NULL)
        ABORT("SymFact V2 LL update diagonal block is missing.");

    int gemm_m = lpanel.stRow(iEnd) - lpanel.stRow(iSt);
    int gemm_n = lpanel.stRow(jEnd) - lpanel.stRow(jSt);
    int gemm_k = supersize(k);
    if (gemm_m <= 0 || gemm_n <= 0 || gemm_k <= 0)
        return 0;

    Ftype *raw_rhs = NULL;
    int raw_ld = gemm_n;
    if (superlu_sym_v2_wpanel_cache())
    {
        for (size_t slot = 0; slot < symV2RawPanelNodes.size(); ++slot)
        {
            if (symV2RawPanelNodes[slot] == k)
            {
                if (A_gpu.symV2RawPanelBufs[slot] == NULL ||
                    A_gpu.symV2RawPanelReadyEvents[slot] == NULL)
                    ABORT("SymFact V2 cached W panel is missing.");
                gpuErrchk(cudaStreamWaitEvent(
                    cuStream, A_gpu.symV2RawPanelReadyEvents[slot], 0));
                raw_rhs = A_gpu.symV2RawPanelBufs[slot] + lpanel.stRow(jSt);
                raw_ld = lpanel.LDA();
                break;
            }
        }
    }
    if (raw_rhs == NULL)
    {
        int threads = 256;
        int blocks = (gemm_k * gemm_n + threads - 1) / threads;
        buildSymRawLRangeGPU<Ftype><<<blocks, threads, 0, cuStream>>>(
            rawBlock, gemm_n, lpanel.gpuPanel, jSt, jEnd,
            symV2DiagBlocksGPU[k], gemm_k);
        gpuErrchk(cudaGetLastError());
        raw_rhs = rawBlock;
        raw_ld = gemm_n;
    }

    Ftype alpha = one<Ftype>();
    Ftype beta = zeroT<Ftype>();
    cublasSetStream(handle, cuStream);
    myCublasGemm<Ftype>(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                        gemm_m, gemm_n, gemm_k, &alpha,
                        lpanel.blkPtrGPU(iSt), lpanel.LDA(),
                        raw_rhs, raw_ld, &beta,
                        gemmBuff, gemm_m);

    scatterSymLowerRangeGPU_driver<Ftype>(
        iSt, iEnd, jSt, jEnd, gemmBuff, gemm_m,
        A_gpu.maxSuperSize, ldt, lpanel.gpuPanel, dA_gpu, cuStream);

    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymSchurCompUpLimitedMemLLGPU(
    int_t lStart, int_t lEnd,
    int_t jStart, int_t jEnd,
    int_t k, xlpanel_t<Ftype> &lpanel,
    cublasHandle_t handle, cudaStream_t cuStream,
    Ftype *rawBlock, Ftype *gemmBuff)
{
    if (lStart >= lEnd || jStart >= jEnd || lpanel.isEmpty())
        return 0;

    symV2ProbeLLRange(k, lpanel, lStart, lEnd, jStart, jEnd, NULL);

    if (!superlu_sym_v2_batch_schur_enabled())
    {
        for (int_t jSt = jStart; jSt < jEnd; ++jSt)
        {
            int_t jNext = jSt + 1;
            int ncols = lpanel.stRow(jNext) - lpanel.stRow(jSt);
            if (ncols <= 0)
                continue;

            int nrows = lpanel.stRow(lEnd) - lpanel.stRow(lStart);
            int maxGemmRows = nrows;
            if (static_cast<int64_t>(nrows) * ncols >
                static_cast<int64_t>(A_gpu.gemmBufferSize))
                maxGemmRows = SUPERLU_MAX(
                    1, static_cast<int>(A_gpu.gemmBufferSize / ncols));

            int_t iEnd = lStart;
            while (iEnd < lEnd)
            {
                int_t iSt = iEnd;
                iEnd = lpanel.getEndBlock(iSt, maxGemmRows);
                if (iEnd > lEnd)
                    iEnd = lEnd;
                if (iEnd <= iSt)
                    iEnd = iSt + 1;

                dSymSchurCompUpdatePartLLGPU(iSt, iEnd, jSt, jNext,
                                             k, lpanel, handle, cuStream,
                                             rawBlock, gemmBuff);
            }
        }
        return 0;
    }

    /* GPU3DV2_BATCH_SCHUR: group adjacent symmetric blocks. */
    const int nrows_total = lpanel.stRow(lEnd) - lpanel.stRow(lStart);
    const int ncols_total = lpanel.stRow(jEnd) - lpanel.stRow(jStart);
    if (nrows_total <= 0 || ncols_total <= 0)
        return 0;

    const int64_t gemm_capacity = SUPERLU_MAX(
        static_cast<int64_t>(1),
        static_cast<int64_t>(A_gpu.gemmBufferSize));
    int max_block_rows = 1;
    for (int_t ii = lStart; ii < lEnd; ++ii)
        max_block_rows = SUPERLU_MAX(max_block_rows,
                                     static_cast<int>(lpanel.nbrow(ii)));

    const bool envelope_requested =
        superlu_sym_v2_lower_envelope_enabled();
    bool gid_sorted = true;
    if (envelope_requested)
    {
        const int_t order_begin = SUPERLU_MIN(lStart, jStart);
        const int_t order_end = SUPERLU_MAX(lEnd, jEnd);
        for (int_t ii = order_begin + 1; ii < order_end; ++ii)
        {
            if (lpanel.gid(ii) < lpanel.gid(ii - 1))
            {
                gid_sorted = false;
                break;
            }
        }
    }
    const bool lower_envelope = envelope_requested && gid_sorted;

    int_t envelope_l_start = lStart;
    int_t jSt = jStart;
    while (jSt < jEnd)
    {
        if (lower_envelope)
        {
            const int_t min_col_gid = lpanel.gid(jSt);
            while (envelope_l_start < lEnd &&
                   lpanel.gid(envelope_l_start) < min_col_gid)
                ++envelope_l_start;
            if (envelope_l_start >= lEnd)
                break;
        }
        else
        {
            envelope_l_start = lStart;
        }

        const int group_nrows =
            lpanel.stRow(lEnd) - lpanel.stRow(envelope_l_start);
        if (group_nrows <= 0)
            break;

        int col_limit = superlu_sym_v2_batch_schur_col_limit(
            group_nrows, gemm_capacity);
        int64_t hard_col_limit = gemm_capacity /
                                 static_cast<int64_t>(max_block_rows);
        if (hard_col_limit < 1)
            hard_col_limit = 1;
        col_limit = SUPERLU_MIN(
            col_limit,
            static_cast<int>(SUPERLU_MIN(
                hard_col_limit, static_cast<int64_t>(2147483647L))));
        const int remaining_cols = static_cast<int>(
            lpanel.stRow(jEnd) - lpanel.stRow(jSt));
        col_limit = SUPERLU_MIN(col_limit, remaining_cols);
        col_limit = SUPERLU_MAX(col_limit, 1);

        int_t jNext = jSt + 1;
        while (jNext < jEnd)
        {
            int candidate_cols =
                lpanel.stRow(jNext + 1) - lpanel.stRow(jSt);
            if (candidate_cols > col_limit)
                break;
            ++jNext;
        }

        int group_cols = lpanel.stRow(jNext) - lpanel.stRow(jSt);
        if (group_cols <= 0)
        {
            jSt = jNext;
            continue;
        }

        int maxGemmRows = group_nrows;
        if (static_cast<int64_t>(group_nrows) * group_cols > gemm_capacity)
            maxGemmRows = static_cast<int>(
                gemm_capacity / static_cast<int64_t>(group_cols));
        maxGemmRows = SUPERLU_MAX(maxGemmRows, 1);

        int_t iEnd = envelope_l_start;
        while (iEnd < lEnd)
        {
            int_t iSt = iEnd;
            iEnd = lpanel.getEndBlock(iSt, maxGemmRows);
            if (iEnd > lEnd)
                iEnd = lEnd;
            if (iEnd <= iSt)
                iEnd = iSt + 1;

            dSymSchurCompUpdatePartLLGPU(iSt, iEnd, jSt, jNext,
                                         k, lpanel, handle, cuStream,
                                         rawBlock, gemmBuff);
        }
        jSt = jNext;
    }

    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymLookAheadUpdateLLGPU(
    int streamId,
    int_t k, int_t laIdx, xlpanel_t<Ftype> &lpanel)
{
    if (lpanel.isEmpty())
        return 0;

    int_t laLoc = lpanel.find(laIdx);
    if (laLoc == GLOBAL_BLOCK_NOT_FOUND)
        return 0;

    if (k >= 0 && static_cast<size_t>(k) < symPanelReadyEventIds.size() &&
        symPanelReadyEventIds[k] >= 0)
    {
        gpuErrchk(cudaStreamWaitEvent(A_gpu.lookAheadLStream[streamId],
                                      A_gpu.panelReadyEvents[symPanelReadyEventIds[k]], 0));
        gpuErrchk(cudaStreamWaitEvent(A_gpu.lookAheadUStream[streamId],
                                      A_gpu.panelReadyEvents[symPanelReadyEventIds[k]], 0));
    }

    cublasHandle_t colHandle = A_gpu.lookAheadLHandle[streamId];
    cudaStream_t colStream = A_gpu.lookAheadLStream[streamId];
    Ftype *colRawBlock = A_gpu.symPartnerLStageBufs[streamId];
    Ftype *colGemmBuff = A_gpu.lookAheadLGemmBuffer[streamId];

    cublasHandle_t rowHandle = A_gpu.lookAheadUHandle[streamId];
    cudaStream_t rowStream = A_gpu.lookAheadUStream[streamId];
    Ftype *rowRawBlock = A_gpu.lookAheadUGemmBuffer[streamId];
    Ftype *rowGemmBuff = A_gpu.gpuGemmBuffs[streamId];

    int_t st_lb = lpanel.haveDiag() ? 1 : 0;
    int_t nlb = lpanel.nblocks();
    int_t laGid = lpanel.gid(laLoc);
    for (int_t ii = st_lb; ii < nlb; ++ii)
    {
        if (ii == laLoc || lpanel.gid(ii) >= laGid)
            dSymSchurCompUpdatePartLLGPU(ii, ii + 1, laLoc, laLoc + 1,
                                         k, lpanel, colHandle, colStream,
                                         colRawBlock, colGemmBuff);
        else
            dSymSchurCompUpdatePartLLGPU(laLoc, laLoc + 1, ii, ii + 1,
                                         k, lpanel, rowHandle, rowStream,
                                         rowRawBlock, rowGemmBuff);
    }

    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymSchurCompUpdateExcludeOneLLGPU(
    int streamId,
    int_t k, int_t ex, xlpanel_t<Ftype> &lpanel)
{
    if (lpanel.isEmpty())
        return 0;

    if (k >= 0 && static_cast<size_t>(k) < symPanelReadyEventIds.size() &&
        symPanelReadyEventIds[k] >= 0)
        gpuErrchk(cudaStreamWaitEvent(A_gpu.cuStreams[streamId],
                                      A_gpu.panelReadyEvents[symPanelReadyEventIds[k]], 0));

    int_t st_lb = lpanel.haveDiag() ? 1 : 0;
    int_t nlb = lpanel.nblocks();
    int_t exLoc = lpanel.find(ex);
    cublasHandle_t handle = A_gpu.cuHandles[streamId];
    cudaStream_t cuStream = A_gpu.cuStreams[streamId];
    Ftype *rawBlock = A_gpu.lookAheadUGemmBuffer[streamId];
    Ftype *gemmBuff = A_gpu.gpuGemmBuffs[streamId];

    if (exLoc == GLOBAL_BLOCK_NOT_FOUND)
    {
        dSymSchurCompUpLimitedMemLLGPU(st_lb, nlb, st_lb, nlb,
                                       k, lpanel, handle, cuStream,
                                       rawBlock, gemmBuff);
    }
    else
    {
        dSymSchurCompUpLimitedMemLLGPU(st_lb, exLoc, st_lb, exLoc,
                                       k, lpanel, handle, cuStream,
                                       rawBlock, gemmBuff);
        dSymSchurCompUpLimitedMemLLGPU(exLoc + 1, nlb, st_lb, exLoc,
                                       k, lpanel, handle, cuStream,
                                       rawBlock, gemmBuff);
        dSymSchurCompUpLimitedMemLLGPU(exLoc + 1, nlb, exLoc + 1, nlb,
                                       k, lpanel, handle, cuStream,
                                       rawBlock, gemmBuff);
    }

    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymSchurCompUpdatePartWithLFragmentsGPU(
    int_t iSt, int_t iEnd, int_t jSt, int_t jEnd,
    int_t k, xlpanel_t<Ftype> &lpanel,
    const int_t *frag_index, Ftype *frag_val,
    cublasHandle_t handle, cudaStream_t cuStream,
    Ftype *gemmBuff)
{
    if (iSt >= iEnd || jSt >= jEnd || lpanel.isEmpty() ||
        frag_index == NULL || frag_val == NULL)
        return 0;
    if (k < 0 || static_cast<size_t>(k) >= symV2PartnerLRecvIndex.size())
        ABORT("SymFact V2 L-fragment metadata is missing.");
    const std::vector<int_t> &frag = symV2PartnerLRecvIndex[k];
    if (frag.empty())
        return 0;

    int_t frag_nblocks = frag[0];
    int_t frag_lda = frag[1];
    if (frag_nblocks <= 0 || frag_lda <= 0)
        return 0;

    jSt = SUPERLU_MAX((int_t)0, jSt);
    jEnd = SUPERLU_MIN(jEnd, frag_nblocks);
    if (jSt >= jEnd)
        return 0;

    if (!superlu_sym_v2_batch_schur_enabled())
    {
        for (int_t jj = jSt; jj < jEnd; ++jj)
        {
            int_t st_ptr = LPANEL_HEADER_SIZE + frag_nblocks + jj;
            int_t frag_start = frag[st_ptr];
            int_t frag_end = frag[st_ptr + 1];
            int gemm_m = lpanel.stRow(iEnd) - lpanel.stRow(iSt);
            int gemm_n = frag_end - frag_start;
            int gemm_k = supersize(k);
            if (gemm_m <= 0 || gemm_n <= 0 || gemm_k <= 0)
                continue;

            Ftype alpha = one<Ftype>();
            Ftype beta = zeroT<Ftype>();
            cublasSetStream(handle, cuStream);
            myCublasGemm<Ftype>(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                gemm_m, gemm_n, gemm_k, &alpha,
                                lpanel.blkPtrGPU(iSt), lpanel.LDA(),
                                frag_val + frag_start, frag_lda, &beta,
                                gemmBuff, gemm_m);

            scatterSymLowerLFragmentRangeGPU_driver<Ftype>(
                iSt, iEnd, jj, jj + 1, gemmBuff, gemm_m,
                A_gpu.maxSuperSize, ldt, lpanel.gpuPanel,
                const_cast<int_t *>(frag_index), frag_val,
                dA_gpu, cuStream);
        }
        return 0;
    }

    /* GPU3DV2_BATCH_SCHUR: group adjacent symmetric blocks. */
    const int_t starts = LPANEL_HEADER_SIZE + frag_nblocks;
    const int_t frag_start = frag[starts + jSt];
    const int_t frag_end = frag[starts + jEnd];
    const int gemm_m = lpanel.stRow(iEnd) - lpanel.stRow(iSt);
    const int gemm_n = frag_end - frag_start;
    const int gemm_k = supersize(k);
    if (gemm_m <= 0 || gemm_n <= 0 || gemm_k <= 0)
        return 0;

    Ftype alpha = one<Ftype>();
    Ftype beta = zeroT<Ftype>();
    cublasSetStream(handle, cuStream);
    myCublasGemm<Ftype>(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                        gemm_m, gemm_n, gemm_k, &alpha,
                        lpanel.blkPtrGPU(iSt), lpanel.LDA(),
                        frag_val + frag_start, frag_lda, &beta,
                        gemmBuff, gemm_m);

    scatterSymLowerLFragmentRangeGPU_driver<Ftype>(
        iSt, iEnd, jSt, jEnd, gemmBuff, gemm_m,
        A_gpu.maxSuperSize, ldt, lpanel.gpuPanel,
        const_cast<int_t *>(frag_index), frag_val,
        dA_gpu, cuStream);

    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymSchurCompUpLimitedMemWithLFragmentsGPU(
    int_t lStart, int_t lEnd,
    int_t fragStart, int_t fragEnd,
    int_t k, xlpanel_t<Ftype> &lpanel,
    const int_t *frag_index, Ftype *frag_val,
    cublasHandle_t handle, cudaStream_t cuStream,
    Ftype *gemmBuff)
{
    if (lStart >= lEnd || fragStart >= fragEnd || lpanel.isEmpty() ||
        frag_index == NULL || frag_val == NULL)
        return 0;
    if (k < 0 || static_cast<size_t>(k) >= symV2PartnerLRecvIndex.size())
        ABORT("SymFact V2 L-fragment metadata is missing.");
    const std::vector<int_t> &frag = symV2PartnerLRecvIndex[k];
    if (frag.empty())
        return 0;

    int_t nlb = lpanel.nblocks();
    int_t nfrag = frag[0];
    lStart = SUPERLU_MAX((int_t)0, lStart);
    fragStart = SUPERLU_MAX((int_t)0, fragStart);
    lEnd = SUPERLU_MIN(lEnd, nlb);
    fragEnd = SUPERLU_MIN(fragEnd, nfrag);
    if (lStart >= lEnd || fragStart >= fragEnd)
        return 0;

    symV2ProbeLLRange(k, lpanel, lStart, lEnd,
                      fragStart, fragEnd, &frag);

    if (!superlu_sym_v2_batch_schur_enabled())
    {
        for (int_t jSt = fragStart; jSt < fragEnd; ++jSt)
        {
            int_t st_ptr = LPANEL_HEADER_SIZE + nfrag + jSt;
            int ncols = frag[st_ptr + 1] - frag[st_ptr];
            if (ncols <= 0)
                continue;

            int nrows = lpanel.stRow(lEnd) - lpanel.stRow(lStart);
            int maxGemmRows = nrows;
            if (static_cast<int64_t>(nrows) * ncols >
                static_cast<int64_t>(A_gpu.gemmBufferSize))
                maxGemmRows = SUPERLU_MAX(
                    1, static_cast<int>(A_gpu.gemmBufferSize / ncols));

            int_t iEnd = lStart;
            while (iEnd < lEnd)
            {
                int_t iSt = iEnd;
                iEnd = lpanel.getEndBlock(iSt, maxGemmRows);
                if (iEnd > lEnd)
                    iEnd = lEnd;
                if (iEnd <= iSt)
                    iEnd = iSt + 1;

                dSymSchurCompUpdatePartWithLFragmentsGPU(
                    iSt, iEnd, jSt, jSt + 1,
                    k, lpanel, frag_index, frag_val,
                    handle, cuStream, gemmBuff);
            }
        }
        return 0;
    }

    /* GPU3DV2_BATCH_SCHUR: group adjacent symmetric blocks. */
    const int_t starts = LPANEL_HEADER_SIZE + nfrag;
    const int nrows_total = lpanel.stRow(lEnd) - lpanel.stRow(lStart);
    const int ncols_total = frag[starts + fragEnd] -
                            frag[starts + fragStart];
    if (nrows_total <= 0 || ncols_total <= 0)
        return 0;

    const int64_t gemm_capacity = SUPERLU_MAX(
        static_cast<int64_t>(1),
        static_cast<int64_t>(A_gpu.gemmBufferSize));
    int max_block_rows = 1;
    for (int_t ii = lStart; ii < lEnd; ++ii)
        max_block_rows = SUPERLU_MAX(max_block_rows,
                                     static_cast<int>(lpanel.nbrow(ii)));

    const bool envelope_requested =
        superlu_sym_v2_lower_envelope_enabled();
    bool row_gid_sorted = true;
    bool frag_gid_sorted = true;
    if (envelope_requested)
    {
        for (int_t ii = lStart + 1; ii < lEnd; ++ii)
        {
            if (lpanel.gid(ii) < lpanel.gid(ii - 1))
            {
                row_gid_sorted = false;
                break;
            }
        }
        for (int_t jj = fragStart + 1; jj < fragEnd; ++jj)
        {
            if (frag[LPANEL_HEADER_SIZE + jj] <
                frag[LPANEL_HEADER_SIZE + jj - 1])
            {
                frag_gid_sorted = false;
                break;
            }
        }
    }
    const bool lower_envelope =
        envelope_requested && row_gid_sorted && frag_gid_sorted;

    int_t envelope_l_start = lStart;
    int_t jSt = fragStart;
    while (jSt < fragEnd)
    {
        if (lower_envelope)
        {
            const int_t min_col_gid = frag[LPANEL_HEADER_SIZE + jSt];
            while (envelope_l_start < lEnd &&
                   lpanel.gid(envelope_l_start) < min_col_gid)
                ++envelope_l_start;
            if (envelope_l_start >= lEnd)
                break;
        }
        else
        {
            envelope_l_start = lStart;
        }

        const int group_nrows =
            lpanel.stRow(lEnd) - lpanel.stRow(envelope_l_start);
        if (group_nrows <= 0)
            break;

        int col_limit = superlu_sym_v2_batch_schur_col_limit(
            group_nrows, gemm_capacity);
        int64_t hard_col_limit = gemm_capacity /
                                 static_cast<int64_t>(max_block_rows);
        if (hard_col_limit < 1)
            hard_col_limit = 1;
        col_limit = SUPERLU_MIN(
            col_limit,
            static_cast<int>(SUPERLU_MIN(
                hard_col_limit, static_cast<int64_t>(2147483647L))));
        const int remaining_cols = static_cast<int>(
            frag[starts + fragEnd] - frag[starts + jSt]);
        col_limit = SUPERLU_MIN(col_limit, remaining_cols);
        col_limit = SUPERLU_MAX(col_limit, 1);

        int_t jNext = jSt + 1;
        while (jNext < fragEnd)
        {
            int candidate_cols = frag[starts + jNext + 1] -
                                 frag[starts + jSt];
            if (candidate_cols > col_limit)
                break;
            ++jNext;
        }

        int group_cols = frag[starts + jNext] - frag[starts + jSt];
        if (group_cols <= 0)
        {
            jSt = jNext;
            continue;
        }

        int maxGemmRows = group_nrows;
        if (static_cast<int64_t>(group_nrows) * group_cols > gemm_capacity)
            maxGemmRows = static_cast<int>(
                gemm_capacity / static_cast<int64_t>(group_cols));
        maxGemmRows = SUPERLU_MAX(maxGemmRows, 1);

        int_t iEnd = envelope_l_start;
        while (iEnd < lEnd)
        {
            int_t iSt = iEnd;
            iEnd = lpanel.getEndBlock(iSt, maxGemmRows);
            if (iEnd > lEnd)
                iEnd = lEnd;
            if (iEnd <= iSt)
                iEnd = iSt + 1;

            dSymSchurCompUpdatePartWithLFragmentsGPU(
                iSt, iEnd, jSt, jNext,
                k, lpanel, frag_index, frag_val,
                handle, cuStream, gemmBuff);
        }
        jSt = jNext;
    }

    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymLookAheadUpdateWithLFragmentsGPU(
    int streamId,
    int_t k, int_t laIdx, xlpanel_t<Ftype> &lpanel)
{
    if (lpanel.isEmpty())
        return 0;
    if (k < 0 || static_cast<size_t>(k) >= symV2PartnerLRecvIndex.size())
        ABORT("SymFact V2 L-fragment metadata is missing.");
    const std::vector<int_t> &frag = symV2PartnerLRecvIndex[k];
    if (frag.empty())
        return 0;

    int_t st_lb = lpanel.haveDiag() ? 1 : 0;
    int_t nlb = lpanel.nblocks();
    int_t nfrag = frag[0];
    int_t laILoc = lpanel.find(laIdx);
    int_t laJLoc = GLOBAL_BLOCK_NOT_FOUND;
    for (int_t jj = 0; jj < nfrag; ++jj)
    {
        if (frag[LPANEL_HEADER_SIZE + jj] == laIdx)
        {
            laJLoc = jj;
            break;
        }
    }

    int_t *frag_index = A_gpu.symPartnerLidxRecvBufs[streamId];
    Ftype *frag_val = A_gpu.symPartnerLvalRecvBufs[streamId];
    if (frag_index == NULL || frag_val == NULL)
        ABORT("SymFact V2 L-fragment GPU buffers are missing.");

    cublasHandle_t colHandle = A_gpu.lookAheadLHandle[streamId];
    cudaStream_t colStream = A_gpu.lookAheadLStream[streamId];
    Ftype *colGemmBuff = A_gpu.lookAheadLGemmBuffer[streamId];

    cublasHandle_t rowHandle = A_gpu.lookAheadUHandle[streamId];
    cudaStream_t rowStream = A_gpu.lookAheadUStream[streamId];
    Ftype *rowGemmBuff = A_gpu.gpuGemmBuffs[streamId];

    if (laJLoc != GLOBAL_BLOCK_NOT_FOUND)
        dSymSchurCompUpLimitedMemWithLFragmentsGPU(st_lb, nlb,
                                                   laJLoc, laJLoc + 1,
                                                   k, lpanel, frag_index,
                                                   frag_val, colHandle,
                                                   colStream, colGemmBuff);

    if (laILoc != GLOBAL_BLOCK_NOT_FOUND)
    {
        if (laJLoc == GLOBAL_BLOCK_NOT_FOUND)
        {
            dSymSchurCompUpLimitedMemWithLFragmentsGPU(laILoc, laILoc + 1,
                                                       0, nfrag,
                                                       k, lpanel, frag_index,
                                                       frag_val, rowHandle,
                                                       rowStream, rowGemmBuff);
        }
        else
        {
            dSymSchurCompUpLimitedMemWithLFragmentsGPU(laILoc, laILoc + 1,
                                                       0, laJLoc,
                                                       k, lpanel, frag_index,
                                                       frag_val, rowHandle,
                                                       rowStream, rowGemmBuff);
            dSymSchurCompUpLimitedMemWithLFragmentsGPU(laILoc, laILoc + 1,
                                                       laJLoc + 1, nfrag,
                                                       k, lpanel, frag_index,
                                                       frag_val, rowHandle,
                                                       rowStream, rowGemmBuff);
        }
    }

    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymSchurCompUpdateExcludeOneWithLFragmentsGPU(
    int streamId,
    int_t k, int_t ex, xlpanel_t<Ftype> &lpanel)
{
    if (lpanel.isEmpty())
        return 0;
    if (k < 0 || static_cast<size_t>(k) >= symV2PartnerLRecvIndex.size())
        ABORT("SymFact V2 L-fragment metadata is missing.");
    const std::vector<int_t> &frag = symV2PartnerLRecvIndex[k];
    if (frag.empty())
        return 0;

    int_t st_lb = lpanel.haveDiag() ? 1 : 0;
    int_t nlb = lpanel.nblocks();
    int_t nfrag = frag[0];
    int_t exILoc = lpanel.find(ex);
    int_t exJLoc = GLOBAL_BLOCK_NOT_FOUND;
    for (int_t jj = 0; jj < nfrag; ++jj)
    {
        if (frag[LPANEL_HEADER_SIZE + jj] == ex)
        {
            exJLoc = jj;
            break;
        }
    }

    int_t *frag_index = A_gpu.symPartnerLidxRecvBufs[streamId];
    Ftype *frag_val = A_gpu.symPartnerLvalRecvBufs[streamId];
    if (frag_index == NULL || frag_val == NULL)
        ABORT("SymFact V2 L-fragment GPU buffers are missing.");

    cublasHandle_t handle = A_gpu.cuHandles[streamId];
    cudaStream_t cuStream = A_gpu.cuStreams[streamId];
    Ftype *gemmBuff = A_gpu.gpuGemmBuffs[streamId];

    auto update_i_range = [&](int_t ist, int_t iend)
    {
        if (ist >= iend)
            return;
        if (exJLoc == GLOBAL_BLOCK_NOT_FOUND)
        {
            dSymSchurCompUpLimitedMemWithLFragmentsGPU(ist, iend, 0, nfrag,
                                                       k, lpanel, frag_index,
                                                       frag_val, handle,
                                                       cuStream, gemmBuff);
        }
        else
        {
            dSymSchurCompUpLimitedMemWithLFragmentsGPU(ist, iend, 0, exJLoc,
                                                       k, lpanel, frag_index,
                                                       frag_val, handle,
                                                       cuStream, gemmBuff);
            dSymSchurCompUpLimitedMemWithLFragmentsGPU(ist, iend,
                                                       exJLoc + 1, nfrag,
                                                       k, lpanel, frag_index,
                                                       frag_val, handle,
                                                       cuStream, gemmBuff);
        }
    };

    if (exILoc == GLOBAL_BLOCK_NOT_FOUND)
    {
        update_i_range(st_lb, nlb);
    }
    else
    {
        update_i_range(st_lb, exILoc);
        update_i_range(exILoc + 1, nlb);
    }

    return 0;
}

int getMPIProcsPerGPU()
{
    if (!(getenv("MPI_PROCESS_PER_GPU")))
    {
        return 1;
    } else {
        int devCount;
        cudaGetDeviceCount(&devCount);
        int envCount = atoi(getenv("MPI_PROCESS_PER_GPU"));
        envCount = SUPERLU_MAX(envCount, 1);
        printf("MPI_PROCESS_PER_GPU=%d, devCount=%d\n", envCount, devCount);
        return SUPERLU_MIN(envCount, devCount);
    }
}

// #define USABLE_GPU_MEM_FRACTION 0.9

size_t getGPUMemPerProcs(MPI_Comm baseCommunicator)
{
    size_t mfree, mtotal;
    // TODO: shared memory communicator should be part of
    //  LU struct
    //  MPI_Comm sharedComm;
    //  MPI_Comm_split_type(baseCommunicator, MPI_COMM_TYPE_SHARED,
    //                      0, MPI_INFO_NULL, &sharedComm);
    //  MPI_Barrier(sharedComm);
    cudaMemGetInfo(&mfree, &mtotal);
    // MPI_Barrier(sharedComm);
    // MPI_Comm_free(&sharedComm);
#if 0
    printf("Total memory %zu & free memory %zu\n", mtotal, mfree);
#endif
    //return (size_t)(USABLE_GPU_MEM_FRACTION * (Ftype)mfree) / getMPIProcsPerGPU();
    return (size_t)(USABLE_GPU_MEM_FRACTION * (double)mfree) / getMPIProcsPerGPU();
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::setLUstruct_GPU()
{
    int i, stream;
    xlu_sym_gpu3d_trace_gpu_setup(grid3d, "enter setLUstruct_GPU");
    double tSetupGpuMemEstimate = SuperLU_timer_();

#if (DEBUGlevel >= 1)
    int iam = 0;
    CHECK_MALLOC(iam, "Enter setLUstruct_GPU()"); fflush(stdout);
#endif

    A_gpu.Pr = Pr;
    A_gpu.Pc = Pc;
    A_gpu.maxSuperSize = ldt;

    /* Sherry: this mapping may be inefficient on Frontier */
    /*Mapping to device*/
    int deviceCount;
    cudaGetDeviceCount(&deviceCount); // How many GPUs?
    int device_id = grid3d->iam % deviceCount;
    cudaSetDevice(device_id);
    xlu_sym_gpu3d_trace_gpu_setup(grid3d, "setLUstruct_GPU after cudaSetDevice");

    double tRegion[5];
    size_t useableGPUMem = getGPUMemPerProcs(grid3d->comm);
    /**
     *  Memory is divided into two parts data memory and buffer memory
     *  data memory is used for useful data
     *  bufferMemory is used for buffers
     * */
    size_t memReqData = 0;

    /*Memory for XSUP*/
    memReqData += (nsupers + 1) * sizeof(int_t);

    tRegion[0] = SuperLU_timer_();

    size_t totalNzvalSize = 0; /* too big for gemmBufferSize */
    size_t max_gemmCsize = 0;  /* Sherry added 2/20/2023 */
    size_t max_nzrow = 0;  /* Yang added 10/20/2023 */
    size_t max_nzcol = 0;
    bool sym_v2_mode = (options->SymFact == YES && symGPU3DVersion == 2);
    bool need_u_panel_storage = needsUPanelStorage();
    int_t localPanelGpuCount = symV2PanelCount();
    int_t localRowGpuCount = symV2RowCount();
    int_t localPanelGpuAlloc = SUPERLU_MAX((int_t)1, localPanelGpuCount);
    int_t localRowGpuAlloc = SUPERLU_MAX((int_t)1, localRowGpuCount);
    A_gpu.useSymV2PanelIndex = sym_v2_mode ? 1 : 0;
    A_gpu.symV2PanelLocalIndex = NULL;
    size_t sym_diag_buf_dim = static_cast<size_t>(SUPERLU_MAX((int_t)1, ldt));
    if (sym_v2_mode)
    {
        sym_diag_buf_dim = 1;
        for (int_t k = 0; k < nsupers; ++k)
        {
            if (symV2PanelRoot(k) == mycol)
                sym_diag_buf_dim =
                    SUPERLU_MAX(sym_diag_buf_dim,
                                static_cast<size_t>(SUPERLU_MAX((int_t)1,
                                                                supersize(k))));
        }
    }
    size_t sym_diag_buf_elems =
        xlu_checked_product(sym_diag_buf_dim, sym_diag_buf_dim,
                            "GPU diagonal factor buffer");

    /*Memory for lpapenl and upanel Data*/
    for (i = 0; i < localPanelGpuCount; ++i)
    {
        int_t gid = symV2PanelGid(i);
        if (gid < nsupers && isNodeInMyGrid[gid] == 1)
        {
            memReqData += lPanelVec[i].totalSize();
            totalNzvalSize += lPanelVec[i].nzvalSize();
            if(lPanelVec[i].nzvalSize()>0)
                max_nzrow = SUPERLU_MAX(lPanelVec[i].nzrows(),max_nzrow);
	    //max_gemmCsize = SUPERoLU_MAX(max_gemmCsize, ???);
        }
    }
    for (i = 0; i < localRowGpuCount; ++i)
    {
        int_t gid = symV2RowGid(i);
        if (gid < nsupers && isNodeInMyGrid[gid] == 1)
        {
            if (!sym_v2_mode)
            {
                memReqData += sizeof(int_t) * uPanelVec[i].indexSize();
                memReqData += sizeof(Ftype) * uPanelVec[i].nzvalSize();
                totalNzvalSize += uPanelVec[i].nzvalSize();
            }
            if(!sym_v2_mode && uPanelVec[i].nzvalSize()>0)
                max_nzcol = SUPERLU_MAX(uPanelVec[i].nzcols(),max_nzcol);
        }
    }
    max_gemmCsize = max_nzcol*max_nzrow;

    memReqData += localPanelGpuAlloc * sizeof(lpanelGPU_t);
    if (need_u_panel_storage)
        memReqData += localRowGpuAlloc * sizeof(upanelGPU_t);
    if (sym_v2_mode)
        memReqData += nsupers * sizeof(int_t);

    memReqData += sizeof(xLUstructGPU_t<Ftype>);

    // Per stream data
    // TODO: estimate based on ancestor size
    int_t maxBuffSize = sp_ienv_dist (8, options);
    int maxsup = sp_ienv_dist(3, options); // max. supernode size
    maxBuffSize = SUPERLU_MAX(maxsup * maxsup, maxBuffSize); // Sherry added 7/10/23

 #if 0
    A_gpu.gemmBufferSize = SUPERLU_MIN(maxBuffSize, totalNzvalSize);
 #else
    A_gpu.gemmBufferSize = SUPERLU_MIN(maxBuffSize, SUPERLU_MAX(max_gemmCsize,totalNzvalSize)); /* Yang added 10/20/2023 */
 #endif

    int_t u_recv_val_count = sym_v2_mode ? 0 : maxUvalCount;
    int_t u_recv_idx_count = sym_v2_mode ? 0 : maxUidxCount;
    int_t lookahead_u_val_count =
        sym_v2_mode ? maxLvalCount : maxUvalCount;

    size_t sym_v2_raw_panel_count =
        (sym_v2_mode && superlu_sym_v2_wpanel_cache())
            ? static_cast<size_t>(maxLvalCount)
            : 0;
    size_t dataPerStream = (3 * sizeof(Ftype) * maxLvalCount +
                            sizeof(Ftype) * sym_v2_raw_panel_count +
                            2 * sizeof(Ftype) * maxSymPartnerLvalCount +
                            sizeof(Ftype) * u_recv_val_count +
                            sizeof(Ftype) * lookahead_u_val_count +
                            2 * sizeof(int_t) * maxLidxCount +
                            sizeof(int_t) * maxSymPartnerLidxCount +
                            sizeof(int_t) * u_recv_idx_count +
                            A_gpu.gemmBufferSize * sizeof(Ftype) +
                            sym_diag_buf_elems * sizeof(Ftype));
#ifdef SLU_SYM_GPU3D_DEBUG_TRACE
    std::printf("[sym-gpu3d-trace] rank %d: GPU memory estimate free_per_rank=%zu memReqData=%zu dataPerStream=%zu totalNzval=%zu gemmBuffer=%zu diagDim=%zu diagElems=%zu maxLval=%lld maxUval=%lld maxLidx=%lld maxUidx=%lld maxSymPartnerLval=%lld maxSymPartnerLidx=%lld\n",
                (grid3d != NULL) ? grid3d->iam : -1,
                useableGPUMem, memReqData, dataPerStream, totalNzvalSize,
                A_gpu.gemmBufferSize, sym_diag_buf_dim, sym_diag_buf_elems,
                (long long)maxLvalCount, (long long)maxUvalCount,
                (long long)maxLidxCount, (long long)maxUidxCount,
                (long long)maxSymPartnerLvalCount,
                (long long)maxSymPartnerLidxCount);
    std::fflush(stdout);
#endif
    size_t two_stream_bytes =
        xlu_checked_product(2, dataPerStream,
                            "GPU two-stream buffer estimate");
    if (memReqData > static_cast<size_t>(-1) - two_stream_bytes)
        ABORT("GPU two-stream memory estimate overflows allocation size.");
    size_t required_two_stream_bytes = memReqData + two_stream_bytes;
    if (required_two_stream_bytes > useableGPUMem)
    {
        printf("Not enough memory on GPU: available=%zu required_for_2_streams=%zu "
               "memReqData=%zu dataPerStream=%zu gemmBuffer=%zu diagDim=%zu "
               "diagElems=%zu maxLval=%lld maxUval=%lld maxSymPartnerLval=%lld, exiting\n",
               useableGPUMem, required_two_stream_bytes, memReqData,
               dataPerStream, A_gpu.gemmBufferSize, sym_diag_buf_dim,
               sym_diag_buf_elems, (long long)maxLvalCount,
               (long long)maxUvalCount, (long long)maxSymPartnerLvalCount);
        exit(-1);
    }

    tRegion[0] = SuperLU_timer_() - tRegion[0];
    symV2SetupProfileAdd(SYM_V2_SETUP_GPU_MEM_ESTIMATE,
                         SuperLU_timer_() - tSetupGpuMemEstimate);
#if ( PRNTlevel>=1 )
    // print the time taken to estimate memory on GPU
    if (grid3d->iam == 0)
    {
        printf("GPU deviceCount=%d\n", deviceCount);
	printf("\t.. totalNzvalSize %ld, gemmBufferSize %ld, diagBufDim %ld, diagBufElems %ld\n",
	       (long) totalNzvalSize, (long) A_gpu.gemmBufferSize,
	       (long) sym_diag_buf_dim, (long) sym_diag_buf_elems);
    }
#endif

    /*Memory for lapenlPanel Data*/
    tRegion[1] = SuperLU_timer_();

    int_t maxNumberOfStream = (useableGPUMem - memReqData) / dataPerStream;

    int numberOfStreams = SUPERLU_MIN(getNumLookAhead(options), maxNumberOfStream);
    numberOfStreams = SUPERLU_MIN(numberOfStreams, MAX_CUDA_STREAMS);
    int rNumberOfStreams;
    MPI_Allreduce(&numberOfStreams, &rNumberOfStreams, 1,
                  MPI_INT, MPI_MIN, grid3d->comm);
    A_gpu.numCudaStreams = rNumberOfStreams;
    if (sym_v2_mode && superlu_sym_v2_wpanel_cache())
        symV2RawPanelNodes.assign(static_cast<size_t>(rNumberOfStreams), -1);

    int symGpuContract = (options->SymFact == YES)
        ? superlu_gpu3d_contract_for_setup() : 0;
    bool needGpuDiagFactor =
        (options->SymFact != YES) ||
        (symGpuContract == 1 || symGpuContract == 2);

#if ( PRNTlevel>=1 )
    if (!grid3d->iam)
        printf("Using %d CUDA LookAhead streams\n", rNumberOfStreams);
    // size_t totalMemoryRequired = memReqData + numberOfStreams * dataPerStream;
#endif

#if 0 /**** Old code ****/
    upanelGPU_t *uPanelVec_GPU = new upanelGPU_t[CEILING(nsupers, Pr)];
    lpanelGPU_t *lPanelVec_GPU = new lpanelGPU_t[CEILING(nsupers, Pc)];
    void *gpuBasePtr, *gpuCurrentPtr;
    cudaMalloc(&gpuBasePtr, totalMemoryRequired);
    gpuCurrentPtr = gpuBasePtr;

    A_gpu.xsup = (int_t *)gpuCurrentPtr;
    gpuCurrentPtr = (int_t *)gpuCurrentPtr + (nsupers + 1);
    cudaMemcpy(A_gpu.xsup, xsup, (nsupers + 1) * sizeof(int_t), cudaMemcpyHostToDevice);

    for (int i = 0; i < CEILING(nsupers, Pc); ++i)
    {
        if (i * Pc + mycol < nsupers && isNodeInMyGrid[i * Pc + mycol] == 1)
        {
            lPanelVec_GPU[i] = lPanelVec[i].copyToGPU(gpuCurrentPtr);
            gpuCurrentPtr = (char *)gpuCurrentPtr + lPanelVec[i].totalSize();
        }
    }
    A_gpu.lPanelVec = (xlpanelGPU_t<Ftype> *)gpuCurrentPtr;
    gpuCurrentPtr = (char *)gpuCurrentPtr + CEILING(nsupers, Pc) * sizeof(xlpanelGPU_t<Ftype>);
    cudaMemcpy(A_gpu.lPanelVec, lPanelVec_GPU,
               CEILING(nsupers, Pc) * sizeof(xlpanelGPU_t<Ftype>), cudaMemcpyHostToDevice);

    for (int i = 0; i < CEILING(nsupers, Pr); ++i)
    {
        if (i * Pr + myrow < nsupers && isNodeInMyGrid[i * Pr + myrow] == 1)
        {
            uPanelVec_GPU[i] = uPanelVec[i].copyToGPU(gpuCurrentPtr);
            gpuCurrentPtr = (char *)gpuCurrentPtr + uPanelVec[i].totalSize();
        }
    }
    A_gpu.uPanelVec = (xupanelGPU_t<Ftype> *)gpuCurrentPtr;
    gpuCurrentPtr = (char *)gpuCurrentPtr + CEILING(nsupers, Pr) * sizeof(xupanelGPU_t<Ftype>);
    cudaMemcpy(A_gpu.uPanelVec, uPanelVec_GPU,
               CEILING(nsupers, Pr) * sizeof(xupanelGPU_t<Ftype>), cudaMemcpyHostToDevice);

    for (int stream = 0; stream < A_gpu.numCudaStreams; stream++)
    {

        cudaStreamCreate(&A_gpu.cuStreams[stream]);
        gpuErrchk(cudaEventCreateWithFlags(&A_gpu.panelReadyEvents[stream],
                                           cudaEventDisableTiming));
        gpuErrchk(cudaEventCreateWithFlags(
            &A_gpu.symV2PartnerLPackReadyEvents[stream],
            cudaEventDisableTiming));
        gpuErrchk(cudaEventRecord(A_gpu.panelReadyEvents[stream],
                                  A_gpu.cuStreams[stream]));
        gpuErrchk(cudaEventRecord(
            A_gpu.symV2PartnerLPackReadyEvents[stream],
            A_gpu.cuStreams[stream]));
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
        gpuErrchk(cudaEventCreate(&A_gpu.diagD2HStartEvents[stream]));
        gpuErrchk(cudaEventCreate(&A_gpu.diagD2HEndEvents[stream]));
#endif
        cublasCreate(&A_gpu.cuHandles[stream]);
        A_gpu.LvalRecvBufs[stream] = (Ftype *)gpuCurrentPtr;
        gpuCurrentPtr = (Ftype *)gpuCurrentPtr + maxLvalCount;
        A_gpu.UvalRecvBufs[stream] = (Ftype *)gpuCurrentPtr;
        gpuCurrentPtr = (Ftype *)gpuCurrentPtr + maxUvalCount;
        A_gpu.symPartnerLvalRecvBufs[stream] = (Ftype *)gpuCurrentPtr;
        gpuCurrentPtr = (Ftype *)gpuCurrentPtr + maxSymPartnerLvalCount;
        A_gpu.LidxRecvBufs[stream] = (int_t *)gpuCurrentPtr;
        gpuCurrentPtr = (int_t *)gpuCurrentPtr + maxLidxCount;
        A_gpu.UidxRecvBufs[stream] = (int_t *)gpuCurrentPtr;
        gpuCurrentPtr = (int_t *)gpuCurrentPtr + maxUidxCount;
        A_gpu.symPartnerLidxRecvBufs[stream] = (int_t *)gpuCurrentPtr;
        gpuCurrentPtr = (int_t *)gpuCurrentPtr + maxSymPartnerLidxCount;

        A_gpu.gpuGemmBuffs[stream] = (Ftype *)gpuCurrentPtr;
        gpuCurrentPtr = (Ftype *)gpuCurrentPtr + A_gpu.gemmBufferSize;
        A_gpu.dFBufs[stream] = (Ftype *)gpuCurrentPtr;
        gpuCurrentPtr = (Ftype *)gpuCurrentPtr + ldt * ldt;

        /*lookAhead buffers and stream*/
        cublasCreate(&A_gpu.lookAheadLHandle[stream]);
        cudaStreamCreate(&A_gpu.lookAheadLStream[stream]);
        A_gpu.lookAheadLGemmBuffer[stream] = (Ftype *)gpuCurrentPtr;
        gpuCurrentPtr = (Ftype *)gpuCurrentPtr + maxLvalCount;
        cublasCreate(&A_gpu.lookAheadUHandle[stream]);
        cudaStreamCreate(&A_gpu.lookAheadUStream[stream]);
        A_gpu.lookAheadUGemmBuffer[stream] = (Ftype *)gpuCurrentPtr;
        gpuCurrentPtr = (Ftype *)gpuCurrentPtr + maxUvalCount;
    }
    // cudaCheckError();
    // allocate
    dA_gpu = (xLUstructGPU_t<Ftype> *)gpuCurrentPtr;

    cudaMemcpy(dA_gpu, &A_gpu, sizeof(xLUstructGPU_t<Ftype>), cudaMemcpyHostToDevice);
    gpuCurrentPtr = (xLUstructGPU_t<Ftype> *)gpuCurrentPtr + 1;

#else /* else of #if 0 ----> this is the current active code - Sherry */
    gpuErrchk(cudaMalloc(&A_gpu.xsup, (nsupers + 1) * sizeof(int_t)));
    gpuErrchk(cudaMemcpy(A_gpu.xsup, xsup, (nsupers + 1) * sizeof(int_t), cudaMemcpyHostToDevice));

    double tLsend, tUsend;
#if 0
    tLsend = SuperLU_timer_();
    xupanelGPU_t<Ftype> *uPanelVec_GPU = copyUpanelsToGPU();
    tLsend = SuperLU_timer_() - tLsend;
    tUsend = SuperLU_timer_();
    xlpanelGPU_t<Ftype> *lPanelVec_GPU = copyLpanelsToGPU();
    tUsend = SuperLU_timer_() - tUsend;
#else
    xupanelGPU_t<Ftype> *uPanelVec_GPU = NULL;
    if (need_u_panel_storage)
        uPanelVec_GPU = new xupanelGPU_t<Ftype>[localRowGpuAlloc]();
    xlpanelGPU_t<Ftype> *lPanelVec_GPU =
        new xlpanelGPU_t<Ftype>[localPanelGpuAlloc]();
    bool sym_v2_skip_u_gpu_setup =
        (options->SymFact == YES && symGPU3DVersion == 2);
    xlu_sym_gpu3d_trace_gpu_setup(grid3d, "setLUstruct_GPU before copy panels to GPU");
    tLsend = SuperLU_timer_();
    for (i = 0; i < localPanelGpuCount; ++i)
    {
        int_t gid = symV2PanelGid(i);
        if (gid < nsupers && isNodeInMyGrid[gid] == 1)
            lPanelVec_GPU[i] = lPanelVec[i].copyToGPU();
    }
    tLsend = SuperLU_timer_() - tLsend;
    symV2SetupProfileAdd(SYM_V2_SETUP_COPY_L_PANELS_TO_GPU, tLsend);
    tUsend = SuperLU_timer_();
    // cudaCheckError();
    for (i = 0; uPanelVec_GPU != NULL && i < localRowGpuCount; ++i)
    {
        int_t gid = symV2RowGid(i);
        if (gid < nsupers && isNodeInMyGrid[gid] == 1)
        {
            if (!sym_v2_skip_u_gpu_setup)
                uPanelVec_GPU[i] = uPanelVec[i].copyToGPU();
        }
    }
    tUsend = SuperLU_timer_() - tUsend;
    xlu_sym_gpu3d_trace_gpu_setup(grid3d, "setLUstruct_GPU after copy panels to GPU");
#endif
    tRegion[1] = SuperLU_timer_() - tRegion[1];

    if (sym_v2_mode)
    {
        double tSymV2IndexCopy = SuperLU_timer_();
        int_t *sym_v2_panel_index =
            xlu_sym_v2_panel_index_table<Ftype>(trf3Dpartition);
        if (sym_v2_panel_index == NULL)
            ABORT("SymFact V2 GPU panel-local index table is missing.");
        gpuErrchk(cudaMalloc(&A_gpu.symV2PanelLocalIndex,
                             nsupers * sizeof(int_t)));
        gpuErrchk(cudaMemcpy(A_gpu.symV2PanelLocalIndex,
                             sym_v2_panel_index,
                             nsupers * sizeof(int_t),
                             cudaMemcpyHostToDevice));
        symV2SetupProfileAdd(SYM_V2_SETUP_SYM_V2_INDEX_COPY,
                             SuperLU_timer_() - tSymV2IndexCopy);
    }

    double tGpuPanelStructCopy = SuperLU_timer_();
    gpuErrchk(cudaMalloc(&A_gpu.lPanelVec,
                         localPanelGpuAlloc * sizeof(xlpanelGPU_t<Ftype>)));
    gpuErrchk(cudaMemcpy(A_gpu.lPanelVec, lPanelVec_GPU,
               localPanelGpuAlloc * sizeof(xlpanelGPU_t<Ftype>),
               cudaMemcpyHostToDevice));
    A_gpu.uPanelVec = NULL;
    if (uPanelVec_GPU != NULL)
    {
        gpuErrchk(cudaMalloc(&A_gpu.uPanelVec,
                             localRowGpuAlloc * sizeof(xupanelGPU_t<Ftype>)));
        gpuErrchk(cudaMemcpy(A_gpu.uPanelVec, uPanelVec_GPU,
                   localRowGpuAlloc * sizeof(xupanelGPU_t<Ftype>),
                   cudaMemcpyHostToDevice));
    }
    symV2SetupProfileAdd(SYM_V2_SETUP_GPU_PANEL_STRUCT_COPY,
                         SuperLU_timer_() - tGpuPanelStructCopy);

    if (uPanelVec_GPU != NULL)
        delete [] uPanelVec_GPU;
    delete [] lPanelVec_GPU;

    tRegion[2] = SuperLU_timer_();
    int dfactBufSize = 1;
    if (needGpuDiagFactor)
    {
        // TODO: does it work with NULL pointer?
        cusolverDnHandle_t cusolverH = NULL;
        gpuCusolverErrchk(cusolverDnCreate(&cusolverH));

        int getrfBufSize = 0;
        gpuCusolverErrchk(cusolverDnDgetrf_bufferSize(cusolverH, ldt, ldt,
                                                      NULL, ldt,
                                                      &getrfBufSize));
        dfactBufSize = getrfBufSize;
        if (options->SymFact == YES)
        {
            int sytrfBufSize = 0;
            int sytriBufSize = 0;
            gpuCusolverErrchk(cusolverDnDsytrf_bufferSize(cusolverH, ldt,
                                                          NULL, ldt,
                                                          &sytrfBufSize));
            gpuCusolverErrchk(cusolverDnDsytri_bufferSize(cusolverH,
                                                          CUBLAS_FILL_MODE_LOWER,
                                                          ldt, NULL, ldt,
                                                          NULL, &sytriBufSize));
            dfactBufSize = SUPERLU_MAX(dfactBufSize, sytrfBufSize);
            dfactBufSize = SUPERLU_MAX(dfactBufSize, sytriBufSize);
        }

        gpuCusolverErrchk(cusolverDnDestroy(cusolverH));
    }
#if ( PRNTlevel >= 1 )
    printf("Size of dfactBuf is %d\n", dfactBufSize);
#endif
    tRegion[2] = SuperLU_timer_() - tRegion[2];
    symV2SetupProfileAdd(SYM_V2_SETUP_GPU_DIAG_FACTOR_SETUP, tRegion[2]);

    tRegion[3] = SuperLU_timer_();

    double tcuMalloc=SuperLU_timer_();
    double tPerStreamBufferAlloc = SuperLU_timer_();
    xlu_sym_gpu3d_trace_gpu_setup(grid3d, "setLUstruct_GPU before per-stream buffer allocation");

    /* Sherry: where are these freed ?? */
    for (stream = 0; stream < A_gpu.numCudaStreams; stream++)
    {
        gpuErrchk(cudaMalloc(&A_gpu.LvalRecvBufs[stream], sizeof(Ftype) * maxLvalCount));
        A_gpu.UvalRecvBufs[stream] = NULL;
        if (u_recv_val_count > 0)
            gpuErrchk(cudaMalloc(&A_gpu.UvalRecvBufs[stream],
                                 sizeof(Ftype) *
                                     static_cast<size_t>(u_recv_val_count)));
        gpuErrchk(cudaMalloc(&A_gpu.symPartnerLvalRecvBufs[stream],
                             sizeof(Ftype) *
                                 static_cast<size_t>(SUPERLU_MAX((int_t)1, maxSymPartnerLvalCount))));
        gpuErrchk(cudaMalloc(&A_gpu.symPartnerLStageBufs[stream],
                             sizeof(Ftype) *
                                 static_cast<size_t>(SUPERLU_MAX((int_t)1, maxSymPartnerLvalCount))));
        A_gpu.symV2RawPanelBufs[stream] = NULL;
        A_gpu.symV2RawPanelReadyEvents[stream] = NULL;
        if (sym_v2_mode && superlu_sym_v2_wpanel_cache())
            gpuErrchk(cudaMalloc(&A_gpu.symV2RawPanelBufs[stream],
                                 sizeof(Ftype) *
                                     static_cast<size_t>(SUPERLU_MAX((int_t)1,
                                                                     maxLvalCount))));
        gpuErrchk(cudaMalloc(&A_gpu.LidxRecvBufs[stream], sizeof(int_t) * maxLidxCount));
        A_gpu.UidxRecvBufs[stream] = NULL;
        if (u_recv_idx_count > 0)
            gpuErrchk(cudaMalloc(&A_gpu.UidxRecvBufs[stream],
                                 sizeof(int_t) *
                                     static_cast<size_t>(u_recv_idx_count)));
        gpuErrchk(cudaMalloc(&A_gpu.symPartnerLidxRecvBufs[stream],
                             sizeof(int_t) *
                                 static_cast<size_t>(SUPERLU_MAX((int_t)1,
                                                                 maxSymPartnerLidxCount))));
        A_gpu.diagFactWork[stream] = NULL;
        A_gpu.diagFactIPIV[stream] = NULL;
        A_gpu.diagFactInfo[stream] = NULL;
        if (needGpuDiagFactor)
        {
            // allocate the space for diagonal factor on GPU
            gpuErrchk(cudaMalloc(&A_gpu.diagFactWork[stream],
                                 sizeof(Ftype) * dfactBufSize));
            gpuErrchk(cudaMalloc(&A_gpu.diagFactIPIV[stream],
                                 sizeof(int) * ldt));
            gpuErrchk(cudaMalloc(&A_gpu.diagFactInfo[stream],
                                 sizeof(int)));
        }

        /*lookAhead buffers and stream*/
        gpuErrchk(cudaMalloc(&A_gpu.lookAheadLGemmBuffer[stream], sizeof(Ftype) * maxLvalCount));

        gpuErrchk(cudaMalloc(&A_gpu.lookAheadUGemmBuffer[stream],
                             sizeof(Ftype) *
                                 static_cast<size_t>(SUPERLU_MAX((int_t)1, lookahead_u_val_count))));
            // Sherry: replace this by new code
	        //cudaMalloc(&A_gpu.dFBufs[stream], ldt * ldt * sizeof(Ftype));
	        //cudaMalloc(&A_gpu.gpuGemmBuffs[stream], A_gpu.gemmBufferSize * sizeof(Ftype));
    }
    xlu_sym_gpu3d_trace_gpu_setup(grid3d, "setLUstruct_GPU after per-stream buffer allocation");
    symV2SetupProfileAdd(SYM_V2_SETUP_PER_STREAM_BUFFER_ALLOC,
                         SuperLU_timer_() - tPerStreamBufferAlloc);

    /* Sherry: dfBufs[] changed to Ftype pointer **, max(batch, numCudaStreams) */
    int mxLeafNode = trf3Dpartition->mxLeafNode, mx_fsize = 0;

    /* Compute gemmCsize[] for batch operations
       !!!!!!! WARNING: this only works for 1 MPI  !!!!!! */
    if (sym_v2_mode && options->batchCount > 0)
        ABORT("SymFact GPU3DVERSION=2 does not support batchCount>0 until LDL-native batch sizing is implemented.");

    if ( options->batchCount > 0 ) {
	trf3Dpartition->gemmCsizes = int32Calloc_dist(mxLeafNode);
	int k, k0, k_st, k_end, offset, Csize;

	for (int ilvl = 0; ilvl < maxLvl; ++ilvl) {  /* Loop through the Pz tree levels */
	    int treeId = trf3Dpartition->myTreeIdxs[ilvl];
	    sForest_t* sforest = trf3Dpartition->sForests[treeId];
	    if (sforest){
		int_t *perm_c_supno = sforest->nodeList ;
        mx_fsize = max((int_t)mx_fsize, sforest->nNodes);

		int maxTopoLevel = sforest->topoInfo.numLvl;/* number of levels at each outer-tree node */
		for (int topoLvl = 0; topoLvl < maxTopoLevel; ++topoLvl) {
		    k_st = sforest->topoInfo.eTreeTopLims[topoLvl];
		    k_end = sforest->topoInfo.eTreeTopLims[topoLvl + 1];

		    for (k0 = k_st; k0 < k_end; ++k0) {
			offset = k0 - k_st;
			k = perm_c_supno[k0];
			Csize = lPanelVec[k].nzrows() * uPanelVec[k].nzcols();
			trf3Dpartition->gemmCsizes[offset] =
			    SUPERLU_MAX(trf3Dpartition->gemmCsizes[offset], Csize);
		    }
		}
	    }
	}
    }

    int num_dfbufs;  /* number of diagonal buffers */
    if ( options->batchCount > 0 ) { /* use batch code */
	num_dfbufs = mxLeafNode;
    } else { /* use pipelined code */
	// num_dfbufs = MAX_CUDA_STREAMS; //
    num_dfbufs = A_gpu.numCudaStreams;
    }
    int num_gemmbufs = num_dfbufs;
#if ( PRNTlevel >= 1 )
    printf(".. setLUstrut_GPU: num_dfbufs %d, num_gemmbufs %d\n", num_dfbufs, num_gemmbufs);
    fflush(stdout);
#endif

    double tDfbufGemmbufAlloc = SuperLU_timer_();
    A_gpu.dFBufs = (Ftype **) SUPERLU_MALLOC(num_dfbufs * sizeof(Ftype *));
    A_gpu.gpuGemmBuffs = (Ftype **) SUPERLU_MALLOC(num_gemmbufs * sizeof(Ftype *));

    int l;
    size_t sum_diag_size = 0;
    size_t sum_gemmC_size = 0;

    if ( options->batchCount > 0 ) { /* set up variable-size buffers for batch code */
	for (i = 0; i < num_dfbufs; ++i) {
	    l = trf3Dpartition->diagDims[i];
	    gpuErrchk(cudaMalloc(&(A_gpu.dFBufs[i]), l * l * sizeof(Ftype)));
	    //printf("\t diagDims[%d] %d\n", i, l);
	    gpuErrchk(cudaMalloc(&(A_gpu.gpuGemmBuffs[i]), trf3Dpartition->gemmCsizes[i] * sizeof(Ftype)));
	    sum_diag_size += static_cast<size_t>(l) * static_cast<size_t>(l);
	    sum_gemmC_size += static_cast<size_t>(trf3Dpartition->gemmCsizes[i]);
	}
    } else { /* uniform-size buffers */
	size_t dfbuf_elems = sym_diag_buf_elems;
        xlu_sym_gpu3d_trace_gpu_setup(grid3d, "setLUstruct_GPU before dfbuf/gemmbuf allocation");
	for (i = 0; i < num_dfbufs; ++i) {
        gpuErrchk(cudaMalloc(&(A_gpu.dFBufs[i]), dfbuf_elems * sizeof(Ftype)));
	    gpuErrchk(cudaMalloc(&(A_gpu.gpuGemmBuffs[i]), A_gpu.gemmBufferSize * sizeof(Ftype)));
	    sum_diag_size += dfbuf_elems;
	    sum_gemmC_size += A_gpu.gemmBufferSize;
	}
        xlu_sym_gpu3d_trace_gpu_setup(grid3d, "setLUstruct_GPU after dfbuf/gemmbuf allocation");
    }

    // Wajih: Adding allocation for batched LU and SCU marshalled data
    // TODO: these are serialized workspaces, so the allocations can be shared

#if 0
    A_gpu.marshall_data.setBatchSize(num_dfbufs);
    A_gpu.sc_marshall_data.setBatchSize(num_dfbufs);
#endif

    // TODO: where should these be freed?
    // Allocate GPU copy for the node list
    gpuErrchk(cudaMalloc(&(A_gpu.dperm_c_supno), sizeof(int) * mx_fsize));
    // Allocate GPU copy of all the gemm buffer pointers and copy the host array to the GPU
    gpuErrchk(cudaMalloc(&(A_gpu.dgpuGemmBuffs), sizeof(Ftype*) * num_gemmbufs));
    gpuErrchk(cudaMemcpy(A_gpu.dgpuGemmBuffs, A_gpu.gpuGemmBuffs, sizeof(Ftype*) * num_gemmbufs, cudaMemcpyHostToDevice));
    gpuErrchk(cudaGetLastError());
    symV2SetupProfileAdd(SYM_V2_SETUP_DFBUF_GEMMBUF_ALLOC,
                         SuperLU_timer_() - tDfbufGemmbufAlloc);

    tcuMalloc = SuperLU_timer_() - tcuMalloc;
#if ( PRNTlevel>=1 )
    printf("Time to allocate GPU memory: %g\n", tcuMalloc);
    printf("\t.. sum_diag_size %zu\t sum_gemmC_size %zu\n", sum_diag_size, sum_gemmC_size);
    fflush(stdout);
#endif

    double tcuStream=SuperLU_timer_();
    double tStreamHandleCreate = SuperLU_timer_();

    for (stream = 0; stream < A_gpu.numCudaStreams; stream++)
    {
        // cublasCreate(&A_gpu.cuHandles[stream]);
        A_gpu.cuSolveHandles[stream] = NULL;
        if (needGpuDiagFactor)
            gpuCusolverErrchk(cusolverDnCreate(&A_gpu.cuSolveHandles[stream]));
    }
    tcuStream = SuperLU_timer_() - tcuStream;

    double tcuStreamCreate=SuperLU_timer_();
    for (stream = 0; stream < A_gpu.numCudaStreams; stream++)
    {
        gpuErrchk(cudaStreamCreate(&A_gpu.cuStreams[stream]));
        gpuErrchk(cudaEventCreateWithFlags(&A_gpu.panelReadyEvents[stream],
                                           cudaEventDisableTiming));
        if (sym_v2_mode && superlu_sym_v2_wpanel_cache())
            gpuErrchk(cudaEventCreateWithFlags(
                &A_gpu.symV2RawPanelReadyEvents[stream],
                cudaEventDisableTiming));
        gpuErrchk(cudaEventCreateWithFlags(
            &A_gpu.symV2PartnerLPackReadyEvents[stream],
            cudaEventDisableTiming));
        gpuErrchk(cudaEventRecord(A_gpu.panelReadyEvents[stream],
                                  A_gpu.cuStreams[stream]));
        gpuErrchk(cudaEventRecord(
            A_gpu.symV2PartnerLPackReadyEvents[stream],
            A_gpu.cuStreams[stream]));
#ifdef SLU_ENABLE_SYM_GPU3D_TIMING
        gpuErrchk(cudaEventCreate(&A_gpu.diagD2HStartEvents[stream]));
        gpuErrchk(cudaEventCreate(&A_gpu.diagD2HEndEvents[stream]));
#endif
        gpublasCheckErrors(cublasCreate(&A_gpu.cuHandles[stream]));
        /*lookAhead buffers and stream*/
        gpublasCheckErrors(cublasCreate(&A_gpu.lookAheadLHandle[stream]));
        gpuErrchk(cudaStreamCreate(&A_gpu.lookAheadLStream[stream]));
        gpublasCheckErrors(cublasCreate(&A_gpu.lookAheadUHandle[stream]));
        gpuErrchk(cudaStreamCreate(&A_gpu.lookAheadUStream[stream]));

    }
    symV2SetupProfileAdd(SYM_V2_SETUP_STREAM_HANDLE_CREATE,
                         SuperLU_timer_() - tStreamHandleCreate);
    xlu_sym_gpu3d_trace_gpu_setup(grid3d, "setLUstruct_GPU after stream/handle creation");
    if (options->SymFact == YES)
    {
        double tDiagPrefetchAlloc = SuperLU_timer_();
        xlu_sym_gpu3d_trace_gpu_setup(grid3d, "setLUstruct_GPU before sym diag prefetch allocation");
        symDiagPrefetchBufs.assign(A_gpu.numCudaStreams, NULL);
        symDiagPrefetchDoneEvents.resize(A_gpu.numCudaStreams);
        symDiagPrefetchNodes.assign(A_gpu.numCudaStreams, -1);
        size_t prefetch_bytes =
            xlu_checked_square_alloc_bytes(ldt, sizeof(Ftype),
                                           "SymFact diagonal prefetch buffer");
        for (stream = 0; stream < A_gpu.numCudaStreams; stream++)
        {
            gpuErrchk(cudaMallocHost((void **)&symDiagPrefetchBufs[stream],
                                     prefetch_bytes));
            gpuErrchk(cudaEventCreateWithFlags(&symDiagPrefetchDoneEvents[stream],
                                               cudaEventDisableTiming));
        }
        symV2SetupProfileAdd(SYM_V2_SETUP_DIAG_PREFETCH_ALLOC,
                             SuperLU_timer_() - tDiagPrefetchAlloc);
        xlu_sym_gpu3d_trace_gpu_setup(grid3d, "setLUstruct_GPU after sym diag prefetch allocation");
    }
    tcuStreamCreate = SuperLU_timer_() - tcuStreamCreate;
    tRegion[3] = SuperLU_timer_() - tRegion[3];

#if ( PRNTlevel >= 1 )
    printf("Time to create cublas streams: %g\n", tcuStream);
    printf("Time to create CUDA streams: %g\n", tcuStreamCreate);
    printf("Time taken to estimate memory on GPU: %f\n", tRegion[0]);
    printf("TRegion L,U send: \t %g\n", tRegion[1]);
    printf("Time to send Lpanel=%g  and U panels =%g \n", tLsend, tUsend);
    printf("TRegion dfactBuf: \t %g\n", tRegion[2]);
    printf("TRegion stream: \t %g\n", tRegion[3]);
    fflush(stdout);
#endif

    // allocate
    xlu_sym_gpu3d_trace_gpu_setup(grid3d, "setLUstruct_GPU before device struct allocation");
    double tDeviceStructCopy = SuperLU_timer_();
    gpuErrchk(cudaMalloc(&dA_gpu, sizeof(xLUstructGPU_t<Ftype>)));
    gpuErrchk(cudaMemcpy(dA_gpu, &A_gpu, sizeof(xLUstructGPU_t<Ftype>), cudaMemcpyHostToDevice));
    symV2SetupProfileAdd(SYM_V2_SETUP_DEVICE_STRUCT_COPY,
                         SuperLU_timer_() - tDeviceStructCopy);
    xlu_sym_gpu3d_trace_gpu_setup(grid3d, "exit setLUstruct_GPU");

#endif /* match #if 0 ... #else ... */

    // cudaCheckError();

#if (DEBUGlevel >= 1)
	CHECK_MALLOC(iam, "Exit setLUstruct_GPU()");
#endif
    return 0;
} /* setLUstruct_GPU */

template <typename Ftype>
int_t xLUstruct_t<Ftype>::copyLUGPUtoHost()
{

    for (int_t i = 0; i < symV2PanelCount(); ++i)
        if (symV2PanelGid(i) < nsupers &&
            isNodeInMyGrid[symV2PanelGid(i)] == 1)
            lPanelVec[i].copyFromGPU();

    if (needsUPanelStorage())
    {
        if (uPanelVec == NULL)
            ABORT("U host panel storage is missing.");
        for (int_t i = 0; i < symV2RowCount(); ++i)
            if (symV2RowGid(i) < nsupers &&
                isNodeInMyGrid[symV2RowGid(i)] == 1)
                uPanelVec[i].copyFromGPU();
    }
    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::copyLUHosttoGPU()
{
    for (int_t i = 0; i < symV2PanelCount(); ++i)
        if (symV2PanelGid(i) < nsupers &&
            isNodeInMyGrid[symV2PanelGid(i)] == 1)
            lPanelVec[i].copyBackToGPU();

    if (needsUPanelStorage())
    {
        if (uPanelVec == NULL)
            ABORT("U host panel storage is missing.");
        if (A_gpu.uPanelVec == NULL)
            ABORT("U GPU panel storage is missing.");
        for (int_t i = 0; i < symV2RowCount(); ++i)
            if (symV2RowGid(i) < nsupers &&
                isNodeInMyGrid[symV2RowGid(i)] == 1)
                uPanelVec[i].copyBackToGPU();
    }
    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::checkGPU()
{

    for (int_t i = 0; i < symV2PanelCount(); ++i)
        lPanelVec[i].checkGPU();

    if (needsUPanelStorage())
    {
        if (uPanelVec == NULL)
            ABORT("U host panel storage is missing.");
        if (A_gpu.uPanelVec == NULL)
            ABORT("U GPU panel storage is missing.");
        for (int_t i = 0; i < symV2RowCount(); ++i)
            uPanelVec[i].checkGPU();
    }

    std::cout << "Checking LU struct completed succesfully"
              << "\n";
    return 0;
}


/**
 * @brief Pack non-zero values into a vector.
 *
 * @param spNzvalArray The array of non-zero values.
 * @param nzvalSize The size of the array of non-zero values.
 * @param valOffset The offset of the non-zero values.
 * @param packedNzvals The vector to store the non-zero values.
 * @param packedNzvalsIndices The vector to store the indices of the non-zero values.
 */
template <typename Ftype>
void packNzvals(std::vector<Ftype> &packedNzvals, std::vector<int_t> &packedNzvalsIndices,
                Ftype *spNzvalArray, int_t nzvalSize, int_t valOffset)
{
    for (int k = 0; k < nzvalSize; k++)
    {
        if (spNzvalArray[k] != 0)
        {
            packedNzvals.push_back(spNzvalArray[k]);
            packedNzvalsIndices.push_back(valOffset + k);
        }
    }
}

const int AVOID_CPU_NZVAL = 1;
template <typename Ftype>
xlpanelGPU_t<Ftype> *xLUstruct_t<Ftype>::copyLpanelsToGPU()
{
    xlpanelGPU_t<Ftype> *lPanelVec_GPU = new xlpanelGPU_t<Ftype>[CEILING(nsupers, Pc)];

    // TODO: set gpuLvalSize, gpuLidxSize
    gpuLvalSize = 0;
    gpuLidxSize = 0;
    for (int_t i = 0; i < CEILING(nsupers, Pc); ++i)
    {
        if (i * Pc + mycol < nsupers && isNodeInMyGrid[i * Pc + mycol] == 1)
        {
            gpuLvalSize += sizeof(Ftype) * lPanelVec[i].nzvalSize();
            gpuLidxSize += sizeof(int_t) * lPanelVec[i].indexSize();
        }
    }

    Ftype *valBuffer = (Ftype *)SUPERLU_MALLOC(gpuLvalSize);
    int_t *idxBuffer = (int_t *)SUPERLU_MALLOC(gpuLidxSize);

    // allocate memory buffer on GPU
    gpuErrchk(cudaMalloc(&gpuLvalBasePtr, gpuLvalSize));
    gpuErrchk(cudaMalloc(&gpuLidxBasePtr, gpuLidxSize));

    size_t valOffset = 0;
    size_t idxOffset = 0;
    Ftype tCopyToCPU = SuperLU_timer_();

    std::vector<Ftype> packedNzvals;
    std::vector<int_t> packedNzvalsIndices;

    // do a memcpy to CPU buffer
    for (int_t i = 0; i < CEILING(nsupers, Pc); ++i)
    {
        if (i * Pc + mycol < nsupers && isNodeInMyGrid[i * Pc + mycol] == 1)
        {
            if (lPanelVec[i].isEmpty())
            {
                xlpanelGPU_t<Ftype> ithLpanel(NULL, NULL);
                lPanelVec[i].gpuPanel = ithLpanel;
                lPanelVec_GPU[i] = ithLpanel;
            }
            else
            {
                xlpanelGPU_t<Ftype> ithLpanel(&gpuLidxBasePtr[idxOffset], &gpuLvalBasePtr[valOffset]);
                lPanelVec[i].gpuPanel = ithLpanel;
                lPanelVec_GPU[i] = ithLpanel;
                if (AVOID_CPU_NZVAL)
                    packNzvals<Ftype>(packedNzvals, packedNzvalsIndices, lPanelVec[i].val, lPanelVec[i].nzvalSize(), valOffset);
                else
                    memcpy(&valBuffer[valOffset], lPanelVec[i].val, sizeof(Ftype) * lPanelVec[i].nzvalSize());

                memcpy(&idxBuffer[idxOffset], lPanelVec[i].index, sizeof(int_t) * lPanelVec[i].indexSize());

                valOffset += lPanelVec[i].nzvalSize();
                idxOffset += lPanelVec[i].indexSize();
            }
        }
    }
    tCopyToCPU = SuperLU_timer_() - tCopyToCPU;
    std::cout << "Time to copy-L to CPU: " << tCopyToCPU << "\n";
    // do a cudaMemcpy to GPU
    Ftype tLsend = SuperLU_timer_();
    if (AVOID_CPU_NZVAL)
        copyToGPU(gpuLvalBasePtr, packedNzvals, packedNzvalsIndices);
    else
    {
#if 0
            cudaMemcpy(gpuLvalBasePtr, valBuffer, gpuLvalSize, cudaMemcpyHostToDevice);
#else
        copyToGPU_Sparse(gpuLvalBasePtr, valBuffer, gpuLvalSize);
#endif
    }
    // find
    gpuErrchk(cudaMemcpy(gpuLidxBasePtr, idxBuffer, gpuLidxSize, cudaMemcpyHostToDevice));
    tLsend = SuperLU_timer_() - tLsend;
    printf("cudaMemcpy time L =%g \n", tLsend);

    SUPERLU_FREE(valBuffer);
    SUPERLU_FREE(idxBuffer);
    return lPanelVec_GPU;
} /* copyLpanelsToGPU */

template <typename Ftype>
xupanelGPU_t<Ftype> *xLUstruct_t<Ftype>::copyUpanelsToGPU()
{
#if (DEBUGlevel >= 1)
    int iam = 0;
    CHECK_MALLOC(iam, "Enter copyUpanelsToGPU()");
#endif

    xupanelGPU_t<Ftype> *uPanelVec_GPU = new xupanelGPU_t<Ftype>[CEILING(nsupers, Pr)];

    gpuUvalSize = 0;
    gpuUidxSize = 0;
    for (int_t i = 0; i < CEILING(nsupers, Pr); ++i)
    {
        if (i * Pr + myrow < nsupers && isNodeInMyGrid[i * Pr + myrow] == 1)
        {
            gpuUvalSize += sizeof(Ftype) * uPanelVec[i].nzvalSize();
            gpuUidxSize += sizeof(int_t) * uPanelVec[i].indexSize();
        }
    }

    // TODO: set gpuUvalSize, gpuUidxSize

    // allocate memory buffer on GPU
    gpuErrchk(cudaMalloc(&gpuUvalBasePtr, gpuUvalSize));
    gpuErrchk(cudaMalloc(&gpuUidxBasePtr, gpuUidxSize));

    size_t valOffset = 0;
    size_t idxOffset = 0;

    Ftype tCopyToCPU = SuperLU_timer_();
    for (int_t i = 0; i < CEILING(nsupers, Pr); ++i)
    {
        if (i * Pr + myrow < nsupers && isNodeInMyGrid[i * Pr + myrow] == 1)
        {
            if (uPanelVec[i].isEmpty())
            {
                xupanelGPU_t<Ftype> ithupanel(NULL, NULL);
                uPanelVec[i].gpuPanel = ithupanel;
                uPanelVec_GPU[i] = ithupanel;
            }
        }
    }

    int_t *idxBuffer = NULL;
    if ( gpuUidxSize>0 ) /* Sherry fix: gpuUidxSize can be 0 */
	idxBuffer = (int_t *)SUPERLU_MALLOC(gpuUidxSize);

    if (AVOID_CPU_NZVAL)
    {
        printf("AVOID_CPU_NZVAL is set\n");
        std::vector<Ftype> packedNzvals;
        std::vector<int_t> packedNzvalsIndices;
        for (int_t i = 0; i < CEILING(nsupers, Pr); ++i)
        {
            if (i * Pr + myrow < nsupers && isNodeInMyGrid[i * Pr + myrow] == 1)
            {
                if (!uPanelVec[i].isEmpty())
                {

                    xupanelGPU_t<Ftype> ithupanel(&gpuUidxBasePtr[idxOffset], &gpuUvalBasePtr[valOffset]);
                    uPanelVec[i].gpuPanel = ithupanel;
                    uPanelVec_GPU[i] = ithupanel;
                    packNzvals<Ftype>(packedNzvals, packedNzvalsIndices, uPanelVec[i].val,
                               uPanelVec[i].nzvalSize(), valOffset);
                    memcpy(&idxBuffer[idxOffset], uPanelVec[i].index, sizeof(int_t) * uPanelVec[i].indexSize());

                    valOffset += uPanelVec[i].nzvalSize();
                    idxOffset += uPanelVec[i].indexSize();
                }
            }
        }
        tCopyToCPU = SuperLU_timer_() - tCopyToCPU;
        printf("copyU to CPU-buff time = %g\n", tCopyToCPU);

        // do a cudaMemcpy to GPU
        Ftype tLsend = SuperLU_timer_();
        copyToGPU(gpuUvalBasePtr, packedNzvals, packedNzvalsIndices);
        gpuErrchk(cudaMemcpy(gpuUidxBasePtr, idxBuffer, gpuUidxSize, cudaMemcpyHostToDevice));
        tLsend = SuperLU_timer_() - tLsend;
        printf("cudaMemcpy time U =%g \n", tLsend);
        // SUPERLU_FREE(valBuffer);
    }
    else /* AVOID_CPU_NZVAL not set */
    {
        // do a memcpy to CPU buffer
        Ftype *valBuffer = (Ftype *)SUPERLU_MALLOC(gpuUvalSize);

        for (int_t i = 0; i < CEILING(nsupers, Pr); ++i)
        {
            if (i * Pr + myrow < nsupers && isNodeInMyGrid[i * Pr + myrow] == 1)
            {
                if (!uPanelVec[i].isEmpty())
                {

                    xupanelGPU_t<Ftype> ithupanel(&gpuUidxBasePtr[idxOffset], &gpuUvalBasePtr[valOffset]);
                    uPanelVec[i].gpuPanel = ithupanel;
                    uPanelVec_GPU[i] = ithupanel;
                    memcpy(&valBuffer[valOffset], uPanelVec[i].val, sizeof(Ftype) * uPanelVec[i].nzvalSize());
                    memcpy(&idxBuffer[idxOffset], uPanelVec[i].index, sizeof(int_t) * uPanelVec[i].indexSize());

                    valOffset += uPanelVec[i].nzvalSize();
                    idxOffset += uPanelVec[i].indexSize();
                }
            }
        }
        tCopyToCPU = SuperLU_timer_() - tCopyToCPU;
        printf("copyU to CPU-buff time = %g\n", tCopyToCPU);

        // do a cudaMemcpy to GPU
        Ftype tLsend = SuperLU_timer_();
        const int USE_GPU_COPY = 1;
        if (USE_GPU_COPY)
        {
            gpuErrchk(cudaMemcpy(gpuUvalBasePtr, valBuffer, gpuUvalSize, cudaMemcpyHostToDevice));
        }
        else
            copyToGPU_Sparse(gpuUvalBasePtr, valBuffer, gpuUvalSize);

        gpuErrchk(cudaMemcpy(gpuUidxBasePtr, idxBuffer, gpuUidxSize, cudaMemcpyHostToDevice));
        tLsend = SuperLU_timer_() - tLsend;
        printf("cudaMemcpy time U =%g \n", tLsend);

        SUPERLU_FREE(valBuffer);
    } /* end else AVOID_CPU_NZVAL not set */

    if ( gpuUidxSize>0 ) /* Sherry fix: gpuUidxSize can be 0 */
	SUPERLU_FREE(idxBuffer);

#if (DEBUGlevel >= 1)
    CHECK_MALLOC(iam, "Exit copyUpanelsToGPU()");
#endif

    return uPanelVec_GPU;

} /* copyUpanelsToGPU */
//#endif


//////// Rest of the code for batch not used anymore
#if (0)
// Marshall Functors for batched execution
template <typename Ftype>
struct MarshallLUFunc {
    int k_st, *ld_batch, *dim_batch;
    Ftype** diag_ptrs;
    xLUstructGPU_t<Ftype> *A_gpu;

    MarshallLUFunc(int k_st, Ftype** diag_ptrs, int *ld_batch, int *dim_batch, xLUstructGPU_t<Ftype> *A_gpu)
    {
        this->k_st = k_st;
        this->ld_batch = ld_batch;
        this->dim_batch = dim_batch;
        this->diag_ptrs = diag_ptrs;
        this->A_gpu = A_gpu;
    }

    __device__ void operator()(const unsigned int &i) const
    {
        int k = A_gpu->dperm_c_supno[k_st + i];
        int_t* xsup = A_gpu->xsup;
        lpanelGPU_t &lpanel = A_gpu->lPanelVec[A_gpu->g2lCol(k)];

        diag_ptrs[i] = lpanel.blkPtr(0);
        ld_batch[i] = lpanel.LDA();
        dim_batch[i] = SuperSize(k);
    }
};

struct MarshallTRSMUFunc {
    int k_st, *diag_ld_batch, *diag_dim_batch, *panel_ld_batch, *panel_dim_batch;
    Ftype** diag_ptrs, **panel_ptrs;
    xLUstructGPU_t<Ftype> *A_gpu;

    MarshallTRSMUFunc(
        int k_st, Ftype** diag_ptrs, int *diag_ld_batch, int *diag_dim_batch, Ftype** panel_ptrs,
        int *panel_ld_batch, int *panel_dim_batch, xLUstructGPU_t<Ftype> *A_gpu
    )
    {
        this->k_st = k_st;
        this->diag_ptrs = diag_ptrs;
        this->diag_ld_batch = diag_ld_batch;
        this->diag_dim_batch = diag_dim_batch;
        this->panel_ptrs = panel_ptrs;
        this->panel_ld_batch = panel_ld_batch;
        this->panel_dim_batch = panel_dim_batch;
        this->A_gpu = A_gpu;
    }

    __device__ void operator()(const unsigned int &i) const
    {
        int k = A_gpu->dperm_c_supno[k_st + i];
        int_t* xsup = A_gpu->xsup;
        int ksupc = SuperSize(k);

        upanelGPU_t &upanel = A_gpu->uPanelVec[A_gpu->g2lRow(k)];
        lpanelGPU_t &lpanel = A_gpu->lPanelVec[A_gpu->g2lCol(k)];

        if(!upanel.isEmpty())
        {
            panel_ptrs[i] = upanel.blkPtr(0);
            panel_ld_batch[i] = upanel.LDA();
            panel_dim_batch[i] = upanel.nzcols();
            diag_ptrs[i] = lpanel.blkPtr(0);
            diag_ld_batch[i] = lpanel.LDA();
            diag_dim_batch[i] = ksupc;
        }
        else
        {
            panel_ptrs[i] = diag_ptrs[i] = NULL;
            panel_ld_batch[i] = diag_ld_batch[i] = 1;
            panel_dim_batch[i] = diag_dim_batch[i] = 0;
        }
    }
};

struct MarshallTRSMLFunc {
    int k_st, *diag_ld_batch, *diag_dim_batch, *panel_ld_batch, *panel_dim_batch;
    Ftype** diag_ptrs, **panel_ptrs;
    xLUstructGPU_t<Ftype> *A_gpu;

    MarshallTRSMLFunc(
        int k_st, Ftype** diag_ptrs, int *diag_ld_batch, int *diag_dim_batch, Ftype** panel_ptrs,
        int *panel_ld_batch, int *panel_dim_batch, xLUstructGPU_t<Ftype> *A_gpu
    )
    {
        this->k_st = k_st;
        this->diag_ptrs = diag_ptrs;
        this->diag_ld_batch = diag_ld_batch;
        this->diag_dim_batch = diag_dim_batch;
        this->panel_ptrs = panel_ptrs;
        this->panel_ld_batch = panel_ld_batch;
        this->panel_dim_batch = panel_dim_batch;
        this->A_gpu = A_gpu;
    }

    __device__ void operator()(const unsigned int &i) const
    {
        int k = A_gpu->dperm_c_supno[k_st + i];
        int_t* xsup = A_gpu->xsup;
        int ksupc = SuperSize(k);

        lpanelGPU_t &lpanel = A_gpu->lPanelVec[A_gpu->g2lCol(k)];

        if(!lpanel.isEmpty())
        {
            Ftype *lPanelStPtr = lpanel.blkPtr(0);
            int_t len = lpanel.nzrows();
            if(lpanel.haveDiag())
            {
                lPanelStPtr = lpanel.blkPtr(1);
                len -= lpanel.nbrow(0);
            }
            panel_ptrs[i] = lPanelStPtr;
            panel_ld_batch[i] = lpanel.LDA();
            panel_dim_batch[i] = len;
            diag_ptrs[i] = lpanel.blkPtr(0);
            diag_ld_batch[i] = lpanel.LDA();
            diag_dim_batch[i] = ksupc;
        }
        else
        {
            panel_ptrs[i] = diag_ptrs[i] = NULL;
            panel_ld_batch[i] = diag_ld_batch[i] = 1;
            panel_dim_batch[i] = diag_dim_batch[i] = 0;
        }
    }
};

struct MarshallInitSCUFunc {
    int k_st, *ist, *iend, *maxGemmRows, *maxGemmCols;
    lpanelGPU_t* lpanels;
    upanelGPU_t* upanels;
    xLUstructGPU_t<Ftype> *A_gpu;

    MarshallInitSCUFunc(
        int k_st, int *ist, int *iend, int *maxGemmRows, int *maxGemmCols,
        lpanelGPU_t* lpanels, upanelGPU_t* upanels, xLUstructGPU_t<Ftype> *A_gpu
    )
    {
        this->k_st = k_st;
        this->ist = ist;
        this->iend = iend;
        this->maxGemmRows = maxGemmRows;
        this->maxGemmCols = maxGemmCols;
        this->lpanels = lpanels;
        this->upanels = upanels;
        this->A_gpu = A_gpu;
    }

    __device__ void operator()(const unsigned int &i) const
    {
        int k = A_gpu->dperm_c_supno[k_st + i];
        size_t gemmBufferSize = A_gpu->gemmBufferSize;

        lpanelGPU_t &lpanel = A_gpu->lPanelVec[A_gpu->g2lCol(k)];
        upanelGPU_t &upanel = A_gpu->uPanelVec[A_gpu->g2lCol(k)];

        lpanels[i] = lpanel;
        upanels[i] = upanel;

        if(!upanel.isEmpty() && !lpanel.isEmpty())
        {
            int_t st_lb = 1;
            int_t nlb = lpanel.nblocks();
            int_t nub = upanel.nblocks();

            ist[i] = iend[i] = st_lb;
            int nrows = lpanel.stRow(nlb) - lpanel.stRow(st_lb);
            int ncols = upanel.nzcols();

            int max_rows = nrows;
            int max_cols = ncols;
            // entire gemm doesn't fit in gemm buffer
            if (nrows * ncols > gemmBufferSize)
            {
                int maxGemmOpSize = (int)sqrt((Ftype)gemmBufferSize);
                int numberofRowChunks = (nrows + maxGemmOpSize - 1) / maxGemmOpSize;
                max_rows = nrows / numberofRowChunks;
                max_cols = gemmBufferSize / max_rows;
            }

            maxGemmRows[i] = max_rows;
            maxGemmCols[i] = max_cols;
        }
        else
        {
            ist[i] = iend[i] = 0;
            maxGemmRows[i] = maxGemmCols[i] = 0;
        }
    }
};

struct MarshallSCUOuter_Predicate
{
    __host__ __device__ bool operator()(const int &x)
    {
        return x == 1;
    }
};

struct MarshallSCUOuterFunc {
    int k_st, *ist, *iend, *jst, *jend, *maxGemmRows, *done_flags;
    xLUstructGPU_t<Ftype> *A_gpu;

    MarshallSCUOuterFunc(int k_st, int *ist, int *iend, int *jst, int *jend, int *maxGemmRows, int* done_flags, xLUstructGPU_t<Ftype> *A_gpu)
    {
        this->k_st = k_st;
        this->ist = ist;
        this->iend = iend;
        this->jst = jst;
        this->jend = jend;
        this->maxGemmRows = maxGemmRows;
        this->done_flags = done_flags;
        this->A_gpu = A_gpu;
    }

    __device__ void operator()(const unsigned int &i) const
    {
        int k = A_gpu->dperm_c_supno[k_st + i];
        lpanelGPU_t &lpanel = A_gpu->lPanelVec[A_gpu->g2lCol(k)];
        upanelGPU_t &upanel = A_gpu->uPanelVec[A_gpu->g2lCol(k)];
        int& iEnd = iend[i];

        // Not done if even one operation still has work to do
        if(!lpanel.isEmpty() && !upanel.isEmpty() && iEnd < lpanel.nblocks())
        {
            int& iSt = ist[i];
            iSt = iEnd;
            iEnd = lpanel.getEndBlock(iSt, maxGemmRows[i]);
            assert(iEnd > iSt);
            jst[i] = jend[i] = 0;
            done_flags[i] = 0;
        }
        else
        {
            done_flags[i] = 1;
        }
    }
};

struct MarshallSCUInner_Predicate
{
    __host__ __device__ bool operator()(const int &x)
    {
        return x == 0;
    }
}

template<typename T>
struct element_diff : public thrust::unary_function<T,T>
{
    T* st, *end;
    element_diff(T* st, T *end)
    {
        this->st = st;
        this->end = end;
    }

    __device__ T operator()(const T &x) const
    {
        return end[x] - st[x];
    }
}


struct MarshallSCUInnerFunc {
    int k_st, *ist, *iend, *jst, *jend, *maxGemmCols;
    xLUstructGPU_t<Ftype> *A_gpu;
    Ftype** A_ptrs, **B_ptrs, **C_ptrs;
    int* lda_array, *ldb_array, *ldc_array, *m_array, *n_array, *k_array;

    MarshallSCUInnerFunc(
        int k_st, int *ist, int *iend, int *jst, int *jend, int *maxGemmCols,
        Ftype** A_ptrs, int* lda_array, Ftype** B_ptrs, int* ldb_array, Ftype **C_ptrs, int *ldc_array,
        int *m_array, int *n_array, int *k_array, xLUstructGPU_t<Ftype> *A_gpu
    )
    {
        this->k_st = k_st;
        this->ist = ist;
        this->iend = iend;
        this->jst = jst;
        this->jend = jend;
        this->maxGemmCols = maxGemmCols;
        this->A_ptrs = A_ptrs;
        this->B_ptrs = B_ptrs;
        this->C_ptrs = C_ptrs;
        this->lda_array = lda_array;
        this->ldb_array = ldb_array;
        this->ldc_array = ldc_array;
        this->m_array = m_array;
        this->n_array = n_array;
        this->k_array = k_array;
        this->A_gpu = A_gpu;
    }

    __device__ void operator()(const unsigned int &i) const
    {
        int k = A_gpu->dperm_c_supno[k_st + i];
        int_t* xsup = A_gpu->xsup;
        lpanelGPU_t &lpanel = A_gpu->lPanelVec[A_gpu->g2lCol(k)];
        upanelGPU_t &upanel = A_gpu->uPanelVec[A_gpu->g2lCol(k)];

        int iSt = ist[i], iEnd = iend[i];
        int& jSt = jst[i], &jEnd = jend[i];

        // Not done if even one operation still has work to do
        if(!lpanel.isEmpty() && !upanel.isEmpty() && jEnd < upanel.nblocks())
        {
            jSt = jEnd;
            jEnd = upanel.getEndBlock(jSt, maxGemmCols[i]);
            assert(jEnd > jSt);

            A_ptrs[i] = lpanel.blkPtr(iSt);
            B_ptrs[i] = upanel.blkPtr(jSt);
            C_ptrs[i] = A_gpu->dgpuGemmBuffs[i];

            lda_array[i] = lpanel.LDA();
            ldb_array[i] = upanel.LDA();
            ldc_array[i] = lpanel.stRow(iEnd) - lpanel.stRow(iSt);

            m_array[i] = ldc_array[i];
            n_array[i] = upanel.stCol(jEnd) - upanel.stCol(jSt);
            k_array[i] = SuperSize(k);
        }
        else
        {
            A_ptrs[i] = B_ptrs[i] = C_ptrs[i] = NULL;
            lda_array[i] = ldb_array[i] = ldc_array[i] = 1;
            m_array[i] = n_array[i] = k_array[i] = 0;
        }
    }
}

// Marshalling routines for batched execution
void xLUstruct_t<Ftype>::marshallBatchedLUData(int k_st, int k_end, int_t *perm_c_supno)
{
    // First gather up all the pointer and meta data on the host
    LUMarshallData& mdata = A_gpu.marshall_data;
    mdata.batchsize = k_end - k_st;

    MarshallLUFunc func(k_st, mdata.dev_diag_ptrs, mdata.dev_diag_ld_array, mdata.dev_diag_dim_array, dA_gpu);

    thrust::for_each(
        thrust::system::cuda::par, thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(mdata.batchsize), func
    );

    // Ftype **diag_ptrs = mdata.host_diag_ptrs.data();
    // int *ld_batch = mdata.host_diag_ld_array.data();
    // int *dim_batch = mdata.host_diag_dim_array.data();

	// mdata.batchsize = 0;

    // for (int_t k0 = k_st; k0 < k_end; k0++)
    // {
    //     int_t k = perm_c_supno[k0];

	// 	if (iam == procIJ(k, k))
	// 	{
    //         assert(mdata.batchsize < mdata.host_diag_ptrs.size());

    //         xlpanel_t<Ftype> &lpanel = lPanelVec[g2lCol(k)];
	// 		diag_ptrs[mdata.batchsize] = lpanel.blkPtrGPU(0);
	// 		ld_batch[mdata.batchsize] = lpanel.LDA();
	// 		dim_batch[mdata.batchsize] = SuperSize(k);
	// 		mdata.batchsize++;
	// 	}
    // }

    // mdata.setMaxDiag();
    // // Then copy the marshalled data over to the GPU
    // gpuErrchk(cudaMemcpy(mdata.dev_diag_ptrs, diag_ptrs, mdata.batchsize * sizeof(Ftype*), cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(mdata.dev_diag_ld_array, ld_batch, mdata.batchsize * sizeof(int), cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(mdata.dev_diag_dim_array, dim_batch, mdata.batchsize * sizeof(int), cudaMemcpyHostToDevice));
}

void xLUstruct_t<Ftype>::marshallBatchedTRSMUData(int k_st, int k_end, int_t *perm_c_supno)
{
    // First gather up all the pointer and meta data on the host
    LUMarshallData& mdata = A_gpu.marshall_data;
    mdata.batchsize = k_end - k_st;

    MarshallTRSMUFunc func(
        k_st, mdata.dev_diag_ptrs, mdata.dev_diag_ld_array, mdata.dev_diag_dim_array,
        mdata.dev_panel_ptrs, mdata.dev_panel_ld_array, mdata.dev_panel_dim_array, dA_gpu
    );

    thrust::for_each(
        thrust::system::cuda::par, thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(mdata.batchsize), func
    );

    // Ftype **panel_ptrs = mdata.host_panel_ptrs.data();
    // int *panel_ld_batch = mdata.host_panel_ld_array.data();
    // int *panel_dim_batch = mdata.host_panel_dim_array.data();
    // Ftype **diag_ptrs = mdata.host_diag_ptrs.data();
    // int *diag_ld_batch = mdata.host_diag_ld_array.data();
    // int *diag_dim_batch = mdata.host_diag_dim_array.data();

	// mdata.batchsize = 0;

    // for (int_t k0 = k_st; k0 < k_end; k0++)
    // {
    //     int_t k = perm_c_supno[k0];
    //     int_t buffer_offset = k0 - k_st;
	// 	int ksupc = SuperSize(k);

	// 	if (myrow == krow(k))
	// 	{
    //         upanel_t& upanel = uPanelVec[g2lRow(k)];
    //         xlpanel_t<Ftype> &lpanel = lPanelVec[g2lCol(k)];
    //         if(!upanel.isEmpty())
    //         {
    //             assert(mdata.batchsize < mdata.host_diag_ptrs.size());

    //             panel_ptrs[mdata.batchsize] = upanel.blkPtrGPU(0);
    //             panel_ld_batch[mdata.batchsize] = upanel.LDA();
    //             panel_dim_batch[mdata.batchsize] = upanel.nzcols();

    //             // Hackathon change: using the original diagonal block instead of the bcast buffer
    //             // diag_ptrs[mdata.batchsize] = A_gpu.dFBufs[buffer_offset];
    //             // diag_ld_batch[mdata.batchsize] = ksupc;
    //             // diag_dim_batch[mdata.batchsize] = ksupc;
    //             diag_ptrs[mdata.batchsize] = lpanel.blkPtrGPU(0);
    //             diag_ld_batch[mdata.batchsize] = lpanel.LDA();
    //             diag_dim_batch[mdata.batchsize] = ksupc;

    //             mdata.batchsize++;
    //         }
	// 	}
    // }

    // mdata.setMaxDiag();
    // mdata.setMaxPanel();

    // // Then copy the marshalled data over to the GPU
    // gpuErrchk(cudaMemcpy(mdata.dev_diag_ptrs, diag_ptrs, mdata.batchsize * sizeof(Ftype*), cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(mdata.dev_diag_ld_array, diag_ld_batch, mdata.batchsize * sizeof(int), cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(mdata.dev_diag_dim_array, diag_dim_batch, mdata.batchsize * sizeof(int), cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(mdata.dev_panel_ptrs, panel_ptrs, mdata.batchsize * sizeof(Ftype*), cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(mdata.dev_panel_ld_array, panel_ld_batch, mdata.batchsize * sizeof(int), cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(mdata.dev_panel_dim_array, panel_dim_batch, mdata.batchsize * sizeof(int), cudaMemcpyHostToDevice));
}

void xLUstruct_t<Ftype>::marshallBatchedTRSMLData(int k_st, int k_end, int_t *perm_c_supno)
{
    // First gather up all the pointer and meta data on the host
    LUMarshallData& mdata = A_gpu.marshall_data;
    mdata.batchsize = k_end - k_st;

    MarshallTRSMLFunc func(
        k_st, mdata.dev_diag_ptrs, mdata.dev_diag_ld_array, mdata.dev_diag_dim_array,
        mdata.dev_panel_ptrs, mdata.dev_panel_ld_array, mdata.dev_panel_dim_array, dA_gpu
    );

    thrust::for_each(
        thrust::system::cuda::par, thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(mdata.batchsize), func
    );

    // Ftype **panel_ptrs = mdata.host_panel_ptrs.data();
    // int *panel_ld_batch = mdata.host_panel_ld_array.data();
    // int *panel_dim_batch = mdata.host_panel_dim_array.data();
    // Ftype **diag_ptrs = mdata.host_diag_ptrs.data();
    // int *diag_ld_batch = mdata.host_diag_ld_array.data();
    // int *diag_dim_batch = mdata.host_diag_dim_array.data();

	// mdata.batchsize = 0;

    // for (int_t k0 = k_st; k0 < k_end; k0++)
    // {
    //     int_t k = perm_c_supno[k0];
    //     int_t buffer_offset = k0 - k_st;
	// 	int ksupc = SuperSize(k);

	// 	if (mycol == kcol(k))
	// 	{
    //         xlpanel_t<Ftype> &lpanel = lPanelVec[g2lCol(k)];
    //         if(!lpanel.isEmpty())
    //         {
    //             assert(mdata.batchsize < mdata.host_diag_ptrs.size());

    //             Ftype *lPanelStPtr = lpanel.blkPtrGPU(0);
    //             int_t len = lpanel.nzrows();
    //             if(lpanel.haveDiag())
    //             {
    //                 /* code */
    //                 lPanelStPtr = lpanel.blkPtrGPU(1);
    //                 len -= lpanel.nbrow(0);
    //             }
    //             panel_ptrs[mdata.batchsize] = lPanelStPtr;
    //             panel_ld_batch[mdata.batchsize] = lpanel.LDA();
    //             panel_dim_batch[mdata.batchsize] = len;

    //             // Hackathon change: using the original diagonal block instead of the bcast buffer
    //             // diag_ptrs[mdata.batchsize] = A_gpu.dFBufs[buffer_offset];
    //             // diag_ld_batch[mdata.batchsize] = ksupc;
    //             // diag_dim_batch[mdata.batchsize] = ksupc;
    //             diag_ptrs[mdata.batchsize] = lpanel.blkPtrGPU(0);
    //             diag_ld_batch[mdata.batchsize] = lpanel.LDA();
    //             diag_dim_batch[mdata.batchsize] = ksupc;

    //             mdata.batchsize++;
    //         }
	// 	}
    // }

    // mdata.setMaxDiag();
    // mdata.setMaxPanel();

    // // Then copy the marshalled data over to the GPU
    // cudaMemcpy(mdata.dev_diag_ptrs, diag_ptrs, mdata.batchsize * sizeof(Ftype*), cudaMemcpyHostToDevice);
    // cudaMemcpy(mdata.dev_diag_ld_array, diag_ld_batch, mdata.batchsize * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(mdata.dev_diag_dim_array, diag_dim_batch, mdata.batchsize * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(mdata.dev_panel_ptrs, panel_ptrs, mdata.batchsize * sizeof(Ftype*), cudaMemcpyHostToDevice);
    // cudaMemcpy(mdata.dev_panel_ld_array, panel_ld_batch, mdata.batchsize * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(mdata.dev_panel_dim_array, panel_dim_batch, mdata.batchsize * sizeof(int), cudaMemcpyHostToDevice);
}

void xLUstruct_t<Ftype>::initSCUMarshallData(int k_st, int k_end, int_t *perm_c_supno)
{
    SCUMarshallData& sc_mdata = A_gpu.sc_marshall_data;
    sc_mdata.batchsize = k_end - k_st;

    MarshallInitSCUFunc func(
        k_st, sc_mdata.dev_ist, sc_mdata.dev_iend, sc_mdata.dev_maxGemmRows, sc_mdata.dev_maxGemmCols,
        sc_mdata.dev_gpu_lpanels, sc_mdata.dev_gpu_upanels, dA_gpu
    );

    thrust::for_each(
        thrust::system::cuda::par, thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(sc_mdata.batchsize), func
    );

    // for (int_t k0 = k_st; k0 < k_end; k0++)
    // {
    //     int_t k = perm_c_supno[k0];
    //     int_t buffer_offset = k0 - k_st;

    //     assert(buffer_offset < sc_mdata.upanels.size());

    //     // Wajih: TODO: figure out what this offset does
    //     int offset = 0;
    //     if (UidxSendCounts[k] > 0 && LidxSendCounts[k] > 0)
    //     {
    //         sc_mdata.upanels[buffer_offset] = getKUpanel(k, offset);
    //         sc_mdata.lpanels[buffer_offset] = getKLpanel(k, offset);

    //         // Set gemm loop parameters for the panels
    //         upanel_t& upanel = sc_mdata.upanels[buffer_offset];
    //         lpanel_t& lpanel = sc_mdata.lpanels[buffer_offset];

    //         sc_mdata.host_gpu_upanels[buffer_offset] = upanel.gpuPanel;
    //         sc_mdata.host_gpu_lpanels[buffer_offset] = lpanel.gpuPanel;

    //         if(!upanel.isEmpty() && !lpanel.isEmpty())
    //         {
    //             int_t st_lb = 0;
    //             if (myrow == krow(k))
    //                 st_lb = 1;

    //             int_t nlb = lpanel.nblocks();
    //             int_t nub = upanel.nblocks();

    //             sc_mdata.ist[buffer_offset] = st_lb;
    //             sc_mdata.iend[buffer_offset] = sc_mdata.ist[buffer_offset];

    //             int nrows = lpanel.stRow(nlb) - lpanel.stRow(st_lb);
    //             int ncols = upanel.nzcols();

    //             int maxGemmRows = nrows;
    //             int maxGemmCols = ncols;
    //             // entire gemm doesn't fit in gemm buffer
    //             if (nrows * ncols > A_gpu.gemmBufferSize)
    //             {
    //                 int maxGemmOpSize = (int)sqrt(A_gpu.gemmBufferSize);
    //                 int numberofRowChunks = (nrows + maxGemmOpSize - 1) / maxGemmOpSize;
    //                 maxGemmRows = nrows / numberofRowChunks;
    //                 maxGemmCols = A_gpu.gemmBufferSize / maxGemmRows;
    //             }

    //             sc_mdata.maxGemmRows[buffer_offset] = maxGemmRows;
    //             sc_mdata.maxGemmCols[buffer_offset] = maxGemmCols;
    //         }
    //     }
    // }

    // sc_mdata.copyPanelDataToGPU();
}

int xLUstruct_t<Ftype>::marshallSCUBatchedDataOuter(int k_st, int k_end, int_t *perm_c_supno)
{
    SCUMarshallData& sc_mdata = A_gpu.sc_marshall_data;
    sc_mdata.batchsize = k_end - k_st;

    // Temporarily use the m array for the done flags
    int *done_flags = sc_mdata.dev_m_array;
    MarshallSCUOuterFunc func(
        k_st, sc_mdata.dev_ist, sc_mdata.dev_iend, sc_mdata.dev_jst, sc_mdata.dev_jend,
        sc_mdata.dev_maxGemmRows, done_flags, dA_gpu
    );

    thrust::for_each(
        thrust::system::cuda::par, thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(sc_mdata.batchsize), func
    );

    bool done = thrust::all_of(
        thrust::system::cuda::par, done_flags, done_flags + sc_mdata.batchsize,
        MarshallSCUOuter_Predicate()
    );

    return done;

    // int done_i = 1;
    // for (int k0 = k_st; k0 < k_end; k0++)
    // {
    //     int k = perm_c_supno[k0];
    //     int buffer_index = k0 - k_st;
    //     lpanel_t& lpanel = sc_mdata.lpanels[buffer_index];
    //     upanel_t& upanel = sc_mdata.upanels[buffer_index];
    //     if(lpanel.isEmpty() || upanel.isEmpty())
    //         continue;

    //     int& iEnd = sc_mdata.iend[buffer_index];
    //     // Not done if even one operation still has work to do
    //     if(iEnd < lpanel.nblocks())
    //     {
    //         done_i = 0;
    //         int& iSt = sc_mdata.ist[buffer_index];
    //         iSt = iEnd;
    //         iEnd = lpanel.getEndBlock(iSt, sc_mdata.maxGemmRows[buffer_index]);
    //         assert(iEnd > iSt);
    //         sc_mdata.jst[buffer_index] = sc_mdata.jend[buffer_index] = 0;
    //     }
    // }

    // return done_i;
}

int xLUstruct_t<Ftype>::marshallSCUBatchedDataInner(int k_st, int k_end, int_t *perm_c_supno)
{
    SCUMarshallData& sc_mdata = A_gpu.sc_marshall_data;
    int knum = k_end - k_st;
    sc_mdata.batchsize = knum;

    MarshallSCUInnerFunc func(
        k_st, sc_mdata.dev_ist, sc_mdata.dev_iend, sc_mdata.dev_jst, sc_mdata.dev_jend, sc_mdata.dev_maxGemmCols,
        sc_mdata.dev_A_ptrs, sc_mdata.dev_lda_array, sc_mdata.dev_B_ptrs, sc_mdata.dev_ldb_array, sc_mdata.dev_C_ptrs,
        sc_mdata.dev_ldc_array, sc_mdata.dev_m_array, sc_mdata.dev_n_array, sc_mdata.dev_k_array, dA_gpu
    );

    thrust::counting_iterator<int> start(0), end(knum);
    thrust::for_each(thrust::system::cuda::par, start, end, func);

    // Set the max dims in the marshalled data
    sc_mdata.max_m = thrust::reduce(thrust::system::cuda::par, sc_mdata.dev_m_array, sc_mdata.dev_m_array + knum, 0, thrust::maximum<int>());
    sc_mdata.max_n = thrust::reduce(thrust::system::cuda::par, sc_mdata.dev_n_array, sc_mdata.dev_n_array + knum, 0, thrust::maximum<int>());
    sc_mdata.max_k = thrust::reduce(thrust::system::cuda::par, sc_mdata.dev_k_array, sc_mdata.dev_k_array + knum, 0, thrust::maximum<int>());
    sc_mdata.max_ilen = thrust::transform_reduce(thrust::system::cuda::par, start, end, element_diff<int>(sc_mdata.dev_ist, sc_mdata.dev_iend), 0, thrust::maximum<int>());
    sc_mdata.max_jlen = thrust::transform_reduce(thrust::system::cuda::par, start, end, element_diff<int>(sc_mdata.dev_jst, sc_mdata.dev_jend), 0, thrust::maximum<int>());

    printf("SCU %d -> %d: %d %d %d %d %d\n", k_st, k_end, sc_mdata.max_m, sc_mdata.max_n, sc_mdata.max_k, sc_mdata.max_ilen, sc_mdata.max_jlen);

    return thrust::all_of(
        thrust::system::cuda::par, sc_mdata.dev_m_array, sc_mdata.dev_m_array + sc_mdata.batchsize,
        MarshallSCUInner_Predicate()
    );
    // int done_j = 1;
    // for(int k0 = k_st; k0 < k_end; k0++)
    // {
    //     int k = perm_c_supno[k0];
    //     int buffer_index = k0 - k_st;
    //     lpanel_t& lpanel = sc_mdata.lpanels[buffer_index];
    //     upanel_t& upanel = sc_mdata.upanels[buffer_index];

    //     if(lpanel.isEmpty() || upanel.isEmpty())
    //         continue;

    //     int iSt = sc_mdata.ist[buffer_index];
    //     int iEnd = sc_mdata.iend[buffer_index];

    //     int& jSt = sc_mdata.jst[buffer_index];
    //     int& jEnd = sc_mdata.jend[buffer_index];

    //     // Not done if even one operation still has work to do
    //     if(jEnd < upanel.nblocks())
    //     {
    //         jSt = jEnd;
    //         jEnd = upanel.getEndBlock(jSt, sc_mdata.maxGemmCols[buffer_index]);

    //         assert(jEnd > jSt);
    //         done_j = 0;
    //         // printf("k = %d, ist = %d, iend = %d, jst = %d, jend = %d\n", k, iSt, iEnd, jSt, jEnd);

    //         sc_mdata.host_m_array[buffer_index] = lpanel.stRow(iEnd) - lpanel.stRow(iSt);
    //         sc_mdata.host_n_array[buffer_index] = upanel.stCol(jEnd) - upanel.stCol(jSt);
    //         sc_mdata.host_k_array[buffer_index] = supersize(k);

    //         sc_mdata.host_A_ptrs[buffer_index] = lpanel.blkPtrGPU(iSt);
    //         sc_mdata.host_B_ptrs[buffer_index] = upanel.blkPtrGPU(jSt);
    //         sc_mdata.host_C_ptrs[buffer_index] = A_gpu.gpuGemmBuffs[buffer_index];

    //         sc_mdata.host_lda_array[buffer_index] = lpanel.LDA();
    //         sc_mdata.host_ldb_array[buffer_index] = upanel.LDA();
    //         sc_mdata.host_ldc_array[buffer_index] = sc_mdata.host_m_array[buffer_index];
    //     }
    //     else
    //     {
    //         sc_mdata.host_A_ptrs[buffer_index] = NULL;
    //         sc_mdata.host_B_ptrs[buffer_index] = NULL;
    //         sc_mdata.host_C_ptrs[buffer_index] = NULL;

    //         sc_mdata.host_m_array[buffer_index] = 0;
    //         sc_mdata.host_n_array[buffer_index] = 0;
    //         sc_mdata.host_k_array[buffer_index] = 0;

    //         sc_mdata.host_lda_array[buffer_index] = 1;
    //         sc_mdata.host_ldb_array[buffer_index] = 1;
    //         sc_mdata.host_ldc_array[buffer_index] = 1;
    //     }
    // }

    // if(done_j == 0)
    // {
    //     // Upload the buffers to the gpu
    //     sc_mdata.setMaxDims();
    //     sc_mdata.copyToGPU();
    // }

    // return done_j;
}
#endif /* match if (0) */
