#pragma once

#include <algorithm>
#include "superlu_ddefs.h"
#include "lupanels_GPU.cuh"
#include "lupanels.hpp"
#include "gpuCommon.hpp"

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
        lj = dA->g2lCol(gj);
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

#define ATOMIC_SCATTER
// Atomic Scatter is need if I want to perform multiple Schur Complement
//  update concurrently
#ifdef ATOMIC_SCATTER
            atomicAddT<Ftype>(&Dst[rowS2D[i] + lddst * colS2D[j]], -Src[i + ldsrc * j]);
#else
            Dst[rowS2D[i] + lddst * colS2D[j]] -= Src[i + ldsrc * j];
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
