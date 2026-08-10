#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "xlupanels.hpp"
#include "cublas_cusolver_wrappers.hpp"
#include "symldl_v2_dual_fragment_update_impl.cuh"

#ifdef HAVE_CUDA

static inline bool symldl_v2_trace_fragment_schur()
{
    const char *env = std::getenv("GPU3DV2_TRACE_FRAGMENT_SCHUR");
    return env != NULL && env[0] != '\0' && env[0] != '0';
}

template <typename Ftype>
static int_t symldl_v2_dual_fragment_lookahead(
    xLUstruct_t<Ftype> *lu, int streamId, int_t k, int_t laIdx)
{
    if (!lu->symV2UsePcFragmentSchurPanel(k))
        ABORT("SymFact V2 Pc-fragment lookahead called while disabled.");
    if (k < 0 ||
        static_cast<size_t>(k) >= lu->symV2PartnerLRecvIndex.size())
        ABORT("SymFact V2 Pc-fragment metadata is missing.");
    if (static_cast<size_t>(k) >= lu->symV2RowFragRecvIndex.size())
        ABORT("SymFact V2 Pc-fragment row metadata is missing.");
    lu->symV2RouteProfileNoteDualFragmentLookahead();

    const std::vector<int_t> &row_frag = lu->symV2RowFragRecvIndex[k];
    const std::vector<int_t> &col_frag = lu->symV2PartnerLRecvIndex[k];
    if (symldl_v2_trace_fragment_schur())
    {
        static int printed = 0;
        if (printed < 32)
        {
            std::fprintf(stderr,
                         "[symv2-frag-schur] rank %d lookahead k %d la %d row_blocks %d col_blocks %d\n",
                         lu->grid3d != NULL ? lu->grid3d->iam : -1,
                         static_cast<int>(k), static_cast<int>(laIdx),
                         row_frag.empty() ? 0 : static_cast<int>(row_frag[0]),
                         col_frag.empty() ? 0 : static_cast<int>(col_frag[0]));
            ++printed;
        }
    }
    if (row_frag.empty() || col_frag.empty())
        return 0;

    int_t *row_idx = lu->A_gpu.symV2RowFragIdxRecvBufs[streamId];
    Ftype *row_val = lu->A_gpu.symV2RowFragValRecvBufs[streamId];
    int_t *col_idx = lu->A_gpu.symPartnerLidxRecvBufs[streamId];
    Ftype *col_val = lu->A_gpu.symPartnerLvalRecvBufs[streamId];
    if (row_idx == NULL || row_val == NULL ||
        col_idx == NULL || col_val == NULL)
        ABORT("SymFact V2 Pc-fragment GPU buffers are missing.");

    int_t nrow = symldl_v2_frag_host_nblocks(row_frag);
    int_t ncol = symldl_v2_frag_host_nblocks(col_frag);
    int_t rowLoc = symldl_v2_frag_host_find(row_frag, laIdx);
    int_t colLoc = symldl_v2_frag_host_find(col_frag, laIdx);
    int_t diagRowLoc = symldl_v2_frag_host_find(row_frag, k);
    int_t diagColLoc = symldl_v2_frag_host_find(col_frag, k);

    cublasHandle_t colHandle = lu->A_gpu.lookAheadLHandle[streamId];
    cudaStream_t colStream = lu->A_gpu.lookAheadLStream[streamId];
    Ftype *colGemmBuff = lu->A_gpu.lookAheadLGemmBuffer[streamId];
    cublasHandle_t rowHandle = lu->A_gpu.lookAheadUHandle[streamId];
    cudaStream_t rowStream = lu->A_gpu.lookAheadUStream[streamId];
    Ftype *rowGemmBuff = lu->A_gpu.gpuGemmBuffs[streamId];

    int_t rowRanges[3][2];
    int nRowRanges = symldl_v2_frag_host_ranges_excluding(
        0, nrow, diagRowLoc, GLOBAL_BLOCK_NOT_FOUND, rowRanges);
    if (colLoc != GLOBAL_BLOCK_NOT_FOUND && colLoc != diagColLoc)
    {
        for (int ir = 0; ir < nRowRanges; ++ir)
            symldl_v2_dual_fragment_limited_update(
                lu, rowRanges[ir][0], rowRanges[ir][1],
                colLoc, colLoc + 1, k,
                row_frag, col_frag, row_idx, row_val, col_idx, col_val,
                colHandle, colStream, colGemmBuff);
    }

    if (rowLoc != GLOBAL_BLOCK_NOT_FOUND && rowLoc != diagRowLoc)
    {
        int_t colRanges[3][2];
        int nColRanges = symldl_v2_frag_host_ranges_excluding(
            0, ncol, colLoc, diagColLoc, colRanges);
        for (int ic = 0; ic < nColRanges; ++ic)
            symldl_v2_dual_fragment_limited_update(
                lu, rowLoc, rowLoc + 1,
                colRanges[ic][0], colRanges[ic][1], k,
                row_frag, col_frag, row_idx, row_val, col_idx, col_val,
                rowHandle, rowStream, rowGemmBuff);
    }
    return 0;
}

template <typename Ftype>
static int_t symldl_v2_dual_fragment_exclude(
    xLUstruct_t<Ftype> *lu, int streamId, int_t k, int_t ex)
{
    if (!lu->symV2UsePcFragmentSchurPanel(k))
        ABORT("SymFact V2 Pc-fragment exclude called while disabled.");
    if (k < 0 ||
        static_cast<size_t>(k) >= lu->symV2PartnerLRecvIndex.size())
        ABORT("SymFact V2 Pc-fragment metadata is missing.");
    if (static_cast<size_t>(k) >= lu->symV2RowFragRecvIndex.size())
        ABORT("SymFact V2 Pc-fragment row metadata is missing.");
    lu->symV2RouteProfileNoteDualFragmentExclude();

    const std::vector<int_t> &row_frag = lu->symV2RowFragRecvIndex[k];
    const std::vector<int_t> &col_frag = lu->symV2PartnerLRecvIndex[k];
    if (symldl_v2_trace_fragment_schur())
    {
        static int printed = 0;
        if (printed < 32)
        {
            std::fprintf(stderr,
                         "[symv2-frag-schur] rank %d exclude k %d ex %d row_blocks %d col_blocks %d\n",
                         lu->grid3d != NULL ? lu->grid3d->iam : -1,
                         static_cast<int>(k), static_cast<int>(ex),
                         row_frag.empty() ? 0 : static_cast<int>(row_frag[0]),
                         col_frag.empty() ? 0 : static_cast<int>(col_frag[0]));
            ++printed;
        }
    }
    if (row_frag.empty() || col_frag.empty())
        return 0;

    int_t *row_idx = lu->A_gpu.symV2RowFragIdxRecvBufs[streamId];
    Ftype *row_val = lu->A_gpu.symV2RowFragValRecvBufs[streamId];
    int_t *col_idx = lu->A_gpu.symPartnerLidxRecvBufs[streamId];
    Ftype *col_val = lu->A_gpu.symPartnerLvalRecvBufs[streamId];
    if (row_idx == NULL || row_val == NULL ||
        col_idx == NULL || col_val == NULL)
        ABORT("SymFact V2 Pc-fragment GPU buffers are missing.");

    int_t nrow = symldl_v2_frag_host_nblocks(row_frag);
    int_t ncol = symldl_v2_frag_host_nblocks(col_frag);
    int_t rowLoc = symldl_v2_frag_host_find(row_frag, ex);
    int_t colLoc = symldl_v2_frag_host_find(col_frag, ex);
    int_t diagRowLoc = symldl_v2_frag_host_find(row_frag, k);
    int_t diagColLoc = symldl_v2_frag_host_find(col_frag, k);

    cublasHandle_t handle = lu->A_gpu.cuHandles[streamId];
    cudaStream_t cuStream = lu->A_gpu.cuStreams[streamId];
    Ftype *gemmBuff = lu->A_gpu.gpuGemmBuffs[streamId];

    int_t rowRanges[3][2];
    int_t colRanges[3][2];
    int nRowRanges = symldl_v2_frag_host_ranges_excluding(
        0, nrow, rowLoc, diagRowLoc, rowRanges);
    int nColRanges = symldl_v2_frag_host_ranges_excluding(
        0, ncol, colLoc, diagColLoc, colRanges);
    for (int ir = 0; ir < nRowRanges; ++ir)
    {
        for (int ic = 0; ic < nColRanges; ++ic)
            symldl_v2_dual_fragment_limited_update(
                lu, rowRanges[ir][0], rowRanges[ir][1],
                colRanges[ic][0], colRanges[ic][1], k,
                row_frag, col_frag, row_idx, row_val, col_idx, col_val,
                handle, cuStream, gemmBuff);
    }
    return 0;
}

#include "symldl_v2_l_fragment_schur_impl.cuh"
#include "symldl_v2_ll_schur_impl.cuh"

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymV2LookAheadUpdateGPU(
    int streamId, int_t k, int_t laIdx, xlpanel_t<Ftype> &lpanel)
{
    if (symV2UsePcFragmentSchurPanel(k))
        return symldl_v2_dual_fragment_lookahead(
            this, streamId, k, laIdx);
    if (symV2IsPr1Fastpath())
        return symldl_v2_ll_lookahead(this, streamId, k, laIdx, lpanel);
    return symldl_v2_l_fragment_lookahead(
        this, streamId, k, laIdx, lpanel);
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymV2SchurCompUpdateExcludeOneGPU(
    int streamId, int_t k, int_t ex, xlpanel_t<Ftype> &lpanel)
{
    if (symV2UsePcFragmentSchurPanel(k))
        return symldl_v2_dual_fragment_exclude(
            this, streamId, k, ex);
    if (symV2IsPr1Fastpath())
        return symldl_v2_ll_exclude(this, streamId, k, ex, lpanel);
    return symldl_v2_l_fragment_exclude(
        this, streamId, k, ex, lpanel);
}

#endif
