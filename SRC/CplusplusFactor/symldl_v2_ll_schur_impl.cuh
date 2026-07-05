#pragma once

#include "xlupanels.hpp"
#include "symldl_v2_ll_update_impl.cuh"

#ifdef HAVE_CUDA

template <typename Ftype>
static int_t symldl_v2_ll_lookahead(
    xLUstruct_t<Ftype> *lu, int streamId, int_t k, int_t laIdx,
    xlpanel_t<Ftype> &lpanel)
{
    if (lpanel.isEmpty())
        return 0;

    int_t laLoc = lpanel.find(laIdx);
    if (laLoc == GLOBAL_BLOCK_NOT_FOUND)
        return 0;

    if (k >= 0 &&
        static_cast<size_t>(k) < lu->symPanelReadyEventIds.size() &&
        lu->symPanelReadyEventIds[static_cast<size_t>(k)] >= 0)
    {
        int event_id = lu->symPanelReadyEventIds[static_cast<size_t>(k)];
        gpuErrchk(cudaStreamWaitEvent(lu->A_gpu.lookAheadLStream[streamId],
                                      lu->A_gpu.panelReadyEvents[event_id],
                                      0));
        gpuErrchk(cudaStreamWaitEvent(lu->A_gpu.lookAheadUStream[streamId],
                                      lu->A_gpu.panelReadyEvents[event_id],
                                      0));
    }

    cublasHandle_t colHandle = lu->A_gpu.lookAheadLHandle[streamId];
    cudaStream_t colStream = lu->A_gpu.lookAheadLStream[streamId];
    Ftype *colRawBlock = lu->A_gpu.symPartnerLStageBufs[streamId];
    Ftype *colGemmBuff = lu->A_gpu.lookAheadLGemmBuffer[streamId];

    cublasHandle_t rowHandle = lu->A_gpu.lookAheadUHandle[streamId];
    cudaStream_t rowStream = lu->A_gpu.lookAheadUStream[streamId];
    Ftype *rowRawBlock = lu->A_gpu.lookAheadUGemmBuffer[streamId];
    Ftype *rowGemmBuff = lu->A_gpu.gpuGemmBuffs[streamId];

    if (colRawBlock == NULL || colGemmBuff == NULL ||
        rowRawBlock == NULL || rowGemmBuff == NULL)
        ABORT("SymFact V2 LL lookahead buffers are missing.");

    int_t st_lb = lpanel.haveDiag() ? 1 : 0;
    int_t nlb = lpanel.nblocks();
    int_t la_gid = lpanel.gid(laLoc);
    for (int_t ii = st_lb; ii < nlb; ++ii)
    {
        if (ii == laLoc || lpanel.gid(ii) >= la_gid)
        {
            symldl_v2_ll_part_update(
                lu, ii, ii + 1, laLoc, laLoc + 1, k, lpanel,
                colHandle, colStream, colRawBlock, colGemmBuff);
        }
        else
        {
            symldl_v2_ll_part_update(
                lu, laLoc, laLoc + 1, ii, ii + 1, k, lpanel,
                rowHandle, rowStream, rowRawBlock, rowGemmBuff);
        }
    }

    return 0;
}

template <typename Ftype>
static int_t symldl_v2_ll_exclude(
    xLUstruct_t<Ftype> *lu, int streamId, int_t k, int_t ex,
    xlpanel_t<Ftype> &lpanel)
{
    if (lpanel.isEmpty())
        return 0;

    if (k >= 0 &&
        static_cast<size_t>(k) < lu->symPanelReadyEventIds.size() &&
        lu->symPanelReadyEventIds[static_cast<size_t>(k)] >= 0)
    {
        int event_id = lu->symPanelReadyEventIds[static_cast<size_t>(k)];
        gpuErrchk(cudaStreamWaitEvent(lu->A_gpu.cuStreams[streamId],
                                      lu->A_gpu.panelReadyEvents[event_id],
                                      0));
    }

    int_t st_lb = lpanel.haveDiag() ? 1 : 0;
    int_t nlb = lpanel.nblocks();
    int_t exLoc = lpanel.find(ex);
    cublasHandle_t handle = lu->A_gpu.cuHandles[streamId];
    cudaStream_t cuStream = lu->A_gpu.cuStreams[streamId];
    Ftype *rawBlock = lu->A_gpu.lookAheadUGemmBuffer[streamId];
    Ftype *gemmBuff = lu->A_gpu.gpuGemmBuffs[streamId];
    if (rawBlock == NULL || gemmBuff == NULL)
        ABORT("SymFact V2 LL Schur buffers are missing.");

    if (exLoc == GLOBAL_BLOCK_NOT_FOUND)
    {
        symldl_v2_ll_limited_update(
            lu, st_lb, nlb, st_lb, nlb, k, lpanel,
            handle, cuStream, rawBlock, gemmBuff);
    }
    else
    {
        symldl_v2_ll_limited_update(
            lu, st_lb, exLoc, st_lb, exLoc, k, lpanel,
            handle, cuStream, rawBlock, gemmBuff);
        symldl_v2_ll_limited_update(
            lu, exLoc + 1, nlb, st_lb, exLoc, k, lpanel,
            handle, cuStream, rawBlock, gemmBuff);
        symldl_v2_ll_limited_update(
            lu, exLoc + 1, nlb, exLoc + 1, nlb, k, lpanel,
            handle, cuStream, rawBlock, gemmBuff);
    }

    return 0;
}

#endif
