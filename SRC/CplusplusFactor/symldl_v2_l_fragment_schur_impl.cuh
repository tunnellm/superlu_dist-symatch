#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "xlupanels.hpp"
#include "symldl_v2_l_fragment_update_impl.cuh"

#ifdef HAVE_CUDA

template <typename Ftype>
static int_t symldl_v2_l_fragment_host_find(
    const std::vector<int_t> &frag, int_t gid)
{
    if (frag.empty())
        return GLOBAL_BLOCK_NOT_FOUND;
    for (int_t i = 0; i < frag[0]; ++i)
        if (frag[LPANEL_HEADER_SIZE + i] == gid)
            return i;
    return GLOBAL_BLOCK_NOT_FOUND;
}

template <typename Ftype>
static int_t symldl_v2_l_fragment_lookahead(
    xLUstruct_t<Ftype> *lu, int streamId, int_t k, int_t laIdx,
    xlpanel_t<Ftype> &lpanel)
{
    if (lpanel.isEmpty())
        return 0;
    if (k < 0 ||
        static_cast<size_t>(k) >= lu->symV2PartnerLRecvIndex.size())
        ABORT("SymFact V2 L-fragment metadata is missing.");

    const std::vector<int_t> &frag = lu->symV2PartnerLRecvIndex[k];
    if (frag.empty())
        return 0;

    int_t st_lb = lpanel.haveDiag() ? 1 : 0;
    int_t nlb = lpanel.nblocks();
    int_t nfrag = frag[0];
    int_t laILoc = lpanel.find(laIdx);
    int_t laJLoc = symldl_v2_l_fragment_host_find<Ftype>(frag, laIdx);

    int_t *frag_index = lu->A_gpu.symPartnerLidxRecvBufs[streamId];
    Ftype *frag_val = lu->A_gpu.symPartnerLvalRecvBufs[streamId];
    if (frag_index == NULL || frag_val == NULL)
        ABORT("SymFact V2 L-fragment GPU buffers are missing.");

    cublasHandle_t colHandle = lu->A_gpu.lookAheadLHandle[streamId];
    cudaStream_t colStream = lu->A_gpu.lookAheadLStream[streamId];
    Ftype *colGemmBuff = lu->A_gpu.lookAheadLGemmBuffer[streamId];

    cublasHandle_t rowHandle = lu->A_gpu.lookAheadUHandle[streamId];
    cudaStream_t rowStream = lu->A_gpu.lookAheadUStream[streamId];
    Ftype *rowGemmBuff = lu->A_gpu.gpuGemmBuffs[streamId];

    if (laJLoc != GLOBAL_BLOCK_NOT_FOUND)
        symldl_v2_l_fragment_limited_update(
            lu, st_lb, nlb, laJLoc, laJLoc + 1, k, lpanel, frag,
            frag_index, frag_val, colHandle, colStream, colGemmBuff);

    if (laILoc != GLOBAL_BLOCK_NOT_FOUND)
    {
        if (laJLoc == GLOBAL_BLOCK_NOT_FOUND)
        {
            symldl_v2_l_fragment_limited_update(
                lu, laILoc, laILoc + 1, 0, nfrag, k, lpanel, frag,
                frag_index, frag_val, rowHandle, rowStream, rowGemmBuff);
        }
        else
        {
            symldl_v2_l_fragment_limited_update(
                lu, laILoc, laILoc + 1, 0, laJLoc, k, lpanel, frag,
                frag_index, frag_val, rowHandle, rowStream, rowGemmBuff);
            symldl_v2_l_fragment_limited_update(
                lu, laILoc, laILoc + 1, laJLoc + 1, nfrag, k, lpanel,
                frag, frag_index, frag_val, rowHandle, rowStream,
                rowGemmBuff);
        }
    }

    return 0;
}

template <typename Ftype>
static int_t symldl_v2_l_fragment_exclude(
    xLUstruct_t<Ftype> *lu, int streamId, int_t k, int_t ex,
    xlpanel_t<Ftype> &lpanel)
{
    if (lpanel.isEmpty())
        return 0;
    if (k < 0 ||
        static_cast<size_t>(k) >= lu->symV2PartnerLRecvIndex.size())
        ABORT("SymFact V2 L-fragment metadata is missing.");

    const std::vector<int_t> &frag = lu->symV2PartnerLRecvIndex[k];
    if (frag.empty())
        return 0;

    int_t st_lb = lpanel.haveDiag() ? 1 : 0;
    int_t nlb = lpanel.nblocks();
    int_t nfrag = frag[0];
    int_t exILoc = lpanel.find(ex);
    int_t exJLoc = symldl_v2_l_fragment_host_find<Ftype>(frag, ex);

    int_t *frag_index = lu->A_gpu.symPartnerLidxRecvBufs[streamId];
    Ftype *frag_val = lu->A_gpu.symPartnerLvalRecvBufs[streamId];
    if (frag_index == NULL || frag_val == NULL)
        ABORT("SymFact V2 L-fragment GPU buffers are missing.");

    cublasHandle_t handle = lu->A_gpu.cuHandles[streamId];
    cudaStream_t cuStream = lu->A_gpu.cuStreams[streamId];
    Ftype *gemmBuff = lu->A_gpu.gpuGemmBuffs[streamId];

    auto update_i_range = [&](int_t ist, int_t iend)
    {
        if (ist >= iend)
            return;
        if (exJLoc == GLOBAL_BLOCK_NOT_FOUND)
        {
            symldl_v2_l_fragment_limited_update(
                lu, ist, iend, 0, nfrag, k, lpanel, frag,
                frag_index, frag_val, handle, cuStream, gemmBuff);
        }
        else
        {
            symldl_v2_l_fragment_limited_update(
                lu, ist, iend, 0, exJLoc, k, lpanel, frag,
                frag_index, frag_val, handle, cuStream, gemmBuff);
            symldl_v2_l_fragment_limited_update(
                lu, ist, iend, exJLoc + 1, nfrag, k, lpanel, frag,
                frag_index, frag_val, handle, cuStream, gemmBuff);
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

#endif
