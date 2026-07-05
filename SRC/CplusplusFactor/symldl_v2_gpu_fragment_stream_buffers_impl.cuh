#pragma once

#include "xlupanels.hpp"
#include "symldl_v2_config.hpp"
#include "symldl_v2_gpu_arena_utils.cuh"

#ifdef HAVE_CUDA

template <typename Ftype>
static void symldl_v2_setup_gpu_fragment_stream_buffers(
    xLUstruct_t<Ftype> *lu, int stream)
{
    lu->A_gpu.symV2RawPanelBufs[stream] = NULL;
    if (!lu->useSymV2Solve())
    {
        lu->A_gpu.symPartnerLvalRecvBufs[stream] = NULL;
        lu->A_gpu.symPartnerLStageBufs[stream] = NULL;
        lu->A_gpu.symPartnerLSendStageBufs[stream] = NULL;
        lu->A_gpu.symPartnerLidxRecvBufs[stream] = NULL;
        lu->A_gpu.symV2RowFragStageBufs[stream] = NULL;
        lu->A_gpu.symV2RowFragValRecvBufs[stream] = NULL;
        lu->A_gpu.symV2RowFragIdxRecvBufs[stream] = NULL;
        lu->A_gpu.symV2RowFragSendMapStageBufs[stream] = NULL;
        return;
    }

    symldl_v2_cuda_malloc_optional(
        (void **) &lu->A_gpu.symPartnerLvalRecvBufs[stream],
        lu->maxSymPartnerLvalCount, sizeof(Ftype),
        "SymFact V2 partner receive buffer allocation overflows.");
    int_t partner_stage_count = lu->maxSymPartnerLvalCount;
    if (lu->Pr <= 1)
        partner_stage_count =
            SUPERLU_MAX(partner_stage_count, lu->maxLvalCount);
    symldl_v2_cuda_malloc_optional(
        (void **) &lu->A_gpu.symPartnerLStageBufs[stream],
        partner_stage_count, sizeof(Ftype),
        "SymFact V2 partner staging buffer allocation overflows.");
    symldl_v2_cuda_malloc_optional(
        (void **) &lu->A_gpu.symPartnerLSendStageBufs[stream],
        lu->maxSymPartnerLSendStageCount, sizeof(Ftype),
        "SymFact V2 partner send staging allocation overflows.");
    symldl_v2_cuda_malloc_optional(
        (void **) &lu->A_gpu.symPartnerLidxRecvBufs[stream],
        lu->maxSymPartnerLidxCount, sizeof(int_t),
        "SymFact V2 partner index buffer allocation overflows.");

    symldl_v2_cuda_malloc_optional(
        (void **) &lu->A_gpu.symV2RowFragStageBufs[stream],
        lu->maxSymV2RowFragStageCount, sizeof(Ftype),
        "SymFact V2 row staging buffer allocation overflows.");
    symldl_v2_cuda_malloc_optional(
        (void **) &lu->A_gpu.symV2RowFragValRecvBufs[stream],
        lu->maxSymV2RowFragValRecvCount, sizeof(Ftype),
        "SymFact V2 row receive buffer allocation overflows.");
    symldl_v2_cuda_malloc_optional(
        (void **) &lu->A_gpu.symV2RowFragIdxRecvBufs[stream],
        lu->maxSymV2RowFragIdxRecvCount, sizeof(int_t),
        "SymFact V2 row index buffer allocation overflows.");
    symldl_v2_cuda_malloc_optional(
        (void **) &lu->A_gpu.symV2RowFragSendMapStageBufs[stream],
        lu->maxSymV2RowFragValSendCount, sizeof(int_t),
        "SymFact V2 row send-map staging allocation overflows.");
    symldl_v2_cuda_malloc_optional(
        (void **) &lu->A_gpu.symV2RawPanelBufs[stream],
        symldl_v2_use_wpanel_cache(lu->grid3d) ? lu->maxLvalCount : 0,
        sizeof(Ftype),
        "SymFact V2 W-panel cache allocation overflows.");
}

#endif
