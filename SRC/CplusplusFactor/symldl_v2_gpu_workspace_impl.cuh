#pragma once

#include "xlupanels.hpp"
#include "symldl_v2_config.hpp"

#ifdef HAVE_CUDA

static inline size_t symldl_v2_cuda_bytes(int_t count, size_t elem_size,
                                          const char *what)
{
    if (count <= 0)
        return 0;
    size_t n = static_cast<size_t>(count);
    if (elem_size != 0 && n > static_cast<size_t>(-1) / elem_size)
        ABORT(what);
    return n * elem_size;
}

static inline void symldl_v2_cuda_malloc_optional(void **ptr, int_t count,
                                                  size_t elem_size,
                                                  const char *what)
{
    size_t bytes = symldl_v2_cuda_bytes(count, elem_size, what);
    if (bytes == 0)
    {
        *ptr = NULL;
        return;
    }
    gpuErrchk(cudaMalloc(ptr, bytes));
}

template <typename Ftype>
static void symldl_v2_setup_raw_panel_ring(xLUstruct_t<Ftype> *lu,
                                           int nstreams)
{
    lu->symV2RawPanelNodes.clear();
    if (lu->useSymV2Solve() && superlu_sym_v2_wpanel_cache())
        lu->symV2RawPanelNodes.assign(static_cast<size_t>(nstreams),
                                      (int_t) -1);
}

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
        superlu_sym_v2_wpanel_cache() ? lu->maxLvalCount : 0,
        sizeof(Ftype),
        "SymFact V2 W-panel cache allocation overflows.");
}

template <typename Ftype>
static void symldl_v2_setup_gpu_panel_index(xLUstruct_t<Ftype> *lu)
{
    lu->A_gpu.useSymV2PanelIndex = 0;
    lu->A_gpu.symV2PanelLocalIndex = NULL;
}

template <>
inline void symldl_v2_setup_gpu_panel_index<double>(xLUstruct_t<double> *lu)
{
    lu->A_gpu.useSymV2PanelIndex = lu->useSymV2Solve() ? 1 : 0;
    lu->A_gpu.symV2PanelLocalIndex = NULL;
    if (!lu->useSymV2Solve())
        return;
    if (lu->trf3Dpartition == NULL ||
        lu->trf3Dpartition->symV2PanelLocalIndex == NULL)
        ABORT("SymFact V2 panel index metadata is missing.");
    gpuErrchk(cudaMalloc(
        (void **) &lu->A_gpu.symV2PanelLocalIndex,
        sizeof(int_t) * static_cast<size_t>(lu->nsupers)));
    gpuErrchk(cudaMemcpy(lu->A_gpu.symV2PanelLocalIndex,
                         lu->trf3Dpartition->symV2PanelLocalIndex,
                         sizeof(int_t) * static_cast<size_t>(lu->nsupers),
                         cudaMemcpyHostToDevice));
}

#endif
