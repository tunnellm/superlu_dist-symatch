#pragma once

#include "xlupanels.hpp"
#include "symldl_v2_config.hpp"

#ifdef HAVE_CUDA

template <typename Ftype>
static void symldl_v2_setup_raw_panel_ring(xLUstruct_t<Ftype> *lu,
                                           int nstreams)
{
    lu->symV2RawPanelNodes.clear();
    if (lu->useSymV2Solve() &&
        symldl_v2_use_wpanel_cache(lu->grid3d))
        lu->symV2RawPanelNodes.assign(static_cast<size_t>(nstreams),
                                      (int_t) -1);
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
