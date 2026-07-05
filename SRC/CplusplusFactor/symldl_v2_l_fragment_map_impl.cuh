#pragma once

#include "xlupanels.hpp"

#ifdef HAVE_CUDA

static inline int_t *symldl_v2_partner_send_map_gpu(
    xLUstruct_t<double> *lu, size_t flat, int count,
    int stream_offset, cudaStream_t stream)
{
    if (flat >= lu->symV2PartnerLSendSizes.size() ||
        flat >= lu->symV2PartnerLMapOffsets.size() ||
        flat >= lu->symV2PartnerLHostSendScratchOffsets.size())
        ABORT("SymFact V2 L-fragment send map slot is invalid.");
    if (count < 0 || lu->symV2PartnerLSendSizes[flat] != count)
        ABORT("SymFact V2 L-fragment send map size is invalid.");
    if (count == 0)
        return NULL;
    if (flat < lu->symL2LSendMapsGPU.size() &&
        lu->symL2LSendMapsGPU[flat] != NULL)
        return lu->symL2LSendMapsGPU[flat];
    if (lu->A_gpu.symV2RowFragSendMapStageBufs[stream_offset] == NULL)
        ABORT("SymFact V2 L-fragment send map staging is missing.");
    size_t offset = lu->symV2PartnerLHostSendScratchOffsets[flat];
    size_t size = static_cast<size_t>(count);
    if (offset + size >
            static_cast<size_t>(lu->maxSymPartnerLSendStageCount) ||
        offset + size < offset)
        ABORT("SymFact V2 L-fragment send map staging is too small.");
    size_t map_offset = lu->symV2PartnerLMapOffsets[flat];
    if (map_offset + size > lu->symV2PartnerLPackedMaps.size() ||
        map_offset + size < map_offset)
        ABORT("SymFact V2 L-fragment send map offset is invalid.");
    int_t *dst =
        lu->A_gpu.symV2RowFragSendMapStageBufs[stream_offset] + offset;
    gpuErrchk(cudaMemcpyAsync(
        dst, lu->symV2PartnerLPackedMaps.data() + map_offset,
        sizeof(int_t) * size, cudaMemcpyHostToDevice, stream));
    return dst;
}

static inline int_t *symldl_v2_panel_send_maps_gpu(
    xLUstruct_t<double> *lu, int_t lk, int stream_offset,
    cudaStream_t stream, size_t *panel_map_base)
{
    if (lk < 0 || static_cast<size_t>(lk) >= lu->symV2PanelCount())
        ABORT("SymFact V2 L-fragment send map panel is invalid.");
    if (lu->Pc <= 0)
        ABORT("SymFact V2 L-fragment send map grid is invalid.");
    size_t base_flat = static_cast<size_t>(lk) * static_cast<size_t>(lu->Pc);
    size_t end_flat = base_flat + static_cast<size_t>(lu->Pc);
    if (end_flat < base_flat ||
        end_flat > lu->symV2PartnerLSendSizes.size() ||
        end_flat > lu->symV2PartnerLMapOffsets.size() ||
        end_flat > lu->symV2PartnerLHostSendScratchOffsets.size())
        ABORT("SymFact V2 L-fragment send map panel slots are invalid.");
    if (lu->A_gpu.symV2RowFragSendMapStageBufs[stream_offset] == NULL)
        ABORT("SymFact V2 L-fragment send map staging is missing.");

    size_t base = lu->symV2PartnerLMapOffsets[base_flat];
    if (panel_map_base != NULL)
        *panel_map_base = base;
    int_t *stage = lu->A_gpu.symV2RowFragSendMapStageBufs[stream_offset];

    for (size_t flat = base_flat; flat < end_flat; ++flat)
    {
        int count = lu->symV2PartnerLSendSizes[flat];
        if (count < 0)
            ABORT("SymFact V2 L-fragment send map size is invalid.");
        if (count == 0)
            continue;
        size_t size = static_cast<size_t>(count);
        size_t offset = lu->symV2PartnerLHostSendScratchOffsets[flat];
        size_t map_offset = lu->symV2PartnerLMapOffsets[flat];
        if (map_offset < base || map_offset - base != offset)
            ABORT("SymFact V2 L-fragment send map staging offset is inconsistent.");
        if (offset + size >
                static_cast<size_t>(lu->maxSymPartnerLSendStageCount) ||
            offset + size < offset)
            ABORT("SymFact V2 L-fragment send map staging is too small.");
        if (map_offset + size > lu->symV2PartnerLPackedMaps.size() ||
            map_offset + size < map_offset)
            ABORT("SymFact V2 L-fragment send map offset is invalid.");
        gpuErrchk(cudaMemcpyAsync(
            stage + offset, lu->symV2PartnerLPackedMaps.data() + map_offset,
            sizeof(int_t) * size, cudaMemcpyHostToDevice, stream));
    }

    return stage;
}

#endif
