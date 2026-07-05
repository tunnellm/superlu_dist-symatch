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

#endif
