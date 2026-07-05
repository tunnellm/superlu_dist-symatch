#pragma once

#include <algorithm>
#include <vector>

#include "xlupanels.hpp"
#include "gpuCommon.hpp"
#include "symldl_v2_config.hpp"

#ifdef HAVE_CUDA

static inline void symldl_v2_cuda_malloc_or_abort(void **ptr, size_t bytes,
                                                 const char *what)
{
    cudaError_t err = cudaMalloc(ptr, bytes);
    if (err == cudaSuccess)
        return;

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cudaError_t mem_err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (mem_err == cudaSuccess)
        fprintf(stderr,
                "%s: requested %zu bytes, free %zu bytes, total %zu bytes.\n",
                what, bytes, free_bytes, total_bytes);
    else
        fprintf(stderr, "%s: requested %zu bytes.\n", what, bytes);
    fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
    ABORT("SymFact V2 GPU allocation failed.");
}

template <typename Ftype>
static void symldl_v2_materialize_pcfrag_metadata(xLUstruct_t<Ftype> *lu)
{
    if (!lu->useSymV2Solve() || !lu->superlu_acc_offload)
        return;

    if (lu->symV2PartnerLSendBufPoolCount > 0)
    {
        if (lu->symV2PartnerLSendBufPoolGPU != NULL)
            ABORT("SymFact V2 partner send value pool already exists.");
        symldl_v2_cuda_malloc_or_abort(
            (void **) &lu->symV2PartnerLSendBufPoolGPU,
            sizeof(Ftype) * lu->symV2PartnerLSendBufPoolCount,
            "SymFact V2 partner send value pool allocation");
        for (size_t flat = 0; flat < lu->symV2PartnerLSendSizes.size();
             ++flat)
        {
            int size = lu->symV2PartnerLSendSizes[flat];
            if (size <= 0)
                continue;
            size_t offset = lu->symV2PartnerLMapOffsets[flat];
            if (offset + static_cast<size_t>(size) >
                    lu->symV2PartnerLSendBufPoolCount ||
                offset + static_cast<size_t>(size) < offset)
                ABORT("SymFact V2 partner send value offset is invalid.");
            lu->symV2PartnerLSendBufsGPU[flat] =
                lu->symV2PartnerLSendBufPoolGPU + offset;
        }
    }

    const bool stream_l2l_send_maps =
        superlu_sym_v2_pc_fragment_ldl_native();
    if (lu->symL2LSendMapPoolCount > 0 && !stream_l2l_send_maps)
    {
        if (lu->symV2PartnerLPackedMaps.size() !=
            lu->symL2LSendMapPoolCount)
            ABORT("SymFact V2 partner-L send map pool size mismatch.");
        if (lu->symL2LSendMapPoolGPU != NULL)
            ABORT("SymFact V2 partner-L send map pool already exists.");
        symldl_v2_cuda_malloc_or_abort(
            (void **) &lu->symL2LSendMapPoolGPU,
            sizeof(int_t) * lu->symL2LSendMapPoolCount,
            "SymFact V2 partner-L send map pool allocation");
        gpuErrchk(cudaMemcpy(lu->symL2LSendMapPoolGPU,
                             lu->symV2PartnerLPackedMaps.data(),
                             sizeof(int_t) * lu->symL2LSendMapPoolCount,
                             cudaMemcpyHostToDevice));
        for (size_t flat = 0; flat < lu->symV2PartnerLSendSizes.size();
             ++flat)
        {
            int size = lu->symV2PartnerLSendSizes[flat];
            if (size <= 0)
                continue;
            size_t offset = lu->symV2PartnerLMapOffsets[flat];
            if (offset + static_cast<size_t>(size) >
                    lu->symL2LSendMapPoolCount ||
                offset + static_cast<size_t>(size) < offset)
                ABORT("SymFact V2 partner-L send map offset is invalid.");
            lu->symL2LSendMapsGPU[flat] =
                lu->symL2LSendMapPoolGPU + offset;
        }
    }

    if (lu->symV2PartnerLHostSendPoolPinnedCount > 0)
    {
        if (lu->symV2PartnerLHostSendPoolPinned != NULL)
            ABORT("SymFact V2 partner send staging pool already exists.");
        gpuErrchk(cudaMallocHost(
            (void **) &lu->symV2PartnerLHostSendPoolPinned,
            sizeof(Ftype) * lu->symV2PartnerLHostSendPoolPinnedCount));
        for (size_t flat = 0; flat < lu->symV2PartnerLSendSizes.size();
             ++flat)
        {
            int size = lu->symV2PartnerLSendSizes[flat];
            if (size <= 0)
                continue;
            size_t offset = lu->symV2PartnerLHostSendScratchOffsets[flat];
            if (offset + static_cast<size_t>(size) >
                    lu->symV2PartnerLHostSendPoolPinnedCount ||
                offset + static_cast<size_t>(size) < offset)
                ABORT("SymFact V2 partner send staging offset is invalid.");
            lu->symV2PartnerLHostSendBufsPinned[flat] =
                lu->symV2PartnerLHostSendPoolPinned + offset;
        }
    }
    else if (superlu_sym_v2_pinned_staging() && !superlu_cuda_aware_mpi())
    {
        for (size_t flat = 0; flat < lu->symV2PartnerLSendSizes.size();
             ++flat)
        {
            int size = lu->symV2PartnerLSendSizes[flat];
            if (size <= 0)
                continue;
            if (lu->symV2PartnerLHostSendBufsPinned[flat] != NULL)
                ABORT("SymFact V2 partner send staging already exists.");
            gpuErrchk(cudaMallocHost(
                (void **) &lu->symV2PartnerLHostSendBufsPinned[flat],
                sizeof(Ftype) * static_cast<size_t>(size)));
        }
    }

    if (lu->symV2PartnerLRecvMapPoolCount > 0)
    {
        if (lu->symV2PartnerLRecvMapPoolGPU != NULL)
            ABORT("SymFact V2 partner receive map pool already exists.");
        std::vector<int_t> packed(lu->symV2PartnerLRecvMapPoolCount, 0);
        if (lu->symV2PartnerLRecvMap.size() !=
                lu->symV2PartnerLRecvMapOffsets.size() ||
            lu->symV2PartnerLRecvMap.size() !=
                lu->symV2PartnerLRecvMapsGPU.size())
            ABORT("SymFact V2 partner receive map table size mismatch.");
        for (size_t pos = 0; pos < lu->symV2PartnerLRecvMap.size(); ++pos)
        {
            if (lu->symV2PartnerLRecvMap[pos].empty())
                continue;
            size_t offset = lu->symV2PartnerLRecvMapOffsets[pos];
            if (offset + lu->symV2PartnerLRecvMap[pos].size() >
                    lu->symV2PartnerLRecvMapPoolCount ||
                offset + lu->symV2PartnerLRecvMap[pos].size() < offset)
                ABORT("SymFact V2 partner receive map offset is invalid.");
            std::copy(lu->symV2PartnerLRecvMap[pos].begin(),
                      lu->symV2PartnerLRecvMap[pos].end(),
                      packed.begin() + offset);
        }
        symldl_v2_cuda_malloc_or_abort(
            (void **) &lu->symV2PartnerLRecvMapPoolGPU,
            sizeof(int_t) * lu->symV2PartnerLRecvMapPoolCount,
            "SymFact V2 partner receive map pool allocation");
        gpuErrchk(cudaMemcpy(lu->symV2PartnerLRecvMapPoolGPU,
                             packed.data(),
                             sizeof(int_t) *
                                 lu->symV2PartnerLRecvMapPoolCount,
                             cudaMemcpyHostToDevice));
        for (size_t pos = 0; pos < lu->symV2PartnerLRecvMap.size(); ++pos)
        {
            if (!lu->symV2PartnerLRecvMap[pos].empty())
                lu->symV2PartnerLRecvMapsGPU[pos] =
                    lu->symV2PartnerLRecvMapPoolGPU +
                    lu->symV2PartnerLRecvMapOffsets[pos];
        }
    }

    if (lu->symV2RowFragRecvMapPoolCount > 0)
    {
        if (lu->symV2RowFragRecvMapPoolGPU != NULL)
            ABORT("SymFact V2 row-fragment receive map pool already exists.");
        std::vector<int_t> packed(lu->symV2RowFragRecvMapPoolCount, 0);
        if (lu->symV2RowFragRecvMap.size() !=
                lu->symV2RowFragRecvMapOffsets.size() ||
            lu->symV2RowFragRecvMap.size() !=
                lu->symV2RowFragRecvMapsGPU.size())
            ABORT("SymFact V2 row-fragment receive map table size mismatch.");
        for (size_t pos = 0; pos < lu->symV2RowFragRecvMap.size(); ++pos)
        {
            if (lu->symV2RowFragRecvMap[pos].empty())
                continue;
            size_t offset = lu->symV2RowFragRecvMapOffsets[pos];
            if (offset + lu->symV2RowFragRecvMap[pos].size() >
                    lu->symV2RowFragRecvMapPoolCount ||
                offset + lu->symV2RowFragRecvMap[pos].size() < offset)
                ABORT("SymFact V2 row-fragment receive map offset is invalid.");
            std::copy(lu->symV2RowFragRecvMap[pos].begin(),
                      lu->symV2RowFragRecvMap[pos].end(),
                      packed.begin() + offset);
        }
        symldl_v2_cuda_malloc_or_abort(
            (void **) &lu->symV2RowFragRecvMapPoolGPU,
            sizeof(int_t) * lu->symV2RowFragRecvMapPoolCount,
            "SymFact V2 row-fragment receive map pool allocation");
        gpuErrchk(cudaMemcpy(lu->symV2RowFragRecvMapPoolGPU,
                             packed.data(),
                             sizeof(int_t) *
                                 lu->symV2RowFragRecvMapPoolCount,
                             cudaMemcpyHostToDevice));
        for (size_t pos = 0; pos < lu->symV2RowFragRecvMap.size(); ++pos)
        {
            if (!lu->symV2RowFragRecvMap[pos].empty())
                lu->symV2RowFragRecvMapsGPU[pos] =
                    lu->symV2RowFragRecvMapPoolGPU +
                    lu->symV2RowFragRecvMapOffsets[pos];
        }
    }

    if (lu->symV2RowDownSendSegPoolCount > 0)
    {
        if (lu->symV2RowDownSendSegPoolGPU != NULL)
            ABORT("SymFact V2 row-down send segment pool already exists.");
        if (lu->symV2RowDownSendSegsHost.size() !=
            lu->symV2RowDownSendSegPoolCount)
            ABORT("SymFact V2 row-down send segment pool size mismatch.");
        symldl_v2_cuda_malloc_or_abort(
            (void **) &lu->symV2RowDownSendSegPoolGPU,
            sizeof(SymV2RowDownSendSegmentGPU) *
                lu->symV2RowDownSendSegPoolCount,
            "SymFact V2 row-down send segment pool allocation");
        gpuErrchk(cudaMemcpy(lu->symV2RowDownSendSegPoolGPU,
                             lu->symV2RowDownSendSegsHost.data(),
                             sizeof(SymV2RowDownSendSegmentGPU) *
                                 lu->symV2RowDownSendSegPoolCount,
                             cudaMemcpyHostToDevice));
        if (lu->symV2RowDownSendSegsGPU.size() !=
                lu->symV2RowDownSendSizes.size() ||
            lu->symV2RowDownSendSegOffsets.size() !=
                lu->symV2RowDownSendSizes.size() ||
            lu->symV2RowDownSendSegCounts.size() !=
                lu->symV2RowDownSendSizes.size())
            ABORT("SymFact V2 row-down send segment table size mismatch.");
        for (size_t slot = 0; slot < lu->symV2RowDownSendSizes.size();
             ++slot)
        {
            if (lu->symV2RowDownSendSizes[slot] <= 0)
                continue;
            int count = lu->symV2RowDownSendSegCounts[slot];
            if (count <= 0)
                ABORT("SymFact V2 row-down send segment slot is empty.");
            size_t offset = lu->symV2RowDownSendSegOffsets[slot];
            if (offset + static_cast<size_t>(count) >
                    lu->symV2RowDownSendSegPoolCount ||
                offset + static_cast<size_t>(count) < offset)
                ABORT("SymFact V2 row-down send segment offset is invalid.");
            lu->symV2RowDownSendSegsGPU[slot] =
                lu->symV2RowDownSendSegPoolGPU + offset;
        }
    }
}
#endif
