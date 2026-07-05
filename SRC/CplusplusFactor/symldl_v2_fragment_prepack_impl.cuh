#pragma once

#include "xlupanels.hpp"
#include "symldl_v2_config.hpp"
#include "symldl_v2_l_fragment_map_impl.cuh"
#include "symldl_v2_fragment_exchange_common.cuh"

#ifdef HAVE_CUDA

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSymV2PrepackLFragmentsGPU(int_t, int_t)
{
    ABORT("SymFact GPU3DVERSION=2 fragment prepack is not implemented.");
    return 0;
}

template <>
inline int_t xLUstruct_t<double>::dSymV2PrepackLFragmentsGPU(
    int_t k, int_t stream_offset)
{
    if (options->SymFact != YES || symGPU3DVersion != 2)
        return 0;
    if (!superlu_acc_offload)
        ABORT("GPU3DVERSION=2 raw L-fragment prepack requires GPU offload.");
    if (k < 0 || k >= nsupers || mycol != symV2PanelRoot(k))
        return 0;

    if (Pr <= 1)
    {
        int_t lk = symV2PanelIndex(k);
        if (lk < 0)
            return 0;
        xlpanel_t<double> &lpanel = lPanelVec[lk];
        if (!lpanel.isEmpty() && symldl_v2_use_wpanel_cache(grid3d))
        {
            if (stream_offset < 0 || stream_offset >= A_gpu.numCudaStreams)
                stream_offset = 0;
            if (static_cast<size_t>(stream_offset) >=
                    symV2RawPanelNodes.size() ||
                A_gpu.symV2RawPanelBufs[stream_offset] == NULL ||
                A_gpu.symV2RawPanelReadyEvents[stream_offset] == NULL)
                ABORT("SymFact V2 W-panel ring is not initialized.");
            cudaStream_t stream = A_gpu.cuStreams[stream_offset];
            gpuErrchk(cudaMemcpyAsync(
                A_gpu.symV2RawPanelBufs[stream_offset], lpanel.gpuPanel.val,
                static_cast<size_t>(lpanel.nzvalSize()) * sizeof(double),
                cudaMemcpyDeviceToDevice, stream));
            gpuErrchk(cudaEventRecord(
                A_gpu.symV2RawPanelReadyEvents[stream_offset], stream));
            symV2RawPanelNodes[stream_offset] = k;
        }
        return 0;
    }

    if (symV2PartnerLSendBufsGPU.empty() || symL2LSendMapsGPU.empty() ||
        symV2PartnerLSendSizes.empty() ||
        symV2PartnerLSendRowActive.empty() ||
        symV2PartnerLPrepacked.empty())
        ABORT("SymFact V2 raw L-fragment prepack buffers are not allocated.");

    int_t lk = symV2PanelIndex(k);
    if (lk < 0 ||
        static_cast<size_t>(lk) >= symV2PartnerLPrepacked.size())
        ABORT("SymFact V2 raw L-fragment prepack has an invalid local panel.");

    symV2PartnerLPrepacked[static_cast<size_t>(lk)] = 0;
    if (symV2UsePcFragmentSchurPanel(k) ||
        symV2PartnerLSendBufPoolCount == 0)
        return 0;

    if (stream_offset < 0 || stream_offset >= A_gpu.numCudaStreams)
        stream_offset = 0;
    cudaStream_t stream = A_gpu.cuStreams[stream_offset];
    xlpanel_t<double> &lpanel = lPanelVec[lk];

    if (lpanel.isEmpty())
    {
        gpuErrchk(cudaEventRecord(
            A_gpu.symV2PartnerLPackReadyEvents[stream_offset], stream));
        symV2PartnerLPrepacked[static_cast<size_t>(lk)] =
            static_cast<unsigned char>(stream_offset + 1);
        return 0;
    }

    if (symldl_v2_use_wpanel_cache(grid3d))
    {
        if (static_cast<size_t>(stream_offset) >=
                symV2RawPanelNodes.size() ||
            A_gpu.symV2RawPanelBufs[stream_offset] == NULL ||
            A_gpu.symV2RawPanelReadyEvents[stream_offset] == NULL)
            ABORT("SymFact V2 W-panel ring is not initialized.");
        gpuErrchk(cudaMemcpyAsync(
            A_gpu.symV2RawPanelBufs[stream_offset], lpanel.gpuPanel.val,
            static_cast<size_t>(lpanel.nzvalSize()) * sizeof(double),
            cudaMemcpyDeviceToDevice, stream));
        gpuErrchk(cudaEventRecord(
            A_gpu.symV2RawPanelReadyEvents[stream_offset], stream));
        symV2RawPanelNodes[stream_offset] = k;
    }

    bool packed_any = false;
    for (int pc = 0; pc < Pc; ++pc)
    {
        size_t flat = static_cast<size_t>(lk) * static_cast<size_t>(Pc) +
                      static_cast<size_t>(pc);
        if (flat >= symV2PartnerLSendSizes.size())
            ABORT("SymFact V2 raw L-fragment prepack size is missing.");
        int size = symV2PartnerLSendSizes[flat];
        if (size <= 0)
            continue;

        bool active_dest = false;
        for (int pr = 0; pr < Pr; ++pr)
        {
            size_t active_pos =
                flat * static_cast<size_t>(Pr) + static_cast<size_t>(pr);
            if (active_pos >= symV2PartnerLSendRowActive.size())
                ABORT("SymFact V2 raw L-fragment prepack row mask is missing.");
            if (symV2PartnerLSendRowActive[active_pos])
            {
                active_dest = true;
                break;
            }
        }
        if (!active_dest)
            continue;

        double *sendbuf = symV2PartnerLSendBufsGPU[flat];
        int_t *sendmap = symL2LSendMapsGPU[flat];
        if (sendbuf == NULL || sendmap == NULL)
            ABORT("SymFact V2 raw L-fragment prepack buffer is missing.");

        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        symldl_v2_lfrag_pack_kernel<<<blocks, threads, 0, stream>>>(
            lpanel.gpuPanel.val, sendbuf, sendmap, size);
        packed_any = true;
    }
    if (packed_any)
        gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaEventRecord(
        A_gpu.symV2PartnerLPackReadyEvents[stream_offset], stream));
    symV2PartnerLPrepacked[static_cast<size_t>(lk)] =
        static_cast<unsigned char>(stream_offset + 1);
    return 0;
}

#endif
