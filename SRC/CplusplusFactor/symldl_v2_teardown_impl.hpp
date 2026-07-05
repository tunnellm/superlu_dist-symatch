#pragma once

template <typename Ftype>
void xLUstruct_t<Ftype>::symV2FreeDiagBlocks()
{
    if (symFactWork != NULL)
        SUPERLU_FREE(symFactWork);
    if (symFactIPIV != NULL)
        SUPERLU_FREE(symFactIPIV);

    for (size_t i = 0; i < symV2DiagBlocks.size(); ++i)
        if (symV2DiagBlocks[i] != NULL)
            SUPERLU_FREE(symV2DiagBlocks[i]);
}

template <typename Ftype>
void xLUstruct_t<Ftype>::symV2FreeStreamHostBuffers(int stream)
{
    if (stream < (int) symPartnerLvalRecvBufs.size() &&
        symPartnerLvalRecvBufs[stream] != NULL)
    {
#ifdef HAVE_CUDA
        if (symPartnerLvalRecvBufs[stream] ==
            symV2PartnerLHostRecvPoolPinned)
        {
        }
        else if (symV2PartnerLHostRecvPinned)
            cudaFreeHost(symPartnerLvalRecvBufs[stream]);
        else
#endif
            SUPERLU_FREE(symPartnerLvalRecvBufs[stream]);
    }

    if (stream < (int) symPartnerLidxRecvBufs.size())
        SUPERLU_FREE(symPartnerLidxRecvBufs[stream]);

#ifdef HAVE_CUDA
    if (stream < (int) symV2RowFragHostRecvBufs.size() &&
        symV2RowFragHostRecvBufs[stream] != NULL &&
        symV2RowFragHostRecvBufs[stream] != symV2RowFragHostRecvPoolPinned)
    {
        if (symV2RowFragHostRecvPinned)
            cudaFreeHost(symV2RowFragHostRecvBufs[stream]);
        else
            SUPERLU_FREE(symV2RowFragHostRecvBufs[stream]);
    }

    if (stream < (int) symV2RowFragHostSendBufs.size() &&
        symV2RowFragHostSendBufs[stream] != NULL &&
        symV2RowFragHostSendBufs[stream] != symV2RowFragHostSendPoolPinned)
    {
        if (symV2RowFragHostSendPinned)
            cudaFreeHost(symV2RowFragHostSendBufs[stream]);
        else
            SUPERLU_FREE(symV2RowFragHostSendBufs[stream]);
    }
#endif
}

#ifdef HAVE_CUDA
template <typename Ftype>
void xLUstruct_t<Ftype>::symV2FreeGpuStorage()
{
    if (symV2PartnerLSendBufPoolGPU != NULL)
        cudaFree(symV2PartnerLSendBufPoolGPU);
    if (symL2LSendMapPoolGPU != NULL)
        cudaFree(symL2LSendMapPoolGPU);
    if (symV2PartnerLRecvMapPoolGPU != NULL)
        cudaFree(symV2PartnerLRecvMapPoolGPU);
    if (symV2RowFragRecvMapPoolGPU != NULL)
        cudaFree(symV2RowFragRecvMapPoolGPU);
    if (symV2RowDownSendSegPoolGPU != NULL)
        cudaFree(symV2RowDownSendSegPoolGPU);
    if (symV2LPanelArenaGPU != NULL)
        cudaFree(symV2LPanelArenaGPU);
    if (symV2StreamArenaGPU != NULL)
        cudaFree(symV2StreamArenaGPU);
    if (symV2GemmArenaGPU != NULL)
        cudaFree(symV2GemmArenaGPU);

    for (size_t i = 0; i < symV2PartnerLHostSendBufsPinned.size(); ++i)
        if (symV2PartnerLHostSendBufsPinned[i] != NULL &&
            symV2PartnerLHostSendBufsPinned[i] !=
                symV2PartnerLHostSendPoolPinned)
            cudaFreeHost(symV2PartnerLHostSendBufsPinned[i]);

    if (symV2PartnerLHostSendPoolPinned != NULL)
        cudaFreeHost(symV2PartnerLHostSendPoolPinned);
    if (symV2PartnerLHostRecvPoolPinned != NULL)
        cudaFreeHost(symV2PartnerLHostRecvPoolPinned);
    if (symV2RowFragHostRecvPoolPinned != NULL)
        cudaFreeHost(symV2RowFragHostRecvPoolPinned);
    if (symV2RowFragHostSendPoolPinned != NULL)
        cudaFreeHost(symV2RowFragHostSendPoolPinned);

    for (size_t i = 0; i < symV2DiagBlocksGPU.size(); ++i)
        if (symV2DiagBlocksGPU[i] != NULL)
            cudaFree(symV2DiagBlocksGPU[i]);

    for (int stream = 0; stream < A_gpu.numCudaStreams; ++stream)
        if (A_gpu.symV2RawPanelBufs[stream] != NULL &&
            symV2StreamArenaGPU == NULL)
            cudaFree(A_gpu.symV2RawPanelBufs[stream]);
}
#endif
