#pragma once

template <typename Ftype>
void xLUstruct_t<Ftype>::symV2FreeDiagBlocks()
{
    if (symFactWork != NULL)
        SUPERLU_FREE(symFactWork);
    if (symFactIPIV != NULL)
        SUPERLU_FREE(symFactIPIV);
    symFactWork = NULL;
    symFactIPIV = NULL;
    symFactWorkSize = 0;
    symFactIPIVSize = 0;

    for (size_t i = 0; i < symV2DiagBlocks.size(); ++i)
        if (symV2DiagBlocks[i] != NULL)
            SUPERLU_FREE(symV2DiagBlocks[i]);
}

template <typename Ftype>
void xLUstruct_t<Ftype>::symV2FreeCpuStorage()
{
    for (size_t slot = 0; slot < symV2CpuSlotRequestCounts.size(); ++slot)
        if (symV2CpuSlotRequestCounts[slot] != 0)
            ABORT("SymFact V2 CPU slot still owns MPI requests at teardown.");
    for (size_t request = 0; request < symV2CpuRequests.size(); ++request)
        if (symV2CpuRequests[request] != MPI_REQUEST_NULL)
            ABORT("SymFact V2 CPU MPI request remains active at teardown.");

    std::vector<Ftype *> *buffers[] = {
        &symV2CpuRawPanelBufs, &symV2CpuPartnerSendBufs,
        &symV2CpuPartnerRecvBufs,
        &symV2CpuRowSendBufs,
        &symV2CpuRowRecvBufs
    };
    for (size_t b = 0; b < sizeof(buffers) / sizeof(buffers[0]); ++b)
    {
        for (size_t i = 0; i < buffers[b]->size(); ++i)
            if ((*buffers[b])[i] != NULL)
                SUPERLU_FREE((*buffers[b])[i]);
        buffers[b]->clear();
    }
    symV2CpuRawPanelCapacity = 0;
    symV2CpuPartnerSendCapacity = 0;
    symV2CpuPartnerRecvCapacity = 0;
    symV2CpuRowSendCapacity = 0;
    symV2CpuRowRecvCapacity = 0;
    if (symV2CpuPanelPending != NULL)
        SUPERLU_FREE(symV2CpuPanelPending);
    if (symV2CpuSlotPending != NULL)
        SUPERLU_FREE(symV2CpuSlotPending);
    symV2CpuPanelPending = NULL;
    symV2CpuSlotPending = NULL;
#ifdef _OPENMP
    if (symV2CpuOutputLocks != NULL)
    {
        size_t lock_count = symV2CpuOutputLockOffsets.empty()
            ? 0 : symV2CpuOutputLockOffsets.back();
        omp_lock_t *locks =
            static_cast<omp_lock_t *>(symV2CpuOutputLocks);
        for (size_t lock = 0; lock < lock_count; ++lock)
            omp_destroy_lock(&locks[lock]);
        SUPERLU_FREE(symV2CpuOutputLocks);
        symV2CpuOutputLocks = NULL;
    }
#endif
    symV2CpuOutputLockOffsets.clear();
    symV2CpuPanelFactorStarted.clear();
    symV2CpuSlotOwner.clear();
    symV2CpuSlotGeneration.clear();
    symV2CpuRequests.clear();
    symV2CpuRequestPeers.clear();
    symV2CpuRequestKinds.clear();
    symV2CpuWaitIndices.clear();
    symV2CpuWaitStatuses.clear();
    symV2CpuPartnerRecvOffsets.clear();
    symV2CpuPartnerSendRowActive.clear();
    symV2CpuPartnerAssembledIndex.clear();
    symV2CpuPartnerAssembleMaps.clear();
    symV2CpuPartnerRecvChunksRemaining.clear();
    symV2CpuPartnerUpdateSubmitted.clear();
    symV2CpuExchangeStates.clear();
    symV2CpuWindowStates.clear();
    symV2CpuWindowDonePanelBcast.clear();
    symV2CpuWindowDonePanelSolve.clear();
    symV2CpuWindowChildrenLeft.clear();
    symV2CpuDeferredTasksActive = 0;
    symV2CpuSlotRequestCounts.clear();
    symV2CpuSlotSendBegins.clear();
    symV2CpuThreadProfiles.clear();
    symV2CpuRowLookupPanelOffsets.clear();
    symV2CpuRowLookups.clear();
    symV2CpuRowLookupPool.clear();
    symV2CpuRequestsPerSlot = 0;
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
