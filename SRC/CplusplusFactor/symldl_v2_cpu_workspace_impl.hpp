#pragma once

#include <limits>

#include "xlupanels.hpp"

template <typename Ftype>
static void symldl_v2_allocate_cpu_slot_buffers(
    std::vector<Ftype *> &buffers, int slots, size_t capacity,
    const char *overflow_message, const char *allocation_message)
{
    buffers.assign(static_cast<size_t>(slots), NULL);
    for (int slot = 0; slot < slots; ++slot)
    {
        buffers[slot] = (Ftype *) SUPERLU_MALLOC(
            symldl_v2_checked_product(capacity, sizeof(Ftype),
                                      overflow_message));
        if (buffers[slot] == NULL)
            ABORT(allocation_message);
    }
}

template <typename Ftype>
static void symldl_v2_resize_cpu_slot_buffers(
    std::vector<Ftype *> &buffers, size_t &capacity, size_t required,
    const char *overflow_message, const char *allocation_message)
{
    required = SUPERLU_MAX(static_cast<size_t>(1), required);
    if (required <= capacity)
        return;
    int slots = static_cast<int>(buffers.size());
    for (size_t slot = 0; slot < buffers.size(); ++slot)
        if (buffers[slot] != NULL)
            SUPERLU_FREE(buffers[slot]);
    buffers.clear();
    symldl_v2_allocate_cpu_slot_buffers(
        buffers, slots, required, overflow_message, allocation_message);
    capacity = required;
}

template <typename Ftype>
static void symldl_v2_resize_cpu_request_workspace(
    xLUstruct_t<Ftype> *lu)
{
    int slots = static_cast<int>(lu->symV2CpuRawPanelBufs.size());
    if (slots <= 0)
        ABORT("SymFact V2 CPU request workspace has no slots.");

    size_t max_payload = SUPERLU_MAX(
        SUPERLU_MAX(lu->symV2CpuPartnerSendCapacity,
                    lu->symV2CpuPartnerRecvCapacity),
        SUPERLU_MAX(lu->symV2CpuRowSendCapacity,
                    lu->symV2CpuRowRecvCapacity));
    size_t chunks = SUPERLU_MAX(
        static_cast<size_t>(1),
        symldl_v2_cpu_mpi_chunk_count(max_payload));
    size_t peer_messages =
        symldl_v2_checked_product(
            static_cast<size_t>(lu->Pr),
            static_cast<size_t>(lu->Pc + 1),
            "SymFact V2 CPU request workspace overflows.") +
        static_cast<size_t>(lu->Pc + 2);
    size_t requests_per_slot =
        symldl_v2_checked_product(
            peer_messages, chunks,
            "SymFact V2 CPU request workspace overflows.") + 8;
    if (requests_per_slot >
        static_cast<size_t>(std::numeric_limits<int>::max()))
        ABORT("SymFact V2 CPU request workspace exceeds MPI limits.");
    size_t total_requests = symldl_v2_checked_product(
        static_cast<size_t>(slots), requests_per_slot,
        "SymFact V2 CPU request workspace overflows.");

    lu->symV2CpuRequestsPerSlot = requests_per_slot;
    lu->symV2CpuRequests.assign(total_requests, MPI_REQUEST_NULL);
    lu->symV2CpuRequestPeers.assign(total_requests, -1);
    lu->symV2CpuWaitIndices.assign(total_requests, -1);
    lu->symV2CpuWaitStatuses.resize(total_requests);
    lu->symV2CpuPartnerRecvChunksRemaining.assign(
        static_cast<size_t>(lu->Pr), 0);
    lu->symV2CpuPartnerUpdateSubmitted.assign(
        static_cast<size_t>(lu->Pr), 0);
    lu->symV2CpuSlotRequestCounts.assign(
        static_cast<size_t>(slots), 0);
    lu->symV2CpuSlotSendBegins.assign(
        static_cast<size_t>(slots), 0);
}

template <typename Ftype>
static void symldl_v2_allocate_cpu_factor_workspace(
    xLUstruct_t<Ftype> *lu)
{
    if (!lu->symV2UsesCpuFactor())
        return;
    double workspace_start = SuperLU_timer_();

    int slots = lu->options != NULL ? lu->options->num_lookaheads : 1;
    slots = SUPERLU_MAX(1, slots);
    lu->symV2CpuRawPanelCapacity =
        static_cast<size_t>(SUPERLU_MAX((int_t) 1, lu->maxLvalCount));
    lu->symV2CpuPartnerSendCapacity =
        static_cast<size_t>(SUPERLU_MAX((int_t) 1, lu->maxLvalCount));
    lu->symV2CpuPartnerRecvCapacity =
        static_cast<size_t>(SUPERLU_MAX((int_t) 1,
                                        lu->maxSymPartnerLvalCount));
    lu->symV2CpuRowSendCapacity =
        static_cast<size_t>(SUPERLU_MAX((int_t) 1, lu->maxLvalCount));
    lu->symV2CpuRowRecvCapacity =
        static_cast<size_t>(SUPERLU_MAX((int_t) 1, lu->maxLvalCount));

    symldl_v2_allocate_cpu_slot_buffers(
        lu->symV2CpuRawPanelBufs, slots, lu->symV2CpuRawPanelCapacity,
        "SymFact V2 CPU raw-panel workspace overflows.",
        "Malloc fails for SymFact V2 CPU raw-panel workspace.");
    symldl_v2_allocate_cpu_slot_buffers(
        lu->symV2CpuPartnerSendBufs, slots,
        lu->symV2CpuPartnerSendCapacity,
        "SymFact V2 CPU partner-send workspace overflows.",
        "Malloc fails for SymFact V2 CPU partner-send workspace.");
    symldl_v2_allocate_cpu_slot_buffers(
        lu->symV2CpuPartnerRecvBufs, slots,
        lu->symV2CpuPartnerRecvCapacity,
        "SymFact V2 CPU partner-receive workspace overflows.",
        "Malloc fails for SymFact V2 CPU partner-receive workspace.");
    symldl_v2_allocate_cpu_slot_buffers(
        lu->symV2CpuRowSendBufs, slots, lu->symV2CpuRowSendCapacity,
        "SymFact V2 CPU row-send workspace overflows.",
        "Malloc fails for SymFact V2 CPU row-send workspace.");
    symldl_v2_allocate_cpu_slot_buffers(
        lu->symV2CpuRowRecvBufs, slots, lu->symV2CpuRowRecvCapacity,
        "SymFact V2 CPU row-receive workspace overflows.",
        "Malloc fails for SymFact V2 CPU row-receive workspace.");

    symldl_v2_resize_cpu_request_workspace(lu);
    lu->symV2CpuPartnerRecvOffsets.assign(
        static_cast<size_t>(lu->Pr),
        std::numeric_limits<size_t>::max());
    lu->symV2CpuReductionPanelSlots.assign(static_cast<size_t>(slots), -1);
    lu->symV2CpuReductionChunksRemaining.assign(
        static_cast<size_t>(slots), 0);
    int profile_threads = 1;
#ifdef _OPENMP
    profile_threads = SUPERLU_MAX(1, omp_get_max_threads());
#endif
    lu->symV2CpuThreadProfiles.assign(
        static_cast<size_t>(profile_threads), SymLDLV2CpuThreadProfile());
    lu->symV2CpuSlotPending = int32Calloc_dist(slots);
    lu->symV2CpuPanelPending = int32Calloc_dist(lu->symV2PanelCount());
    if (lu->symV2CpuSlotPending == NULL ||
        lu->symV2CpuPanelPending == NULL)
        ABORT("Malloc fails for SymFact V2 CPU scheduler state.");

    lu->symV2CpuOutputLockOffsets.assign(
        static_cast<size_t>(lu->symV2PanelCount()) + 1, 0);
    size_t output_locks = 0;
    for (int_t local_panel = 0;
         local_panel < lu->symV2PanelCount(); ++local_panel)
    {
        lu->symV2CpuOutputLockOffsets[
            static_cast<size_t>(local_panel)] = output_locks;
        xlpanel_t<Ftype> &panel = lu->lPanelVec[local_panel];
        if (!panel.isEmpty())
            output_locks += static_cast<size_t>(panel.nblocks());
    }
    lu->symV2CpuOutputLockOffsets[
        static_cast<size_t>(lu->symV2PanelCount())] = output_locks;
#ifdef _OPENMP
    if (output_locks > 0)
    {
        lu->symV2CpuOutputLocks = SUPERLU_MALLOC(
            symldl_v2_checked_product(
                output_locks, sizeof(omp_lock_t),
                "SymFact V2 CPU output lock table overflows."));
        if (lu->symV2CpuOutputLocks == NULL)
            ABORT("Malloc fails for SymFact V2 CPU output locks.");
        omp_lock_t *locks =
            static_cast<omp_lock_t *>(lu->symV2CpuOutputLocks);
        for (size_t lock = 0; lock < output_locks; ++lock)
            omp_init_lock(&locks[lock]);
    }
#endif

    for (int_t k = 0; k < lu->nsupers; ++k)
    {
        if (lu->isNodeInMyGrid == NULL || lu->isNodeInMyGrid[k] == 0 ||
            lu->symV2PanelRoot(k) != lu->mycol)
            continue;
        int_t n = lu->supersize(k);
        if (n <= 0)
            ABORT("SymFact V2 CPU diagonal workspace has invalid size.");
        if (lu->symV2DiagBlocks[static_cast<size_t>(k)] != NULL)
            continue;
        lu->symV2DiagBlocks[static_cast<size_t>(k)] =
            (Ftype *) SUPERLU_MALLOC(symldl_v2_checked_product(
                symldl_v2_checked_product(
                    static_cast<size_t>(n), static_cast<size_t>(n),
                    "SymFact V2 CPU diagonal workspace overflows."),
                sizeof(Ftype),
                "SymFact V2 CPU diagonal workspace overflows."));
        if (lu->symV2DiagBlocks[static_cast<size_t>(k)] == NULL)
            ABORT("Malloc fails for SymFact V2 CPU diagonal workspace.");
    }
    lu->symV2CpuWorkspaceInitTime += SuperLU_timer_() - workspace_start;
}
