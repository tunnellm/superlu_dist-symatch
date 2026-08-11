#pragma once

#include <algorithm>
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
    lu->symV2CpuRequestKinds.assign(
        total_requests, SYM_LDL_V2_CPU_REQUEST_NONE);
    lu->symV2CpuWaitIndices.assign(total_requests, -1);
    lu->symV2CpuWaitStatuses.resize(total_requests);
    lu->symV2CpuSlotRequestCounts.assign(
        static_cast<size_t>(slots), 0);
    lu->symV2CpuSlotSendBegins.assign(
        static_cast<size_t>(slots), 0);
}

template <typename Ftype>
static void symldl_v2_build_cpu_row_lookups(xLUstruct_t<Ftype> *lu)
{
    size_t panel_count = static_cast<size_t>(lu->symV2PanelCount());
    lu->symV2CpuRowLookupPanelOffsets.assign(panel_count + 1, 0);

    size_t block_count = 0;
    for (size_t local_panel = 0; local_panel < panel_count; ++local_panel)
    {
        lu->symV2CpuRowLookupPanelOffsets[local_panel] = block_count;
        xlpanel_t<Ftype> &panel = lu->lPanelVec[local_panel];
        if (panel.isEmpty())
            continue;
        int_t panel_block_count = panel.nblocks();
        if (panel_block_count <= 0)
            ABORT("SymFact V2 CPU row lookup panel has no blocks.");
        size_t panel_blocks = static_cast<size_t>(panel_block_count);
        if (block_count > std::numeric_limits<size_t>::max() - panel_blocks)
            ABORT("SymFact V2 CPU row lookup table overflows.");
        block_count += panel_blocks;
    }
    lu->symV2CpuRowLookupPanelOffsets[panel_count] = block_count;
    lu->symV2CpuRowLookups.assign(
        block_count, SymLDLV2CpuRowLookup());

    size_t lookup_entries = 0;
    for (size_t local_panel = 0; local_panel < panel_count; ++local_panel)
    {
        xlpanel_t<Ftype> &panel = lu->lPanelVec[local_panel];
        if (panel.isEmpty())
            continue;
        size_t block_base =
            lu->symV2CpuRowLookupPanelOffsets[local_panel];
        for (int_t block = 0; block < panel.nblocks(); ++block)
        {
            int_t rows = panel.nbrow(block);
            if (rows <= 0)
                ABORT("SymFact V2 CPU row lookup has an empty block.");
            int_t *row_list = panel.rowList(block);
            int_t max_row = -1;
            for (int_t row = 0; row < rows; ++row)
            {
                if (row_list[row] < 0)
                    ABORT("SymFact V2 CPU row lookup has a negative row.");
                max_row = SUPERLU_MAX(max_row, row_list[row]);
            }
            size_t dense_extent = static_cast<size_t>(max_row) + 1;
            size_t row_count = static_cast<size_t>(rows);
            bool dense = dense_extent <= row_count ||
                         dense_extent - row_count <= row_count;
            size_t extent = dense ? dense_extent : row_count;
            if (lookup_entries >
                std::numeric_limits<size_t>::max() - extent)
                ABORT("SymFact V2 CPU row lookup pool overflows.");
            SymLDLV2CpuRowLookup &lookup =
                lu->symV2CpuRowLookups[block_base +
                                       static_cast<size_t>(block)];
            lookup.offset = lookup_entries;
            lookup.extent = extent;
            lookup.dense = dense ? 1 : 0;
            lookup_entries += extent;
        }
    }

    lu->symV2CpuRowLookupPool.assign(lookup_entries, (int_t) -1);
    for (size_t local_panel = 0; local_panel < panel_count; ++local_panel)
    {
        xlpanel_t<Ftype> &panel = lu->lPanelVec[local_panel];
        if (panel.isEmpty())
            continue;
        size_t block_base =
            lu->symV2CpuRowLookupPanelOffsets[local_panel];
        for (int_t block = 0; block < panel.nblocks(); ++block)
        {
            int_t rows = panel.nbrow(block);
            int_t *row_list = panel.rowList(block);
            SymLDLV2CpuRowLookup &lookup =
                lu->symV2CpuRowLookups[block_base +
                                       static_cast<size_t>(block)];
            if (lookup.offset > lu->symV2CpuRowLookupPool.size() ||
                lookup.extent >
                    lu->symV2CpuRowLookupPool.size() - lookup.offset)
                ABORT("SymFact V2 CPU row lookup range is invalid.");
            int_t *entries =
                lu->symV2CpuRowLookupPool.data() + lookup.offset;
            if (lookup.dense)
            {
                for (int_t row = 0; row < rows; ++row)
                {
                    size_t gid = static_cast<size_t>(row_list[row]);
                    if (gid >= lookup.extent || entries[gid] != -1)
                        ABORT("SymFact V2 CPU destination rows are duplicated.");
                    entries[gid] = row;
                }
            }
            else
            {
                for (int_t row = 0; row < rows; ++row)
                    entries[row] = row;
                std::sort(
                    entries, entries + rows,
                    [row_list](int_t left, int_t right)
                    {
                        int_t left_gid = row_list[left];
                        int_t right_gid = row_list[right];
                        return left_gid < right_gid ||
                               (left_gid == right_gid && left < right);
                    });
                for (int_t row = 1; row < rows; ++row)
                    if (row_list[entries[row - 1]] ==
                        row_list[entries[row]])
                        ABORT("SymFact V2 CPU destination rows are duplicated.");
            }
        }
    }
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
    SymLDLV2CpuExchangeRoute route = symldl_v2_cpu_exchange_route(lu);
    lu->symV2CpuRawPanelCapacity =
        static_cast<size_t>(SUPERLU_MAX((int_t) 1, lu->maxLvalCount));
    bool needs_partner = route == SYM_LDL_V2_CPU_ROUTE_PC1_PARTNER_ONLY ||
                         route == SYM_LDL_V2_CPU_ROUTE_DUAL_FRAGMENT;
    bool needs_row = route == SYM_LDL_V2_CPU_ROUTE_DUAL_FRAGMENT;
    bool needs_partner_plan = route != SYM_LDL_V2_CPU_ROUTE_COLLAPSED;
    lu->symV2CpuPartnerSendCapacity = needs_partner_plan
        ? static_cast<size_t>(SUPERLU_MAX((int_t) 1, lu->maxLvalCount)) : 0;
    lu->symV2CpuPartnerRecvCapacity = needs_partner_plan
        ? static_cast<size_t>(SUPERLU_MAX((int_t) 1,
                                          lu->maxSymPartnerLvalCount)) : 0;
    lu->symV2CpuRowSendCapacity = needs_row
        ? static_cast<size_t>(SUPERLU_MAX((int_t) 1, lu->maxLvalCount)) : 0;
    lu->symV2CpuRowRecvCapacity = needs_row
        ? static_cast<size_t>(SUPERLU_MAX((int_t) 1, lu->maxLvalCount)) : 0;

    symldl_v2_allocate_cpu_slot_buffers(
        lu->symV2CpuRawPanelBufs, slots, lu->symV2CpuRawPanelCapacity,
        "SymFact V2 CPU raw-panel workspace overflows.",
        "Malloc fails for SymFact V2 CPU raw-panel workspace.");
    if (needs_partner)
    {
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
    }
    if (needs_row)
    {
        symldl_v2_allocate_cpu_slot_buffers(
            lu->symV2CpuRowSendBufs, slots, lu->symV2CpuRowSendCapacity,
            "SymFact V2 CPU row-send workspace overflows.",
            "Malloc fails for SymFact V2 CPU row-send workspace.");
        symldl_v2_allocate_cpu_slot_buffers(
            lu->symV2CpuRowRecvBufs, slots, lu->symV2CpuRowRecvCapacity,
            "SymFact V2 CPU row-receive workspace overflows.",
            "Malloc fails for SymFact V2 CPU row-receive workspace.");
    }
    symldl_v2_resize_cpu_request_workspace(lu);
    size_t exchange_peers = symldl_v2_checked_product(
        static_cast<size_t>(slots), static_cast<size_t>(lu->Pr),
        "SymFact V2 CPU exchange state overflows.");
    lu->symV2CpuPartnerRecvOffsets.assign(
        exchange_peers,
        std::numeric_limits<size_t>::max());
    lu->symV2CpuPartnerRecvChunksRemaining.assign(exchange_peers, 0);
    lu->symV2CpuPartnerUpdateSubmitted.assign(exchange_peers, 0);
    lu->symV2CpuExchangeStates.assign(
        static_cast<size_t>(slots), SymLDLV2CpuExchangeState());
    lu->symV2CpuSlotOwner.assign(static_cast<size_t>(slots), -1);
    lu->symV2CpuSlotGeneration.assign(static_cast<size_t>(slots), 0);
    lu->symV2CpuPanelFactorStarted.assign(
        static_cast<size_t>(lu->symV2PanelCount()), 0);
    lu->symV2CpuDeferredTasksActive = 0;
    lu->symV2CpuReductionPanelSlots.assign(static_cast<size_t>(slots), -1);
    lu->symV2CpuReductionChunksRemaining.assign(
        static_cast<size_t>(slots), 0);
    int profile_threads = SUPERLU_MAX(1, lu->nThreads);
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
        {
            for (int_t block = 1; block < panel.nblocks(); ++block)
                if (panel.gid(block - 1) >= panel.gid(block))
                    ABORT("SymFact V2 CPU local panel blocks are not ordered.");
            output_locks += static_cast<size_t>(panel.nblocks());
        }
    }
    lu->symV2CpuOutputLockOffsets[
        static_cast<size_t>(lu->symV2PanelCount())] = output_locks;
    symldl_v2_build_cpu_row_lookups(lu);
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
