#pragma once

#include "xlupanels.hpp"
#include "symldl_v2_cpu_panel_impl.hpp"
#include "symldl_v2_cpu_exchange_impl.hpp"
#include "symldl_v2_cpu_update_impl.hpp"
#include "symldl_v2_cpu_hybrid_exchange_impl.hpp"
#include "symldl_v2_cpu_specialized_exchange_impl.hpp"
#include "symldl_v2_factor_gpu_bridge.hpp"
#include "symldl_v2_cpu_window_scheduler_impl.hpp"

template <typename Ftype>
static int symldl_v2_cpu_deferred_tasks_active(xLUstruct_t<Ftype> *lu)
{
    int active = 0;
#ifdef _OPENMP
#pragma omp atomic read
#endif
    active = lu->symV2CpuDeferredTasksActive;
    if (active < 0)
        ABORT("SymFact V2 CPU deferred-task count is invalid.");
    return active;
}

template <typename Ftype>
static bool symldl_v2_cpu_has_active_exchange(xLUstruct_t<Ftype> *lu)
{
    for (size_t slot = 0; slot < lu->symV2CpuExchangeStates.size(); ++slot)
        if (lu->symV2CpuExchangeStates[slot].active)
            return true;
    return false;
}

struct SymLDLV2CpuProgressState
{
    int last_active_tasks = -1;
};

template <typename Ftype>
static bool symldl_v2_cpu_scheduler_progress(
    xLUstruct_t<Ftype> *lu, SymLDLV2CpuProgressState *progress_state)
{
    if (!symldl_v2_cpu_async_exchange_enabled() ||
        !symldl_v2_cpu_has_active_exchange(lu))
        return false;
    if (progress_state == NULL)
        ABORT("SymFact V2 CPU scheduler progress state is missing.");

    int active_tasks = symldl_v2_cpu_deferred_tasks_active(lu);
    bool poll = active_tasks == 0 ||
                active_tasks != progress_state->last_active_tasks;
    bool progressed = false;
    if (poll)
    {
        progressed =
            symldl_v2_cpu_progress_all_exchanges(lu, false);
        active_tasks = symldl_v2_cpu_deferred_tasks_active(lu);
    }

    if (!progressed && active_tasks == 0 &&
        symldl_v2_cpu_has_active_exchange(lu))
    {
        progressed =
            symldl_v2_cpu_progress_all_exchanges(lu, true);
    }
    if (active_tasks > 0 &&
        active_tasks != progress_state->last_active_tasks)
        ++lu->symV2CpuProgressYieldsWithTasks;
    progress_state->last_active_tasks = active_tasks;
    return progressed;
}

template <typename Ftype>
static void symldl_v2_cpu_wait_for_counter(
    xLUstruct_t<Ftype> *lu, int *counter, bool slot_backpressure)
{
    if (counter == NULL)
        ABORT("SymFact V2 CPU scheduler counter is missing.");
    bool blocked = false;
    SymLDLV2CpuProgressState progress_state;
    double wait_start = SuperLU_timer_();
    for (;;)
    {
        int value = 0;
#ifdef _OPENMP
#pragma omp atomic read
#endif
        value = *counter;
        if (value == 0)
        {
            if (blocked)
            {
                double elapsed = SuperLU_timer_() - wait_start;
                lu->symV2CpuSchedulerIdleTime +=
                    elapsed;
                if (slot_backpressure)
                    lu->symV2CpuSlotBackpressureTime += elapsed;
            }
            return;
        }
        if (value < 0)
            ABORT("SymFact V2 CPU scheduler counter is invalid.");
        if (!blocked)
        {
            blocked = true;
            if (slot_backpressure)
                ++lu->symV2CpuSlotBackpressureEvents;
        }
        symldl_v2_cpu_scheduler_progress(lu, &progress_state);
#ifdef _OPENMP
#pragma omp taskyield
#endif
    }
}

template <typename Ftype>
static int_t symldl_v2_cpu_completion_factor_forest(
    xLUstruct_t<Ftype> *lu, sForest_t *forest,
    diagFactBufs_type<Ftype> **diag_buffers,
    gEtreeInfo_t *etree)
{
    if (forest == NULL || forest->nNodes <= 0)
        return 0;
    if (etree == NULL || etree->setree == NULL)
        ABORT("SymFact V2 CPU scheduler requires the elimination tree.");
    int slots = static_cast<int>(lu->symV2CpuRawPanelBufs.size());
    if (slots <= 0)
        ABORT("SymFact V2 CPU scheduler has no panel slots.");

    double scheduler_start = SuperLU_timer_();
#pragma omp parallel
    {
#pragma omp single
        {
#ifdef _OPENMP
            lu->symV2CpuWorkerCount = omp_get_num_threads();
#else
            lu->symV2CpuWorkerCount = 1;
#endif
            for (int_t position = 0; position < forest->nNodes; ++position)
            {
                int_t k = forest->nodeList[position];
                int slot = static_cast<int>(position % slots);
                symldl_v2_cpu_wait_for_counter(
                    lu, &lu->symV2CpuSlotPending[slot], true);
                symldl_v2_cpu_drain_slot_sends(lu, slot);
                if (symldl_v2_cpu_ownership_check_enabled())
                {
                    uint64_t generation =
                        lu->symV2CpuSlotGeneration[static_cast<size_t>(slot)];
                    if (generation == std::numeric_limits<uint64_t>::max())
                        ABORT("SymFact V2 CPU slot generation overflows.");
                    lu->symV2CpuSlotGeneration[
                        static_cast<size_t>(slot)] = generation + 1;
                    lu->symV2CpuSlotOwner[static_cast<size_t>(slot)] = k;
                }
                if (lu->mycol == lu->symV2PanelRoot(k))
                {
                    int_t local_panel = lu->symV2PanelIndex(k);
                    if (local_panel < 0 ||
                        local_panel >= lu->symV2PanelCount() ||
                        lu->symV2PanelGid(local_panel) != k)
                        ABORT("SymFact V2 CPU factor panel is invalid.");
                    symldl_v2_cpu_wait_for_counter(
                        lu, &lu->symV2CpuPanelPending[local_panel], false);
                    symldl_v2_cpu_mark_panel_factor_started(
                        lu, local_panel);
                }

                double panel_issue_start = SuperLU_timer_();
                ++lu->symV2CpuPanelsIssued;
                bool hybrid = symldl_v2_cpu_scheduler_kind() ==
                              SYM_LDL_V2_CPU_SCHEDULER_HYBRID;
                if (!hybrid)
                    symldl_v2_cpu_capture_raw_panel(lu, k, slot);
#ifdef HAVE_CUDA
                pdgstrf3d_symv2_diag_panel_cuda_bridge(
                    static_cast<void *>(lu), k, slot, slot, diag_buffers);
#else
                lu->dSymDiagFactorPanelSolve(
                    k, slot, slot, diag_buffers);
#endif
                int_t parent = etree->setree[k];
                SymLDLV2CpuExchangeRoute route =
                    symldl_v2_cpu_exchange_route(lu);
                ++lu->symV2CpuRoutePanels[static_cast<int>(route)];

                if (route == SYM_LDL_V2_CPU_ROUTE_COLLAPSED)
                {
                    if (hybrid)
                    {
                        symldl_v2_cpu_hybrid_reconstruct_collapsed_raw(
                            lu, k, slot);
                        int_t local_panel = lu->symV2PanelIndex(k);
                        xlpanel_t<Ftype> &row_panel =
                            lu->lPanelVec[local_panel];
                        xlpanel_t<Ftype> column_panel(
                            row_panel.index,
                            lu->symV2CpuRawPanelBufs[slot]);
                        symldl_v2_cpu_hybrid_submit_collapsed(
                            lu, k, parent, slot, row_panel, column_panel);
                    }
                    else
                    {
                        Ftype *raw_values = NULL;
                        xlpanel_t<Ftype> panel = symldl_v2_cpu_exchange_panel(
                            lu, k, slot, &raw_values);
                        symldl_v2_cpu_submit_collapsed_schur_tasks(
                            lu, k, parent, slot, panel, raw_values);
                    }
                    ++lu->symV2CpuReleaseEventsLocal;
                }
                else if (route == SYM_LDL_V2_CPU_ROUTE_PC1_PARTNER_ONLY)
                {
                    if (hybrid)
                        symldl_v2_cpu_issue_hybrid_exchange(
                            lu, k, parent, slot, route);
                    else
                        symldl_v2_cpu_issue_pc1_exchange(
                            lu, k, parent, slot);
                    if (symldl_v2_cpu_async_exchange_enabled())
                        symldl_v2_cpu_progress_all_exchanges(lu, false);
                    else
                        symldl_v2_cpu_complete_exchange(lu, slot);
                }
                else
                {
                    if (hybrid)
                    {
                        symldl_v2_cpu_issue_hybrid_exchange(
                            lu, k, parent, slot, route);
                        symldl_v2_cpu_progress_all_exchanges(
                            lu, false);
                    }
                    else if (symldl_v2_cpu_async_exchange_enabled())
                    {
                        symldl_v2_cpu_issue_fragment_exchange(
                            lu, k, parent, slot, true);
                        symldl_v2_cpu_progress_all_exchanges(
                            lu, false);
                    }
                    else
                        symldl_v2_cpu_exchange_fragments_and_update(
                            lu, k, parent, slot, true);
                }
                if (route == SYM_LDL_V2_CPU_ROUTE_COLLAPSED)
                    ++lu->symV2CpuPanelsCompleted;
                lu->symV2CpuPanelIssueTime +=
                    SuperLU_timer_() - panel_issue_start;
                uint64_t active_slots = 0;
                for (int active_slot = 0; active_slot < slots;
                     ++active_slot)
                    if (lu->symV2CpuSlotPending[active_slot] != 0 ||
                        lu->symV2CpuSlotRequestCounts[
                            static_cast<size_t>(active_slot)] != 0)
                        ++active_slots;
                lu->symV2CpuActiveSlotHighWater = SUPERLU_MAX(
                    lu->symV2CpuActiveSlotHighWater, active_slots);
            }
            SymLDLV2CpuProgressState drain_progress_state;
            while (symldl_v2_cpu_has_active_exchange(lu))
            {
                symldl_v2_cpu_scheduler_progress(
                    lu, &drain_progress_state);
#ifdef _OPENMP
#pragma omp taskyield
#endif
            }
#pragma omp taskwait
            if (symldl_v2_cpu_deferred_tasks_active(lu) != 0)
                ABORT("SymFact V2 CPU tasks remain active after taskwait.");
        }
    }
    lu->symV2CpuSchedulerTime += SuperLU_timer_() - scheduler_start;

    for (int slot = 0; slot < slots; ++slot)
    {
        if (lu->symV2CpuExchangeStates[static_cast<size_t>(slot)].active)
            ABORT("SymFact V2 CPU exchange remains active after factorization.");
        symldl_v2_cpu_drain_slot_sends(lu, slot);
        if (lu->symV2CpuSlotPending[slot] != 0)
            ABORT("SymFact V2 CPU slot remains active after forest factorization.");
    }
    return 0;
}

template <typename Ftype>
static int_t symldl_v2_cpu_factor_forest(
    xLUstruct_t<Ftype> *lu, sForest_t *forest,
    diagFactBufs_type<Ftype> **diag_buffers, gEtreeInfo_t *etree)
{
    if (symldl_v2_cpu_scheduler_kind() ==
        SYM_LDL_V2_CPU_SCHEDULER_WINDOW)
        return symldl_v2_cpu_window_factor_forest(
            lu, forest, diag_buffers, etree);
    return symldl_v2_cpu_completion_factor_forest(
        lu, forest, diag_buffers, etree);
}
