#pragma once

#include "xlupanels.hpp"
#include "symldl_v2_cpu_panel_impl.hpp"
#include "symldl_v2_cpu_exchange_impl.hpp"
#include "symldl_v2_cpu_update_impl.hpp"

template <typename Ftype>
static void symldl_v2_cpu_wait_for_counter(
    xLUstruct_t<Ftype> *lu, int *counter, bool slot_backpressure)
{
    if (counter == NULL)
        ABORT("SymFact V2 CPU scheduler counter is missing.");
    bool blocked = false;
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
#ifdef _OPENMP
#pragma omp taskyield
#endif
    }
}

template <typename Ftype>
static int_t symldl_v2_cpu_factor_forest(
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
                if (lu->mycol == lu->symV2PanelRoot(k))
                {
                    int_t local_panel = lu->symV2PanelIndex(k);
                    if (local_panel < 0 ||
                        local_panel >= lu->symV2PanelCount() ||
                        lu->symV2PanelGid(local_panel) != k)
                        ABORT("SymFact V2 CPU factor panel is invalid.");
                    symldl_v2_cpu_wait_for_counter(
                        lu, &lu->symV2CpuPanelPending[local_panel], false);
                }

                double panel_issue_start = SuperLU_timer_();
                ++lu->symV2CpuPanelsIssued;
                symldl_v2_cpu_capture_raw_panel(lu, k, slot);
                lu->dSymDiagFactorPanelSolve(
                    k, slot, slot, diag_buffers);
                int_t parent = etree->setree[k];

                if (lu->Pr == 1 && lu->Pc == 1)
                {
                    Ftype *raw_values = NULL;
                    xlpanel_t<Ftype> panel = symldl_v2_cpu_exchange_panel(
                        lu, k, slot, &raw_values);
                    symldl_v2_cpu_submit_collapsed_schur_tasks(
                        lu, k, parent, slot, panel, raw_values);
                }
                else
                {
                    symldl_v2_cpu_exchange_fragments_and_update(
                        lu, k, parent, slot, true);
                }
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
#pragma omp taskwait
        }
    }
    lu->symV2CpuSchedulerTime += SuperLU_timer_() - scheduler_start;

    for (int slot = 0; slot < slots; ++slot)
    {
        symldl_v2_cpu_drain_slot_sends(lu, slot);
        if (lu->symV2CpuSlotPending[slot] != 0)
            ABORT("SymFact V2 CPU slot remains active after forest factorization.");
    }
    return 0;
}
