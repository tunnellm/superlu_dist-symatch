#pragma once

#include "xlupanels.hpp"
#include "symldl_v2_cpu_panel_impl.hpp"
#include "symldl_v2_cpu_window_exchange_impl.hpp"
#include "symldl_v2_cpu_window_update_impl.hpp"
#include "symldl_v2_factor_gpu_bridge.hpp"

template <typename Ftype>
static void symldl_v2_cpu_window_factor_panel(
    xLUstruct_t<Ftype> *lu, int_t k,
    diagFactBufs_type<Ftype> **diag_buffers)
{
    if (lu->mycol == lu->symV2PanelRoot(k))
    {
        int_t local_panel = lu->symV2PanelIndex(k);
        if (local_panel < 0 || local_panel >= lu->symV2PanelCount() ||
            lu->symV2PanelGid(local_panel) != k)
            ABORT("SymFact V2 CPU window factor panel is invalid.");
        symldl_v2_cpu_mark_panel_factor_started(lu, local_panel);
    }
#ifdef HAVE_CUDA
    pdgstrf3d_symv2_diag_panel_cuda_bridge(
        static_cast<void *>(lu), k, 0, 0, diag_buffers);
#else
    lu->dSymDiagFactorPanelSolve(k, 0, 0, diag_buffers);
#endif
}

template <typename Ftype>
static int_t symldl_v2_cpu_window_factor_forest(
    xLUstruct_t<Ftype> *lu, sForest_t *forest,
    diagFactBufs_type<Ftype> **diag_buffers, gEtreeInfo_t *etree)
{
    if (forest == NULL || forest->nNodes <= 0)
        return 0;
    if (etree == NULL || etree->setree == NULL)
        ABORT("SymFact V2 CPU window scheduler requires the elimination tree.");
    int slots = static_cast<int>(lu->symV2CpuWindowStates.size());
    if (slots <= 0)
        ABORT("SymFact V2 CPU window scheduler has no panel slots.");

    int_t node_count = forest->nNodes;
    int_t *nodes = forest->nodeList;
    treeTopoInfo_t *topology = &forest->topoInfo;
    int_t *inverse = topology->myIperm;
    int_t *level_limits = topology->eTreeTopLims;
    if (inverse == NULL || level_limits == NULL || topology->numLvl <= 0)
        ABORT("SymFact V2 CPU window topology is incomplete.");
    if (static_cast<size_t>(node_count) >
            lu->symV2CpuWindowDonePanelBcast.size() ||
        static_cast<size_t>(node_count) >
            lu->symV2CpuWindowDonePanelSolve.size() ||
        static_cast<size_t>(node_count) >
            lu->symV2CpuWindowChildrenLeft.size())
        ABORT("SymFact V2 CPU window scheduler state is undersized.");
    std::fill(lu->symV2CpuWindowDonePanelBcast.begin(),
              lu->symV2CpuWindowDonePanelBcast.begin() + node_count, 0);
    std::fill(lu->symV2CpuWindowDonePanelSolve.begin(),
              lu->symV2CpuWindowDonePanelSolve.begin() + node_count, 0);
    std::fill(lu->symV2CpuWindowChildrenLeft.begin(),
              lu->symV2CpuWindowChildrenLeft.begin() + node_count, 0);
    for (int slot = 0; slot < slots; ++slot)
        if (lu->symV2CpuWindowStates[static_cast<size_t>(slot)].ready)
            ABORT("SymFact V2 CPU window slot remains active at forest entry.");

    for (int_t position = 0; position < node_count; ++position)
    {
        int_t parent = etree->setree[nodes[position]];
        int_t parent_position = parent >= 0 && parent < lu->nsupers
                                    ? inverse[parent]
                                    : -1;
        if (parent_position >= 0 && parent_position < node_count)
            ++lu->symV2CpuWindowChildrenLeft[
                static_cast<size_t>(parent_position)];
    }

    int_t first_level_end = level_limits[1];
    if (first_level_end <= 0 || first_level_end > node_count)
        ABORT("SymFact V2 CPU window first topology level is invalid.");
    int half = slots >= 2 ? slots / 2 : 1;
    int_t window_capacity = SUPERLU_MIN(
        static_cast<int_t>(half), first_level_end);
    window_capacity = SUPERLU_MAX((int_t) 1, window_capacity);
    lu->symV2CpuWindowMaxWidth = SUPERLU_MAX(
        lu->symV2CpuWindowMaxWidth,
        static_cast<uint64_t>(window_capacity));

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
            for (int_t position = 0; position < first_level_end; ++position)
            {
                symldl_v2_cpu_window_factor_panel(
                    lu, nodes[position], diag_buffers);
                lu->symV2CpuWindowDonePanelSolve[
                    static_cast<size_t>(position)] = 1;
            }

            int_t current_begin = 0;
            int_t current_count = SUPERLU_MIN(
                window_capacity, node_count - current_begin);
            int parity = 0;
            int current_base = slots >= 2 ? 0 : 0;
            for (int_t relative = 0; relative < current_count; ++relative)
            {
                int_t position = current_begin + relative;
                int slot = current_base + static_cast<int>(relative);
                int_t k = nodes[position];
                symldl_v2_cpu_window_exchange(
                    lu, k, etree->setree[k], slot);
                lu->symV2CpuWindowDonePanelBcast[
                    static_cast<size_t>(position)] = 1;
            }

            while (current_begin < node_count)
            {
                if (current_count <= 0)
                    ABORT("SymFact V2 CPU window scheduler made no progress.");
                ++lu->symV2CpuWindowIterations;

                double lookahead_start = SuperLU_timer_();
#pragma omp taskgroup
                {
                    for (int_t relative = 0; relative < current_count;
                         ++relative)
                    {
                        int_t position = current_begin + relative;
                        int slot = current_base + static_cast<int>(relative);
                        xlpanel_t<Ftype> row_panel;
                        xlpanel_t<Ftype> column_panel;
                        symldl_v2_cpu_window_panels(
                            lu, slot, &row_panel, &column_panel);
                        int_t k = nodes[position];
                        symldl_v2_cpu_window_lookahead(
                            lu, k, etree->setree[k], row_panel,
                            column_panel);
                    }
                    lu->symV2CpuWindowLookaheadTime +=
                        SuperLU_timer_() - lookahead_start;
                    double lookahead_wait_start = SuperLU_timer_();
#pragma omp taskwait
                    lu->symV2CpuWindowLookaheadWaitTime +=
                        SuperLU_timer_() - lookahead_wait_start;
                }

                for (int_t relative = 0; relative < current_count;
                     ++relative)
                {
                    int_t position = current_begin + relative;
                    int_t k = nodes[position];
                    int_t parent = etree->setree[k];
                    int_t parent_position =
                        parent >= 0 && parent < lu->nsupers
                            ? inverse[parent]
                            : -1;
                    if (parent_position < 0 || parent_position >= node_count)
                        continue;
                    int &children = lu->symV2CpuWindowChildrenLeft[
                        static_cast<size_t>(parent_position)];
                    if (children <= 0)
                        ABORT("SymFact V2 CPU window child count underflows.");
                    --children;
                    if (children == 0 &&
                        !lu->symV2CpuWindowDonePanelSolve[
                            static_cast<size_t>(parent_position)])
                    {
                        symldl_v2_cpu_window_factor_panel(
                            lu, parent, diag_buffers);
                        lu->symV2CpuWindowDonePanelSolve[
                            static_cast<size_t>(parent_position)] = 1;
                    }
                }

                int_t next_begin = current_begin + current_count;
                int_t next_count = 0;
                while (next_begin + next_count < node_count &&
                       next_count < window_capacity &&
                       lu->symV2CpuWindowChildrenLeft[
                           static_cast<size_t>(next_begin + next_count)] == 0)
                    ++next_count;
                int next_base = slots >= 2 ? ((parity + 1) & 1) * half : 0;

                double exclude_start = SuperLU_timer_();
#pragma omp taskgroup
                {
                    for (int_t relative = 0; relative < current_count;
                         ++relative)
                    {
                        int_t position = current_begin + relative;
                        int slot = current_base + static_cast<int>(relative);
                        xlpanel_t<Ftype> row_panel;
                        xlpanel_t<Ftype> column_panel;
                        symldl_v2_cpu_window_panels(
                            lu, slot, &row_panel, &column_panel);
                        int_t k = nodes[position];
                        symldl_v2_cpu_window_exclude(
                            lu, k, etree->setree[k], row_panel,
                            column_panel);
                    }
                    lu->symV2CpuWindowExcludeTime +=
                        SuperLU_timer_() - exclude_start;

                    if (slots >= 2)
                    {
                        double issue_start = SuperLU_timer_();
                        for (int_t relative = 0; relative < next_count;
                             ++relative)
                        {
                            int_t position = next_begin + relative;
                            int slot = next_base + static_cast<int>(relative);
                            int_t k = nodes[position];
                            symldl_v2_cpu_window_exchange(
                                lu, k, etree->setree[k], slot);
                            lu->symV2CpuWindowDonePanelBcast[
                                static_cast<size_t>(position)] = 1;
                        }
                        lu->symV2CpuWindowNextIssueTime +=
                            SuperLU_timer_() - issue_start;
                    }
                    double exclude_wait_start = SuperLU_timer_();
#pragma omp taskwait
                    lu->symV2CpuWindowExcludeWaitTime +=
                        SuperLU_timer_() - exclude_wait_start;
                }

                for (int_t relative = 0; relative < current_count;
                     ++relative)
                    symldl_v2_cpu_window_release(
                        lu, current_base + static_cast<int>(relative));

                if (slots < 2 && next_count > 0)
                {
                    double issue_start = SuperLU_timer_();
                    int_t k = nodes[next_begin];
                    symldl_v2_cpu_window_exchange(
                        lu, k, etree->setree[k], 0);
                    lu->symV2CpuWindowDonePanelBcast[
                        static_cast<size_t>(next_begin)] = 1;
                    lu->symV2CpuWindowNextIssueTime +=
                        SuperLU_timer_() - issue_start;
                }

                current_begin = next_begin;
                current_count = next_count;
                current_base = next_base;
                ++parity;
            }
        }
    }
    lu->symV2CpuSchedulerTime += SuperLU_timer_() - scheduler_start;

    for (int_t position = 0; position < node_count; ++position)
        if (!lu->symV2CpuWindowDonePanelBcast[
                static_cast<size_t>(position)] ||
            !lu->symV2CpuWindowDonePanelSolve[
                static_cast<size_t>(position)])
            ABORT("SymFact V2 CPU window forest did not complete.");
    for (int slot = 0; slot < slots; ++slot)
        if (lu->symV2CpuWindowStates[static_cast<size_t>(slot)].ready ||
            lu->symV2CpuSlotRequestCounts[static_cast<size_t>(slot)] != 0)
            ABORT("SymFact V2 CPU window slot remains active after forest.");
    return 0;
}
