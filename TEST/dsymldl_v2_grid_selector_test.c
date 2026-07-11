#include "dsymldl_v2_grid_selector.h"
#include "dsymldl_v2_grid_model.h"
#include "dsymldl_v2_grid_report.h"
#include "dsymldl_v2_partition_plan.h"
#include "dsymldl_v2_runtime_model.h"
#include "dsymldl_v2_workspace_size.h"

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static dSymLDLV2GridSelection enumerate(dSymLDLV2GridRequest request,
                                        int ranks)
{
    char error[256];
    dSymLDLV2GridSelection selection;
    dSymLDLV2GridSelectionInit(&selection);
    assert(dSymLDLV2EnumerateGridCandidates(&request, ranks, &selection,
                                            error, sizeof(error)));
    return selection;
}

static void set_metric(dSymLDLV2GridCandidate *candidate,
                       dSymLDLV2RuntimeMetricKind metric,
                       double total, double critical)
{
    candidate->performance.metric[metric].active = 1;
    candidate->performance.metric[metric].total = total;
    candidate->performance.metric[metric].critical = critical;
}

static void test_parse(void)
{
    char error[128];
    int value = -1;

    assert(dSymLDLV2ParseGridDimension("auto", &value, error,
                                      sizeof(error)));
    assert(value == SUPERLU_GRID_AUTO);
    assert(dSymLDLV2ParseGridDimension("16", &value, error,
                                      sizeof(error)));
    assert(value == 16);
    assert(!dSymLDLV2ParseGridDimension("0", &value, error,
                                       sizeof(error)));
    assert(!dSymLDLV2ParseGridDimension("-1", &value, error,
                                       sizeof(error)));
    assert(!dSymLDLV2ParseGridDimension("4x", &value, error,
                                       sizeof(error)));
    assert(!dSymLDLV2ParseGridDimension("AUTO", &value, error,
                                       sizeof(error)));
    assert(!dSymLDLV2ParseGridDimension(" auto", &value, error,
                                       sizeof(error)));
    assert(!dSymLDLV2ParseGridDimension("99999999999999999999", &value,
                                       error, sizeof(error)));
}

static void test_rank_order(void)
{
    assert(dSymLDLV2RuntimeRankForCoordinates(
               DSYMLDL_V2_RANK_ORDER_Z_MAJOR, 2, 3, 2, 0, 0, 0) == 0);
    assert(dSymLDLV2RuntimeRankForCoordinates(
               DSYMLDL_V2_RANK_ORDER_Z_MAJOR, 2, 3, 2, 0, 1, 0) == 1);
    assert(dSymLDLV2RuntimeRankForCoordinates(
               DSYMLDL_V2_RANK_ORDER_Z_MAJOR, 2, 3, 2, 1, 0, 0) == 3);
    assert(dSymLDLV2RuntimeRankForCoordinates(
               DSYMLDL_V2_RANK_ORDER_Z_MAJOR, 2, 3, 2, 0, 0, 1) == 6);
    assert(dSymLDLV2RuntimeRankForCoordinates(
               DSYMLDL_V2_RANK_ORDER_XY_MAJOR, 2, 3, 2, 0, 0, 1) == 1);
    assert(dSymLDLV2RuntimeRankForCoordinates(
               DSYMLDL_V2_RANK_ORDER_XY_MAJOR, 2, 3, 2, 0, 1, 0) == 2);
    assert(dSymLDLV2RuntimeRankForCoordinates(
               DSYMLDL_V2_RANK_ORDER_XY_MAJOR, 2, 3, 2, 1, 0, 0) == 6);
    assert(dSymLDLV2RuntimeRankForCoordinates(
               DSYMLDL_V2_RANK_ORDER_Z_MAJOR, 2, 3, 2, 2, 0, 0) == -1);
}

static void test_all_auto(void)
{
    dSymLDLV2GridRequest request;
    dSymLDLV2GridRequestInit(&request);
    dSymLDLV2GridSelection selection = enumerate(request, 16);

    assert(selection.candidate_count == 15);
    assert(selection.candidates[0].pr == 1);
    assert(selection.candidates[0].pc == 16);
    assert(selection.candidates[0].pz == 1);
    assert(selection.candidates[14].pr == 1);
    assert(selection.candidates[14].pc == 1);
    assert(selection.candidates[14].pz == 16);
    for (size_t i = 0; i < selection.candidate_count; ++i)
    {
        dSymLDLV2GridCandidate *candidate = &selection.candidates[i];
        assert(candidate->pr * candidate->pc * candidate->pz == 16);
        assert((candidate->pz & (candidate->pz - 1)) == 0);
    }
    dSymLDLV2GridSelectionDestroy(&selection);
}

static void test_partial_auto(void)
{
    dSymLDLV2GridRequest request;
    dSymLDLV2GridRequestInit(&request);
    request.pz = 4;
    dSymLDLV2GridSelection selection = enumerate(request, 16);
    assert(selection.candidate_count == 3);
    for (size_t i = 0; i < selection.candidate_count; ++i)
        assert(selection.candidates[i].pz == 4);
    dSymLDLV2GridSelectionDestroy(&selection);

    dSymLDLV2GridRequestInit(&request);
    request.pr = 2;
    selection = enumerate(request, 16);
    assert(selection.candidate_count == 4);
    dSymLDLV2GridSelectionDestroy(&selection);

    dSymLDLV2GridRequestInit(&request);
    request.pr = 2;
    request.pc = 4;
    selection = enumerate(request, 16);
    assert(selection.candidate_count == 1);
    assert(selection.candidates[0].pz == 2);
    dSymLDLV2GridSelectionDestroy(&selection);

    dSymLDLV2GridRequestInit(&request);
    request.pc = 4;
    request.pz = 2;
    selection = enumerate(request, 16);
    assert(selection.candidate_count == 1);
    assert(selection.candidates[0].pr == 2);
    dSymLDLV2GridSelectionDestroy(&selection);

    dSymLDLV2GridRequestInit(&request);
    request.pr = 2;
    request.pz = 2;
    selection = enumerate(request, 16);
    assert(selection.candidate_count == 1);
    assert(selection.candidates[0].pc == 4);
    dSymLDLV2GridSelectionDestroy(&selection);

    dSymLDLV2GridRequestInit(&request);
    request.pr = 2;
    request.pc = 4;
    request.pz = 2;
    selection = enumerate(request, 16);
    assert(selection.candidate_count == 1);
    dSymLDLV2GridSelectionDestroy(&selection);
}

static void test_invalid_requests(void)
{
    char error[256];
    dSymLDLV2GridRequest request;
    dSymLDLV2GridSelection selection;
    dSymLDLV2GridSelectionInit(&selection);

    dSymLDLV2GridRequestInit(&request);
    request.pz = 3;
    assert(!dSymLDLV2EnumerateGridCandidates(&request, 12, &selection,
                                             error, sizeof(error)));

    dSymLDLV2GridRequestInit(&request);
    request.pr = 3;
    request.pc = 3;
    request.pz = 2;
    assert(!dSymLDLV2EnumerateGridCandidates(&request, 16, &selection,
                                             error, sizeof(error)));

    dSymLDLV2GridRequestInit(&request);
    request.max_pr = 1;
    request.max_pc = 1;
    request.max_pz = 4;
    assert(!dSymLDLV2EnumerateGridCandidates(&request, 16, &selection,
                                             error, sizeof(error)));

    dSymLDLV2GridRequestInit(&request);
    request.max_pr = -1;
    assert(!dSymLDLV2EnumerateGridCandidates(&request, 16, &selection,
                                             error, sizeof(error)));
}

static void test_selection(void)
{
    char error[256];
    dSymLDLV2GridRequest request;
    dSymLDLV2GridRequestInit(&request);
    request.pz = 4;
    dSymLDLV2GridSelection selection = enumerate(request, 16);

    set_metric(&selection.candidates[0], DSYMLDL_V2_RUNTIME_GPU_FLOPS,
               10.0, 10.0);
    set_metric(&selection.candidates[0],
               DSYMLDL_V2_RUNTIME_INTER_NODE_BYTES, 100.0, 100.0);
    set_metric(&selection.candidates[1], DSYMLDL_V2_RUNTIME_GPU_FLOPS,
               100.0, 100.0);
    set_metric(&selection.candidates[1],
               DSYMLDL_V2_RUNTIME_INTER_NODE_BYTES, 10.0, 10.0);
    set_metric(&selection.candidates[2], DSYMLDL_V2_RUNTIME_GPU_FLOPS,
               40.0, 40.0);
    set_metric(&selection.candidates[2],
               DSYMLDL_V2_RUNTIME_INTER_NODE_BYTES, 40.0, 40.0);

    assert(dSymLDLV2SelectGrid(&selection, error, sizeof(error)));
    assert(selection.selected_index == 2);
    assert(selection.pareto_count == 3);
    assert(selection.confidence ==
           DSYMLDL_V2_GRID_CONFIDENCE_ROBUST_COMPROMISE);

    memset(&selection.candidates[0].performance, 0,
           sizeof(selection.candidates[0].performance));
    memset(&selection.candidates[1].performance, 0,
           sizeof(selection.candidates[1].performance));
    selection.candidates[2].status = DSYMLDL_V2_GRID_REJECT_GPU_MEMORY;
    set_metric(&selection.candidates[0], DSYMLDL_V2_RUNTIME_CPU_FLOPS,
               1.0, 1.0);
    set_metric(&selection.candidates[1], DSYMLDL_V2_RUNTIME_CPU_FLOPS,
               1.0, 1.0);
    selection.candidates[0].memory.host_high_water_per_node = 20;
    selection.candidates[1].memory.host_high_water_per_node = 30;
    assert(dSymLDLV2SelectGrid(&selection, error, sizeof(error)));
    assert(selection.selected_index == 0);
    assert(selection.pareto_count == 2);
    assert(selection.confidence ==
           DSYMLDL_V2_GRID_CONFIDENCE_AMBIGUOUS_TIE);

    set_metric(&selection.candidates[1], DSYMLDL_V2_RUNTIME_CPU_FLOPS,
               2.0, 2.0);
    assert(dSymLDLV2SelectGrid(&selection, error, sizeof(error)));
    assert(selection.selected_index == 0);
    assert(selection.pareto_count == 1);
    assert(selection.confidence == DSYMLDL_V2_GRID_CONFIDENCE_DOMINANT);
    dSymLDLV2GridSelectionDestroy(&selection);
}

static void test_partition_plan(void)
{
    const int_t nsupers = 7;
    int_t setree[7] = {2, 2, 6, 5, 5, 6, 7};
    int_t xsup[8] = {0, 2, 5, 7, 11, 14, 18, 22};
    int_t panel_rows[7] = {12, 8, 18, 9, 15, 20, 24};
    treeList_t *tree_list = setree2list(nsupers, setree);
    dSymLDLV2PartitionPlanInput input = {
        nsupers, setree, xsup, panel_rows, tree_list, 0.0
    };
    dSymLDLV2PartitionPlan first = {0};
    dSymLDLV2PartitionPlan second = {0};
    char error[256];

    assert(tree_list != NULL);
    assert(dSymLDLV2BuildPartitionPlan(&input, 2, 2, 2, &first,
                                       error, sizeof(error)));
    assert(dSymLDLV2BuildPartitionPlan(&input, 2, 2, 2, &second,
                                       error, sizeof(error)));
    assert(first.nsupers == nsupers);
    assert(first.max_z_levels == 2);
    assert(first.forest_count == 3);
    assert(first.factor_level_count > 0);
    assert(first.factor_level_ptr[first.factor_level_count] == nsupers);
    assert(memcmp(first.diag_root, second.diag_root,
                  (size_t) nsupers * sizeof(int)) == 0);
    assert(memcmp(first.panel_root, second.panel_root,
                  (size_t) nsupers * sizeof(int)) == 0);

    int seen[7] = {0};
    for (int_t tree = 0; tree < first.forest_count; ++tree)
    {
        sForest_t *forest = first.forests[tree];
        if (forest == NULL)
            continue;
        for (int_t i = 0; i < forest->nNodes; ++i)
        {
            assert(forest->nodeList[i] >= 0);
            assert(forest->nodeList[i] < nsupers);
            ++seen[forest->nodeList[i]];
        }
    }
    for (int_t k = 0; k < nsupers; ++k)
    {
        assert(seen[k] == 1);
        assert(first.diag_root[k] >= 0 && first.diag_root[k] < first.pr);
        assert(first.panel_root[k] >= 0 && first.panel_root[k] < first.pc);
        assert(first.forest_of_supernode[k] >= 0);
        assert(first.forest_of_supernode[k] < first.forest_count);
    }
    assert(first.maximum_rank_load > 0.0);
    assert(first.maximum_level_rank_load > 0.0);

    dSymLDLV2PartitionPlanDestroy(&first);
    dSymLDLV2PartitionPlanDestroy(&second);
    free_treelist(nsupers, tree_list);
}

static void test_invalid_structure(void)
{
    const int_t nsupers = 2;
    int_t setree[2] = {1, 2};
    int_t xsup[3] = {0, 2, 4};
    int_t panel_rows[2] = {4, 2};
    int_t supno[4] = {0, 0, 1, 1};
    int_t xlsub[5] = {0, 4, 4, 6, 6};
    int_t lsub[6] = {0, 1, 2, 3, 2, 3};
    Glu_persist_t persist;
    Glu_freeable_t symbolic;
    dSymLDLV2StructuralSummary structure = {0};
    dSymLDLV2PartitionPlan plan = {0};
    char error[256];

    memset(&persist, 0, sizeof(persist));
    memset(&symbolic, 0, sizeof(symbolic));
    persist.xsup = xsup;
    persist.supno = supno;
    symbolic.xlsub = xlsub;
    symbolic.lsub = lsub;
    assert(dSymLDLV2BuildStructuralSummary(
        nsupers, &persist, &symbolic, &structure, error, sizeof(error)));
    dSymLDLV2StructuralSummaryDestroy(&structure);

    lsub[0] = 2;
    lsub[1] = 3;
    lsub[2] = 0;
    lsub[3] = 1;
    assert(dSymLDLV2BuildStructuralSummary(
        nsupers, &persist, &symbolic, &structure, error, sizeof(error)));
    assert(structure.panel_block_offsets[0] == 0);
    assert(structure.panel_block_offsets[1] == 2);
    assert(structure.block_row_supernode[0] == 0);
    assert(structure.block_row_supernode[1] == 1);
    assert(structure.block_row_count[0] == 2);
    assert(structure.block_row_count[1] == 2);
    dSymLDLV2StructuralSummaryDestroy(&structure);
    lsub[0] = 0;
    lsub[1] = 1;
    lsub[2] = 2;
    lsub[3] = 3;

    treeList_t *tree_list = setree2list(nsupers, setree);
    dSymLDLV2PartitionPlanInput input = {
        nsupers, setree, xsup, panel_rows, tree_list, NAN
    };
    assert(!dSymLDLV2BuildPartitionPlan(&input, 1, 1, 1, &plan,
                                        error, sizeof(error)));
    input.owner_affinity_weight = 0.0;
    setree[0] = -1;
    assert(!dSymLDLV2BuildPartitionPlan(&input, 1, 1, 1, &plan,
                                        error, sizeof(error)));
    setree[0] = 1;
    free_treelist(nsupers, tree_list);
}

static void test_grid_model(void)
{
    const int_t nsupers = 3;
    int_t setree[3] = {1, 2, 3};
    int_t xsup[4] = {0, 2, 4, 6};
    int_t supno[6] = {0, 0, 1, 1, 2, 2};
    int_t xlsub[7] = {0, 6, 6, 10, 10, 12, 12};
    int_t lsub[12] = {0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 4, 5};
    Glu_persist_t persist;
    Glu_freeable_t symbolic;
    dSymLDLV2StructuralSummary structure = {0};
    dSymLDLV2PartitionPlan plan = {0};
    dSymLDLV2GridCandidate candidate = {0};
    char error[256];

    memset(&persist, 0, sizeof(persist));
    memset(&symbolic, 0, sizeof(symbolic));
    candidate.pr = 2;
    candidate.pc = 2;
    candidate.pz = 1;
    persist.xsup = xsup;
    persist.supno = supno;
    symbolic.xlsub = xlsub;
    symbolic.lsub = lsub;
    assert(dSymLDLV2BuildStructuralSummary(
        nsupers, &persist, &symbolic, &structure, error, sizeof(error)));
    assert(structure.block_count == 6);
    assert(structure.panel_rows[0] == 6);
    assert(structure.panel_rows[1] == 4);
    assert(structure.panel_rows[2] == 2);
    assert(structure.symbolic_row_entries == 12);
    assert(structure.factor_value_entries == 24);
    structure.input_nnz = 12;

    treeList_t *tree_list = setree2list(nsupers, setree);
    dSymLDLV2PartitionPlanInput partition_input = {
        nsupers, setree, xsup, structure.panel_rows, tree_list, 0.0
    };
    assert(dSymLDLV2BuildPartitionPlan(&partition_input, 2, 2, 1, &plan,
                                       error, sizeof(error)));
    dSymLDLV2GridModelInput model;
    memset(&model, 0, sizeof(model));
    model.partition_input = &partition_input;
    model.structure = &structure;
    model.available.ranks_per_node = 4;
    model.runtime.gpu_offload = 1;
    model.runtime.gpu_offload = 1;
    model.runtime.gpu_streams = 4;
    model.runtime.lookahead_depth = 2;
    model.runtime.gemm_buffer_values = 128;
    model.runtime.max_supernode_size = 4;
    model.runtime.pinned_staging = 1;
    model.runtime.pc_fragment_schur = 1;
    model.runtime.pc_fragment_ldl_native = 1;
    assert(dSymLDLV2EvaluateGridCandidate(&model, &plan, &candidate,
                                          error, sizeof(error)));
    assert(candidate.status == DSYMLDL_V2_GRID_FEASIBLE);
    assert(candidate.memory.host_persistent_factor > 0);
    assert(candidate.memory.host_symbolic_workspace_high_water > 0);
    assert(candidate.memory.host_distribution_workspace_high_water >=
           candidate.memory.host_symbolic_workspace_high_water);
    assert(candidate.memory.host_high_water_per_rank >=
           candidate.memory.host_distribution_workspace_high_water);
    assert(candidate.memory.gpu_high_water_per_rank > 0);
    assert(candidate.memory.gpu_runtime_overhead >=
           (uint64_t) 256 * 1024 * 1024);
    assert(candidate.memory.host_communication_staging >=
           candidate.memory.pinned_staging);
    assert(candidate.performance.total_factor_flops > 0.0);
    assert(candidate.performance.metric[DSYMLDL_V2_RUNTIME_GPU_FLOPS]
               .active);
    assert(candidate.performance.metric[DSYMLDL_V2_RUNTIME_GPU_FLOPS]
               .critical > 0.0);
    assert(candidate.performance.metric[DSYMLDL_V2_RUNTIME_CPU_FLOPS]
               .critical > 0.0);
    assert(candidate.performance.estimated_gpu_streams == 4);

    uint64_t stream_workspace =
        candidate.memory.gpu_factor_workspace_high_water +
        candidate.memory.gpu_communication_staging;
    uint64_t nonstream_workspace =
        candidate.memory.gpu_high_water_per_rank - stream_workspace;
    assert(stream_workspace > 0 && stream_workspace % 4 == 0);
    uint64_t per_stream_workspace = stream_workspace / 4;

    model.available.gpu_memory_known = 1;
    model.available.available_gpu_bytes =
        nonstream_workspace + 2 * per_stream_workspace;
    assert(dSymLDLV2EvaluateGridCandidate(&model, &plan, &candidate,
                                          error, sizeof(error)));
    assert(candidate.status == DSYMLDL_V2_GRID_FEASIBLE);
    assert(candidate.performance.estimated_gpu_streams == 2);

    model.available.available_gpu_bytes =
        nonstream_workspace + 4 * per_stream_workspace;
    assert(dSymLDLV2EvaluateGridCandidate(&model, &plan, &candidate,
                                          error, sizeof(error)));
    assert(candidate.status == DSYMLDL_V2_GRID_FEASIBLE);

    model.available.available_gpu_bytes =
        nonstream_workspace + per_stream_workspace;
    assert(dSymLDLV2EvaluateGridCandidate(&model, &plan, &candidate,
                                          error, sizeof(error)));
    assert(candidate.status == DSYMLDL_V2_GRID_REJECT_GPU_MEMORY);

    model.available.available_gpu_bytes = 1;
    assert(dSymLDLV2EvaluateGridCandidate(&model, &plan, &candidate,
                                          error, sizeof(error)));
    assert(candidate.status == DSYMLDL_V2_GRID_REJECT_GPU_MEMORY);

    model.runtime.pc_fragment_ldl_native = 0;
    assert(dSymLDLV2EvaluateGridCandidate(&model, &plan, &candidate,
                                          error, sizeof(error)));
    assert(candidate.status == DSYMLDL_V2_GRID_REJECT_IMPLEMENTATION);
    model.runtime.pc_fragment_ldl_native = 1;

    model.runtime.cuda_aware_mpi = 1;
    assert(dSymLDLV2EvaluateGridCandidate(&model, &plan, &candidate,
                                          error, sizeof(error)));
    assert(candidate.status == DSYMLDL_V2_GRID_REJECT_IMPLEMENTATION);
    model.runtime.cuda_aware_mpi = 0;

    model.runtime.gpu_offload = 0;
    assert(dSymLDLV2EvaluateGridCandidate(&model, &plan, &candidate,
                                          error, sizeof(error)));
    assert(candidate.status == DSYMLDL_V2_GRID_FEASIBLE);
    assert(!candidate.performance.metric[DSYMLDL_V2_RUNTIME_GPU_FLOPS]
                .active);
    assert(!candidate.performance.metric[DSYMLDL_V2_RUNTIME_TASK_LAUNCHES]
                .active);
    assert(!candidate.performance.metric[
                DSYMLDL_V2_RUNTIME_HOST_STAGING_BYTES].active);
    assert(candidate.performance.metric[DSYMLDL_V2_RUNTIME_CPU_FLOPS]
               .active);
    model.runtime.gpu_offload = 1;

    /* Exercise capacity-driven stream selection at common GPU sizes without
       allocating the modeled workspace. */
    structure.factor_value_entries = UINT64_C(1) << 30;
    model.runtime.gpu_streams = 8;
    model.runtime.gemm_buffer_values = (size_t) UINT64_C(1) << 30;
    model.available.available_gpu_bytes = UINT64_C(40) << 30;
    assert(dSymLDLV2EvaluateGridCandidate(&model, &plan, &candidate,
                                          error, sizeof(error)));
    assert(candidate.status == DSYMLDL_V2_GRID_FEASIBLE);
    assert(candidate.performance.estimated_gpu_streams == 4);
    assert(candidate.memory.gpu_high_water_per_rank <=
           model.available.available_gpu_bytes);

    model.available.available_gpu_bytes = UINT64_C(80) << 30;
    assert(dSymLDLV2EvaluateGridCandidate(&model, &plan, &candidate,
                                          error, sizeof(error)));
    assert(candidate.status == DSYMLDL_V2_GRID_FEASIBLE);
    assert(candidate.performance.estimated_gpu_streams == 8);
    assert(candidate.memory.gpu_high_water_per_rank <=
           model.available.available_gpu_bytes);

    model.available.available_gpu_bytes = UINT64_C(100) << 30;
    assert(dSymLDLV2EvaluateGridCandidate(&model, &plan, &candidate,
                                          error, sizeof(error)));
    assert(candidate.status == DSYMLDL_V2_GRID_FEASIBLE);
    assert(candidate.performance.estimated_gpu_streams == 8);
    assert(candidate.memory.gpu_high_water_per_rank <=
           model.available.available_gpu_bytes);

    structure.factor_value_entries = 24;
    model.runtime.gpu_streams = 4;
    model.runtime.gemm_buffer_values = 128;

    structure.block_row_supernode[0] = -1;
    assert(!dSymLDLV2EvaluateGridCandidate(&model, &plan, &candidate,
                                           error, sizeof(error)));
    structure.block_row_supernode[0] = 0;

    dSymLDLV2PartitionPlanDestroy(&plan);
    dSymLDLV2StructuralSummaryDestroy(&structure);
    free_treelist(nsupers, tree_list);
}

static void test_workspace_sizing(void)
{
    dSymLDLV2StreamWorkspaceCounts counts;
    size_t bytes = 0;
    memset(&counts, 0, sizeof(counts));
    counts.l_value_count = 1;
    counts.u_value_count = 1;
    counts.partner_value_count = 1;
    counts.partner_stage_count = 1;
    counts.partner_send_stage_count = 1;
    counts.row_stage_count = 1;
    counts.row_receive_value_count = 1;
    counts.raw_panel_count = 1;
    counts.l_index_count = 1;
    counts.u_index_count = 1;
    counts.partner_index_count = 1;
    counts.row_receive_index_count = 1;
    counts.row_send_map_count = 1;
    counts.diagonal_work_count = 1;
    counts.lookahead_l_count = 1;
    counts.lookahead_row_count = 1;
    counts.pc_fragment_schur = 1;
    counts.need_diagonal_work = 1;
    assert(dSymLDLV2StreamWorkspaceBytes(
        &counts, sizeof(double), sizeof(int_t), sizeof(int), &bytes));
    assert(bytes == 17 * DSYMLDL_V2_GPU_ARENA_ALIGNMENT);
    assert(dSymLDLV2GemmWorkspaceBytes(
        1, 1, sizeof(double), &bytes));
    assert(bytes == 2 * DSYMLDL_V2_GPU_ARENA_ALIGNMENT);
    assert(!dSymLDLV2GemmWorkspaceBytes(
        SIZE_MAX, 1, sizeof(double), &bytes));

    counts.l_value_count = SIZE_MAX;
    assert(!dSymLDLV2StreamWorkspaceBytes(
        &counts, sizeof(double), sizeof(int_t), sizeof(int), &bytes));
}

static void test_runtime_stream_cap(void)
{
    dSymLDLV2MemoryEstimate memory = {0};

    dSymLDLV2SetCurrentGridPrediction(NULL);
    assert(dSymLDLV2GetCurrentGridGPUStreamCap() == 0);
    assert(dSymLDLV2GetCurrentGridGPUStreamsUsed() == 0);

    dSymLDLV2SetCurrentGridGPUStreamCap(4);
    dSymLDLV2SetCurrentGridGPUStreamsUsed(3);
    assert(dSymLDLV2GetCurrentGridGPUStreamCap() == 4);
    assert(dSymLDLV2GetCurrentGridGPUStreamsUsed() == 3);

    dSymLDLV2SetCurrentGridPrediction(&memory);
    assert(dSymLDLV2GetCurrentGridGPUStreamCap() == 4);
    assert(dSymLDLV2GetCurrentGridGPUStreamsUsed() == 0);

    dSymLDLV2SetCurrentGridPrediction(NULL);
    assert(dSymLDLV2GetCurrentGridGPUStreamCap() == 0);

    dSymLDLV2SetCurrentGridGPUStreamCap(-1);
    dSymLDLV2SetCurrentGridGPUStreamsUsed(-1);
    assert(dSymLDLV2GetCurrentGridGPUStreamCap() == 0);
    assert(dSymLDLV2GetCurrentGridGPUStreamsUsed() == 0);
}

static void test_performance_shape(void)
{
    const int_t nsupers = 7;
    int_t setree[7] = {2, 2, 6, 5, 5, 6, 7};
    int_t xsup[8] = {0, 2, 5, 7, 11, 14, 18, 22};
    int_t panel_rows[7];
    int_t panel_offsets[8];
    int_t block_gids[28];
    int_t block_rows[28];
    dSymLDLV2StructuralSummary structure;
    dSymLDLV2GridModelInput model;
    dSymLDLV2GridRequest request;
    char error[256];
    int_t block = 0;

    memset(&structure, 0, sizeof(structure));
    for (int_t k = 0; k < nsupers; ++k)
    {
        panel_offsets[k] = block;
        panel_rows[k] = 0;
        for (int_t row = k; row < nsupers; ++row)
        {
            int_t rows = xsup[row + 1] - xsup[row];
            block_gids[block] = row;
            block_rows[block++] = rows;
            panel_rows[k] += rows;
        }
        structure.symbolic_row_entries += (uint64_t) panel_rows[k];
        structure.factor_value_entries +=
            (uint64_t) panel_rows[k] * (uint64_t) (xsup[k + 1] - xsup[k]);
    }
    panel_offsets[nsupers] = block;
    assert(block == 28);
    structure.nsupers = nsupers;
    structure.n = xsup[nsupers];
    structure.input_nnz = structure.factor_value_entries;
    structure.block_count = block;
    structure.panel_block_offsets = panel_offsets;
    structure.block_row_supernode = block_gids;
    structure.block_row_count = block_rows;
    structure.panel_rows = panel_rows;

    treeList_t *tree_list = setree2list(nsupers, setree);
    dSymLDLV2PartitionPlanInput partition_input = {
        nsupers, setree, xsup, panel_rows, tree_list, 0.0
    };
    dSymLDLV2GridRequestInit(&request);
    dSymLDLV2GridSelection selection = enumerate(request, 8);
    memset(&model, 0, sizeof(model));
    model.partition_input = &partition_input;
    model.structure = &structure;
    model.available.ranks_per_node = 4;
    model.runtime.gpu_streams = 8;
    model.runtime.lookahead_depth = 8;
    model.runtime.solve_nrhs = 1;
    model.runtime.gemm_buffer_values = 128;
    model.runtime.max_supernode_size = 4;
    model.runtime.pinned_staging = 1;
    model.runtime.pooled_pinned_staging = 1;
    model.runtime.pc_fragment_schur = 1;
    model.runtime.pc_fragment_ldl_native = 1;
    int saw_pr1_pc_candidate = 0;

    for (size_t i = 0; i < selection.candidate_count; ++i)
    {
        dSymLDLV2PartitionPlan plan = {0};
        assert(dSymLDLV2BuildPartitionPlan(
            &partition_input, selection.candidates[i].pr,
            selection.candidates[i].pc, selection.candidates[i].pz,
            &plan, error, sizeof(error)));
        assert(dSymLDLV2EvaluateGridCandidate(
            &model, &plan, &selection.candidates[i], error, sizeof(error)));
        if (selection.candidates[i].pr == 1 &&
            selection.candidates[i].pc > 1)
        {
            assert(selection.candidates[i]
                       .performance
                       .metric[DSYMLDL_V2_RUNTIME_INTER_NODE_BYTES]
                       .total > 0.0);
            saw_pr1_pc_candidate = 1;
        }
        dSymLDLV2PartitionPlanDestroy(&plan);
    }
    assert(saw_pr1_pc_candidate);
    assert(dSymLDLV2SelectGrid(&selection, error, sizeof(error)));
    const dSymLDLV2GridCandidate *selected =
        &selection.candidates[selection.selected_index];
    assert(selected->status == DSYMLDL_V2_GRID_FEASIBLE);
    assert(!selected->performance.pareto_dominated);
    assert(selection.pareto_count > 0);

    dSymLDLV2PartitionPlan topology_plan = {0};
    assert(dSymLDLV2BuildPartitionPlan(
        &partition_input, 2, 2, 2, &topology_plan,
        error, sizeof(error)));
    int node_of_rank[8] = {0, 0, 0, 0, 1, 1, 1, 1};
    dSymLDLV2GridTopology topology = {
        8, 2, 1, DSYMLDL_V2_RANK_ORDER_Z_MAJOR, node_of_rank
    };
    dSymLDLV2PerformanceEstimate z_order;
    dSymLDLV2PerformanceEstimate xy_order;
    dSymLDLV2PerformanceEstimate one_stream;
    dSymLDLV2PerformanceEstimate eight_streams;
    dSymLDLV2PerformanceEstimate one_layer;
    model.topology = &topology;
    model.runtime.rank_order = DSYMLDL_V2_RANK_ORDER_Z_MAJOR;
    assert(dSymLDLV2EvaluateRuntimeModel(
        &model, &topology_plan, 8, 8, &z_order,
        error, sizeof(error)));
    topology.rank_order = DSYMLDL_V2_RANK_ORDER_XY_MAJOR;
    model.runtime.rank_order = DSYMLDL_V2_RANK_ORDER_XY_MAJOR;
    assert(dSymLDLV2EvaluateRuntimeModel(
        &model, &topology_plan, 8, 8, &xy_order,
        error, sizeof(error)));
    assert(z_order.metric[DSYMLDL_V2_RUNTIME_INTER_NODE_BYTES].total !=
               xy_order.metric[DSYMLDL_V2_RUNTIME_INTER_NODE_BYTES].total ||
           z_order.metric[DSYMLDL_V2_RUNTIME_INTRA_NODE_BYTES].total !=
               xy_order.metric[DSYMLDL_V2_RUNTIME_INTRA_NODE_BYTES].total);

    topology.rank_order = DSYMLDL_V2_RANK_ORDER_Z_MAJOR;
    model.runtime.rank_order = DSYMLDL_V2_RANK_ORDER_Z_MAJOR;
    assert(dSymLDLV2EvaluateRuntimeModel(
        &model, &topology_plan, 8, 1, &one_stream,
        error, sizeof(error)));
    assert(dSymLDLV2EvaluateRuntimeModel(
        &model, &topology_plan, 8, 8, &eight_streams,
        error, sizeof(error)));
    assert(one_stream.metric[DSYMLDL_V2_RUNTIME_GPU_FLOPS].critical >=
           eight_streams.metric[DSYMLDL_V2_RUNTIME_GPU_FLOPS].critical);

    dSymLDLV2PartitionPlan one_layer_plan = {0};
    assert(dSymLDLV2BuildPartitionPlan(
        &partition_input, 4, 2, 1, &one_layer_plan,
        error, sizeof(error)));
    assert(dSymLDLV2EvaluateRuntimeModel(
        &model, &one_layer_plan, 8, 8, &one_layer,
        error, sizeof(error)));
    assert(fabs(one_layer.total_factor_flops -
                eight_streams.total_factor_flops) <=
           1.0e-12 * one_layer.total_factor_flops);
    assert(eight_streams.metric[DSYMLDL_V2_RUNTIME_INTER_NODE_BYTES].total >
           0.0);
    dSymLDLV2PartitionPlanDestroy(&one_layer_plan);
    dSymLDLV2PartitionPlanDestroy(&topology_plan);

    dSymLDLV2PartitionPlan plan_2x3 = {0};
    dSymLDLV2PartitionPlan plan_3x2 = {0};
    int six_rank_nodes[6] = {0, 0, 0, 1, 1, 1};
    dSymLDLV2GridTopology six_rank_topology = {
        6, 2, 1, DSYMLDL_V2_RANK_ORDER_Z_MAJOR, six_rank_nodes
    };
    dSymLDLV2PerformanceEstimate metrics_2x3;
    dSymLDLV2PerformanceEstimate metrics_3x2;
    assert(dSymLDLV2BuildPartitionPlan(
        &partition_input, 2, 3, 1, &plan_2x3,
        error, sizeof(error)));
    assert(dSymLDLV2BuildPartitionPlan(
        &partition_input, 3, 2, 1, &plan_3x2,
        error, sizeof(error)));
    model.topology = &six_rank_topology;
    assert(dSymLDLV2EvaluateRuntimeModel(
        &model, &plan_2x3, 8, 8, &metrics_2x3,
        error, sizeof(error)));
    assert(dSymLDLV2EvaluateRuntimeModel(
        &model, &plan_3x2, 8, 8, &metrics_3x2,
        error, sizeof(error)));
    int shape_differs = 0;
    for (int metric = 0; metric < DSYMLDL_V2_RUNTIME_METRIC_COUNT; ++metric)
        if (metrics_2x3.metric[metric].critical !=
                metrics_3x2.metric[metric].critical ||
            metrics_2x3.metric[metric].total !=
                metrics_3x2.metric[metric].total)
            shape_differs = 1;
    assert(shape_differs);
    dSymLDLV2PartitionPlanDestroy(&plan_2x3);
    dSymLDLV2PartitionPlanDestroy(&plan_3x2);

    dSymLDLV2GridSelectionDestroy(&selection);
    free_treelist(nsupers, tree_list);
}

int main(void)
{
    test_parse();
    test_rank_order();
    test_all_auto();
    test_partial_auto();
    test_invalid_requests();
    test_selection();
    test_partition_plan();
    test_invalid_structure();
    test_grid_model();
    test_workspace_sizing();
    test_runtime_stream_cap();
    test_performance_shape();
    puts("SymLDL grid selector tests passed");
    return 0;
}
