/*! @file
 * \brief MPI smoke for the retained-L true-symmetric CPU solve runtime.
 */

#include "dsymldl_v2_cpu_solve.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { TEST_NSUPERS = 12, TEST_NRHS_CAPACITY = 3 };

static void *
test_alloc(size_t count, size_t size)
{
    void *ptr = calloc(count != 0 ? count : 1, size);
    if (ptr == NULL)
        ABORT("Malloc fails for the SymLDL CPU solve test.");
    return ptr;
}

static double
test_l_value(int_t row, int_t col)
{
    if (row <= col)
        return 0.0;
    if (col < 8) {
        if ((col & 1) == 0 && row == col + 1)
            return 0.035 + 0.002 * (double) (col / 2);
        if (row == (col < 4 ? 8 : 9))
            return -0.018 + 0.001 * (double) (col % 4);
    }
    if ((col == 8 || col == 9) && row == 10)
        return 0.027 - 0.003 * (double) (col - 8);
    if (col == 10 && row == 11)
        return -0.021;
    return 0.0;
}

static double
test_d_value(int_t gid)
{
    return 1.25 + 0.05 * (double) gid;
}

static int
test_root(int_t gid, int count, uint32_t seed)
{
    uint32_t value = (uint32_t) gid + seed;
    value ^= value >> 16;
    value *= UINT32_C(0x7feb352d);
    value ^= value >> 15;
    value *= UINT32_C(0x846ca68b);
    value ^= value >> 16;
    return (int) (value % (uint32_t) count);
}

static int
test_forest_for_gid(int_t gid, int npdep)
{
    if (npdep == 1)
        return 0;
    if (npdep == 2) {
        if (gid >= 10)
            return 0;
        return gid < 4 || gid == 8 ? 1 : 2;
    }
    if (npdep == 4) {
        if (gid >= 10)
            return 0;
        if (gid == 8)
            return 1;
        if (gid == 9)
            return 2;
        return 3 + (int) (gid / 2);
    }
    ABORT("SymLDL CPU test supports only one, two, or four Z layers.");
    return -1;
}

static void
test_forests_create(dtrf3Dpartition_t *partition, gridinfo3d_t *grid3d)
{
    const int npdep = grid3d->zscp.Np;
    const int max_level = (int) log2i(npdep) + 1;
    const int forest_count = (1 << max_level) - 1;
    int_t *counts = (int_t *) test_alloc((size_t) forest_count, sizeof(int_t));
    int_t *write = (int_t *) test_alloc((size_t) forest_count, sizeof(int_t));

    partition->sForests = (sForest_t **)
        test_alloc((size_t) forest_count, sizeof(*partition->sForests));
    for (int_t gid = 0; gid < TEST_NSUPERS; ++gid)
        ++counts[test_forest_for_gid(gid, npdep)];
    for (int forest = 0; forest < forest_count; ++forest) {
        sForest_t *entry = (sForest_t *) test_alloc(1, sizeof(*entry));
        entry->nNodes = counts[forest];
        entry->nodeList = (int_t *)
            test_alloc((size_t) counts[forest], sizeof(int_t));
        partition->sForests[forest] = entry;
    }
    for (int_t gid = 0; gid < TEST_NSUPERS; ++gid) {
        int forest = test_forest_for_gid(gid, npdep);
        partition->sForests[forest]->nodeList[write[forest]++] = gid;
    }
    partition->myTreeIdxs = getGridTrees(grid3d);
    partition->myZeroTrIdxs = getReplicatedTrees(grid3d);
    if (partition->myTreeIdxs == NULL || partition->myZeroTrIdxs == NULL)
        ABORT("SymLDL CPU test forest metadata allocation failed.");

    for (int_t gid = 0; gid < TEST_NSUPERS; ++gid)
        partition->superGridMap[gid] = NOT_IN_GRID;
    for (int level = 0; level < max_level; ++level) {
        int forest = (int) partition->myTreeIdxs[level];
        SupernodeToGridMap_t map = partition->myZeroTrIdxs[level]
                                           ? IN_GRID_ZERO : IN_GRID_AIJ;
        if (forest < 0 || forest >= forest_count)
            ABORT("SymLDL CPU test has an invalid local forest.");
        for (int_t pos = 0; pos < partition->sForests[forest]->nNodes; ++pos)
            partition->superGridMap[
                partition->sForests[forest]->nodeList[pos]] = map;
    }
    free(write);
    free(counts);
}

static void
test_rhs(int nrhs, double *rhs, double *post_forward,
         double *post_diagonal, double *expected)
{
    memset(post_forward, 0,
           sizeof(double) * TEST_NSUPERS * TEST_NRHS_CAPACITY);
    memset(post_diagonal, 0,
           sizeof(double) * TEST_NSUPERS * TEST_NRHS_CAPACITY);
    memset(rhs, 0, sizeof(double) * TEST_NSUPERS * nrhs);

    for (int rhs_id = 0; rhs_id < nrhs; ++rhs_id)
        for (int_t gid = 0; gid < TEST_NSUPERS; ++gid)
            expected[gid + (int_t) rhs_id * TEST_NSUPERS] =
                0.75 + 0.125 * (double) gid + 0.03125 * rhs_id;

    for (int rhs_id = 0; rhs_id < nrhs; ++rhs_id) {
        for (int_t col = 0; col < TEST_NSUPERS; ++col) {
            double value = expected[col + (int_t) rhs_id * TEST_NSUPERS];
            for (int_t row = col + 1; row < TEST_NSUPERS; ++row)
                value += test_l_value(row, col) *
                         expected[row + (int_t) rhs_id * TEST_NSUPERS];
            post_diagonal[col + (int_t) rhs_id * TEST_NSUPERS] = value;
            post_forward[col + (int_t) rhs_id * TEST_NSUPERS] =
                test_d_value(col) * value;
        }
        for (int_t row = 0; row < TEST_NSUPERS; ++row) {
            double value = post_forward[
                row + (int_t) rhs_id * TEST_NSUPERS];
            for (int_t col = 0; col < row; ++col)
                value += test_l_value(row, col) *
                         post_forward[col +
                                      (int_t) rhs_id * TEST_NSUPERS];
            rhs[row + (int_t) rhs_id * TEST_NSUPERS] = value;
        }
    }
}

static dSymLDLSolveGraph *
test_graph_create(gridinfo3d_t *grid3d, dtrf3Dpartition_t *partition,
                  double ***panel_values_out)
{
    gridinfo_t *grid = &grid3d->grid2d;
    int myrow = MYROW(grid->iam, grid);
    int mycol = MYCOL(grid->iam, grid);
    int_t xsup[TEST_NSUPERS + 1];
    int_t *ilsum;
    int_t panel_count = 0;
    int_t row_count = 0;
    int_t block_count = 0;
    int_t edge = 0;
    int_t row_entry = 0;
    int *local_owner;
    int *local_owner_count;
    int *owner_count;
    dSymLDLPanelDesc *panels;
    dSymLDLBlockDesc *blocks;
    int_t *rows;
    double **panel_values;

    memset(partition, 0, sizeof(*partition));
    partition->nsupers = TEST_NSUPERS;
    partition->symV2DiagRoot = (int *) test_alloc(TEST_NSUPERS, sizeof(int));
    partition->symV2PanelRoot = (int *) test_alloc(TEST_NSUPERS, sizeof(int));
    partition->symV2DiagOwner = (int *) test_alloc(TEST_NSUPERS, sizeof(int));
    partition->symV2PanelLocalIndex =
        (int_t *) test_alloc(TEST_NSUPERS, sizeof(int_t));
    partition->symV2RowLocalIndex =
        (int_t *) test_alloc(TEST_NSUPERS, sizeof(int_t));
    partition->superGridMap = (SupernodeToGridMap_t *)
        test_alloc(TEST_NSUPERS, sizeof(SupernodeToGridMap_t));
    test_forests_create(partition, grid3d);

    for (int_t gid = 0; gid <= TEST_NSUPERS; ++gid)
        xsup[gid] = gid;
    for (int_t gid = 0; gid < TEST_NSUPERS; ++gid) {
        partition->symV2DiagRoot[gid] =
            test_root(gid, grid->nprow, UINT32_C(0x91e10da5));
        partition->symV2PanelRoot[gid] =
            test_root(gid, grid->npcol, UINT32_C(0x243f6a88));
        partition->symV2PanelLocalIndex[gid] = -1;
        partition->symV2RowLocalIndex[gid] = -1;
    }
    local_owner = (int *) test_alloc(TEST_NSUPERS, sizeof(int));
    local_owner_count = (int *) test_alloc(TEST_NSUPERS, sizeof(int));
    owner_count = (int *) test_alloc(TEST_NSUPERS, sizeof(int));
    for (int_t gid = 0; gid < TEST_NSUPERS; ++gid) {
        int owns = partition->superGridMap[gid] == IN_GRID_AIJ &&
                   grid->iam == PNUM(partition->symV2DiagRoot[gid],
                                     partition->symV2PanelRoot[gid], grid);
        local_owner[gid] = owns ? grid3d->iam : INT_MAX;
        local_owner_count[gid] = owns;
    }
    MPI_Allreduce(local_owner, partition->symV2DiagOwner, TEST_NSUPERS,
                  MPI_INT, MPI_MIN, grid3d->comm);
    MPI_Allreduce(local_owner_count, owner_count, TEST_NSUPERS,
                  MPI_INT, MPI_SUM, grid3d->comm);
    for (int_t gid = 0; gid < TEST_NSUPERS; ++gid)
        if (partition->symV2DiagOwner[gid] == INT_MAX ||
            owner_count[gid] != 1)
            ABORT("SymLDL CPU test diagonal ownership is not unique.");
    free(owner_count);
    free(local_owner_count);
    free(local_owner);

    for (int_t gid = 0; gid < TEST_NSUPERS; ++gid) {
        if (partition->superGridMap[gid] == NOT_IN_GRID)
            continue;
        panel_count += partition->symV2PanelRoot[gid] == mycol;
        row_count += partition->symV2DiagRoot[gid] == myrow;
        if (partition->symV2PanelRoot[gid] == mycol)
            for (int_t target = gid + 1; target < TEST_NSUPERS; ++target)
                block_count +=
                               partition->superGridMap[target] != NOT_IN_GRID &&
                               partition->symV2DiagRoot[target] == myrow &&
                               test_l_value(target, gid) != 0.0;
    }

    partition->symV2LocalPanelCount = panel_count;
    partition->symV2LocalRowCount = row_count;
    partition->symV2LocalPanelGids =
        (int_t *) test_alloc((size_t) panel_count, sizeof(int_t));
    partition->symV2LocalRowGids =
        (int_t *) test_alloc((size_t) row_count, sizeof(int_t));
    ilsum = (int_t *) test_alloc((size_t) row_count + 1, sizeof(int_t));
    panels = (dSymLDLPanelDesc *)
        test_alloc((size_t) panel_count, sizeof(*panels));
    blocks = (dSymLDLBlockDesc *)
        test_alloc((size_t) block_count, sizeof(*blocks));
    rows = (int_t *) test_alloc((size_t) block_count, sizeof(int_t));
    panel_values = (double **)
        test_alloc((size_t) panel_count, sizeof(double *));

    panel_count = 0;
    row_count = 0;
    for (int_t gid = 0; gid < TEST_NSUPERS; ++gid) {
        if (partition->superGridMap[gid] == NOT_IN_GRID)
            continue;
        if (partition->symV2PanelRoot[gid] == mycol) {
            partition->symV2PanelLocalIndex[gid] = panel_count;
            partition->symV2LocalPanelGids[panel_count++] = gid;
        }
        if (partition->symV2DiagRoot[gid] == myrow) {
            partition->symV2RowLocalIndex[gid] = row_count;
            partition->symV2LocalRowGids[row_count] = gid;
            ilsum[row_count] = row_count;
            ++row_count;
        }
    }
    ilsum[row_count] = row_count;

    for (int_t slot = 0; slot < panel_count; ++slot) {
        int_t gid = partition->symV2LocalPanelGids[slot];
        int has_diag = partition->symV2DiagRoot[gid] == myrow;
        int_t local_blocks = 0;
        int_t value_pos;
        for (int_t target = gid + 1; target < TEST_NSUPERS; ++target)
            local_blocks +=
                            partition->superGridMap[target] != NOT_IN_GRID &&
                            partition->symV2DiagRoot[target] == myrow &&
                            test_l_value(target, gid) != 0.0;

        panels[slot].gid = gid;
        panels[slot].width = 1;
        panels[slot].diag_luptr = 0;
        panels[slot].block_begin = edge;
        panels[slot].block_count = local_blocks;
        panels[slot].has_diag = has_diag;
        panels[slot].active = has_diag || local_blocks > 0;
        panels[slot].nsupr = has_diag + local_blocks;
        panels[slot].value_count = panels[slot].nsupr;
        if (!panels[slot].active)
            continue;
        panel_values[slot] = (double *)
            test_alloc((size_t) panels[slot].value_count, sizeof(double));
        panels[slot].values = panel_values[slot];
        value_pos = 0;
        if (has_diag)
            panel_values[slot][value_pos++] =
                partition->superGridMap[gid] == IN_GRID_AIJ
                    ? 1.0 / test_d_value(gid) : 0.0;
        for (int_t target = gid + 1; target < TEST_NSUPERS; ++target) {
            double value = test_l_value(target, gid);
            if (partition->superGridMap[target] == NOT_IN_GRID ||
                partition->symV2DiagRoot[target] != myrow || value == 0.0)
                continue;
            blocks[edge].panel_id = slot;
            blocks[edge].target_gid = target;
            blocks[edge].luptr = value_pos;
            blocks[edge].nbrow = 1;
            blocks[edge].row_begin = row_entry;
            rows[row_entry++] = target;
            panel_values[slot][value_pos++] =
                partition->superGridMap[gid] == IN_GRID_AIJ ? value : 0.0;
            ++edge;
        }
        if (value_pos != panels[slot].value_count)
            ABORT("SymLDL CPU test panel fill is inconsistent.");
    }
    if (edge != block_count || row_entry != block_count)
        ABORT("SymLDL CPU test retained graph fill is inconsistent.");

    dSymLDLSolveGraph *graph = dSymLDLSolveGraphCreate(
        TEST_NSUPERS, TEST_NSUPERS, panel_count, panels,
        block_count, blocks, block_count, rows, xsup, ilsum,
        partition, grid3d);
    free(ilsum);
    free(rows);
    free(blocks);
    free(panels);
    if (graph == NULL)
        ABORT("SymLDL CPU test graph creation failed.");
    *panel_values_out = panel_values;
    return graph;
}

static void
test_partition_free(dtrf3Dpartition_t *partition, int npdep)
{
    int forest_count = 2 * npdep - 1;
    free(partition->symV2LocalRowGids);
    free(partition->symV2LocalPanelGids);
    free(partition->superGridMap);
    free(partition->symV2RowLocalIndex);
    free(partition->symV2PanelLocalIndex);
    free(partition->symV2DiagOwner);
    free(partition->symV2PanelRoot);
    free(partition->symV2DiagRoot);
    free(partition->myZeroTrIdxs);
    free(partition->myTreeIdxs);
    for (int forest = 0; forest < forest_count; ++forest) {
        free(partition->sForests[forest]->nodeList);
        free(partition->sForests[forest]);
    }
    free(partition->sForests);
}

static double
test_phase_error(const dSymLDLSolveGraph *graph,
                 const dtrf3Dpartition_t *partition,
                 gridinfo3d_t *grid3d, const double *x, int nrhs,
                 const double *expected)
{
    double local[TEST_NSUPERS * TEST_NRHS_CAPACITY];
    double result[TEST_NSUPERS * TEST_NRHS_CAPACITY];
    double error = 0.0;
    memset(local, 0, sizeof(local));

    for (int_t gid = 0; gid < TEST_NSUPERS; ++gid) {
        int_t slot = graph->row_local_index[gid];
        if (grid3d->iam != partition->symV2DiagOwner[gid])
            continue;
        if (slot < 0)
            ABORT("SymLDL CPU test owner has no local solution row.");
        int_t offset = graph->ilsum[slot] * nrhs + (slot + 1) * XK_H;
        for (int rhs_id = 0; rhs_id < nrhs; ++rhs_id)
            local[gid + (int_t) rhs_id * TEST_NSUPERS] =
                x[offset + rhs_id];
    }
    MPI_Allreduce(local, result, TEST_NSUPERS * nrhs, MPI_DOUBLE,
                  MPI_SUM, grid3d->comm);
    for (int i = 0; i < TEST_NSUPERS * nrhs; ++i)
        error = SUPERLU_MAX(error, fabs(result[i] - expected[i]));
    return error;
}

static double
test_run(dSymLDLCPUSolveHandle *handle, const dSymLDLSolveGraph *graph,
         const dtrf3Dpartition_t *partition, gridinfo3d_t *grid3d, int nrhs)
{
    int_t x_count = graph->ilsum[graph->row_count] * nrhs +
                    (graph->row_count + 1) * XK_H;
    double rhs[TEST_NSUPERS * TEST_NRHS_CAPACITY];
    double post_forward[TEST_NSUPERS * TEST_NRHS_CAPACITY];
    double post_diagonal[TEST_NSUPERS * TEST_NRHS_CAPACITY];
    double expected[TEST_NSUPERS * TEST_NRHS_CAPACITY];
    double *x = (double *) test_alloc((size_t) x_count, sizeof(double));
    double error = 0.0;
    test_rhs(nrhs, rhs, post_forward, post_diagonal, expected);

    for (int_t slot = 0; slot < graph->row_count; ++slot) {
        int_t gid = graph->row_gids[slot];
        int_t offset = graph->ilsum[slot] * nrhs + (slot + 1) * XK_H;
        x[offset - XK_H] = (double) gid;
        if (grid3d->iam == partition->symV2DiagOwner[gid])
            for (int rhs_id = 0; rhs_id < nrhs; ++rhs_id)
                x[offset + rhs_id] =
                    rhs[gid + (int_t) rhs_id * TEST_NSUPERS];
    }
    if (dSymLDLCPUForward(handle, x, x_count, nrhs) != 0)
        ABORT("SymLDL CPU test forward solve failed.");
    error = SUPERLU_MAX(
        error, test_phase_error(
                   graph, partition, grid3d, x, nrhs, post_forward));
    if (dSymLDLCPUDiagonal(handle, x, x_count, nrhs) != 0)
        ABORT("SymLDL CPU test diagonal solve failed.");
    error = SUPERLU_MAX(
        error, test_phase_error(
                   graph, partition, grid3d, x, nrhs, post_diagonal));
    if (dSymLDLCPUBackward(handle, x, x_count, nrhs) != 0)
        ABORT("SymLDL CPU test backward solve failed.");
    error = SUPERLU_MAX(
        error, test_phase_error(
                   graph, partition, grid3d, x, nrhs, expected));
    free(x);
    return error;
}

int
main(int argc, char **argv)
{
    int provided;
    int nprow = argc > 1 ? atoi(argv[1]) : 1;
    int npcol = argc > 2 ? atoi(argv[2]) : 1;
    int npdep = argc > 3 ? atoi(argv[3]) : 1;
    int repeats = argc > 4 ? atoi(argv[4]) : 5;
    int world_size;
    gridinfo3d_t grid3d;
    dtrf3Dpartition_t partition;
    dSymLDLSolveGraph *graph;
    dSymLDLCPUSolveHandle *handle;
    double **panel_values;
    double max_error = 0.0;
    double timers[6];
    double consumed[6];

    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (provided < MPI_THREAD_FUNNELED || nprow <= 0 || npcol <= 0 ||
        (npdep != 1 && npdep != 2 && npdep != 4) ||
        nprow * npcol * npdep != world_size || repeats <= 0)
        ABORT("SymLDL CPU test process grid is invalid.");
    superlu_gridinit3d(MPI_COMM_WORLD, nprow, npcol, npdep, &grid3d);
    graph = test_graph_create(&grid3d, &partition, &panel_values);
    handle = dSymLDLCPUSolveCreate(
        graph, TEST_NRHS_CAPACITY, &partition, &grid3d);
    if (handle == NULL)
        ABORT("SymLDL CPU test handle creation failed.");

    for (int iteration = 0; iteration < repeats; ++iteration) {
        int nrhs = iteration % 2 == 0 ? 1 : TEST_NRHS_CAPACITY;
        max_error = SUPERLU_MAX(
            max_error, test_run(handle, graph, &partition, &grid3d, nrhs));
    }
    if (grid3d.iam == 0)
        printf("SymLDL CPU retained-graph test: grid=%dx%dx%d repeats=%d "
               "max_error=%.6e\n", nprow, npcol, npdep, repeats,
               max_error);
    if (max_error > 1e-11)
        ABORT("SymLDL CPU retained-graph test is numerically incorrect.");

    dSymLDLCPUSolveTakeTimers(
        handle, &timers[0], &timers[1], &timers[2], &timers[3],
        &timers[4], &timers[5]);
    dSymLDLCPUSolveTakeTimers(
        handle, &consumed[0], &consumed[1], &consumed[2], &consumed[3],
        &consumed[4], &consumed[5]);
    for (int timer = 0; timer < 6; ++timer) {
        if (timers[timer] < 0.0 || consumed[timer] != 0.0)
            ABORT("SymLDL CPU solve timer consumption is inconsistent.");
    }

    dSymLDLCPUSolveDestroy(handle);
    dSymLDLSolveGraphDestroy(graph);
    for (int_t slot = 0; slot < partition.symV2LocalPanelCount; ++slot)
        free(panel_values[slot]);
    free(panel_values);
    test_partition_free(&partition, npdep);
    superlu_gridexit3d(&grid3d);
    MPI_Finalize();
    return 0;
}
