

#include "superlu_ddefs.h"
#include <string.h>

extern int_t calcTopInfoForest(sForest_t *forest, int_t nsupers,
                               int_t *setree);

typedef struct {
    double panel_work;
    double row_work;
    double rank_work;
    double tree_weight;
} dSymV2LDLCost_t;

static void dSymV2CostFromDims(double ksupc, double lrows, int nprow,
                               dSymV2LDLCost_t *cost)
{
    if (lrows < ksupc)
        lrows = ksupc;

    double below = lrows - ksupc;
    double diag_cost = ksupc * ksupc * ksupc;
    double panel_factor_cost = below * ksupc * ksupc;
    double ll_schur_cost = 0.5 * below * below * ksupc;
    double partner_comm_cost = (nprow > 1) ? below * ksupc : 0.0;
    double solve_cost = ksupc * ksupc + 2.0 * below * ksupc;

    cost->panel_work =
        SUPERLU_MAX(1.0, panel_factor_cost + ll_schur_cost +
                         partner_comm_cost);
    cost->row_work =
        SUPERLU_MAX(1.0, diag_cost + solve_cost + partner_comm_cost);
    cost->rank_work =
        SUPERLU_MAX(1.0, diag_cost + panel_factor_cost + solve_cost +
                         0.25 * ll_schur_cost + partner_comm_cost);
    cost->tree_weight =
        SUPERLU_MAX(1.0, diag_cost + panel_factor_cost + ll_schur_cost +
                         partner_comm_cost + solve_cost);
}

static void dSymV2CalcLDLTreeWeight(int_t nsupers, int_t *setree,
                                    treeList_t *treeList, int_t *xsup,
                                    Glu_freeable_t *Glu_freeable,
                                    int_t **Lrowind_bc_ptr,
                                    gridinfo3d_t *grid3d)
{
    gridinfo_t *grid = &(grid3d->grid2d);
    int_t *mylsize = INT_T_ALLOC(nsupers);

    for (int_t k = 0; k < nsupers; ++k)
        mylsize[k] = 0;

    if (Lrowind_bc_ptr != NULL)
    {
        int_t mycol = MYCOL(grid->iam, grid);
        for (int_t k = 0; k < nsupers; ++k)
        {
            if (mycol == PCOL(k, grid))
            {
                int_t lk = LBj(k, grid);
                int_t *lsub = Lrowind_bc_ptr[lk];
                if (lsub != NULL)
                    mylsize[k] = lsub[1];
            }
        }
    }
    else if (Glu_freeable != NULL &&
             Glu_freeable->xlsub != NULL &&
             Glu_freeable->lsub != NULL)
    {
        for (int_t k = 0; k < nsupers; ++k)
        {
            int_t fsupc = xsup[k];
            int_t nrows = Glu_freeable->xlsub[fsupc + 1] -
                          Glu_freeable->xlsub[fsupc];
            mylsize[k] = SUPERLU_MAX((int_t)0, nrows);
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, mylsize, nsupers, mpi_int_t, MPI_MAX,
                  grid->comm);

    for (int_t k = 0; k < nsupers; ++k)
    {
        double ksupc = (double)SuperSize(k);
        double depth = (double)SUPERLU_MAX((int_t)1, treeList[k].depth);
        double lrows = (double)mylsize[k];
        if (lrows <= 0.0)
            lrows = ksupc + depth;
        if (lrows < ksupc)
            lrows = ksupc;

        dSymV2LDLCost_t cost;
        dSymV2CostFromDims(ksupc, lrows, grid->nprow, &cost);
        treeList[k].weight = cost.tree_weight;
        treeList[k].iWeight = treeList[k].weight;
        treeList[k].scuWeight = treeList[k].weight;
    }

    treeList[nsupers].iWeight = 0.0;
    for (int_t k = 0; k < nsupers; ++k)
    {
        int_t parent = setree[k];
        treeList[parent].iWeight += treeList[k].iWeight;
    }

    SUPERLU_FREE(mylsize);
}

static void dSymV2ResetLDLMetadata(dtrf3Dpartition_t *trf3Dpart)
{
    trf3Dpart->symV2DiagOwner = NULL;
    trf3Dpart->symV2PanelRoot = NULL;
    trf3Dpart->symV2DiagRoot = NULL;
    trf3Dpart->symV2PanelLocalIndex = NULL;
    trf3Dpart->symV2RowLocalIndex = NULL;
    trf3Dpart->symV2LocalPanelGids = NULL;
    trf3Dpart->symV2LocalRowGids = NULL;
    trf3Dpart->symV2LocalPanelCount = 0;
    trf3Dpart->symV2LocalRowCount = 0;
    trf3Dpart->symV2ScheduleEnabled = 0;
    trf3Dpart->symV2FactorLevelCount = 0;
    trf3Dpart->symV2FactorLevelPtr = NULL;
    trf3Dpart->symV2FactorNodes = NULL;
    trf3Dpart->symV2NodeLevel = NULL;
    trf3Dpart->symV2NodeOrder = NULL;
    trf3Dpart->symV2NodeIperm = NULL;
}

static void dSymV2EstimateSupernodeWork(int_t k, int_t *xsup,
                                        Glu_freeable_t *Glu_freeable,
                                        gridinfo_t *grid,
                                        dSymV2LDLCost_t *cost)
{
    double ksupc = (double)SuperSize(k);
    double lrows = ksupc;

    if (Glu_freeable != NULL &&
        Glu_freeable->xlsub != NULL &&
        Glu_freeable->lsub != NULL &&
        xsup != NULL)
    {
        int_t fsupc = xsup[k];
        int_t nrows = Glu_freeable->xlsub[fsupc + 1] -
                      Glu_freeable->xlsub[fsupc];
        if (nrows > 0)
            lrows = (double)nrows;
    }
    dSymV2CostFromDims(ksupc, lrows, grid->nprow, cost);
}

static void dSymV2ChooseOwnerPair(const double *panel_load,
                                  const double *row_load,
                                  const double *rank_load,
                                  gridinfo_t *grid,
                                  const dSymV2LDLCost_t *cost,
                                  int *diag_root, int *panel_root)
{
    double best_score = -1.0;
    int best_pr = 0;
    int best_pc = 0;

    for (int pr = 0; pr < grid->nprow; ++pr)
    {
        for (int pc = 0; pc < grid->npcol; ++pc)
        {
            int rank = PNUM(pr, pc, grid);
            double projected_panel = panel_load[pc] + cost->panel_work;
            double projected_row = row_load[pr] + cost->row_work;
            double projected_rank = rank_load[rank] + cost->rank_work;
            double max_projected = SUPERLU_MAX(projected_panel,
                                      SUPERLU_MAX(projected_row,
                                                  projected_rank));
            double score = max_projected +
                           0.05 * (projected_panel + projected_row +
                                   projected_rank);
            if (best_score < 0.0 || score < best_score)
            {
                best_score = score;
                best_pr = pr;
                best_pc = pc;
            }
        }
    }

    *diag_root = best_pr;
    *panel_root = best_pc;
}

static void dSymV2InitLDLOwners(int_t nsupers,
                                dtrf3Dpartition_t *trf3Dpart,
                                int_t *xsup,
                                Glu_freeable_t *Glu_freeable,
                                gridinfo3d_t *grid3d)
{
    gridinfo_t *grid = &(grid3d->grid2d);
    int global_rank;
    int *local_owner;
    size_t owner_bytes = (size_t) nsupers * sizeof(int);
    double *panel_load = (double *) SUPERLU_MALLOC(grid->npcol * sizeof(double));
    double *row_load = (double *) SUPERLU_MALLOC(grid->nprow * sizeof(double));
    double *rank_load =
        (double *) SUPERLU_MALLOC(grid->nprow * grid->npcol * sizeof(double));

    if (panel_load == NULL || row_load == NULL || rank_load == NULL)
        ABORT("Malloc fails for SymFact V2 LDL load metadata.");
    for (int pc = 0; pc < grid->npcol; ++pc)
        panel_load[pc] = 0.0;
    for (int pr = 0; pr < grid->nprow; ++pr)
        row_load[pr] = 0.0;
    for (int p = 0; p < grid->nprow * grid->npcol; ++p)
        rank_load[p] = 0.0;

    dSymV2ResetLDLMetadata(trf3Dpart);

    if (!(trf3Dpart->symV2DiagOwner = (int *) SUPERLU_MALLOC(owner_bytes)) ||
        !(trf3Dpart->symV2PanelRoot = (int *) SUPERLU_MALLOC(owner_bytes)) ||
        !(trf3Dpart->symV2DiagRoot = (int *) SUPERLU_MALLOC(owner_bytes)) ||
        !(trf3Dpart->symV2PanelLocalIndex = INT_T_ALLOC(nsupers)) ||
        !(trf3Dpart->symV2RowLocalIndex = INT_T_ALLOC(nsupers)) ||
        !(local_owner = (int *) SUPERLU_MALLOC(owner_bytes)))
        ABORT("Malloc fails for SymFact V2 LDL owner metadata.");

    MPI_Comm_rank(grid3d->comm, &global_rank);
    for (int_t k = 0; k < nsupers; ++k)
    {
        dSymV2LDLCost_t cost;
        int panel_root;
        int diag_root;
        int owner_rank;
        dSymV2EstimateSupernodeWork(k, xsup, Glu_freeable, grid, &cost);
        dSymV2ChooseOwnerPair(panel_load, row_load, rank_load, grid, &cost,
                              &diag_root, &panel_root);
        owner_rank = PNUM(diag_root, panel_root, grid);
        panel_load[panel_root] += cost.panel_work;
        row_load[diag_root] += cost.row_work;
        rank_load[owner_rank] += cost.rank_work;
        trf3Dpart->symV2PanelRoot[k] = panel_root;
        trf3Dpart->symV2DiagRoot[k] = diag_root;
        trf3Dpart->symV2PanelLocalIndex[k] = -1;
        trf3Dpart->symV2RowLocalIndex[k] = -1;
        local_owner[k] =
            (grid3d->zscp.Iam == 0 &&
             grid->iam == PNUM(diag_root, panel_root, grid))
                ? global_rank
                : INT_MAX;
    }

    MPI_Allreduce(local_owner, trf3Dpart->symV2DiagOwner, (int) nsupers,
                  MPI_INT, MPI_MIN, grid3d->comm);
    for (int_t k = 0; k < nsupers; ++k)
        if (trf3Dpart->symV2DiagOwner[k] == INT_MAX)
            ABORT("SymFact V2 LDL owner metadata is missing a diagonal owner.");

    int myrow = MYROW(grid->iam, grid);
    int mycol = MYCOL(grid->iam, grid);
    for (int_t k = 0; k < nsupers; ++k)
    {
        if (trf3Dpart->symV2PanelRoot[k] == mycol)
            ++trf3Dpart->symV2LocalPanelCount;
        if (trf3Dpart->symV2DiagRoot[k] == myrow)
            ++trf3Dpart->symV2LocalRowCount;
    }
    if (trf3Dpart->symV2LocalPanelCount > 0 &&
        !(trf3Dpart->symV2LocalPanelGids =
              INT_T_ALLOC(trf3Dpart->symV2LocalPanelCount)))
        ABORT("Malloc fails for SymFact V2 local panel gids.");
    if (trf3Dpart->symV2LocalRowCount > 0 &&
        !(trf3Dpart->symV2LocalRowGids =
              INT_T_ALLOC(trf3Dpart->symV2LocalRowCount)))
        ABORT("Malloc fails for SymFact V2 local row gids.");

    int_t local_panel = 0;
    int_t local_row = 0;
    for (int_t k = 0; k < nsupers; ++k)
    {
        if (trf3Dpart->symV2PanelRoot[k] == mycol)
        {
            trf3Dpart->symV2PanelLocalIndex[k] = local_panel;
            trf3Dpart->symV2LocalPanelGids[local_panel++] = k;
        }
        if (trf3Dpart->symV2DiagRoot[k] == myrow)
        {
            trf3Dpart->symV2RowLocalIndex[k] = local_row;
            trf3Dpart->symV2LocalRowGids[local_row++] = k;
        }
    }

    SUPERLU_FREE(local_owner);
    SUPERLU_FREE(panel_load);
    SUPERLU_FREE(row_load);
    SUPERLU_FREE(rank_load);
}

static void dSymV2BuildLocalLDLIndexes(int_t nsupers,
                                       dtrf3Dpartition_t *trf3Dpart,
                                       gridinfo3d_t *grid3d)
{
    gridinfo_t *grid = &(grid3d->grid2d);
    int myrow = MYROW(grid->iam, grid);
    int mycol = MYCOL(grid->iam, grid);

    if (trf3Dpart->symV2PanelLocalIndex == NULL ||
        trf3Dpart->symV2RowLocalIndex == NULL ||
        trf3Dpart->superGridMap == NULL)
        ABORT("SymFact V2 LDL local index metadata is not initialized.");

    if (trf3Dpart->symV2LocalPanelGids != NULL)
        SUPERLU_FREE(trf3Dpart->symV2LocalPanelGids);
    if (trf3Dpart->symV2LocalRowGids != NULL)
        SUPERLU_FREE(trf3Dpart->symV2LocalRowGids);
    trf3Dpart->symV2LocalPanelGids = NULL;
    trf3Dpart->symV2LocalRowGids = NULL;
    trf3Dpart->symV2LocalPanelCount = 0;
    trf3Dpart->symV2LocalRowCount = 0;

    for (int_t k = 0; k < nsupers; ++k)
    {
        trf3Dpart->symV2PanelLocalIndex[k] = -1;
        trf3Dpart->symV2RowLocalIndex[k] = -1;
        if (trf3Dpart->superGridMap[k] == NOT_IN_GRID)
            continue;
        if (trf3Dpart->symV2PanelRoot[k] == mycol)
            ++trf3Dpart->symV2LocalPanelCount;
        if (trf3Dpart->symV2DiagRoot[k] == myrow)
            ++trf3Dpart->symV2LocalRowCount;
    }

    if (trf3Dpart->symV2LocalPanelCount > 0)
    {
        trf3Dpart->symV2LocalPanelGids =
            INT_T_ALLOC(trf3Dpart->symV2LocalPanelCount);
        if (trf3Dpart->symV2LocalPanelGids == NULL)
            ABORT("Malloc fails for SymFact V2 local panel gids.");
    }
    if (trf3Dpart->symV2LocalRowCount > 0)
    {
        trf3Dpart->symV2LocalRowGids =
            INT_T_ALLOC(trf3Dpart->symV2LocalRowCount);
        if (trf3Dpart->symV2LocalRowGids == NULL)
            ABORT("Malloc fails for SymFact V2 local row gids.");
    }

    int_t local_panel = 0;
    int_t local_row = 0;
    for (int_t k = 0; k < nsupers; ++k)
    {
        if (trf3Dpart->superGridMap[k] == NOT_IN_GRID)
            continue;
        if (trf3Dpart->symV2PanelRoot[k] == mycol)
        {
            trf3Dpart->symV2PanelLocalIndex[k] = local_panel;
            trf3Dpart->symV2LocalPanelGids[local_panel++] = k;
        }
        if (trf3Dpart->symV2DiagRoot[k] == myrow)
        {
            trf3Dpart->symV2RowLocalIndex[k] = local_row;
            trf3Dpart->symV2LocalRowGids[local_row++] = k;
        }
    }
}

static void dSymV2UpdateLDLDiagOwners(int_t nsupers,
                                      dtrf3Dpartition_t *trf3Dpart,
                                      gridinfo3d_t *grid3d)
{
    gridinfo_t *grid = &(grid3d->grid2d);
    int global_rank;
    int *local_owner;

    if (trf3Dpart->symV2DiagOwner == NULL ||
        trf3Dpart->symV2PanelRoot == NULL ||
        trf3Dpart->symV2DiagRoot == NULL ||
        trf3Dpart->superGridMap == NULL)
        ABORT("SymFact V2 LDL owner metadata is not initialized.");

    local_owner = (int *) SUPERLU_MALLOC((size_t) nsupers * sizeof(int));
    if (local_owner == NULL)
        ABORT("Malloc fails for SymFact V2 diagonal owner workspace.");

    MPI_Comm_rank(grid3d->comm, &global_rank);
    for (int_t k = 0; k < nsupers; ++k)
    {
        int owner_2d = PNUM(trf3Dpart->symV2DiagRoot[k],
                            trf3Dpart->symV2PanelRoot[k], grid);
        local_owner[k] =
            (trf3Dpart->superGridMap[k] == IN_GRID_AIJ &&
             grid->iam == owner_2d)
                ? global_rank
                : INT_MAX;
    }

    MPI_Allreduce(local_owner, trf3Dpart->symV2DiagOwner, (int) nsupers,
                  MPI_INT, MPI_MIN, grid3d->comm);
    for (int_t k = 0; k < nsupers; ++k)
        if (trf3Dpart->symV2DiagOwner[k] == INT_MAX)
            ABORT("SymFact V2 LDL owner metadata is missing a diagonal owner.");

    SUPERLU_FREE(local_owner);
}

static void dSymV2ComputeForestDiagDims(int_t nsupers,
                                        dtrf3Dpartition_t *trf3Dpart,
                                        int_t *xsup,
                                        gridinfo3d_t *grid3d)
{
    int_t maxLvl = trf3Dpart->maxLvl;
    int_t mxLeafNode = 0;
    gridinfo_t *grid = &(grid3d->grid2d);
    int mycol = MYCOL(grid->iam, grid);

    if (trf3Dpart->diagDims != NULL)
        SUPERLU_FREE(trf3Dpart->diagDims);
    trf3Dpart->diagDims = NULL;

    for (int_t ilvl = 0; ilvl < maxLvl; ++ilvl)
    {
        sForest_t *sforest =
            trf3Dpart->sForests[trf3Dpart->myTreeIdxs[ilvl]];
        if (sforest != NULL && sforest->topoInfo.eTreeTopLims != NULL)
            mxLeafNode = SUPERLU_MAX(mxLeafNode,
                                     sforest->topoInfo.eTreeTopLims[1]);
    }

    trf3Dpart->mxLeafNode = (int) mxLeafNode;
    trf3Dpart->diagDims = int32Calloc_dist((int) SUPERLU_MAX(mxLeafNode, (int_t)1));
    if (trf3Dpart->diagDims == NULL)
        ABORT("Calloc fails for SymFact V2 LDL diagonal dimensions.");

    for (int_t ilvl = 0; ilvl < maxLvl; ++ilvl)
    {
        sForest_t *sforest =
            trf3Dpart->sForests[trf3Dpart->myTreeIdxs[ilvl]];
        if (sforest == NULL || sforest->topoInfo.eTreeTopLims == NULL)
            continue;
        for (int_t topoLvl = 0; topoLvl < sforest->topoInfo.numLvl;
             ++topoLvl)
        {
            int_t k_st = sforest->topoInfo.eTreeTopLims[topoLvl];
            int_t k_end = sforest->topoInfo.eTreeTopLims[topoLvl + 1];
            for (int_t k0 = k_st; k0 < k_end; ++k0)
            {
                int_t offset = k0 - k_st;
                int_t k = sforest->nodeList[k0];
                int_t ksupc = xsup[k + 1] - xsup[k];
                if (k >= 0 && k < nsupers &&
                    trf3Dpart->symV2PanelRoot[k] == mycol)
                    trf3Dpart->diagDims[offset] =
                        SUPERLU_MAX(trf3Dpart->diagDims[offset],
                                    (int) ksupc);
            }
        }
    }
}

static void dSymV2ValidateLDLForests(int_t nsupers,
                                     dtrf3Dpartition_t *trf3Dpart,
                                     int_t *gNodeCount,
                                     int_t **gNodeLists)
{
    int_t maxLvl = trf3Dpart->maxLvl;
    int_t numForests = (1 << maxLvl) - 1;
    int *seen = int32Calloc_dist((int) nsupers);
    if (seen == NULL)
        ABORT("Calloc fails for SymFact V2 LDL forest validation.");

    for (int_t tree = 0; tree < numForests; ++tree)
    {
        for (int_t i = 0; i < gNodeCount[tree]; ++i)
        {
            int_t k = gNodeLists[tree][i];
            if (k < 0 || k >= nsupers)
                ABORT("SymFact V2 LDL forest contains an invalid supernode.");
            ++seen[k];
        }
    }
    for (int_t k = 0; k < nsupers; ++k)
        if (seen[k] != 1)
            ABORT("SymFact V2 LDL forests do not partition supernodes exactly once.");
    SUPERLU_FREE(seen);
}

static void dSymV2TraceLDLForests(dtrf3Dpartition_t *trf3Dpart,
                                  int_t *gNodeCount,
                                  gridinfo3d_t *grid3d)
{
    const char *env = getenv("GPU3DV2_TRACE");
    if (env == NULL || env[0] == '\0' || env[0] == '0')
        return;

    int_t maxLvl = trf3Dpart->maxLvl;
    int_t numForests = (1 << maxLvl) - 1;
    double local_weight = 0.0;
    int_t local_nodes = 0;

    for (int_t ilvl = 0; ilvl < maxLvl; ++ilvl)
    {
        int_t tree = trf3Dpart->myTreeIdxs[ilvl];
        sForest_t *sforest = trf3Dpart->sForests[tree];
        if (sforest != NULL && !trf3Dpart->myZeroTrIdxs[ilvl])
        {
            local_nodes += sforest->nNodes;
            local_weight += sforest->weight;
        }
    }

    fprintf(stderr,
            "[sym-v2-trace] rank %d z=%d LDL forests maxLvl=%lld numForests=%lld localActiveNodes=%lld localActiveWeight=%.6e localPanels=%lld localRows=%lld\n",
            grid3d ? grid3d->iam : -1,
            grid3d ? grid3d->zscp.Iam : -1,
            (long long) maxLvl, (long long) numForests,
            (long long) local_nodes, local_weight,
            (long long) trf3Dpart->symV2LocalPanelCount,
            (long long) trf3Dpart->symV2LocalRowCount);
    for (int_t tree = 0; tree < numForests; ++tree)
    {
        sForest_t *sforest = trf3Dpart->sForests[tree];
        fprintf(stderr,
                "[sym-v2-trace] rank %d forest=%lld nodes=%lld weight=%.6e present=%d\n",
                grid3d ? grid3d->iam : -1,
                (long long) tree,
                (long long) gNodeCount[tree],
                sforest ? sforest->weight : 0.0,
                sforest != NULL);
    }
    fflush(stderr);
}

static void dSymV2BuildLDLSchedule(int_t nsupers, int_t *setree,
                                   dtrf3Dpartition_t *trf3Dpart,
                                   int_t *xsup, gridinfo3d_t *grid3d)
{
    gridinfo_t *grid = &(grid3d->grid2d);
    int mycol = MYCOL(grid->iam, grid);
    sForest_t tmp_forest;
    int_t nlevels;
    int_t *level_ptr;
    int_t *nodes = intMalloc_dist(nsupers);
    int_t *node_level = intMalloc_dist(nsupers);
    int_t *node_order = intMalloc_dist(nsupers);
    int_t *node_iperm = intMalloc_dist(nsupers + 1);
    int_t max_width = 1;

    memset(&tmp_forest, 0, sizeof(tmp_forest));
    tmp_forest.nNodes = nsupers;
    tmp_forest.numTrees = 1;
    tmp_forest.nodeList = INT_T_ALLOC(nsupers);
    if (tmp_forest.nodeList == NULL)
        ABORT("Malloc fails for SymFact V2 LDL temporary forest.");
    for (int_t k = 0; k < nsupers; ++k)
        tmp_forest.nodeList[k] = k;
    calcTopInfoForest(&tmp_forest, nsupers, setree);

    nlevels = tmp_forest.topoInfo.numLvl;
    level_ptr = intMalloc_dist(nlevels + 1);
    if (level_ptr == NULL || nodes == NULL || node_level == NULL ||
        node_order == NULL || node_iperm == NULL)
        ABORT("Malloc fails for SymFact V2 LDL schedule metadata.");

    for (int_t k = 0; k < nsupers; ++k)
    {
        node_level[k] = -1;
        node_order[k] = -1;
        nodes[k] = tmp_forest.nodeList[k];
        node_iperm[k] = tmp_forest.topoInfo.myIperm[k];
    }
    node_iperm[nsupers] = nsupers;

    for (int_t level = 0; level <= nlevels; ++level)
        level_ptr[level] = tmp_forest.topoInfo.eTreeTopLims[level];
    for (int_t level = 0; level < nlevels; ++level)
    {
        int_t begin = level_ptr[level];
        int_t end = level_ptr[level + 1];
        max_width = SUPERLU_MAX(max_width, end - begin);
        for (int_t pos = begin; pos < end; ++pos)
        {
            int_t k = nodes[pos];
            node_level[k] = level;
            node_order[k] = pos;
        }
    }

    trf3Dpart->symV2ScheduleEnabled = 1;
    trf3Dpart->symV2FactorLevelCount = nlevels;
    trf3Dpart->symV2FactorLevelPtr = level_ptr;
    trf3Dpart->symV2FactorNodes = nodes;
    trf3Dpart->symV2NodeLevel = node_level;
    trf3Dpart->symV2NodeOrder = node_order;
    trf3Dpart->symV2NodeIperm = node_iperm;
    trf3Dpart->mxLeafNode = (int) max_width;
    trf3Dpart->diagDims = int32Calloc_dist((int) max_width);
    if (trf3Dpart->diagDims == NULL)
        ABORT("Calloc fails for SymFact V2 LDL diagonal dimensions.");
    for (int_t level = 0; level < nlevels; ++level)
    {
        int_t begin = level_ptr[level];
        int_t end = level_ptr[level + 1];
        for (int_t offset = 0; offset < end - begin; ++offset)
        {
            int_t k = nodes[begin + offset];
            int_t ksupc = xsup[k + 1] - xsup[k];
            if (trf3Dpart->symV2PanelRoot[k] == mycol)
                trf3Dpart->diagDims[offset] =
                    SUPERLU_MAX(trf3Dpart->diagDims[offset], (int) ksupc);
        }
    }

    SUPERLU_FREE(tmp_forest.nodeList);
    SUPERLU_FREE(tmp_forest.topoInfo.eTreeTopLims);
    SUPERLU_FREE(tmp_forest.topoInfo.myIperm);
}

static sForest_t *dSymV2CreateLDLForest(int_t nsupers,
                                        dtrf3Dpartition_t *trf3Dpart,
                                        treeList_t *treeList)
{
    sForest_t *forest = (sForest_t *) SUPERLU_MALLOC(sizeof(sForest_t));
    if (forest == NULL)
        ABORT("Malloc fails for SymFact V2 LDL forest.");

    forest->nNodes = nsupers;
    forest->numTrees = 1;
    forest->numLvl = trf3Dpart->symV2FactorLevelCount;
    forest->nodeList = INT_T_ALLOC(nsupers);
    forest->topoInfo.numLvl = trf3Dpart->symV2FactorLevelCount;
    forest->topoInfo.eTreeTopLims =
        INT_T_ALLOC(trf3Dpart->symV2FactorLevelCount + 1);
    forest->topoInfo.myIperm = INT_T_ALLOC(nsupers + 1);
    if (forest->nodeList == NULL ||
        forest->topoInfo.eTreeTopLims == NULL ||
        forest->topoInfo.myIperm == NULL)
        ABORT("Malloc fails for SymFact V2 LDL forest arrays.");

    for (int_t k = 0; k < nsupers; ++k)
    {
        forest->nodeList[k] = trf3Dpart->symV2FactorNodes[k];
        forest->topoInfo.myIperm[k] = trf3Dpart->symV2NodeIperm[k];
    }
    forest->topoInfo.myIperm[nsupers] = nsupers;
    for (int_t lvl = 0; lvl <= trf3Dpart->symV2FactorLevelCount; ++lvl)
        forest->topoInfo.eTreeTopLims[lvl] =
            trf3Dpart->symV2FactorLevelPtr[lvl];

    forest->weight = 0.0;
    for (int_t k = 0; k < nsupers; ++k)
        forest->weight += treeList[forest->nodeList[k]].weight;
    forest->cost = 0.0;
    return forest;
}

static void dSymV2InstallLDLForest(int_t nsupers,
                                   dtrf3Dpartition_t *trf3Dpart,
                                   int_t *setree,
                                   treeList_t *treeList,
                                   int_t *xsup,
                                   gridinfo3d_t *grid3d)
{
    int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
    sForest_t **sForests = getGreedyLoadBalForests(maxLvl, nsupers,
                                                    setree, treeList);
    int_t *myTreeIdxs = getGridTrees(grid3d);
    int_t *myZeroTrIdxs = getReplicatedTrees(grid3d);
    int_t *gNodeCount = getNodeCountsFr(maxLvl, sForests);
    int_t **gNodeLists = getNodeListFr(maxLvl, sForests);
    int_t *myNodeCount = getMyNodeCountsFr(maxLvl, myTreeIdxs, sForests);
    int_t **treePerm = getTreePermFr(myTreeIdxs, sForests, grid3d);
    int_t *supernode2treeMap =
        createSupernode2TreeMap(nsupers, maxLvl, gNodeCount, gNodeLists);
    int *supernodeMask = int32Calloc_dist(nsupers);
    SupernodeToGridMap_t *superGridMap =
        createSuperGridMap(nsupers, maxLvl, myTreeIdxs, myZeroTrIdxs,
                           gNodeCount, gNodeLists);

    if (sForests == NULL || myTreeIdxs == NULL || myZeroTrIdxs == NULL ||
        gNodeCount == NULL || gNodeLists == NULL || myNodeCount == NULL ||
        treePerm == NULL || supernode2treeMap == NULL ||
        supernodeMask == NULL || superGridMap == NULL)
        ABORT("Malloc fails for SymFact V2 LDL forest metadata.");

    for (int_t k = 0; k < nsupers; ++k)
    {
        supernodeMask[k] = 0;
    }
    for (int_t lvl = 0; lvl < maxLvl; ++lvl)
    {
        for (int_t nd = 0; nd < myNodeCount[lvl]; ++nd)
        {
            supernodeMask[treePerm[lvl][nd]] = 1;
        }
    }

    trf3Dpart->sForests = sForests;
    trf3Dpart->myTreeIdxs = myTreeIdxs;
    trf3Dpart->myZeroTrIdxs = myZeroTrIdxs;
    trf3Dpart->myNodeCount = myNodeCount;
    trf3Dpart->treePerm = treePerm;
    trf3Dpart->supernode2treeMap = supernode2treeMap;
    trf3Dpart->supernodeMask = supernodeMask;
    trf3Dpart->superGridMap = superGridMap;
    trf3Dpart->maxLvl = maxLvl;
    trf3Dpart->symV2ScheduleEnabled = 1;

    dSymV2ValidateLDLForests(nsupers, trf3Dpart, gNodeCount, gNodeLists);
    dSymV2BuildLocalLDLIndexes(nsupers, trf3Dpart, grid3d);
    dSymV2UpdateLDLDiagOwners(nsupers, trf3Dpart, grid3d);
    dSymV2ComputeForestDiagDims(nsupers, trf3Dpart, xsup, grid3d);
    dSymV2TraceLDLForests(trf3Dpart, gNodeCount, grid3d);

    SUPERLU_FREE(gNodeCount);
    SUPERLU_FREE(gNodeLists);
}

static void dTrfPartitionInitImpl(int_t nsupers,  dLUstruct_t *LUstruct,
                                  gridinfo3d_t *grid3d,
                                  int use_sym_v2_weights,
                                  Glu_freeable_t *Glu_freeable)
{

    gridinfo_t* grid = &(grid3d->grid2d);
    int iam = grid3d->iam;
#if ( DEBUGlevel>=1 )
    CHECK_MALLOC (iam, "Enter dnewTrfPartitionInit()");
#endif

    // check parameters
    if (LUstruct == NULL || LUstruct->trf3Dpart == NULL || grid3d == NULL)
    {
        fprintf(stderr, "Error: Invalid arguments to dnewTrfPartitionInit().\n");
        return;
    }

      // Calculation of supernodal etree
    int_t *setree = supernodal_etree(nsupers, LUstruct->etree, LUstruct->Glu_persist->supno, LUstruct->Glu_persist->xsup);

    // Conversion of supernodal etree to list
    treeList_t *treeList = setree2list(nsupers, setree);

// YL: The essential difference between this function and dinitTrf3Dpartition_allgrid to avoid calling pddistribute* twice is that Piyush has removed the treelist weight update function below (and iperm_c_supno as well), which requires the LU data structure
#if 0
    /*update treelist with weight and depth*/
    getSCUweight_allgrid(nsupers, treeList, xsup,
        LUstruct->Llu->Lrowind_bc_ptr, LUstruct->Llu->Ufstnz_br_ptr,
        grid3d);
#endif
    // Calculation of tree weight
    calcTreeWeight(nsupers, setree, treeList, LUstruct->Glu_persist->xsup);
    if (use_sym_v2_weights)
    {
        dSymV2CalcLDLTreeWeight(nsupers, setree, treeList,
                                LUstruct->Glu_persist->xsup,
                                Glu_freeable,
                                NULL, grid3d);
    }

    // Calculation of maximum level
    int_t maxLvl = log2i(grid3d->zscp.Np) + 1;

    // Generation of forests
    sForest_t **sForests = getForests(maxLvl, nsupers, setree, treeList);

    dtrf3Dpartition_t *trf3Dpart = LUstruct->trf3Dpart;
    trf3Dpart->sForests = sForests;
    trf3Dpart->nsupers = nsupers;
    trf3Dpart->symV2DiagOwner = NULL;
    trf3Dpart->symV2PanelRoot = NULL;
    trf3Dpart->symV2DiagRoot = NULL;
    trf3Dpart->symV2PanelLocalIndex = NULL;
    trf3Dpart->symV2RowLocalIndex = NULL;
    trf3Dpart->symV2LocalPanelGids = NULL;
    trf3Dpart->symV2LocalRowGids = NULL;
    trf3Dpart->symV2LocalPanelCount = 0;
    trf3Dpart->symV2LocalRowCount = 0;
    dSymV2ResetLDLMetadata(trf3Dpart);
    if (use_sym_v2_weights)
        dSymV2InitLDLOwners(nsupers, trf3Dpart,
                            LUstruct->Glu_persist->xsup,
                            Glu_freeable, grid3d);
      int_t *myTreeIdxs = getGridTrees(grid3d);
    int_t *myZeroTrIdxs = getReplicatedTrees(grid3d);
    int_t *gNodeCount = getNodeCountsFr(maxLvl, sForests);
    int_t **gNodeLists = getNodeListFr(maxLvl, sForests); // reuse NodeLists stored in sForests[]

    // dinit3DLUstructForest(myTreeIdxs, myZeroTrIdxs,
    //                       sForests, LUstruct, grid3d);
    int_t *myNodeCount = getMyNodeCountsFr(maxLvl, myTreeIdxs, sForests);
    int_t **treePerm = getTreePermFr(myTreeIdxs, sForests, grid3d);
    int* supernodeMask = SUPERLU_MALLOC(nsupers*sizeof(int));
    for (int ii = 0; ii < nsupers; ++ii)
        supernodeMask[ii]=0;
    for (int lvl = 0; lvl < maxLvl; ++lvl)
    {
        // printf("iam %5d lvl %5d myNodeCount[lvl] %5d\n",grid3d->iam, lvl,myNodeCount[lvl]);
        for (int nd = 0; nd < myNodeCount[lvl]; ++nd)
        {
            supernodeMask[treePerm[lvl][nd]]=1;
        }
    }





    // dLUValSubBuf_t *LUvsb = SUPERLU_MALLOC(sizeof(dLUValSubBuf_t));
    // dLluBufInit(LUvsb, LUstruct);

#if (DEBUGlevel>=1)
    // let count sum of gnodecount
    int_t gNodeCountSum = 0;
    for (int_t i = 0; i < (1 << maxLvl) - 1; ++i)
    {
        gNodeCountSum += gNodeCount[i];
    }
    printf(" Iam: %d, Nsupers %d, gnodecountSum =%d \n", grid3d->iam, nsupers, gNodeCountSum);
#endif

    /* Sherry 2/17/23
       Compute buffer sizes needed for diagonal LU blocks and C matrices in GEMM. */


    iam = grid->iam;  /* 'grid' is 2D grid */
    int k, k0, k_st, k_end, offset, nsupc, krow, kcol;
    int myrow = MYROW (iam, grid);
    int mycol = MYCOL (iam, grid);
    int_t *xsup  = LUstruct->Glu_persist->xsup;

#if 0
    int krow = PROW (k, grid);
    int kcol = PCOL (k, grid);
    int_t** Lrowind_bc_ptr = Llu->Lrowind_bc_ptr;
    double** Lnzval_bc_ptr = Llu->Lnzval_bc_ptr;

    int_t** Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;
    double** Unzval_br_ptr = Llu->Unzval_br_ptr;
#endif

    int mxLeafNode = 0; // Yang: only need to check the leaf level of topoInfo as the factorization proceeds level by level
    for (int ilvl = 0; ilvl < maxLvl; ++ilvl) {
        if (sForests[myTreeIdxs[ilvl]] && sForests[myTreeIdxs[ilvl]]->topoInfo.eTreeTopLims[1] > mxLeafNode )
            mxLeafNode    = sForests[myTreeIdxs[ilvl]]->topoInfo.eTreeTopLims[1];
    }

    // Yang: use ldts to track the maximum needed buffer sizes per node of topoInfo
    //int *ldts = (int*) SUPERLU_MALLOC(mxLeafNode*sizeof(int));
    //for (int i = 0; i < mxLeafNode; ++i) {  //????????
    //ldts[i]=1;
    //}
    int *ldts = int32Calloc_dist(mxLeafNode);

    for (int ilvl = 0; ilvl < maxLvl; ++ilvl) {  /* Loop through the Pz tree levels */
        int treeId = myTreeIdxs[ilvl];
        sForest_t* sforest = sForests[treeId];
        if (sforest){
            int_t *perm_node = sforest->nodeList ; /* permuted list, in order of factorization */
	    int maxTopoLevel = sforest->topoInfo.numLvl;/* number of levels at each outer-tree node */
            for (int topoLvl = 0; topoLvl < maxTopoLevel; ++topoLvl)
            {
                /* code */
                k_st = sforest->topoInfo.eTreeTopLims[topoLvl];
                k_end = sforest->topoInfo.eTreeTopLims[topoLvl + 1];
		//printf("\t..topoLvl %d, k_st %d, k_end %d\n", topoLvl, k_st, k_end);

                for (int k0 = k_st; k0 < k_end; ++k0)
                {
                    offset = k0 - k_st;
                    k = perm_node[k0];
                    nsupc = (xsup[k+1]-xsup[k]);
                    krow = use_sym_v2_weights
                               ? trf3Dpart->symV2DiagRoot[k]
                               : PROW(k, grid);
                    kcol = use_sym_v2_weights
                               ? trf3Dpart->symV2PanelRoot[k]
                               : PCOL(k, grid);
                    if (use_sym_v2_weights ? (mycol == kcol)
                                            : (myrow == krow || mycol == kcol))
                    {
		        ldts[offset] = SUPERLU_MAX(ldts[offset], nsupc);
                    }
#if 0 /* GPU gemm buffers can only be set on GPU side, because here we only know
	 the size of U data structure on CPU.  It is different on GPU */
                    if ( mycol == kcol ) { /* processes owning L panel */

		    }
                    if ( myrow == krow )
			gemmCsizes[offset] = SUPERLU_MAX(ldts[offset], ???);
#endif
                }
            }
        }
    }




    trf3Dpart->gEtreeInfo = fillEtreeInfo(nsupers, setree, treeList);
    // trf3Dpart->iperm_c_supno = iperm_c_supno;
    trf3Dpart->myNodeCount = myNodeCount;
    trf3Dpart->myTreeIdxs = myTreeIdxs;
    trf3Dpart->myZeroTrIdxs = myZeroTrIdxs;
    trf3Dpart->sForests = sForests;
    trf3Dpart->treePerm = treePerm;
    trf3Dpart->maxLvl = maxLvl;
    // trf3Dpart->LUvsb = LUvsb;
    trf3Dpart->supernode2treeMap = createSupernode2TreeMap(nsupers, maxLvl, gNodeCount, gNodeLists);
    trf3Dpart->superGridMap = createSuperGridMap(nsupers, maxLvl, myTreeIdxs, myZeroTrIdxs, gNodeCount, gNodeLists);
    trf3Dpart->supernodeMask = supernodeMask;
    trf3Dpart->mxLeafNode = mxLeafNode;  // Sherry added these 3
    trf3Dpart->diagDims = ldts;
    //trf3Dpart->gemmCsizes = gemmCsizes;
    // Sherry added
    // Deallocate storage
    SUPERLU_FREE(gNodeCount);
    SUPERLU_FREE(gNodeLists);
    free_treelist(nsupers, treeList);
#if ( DEBUGlevel>=1 )
    CHECK_MALLOC (iam, "Exit dnewTrfPartitionInit()");
#endif

}

void dnewTrfPartitionInit(int_t nsupers,  dLUstruct_t *LUstruct,
                          gridinfo3d_t *grid3d)
{
    dTrfPartitionInitImpl(nsupers, LUstruct, grid3d, 0, NULL);
}

void dSymV2TrfPartitionInit(int_t nsupers,  dLUstruct_t *LUstruct,
                            Glu_freeable_t *Glu_freeable,
                            gridinfo3d_t *grid3d,
                            superlu_dist_options_t *options)
{
    dtrf3Dpartition_t *trf3Dpart;
    int_t *setree;
    treeList_t *treeList;

    if (options == NULL || options->SymFact != YES)
        ABORT("dSymV2TrfPartitionInit requires SymFact=YES.");
    if (LUstruct == NULL || LUstruct->trf3Dpart == NULL ||
        LUstruct->Glu_persist == NULL || grid3d == NULL)
        ABORT("dSymV2TrfPartitionInit received invalid arguments.");

    trf3Dpart = LUstruct->trf3Dpart;
    trf3Dpart->nsupers = nsupers;
    trf3Dpart->iperm_c_supno = NULL;
    trf3Dpart->myNodeCount = NULL;
    trf3Dpart->myTreeIdxs = NULL;
    trf3Dpart->myZeroTrIdxs = NULL;
    trf3Dpart->treePerm = NULL;
    trf3Dpart->sForests = NULL;
    trf3Dpart->supernode2treeMap = NULL;
    trf3Dpart->LUvsb = NULL;
    trf3Dpart->gemmCsizes = NULL;
    dSymV2ResetLDLMetadata(trf3Dpart);

    setree = supernodal_etree(nsupers, LUstruct->etree,
                              LUstruct->Glu_persist->supno,
                              LUstruct->Glu_persist->xsup);
    treeList = setree2list(nsupers, setree);
    dSymV2CalcLDLTreeWeight(nsupers, setree, treeList,
                            LUstruct->Glu_persist->xsup,
                            Glu_freeable, NULL, grid3d);
    trf3Dpart->gEtreeInfo = fillEtreeInfo(nsupers, setree, treeList);
    dSymV2InitLDLOwners(nsupers, trf3Dpart,
                        LUstruct->Glu_persist->xsup,
                        Glu_freeable, grid3d);
    dSymV2BuildLDLSchedule(nsupers, setree, trf3Dpart,
                           LUstruct->Glu_persist->xsup, grid3d);

    dSymV2InstallLDLForest(nsupers, trf3Dpart, setree, treeList,
                           LUstruct->Glu_persist->xsup, grid3d);

    free_treelist(nsupers, treeList);
}


// function to broad permuted sparse matrix and symbolic factorization data from
// 2d to 3d grid

void dbcastPermutedSparseA(SuperMatrix *A,
                          dScalePermstruct_t *ScalePermstruct,
                          Glu_freeable_t *Glu_freeable,
                          dLUstruct_t *LUstruct, gridinfo3d_t *grid3d)
{
    int_t m = A->nrow;
	int_t n = A->ncol;
    Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    dLocalLU_t *Llu = LUstruct->Llu;
    NRformat_loc *Astore   = (NRformat_loc *) A->Store;
    // check if the varaibles are not NULL
    if (A == NULL || ScalePermstruct == NULL ||
        Glu_freeable == NULL || LUstruct == NULL  || grid3d == NULL ||
        Glu_persist == NULL || Llu == NULL || Astore == NULL)
    {
        fprintf(stderr, "Error: Invalid arguments to dbcastPermutedSparseA().\n");
        return;
    }

    /* broadcast etree */
    int_t *etree = LUstruct->etree;
    if(etree)
        MPI_Bcast( etree, n, mpi_int_t, 0,  grid3d->zscp.comm);

    // list of all the arrays to be broadcasted
    // A, ScalePermstruct, Glu_freeable, LUstruct
    int_t nsupers;

    if (!grid3d->zscp.Iam)
        nsupers = Glu_persist->supno[n-1] + 1;
    // broadcast the number of supernodes
    MPI_Bcast(&nsupers, 1, mpi_int_t, 0, grid3d->zscp.comm);

    /* ==== Broadcasting GLU_persist   ======= */
    // what is the size of xsup and supno?
    allocBcastArray( (void **) &(Glu_persist->xsup), (nsupers+1)*sizeof(int_t),
        0, grid3d->zscp.comm);
    allocBcastArray( (void **) &(Glu_persist->supno), (n)*sizeof(int_t),
        0, grid3d->zscp.comm);
    int_t *xsup = Glu_persist->xsup;    /* supernode and column mapping */
    int_t *supno = Glu_persist->supno;


    /* ==== Broadcasting ScalePermstruct ======= */
//     typedef struct {
//     DiagScale_t DiagScale; // enum 1
//     double *R;  (double*) dimension (A->nrow)
//     double *C;   (double*) dimension (A->ncol)
//     int_t  *perm_r; (int_t*) dimension (A->nrow)
//     int_t  *perm_c; (int_t*) dimension (A->ncol)
// } dScalePermstruct_t;

    MPI_Bcast(&(ScalePermstruct->DiagScale), sizeof(DiagScale_t), MPI_BYTE, 0, grid3d->zscp.comm);

/***** YL: remove the allocation in the following as perm_r/perm_c has been allocated on all grids by dScalePermstructInit
*/
#if 1
    MPI_Bcast(ScalePermstruct->perm_r, m*sizeof(int_t), MPI_BYTE, 0, grid3d->zscp.comm);
    MPI_Bcast(ScalePermstruct->perm_c, n*sizeof(int_t), MPI_BYTE, 0, grid3d->zscp.comm);
#else
    allocBcastArray ( &(ScalePermstruct->perm_r), m*sizeof(int_t),
        0, grid3d->zscp.comm);
    allocBcastArray ( &(ScalePermstruct->perm_c), n*sizeof(int_t),
        0, grid3d->zscp.comm);
#endif
    if(ScalePermstruct->DiagScale==ROW || ScalePermstruct->DiagScale==BOTH)
    allocBcastArray ( (void **) &(ScalePermstruct->R), m*sizeof(double),
        0, grid3d->zscp.comm);
    if(ScalePermstruct->DiagScale==COL || ScalePermstruct->DiagScale==BOTH)
    allocBcastArray ( (void **) &(ScalePermstruct->C), n*sizeof(double),
        0, grid3d->zscp.comm);


    /* ==== Broadcasting Glu_freeable ======= */
//     typedef struct {
//     int_t     *lsub;     /* compressed L subscripts */
//     int_t     *xlsub;        // i think its size is n+1
//     int_t     *usub;     /* compressed U subscripts i think nzumax*/
//     int_t     *xusub;
//     int_t     nzlmax;    /* current max size of lsub */
//     int_t     nzumax;    /*    "    "    "      usub */
//     LU_space_t MemModel; /* 0 - system malloc'd; 1 - user provided */
//     //int_t     *llvl;     /* keep track of level in L for level-based ILU */
//     //int_t     *ulvl;     /* keep track of level in U for level-based ILU */
//     int64_t nnzLU;   /* number of nonzeros in L+U*/
// } Glu_freeable_t;

    allocBcastLargeArray( (void **) &(Glu_freeable->lsub), Glu_freeable->nzlmax*sizeof(int_t),
        0, grid3d->zscp.comm);
    allocBcastArray( (void **) &(Glu_freeable->xlsub), (n+1)*sizeof(int_t),
        0, grid3d->zscp.comm);
    allocBcastLargeArray( (void **) &(Glu_freeable->usub), Glu_freeable->nzumax*sizeof(int_t),
        0, grid3d->zscp.comm);
    allocBcastArray( (void **) &(Glu_freeable->xusub), (n+1)*sizeof(int_t),
        0, grid3d->zscp.comm);
    MPI_Bcast(&(Glu_freeable->nzlmax), sizeof(int_t), MPI_BYTE, 0, grid3d->zscp.comm);
    MPI_Bcast(&(Glu_freeable->nzumax), sizeof(int_t), MPI_BYTE, 0, grid3d->zscp.comm);
    MPI_Bcast(&(Glu_freeable->nnzLU), sizeof(int64_t), MPI_BYTE, 0, grid3d->zscp.comm);

    if(grid3d->zscp.Iam)
    {
        Glu_freeable->MemModel = SYSTEM;
    }


    /* ==== Broadcasting permuted sparse matrix ======= */
    // Astore = (NRformat_loc *) A->Store;
//     /typedef struct {
//     int_t nnz_loc;   /* number of nonzeros in the local submatrix */
//     int_t m_loc;     /* number of rows local to this processor */
//     int_t fst_row;   /* global index of the first row */
//     void  *nzval;    /* pointer to array of nonzero values, packed by row */
//     int_t *rowptr;   /* pointer to array of beginning of rows in nzval[]
// 			and colind[]  */
//     int_t *colind;   /* pointer to array of column indices of the nonzeros */
//                      /* Note:
// 			Zero-based indexing is used;
// 			rowptr[] has n_loc + 1 entries, the last one pointing
// 			beyond the last row, so that rowptr[n_loc] = nnz_loc.*/
// } NRformat_loc;


    // NRformat_loc *Astore = (NRformat_loc *) A->Store;
    MPI_Bcast(&(Astore->nnz_loc), sizeof(int_t), MPI_BYTE, 0, grid3d->zscp.comm);
    MPI_Bcast(&(Astore->m_loc), sizeof(int_t), MPI_BYTE, 0, grid3d->zscp.comm);
    MPI_Bcast(&(Astore->fst_row), sizeof(int_t), MPI_BYTE, 0, grid3d->zscp.comm);


/***** YL: remove the allocation in the following as dGatherNRformat_loc3d_allgrid instead of dGatherNRformat_loc3d has been called, which already allocate A->Store on all grids
 * Note the the broadcast is still needed as the A->Store has been scaled by dscaleMatrixDiagonally only on grid 0
*/
#if 1
    MPI_Bcast(Astore->nzval, Astore->nnz_loc*sizeof(double), MPI_BYTE, 0, grid3d->zscp.comm);
    MPI_Bcast(Astore->rowptr, (Astore->m_loc+1)*sizeof(int_t), MPI_BYTE, 0, grid3d->zscp.comm);
    MPI_Bcast(Astore->colind, Astore->nnz_loc*sizeof(int_t), MPI_BYTE, 0, grid3d->zscp.comm);
#else
    allocBcastArray( (void **) &(Astore->nzval), Astore->nnz_loc*sizeof(double),
        0, grid3d->zscp.comm);
    allocBcastArray( (void **) &(Astore->rowptr), (Astore->m_loc+1)*sizeof(int_t),
        0, grid3d->zscp.comm);
    allocBcastArray( (void **) &(Astore->colind), Astore->nnz_loc*sizeof(int_t),
        0, grid3d->zscp.comm);
#endif

}
