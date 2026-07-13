

#include "superlu_ddefs.h"
#include <string.h>
#include <stdlib.h>

extern int_t calcTopInfoForest(sForest_t *forest, int_t nsupers,
                               int_t *setree);

typedef struct {
    double panel_work;
    double row_work;
    double rank_work;
    double comm_work;
    double tree_weight;
} dSymV2LDLCost_t;

static size_t dSymV2CheckedProduct(size_t left, size_t right,
                                   const char *message)
{
    if (left != 0 && right > ((size_t)-1) / left)
        ABORT(message);
    return left * right;
}

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
    double rank_schur_fraction =
        1.0 / (double) SUPERLU_MAX(1, nprow);

    cost->panel_work =
        SUPERLU_MAX(1.0, panel_factor_cost + ll_schur_cost +
                         partner_comm_cost);
    cost->row_work =
        SUPERLU_MAX(1.0, diag_cost + solve_cost + partner_comm_cost);
    cost->rank_work =
        SUPERLU_MAX(1.0, diag_cost + panel_factor_cost + solve_cost +
                         rank_schur_fraction * ll_schur_cost +
                         partner_comm_cost);
    cost->comm_work = partner_comm_cost;
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

static double dSymV2OwnerAffinityWeight(void)
{
    const char *env = getenv("GPU3DV2_OWNER_AFFINITY");
    if (env == NULL || env[0] == '\0')
        return 0.0;
    char *end = NULL;
    double value = strtod(env, &end);
    if (end == env || *end != '\0' || value < 0.0)
        ABORT("GPU3DV2_OWNER_AFFINITY must be a nonnegative number.");
    return value;
}

static void dSymV2ChooseOwnerPair(const double *panel_load,
                                  const double *row_load,
                                  const double *rank_load,
                                  gridinfo_t *grid,
                                  const dSymV2LDLCost_t *cost,
                                  int parent_pr, int parent_pc,
                                  double affinity_weight,
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
            double affinity_penalty = 0.0;
            if (affinity_weight > 0.0 && parent_pr >= 0 && parent_pc >= 0)
            {
                if (pc != parent_pc)
                    affinity_penalty += cost->comm_work;
                if (pr != parent_pr)
                    affinity_penalty += 0.5 * cost->comm_work;
            }
            double score = max_projected +
                           0.05 * (projected_panel + projected_row +
                                   projected_rank) +
                           affinity_weight * affinity_penalty;
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
                                int_t *setree,
                                int_t *xsup,
                                Glu_freeable_t *Glu_freeable,
                                const int *active_z,
                                gridinfo3d_t *grid3d)
{
    gridinfo_t *grid = &(grid3d->grid2d);
    const int pz = grid3d->zscp.Np;
    const size_t ranks_2d = dSymV2CheckedProduct(
        (size_t) grid->nprow, (size_t) grid->npcol,
        "SymFact V2 LDL 2D owner count overflows.");
    const size_t panel_count = dSymV2CheckedProduct(
        (size_t) pz, (size_t) grid->npcol,
        "SymFact V2 LDL panel load count overflows.");
    const size_t row_count = dSymV2CheckedProduct(
        (size_t) pz, (size_t) grid->nprow,
        "SymFact V2 LDL row load count overflows.");
    const size_t rank_count = dSymV2CheckedProduct(
        (size_t) pz, ranks_2d,
        "SymFact V2 LDL rank load count overflows.");
    const size_t owner_bytes = dSymV2CheckedProduct(
        (size_t) nsupers, sizeof(int),
        "SymFact V2 LDL owner metadata size overflows.");
    const size_t panel_bytes = dSymV2CheckedProduct(
        panel_count, sizeof(double),
        "SymFact V2 LDL panel load size overflows.");
    const size_t row_bytes = dSymV2CheckedProduct(
        row_count, sizeof(double),
        "SymFact V2 LDL row load size overflows.");
    const size_t rank_bytes = dSymV2CheckedProduct(
        rank_count, sizeof(double),
        "SymFact V2 LDL rank load size overflows.");
    double *panel_load = (double *) SUPERLU_MALLOC(panel_bytes);
    double *row_load = (double *) SUPERLU_MALLOC(row_bytes);
    double *rank_load = (double *) SUPERLU_MALLOC(rank_bytes);

    if (active_z == NULL || panel_load == NULL || row_load == NULL ||
        rank_load == NULL)
        ABORT("Malloc fails for SymFact V2 LDL load metadata.");
    memset(panel_load, 0, panel_bytes);
    memset(row_load, 0, row_bytes);
    memset(rank_load, 0, rank_bytes);

    if (!(trf3Dpart->symV2DiagOwner = (int *) SUPERLU_MALLOC(owner_bytes)) ||
        !(trf3Dpart->symV2PanelRoot = (int *) SUPERLU_MALLOC(owner_bytes)) ||
        !(trf3Dpart->symV2DiagRoot = (int *) SUPERLU_MALLOC(owner_bytes)) ||
        !(trf3Dpart->symV2PanelLocalIndex = INT_T_ALLOC(nsupers)) ||
        !(trf3Dpart->symV2RowLocalIndex = INT_T_ALLOC(nsupers)))
        ABORT("Malloc fails for SymFact V2 LDL owner metadata.");

    const double affinity_weight = dSymV2OwnerAffinityWeight();
    for (int_t k = 0; k < nsupers; ++k)
    {
        trf3Dpart->symV2PanelRoot[k] = -1;
        trf3Dpart->symV2DiagRoot[k] = -1;
        trf3Dpart->symV2PanelLocalIndex[k] = -1;
        trf3Dpart->symV2RowLocalIndex[k] = -1;
    }
    for (int_t order = 0; order < nsupers; ++order)
    {
        const int_t k = affinity_weight > 0.0
                            ? nsupers - 1 - order
                            : order;
        dSymV2LDLCost_t cost;
        int panel_root;
        int diag_root;
        int owner_rank;
        const int z = active_z[k];
        if (z < 0 || z >= pz)
            ABORT("SymFact V2 LDL owner has an invalid active Z layer.");
        const int_t parent = setree != NULL ? setree[k] : nsupers;
        const int parent_pr =
            (parent >= 0 && parent < nsupers && active_z[parent] == z)
                ? trf3Dpart->symV2DiagRoot[parent]
                : -1;
        const int parent_pc =
            (parent >= 0 && parent < nsupers && active_z[parent] == z)
                ? trf3Dpart->symV2PanelRoot[parent]
                : -1;
        const size_t panel_offset = (size_t) z * (size_t) grid->npcol;
        const size_t row_offset = (size_t) z * (size_t) grid->nprow;
        const size_t rank_offset = (size_t) z * ranks_2d;
        dSymV2EstimateSupernodeWork(k, xsup, Glu_freeable, grid, &cost);
        dSymV2ChooseOwnerPair(panel_load + panel_offset,
                              row_load + row_offset,
                              rank_load + rank_offset, grid, &cost,
                              parent_pr, parent_pc, affinity_weight,
                              &diag_root, &panel_root);
        owner_rank = PNUM(diag_root, panel_root, grid);
        panel_load[panel_offset + (size_t) panel_root] += cost.panel_work;
        row_load[row_offset + (size_t) diag_root] += cost.row_work;
        rank_load[rank_offset + (size_t) owner_rank] += cost.rank_work;
        trf3Dpart->symV2PanelRoot[k] = panel_root;
        trf3Dpart->symV2DiagRoot[k] = diag_root;
    }

    const char *trace = getenv("GPU3DV2_TRACE");
    if (grid3d->iam == 0 && trace != NULL && trace[0] != '\0' &&
        trace[0] != '0')
    {
        double total = 0.0;
        double maximum = 0.0;
        for (size_t rank = 0; rank < rank_count; ++rank)
        {
            total += rank_load[rank];
            maximum = SUPERLU_MAX(maximum, rank_load[rank]);
        }
        const double average = rank_count > 0
                                   ? total / (double) rank_count
                                   : 0.0;
        const double imbalance = average > 0.0 ? maximum / average : 0.0;
        fprintf(stderr,
                "[sym-v2-trace] LDL physical owner load ranks=%llu "
                "total=%.6e average=%.6e maximum=%.6e imbalance=%.6f\n",
                (unsigned long long) rank_count, total, average, maximum,
                imbalance);
        fflush(stderr);
    }

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
    int *local_owner_count;
    int *owner_count;

    if (trf3Dpart->symV2DiagOwner == NULL ||
        trf3Dpart->symV2PanelRoot == NULL ||
        trf3Dpart->symV2DiagRoot == NULL ||
        trf3Dpart->superGridMap == NULL)
        ABORT("SymFact V2 LDL owner metadata is not initialized.");

    local_owner = (int *) SUPERLU_MALLOC((size_t) nsupers * sizeof(int));
    local_owner_count = int32Calloc_dist((int) nsupers);
    owner_count = int32Calloc_dist((int) nsupers);
    if (local_owner == NULL || local_owner_count == NULL ||
        owner_count == NULL)
        ABORT("Malloc fails for SymFact V2 diagonal owner workspace.");

    MPI_Comm_rank(grid3d->comm, &global_rank);
    for (int_t k = 0; k < nsupers; ++k)
    {
        int owner_2d = PNUM(trf3Dpart->symV2DiagRoot[k],
                            trf3Dpart->symV2PanelRoot[k], grid);
        int owns_diag = trf3Dpart->superGridMap[k] == IN_GRID_AIJ &&
                        grid->iam == owner_2d;
        local_owner[k] = owns_diag ? global_rank : INT_MAX;
        local_owner_count[k] = owns_diag ? 1 : 0;
    }

    MPI_Allreduce(local_owner, trf3Dpart->symV2DiagOwner, (int) nsupers,
                  MPI_INT, MPI_MIN, grid3d->comm);
    MPI_Allreduce(local_owner_count, owner_count, (int) nsupers,
                  MPI_INT, MPI_SUM, grid3d->comm);
    for (int_t k = 0; k < nsupers; ++k)
    {
        if (trf3Dpart->symV2DiagOwner[k] == INT_MAX)
            ABORT("SymFact V2 LDL owner metadata is missing a diagonal owner.");
        if (owner_count[k] != 1)
            ABORT("SymFact V2 LDL owner metadata has an invalid diagonal owner count.");
    }

    SUPERLU_FREE(local_owner);
    SUPERLU_FREE(local_owner_count);
    SUPERLU_FREE(owner_count);
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

static int *dSymV2BuildActiveZMap(int_t nsupers, int_t maxLvl,
                                  int_t *gNodeCount, int_t **gNodeLists,
                                  gridinfo3d_t *grid3d)
{
    const int pz = grid3d->zscp.Np;
    const int_t numForests = (((int_t) 1) << maxLvl) - 1;
    const size_t forest_bytes = dSymV2CheckedProduct(
        (size_t) numForests, sizeof(int),
        "SymFact V2 LDL forest owner size overflows.");
    const size_t active_bytes = dSymV2CheckedProduct(
        (size_t) nsupers, sizeof(int),
        "SymFact V2 LDL active Z map size overflows.");
    int *forest_owner_z = (int *) SUPERLU_MALLOC(forest_bytes);
    int *active_z = (int *) SUPERLU_MALLOC(active_bytes);

    if (forest_owner_z == NULL || active_z == NULL)
        ABORT("Malloc fails for SymFact V2 LDL active Z metadata.");
    for (int_t tree = 0; tree < numForests; ++tree)
        forest_owner_z[tree] = -1;
    for (int_t k = 0; k < nsupers; ++k)
        active_z[k] = -1;

    for (int z = 0; z < pz; ++z)
    {
        int_t tree = (int_t) pz - 1 + (int_t) z;
        for (int_t level = 0; level < maxLvl; ++level)
        {
            if (tree < 0 || tree >= numForests)
                ABORT("SymFact V2 LDL forest has an invalid Z mapping.");
            const int_t stride = ((int_t) 1) << level;
            if (((int_t) z % stride) == 0)
            {
                if (forest_owner_z[tree] >= 0 &&
                    forest_owner_z[tree] != z)
                    ABORT("SymFact V2 LDL forest has multiple active Z owners.");
                forest_owner_z[tree] = z;
            }
            tree = (tree - 1) / 2;
        }
    }

    for (int_t tree = 0; tree < numForests; ++tree)
    {
        const int z = forest_owner_z[tree];
        if (z < 0 || z >= pz)
            ABORT("SymFact V2 LDL forest is missing an active Z owner.");
        for (int_t node = 0; node < gNodeCount[tree]; ++node)
        {
            const int_t k = gNodeLists[tree][node];
            if (k < 0 || k >= nsupers)
                ABORT("SymFact V2 LDL active Z map has an invalid supernode.");
            if (active_z[k] >= 0)
                ABORT("SymFact V2 LDL supernode has multiple active Z owners.");
            active_z[k] = z;
        }
    }
    for (int_t k = 0; k < nsupers; ++k)
        if (active_z[k] < 0)
            ABORT("SymFact V2 LDL supernode is missing an active Z owner.");

    SUPERLU_FREE(forest_owner_z);
    return active_z;
}

static void dSymV2ValidateActiveZMap(int_t nsupers, const int *active_z,
                                     dtrf3Dpartition_t *trf3Dpart,
                                     gridinfo3d_t *grid3d)
{
    if (active_z == NULL || trf3Dpart->superGridMap == NULL)
        ABORT("SymFact V2 LDL active Z validation metadata is missing.");

    const int my_z = grid3d->zscp.Iam;
    for (int_t k = 0; k < nsupers; ++k)
    {
        const int active_here = active_z[k] == my_z;
        const int initialized_here =
            trf3Dpart->superGridMap[k] == IN_GRID_AIJ;
        if (active_here != initialized_here)
            ABORT("SymFact V2 LDL active Z map disagrees with the forest map.");
    }
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
                                   Glu_freeable_t *Glu_freeable,
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
    int *active_z = dSymV2BuildActiveZMap(
        nsupers, maxLvl, gNodeCount, gNodeLists, grid3d);
    dSymV2ValidateActiveZMap(nsupers, active_z, trf3Dpart, grid3d);
    dSymV2InitLDLOwners(nsupers, trf3Dpart, setree, xsup,
                        Glu_freeable, active_z, grid3d);
    dSymV2BuildLDLSchedule(nsupers, setree, trf3Dpart, xsup, grid3d);
    dSymV2BuildLocalLDLIndexes(nsupers, trf3Dpart, grid3d);
    dSymV2UpdateLDLDiagOwners(nsupers, trf3Dpart, grid3d);
    dSymV2ComputeForestDiagDims(nsupers, trf3Dpart, xsup, grid3d);
    dSymV2TraceLDLForests(trf3Dpart, gNodeCount, grid3d);

    SUPERLU_FREE(active_z);
    SUPERLU_FREE(gNodeCount);
    SUPERLU_FREE(gNodeLists);
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
    dSymV2InstallLDLForest(nsupers, trf3Dpart, setree, treeList,
                           LUstruct->Glu_persist->xsup,
                           Glu_freeable, grid3d);

    free_treelist(nsupers, treeList);
}
