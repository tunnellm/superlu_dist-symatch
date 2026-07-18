

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

typedef enum {
    DSYM_V2_OWNER_BLOCK_CYCLIC = 0,
    DSYM_V2_OWNER_GREEDY = 1,
    DSYM_V2_OWNER_PAPER_ID_ROW = 2
} dSymV2OwnerMapping_t;

typedef enum {
    DSYM_V2_OWNER_COST_LDL = 0,
    DSYM_V2_OWNER_COST_PAPER = 1
} dSymV2OwnerCost_t;

typedef struct {
    double *row_work;
    double *column_work;
    double *diagonal_work;
    int_t *id_order;
} dSymV2PaperWork_t;

typedef struct {
    int_t gid;
    int_t depth;
} dSymV2PaperOrderEntry_t;

static int dSymV2PartitionBoolFlag(const char *name, int default_value)
{
    const char *env = getenv(name);
    if (env == NULL || env[0] == '\0')
        return default_value;
    if (strcmp(env, "0") == 0)
        return 0;
    if (strcmp(env, "1") == 0)
        return 1;
    ABORT("SymFact V2 partition flags must be 0 or 1.");
    return default_value;
}

static int dSymV2GreedyMappingEnabled(void)
{
    return dSymV2PartitionBoolFlag("GPU3DV2_GREEDY_MAPPING", 1);
}

static dSymV2OwnerMapping_t dSymV2OwnerMapping(void)
{
    const char *env = getenv("GPU3DV2_OWNER_MAPPING");
    if (env == NULL || env[0] == '\0')
        return dSymV2GreedyMappingEnabled()
                   ? DSYM_V2_OWNER_GREEDY
                   : DSYM_V2_OWNER_BLOCK_CYCLIC;
    if (strcmp(env, "block-cyclic") == 0)
        return DSYM_V2_OWNER_BLOCK_CYCLIC;
    if (strcmp(env, "greedy") == 0)
        return DSYM_V2_OWNER_GREEDY;
    if (strcmp(env, "paper-id-row") == 0)
        return DSYM_V2_OWNER_PAPER_ID_ROW;
    ABORT("GPU3DV2_OWNER_MAPPING must be block-cyclic, greedy, or paper-id-row.");
    return DSYM_V2_OWNER_GREEDY;
}

static dSymV2OwnerCost_t dSymV2OwnerCost(void)
{
    const char *env = getenv("GPU3DV2_OWNER_COST");
    if (env == NULL || env[0] == '\0' || strcmp(env, "ldl") == 0)
        return DSYM_V2_OWNER_COST_LDL;
    if (strcmp(env, "paper") == 0)
        return DSYM_V2_OWNER_COST_PAPER;
    ABORT("GPU3DV2_OWNER_COST must be ldl or paper.");
    return DSYM_V2_OWNER_COST_LDL;
}

static const char *dSymV2OwnerMappingName(dSymV2OwnerMapping_t mapping)
{
    switch (mapping)
    {
        case DSYM_V2_OWNER_BLOCK_CYCLIC:
            return "block-cyclic";
        case DSYM_V2_OWNER_GREEDY:
            return "greedy";
        case DSYM_V2_OWNER_PAPER_ID_ROW:
            return "paper increasing-depth rows/cyclic columns";
    }
    return "unknown";
}

static const char *dSymV2OwnerCostName(dSymV2OwnerCost_t cost)
{
    return cost == DSYM_V2_OWNER_COST_PAPER ? "paper" : "LDL";
}

static int dSymV2LDLForestWeightsEnabled(void)
{
    return dSymV2PartitionBoolFlag("GPU3DV2_LDL_FOREST_WEIGHTS", 1);
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

static int dSymV2PaperOrderCompare(const void *left, const void *right)
{
    const dSymV2PaperOrderEntry_t *a =
        (const dSymV2PaperOrderEntry_t *) left;
    const dSymV2PaperOrderEntry_t *b =
        (const dSymV2PaperOrderEntry_t *) right;
    if (a->depth < b->depth)
        return -1;
    if (a->depth > b->depth)
        return 1;
    if (a->gid < b->gid)
        return -1;
    if (a->gid > b->gid)
        return 1;
    return 0;
}

static int dSymV2IntCompare(const void *left, const void *right)
{
    const int_t a = *(const int_t *) left;
    const int_t b = *(const int_t *) right;
    return (a > b) - (a < b);
}

static void dSymV2BuildPaperIDOrder(int_t nsupers, treeList_t *treeList,
                                    int_t *order)
{
    dSymV2PaperOrderEntry_t *entries =
        (dSymV2PaperOrderEntry_t *) SUPERLU_MALLOC(
            (size_t) nsupers * sizeof(dSymV2PaperOrderEntry_t));
    int_t *stack = INT_T_ALLOC(nsupers);
    int_t *depth = INT_T_ALLOC(nsupers);
    if (entries == NULL || stack == NULL || depth == NULL)
        ABORT("Malloc fails for SymFact V2 paper owner ordering.");

    for (int_t k = 0; k < nsupers; ++k)
        depth[k] = -1;

    int_t stack_size = 0;
    for (int_t r = 0; r < treeList[nsupers].numChild; ++r)
    {
        int_t root = treeList[nsupers].childrenList[r];
        depth[root] = 0;
        stack[stack_size++] = root;
    }
    while (stack_size > 0)
    {
        int_t parent = stack[--stack_size];
        for (int_t c = 0; c < treeList[parent].numChild; ++c)
        {
            int_t child = treeList[parent].childrenList[c];
            if (depth[child] >= 0)
                ABORT("SymFact V2 paper owner ordering found an invalid tree.");
            depth[child] = depth[parent] + 1;
            stack[stack_size++] = child;
        }
    }

    for (int_t k = 0; k < nsupers; ++k)
    {
        if (depth[k] < 0)
            ABORT("SymFact V2 paper owner ordering missed a supernode.");
        entries[k].gid = k;
        entries[k].depth = depth[k];
    }
    qsort(entries, (size_t) nsupers, sizeof(dSymV2PaperOrderEntry_t),
          dSymV2PaperOrderCompare);
    for (int_t k = 0; k < nsupers; ++k)
        order[k] = entries[k].gid;

    SUPERLU_FREE(depth);
    SUPERLU_FREE(stack);
    SUPERLU_FREE(entries);
}

static void dSymV2FreePaperWork(dSymV2PaperWork_t *paper)
{
    SUPERLU_FREE(paper->row_work);
    SUPERLU_FREE(paper->column_work);
    SUPERLU_FREE(paper->diagonal_work);
    SUPERLU_FREE(paper->id_order);
    memset(paper, 0, sizeof(*paper));
}

static void dSymV2BuildPaperWork(int_t nsupers, int_t *xsup, int_t *supno,
                                 Glu_freeable_t *Glu_freeable,
                                 treeList_t *treeList,
                                 dSymV2PaperWork_t *paper)
{
    const double operation_cost = 1000.0;
    int_t *block_gids;
    int_t *block_rows;

    memset(paper, 0, sizeof(*paper));
    if (xsup == NULL || supno == NULL || Glu_freeable == NULL ||
        Glu_freeable->xlsub == NULL || Glu_freeable->lsub == NULL ||
        treeList == NULL)
        ABORT("The paper owner cost requires replicated symbolic L structure.");

    paper->row_work = (double *) SUPERLU_MALLOC(
        (size_t) nsupers * sizeof(double));
    paper->column_work = (double *) SUPERLU_MALLOC(
        (size_t) nsupers * sizeof(double));
    paper->diagonal_work = (double *) SUPERLU_MALLOC(
        (size_t) nsupers * sizeof(double));
    paper->id_order = INT_T_ALLOC(nsupers);
    block_gids = INT_T_ALLOC(nsupers);
    block_rows = INT_T_ALLOC(nsupers);
    if (paper->row_work == NULL || paper->column_work == NULL ||
        paper->diagonal_work == NULL || paper->id_order == NULL ||
        block_gids == NULL || block_rows == NULL)
        ABORT("Malloc fails for SymFact V2 paper owner work.");

    for (int_t k = 0; k < nsupers; ++k)
    {
        paper->row_work[k] = 0.0;
        paper->column_work[k] = 0.0;
        paper->diagonal_work[k] = 0.0;
        block_rows[k] = 0;
    }

    for (int_t k = 0; k < nsupers; ++k)
    {
        const double width = (double) (xsup[k + 1] - xsup[k]);
        const int_t fsupc = xsup[k];
        const int_t begin = Glu_freeable->xlsub[fsupc];
        const int_t end = Glu_freeable->xlsub[fsupc + 1];
        int_t block_count = 0;

        if (width <= 0.0 || begin < 0 || end < begin ||
            end > Glu_freeable->nzlmax)
            ABORT("The paper owner cost found invalid symbolic L bounds.");

        double work = width * width * width / 3.0 + operation_cost;
        paper->row_work[k] += work;
        paper->column_work[k] += work;
        paper->diagonal_work[k] += work;

        for (int_t p = begin; p < end; ++p)
        {
            const int_t row = Glu_freeable->lsub[p];
            if (row < 0 || row >= xsup[nsupers])
                ABORT("The paper owner cost found an invalid L row.");
            const int_t gid = supno[row];
            if (gid < k || gid >= nsupers)
                ABORT("The paper owner cost found an invalid lower block.");
            if (gid > k)
            {
                if (block_rows[gid] == 0)
                {
                    if (block_count >= nsupers)
                        ABORT("The paper owner block count exceeds the matrix size.");
                    block_gids[block_count++] = gid;
                }
                ++block_rows[gid];
            }
        }
        qsort(block_gids, (size_t) block_count, sizeof(int_t),
              dSymV2IntCompare);

        for (int_t b = 0; b < block_count; ++b)
        {
            const int_t gid = block_gids[b];
            const double rows = (double) block_rows[gid];
            work = rows * width * width + operation_cost;
            paper->row_work[gid] += work;
            paper->column_work[k] += work;
        }

        double prefix_rows = 0.0;
        double total_rows = 0.0;
        for (int_t b = 0; b < block_count; ++b)
            total_rows += (double) block_rows[block_gids[b]];
        for (int_t b = 0; b < block_count; ++b)
        {
            const int_t gid = block_gids[b];
            const double rows = (double) block_rows[gid];
            const double suffix_rows = total_rows - prefix_rows - rows;
            const double diagonal_update =
                rows * (rows + 1.0) * width;

            paper->row_work[gid] +=
                diagonal_update + 2.0 * rows * width * prefix_rows +
                operation_cost * (double) (b + 1);
            paper->column_work[gid] +=
                diagonal_update + 2.0 * rows * width * suffix_rows +
                operation_cost * (double) (block_count - b);
            paper->diagonal_work[gid] +=
                diagonal_update + operation_cost;
            prefix_rows += rows;
            block_rows[gid] = 0;
        }
    }

    dSymV2BuildPaperIDOrder(nsupers, treeList, paper->id_order);
    SUPERLU_FREE(block_rows);
    SUPERLU_FREE(block_gids);
}

static void dSymV2PaperCostForSupernode(int_t k,
                                        const dSymV2PaperWork_t *paper,
                                        dSymV2LDLCost_t *cost)
{
    cost->panel_work = SUPERLU_MAX(1.0, paper->column_work[k]);
    cost->row_work = SUPERLU_MAX(1.0, paper->row_work[k]);
    cost->rank_work = SUPERLU_MAX(1.0, paper->diagonal_work[k]);
    cost->comm_work = 0.0;
    cost->tree_weight = SUPERLU_MAX(
        1.0, cost->panel_work + cost->row_work - cost->rank_work);
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
                                int_t *supno,
                                Glu_freeable_t *Glu_freeable,
                                treeList_t *treeList,
                                gridinfo3d_t *grid3d)
{
    gridinfo_t *grid = &(grid3d->grid2d);
    int global_rank;
    int *local_owner;
    size_t owner_bytes = (size_t) nsupers * sizeof(int);
    const dSymV2OwnerMapping_t owner_mapping = dSymV2OwnerMapping();
    const dSymV2OwnerCost_t owner_cost = dSymV2OwnerCost();
    dSymV2PaperWork_t paper_work;
    double *panel_load = NULL;
    double *row_load = NULL;
    double *rank_load = NULL;

    memset(&paper_work, 0, sizeof(paper_work));
    if (owner_mapping == DSYM_V2_OWNER_PAPER_ID_ROW &&
        owner_cost != DSYM_V2_OWNER_COST_PAPER)
        ABORT("The paper increasing-depth owner mapping requires paper owner cost.");
    if (owner_cost == DSYM_V2_OWNER_COST_PAPER)
        dSymV2BuildPaperWork(nsupers, xsup, supno, Glu_freeable, treeList,
                             &paper_work);

    if (owner_mapping != DSYM_V2_OWNER_BLOCK_CYCLIC)
    {
        panel_load = (double *) SUPERLU_MALLOC(
            grid->npcol * sizeof(double));
        row_load = (double *) SUPERLU_MALLOC(
            grid->nprow * sizeof(double));
        rank_load = (double *) SUPERLU_MALLOC(
            grid->nprow * grid->npcol * sizeof(double));
        if (panel_load == NULL || row_load == NULL || rank_load == NULL)
            ABORT("Malloc fails for SymFact V2 LDL load metadata.");
        for (int pc = 0; pc < grid->npcol; ++pc)
            panel_load[pc] = 0.0;
        for (int pr = 0; pr < grid->nprow; ++pr)
            row_load[pr] = 0.0;
        for (int p = 0; p < grid->nprow * grid->npcol; ++p)
            rank_load[p] = 0.0;
    }

    dSymV2ResetLDLMetadata(trf3Dpart);

    if (!(trf3Dpart->symV2DiagOwner = (int *) SUPERLU_MALLOC(owner_bytes)) ||
        !(trf3Dpart->symV2PanelRoot = (int *) SUPERLU_MALLOC(owner_bytes)) ||
        !(trf3Dpart->symV2DiagRoot = (int *) SUPERLU_MALLOC(owner_bytes)) ||
        !(trf3Dpart->symV2PanelLocalIndex = INT_T_ALLOC(nsupers)) ||
        !(trf3Dpart->symV2RowLocalIndex = INT_T_ALLOC(nsupers)) ||
        !(local_owner = (int *) SUPERLU_MALLOC(owner_bytes)))
        ABORT("Malloc fails for SymFact V2 LDL owner metadata.");

    MPI_Comm_rank(grid3d->comm, &global_rank);
    const double affinity_weight =
        owner_mapping == DSYM_V2_OWNER_GREEDY
            ? dSymV2OwnerAffinityWeight()
            : 0.0;
    if (owner_cost == DSYM_V2_OWNER_COST_PAPER && affinity_weight > 0.0)
        ABORT("Paper owner cost does not define the owner-affinity term.");
    if (grid3d->iam == 0)
        printf("SymFact V2 LDL owner mapping: %s; owner cost: %s.\n",
               dSymV2OwnerMappingName(owner_mapping),
               dSymV2OwnerCostName(owner_cost));
    for (int_t k = 0; k < nsupers; ++k)
    {
        trf3Dpart->symV2PanelRoot[k] = -1;
        trf3Dpart->symV2DiagRoot[k] = -1;
    }
    for (int_t order = 0; order < nsupers; ++order)
    {
        const int_t k = owner_mapping == DSYM_V2_OWNER_PAPER_ID_ROW
                            ? paper_work.id_order[order]
                            : (affinity_weight > 0.0
                                   ? nsupers - 1 - order
                                   : order);
        int panel_root;
        int diag_root;
        int owner_rank;
        if (owner_mapping == DSYM_V2_OWNER_GREEDY)
        {
            dSymV2LDLCost_t cost;
            const int_t parent = setree != NULL ? setree[k] : nsupers;
            const int parent_pr =
                (parent >= 0 && parent < nsupers)
                    ? trf3Dpart->symV2DiagRoot[parent]
                    : -1;
            const int parent_pc =
                (parent >= 0 && parent < nsupers)
                    ? trf3Dpart->symV2PanelRoot[parent]
                    : -1;
            if (owner_cost == DSYM_V2_OWNER_COST_PAPER)
                dSymV2PaperCostForSupernode(k, &paper_work, &cost);
            else
                dSymV2EstimateSupernodeWork(k, xsup, Glu_freeable, grid,
                                            &cost);
            dSymV2ChooseOwnerPair(panel_load, row_load, rank_load, grid,
                                  &cost, parent_pr, parent_pc,
                                  affinity_weight, &diag_root, &panel_root);
            owner_rank = PNUM(diag_root, panel_root, grid);
            panel_load[panel_root] += cost.panel_work;
            row_load[diag_root] += cost.row_work;
            rank_load[owner_rank] += cost.rank_work;
        }
        else if (owner_mapping == DSYM_V2_OWNER_PAPER_ID_ROW)
        {
            diag_root = 0;
            for (int pr = 1; pr < grid->nprow; ++pr)
                if (row_load[pr] < row_load[diag_root])
                    diag_root = pr;
            panel_root = PCOL(k, grid);
            owner_rank = PNUM(diag_root, panel_root, grid);
            row_load[diag_root] += paper_work.row_work[k];
            panel_load[panel_root] += paper_work.column_work[k];
            rank_load[owner_rank] += paper_work.diagonal_work[k];
        }
        else
        {
            diag_root = PROW(k, grid);
            panel_root = PCOL(k, grid);
            owner_rank = PNUM(diag_root, panel_root, grid);
        }
        trf3Dpart->symV2PanelRoot[k] = panel_root;
        trf3Dpart->symV2DiagRoot[k] = diag_root;
        trf3Dpart->symV2PanelLocalIndex[k] = -1;
        trf3Dpart->symV2RowLocalIndex[k] = -1;
        local_owner[k] =
            (grid3d->zscp.Iam == 0 &&
             grid->iam == owner_rank)
                ? global_rank
                : INT_MAX;
    }

    if (grid3d->iam == 0 && owner_cost == DSYM_V2_OWNER_COST_PAPER &&
        owner_mapping != DSYM_V2_OWNER_BLOCK_CYCLIC)
    {
        double total_row_work = 0.0;
        double total_column_work = 0.0;
        double max_row_work = 0.0;
        double max_column_work = 0.0;
        for (int_t k = 0; k < nsupers; ++k)
        {
            total_row_work += paper_work.row_work[k];
            total_column_work += paper_work.column_work[k];
        }
        for (int pr = 0; pr < grid->nprow; ++pr)
            max_row_work = SUPERLU_MAX(max_row_work, row_load[pr]);
        for (int pc = 0; pc < grid->npcol; ++pc)
            max_column_work = SUPERLU_MAX(max_column_work,
                                          panel_load[pc]);
        printf("SymFact V2 paper owner balance: row %.6f, column %.6f.\n",
               total_row_work /
                   ((double) grid->nprow * SUPERLU_MAX(1.0, max_row_work)),
               total_column_work /
                   ((double) grid->npcol *
                    SUPERLU_MAX(1.0, max_column_work)));
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
    if (panel_load != NULL)
        SUPERLU_FREE(panel_load);
    if (row_load != NULL)
        SUPERLU_FREE(row_load);
    if (rank_load != NULL)
        SUPERLU_FREE(rank_load);
    dSymV2FreePaperWork(&paper_work);
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
    const int ldl_forest_weights = dSymV2LDLForestWeightsEnabled();
    if (ldl_forest_weights)
        dSymV2CalcLDLTreeWeight(nsupers, setree, treeList,
                                LUstruct->Glu_persist->xsup,
                                Glu_freeable, NULL, grid3d);
    else
        calcTreeWeight(nsupers, setree, treeList,
                       LUstruct->Glu_persist->xsup);
    if (grid3d->iam == 0)
        printf("SymFact V2 forest weights: %s.\n",
               ldl_forest_weights ? "LDL" : "original");
    trf3Dpart->gEtreeInfo = fillEtreeInfo(nsupers, setree, treeList);
    dSymV2InitLDLOwners(nsupers, trf3Dpart, setree,
                        LUstruct->Glu_persist->xsup,
                        LUstruct->Glu_persist->supno,
                        Glu_freeable, treeList, grid3d);
    dSymV2BuildLDLSchedule(nsupers, setree, trf3Dpart,
                           LUstruct->Glu_persist->xsup, grid3d);

    dSymV2InstallLDLForest(nsupers, trf3Dpart, setree, treeList,
                           LUstruct->Glu_persist->xsup, grid3d);

    free_treelist(nsupers, treeList);
}
