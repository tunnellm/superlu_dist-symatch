

#include "superlu_ddefs.h"
#include "dsymldl_v2_partition_plan.h"
#include "dsymldl_v2_grid_report.h"
#include <errno.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

extern int_t calcTopInfoForest(sForest_t *forest, int_t nsupers,
                               int_t *setree);

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

static int_t *dSymV2CollectPanelRows(int_t nsupers, int_t *xsup,
                                     Glu_freeable_t *Glu_freeable,
                                     int_t **Lrowind_bc_ptr,
                                     gridinfo3d_t *grid3d)
{
    gridinfo_t *grid = &(grid3d->grid2d);
    int_t *panel_rows = INT_T_ALLOC(nsupers);
    if (panel_rows == NULL)
        ABORT("Malloc fails for SymFact V2 panel row summary.");
    for (int_t k = 0; k < nsupers; ++k)
        panel_rows[k] = 0;

    if (Lrowind_bc_ptr != NULL)
    {
        int_t mycol = MYCOL(grid->iam, grid);
        for (int_t k = 0; k < nsupers; ++k)
        {
            if (mycol != PCOL(k, grid))
                continue;
            int_t *lsub = Lrowind_bc_ptr[LBj(k, grid)];
            if (lsub != NULL)
                panel_rows[k] = lsub[1];
        }
    }
    else if (Glu_freeable != NULL && Glu_freeable->xlsub != NULL &&
             Glu_freeable->lsub != NULL)
    {
        for (int_t k = 0; k < nsupers; ++k)
        {
            int_t first_column = xsup[k];
            panel_rows[k] = SUPERLU_MAX(
                (int_t) 0,
                Glu_freeable->xlsub[first_column + 1] -
                    Glu_freeable->xlsub[first_column]);
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, panel_rows, nsupers, mpi_int_t, MPI_MAX,
                  grid->comm);
    for (int_t k = 0; k < nsupers; ++k)
        panel_rows[k] = SUPERLU_MAX(panel_rows[k], xsup[k + 1] - xsup[k]);
    return panel_rows;
}

static void dSymV2InstallLDLOwners(
    int_t nsupers, dtrf3Dpartition_t *trf3Dpart,
    const dSymLDLV2PartitionPlan *plan, gridinfo3d_t *grid3d)
{
    gridinfo_t *grid = &(grid3d->grid2d);
    size_t owner_bytes = (size_t) nsupers * sizeof(int);
    int global_rank;
    int *local_owner;

    dSymV2ResetLDLMetadata(trf3Dpart);
    trf3Dpart->symV2DiagOwner = (int *) SUPERLU_MALLOC(owner_bytes);
    trf3Dpart->symV2PanelRoot = (int *) SUPERLU_MALLOC(owner_bytes);
    trf3Dpart->symV2DiagRoot = (int *) SUPERLU_MALLOC(owner_bytes);
    trf3Dpart->symV2PanelLocalIndex = INT_T_ALLOC(nsupers);
    trf3Dpart->symV2RowLocalIndex = INT_T_ALLOC(nsupers);
    local_owner = (int *) SUPERLU_MALLOC(owner_bytes);
    if (trf3Dpart->symV2DiagOwner == NULL ||
        trf3Dpart->symV2PanelRoot == NULL ||
        trf3Dpart->symV2DiagRoot == NULL ||
        trf3Dpart->symV2PanelLocalIndex == NULL ||
        trf3Dpart->symV2RowLocalIndex == NULL || local_owner == NULL)
        ABORT("Malloc fails for SymFact V2 LDL owner metadata.");

    memcpy(trf3Dpart->symV2PanelRoot, plan->panel_root, owner_bytes);
    memcpy(trf3Dpart->symV2DiagRoot, plan->diag_root, owner_bytes);
    MPI_Comm_rank(grid3d->comm, &global_rank);
    for (int_t k = 0; k < nsupers; ++k)
    {
        int owner = PNUM(plan->diag_root[k], plan->panel_root[k], grid);
        trf3Dpart->symV2PanelLocalIndex[k] = -1;
        trf3Dpart->symV2RowLocalIndex[k] = -1;
        local_owner[k] = grid3d->zscp.Iam == 0 && grid->iam == owner
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
        if (plan->panel_root[k] == mycol)
            ++trf3Dpart->symV2LocalPanelCount;
        if (plan->diag_root[k] == myrow)
            ++trf3Dpart->symV2LocalRowCount;
    }
    if (trf3Dpart->symV2LocalPanelCount > 0)
        trf3Dpart->symV2LocalPanelGids =
            INT_T_ALLOC(trf3Dpart->symV2LocalPanelCount);
    if (trf3Dpart->symV2LocalRowCount > 0)
        trf3Dpart->symV2LocalRowGids =
            INT_T_ALLOC(trf3Dpart->symV2LocalRowCount);
    if ((trf3Dpart->symV2LocalPanelCount > 0 &&
         trf3Dpart->symV2LocalPanelGids == NULL) ||
        (trf3Dpart->symV2LocalRowCount > 0 &&
         trf3Dpart->symV2LocalRowGids == NULL))
        ABORT("Malloc fails for SymFact V2 local owner metadata.");

    int_t local_panel = 0;
    int_t local_row = 0;
    for (int_t k = 0; k < nsupers; ++k)
    {
        if (plan->panel_root[k] == mycol)
        {
            trf3Dpart->symV2PanelLocalIndex[k] = local_panel;
            trf3Dpart->symV2LocalPanelGids[local_panel++] = k;
        }
        if (plan->diag_root[k] == myrow)
        {
            trf3Dpart->symV2RowLocalIndex[k] = local_row;
            trf3Dpart->symV2LocalRowGids[local_row++] = k;
        }
    }
    SUPERLU_FREE(local_owner);
}

static void dSymV2InstallLDLSchedule(
    dtrf3Dpartition_t *trf3Dpart, dSymLDLV2PartitionPlan *plan)
{
    trf3Dpart->symV2ScheduleEnabled = 1;
    trf3Dpart->symV2FactorLevelCount = plan->factor_level_count;
    trf3Dpart->symV2FactorLevelPtr = plan->factor_level_ptr;
    trf3Dpart->symV2FactorNodes = plan->factor_nodes;
    trf3Dpart->symV2NodeLevel = plan->node_level;
    trf3Dpart->symV2NodeOrder = plan->node_order;
    trf3Dpart->symV2NodeIperm = plan->node_iperm;
    plan->factor_level_ptr = NULL;
    plan->factor_nodes = NULL;
    plan->node_level = NULL;
    plan->node_order = NULL;
    plan->node_iperm = NULL;
}

static double dSymV2OwnerAffinityWeight(void)
{
    const char *env = getenv("GPU3DV2_OWNER_AFFINITY");
    if (env == NULL || env[0] == '\0')
        return 0.0;
    char *end = NULL;
    errno = 0;
    double value = strtod(env, &end);
    if (errno == ERANGE || end == env || *end != '\0' ||
        !isfinite(value) || value < 0.0)
        ABORT("GPU3DV2_OWNER_AFFINITY must be a nonnegative number.");
    return value;
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

static void dSymV2InstallLDLForest(int_t nsupers,
                                   dtrf3Dpartition_t *trf3Dpart,
                                   dSymLDLV2PartitionPlan *plan,
                                   int_t *xsup,
                                   gridinfo3d_t *grid3d)
{
    int_t maxLvl = plan->max_z_levels;
    sForest_t **sForests = plan->forests;
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
    plan->forests = NULL;

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
    int_t *panel_rows;
    treeList_t *treeList;
    dSymLDLV2PartitionPlanInput plan_input;
    dSymLDLV2PartitionPlan plan;
    dSymLDLV2GridRuntimeConfig grid_runtime;
    char plan_error[256];

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
    trf3Dpart->supernodeMask = NULL;
    trf3Dpart->superGridMap = NULL;
    trf3Dpart->LUvsb = NULL;
    trf3Dpart->diagDims = NULL;
    trf3Dpart->gemmCsizes = NULL;
    dSymV2ResetLDLMetadata(trf3Dpart);

    setree = supernodal_etree(nsupers, LUstruct->etree,
                              LUstruct->Glu_persist->supno,
                              LUstruct->Glu_persist->xsup);
    treeList = setree2list(nsupers, setree);
    panel_rows = dSymV2CollectPanelRows(
        nsupers, LUstruct->Glu_persist->xsup, Glu_freeable, NULL, grid3d);
    memset(&plan_input, 0, sizeof(plan_input));
    memset(&plan, 0, sizeof(plan));
    plan_input.nsupers = nsupers;
    plan_input.setree = setree;
    plan_input.xsup = LUstruct->Glu_persist->xsup;
    plan_input.lrows = panel_rows;
    plan_input.tree_list = treeList;
    double owner_affinity_weight = dSymV2OwnerAffinityWeight();
    double minimum_owner_affinity;
    double maximum_owner_affinity;
    MPI_Allreduce(&owner_affinity_weight, &minimum_owner_affinity, 1,
                  MPI_DOUBLE, MPI_MIN, grid3d->comm);
    MPI_Allreduce(&owner_affinity_weight, &maximum_owner_affinity, 1,
                  MPI_DOUBLE, MPI_MAX, grid3d->comm);
    if (minimum_owner_affinity != maximum_owner_affinity)
        ABORT("GPU3DV2_OWNER_AFFINITY differs between MPI ranks.");
    plan_input.owner_affinity_weight = owner_affinity_weight;
    if (!dSymLDLV2BuildPartitionPlan(
            &plan_input, grid3d->nprow, grid3d->npcol, grid3d->npdep,
            &plan, plan_error, sizeof(plan_error)))
    {
        fprintf(stderr, "%s\n", plan_error);
        ABORT("SymFact V2 partition planning failed.");
    }

    int grid_report_enabled = dSymLDLV2GridReportEnabled();
    int minimum_grid_report_enabled;
    int maximum_grid_report_enabled;
    MPI_Allreduce(&grid_report_enabled, &minimum_grid_report_enabled, 1,
                  MPI_INT, MPI_MIN, grid3d->comm);
    MPI_Allreduce(&grid_report_enabled, &maximum_grid_report_enabled, 1,
                  MPI_INT, MPI_MAX, grid3d->comm);
    if (minimum_grid_report_enabled != maximum_grid_report_enabled)
        ABORT("SymLDL grid report setting differs between MPI ranks.");

    if (grid_report_enabled)
    {
        dSymLDLV2StructuralSummary structure;
        dSymLDLV2GridRuntimeConfigInit(options, &grid_runtime);
        memset(&structure, 0, sizeof(structure));
        if (!dSymLDLV2BuildStructuralSummary(
                nsupers, LUstruct->Glu_persist, Glu_freeable,
                &structure, plan_error, sizeof(plan_error)) ||
            !dSymLDLV2ReportGridCandidates(
                &plan_input, &structure, grid3d->comm,
                grid3d->nprow, grid3d->npcol, grid3d->npdep,
                &grid_runtime,
                plan_error, sizeof(plan_error)))
        {
            dSymLDLV2StructuralSummaryDestroy(&structure);
            fprintf(stderr, "%s\n", plan_error);
            ABORT("SymLDL automatic grid report failed.");
        }
        dSymLDLV2StructuralSummaryDestroy(&structure);
    }

    trf3Dpart->gEtreeInfo = fillEtreeInfo(nsupers, setree, treeList);
    dSymV2InstallLDLOwners(nsupers, trf3Dpart, &plan, grid3d);
    dSymV2InstallLDLSchedule(trf3Dpart, &plan);
    dSymV2InstallLDLForest(nsupers, trf3Dpart, &plan,
                           LUstruct->Glu_persist->xsup, grid3d);

    dSymLDLV2PartitionPlanDestroy(&plan);
    SUPERLU_FREE(panel_rows);
    free_treelist(nsupers, treeList);
}
