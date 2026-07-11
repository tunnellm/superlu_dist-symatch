#include "dsymldl_v2_partition_plan.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int_t calcTopInfoForest(sForest_t *forest, int_t nsupers,
                               int_t *setree);

static void dSymLDLV2SetPlanError(char *error, size_t error_size,
                                  const char *message)
{
    if (error != NULL && error_size > 0)
    {
        snprintf(error, error_size, "%s", message);
        error[error_size - 1] = '\0';
    }
}

static int dSymLDLV2PlanIsPowerOfTwo(int value)
{
    return value > 0 && (value & (value - 1)) == 0;
}

static int_t dSymLDLV2PlanLog2(int value)
{
    int_t result = 0;
    while (value > 1)
    {
        value >>= 1;
        ++result;
    }
    return result;
}

static int dSymLDLV2PlanProduct(size_t left, size_t right, size_t *product)
{
    if (product == NULL || (left != 0 && right > SIZE_MAX / left))
        return 0;
    *product = left * right;
    return 1;
}

static int dSymLDLV2PlanAllocationFits(size_t count, size_t element_size)
{
    return element_size == 0 || count <= SIZE_MAX / element_size;
}

static double dSymLDLV2Maximum(const double *values, size_t count)
{
    double maximum = 0.0;
    for (size_t i = 0; i < count; ++i)
        if (values[i] > maximum)
            maximum = values[i];
    return maximum;
}

static double dSymLDLV2PanelRows(const dSymLDLV2PartitionPlanInput *input,
                                 int_t k)
{
    double columns = (double) (input->xsup[k + 1] - input->xsup[k]);
    double rows = input->lrows != NULL ? (double) input->lrows[k] : columns;
    return rows > columns ? rows : columns;
}

void dSymLDLV2SupernodeCostFromDimensions(double supernode_columns,
                                          double panel_rows, int pr,
                                          dSymLDLV2SupernodeCost *cost)
{
    if (cost == NULL)
        return;
    if (panel_rows < supernode_columns)
        panel_rows = supernode_columns;

    double below = panel_rows - supernode_columns;
    double diag_cost = supernode_columns * supernode_columns *
                       supernode_columns;
    double panel_factor_cost = below * supernode_columns *
                               supernode_columns;
    double ll_schur_cost = 0.5 * below * below * supernode_columns;
    double partner_communication =
        pr > 1 ? below * supernode_columns : 0.0;
    double solve_cost = supernode_columns * supernode_columns +
                        2.0 * below * supernode_columns;

    cost->panel_work = SUPERLU_MAX(
        1.0, panel_factor_cost + ll_schur_cost + partner_communication);
    cost->row_work = SUPERLU_MAX(
        1.0, diag_cost + solve_cost + partner_communication);
    cost->rank_work = SUPERLU_MAX(
        1.0, diag_cost + panel_factor_cost + solve_cost +
                 0.25 * ll_schur_cost + partner_communication);
    cost->communication_work = partner_communication;
    cost->tree_weight = SUPERLU_MAX(
        1.0, diag_cost + panel_factor_cost + ll_schur_cost +
                 partner_communication + solve_cost);
}

static void dSymLDLV2ChooseOwner(
    const dSymLDLV2PartitionPlanInput *input,
    dSymLDLV2PartitionPlan *plan, int_t k,
    const dSymLDLV2SupernodeCost *cost)
{
    double best_score = -1.0;
    int best_pr = 0;
    int best_pc = 0;
    int_t parent = input->setree != NULL ? input->setree[k] : input->nsupers;
    int parent_pr = parent >= 0 && parent < input->nsupers
                        ? plan->diag_root[parent]
                        : -1;
    int parent_pc = parent >= 0 && parent < input->nsupers
                        ? plan->panel_root[parent]
                        : -1;

    for (int pr = 0; pr < plan->pr; ++pr)
    {
        for (int pc = 0; pc < plan->pc; ++pc)
        {
            int rank = pr * plan->pc + pc;
            double projected_panel = plan->panel_load[pc] + cost->panel_work;
            double projected_row = plan->row_load[pr] + cost->row_work;
            double projected_rank = plan->rank_load[rank] + cost->rank_work;
            double maximum = SUPERLU_MAX(
                projected_panel, SUPERLU_MAX(projected_row, projected_rank));
            double affinity_penalty = 0.0;
            if (input->owner_affinity_weight > 0.0 &&
                parent_pr >= 0 && parent_pc >= 0)
            {
                if (pc != parent_pc)
                    affinity_penalty += cost->communication_work;
                if (pr != parent_pr)
                    affinity_penalty += 0.5 * cost->communication_work;
            }
            double score = maximum +
                           0.05 * (projected_panel + projected_row +
                                   projected_rank) +
                           input->owner_affinity_weight * affinity_penalty;
            if (best_score < 0.0 || score < best_score)
            {
                best_score = score;
                best_pr = pr;
                best_pc = pc;
            }
        }
    }

    plan->diag_root[k] = best_pr;
    plan->panel_root[k] = best_pc;
    plan->panel_load[best_pc] += cost->panel_work;
    plan->row_load[best_pr] += cost->row_work;
    plan->rank_load[best_pr * plan->pc + best_pc] += cost->rank_work;
}

static int dSymLDLV2BuildSchedule(const dSymLDLV2PartitionPlanInput *input,
                                  dSymLDLV2PartitionPlan *plan)
{
    sForest_t forest;
    memset(&forest, 0, sizeof(forest));
    forest.nNodes = input->nsupers;
    forest.numTrees = 1;
    forest.nodeList = INT_T_ALLOC(input->nsupers);
    if (forest.nodeList == NULL)
        return 0;
    for (int_t k = 0; k < input->nsupers; ++k)
        forest.nodeList[k] = k;

    calcTopInfoForest(&forest, input->nsupers, (int_t *) input->setree);
    plan->factor_level_count = forest.topoInfo.numLvl;
    plan->factor_nodes = forest.nodeList;
    plan->factor_level_ptr = forest.topoInfo.eTreeTopLims;
    plan->node_iperm = forest.topoInfo.myIperm;
    plan->node_level = INT_T_ALLOC(input->nsupers);
    plan->node_order = INT_T_ALLOC(input->nsupers);
    if (plan->node_level == NULL || plan->node_order == NULL)
        return 0;

    for (int_t k = 0; k < input->nsupers; ++k)
    {
        plan->node_level[k] = -1;
        plan->node_order[k] = -1;
    }
    for (int_t level = 0; level < plan->factor_level_count; ++level)
    {
        for (int_t position = plan->factor_level_ptr[level];
             position < plan->factor_level_ptr[level + 1]; ++position)
        {
            int_t k = plan->factor_nodes[position];
            plan->node_level[k] = level;
            plan->node_order[k] = position;
        }
    }
    return 1;
}

static int dSymLDLV2BuildOwners(const dSymLDLV2PartitionPlanInput *input,
                                dSymLDLV2PartitionPlan *plan)
{
    size_t rank_count;
    if (!dSymLDLV2PlanProduct((size_t) plan->pr, (size_t) plan->pc,
                              &rank_count) ||
        !dSymLDLV2PlanAllocationFits((size_t) input->nsupers,
                                     sizeof(int)) ||
        !dSymLDLV2PlanAllocationFits(rank_count, sizeof(double)))
        return 0;
    plan->diag_root = (int *) malloc((size_t) input->nsupers * sizeof(int));
    plan->panel_root =
        (int *) malloc((size_t) input->nsupers * sizeof(int));
    plan->panel_load = (double *) calloc((size_t) plan->pc, sizeof(double));
    plan->row_load = (double *) calloc((size_t) plan->pr, sizeof(double));
    plan->rank_load = (double *) calloc(rank_count, sizeof(double));
    if (plan->diag_root == NULL || plan->panel_root == NULL ||
        plan->panel_load == NULL || plan->row_load == NULL ||
        plan->rank_load == NULL)
        return 0;

    for (int_t k = 0; k < input->nsupers; ++k)
    {
        plan->diag_root[k] = -1;
        plan->panel_root[k] = -1;
    }
    for (int_t order = 0; order < input->nsupers; ++order)
    {
        int_t k = input->owner_affinity_weight > 0.0
                      ? input->nsupers - 1 - order
                      : order;
        double columns = (double) (input->xsup[k + 1] - input->xsup[k]);
        dSymLDLV2SupernodeCost cost;
        dSymLDLV2SupernodeCostFromDimensions(
            columns, dSymLDLV2PanelRows(input, k), plan->pr, &cost);
        dSymLDLV2ChooseOwner(input, plan, k, &cost);
    }

    plan->maximum_panel_load = dSymLDLV2Maximum(plan->panel_load, plan->pc);
    plan->maximum_row_load = dSymLDLV2Maximum(plan->row_load, plan->pr);
    plan->maximum_rank_load =
        dSymLDLV2Maximum(plan->rank_load, rank_count);
    return 1;
}

static int dSymLDLV2BuildLevelLoads(
    const dSymLDLV2PartitionPlanInput *input,
    dSymLDLV2PartitionPlan *plan)
{
    size_t panel_count;
    size_t row_count;
    size_t rank_count;
    if (plan->factor_level_count < 1 ||
        !dSymLDLV2PlanProduct((size_t) plan->factor_level_count,
                              (size_t) plan->pc, &panel_count) ||
        !dSymLDLV2PlanProduct((size_t) plan->factor_level_count,
                              (size_t) plan->pr, &row_count) ||
        !dSymLDLV2PlanProduct(panel_count, (size_t) plan->pr,
                              &rank_count) ||
        !dSymLDLV2PlanAllocationFits(panel_count, sizeof(double)) ||
        !dSymLDLV2PlanAllocationFits(row_count, sizeof(double)) ||
        !dSymLDLV2PlanAllocationFits(rank_count, sizeof(double)))
        return 0;
    double *panel = (double *) calloc(panel_count, sizeof(double));
    double *row = (double *) calloc(row_count, sizeof(double));
    double *rank = (double *) calloc(rank_count, sizeof(double));
    if (panel == NULL || row == NULL || rank == NULL)
    {
        free(panel);
        free(row);
        free(rank);
        return 0;
    }

    for (int_t k = 0; k < input->nsupers; ++k)
    {
        int_t level = plan->node_level[k];
        int pr = plan->diag_root[k];
        int pc = plan->panel_root[k];
        double columns = (double) (input->xsup[k + 1] - input->xsup[k]);
        dSymLDLV2SupernodeCost cost;
        dSymLDLV2SupernodeCostFromDimensions(
            columns, dSymLDLV2PanelRows(input, k), plan->pr, &cost);
        panel[(size_t) level * (size_t) plan->pc + (size_t) pc] +=
            cost.panel_work;
        row[(size_t) level * (size_t) plan->pr + (size_t) pr] +=
            cost.row_work;
        rank[(size_t) level * (size_t) plan->pr * (size_t) plan->pc +
             (size_t) pr * (size_t) plan->pc + (size_t) pc] +=
            cost.rank_work;
    }

    plan->maximum_level_panel_load =
        dSymLDLV2Maximum(panel, panel_count);
    plan->maximum_level_row_load = dSymLDLV2Maximum(row, row_count);
    plan->maximum_level_rank_load = dSymLDLV2Maximum(rank, rank_count);
    free(panel);
    free(row);
    free(rank);
    return 1;
}

static treeList_t *dSymLDLV2CandidateTreeList(
    const dSymLDLV2PartitionPlanInput *input, int pr)
{
    size_t count = (size_t) input->nsupers + 1;
    if (!dSymLDLV2PlanAllocationFits(count, sizeof(treeList_t)))
        return NULL;
    treeList_t *tree_list = (treeList_t *) malloc(
        count * sizeof(treeList_t));
    if (tree_list == NULL)
        return NULL;
    memcpy(tree_list, input->tree_list, count * sizeof(treeList_t));

    for (int_t k = 0; k < input->nsupers; ++k)
    {
        double columns = (double) (input->xsup[k + 1] - input->xsup[k]);
        dSymLDLV2SupernodeCost cost;
        dSymLDLV2SupernodeCostFromDimensions(
            columns, dSymLDLV2PanelRows(input, k), pr, &cost);
        tree_list[k].weight = cost.tree_weight;
        tree_list[k].iWeight = cost.tree_weight;
        tree_list[k].scuWeight = cost.tree_weight;
    }
    tree_list[input->nsupers].iWeight = 0.0;
    for (int_t k = 0; k < input->nsupers; ++k)
        tree_list[input->setree[k]].iWeight += tree_list[k].iWeight;
    return tree_list;
}

static int dSymLDLV2BuildForests(const dSymLDLV2PartitionPlanInput *input,
                                 dSymLDLV2PartitionPlan *plan)
{
    treeList_t *tree_list = dSymLDLV2CandidateTreeList(input, plan->pr);
    if (tree_list == NULL)
        return 0;

    plan->max_z_levels = dSymLDLV2PlanLog2(plan->pz) + 1;
    if (plan->max_z_levels >= (int_t) (sizeof(int) * CHAR_BIT - 1))
    {
        free(tree_list);
        return 0;
    }
    plan->forest_count = ((int_t) 1 << plan->max_z_levels) - 1;
    plan->forests = getGreedyLoadBalForests(
        plan->max_z_levels, input->nsupers, (int_t *) input->setree,
        tree_list);
    free(tree_list);
    if (plan->forests == NULL)
        return 0;

    plan->forest_of_supernode = INT_T_ALLOC(input->nsupers);
    if (plan->forest_of_supernode == NULL)
        return 0;
    for (int_t k = 0; k < input->nsupers; ++k)
        plan->forest_of_supernode[k] = -1;
    for (int_t tree = 0; tree < plan->forest_count; ++tree)
    {
        sForest_t *forest = plan->forests[tree];
        if (forest == NULL)
            continue;
        for (int_t i = 0; i < forest->nNodes; ++i)
        {
            int_t k = forest->nodeList[i];
            if (k < 0 || k >= input->nsupers ||
                plan->forest_of_supernode[k] != -1)
                return 0;
            plan->forest_of_supernode[k] = tree;
        }
    }
    for (int_t k = 0; k < input->nsupers; ++k)
        if (plan->forest_of_supernode[k] < 0)
            return 0;

    if (!dSymLDLV2PlanAllocationFits((size_t) plan->pz, sizeof(int_t)) ||
        !dSymLDLV2PlanAllocationFits((size_t) plan->pz, sizeof(double)))
        return 0;
    plan->active_supernodes_per_z =
        (int_t *) calloc((size_t) plan->pz, sizeof(int_t));
    plan->replicated_supernodes_per_z =
        (int_t *) calloc((size_t) plan->pz, sizeof(int_t));
    plan->active_weight_per_z =
        (double *) calloc((size_t) plan->pz, sizeof(double));
    plan->replicated_weight_per_z =
        (double *) calloc((size_t) plan->pz, sizeof(double));
    if (plan->active_supernodes_per_z == NULL ||
        plan->replicated_supernodes_per_z == NULL ||
        plan->active_weight_per_z == NULL ||
        plan->replicated_weight_per_z == NULL)
        return 0;

    for (int z = 0; z < plan->pz; ++z)
    {
        int_t tree = plan->pz - 1 + z;
        for (int_t level = 0; level < plan->max_z_levels; ++level)
        {
            sForest_t *forest = plan->forests[tree];
            int replicated = z % ((int) 1 << level) != 0;
            if (forest != NULL)
            {
                if (replicated)
                {
                    plan->replicated_supernodes_per_z[z] += forest->nNodes;
                    plan->replicated_weight_per_z[z] += forest->weight;
                }
                else
                {
                    plan->active_supernodes_per_z[z] += forest->nNodes;
                    plan->active_weight_per_z[z] += forest->weight;
                }
            }
            if (level + 1 < plan->max_z_levels)
                tree = (tree - 1) / 2;
        }
    }
    return 1;
}

int dSymLDLV2BuildPartitionPlan(
    const dSymLDLV2PartitionPlanInput *input, int pr, int pc, int pz,
    dSymLDLV2PartitionPlan *plan, char *error, size_t error_size)
{
    if (input == NULL || plan == NULL || input->nsupers < 1 ||
        input->setree == NULL || input->xsup == NULL ||
        input->tree_list == NULL)
    {
        dSymLDLV2SetPlanError(error, error_size,
                              "SymLDL partition plan input is incomplete.");
        return 0;
    }
    if (pr < 1 || pc < 1 || !dSymLDLV2PlanIsPowerOfTwo(pz) ||
        !isfinite(input->owner_affinity_weight) ||
        input->owner_affinity_weight < 0.0)
    {
        dSymLDLV2SetPlanError(error, error_size,
                              "SymLDL partition plan dimensions are invalid.");
        return 0;
    }
    if ((uint64_t) input->nsupers + 1 > SIZE_MAX / sizeof(int_t) ||
        (uint64_t) input->nsupers + 1 > SIZE_MAX / sizeof(treeList_t) ||
        (size_t) pr > SIZE_MAX / (size_t) pc)
    {
        dSymLDLV2SetPlanError(error, error_size,
                              "SymLDL partition plan dimensions overflow.");
        return 0;
    }

    if (input->xsup[0] != 0)
    {
        dSymLDLV2SetPlanError(error, error_size,
                              "SymLDL partition plan structure is invalid.");
        return 0;
    }
    for (int_t k = 0; k < input->nsupers; ++k)
    {
        if (input->xsup[k] < 0 || input->xsup[k + 1] <= input->xsup[k] ||
            input->setree[k] < 0 || input->setree[k] > input->nsupers ||
            (input->lrows != NULL &&
             input->lrows[k] < input->xsup[k + 1] - input->xsup[k]))
        {
            dSymLDLV2SetPlanError(
                error, error_size,
                "SymLDL partition plan structure is invalid.");
            return 0;
        }
    }

    memset(plan, 0, sizeof(*plan));
    plan->pr = pr;
    plan->pc = pc;
    plan->pz = pz;
    plan->nsupers = input->nsupers;

    if (!dSymLDLV2BuildSchedule(input, plan) ||
        !dSymLDLV2BuildOwners(input, plan) ||
        !dSymLDLV2BuildLevelLoads(input, plan) ||
        !dSymLDLV2BuildForests(input, plan))
    {
        dSymLDLV2PartitionPlanDestroy(plan);
        dSymLDLV2SetPlanError(error, error_size,
                              "SymLDL partition plan allocation failed.");
        return 0;
    }
    return 1;
}

void dSymLDLV2PartitionPlanDestroy(dSymLDLV2PartitionPlan *plan)
{
    if (plan == NULL)
        return;
    free(plan->diag_root);
    free(plan->panel_root);
    free(plan->panel_load);
    free(plan->row_load);
    free(plan->rank_load);
    free(plan->factor_level_ptr);
    free(plan->factor_nodes);
    free(plan->node_level);
    free(plan->node_order);
    free(plan->node_iperm);
    free(plan->forest_of_supernode);
    free(plan->active_supernodes_per_z);
    free(plan->replicated_supernodes_per_z);
    free(plan->active_weight_per_z);
    free(plan->replicated_weight_per_z);

    if (plan->forests != NULL)
    {
        for (int_t tree = 0; tree < plan->forest_count; ++tree)
        {
            sForest_t *forest = plan->forests[tree];
            if (forest == NULL)
                continue;
            SUPERLU_FREE(forest->nodeList);
            SUPERLU_FREE(forest->topoInfo.eTreeTopLims);
            SUPERLU_FREE(forest->topoInfo.myIperm);
            SUPERLU_FREE(forest);
        }
        SUPERLU_FREE(plan->forests);
    }
    memset(plan, 0, sizeof(*plan));
}
