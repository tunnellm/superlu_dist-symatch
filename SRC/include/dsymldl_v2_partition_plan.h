#ifndef DSYMLDL_V2_PARTITION_PLAN_H
#define DSYMLDL_V2_PARTITION_PLAN_H

#include "superlu_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double panel_work;
    double row_work;
    double rank_work;
    double communication_work;
    double tree_weight;
} dSymLDLV2SupernodeCost;

typedef struct {
    int_t nsupers;
    const int_t *setree;
    const int_t *xsup;
    const int_t *lrows;
    const treeList_t *tree_list;
    double owner_affinity_weight;
} dSymLDLV2PartitionPlanInput;

typedef struct {
    int pr;
    int pc;
    int pz;
    int_t nsupers;

    int *diag_root;
    int *panel_root;
    double *panel_load;
    double *row_load;
    double *rank_load;
    double maximum_panel_load;
    double maximum_row_load;
    double maximum_rank_load;
    double maximum_level_panel_load;
    double maximum_level_row_load;
    double maximum_level_rank_load;

    int_t factor_level_count;
    int_t *factor_level_ptr;
    int_t *factor_nodes;
    int_t *node_level;
    int_t *node_order;
    int_t *node_iperm;

    int_t max_z_levels;
    int_t forest_count;
    sForest_t **forests;
    int_t *forest_of_supernode;
    int_t *active_supernodes_per_z;
    int_t *replicated_supernodes_per_z;
    double *active_weight_per_z;
    double *replicated_weight_per_z;
} dSymLDLV2PartitionPlan;

void dSymLDLV2SupernodeCostFromDimensions(double supernode_columns,
                                          double panel_rows, int pr,
                                          dSymLDLV2SupernodeCost *cost);

int dSymLDLV2BuildPartitionPlan(
    const dSymLDLV2PartitionPlanInput *input, int pr, int pc, int pz,
    dSymLDLV2PartitionPlan *plan, char *error, size_t error_size);

void dSymLDLV2PartitionPlanDestroy(dSymLDLV2PartitionPlan *plan);

#ifdef __cplusplus
}
#endif

#endif
