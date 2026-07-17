#ifndef DSYMLDL_V2_SOLVE_GRAPH_H
#define DSYMLDL_V2_SOLVE_GRAPH_H

#include "superlu_ddefs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int_t gid;
    int_t width;
    int_t nsupr;
    int_t diag_luptr;
    int_t block_begin;
    int_t block_count;
    int_t value_count;
    int has_diag;
    int active;
    double *values;
} dSymLDLPanelDesc;

typedef struct {
    int_t panel_id;
    int_t target_gid;
    int_t luptr;
    int_t nbrow;
    int_t row_begin;
} dSymLDLBlockDesc;

typedef struct {
    int root_rank;
    int parent_rank;
    int children[2];
    int rank_index;
    unsigned char child_count;
    unsigned char active;
} dSymLDLTreeNode;

typedef struct {
    int sender_z;
    int receiver_z;
    int_t gid_begin;
    int_t gid_count;
    int_t value_count;
    unsigned char active;
} dSymLDLZStage;

typedef struct dSymLDLSolveGraph {
    int_t n;
    int_t nsupers;
    int_t panel_count;
    int_t block_count;
    int_t factor_row_count;
    int_t row_count;
    int_t backward_lsum_rows;
    int maxsup;

    dSymLDLPanelDesc *panels;
    dSymLDLBlockDesc *blocks;
    int_t *rows;
    int_t *panel_gids;
    int_t *row_gids;
    int_t *panel_local_index;
    int_t *row_local_index;
    int_t *source_edge_offsets;
    int_t *source_edge_ids;
    int_t *target_ilsum;
    int_t *xsup;
    int_t *ilsum;
    int *diag_roots;
    int *panel_roots;

    dSymLDLTreeNode *forward_bcast;
    dSymLDLTreeNode *forward_reduce;
    dSymLDLTreeNode *backward_bcast;
    dSymLDLTreeNode *backward_reduce;
    int *forward_local_dependencies;
    int *backward_local_dependencies;

    int z_rank;
    int z_size;
    int z_stage_count;
    int_t z_gid_count;
    dSymLDLZStage *z_stages;
    int_t *z_stage_gids;
} dSymLDLSolveGraph;

dSymLDLSolveGraph *dSymLDLSolveGraphCreate(
    int_t n, int_t nsupers,
    int_t panel_count, const dSymLDLPanelDesc *panels,
    int_t block_count, const dSymLDLBlockDesc *blocks,
    int_t factor_row_count, const int_t *rows,
    const int_t *xsup, const int_t *ilsum,
    dtrf3Dpartition_t *partition, gridinfo3d_t *grid3d);

void dSymLDLSolveGraphDestroy(dSymLDLSolveGraph *graph);

#ifdef __cplusplus
}
#endif

#endif
