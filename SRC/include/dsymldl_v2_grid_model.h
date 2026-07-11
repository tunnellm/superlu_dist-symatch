#ifndef DSYMLDL_V2_GRID_MODEL_H
#define DSYMLDL_V2_GRID_MODEL_H

#include "dsymldl_v2_grid_selector.h"
#include "dsymldl_v2_partition_plan.h"
#include "superlu_ddefs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int_t nsupers;
    int_t n;
    uint64_t input_nnz;
    uint64_t symbolic_row_entries;
    uint64_t factor_value_entries;
    int_t block_count;
    int_t *panel_block_offsets;
    int_t *block_row_supernode;
    int_t *block_row_count;
    int_t *panel_rows;
} dSymLDLV2StructuralSummary;

typedef struct {
    uint64_t available_gpu_bytes;
    uint64_t available_host_bytes_per_node;
    int ranks_per_node;
    int ranks_per_gpu;
    int gpu_memory_known;
    int host_memory_known;
} dSymLDLV2AvailableMemory;

typedef enum {
    DSYMLDL_V2_RANK_ORDER_Z_MAJOR = 0,
    DSYMLDL_V2_RANK_ORDER_XY_MAJOR = 1
} dSymLDLV2RankOrder;

typedef struct {
    int communicator_size;
    int node_count;
    int topology_known;
    dSymLDLV2RankOrder rank_order;
    const int *node_of_rank;
} dSymLDLV2GridTopology;

typedef struct {
    int gpu_offload;
    int gpu_streams;
    int lookahead_depth;
    int solve_nrhs;
    size_t gemm_buffer_values;
    int max_supernode_size;
    int pinned_staging;
    int pooled_pinned_staging;
    int wpanel_cache;
    int pc_fragment_schur;
    int pc_fragment_ldl_native;
    int cuda_aware_mpi;
    dSymLDLV2RankOrder rank_order;
} dSymLDLV2GridRuntimeConfig;

typedef struct {
    const dSymLDLV2PartitionPlanInput *partition_input;
    const dSymLDLV2StructuralSummary *structure;
    dSymLDLV2AvailableMemory available;
    const dSymLDLV2GridTopology *topology;
    dSymLDLV2GridRuntimeConfig runtime;
    uint64_t gpu_reserve_bytes;
    uint64_t host_reserve_bytes_per_node;
} dSymLDLV2GridModelInput;

int dSymLDLV2BuildStructuralSummary(
    int_t nsupers, const Glu_persist_t *persist,
    const Glu_freeable_t *symbolic, dSymLDLV2StructuralSummary *summary,
    char *error, size_t error_size);

void dSymLDLV2StructuralSummaryDestroy(
    dSymLDLV2StructuralSummary *summary);

int dSymLDLV2DetectAvailableMemory(MPI_Comm communicator,
                                    dSymLDLV2AvailableMemory *available,
                                    char *error, size_t error_size);

int dSymLDLV2EvaluateGridCandidate(
    const dSymLDLV2GridModelInput *input,
    const dSymLDLV2PartitionPlan *plan,
    dSymLDLV2GridCandidate *candidate, char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
