#ifndef DSYMLDL_V2_RUNTIME_MODEL_H
#define DSYMLDL_V2_RUNTIME_MODEL_H

#include "dsymldl_v2_grid_model.h"

#ifdef __cplusplus
extern "C" {
#endif

int dSymLDLV2RuntimeRankForCoordinates(
    dSymLDLV2RankOrder rank_order, int pr_count, int pc_count, int pz_count,
    int pr, int pc, int pz);

int dSymLDLV2EvaluateRuntimeModel(
    const dSymLDLV2GridModelInput *input,
    const dSymLDLV2PartitionPlan *plan,
    int requested_streams, int estimated_streams,
    dSymLDLV2PerformanceEstimate *performance,
    char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
