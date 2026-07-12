#ifndef DSYMLDL_V2_GRID_CALIBRATION_H
#define DSYMLDL_V2_GRID_CALIBRATION_H

#include "dsymldl_v2_grid_model.h"

#ifdef __cplusplus
extern "C" {
#endif

void dSymLDLV2CalibrationProfileInit(
    dSymLDLV2CalibrationProfile *profile);

/* Measure effective resource rates on the allocation used by communicator.
   The result is collective and identical on every rank. */
int dSymLDLV2CalibrateGridModel(
    MPI_Comm communicator, const dSymLDLV2GridTopology *topology,
    const dSymLDLV2GridRuntimeConfig *runtime,
    const dSymLDLV2PartitionPlanInput *partition_input,
    const dSymLDLV2GridSelection *selection,
    double budget_seconds, dSymLDLV2CalibrationProfile *profile,
    char *error, size_t error_size);

double dSymLDLV2CalibrationBudgetSeconds(int node_count);

int dSymLDLV2GridCalibrationEnabled(void);

#ifdef __cplusplus
}
#endif

#endif
