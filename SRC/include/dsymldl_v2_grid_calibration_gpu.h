#ifndef DSYMLDL_V2_GRID_CALIBRATION_GPU_H
#define DSYMLDL_V2_GRID_CALIBRATION_GPU_H

#include "dsymldl_v2_grid_selector.h"

#ifdef __cplusplus
extern "C" {
#endif

int dSymLDLV2CalibrateCUDADevice(
    int device_id, int representative_size, double budget_seconds,
    dSymLDLV2CalibrationCoefficient *gpu_flops,
    dSymLDLV2CalibrationCoefficient *gpu_local_bytes,
    dSymLDLV2CalibrationCoefficient *launches,
    dSymLDLV2CalibrationCoefficient *gpu_synchronizations,
    dSymLDLV2CalibrationCoefficient *host_to_device,
    dSymLDLV2CalibrationCoefficient *device_to_host,
    char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
