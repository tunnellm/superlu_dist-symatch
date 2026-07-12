#ifndef DSYMLDL_V2_GRID_REPORT_H
#define DSYMLDL_V2_GRID_REPORT_H

#include "dsymldl_v2_grid_model.h"
#include "symldl_v2_grid_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

int dSymLDLV2GridReportEnabled(void);

int dSymLDLV2GetCurrentGridPrediction(
    dSymLDLV2MemoryEstimate *memory);

void dSymLDLV2SetCurrentGridPrediction(
    const dSymLDLV2MemoryEstimate *memory);

int dSymLDLV2GetCurrentGridGPUUsage(uint64_t *bytes);

int dSymLDLV2ApplyGridEnvironment(
    dSymLDLV2GridRequest *request, char *error, size_t error_size);

void dSymLDLV2GridRuntimeConfigInit(
    const superlu_dist_options_t *options,
    dSymLDLV2GridRuntimeConfig *runtime);

int dSymLDLV2SelectGridForStructure(
    const dSymLDLV2PartitionPlanInput *partition_input,
    const dSymLDLV2StructuralSummary *structure, MPI_Comm communicator,
    const dSymLDLV2GridRequest *request,
    const dSymLDLV2GridRuntimeConfig *runtime,
    int calibrate,
    dSymLDLV2GridSelection *selection, char *error, size_t error_size);

void dSymLDLV2PrintGridSelection(
    const dSymLDLV2GridSelection *selection,
    int current_pr, int current_pc, int current_pz, int report_only);

int dSymLDLV2ReportGridCandidates(
    const dSymLDLV2PartitionPlanInput *partition_input,
    const dSymLDLV2StructuralSummary *structure, MPI_Comm communicator,
    int current_pr, int current_pc, int current_pz,
    const dSymLDLV2GridRuntimeConfig *runtime,
    char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
