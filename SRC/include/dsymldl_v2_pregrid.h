#ifndef DSYMLDL_V2_PREGRID_H
#define DSYMLDL_V2_PREGRID_H

#include "dsymldl_v2_grid_report.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Evaluate a request from replicated symbolic data without installing it.
   The output selection must be initialized with dSymLDLV2GridSelectionInit. */
int dSymLDLV2SelectGridFromSymbolic(
    int_t nsupers, const Glu_persist_t *persist,
    const Glu_freeable_t *symbolic, const int_t *etree,
    uint64_t input_nnz, MPI_Comm communicator,
    const dSymLDLV2GridRequest *request,
    const dSymLDLV2GridRuntimeConfig *runtime,
    dSymLDLV2GridSelection *selection,
    char *error, size_t error_size);

/* Run preprocessing on a caller-owned provisional grid and preserve A.
   The output selection must be initialized with dSymLDLV2GridSelectionInit. */
int dSymLDLV2AnalyzeDistributedMatrix(
    superlu_dist_options_t *options, SuperMatrix *A,
    gridinfo3d_t *analysis_grid, const dSymLDLV2GridRequest *request,
    const dSymLDLV2GridRuntimeConfig *runtime,
    dSymLDLV2GridSelection *selection,
    char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
