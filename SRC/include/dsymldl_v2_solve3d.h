#ifndef DSYMLDL_V2_SOLVE3D_H
#define DSYMLDL_V2_SOLVE3D_H

#include "superlu_defs.h"
#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int_t gid;
    int_t fst_row;
    int_t width;
    int_t nsupr;
    int_t diag_luptr;
    int_t block_begin;
    int_t block_count;
    int_t row_begin;
    int_t row_count;
    int_t pivot_begin;
    int_t value_count;
    int owner;
    double *values;
} dSymLDL3DPanelDesc;

typedef struct {
    int_t panel;
    int_t target_gid;
    int_t target_owner;
    int_t luptr;
    int_t nbrow;
    int_t row_begin;
} dSymLDL3DBlockDesc;

typedef void *dSymLDL3DSolveGPUHandle;

int dSymLDL3DSolveGPUAvailable(void);

dSymLDL3DSolveGPUHandle dSymLDL3DSolveGPUCreate(
    int_t n, int_t nsupers, int nrhs, int_t maxsup, int_t nlevels,
    const int_t *level_panel_ptr, const int_t *level_block_ptr,
    int_t panel_count, const dSymLDL3DPanelDesc *panels,
    int_t block_count, const dSymLDL3DBlockDesc *blocks,
    int_t row_count, const int_t *rows, MPI_Comm comm);

int dSymLDL3DSolveGPURun(dSymLDL3DSolveGPUHandle handle,
                         double *x, const int_t *local_x_offsets,
                         const int_t *local_row_gids,
                         const int_t *local_row_first,
                         const int_t *local_row_width,
                         const int *local_row_owner,
                         int_t local_row_count, int global_rank);

void dSymLDL3DSolveGPUTakeTimers(dSymLDL3DSolveGPUHandle handle,
                                 double *h2d, double *forward,
                                 double *diagonal, double *backward,
                                 double *d2h);

void dSymLDL3DSolveGPUDestroy(dSymLDL3DSolveGPUHandle handle);

#ifdef __cplusplus
}
#endif

#endif
