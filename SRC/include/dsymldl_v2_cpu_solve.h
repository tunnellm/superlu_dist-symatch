#ifndef DSYMLDL_V2_CPU_SOLVE_H
#define DSYMLDL_V2_CPU_SOLVE_H

#include "dsymldl_v2_solve_graph.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dSymLDLCPUSolveHandle dSymLDLCPUSolveHandle;

typedef struct {
    uint64_t forward_x_messages;
    uint64_t forward_x_bytes;
    uint64_t forward_partial_messages;
    uint64_t forward_partial_bytes;
    uint64_t backward_x_messages;
    uint64_t backward_x_bytes;
    uint64_t backward_partial_messages;
    uint64_t backward_partial_bytes;
} dSymLDLCPUSolveCommStats;

dSymLDLCPUSolveHandle *dSymLDLCPUSolveCreate(
    const dSymLDLSolveGraph *graph, int nrhs_capacity,
    dtrf3Dpartition_t *partition, gridinfo3d_t *grid3d);

int dSymLDLCPUForward(dSymLDLCPUSolveHandle *handle,
                      double *x, int_t x_count, int nrhs);
int dSymLDLCPUDiagonal(dSymLDLCPUSolveHandle *handle,
                       double *x, int_t x_count, int nrhs);
int dSymLDLCPUBackward(dSymLDLCPUSolveHandle *handle,
                       double *x, int_t x_count, int nrhs);

void dSymLDLCPUSolveTakeTimers(
    dSymLDLCPUSolveHandle *handle, double *setup, double *forward,
    double *diagonal, double *backward, double *mpi_progress,
    double *numeric_compute);

void dSymLDLCPUSolveTakeCommStats(
    dSymLDLCPUSolveHandle *handle, dSymLDLCPUSolveCommStats *stats);

void dSymLDLCPUSolveDestroy(dSymLDLCPUSolveHandle *handle);

#ifdef __cplusplus
}
#endif

#endif
