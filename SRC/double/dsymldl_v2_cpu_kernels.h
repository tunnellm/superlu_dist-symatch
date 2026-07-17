#ifndef DSYMLDL_V2_CPU_KERNELS_H
#define DSYMLDL_V2_CPU_KERNELS_H

#include "dsymldl_v2_solve_graph.h"

void dSymLDLCPUForwardBlock(const dSymLDLSolveGraph *graph, int_t edge,
                            int nrhs, const double *source_x,
                            double *output);

void dSymLDLCPUBackwardBlock(const dSymLDLSolveGraph *graph, int_t edge,
                             int nrhs, const double *source_x,
                             double *gather, double *output);

void dSymLDLCPUDiagonalBlock(const dSymLDLPanelDesc *panel, int nrhs,
                             const double *input, double *output);

#endif
