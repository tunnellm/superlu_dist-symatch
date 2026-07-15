/*! @file
 * \brief Non-CUDA stubs for the SymLDL NVSHMEM solve backend.
 */

#include "superlu_ddefs.h"
#include "dsymldl_v2_nvshmem_solve.h"

int
dSymLDLNVSHMEMSolveAvailable(void)
{
    return 0;
}

dSymLDLNVSHMEMSolveHandle
dSymLDLNVSHMEMSolveCreate(
    int_t n, int_t nsupers, int nrhs, int_t x_count, int_t lsum_count,
    int_t panel_count, const dSymLDLNVPanelDesc *panels,
    int_t block_count, const dSymLDLNVBlockDesc *blocks,
    int_t row_count, const int_t *rows,
    const int_t *xsup, const int_t *ilsum,
    dtrf3Dpartition_t *trf3Dpartition, gridinfo3d_t *grid3d)
{
    (void) n;
    (void) nsupers;
    (void) nrhs;
    (void) x_count;
    (void) lsum_count;
    (void) panel_count;
    (void) panels;
    (void) block_count;
    (void) blocks;
    (void) row_count;
    (void) rows;
    (void) xsup;
    (void) ilsum;
    (void) trf3Dpartition;
    (void) grid3d;
    return NULL;
}

int
dSymLDLNVSHMEMForward(dSymLDLNVSHMEMSolveHandle handle,
                      double *x, int_t x_count)
{
    (void) handle;
    (void) x;
    (void) x_count;
    return -1;
}

int
dSymLDLNVSHMEMDiagonal(dSymLDLNVSHMEMSolveHandle handle,
                       double *x, int_t x_count)
{
    (void) handle;
    (void) x;
    (void) x_count;
    return -1;
}

int
dSymLDLNVSHMEMBackward(dSymLDLNVSHMEMSolveHandle handle,
                       double *x, int_t x_count)
{
    (void) handle;
    (void) x;
    (void) x_count;
    return -1;
}

void
dSymLDLNVSHMEMSolveTakeTimers(dSymLDLNVSHMEMSolveHandle handle,
                              double *setup, double *forward,
                              double *sparse_reduce,
                              double *sparse_broadcast,
                              double *diagonal, double *backward,
                              double *h2d, double *d2h)
{
    (void) handle;
    if (setup != NULL) *setup = 0.0;
    if (forward != NULL) *forward = 0.0;
    if (sparse_reduce != NULL) *sparse_reduce = 0.0;
    if (sparse_broadcast != NULL) *sparse_broadcast = 0.0;
    if (diagonal != NULL) *diagonal = 0.0;
    if (backward != NULL) *backward = 0.0;
    if (h2d != NULL) *h2d = 0.0;
    if (d2h != NULL) *d2h = 0.0;
}

void
dSymLDLNVSHMEMSolveDestroy(dSymLDLNVSHMEMSolveHandle handle)
{
    (void) handle;
}
