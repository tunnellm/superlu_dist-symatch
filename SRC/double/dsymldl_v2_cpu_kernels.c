/*! @file
 * \brief Host numeric kernels for the event-driven true-symmetric solve.
 */

#include "dsymldl_v2_cpu_kernels.h"

#include <limits.h>

static int
symldl_cpu_blas_count(int_t count)
{
    int value = (int) count;
    if (count < 0 || (int_t) value != count)
        ABORT("SymLDL CPU solve BLAS dimension overflows int.");
    return value;
}

static void
symldl_cpu_gemm(const char *transa, int_t m_in, int_t n_in, int_t k_in,
                double alpha, const double *a, int_t lda_in,
                const double *b, int_t ldb_in, double beta,
                double *c, int_t ldc_in)
{
    const char transb = 'N';
    int m = symldl_cpu_blas_count(m_in);
    int n = symldl_cpu_blas_count(n_in);
    int k = symldl_cpu_blas_count(k_in);
    int lda = symldl_cpu_blas_count(lda_in);
    int ldb = symldl_cpu_blas_count(ldb_in);
    int ldc = symldl_cpu_blas_count(ldc_in);
    if (m == 0 || n == 0)
        return;
#if defined(USE_VENDOR_BLAS)
    dgemm_(transa, &transb, &m, &n, &k, &alpha, (double *) a, &lda,
           (double *) b, &ldb, &beta, c, &ldc, 1, 1);
#else
    dgemm_(transa, &transb, &m, &n, &k, &alpha, (double *) a, &lda,
           (double *) b, &ldb, &beta, c, &ldc);
#endif
}

void
dSymLDLCPUForwardBlock(const dSymLDLSolveGraph *graph, int_t edge,
                       int nrhs, const double *source_x, double *output)
{
    const dSymLDLBlockDesc *block = &graph->blocks[edge];
    const dSymLDLPanelDesc *panel = &graph->panels[block->panel_id];
    if (nrhs == 1) {
        for (int_t row = 0; row < block->nbrow; ++row) {
            const double *a = panel->values + block->luptr + row;
            double sum = 0.0;
            for (int_t col = 0; col < panel->width; ++col)
                sum += a[col * panel->nsupr] * source_x[col];
            output[row] = -sum;
        }
        return;
    }
    symldl_cpu_gemm("N", block->nbrow, nrhs, panel->width, -1.0,
                    panel->values + block->luptr, panel->nsupr,
                    source_x, panel->width, 0.0, output, block->nbrow);
}

void
dSymLDLCPUBackwardBlock(const dSymLDLSolveGraph *graph, int_t edge,
                        int nrhs, const double *source_x, double *gather,
                        double *output)
{
    const dSymLDLBlockDesc *block = &graph->blocks[edge];
    const dSymLDLPanelDesc *panel = &graph->panels[block->panel_id];
    int_t source = block->target_gid;
    int_t source_first = graph->xsup[source];
    int_t source_width = graph->xsup[source + 1] - source_first;

    if (nrhs == 1) {
        for (int_t col = 0; col < panel->width; ++col) {
            const double *a = panel->values + block->luptr +
                              col * panel->nsupr;
            double sum = 0.0;
            for (int_t row = 0; row < block->nbrow; ++row) {
                int_t relative = graph->rows[block->row_begin + row] -
                                 source_first;
                sum += a[row] * source_x[relative];
            }
            output[col] = -sum;
        }
        return;
    }

    for (int rhs = 0; rhs < nrhs; ++rhs)
        for (int_t row = 0; row < block->nbrow; ++row) {
            int_t relative = graph->rows[block->row_begin + row] -
                             source_first;
            gather[row + (int_t) rhs * block->nbrow] =
                source_x[relative + (int_t) rhs * source_width];
        }
    symldl_cpu_gemm("T", panel->width, nrhs, block->nbrow, -1.0,
                    panel->values + block->luptr, panel->nsupr,
                    gather, block->nbrow, 0.0, output, panel->width);
}

void
dSymLDLCPUDiagonalBlock(const dSymLDLPanelDesc *panel, int nrhs,
                        const double *input, double *output)
{
    if (nrhs == 1) {
        for (int_t row = 0; row < panel->width; ++row) {
            double sum = 0.0;
            for (int_t col = 0; col < panel->width; ++col)
                sum += panel->values[panel->diag_luptr + row +
                                     col * panel->nsupr] * input[col];
            output[row] = sum;
        }
        return;
    }
    symldl_cpu_gemm("N", panel->width, nrhs, panel->width, 1.0,
                    panel->values + panel->diag_luptr, panel->nsupr,
                    input, panel->width, 0.0, output, panel->width);
}
