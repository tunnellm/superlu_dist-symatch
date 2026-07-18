#pragma once

#include "superlu_ddefs.h"

template <typename Ftype>
static inline void symldl_v2_cpu_gemm(
    const char *transa, const char *transb, int m, int n, int k,
    Ftype alpha, Ftype *a, int lda, Ftype *b, int ldb,
    Ftype beta, Ftype *c, int ldc);

template <>
inline void symldl_v2_cpu_gemm<double>(
    const char *transa, const char *transb, int m, int n, int k,
    double alpha, double *a, int lda, double *b, int ldb,
    double beta, double *c, int ldc)
{
    superlu_dgemm(transa, transb, m, n, k, alpha,
                  a, lda, b, ldb, beta, c, ldc);
}

template <typename Ftype>
static inline void symldl_v2_cpu_axpy(
    int n, Ftype alpha, Ftype *x, int incx, Ftype *y, int incy);

template <>
inline void symldl_v2_cpu_axpy<double>(
    int n, double alpha, double *x, int incx, double *y, int incy)
{
    superlu_daxpy(n, alpha, x, incx, y, incy);
}
