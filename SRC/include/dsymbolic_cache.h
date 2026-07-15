#ifndef DSYMBOLIC_CACHE_H
#define DSYMBOLIC_CACHE_H

#include "superlu_ddefs.h"

#ifdef __cplusplus
extern "C" {
#endif

int dSymbolicCacheRead(
    const char *path, int_t n, const superlu_dist_options_t *options,
    dScalePermstruct_t *scale_permutation, dLUstruct_t *lu,
    Glu_freeable_t **symbolic, gridinfo3d_t *grid3d,
    char *error, size_t error_size);

int dSymbolicCacheWrite(
    const char *path, int_t n, const superlu_dist_options_t *options,
    const dScalePermstruct_t *scale_permutation, const dLUstruct_t *lu,
    const Glu_freeable_t *symbolic, gridinfo3d_t *grid3d,
    char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
