#ifndef DSYMLDL_V2_DRIVER_H
#define DSYMLDL_V2_DRIVER_H

#include "superlu_ddefs.h"

#ifdef __cplusplus
extern "C" {
#endif

int dSymV2SolveEnabled(superlu_dist_options_t *options, int gpu3dVersion);
int_t *dSymV2CreateIdentityIpermSupno(int_t nsupers);
void dSymV2LluBufInit(dLUValSubBuf_t *LUvsb);
void dSymV2PrintFactorStats(int_t n, dLUstruct_t *LUstruct,
                            dtrf3Dpartition_t *trf3Dpartition,
                            gridinfo3d_t *grid3d,
                            superlu_dist_options_t *options,
                            SuperLUStat_t *stat, int *info,
                            int parSymbFact, float flinfo,
                            float dist_mem_use, float GA_mem_use,
                            superlu_dist_mem_usage_t symb_mem_usage,
                            superlu_dist_mem_usage_t *num_mem_usage);

#ifdef __cplusplus
}
#endif

#endif
