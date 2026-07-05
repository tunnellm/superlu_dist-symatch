#ifndef DSYMLDL_V2_PARTITION_H
#define DSYMLDL_V2_PARTITION_H

#include "superlu_ddefs.h"

#ifdef __cplusplus
extern "C" {
#endif

void dSymV2TrfPartitionInit(int_t nsupers, dLUstruct_t *LUstruct,
                            Glu_freeable_t *Glu_freeable,
                            gridinfo3d_t *grid3d,
                            superlu_dist_options_t *options);

#ifdef __cplusplus
}
#endif

#endif
