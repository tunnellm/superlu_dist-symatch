#pragma once

#include "superlu_ddefs.h"
#include "superlu_zdefs.h"
#include "superlu_sdefs.h"

int_t pdgstrf3d_v100(superlu_dist_options_t *options, int m, int n, double anorm,
                     dtrf3Dpartition_t *trf3Dpartition, SCT_t *SCT,
                     dLUstruct_t *LUstruct, gridinfo3d_t *grid3d,
                     SuperLUStat_t *stat, int *info);

#ifdef __cplusplus
extern "C"
{
#endif

    // Left for backward compatibility
    struct LUstruct_v100;
    typedef struct LUstruct_v100 *LUgpu_Handle;

    extern LUgpu_Handle createLUgpuHandle(int_t nsupers, int_t ldt_, dtrf3Dpartition_t *trf3Dpartition,
                                          dLUstruct_t *LUstruct, gridinfo3d_t *grid3d,
                                          SCT_t *SCT_, superlu_dist_options_t *options_, SuperLUStat_t *stat,
                                          double thresh_, int *info_);

    extern void destroyLUgpuHandle(LUgpu_Handle LuH);

    extern int dgatherFactoredLU3Dto2D(LUgpu_Handle LuH);

    extern int copyLUGPU2Host(LUgpu_Handle LuH, dLUstruct_t *LUstruct);

    extern int pdgstrf3d_LUpackedInterface(LUgpu_Handle LUHand);

    struct dLUstruct_v1;
    typedef struct dLUstruct_v1 *dLUgpu_Handle;

    dLUgpu_Handle dCreateLUgpuHandle(int_t nsupers, int_t ldt_, dtrf3Dpartition_t *trf3Dpartition, dLUstruct_t *LUstruct, gridinfo3d_t *grid3d, SCT_t *SCT_, superlu_dist_options_t *options_, SuperLUStat_t *stat, double thresh_, int *info_);

    void dDestroyLUgpuHandle(dLUgpu_Handle LuH);

    int dGatherFactoredLU3Dto2D(dLUgpu_Handle LuH);

    int dCopyLUGPU2Host(dLUgpu_Handle LuH, dLUstruct_t *LUstruct);
    void dPrintLUgpuSetupProfile(dLUgpu_Handle LuH);
    void dPrintLUgpuFactorProfile(dLUgpu_Handle LuH);
    void dSymLDLFactorGPUSynchronize(dLUgpu_Handle LuH);
    int dSymLDLFactorGPUCopyPanelToHost(dLUgpu_Handle LuH, int_t k);

    int pdgstrf3d_LUv1(dLUgpu_Handle LUHand);
    int pdgstrf3d_LUv2(dLUgpu_Handle LUHand);

    typedef void *dSymLDLSolveGPU_Handle;
    dSymLDLSolveGPU_Handle dSymLDLSolveGPUCreate(int_t nsupers, int_t maxsup,
                                                 int_t max_panel_rows, int nrhs,
                                                 gridinfo3d_t *grid3d);
    void dSymLDLSolveGPUDestroy(dSymLDLSolveGPU_Handle handle);
    int dSymLDLSolveGPUSetPanel(dSymLDLSolveGPU_Handle handle, int_t k,
                                const double *lusup, int_t count);
    int dSymLDLSolveGPUAttachFactorPanel(dSymLDLSolveGPU_Handle handle,
                                         dLUgpu_Handle factor_handle,
                                         int_t k);
    int dSymLDLSolveGPUSetPanelSchedule(dSymLDLSolveGPU_Handle handle, int_t k,
                                        const int *row_to_send_pos,
                                        int_t row_count, int_t nblocks,
                                        const int_t *block_luptr,
                                        const int_t *block_nbrow,
                                        const int_t *block_row_start);
    int dSymLDLSolveGPUGemm(dSymLDLSolveGPU_Handle handle, int_t k,
                            int_t a_offset, char transa, char transb,
                            int_t m, int_t n, int_t kdim,
                            double alpha, int_t lda, const double *b,
                            int_t ldb, double beta, double *c, int_t ldc);
    int dSymLDLSolveGPUForwardPanel(dSymLDLSolveGPU_Handle handle, int_t k,
                                    int_t ksupc, int nrhs, int_t nsupr,
                                    int_t nblocks, const int_t *block_luptr,
                                    const int_t *block_nbrow,
                                    const int_t *block_row_start,
                                    const double *xk, int total_send,
                                    double *send_vals);
    int dSymLDLSolveGPUBackwardPanel(dSymLDLSolveGPU_Handle handle, int_t k,
                                     int_t ksupc, int nrhs, int_t nsupr,
                                     int_t nblocks, const int_t *block_luptr,
                                     const int_t *block_nbrow,
                                     int_t row_count,
                                     const double *row_values,
                                     double *delta_send);
    void dSymLDLSolveGPUTakeTimers(dSymLDLSolveGPU_Handle handle,
                                   double *h2d, double *compute, double *d2h);

    // Forward declaration of structs 
    // Forward declarations
// struct strf3Dpartition_t;
// typedef struct strf3Dpartition_t strf3Dpartition_t;
// struct sLUstruct_t;
// typedef struct sLUstruct_t sLUstruct_t;
struct ctrf3Dpartition_t;
typedef struct ctrf3Dpartition_t ctrf3Dpartition_t;
struct cLUstruct_t;
typedef struct cLUstruct_t cLUstruct_t;
// struct ztrf3Dpartition_t;
// typedef struct ztrf3Dpartition_t ztrf3Dpartition_t;
// struct zLUstruct_t;
// typedef struct zLUstruct_t zLUstruct_t;

//  Define the single precision real interface
    struct sLUstruct_v1;
    typedef struct sLUstruct_v1 *sLUgpu_Handle;

    // extern sLUgpu_Handle sCreateLUgpuHandle(int_t nsupers, int_t ldt_, 
    // strf3Dpartition_t *trf3Dpartition,
    //                                        sLUstruct_t *LUstruct, gridinfo3d_t *grid3d,
    //                                        SCT_t *SCT_, superlu_dist_options_t *options_, SuperLUStat_t *stat,
    //                                        double thresh_, int *info_);
    extern sLUgpu_Handle sCreateLUgpuHandle(int_t nsupers, int_t ldt_, strf3Dpartition_t *trf3Dpartition,
                                     sLUstruct_t *LUstruct, gridinfo3d_t *grid3d,
                                     SCT_t *SCT_, superlu_dist_options_t *options_, SuperLUStat_t *stat,
                                     float thresh_, int *info_);

    void sDestroyLUgpuHandle(sLUgpu_Handle LuH);

    extern int sgatherFactoredLU3Dto2D(sLUgpu_Handle LuH);

    int sCopyLUGPU2Host(sLUgpu_Handle LuH, sLUstruct_t *LUstruct);
    int psgstrf3d_LUv1(sLUgpu_Handle LUHand);

    extern int psgstrf3d_LUpackedInterface(sLUgpu_Handle LUHand);

    // struct cLUstruct_v1;
    // typedef struct cLUstruct_v1 *cLUgpu_Handle;

    // extern cLUgpu_Handle cCreateLUgpuHandle(int_t nsupers, int_t ldt_, ctrf3Dpartition_t *trf3Dpartition,
    //                                        cLUstruct_t *LUstruct, gridinfo3d_t *grid3d,
    //                                        SCT_t *SCT_, superlu_dist_options_t *options_, SuperLUStat_t *stat,
    //                                        float thresh_, int *info_);

    // extern void cdestroyLUgpuHandle(cLUgpu_Handle LuH);

    // extern int cgatherFactoredLU3Dto2D(cLUgpu_Handle LuH);

    // extern int ccopyLUGPU2Host(cLUgpu_Handle LuH, cLUstruct_t *LUstruct);

    // extern int pcgstrf3d_LUpackedInterface(cLUgpu_Handle LUHand);

    struct zLUstruct_v1;
    typedef struct zLUstruct_v1 *zLUgpu_Handle;

    extern zLUgpu_Handle zCreateLUgpuHandle(int_t nsupers, int_t ldt_, ztrf3Dpartition_t *trf3Dpartition,
                                           zLUstruct_t *LUstruct, gridinfo3d_t *grid3d,
                                           SCT_t *SCT_, superlu_dist_options_t *options_, SuperLUStat_t *stat,
                                           double thresh_, int *info_);

    void zDestroyLUgpuHandle(zLUgpu_Handle LuH);

    extern int zgatherFactoredLU3Dto2D(zLUgpu_Handle LuH);

    int zCopyLUGPU2Host(zLUgpu_Handle LuH, zLUstruct_t *LUstruct);
    int pzgstrf3d_LUv1(zLUgpu_Handle LUHand);
    extern int pzgstrf3d_LUpackedInterface(zLUgpu_Handle LUHand);

    // Forward declaration for batched interface
    // Single 
    struct sBatchFactorizeWorkspace;
    typedef struct sBatchFactorizeWorkspace* sBatchFactorize_Handle;

    extern int ssparseTreeFactorBatchGPU(sBatchFactorize_Handle ws, sForest_t *sforest);

    extern sBatchFactorize_Handle sgetBatchFactorizeWorkspace(
        int_t nsupers, int_t ldt, strf3Dpartition_t *trf3Dpartition, sLUstruct_t *LUstruct, 
        gridinfo3d_t *grid3d, superlu_dist_options_t *options, SuperLUStat_t *stat, int *info
    );

    extern void scopyGPULUDataToHost(
        sBatchFactorize_Handle ws, sLUstruct_t *LUstruct, gridinfo3d_t *grid3d,
        SCT_t *SCT_, superlu_dist_options_t *options, SuperLUStat_t *stat
    );

    extern void sfreeBatchFactorizeWorkspace(sBatchFactorize_Handle ws);struct dBatchFactorizeWorkspace;

    // Double 
    typedef struct dBatchFactorizeWorkspace* dBatchFactorize_Handle;

    extern int dsparseTreeFactorBatchGPU(dBatchFactorize_Handle ws, sForest_t *sforest);

    extern dBatchFactorize_Handle dgetBatchFactorizeWorkspace(
        int_t nsupers, int_t ldt, dtrf3Dpartition_t *trf3Dpartition, dLUstruct_t *LUstruct, 
        gridinfo3d_t *grid3d, superlu_dist_options_t *options, SuperLUStat_t *stat, int *info
    );

    extern void dcopyGPULUDataToHost(
        dBatchFactorize_Handle ws, dLUstruct_t *LUstruct, gridinfo3d_t *grid3d,
        SCT_t *SCT_, superlu_dist_options_t *options, SuperLUStat_t *stat
    );

    extern void dfreeBatchFactorizeWorkspace(dBatchFactorize_Handle ws);

    // Double complex
    typedef struct zBatchFactorizeWorkspace* zBatchFactorize_Handle;

    extern int zsparseTreeFactorBatchGPU(zBatchFactorize_Handle ws, sForest_t *sforest);

    extern zBatchFactorize_Handle zgetBatchFactorizeWorkspace(
        int_t nsupers, int_t ldt, ztrf3Dpartition_t *trf3Dpartition, zLUstruct_t *LUstruct, 
        gridinfo3d_t *grid3d, superlu_dist_options_t *options, SuperLUStat_t *stat, int *info
    );

    extern void zcopyGPULUDataToHost(
        zBatchFactorize_Handle ws, zLUstruct_t *LUstruct, gridinfo3d_t *grid3d,
        SCT_t *SCT_, superlu_dist_options_t *options, SuperLUStat_t *stat
    );

    extern void zfreeBatchFactorizeWorkspace(zBatchFactorize_Handle ws);
#ifdef __cplusplus
}
#endif
