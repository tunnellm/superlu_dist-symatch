    void dPrintLUgpuSetupProfile(dLUgpu_Handle LuH)
    {
        (void) LuH;
    }

    void dPrintLUgpuFactorProfile(dLUgpu_Handle LuH)
    {
        (void) LuH;
    }

    void dSymLDLFactorGPUSynchronize(dLUgpu_Handle LuH)
    {
#ifdef HAVE_CUDA
        xLUstruct_t<double> *LU_v2 = reinterpret_cast<xLUstruct_t<double> *>(LuH);
        if (LU_v2 == NULL || !LU_v2->superlu_acc_offload)
            return;
        for (int stream = 0; stream < LU_v2->A_gpu.numCudaStreams; ++stream)
            cudaStreamSynchronize(LU_v2->A_gpu.cuStreams[stream]);
#else
        (void) LuH;
#endif
    }

    int dSymLDLFactorGPUCopyPanelToHost(dLUgpu_Handle LuH, int_t k)
    {
#ifdef HAVE_CUDA
        xLUstruct_t<double> *LU_v2 = reinterpret_cast<xLUstruct_t<double> *>(LuH);
        if (LU_v2 == NULL || k < 0 || k >= LU_v2->nsupers)
            return -1;
        if (!LU_v2->useSymV2Solve())
            return -2;
        int_t local = LU_v2->symV2PanelIndex(k);
        if (local < 0 || local >= LU_v2->symV2PanelCount())
            return -3;
        if (LU_v2->lPanelVec[local].isEmpty())
            return 0;
        if (LU_v2->lPanelVec[local].gpuPanel.val == NULL)
            return -4;
        LU_v2->lPanelVec[local].copyFromGPU();
        return 0;
#else
        (void) LuH;
        (void) k;
        return -1;
#endif
    }

#include "symldl_v2_solve_gpu_helpers.cuh"
#include "symldl_v2_solve_gpu_setup_impl.cuh"
#include "symldl_v2_solve_gpu_runtime_impl.cuh"

    void dSymLDLSolveGPUTakeTimers(dSymLDLSolveGPU_Handle handle,
                                   double *h2d, double *compute, double *d2h)
    {
        dSymLDLSolveGPUState *state =
            reinterpret_cast<dSymLDLSolveGPUState *>(handle);
        if (h2d) *h2d = 0.0;
        if (compute) *compute = 0.0;
        if (d2h) *d2h = 0.0;
        if (state == NULL)
            return;
        if (h2d) *h2d = state->t_h2d;
        if (compute) *compute = state->t_compute;
        if (d2h) *d2h = state->t_d2h;
        state->t_h2d = 0.0;
        state->t_compute = 0.0;
        state->t_d2h = 0.0;
    }
