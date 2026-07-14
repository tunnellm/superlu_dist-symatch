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

    int dSymLDLFactorGPUGetPanel(dLUgpu_Handle LuH, int_t k,
                                 double **values, int_t *count)
    {
#ifdef HAVE_CUDA
        xLUstruct_t<double> *LU_v2 = reinterpret_cast<xLUstruct_t<double> *>(LuH);
        if (values != NULL) *values = NULL;
        if (count != NULL) *count = 0;
        if (LU_v2 == NULL || values == NULL || count == NULL ||
            k < 0 || k >= LU_v2->nsupers)
            return -1;
        if (!LU_v2->useSymV2Solve())
            return -2;
        int_t local = LU_v2->symV2PanelIndex(k);
        if (local < 0 || local >= LU_v2->symV2PanelCount())
            return -3;
        if (LU_v2->lPanelVec[local].isEmpty())
            return 0;
        *values = LU_v2->lPanelVec[local].gpuPanel.val;
        *count = LU_v2->lPanelVec[local].nzvalSize();
        return (*values != NULL && *count > 0) ? 0 : -4;
#else
        (void) LuH;
        (void) k;
        if (values != NULL) *values = NULL;
        if (count != NULL) *count = 0;
        return -1;
#endif
    }
