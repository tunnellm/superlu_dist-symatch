// This file is included inside the SymLDL GPU solve C API implementation.
// Keep declarations here limited to panel metadata setup.

    int dSymLDLSolveGPUSetPanel(dSymLDLSolveGPU_Handle handle, int_t k,
                                const double *lusup, int_t count)
    {
        dSymLDLSolveGPUState *state =
            reinterpret_cast<dSymLDLSolveGPUState *>(handle);
        if (state == NULL || k < 0 || k >= state->nsupers)
            return -1;
        if (count <= 0 || lusup == NULL)
            return 0;

        if (state->d_lusup[k] != NULL && state->owns_lusup[k])
            gpuErrchk(cudaFree(state->d_lusup[k]));
        gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&state->d_lusup[k]),
                             static_cast<size_t>(count) * sizeof(double)));
        gpuErrchk(cudaMemcpy(state->d_lusup[k], lusup,
                             static_cast<size_t>(count) * sizeof(double),
                             cudaMemcpyHostToDevice));
        state->lusup_count[k] = count;
        state->owns_lusup[k] = 1;
        return 0;
    }

    int dSymLDLSolveGPUAttachFactorPanel(dSymLDLSolveGPU_Handle handle,
                                         dLUgpu_Handle factor_handle,
                                         int_t k)
    {
        dSymLDLSolveGPUState *state =
            reinterpret_cast<dSymLDLSolveGPUState *>(handle);
        xLUstruct_t<double> *LU_v2 =
            reinterpret_cast<xLUstruct_t<double> *>(factor_handle);
        if (state == NULL || LU_v2 == NULL || k < 0 || k >= state->nsupers ||
            k >= LU_v2->nsupers)
            return -1;
        if (!LU_v2->useSymV2Solve())
            return -2;

        int_t local = LU_v2->symV2PanelIndex(k);
        if (local < 0 || local >= LU_v2->symV2PanelCount())
            return -3;
        if (LU_v2->lPanelVec[local].isEmpty())
            return 0;

        double *d_lusup = LU_v2->lPanelVec[local].gpuPanel.val;
        int_t count = LU_v2->lPanelVec[local].nzvalSize();
        if (d_lusup == NULL || count <= 0)
            return -4;

        if (state->d_lusup[k] != NULL && state->owns_lusup[k])
            gpuErrchk(cudaFree(state->d_lusup[k]));
        state->d_lusup[k] = d_lusup;
        state->lusup_count[k] = count;
        state->owns_lusup[k] = 0;
        return 0;
    }

    int dSymLDLSolveGPUSetPanelSchedule(dSymLDLSolveGPU_Handle handle, int_t k,
                                        const int *row_to_send_pos,
                                        int_t row_count, int_t nblocks,
                                        const int_t *block_luptr,
                                        const int_t *block_nbrow,
                                        const int_t *block_row_start)
    {
        dSymLDLSolveGPUState *state =
            reinterpret_cast<dSymLDLSolveGPUState *>(handle);
        if (state == NULL || k < 0 || k >= state->nsupers)
            return -1;
        if (row_count <= 0 || row_to_send_pos == NULL ||
            nblocks < 0 || (nblocks > 0 &&
             (block_luptr == NULL || block_nbrow == NULL ||
              block_row_start == NULL)))
            return -2;
        if (nblocks == 0)
            return 0;

        if (state->d_row_to_send_pos[k] != NULL)
            gpuErrchk(cudaFree(state->d_row_to_send_pos[k]));
        if (state->d_block_luptr[k] != NULL)
            gpuErrchk(cudaFree(state->d_block_luptr[k]));
        if (state->d_block_nbrow[k] != NULL)
            gpuErrchk(cudaFree(state->d_block_nbrow[k]));
        if (state->d_block_row_start[k] != NULL)
            gpuErrchk(cudaFree(state->d_block_row_start[k]));
        gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&state->d_row_to_send_pos[k]),
                             static_cast<size_t>(row_count) * sizeof(int)));
        gpuErrchk(cudaMemcpy(state->d_row_to_send_pos[k], row_to_send_pos,
                             static_cast<size_t>(row_count) * sizeof(int),
                             cudaMemcpyHostToDevice));
        gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&state->d_block_luptr[k]),
                             static_cast<size_t>(nblocks) * sizeof(int_t)));
        gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&state->d_block_nbrow[k]),
                             static_cast<size_t>(nblocks) * sizeof(int_t)));
        gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&state->d_block_row_start[k]),
                             static_cast<size_t>(nblocks) * sizeof(int_t)));
        gpuErrchk(cudaMemcpy(state->d_block_luptr[k], block_luptr,
                             static_cast<size_t>(nblocks) * sizeof(int_t),
                             cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(state->d_block_nbrow[k], block_nbrow,
                             static_cast<size_t>(nblocks) * sizeof(int_t),
                             cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(state->d_block_row_start[k], block_row_start,
                             static_cast<size_t>(nblocks) * sizeof(int_t),
                             cudaMemcpyHostToDevice));
        state->row_to_send_count[k] = row_count;
        state->block_count[k] = nblocks;
        return 0;
    }
