// This file is included inside the SymLDL GPU solve C API implementation.
// Keep declarations here limited to solve-handle setup and panel metadata.

    dSymLDLSolveGPU_Handle dSymLDLSolveGPUCreate(int_t nsupers, int_t maxsup,
                                                 int_t max_panel_rows, int nrhs,
                                                 gridinfo3d_t *grid3d)
    {
        if (nsupers < 0 || maxsup < 0 || max_panel_rows < 0 || nrhs < 0)
            ABORT("Invalid SymLDL GPU solve workspace size.");

        int deviceCount = 0;
        gpuErrchk(cudaGetDeviceCount(&deviceCount));
        if (deviceCount <= 0)
            ABORT("SymLDL GPU solve requires a CUDA device.");
        int device_id = (grid3d != NULL) ? (grid3d->iam % deviceCount) : 0;
        gpuErrchk(cudaSetDevice(device_id));

        dSymLDLSolveGPUState *state = new dSymLDLSolveGPUState;
        state->nsupers = nsupers;
        state->nrhs = nrhs;
        state->d_lusup = new double *[static_cast<size_t>(nsupers)];
        state->lusup_count = new int_t[static_cast<size_t>(nsupers)];
        state->owns_lusup = new int[static_cast<size_t>(nsupers)];
        state->d_row_to_send_pos = new int *[static_cast<size_t>(nsupers)];
        state->row_to_send_count = new int_t[static_cast<size_t>(nsupers)];
        state->d_block_luptr = new int_t *[static_cast<size_t>(nsupers)];
        state->d_block_nbrow = new int_t *[static_cast<size_t>(nsupers)];
        state->d_block_row_start = new int_t *[static_cast<size_t>(nsupers)];
        state->block_count = new int_t[static_cast<size_t>(nsupers)];
        state->d_b = NULL;
        state->d_c = NULL;
        state->d_send_vals = NULL;
        state->d_row_values = NULL;
        state->d_delta = NULL;
        state->d_b_cap = 0;
        state->d_c_cap = 0;
        state->d_send_vals_cap = 0;
        state->d_row_values_cap = 0;
        state->d_delta_cap = 0;
        state->handle = NULL;
        state->stream = NULL;
        state->t_h2d = 0.0;
        state->t_compute = 0.0;
        state->t_d2h = 0.0;

        for (int_t k = 0; k < nsupers; ++k)
        {
            state->d_lusup[k] = NULL;
            state->lusup_count[k] = 0;
            state->owns_lusup[k] = 0;
            state->d_row_to_send_pos[k] = NULL;
            state->row_to_send_count[k] = 0;
            state->d_block_luptr[k] = NULL;
            state->d_block_nbrow[k] = NULL;
            state->d_block_row_start[k] = NULL;
            state->block_count[k] = 0;
        }

        gpuErrchk(cudaStreamCreate(&state->stream));
        gpublasCheckErrors(cublasCreate(&state->handle));
        gpublasCheckErrors(cublasSetStream(state->handle, state->stream));

        int_t max_rhs_rows = (maxsup > max_panel_rows) ? maxsup : max_panel_rows;
        int_t max_rhs_count = max_rhs_rows * static_cast<int_t>(nrhs);
        int_t max_out_count = max_rhs_count;
        if (max_rhs_count > 0)
            symldl_gpu_ensure_buffer(&state->d_b, &state->d_b_cap, max_rhs_count);
        if (max_out_count > 0)
            symldl_gpu_ensure_buffer(&state->d_c, &state->d_c_cap, max_out_count);

        return reinterpret_cast<dSymLDLSolveGPU_Handle>(state);
    }

    void dSymLDLSolveGPUDestroy(dSymLDLSolveGPU_Handle handle)
    {
        dSymLDLSolveGPUState *state =
            reinterpret_cast<dSymLDLSolveGPUState *>(handle);
        if (state == NULL)
            return;

        for (int_t k = 0; k < state->nsupers; ++k)
        {
            if (state->d_lusup[k] != NULL && state->owns_lusup[k])
                gpuErrchk(cudaFree(state->d_lusup[k]));
            if (state->d_row_to_send_pos[k] != NULL)
                gpuErrchk(cudaFree(state->d_row_to_send_pos[k]));
            if (state->d_block_luptr[k] != NULL)
                gpuErrchk(cudaFree(state->d_block_luptr[k]));
            if (state->d_block_nbrow[k] != NULL)
                gpuErrchk(cudaFree(state->d_block_nbrow[k]));
            if (state->d_block_row_start[k] != NULL)
                gpuErrchk(cudaFree(state->d_block_row_start[k]));
        }
        if (state->d_b != NULL)
            gpuErrchk(cudaFree(state->d_b));
        if (state->d_c != NULL)
            gpuErrchk(cudaFree(state->d_c));
        if (state->d_send_vals != NULL)
            gpuErrchk(cudaFree(state->d_send_vals));
        if (state->d_row_values != NULL)
            gpuErrchk(cudaFree(state->d_row_values));
        if (state->d_delta != NULL)
            gpuErrchk(cudaFree(state->d_delta));
        if (state->handle != NULL)
            gpublasCheckErrors(cublasDestroy(state->handle));
        if (state->stream != NULL)
            gpuErrchk(cudaStreamDestroy(state->stream));

        delete [] state->d_lusup;
        delete [] state->lusup_count;
        delete [] state->owns_lusup;
        delete [] state->d_row_to_send_pos;
        delete [] state->row_to_send_count;
        delete [] state->d_block_luptr;
        delete [] state->d_block_nbrow;
        delete [] state->d_block_row_start;
        delete [] state->block_count;
        delete state;
    }

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
