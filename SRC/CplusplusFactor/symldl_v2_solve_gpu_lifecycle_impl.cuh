// This file is included inside the SymLDL GPU solve C API implementation.
// Keep declarations here limited to solve-handle lifecycle.

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
