// This file is included inside the SymLDL GPU solve C API implementation.
// Keep declarations here limited to runtime solve panel operations.

    int dSymLDLSolveGPUGemm(dSymLDLSolveGPU_Handle handle, int_t k,
                            int_t a_offset, char transa, char transb,
                            int_t m_in, int_t n_in, int_t kdim_in,
                            double alpha, int_t lda_in, const double *b,
                            int_t ldb_in, double beta, double *c, int_t ldc_in)
    {
        dSymLDLSolveGPUState *state =
            reinterpret_cast<dSymLDLSolveGPUState *>(handle);
        if (state == NULL || k < 0 || k >= state->nsupers)
            return -1;
        if (m_in == 0 || n_in == 0)
            return 0;
        if (state->d_lusup[k] == NULL || a_offset < 0 ||
            a_offset >= state->lusup_count[k])
            return -2;

        int m = symldl_gpu_count_to_int(m_in);
        int n = symldl_gpu_count_to_int(n_in);
        int kdim = symldl_gpu_count_to_int(kdim_in);
        int lda = symldl_gpu_count_to_int(lda_in);
        int ldb = symldl_gpu_count_to_int(ldb_in);
        int ldc = symldl_gpu_count_to_int(ldc_in);
        int_t b_count = ldb_in * n_in;
        int_t c_count = ldc_in * n_in;

        if (b == NULL || c == NULL || b_count < 0 || c_count < 0)
            return -3;
        symldl_gpu_ensure_buffer(&state->d_b, &state->d_b_cap, b_count);
        symldl_gpu_ensure_buffer(&state->d_c, &state->d_c_cap, c_count);

        double t = SuperLU_timer_();
        if (b_count > 0)
            gpuErrchk(cudaMemcpyAsync(state->d_b, b,
                                      static_cast<size_t>(b_count) * sizeof(double),
                                      cudaMemcpyHostToDevice, state->stream));
        if (beta != 0.0 && c_count > 0)
            gpuErrchk(cudaMemcpyAsync(state->d_c, c,
                                      static_cast<size_t>(c_count) * sizeof(double),
                                      cudaMemcpyHostToDevice, state->stream));
        gpuErrchk(cudaStreamSynchronize(state->stream));
        state->t_h2d += SuperLU_timer_() - t;

        t = SuperLU_timer_();
        const double *a_dev = state->d_lusup[k] + a_offset;
        gpublasCheckErrors(cublasDgemm(state->handle, symldl_gpu_op(transa),
                                       symldl_gpu_op(transb), m, n, kdim,
                                       &alpha, a_dev, lda, state->d_b, ldb,
                                       &beta, state->d_c, ldc));
        gpuErrchk(cudaStreamSynchronize(state->stream));
        state->t_compute += SuperLU_timer_() - t;

        t = SuperLU_timer_();
        if (c_count > 0)
            gpuErrchk(cudaMemcpyAsync(c, state->d_c,
                                      static_cast<size_t>(c_count) * sizeof(double),
                                      cudaMemcpyDeviceToHost, state->stream));
        gpuErrchk(cudaStreamSynchronize(state->stream));
        state->t_d2h += SuperLU_timer_() - t;

        return 0;
    }

    int dSymLDLSolveGPUForwardPanel(dSymLDLSolveGPU_Handle handle, int_t k,
                                    int_t ksupc_in, int nrhs_in, int_t nsupr_in,
                                    int_t nblocks, const int_t *block_luptr,
                                    const int_t *block_nbrow,
                                    const int_t *block_row_start,
                                    const double *xk, int total_send,
                                    double *send_vals)
    {
        dSymLDLSolveGPUState *state =
            reinterpret_cast<dSymLDLSolveGPUState *>(handle);
        if (state == NULL || k < 0 || k >= state->nsupers)
            return -1;
        if (nblocks == 0 || total_send == 0)
            return 0;
        if (state->d_lusup[k] == NULL || state->d_row_to_send_pos[k] == NULL ||
            state->d_block_luptr[k] == NULL ||
            state->d_block_nbrow[k] == NULL ||
            state->d_block_row_start[k] == NULL ||
            xk == NULL || send_vals == NULL)
            return -2;
        if (state->block_count[k] != nblocks)
            return -3;

        int ksupc = symldl_gpu_count_to_int(ksupc_in);
        int nrhs = symldl_gpu_count_to_int(nrhs_in);
        int nsupr = symldl_gpu_count_to_int(nsupr_in);
        int_t xk_count = ksupc_in * static_cast<int_t>(nrhs);
        int_t send_count = static_cast<int_t>(total_send) * nrhs;

        symldl_gpu_ensure_buffer(&state->d_b, &state->d_b_cap, xk_count);
        symldl_gpu_ensure_buffer(&state->d_send_vals, &state->d_send_vals_cap,
                                 send_count);

        double t = SuperLU_timer_();
        gpuErrchk(cudaMemcpyAsync(state->d_b, xk,
                                  static_cast<size_t>(xk_count) * sizeof(double),
                                  cudaMemcpyHostToDevice, state->stream));
        gpuErrchk(cudaStreamSynchronize(state->stream));
        state->t_h2d += SuperLU_timer_() - t;

        t = SuperLU_timer_();
        const int threads = 256;
        dim3 grid(symldl_gpu_count_to_int(nblocks), nrhs);
        symldl_forward_panel_kernel<<<grid, threads, 0, state->stream>>>(
            state->d_lusup[k], nsupr, ksupc, nrhs,
            state->d_block_luptr[k], state->d_block_nbrow[k],
            state->d_block_row_start[k], nblocks,
            state->d_row_to_send_pos[k], state->d_b, state->d_send_vals);
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaStreamSynchronize(state->stream));
        state->t_compute += SuperLU_timer_() - t;

        t = SuperLU_timer_();
        gpuErrchk(cudaMemcpyAsync(send_vals, state->d_send_vals,
                                  static_cast<size_t>(send_count) * sizeof(double),
                                  cudaMemcpyDeviceToHost, state->stream));
        gpuErrchk(cudaStreamSynchronize(state->stream));
        state->t_d2h += SuperLU_timer_() - t;

        (void) block_luptr;
        (void) block_nbrow;
        (void) block_row_start;
        return 0;
    }

    int dSymLDLSolveGPUBackwardPanel(dSymLDLSolveGPU_Handle handle, int_t k,
                                     int_t ksupc_in, int nrhs_in, int_t nsupr_in,
                                     int_t nblocks, const int_t *block_luptr,
                                     const int_t *block_nbrow,
                                     int_t row_count,
                                     const double *row_values,
                                     double *delta_send)
    {
        dSymLDLSolveGPUState *state =
            reinterpret_cast<dSymLDLSolveGPUState *>(handle);
        if (state == NULL || k < 0 || k >= state->nsupers)
            return -1;
        if (nblocks == 0 || row_count == 0)
            return 0;
        if (state->d_lusup[k] == NULL || state->d_block_luptr[k] == NULL ||
            state->d_block_nbrow[k] == NULL ||
            state->d_block_row_start[k] == NULL ||
            row_values == NULL || delta_send == NULL)
            return -2;
        if (state->block_count[k] != nblocks)
            return -3;

        int ksupc = symldl_gpu_count_to_int(ksupc_in);
        int nrhs = symldl_gpu_count_to_int(nrhs_in);
        int nsupr = symldl_gpu_count_to_int(nsupr_in);
        int_t row_values_count = row_count * static_cast<int_t>(nrhs);
        int_t delta_count = ksupc_in * static_cast<int_t>(nrhs);

        symldl_gpu_ensure_buffer(&state->d_row_values,
                                 &state->d_row_values_cap, row_values_count);
        symldl_gpu_ensure_buffer(&state->d_delta, &state->d_delta_cap,
                                 delta_count);

        double t = SuperLU_timer_();
        gpuErrchk(cudaMemcpyAsync(state->d_row_values, row_values,
                                  static_cast<size_t>(row_values_count) *
                                      sizeof(double),
                                  cudaMemcpyHostToDevice, state->stream));
        gpuErrchk(cudaStreamSynchronize(state->stream));
        state->t_h2d += SuperLU_timer_() - t;

        t = SuperLU_timer_();
        gpuErrchk(cudaMemsetAsync(state->d_delta, 0,
                                  static_cast<size_t>(delta_count) *
                                      sizeof(double),
                                  state->stream));
        const int threads = 256;
        dim3 grid(symldl_gpu_count_to_int(nblocks), nrhs);
        symldl_backward_panel_kernel<<<grid, threads, 0, state->stream>>>(
            state->d_lusup[k], nsupr, ksupc, nrhs,
            state->d_block_luptr[k], state->d_block_nbrow[k],
            state->d_block_row_start[k], nblocks, state->d_row_values,
            state->d_delta);
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaStreamSynchronize(state->stream));
        state->t_compute += SuperLU_timer_() - t;

        (void) block_luptr;
        (void) block_nbrow;

        t = SuperLU_timer_();
        gpuErrchk(cudaMemcpyAsync(delta_send, state->d_delta,
                                  static_cast<size_t>(delta_count) * sizeof(double),
                                  cudaMemcpyDeviceToHost, state->stream));
        gpuErrchk(cudaStreamSynchronize(state->stream));
        state->t_d2h += SuperLU_timer_() - t;
        return 0;
    }
