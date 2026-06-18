
#include "superlu_upacked.h"
#include "lupanels.hpp"
#include "xlupanels.hpp"
#include "lupanels_impl.hpp"
#include "pdgstrf3d_upacked_impl.hpp" //unneeded?


extern "C"
{

    dLUgpu_Handle dCreateLUgpuHandle(int_t nsupers, int_t ldt_, dtrf3Dpartition_t *trf3Dpartition,
                                     dLUstruct_t *LUstruct, gridinfo3d_t *grid3d,
                                     SCT_t *SCT_, superlu_dist_options_t *options_, SuperLUStat_t *stat,
                                     double thresh_, int *info_)
    {
#if (DEBUGlevel >= 1)
        CHECK_MALLOC(grid3d->iam, "Enter createLUgpuHandle");
#endif

        xLUstruct_t<double> *instance = new xLUstruct_t<double>(nsupers, ldt_, trf3Dpartition,
                                                                    LUstruct, grid3d,
                                                                    SCT_, options_, stat,
                                                                    thresh_, info_);

        return reinterpret_cast<dLUgpu_Handle>(instance);

#if (DEBUGlevel >= 1)
        CHECK_MALLOC(grid3d->iam, "Exit createLUgpuHandle");
#endif
    }

    void dDestroyLUgpuHandle(dLUgpu_Handle LuH)
    {
	// printf("\t... before delete luH\n"); fflush(stdout);

        delete reinterpret_cast<xLUstruct_t<double> *>(LuH);
	
        // printf("\t... after delete luH\n"); fflush(stdout);
    }

    // I think the following is not used 
    int dGatherFactoredLU3Dto2D(dLUgpu_Handle LuH);

    int dCopyLUGPU2Host(dLUgpu_Handle LuH, dLUstruct_t *LUstruct)
    {
        
        xLUstruct_t<double> *LU_v1 = reinterpret_cast<xLUstruct_t<double> *>(LuH);
        double tXferGpu2Host = SuperLU_timer_();
        if (LU_v1->superlu_acc_offload)
        {
#ifdef HAVE_CUDA
            cudaStreamSynchronize(LU_v1->A_gpu.cuStreams[0]); // in theory I don't need it
            LU_v1->copyLUGPUtoHost();
#endif
        }

        if (!LU_v1->useSymV2Solve())
            LU_v1->packedU2skyline(LUstruct);
        tXferGpu2Host = SuperLU_timer_() - tXferGpu2Host;
#if ( PRNTlevel >= 1 )	
        printf("Time to send data back= %g\n", tXferGpu2Host);
#endif
        return 0;
    }

    int pdgstrf3d_LUv1(dLUgpu_Handle LUHand) // pdgstrf3d_Upacked 
    {
        xLUstruct_t<double> *LU_v1 = reinterpret_cast<xLUstruct_t<double> *>(LUHand);
        return LU_v1->pdgstrf3d();
        
    }

	int pdgstrf3d_LUv2(dLUgpu_Handle LUHand)
	{
	    xLUstruct_t<double> *LU_v2 = reinterpret_cast<xLUstruct_t<double> *>(LUHand);
	    return LU_v2->pdgstrf3dSymV2();
	}

    struct dSymLDLSolveGPUState
    {
        int_t nsupers;
        int nrhs;
        double **d_lusup;
        int_t *lusup_count;
        int **d_row_to_send_pos;
        int_t *row_to_send_count;
        double *d_b;
        double *d_c;
        double *d_send_vals;
        double *d_row_values;
        double *d_delta;
        int_t d_b_cap;
        int_t d_c_cap;
        int_t d_send_vals_cap;
        int_t d_row_values_cap;
        int_t d_delta_cap;
        cublasHandle_t handle;
        cudaStream_t stream;
        double t_h2d;
        double t_compute;
        double t_d2h;
    };

    __global__ void symldl_scatter_forward_send_kernel(
        const double *gemm, double *send_vals, const int *row_to_send_pos,
        int_t row_start, int_t nbrow, int nrhs)
    {
        int_t idx = static_cast<int_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        int_t count = nbrow * static_cast<int_t>(nrhs);
        if (idx >= count)
            return;

        int_t r = idx % nbrow;
        int rhs = static_cast<int>(idx / nbrow);
        int pos = row_to_send_pos[row_start + r];
        send_vals[static_cast<int_t>(pos) * nrhs + rhs] =
            -gemm[r + static_cast<int_t>(rhs) * nbrow];
    }

    __global__ void symldl_pack_backward_rows_kernel(
        const double *row_values, double *rhs, int_t row_start, int_t nbrow,
        int nrhs)
    {
        int_t idx = static_cast<int_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        int_t count = nbrow * static_cast<int_t>(nrhs);
        if (idx >= count)
            return;

        int_t r = idx % nbrow;
        int rhs_id = static_cast<int>(idx / nbrow);
        rhs[r + static_cast<int_t>(rhs_id) * nbrow] =
            row_values[(row_start + r) * nrhs + rhs_id];
    }

    static int symldl_gpu_count_to_int(int_t value)
    {
        int out = static_cast<int>(value);
        if (value < 0 || static_cast<int_t>(out) != value)
            ABORT("SymLDL GPU solve count overflows int.");
        return out;
    }

    static cublasOperation_t symldl_gpu_op(char trans)
    {
        return (trans == 'T' || trans == 't') ? CUBLAS_OP_T : CUBLAS_OP_N;
    }

    static void symldl_gpu_ensure_buffer(double **buffer, int_t *cap, int_t count)
    {
        if (count <= *cap)
            return;
        if (*buffer != NULL)
            gpuErrchk(cudaFree(*buffer));
        gpuErrchk(cudaMalloc(reinterpret_cast<void **>(buffer),
                             static_cast<size_t>(count) * sizeof(double)));
        *cap = count;
    }

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
        state->d_row_to_send_pos = new int *[static_cast<size_t>(nsupers)];
        state->row_to_send_count = new int_t[static_cast<size_t>(nsupers)];
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
            state->d_row_to_send_pos[k] = NULL;
            state->row_to_send_count[k] = 0;
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
            if (state->d_lusup[k] != NULL)
                gpuErrchk(cudaFree(state->d_lusup[k]));
            if (state->d_row_to_send_pos[k] != NULL)
                gpuErrchk(cudaFree(state->d_row_to_send_pos[k]));
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
        delete [] state->d_row_to_send_pos;
        delete [] state->row_to_send_count;
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

        if (state->d_lusup[k] != NULL)
            gpuErrchk(cudaFree(state->d_lusup[k]));
        gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&state->d_lusup[k]),
                             static_cast<size_t>(count) * sizeof(double)));
        gpuErrchk(cudaMemcpy(state->d_lusup[k], lusup,
                             static_cast<size_t>(count) * sizeof(double),
                             cudaMemcpyHostToDevice));
        state->lusup_count[k] = count;
        return 0;
    }

    int dSymLDLSolveGPUSetPanelSchedule(dSymLDLSolveGPU_Handle handle, int_t k,
                                        const int *row_to_send_pos,
                                        int_t row_count)
    {
        dSymLDLSolveGPUState *state =
            reinterpret_cast<dSymLDLSolveGPUState *>(handle);
        if (state == NULL || k < 0 || k >= state->nsupers)
            return -1;
        if (row_count <= 0 || row_to_send_pos == NULL)
            return 0;

        if (state->d_row_to_send_pos[k] != NULL)
            gpuErrchk(cudaFree(state->d_row_to_send_pos[k]));
        gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&state->d_row_to_send_pos[k]),
                             static_cast<size_t>(row_count) * sizeof(int)));
        gpuErrchk(cudaMemcpy(state->d_row_to_send_pos[k], row_to_send_pos,
                             static_cast<size_t>(row_count) * sizeof(int),
                             cudaMemcpyHostToDevice));
        state->row_to_send_count[k] = row_count;
        return 0;
    }

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
            block_luptr == NULL || block_nbrow == NULL ||
            block_row_start == NULL || xk == NULL || send_vals == NULL)
            return -2;

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
        const double alpha = 1.0;
        const double beta = 0.0;
        const int threads = 256;
        for (int_t block = 0; block < nblocks; ++block)
        {
            int_t nbrow_in = block_nbrow[block];
            int nbrow = symldl_gpu_count_to_int(nbrow_in);
            int_t out_count = nbrow_in * static_cast<int_t>(nrhs);
            symldl_gpu_ensure_buffer(&state->d_c, &state->d_c_cap, out_count);
            const double *a_dev = state->d_lusup[k] + block_luptr[block];
            gpublasCheckErrors(cublasDgemm(state->handle, CUBLAS_OP_N,
                                           CUBLAS_OP_N, nbrow, nrhs, ksupc,
                                           &alpha, a_dev, nsupr,
                                           state->d_b, ksupc, &beta,
                                           state->d_c, nbrow));
            int grid = static_cast<int>((out_count + threads - 1) / threads);
            symldl_scatter_forward_send_kernel<<<grid, threads, 0,
                state->stream>>>(state->d_c, state->d_send_vals,
                                 state->d_row_to_send_pos[k],
                                 block_row_start[block], nbrow_in, nrhs);
            gpuErrchk(cudaGetLastError());
        }
        gpuErrchk(cudaStreamSynchronize(state->stream));
        state->t_compute += SuperLU_timer_() - t;

        t = SuperLU_timer_();
        gpuErrchk(cudaMemcpyAsync(send_vals, state->d_send_vals,
                                  static_cast<size_t>(send_count) * sizeof(double),
                                  cudaMemcpyDeviceToHost, state->stream));
        gpuErrchk(cudaStreamSynchronize(state->stream));
        state->t_d2h += SuperLU_timer_() - t;
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
        if (state->d_lusup[k] == NULL || block_luptr == NULL ||
            block_nbrow == NULL || row_values == NULL || delta_send == NULL)
            return -2;

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
        const double alpha = -1.0;
        const double beta = 1.0;
        const int threads = 256;
        int_t row_start = 0;
        for (int_t block = 0; block < nblocks; ++block)
        {
            int_t nbrow_in = block_nbrow[block];
            int nbrow = symldl_gpu_count_to_int(nbrow_in);
            int_t rhs_count = nbrow_in * static_cast<int_t>(nrhs);
            symldl_gpu_ensure_buffer(&state->d_b, &state->d_b_cap, rhs_count);
            int grid = static_cast<int>((rhs_count + threads - 1) / threads);
            symldl_pack_backward_rows_kernel<<<grid, threads, 0,
                state->stream>>>(state->d_row_values, state->d_b, row_start,
                                 nbrow_in, nrhs);
            gpuErrchk(cudaGetLastError());
            const double *a_dev = state->d_lusup[k] + block_luptr[block];
            gpublasCheckErrors(cublasDgemm(state->handle, CUBLAS_OP_T,
                                           CUBLAS_OP_N, ksupc, nrhs, nbrow,
                                           &alpha, a_dev, nsupr,
                                           state->d_b, nbrow, &beta,
                                           state->d_delta, ksupc));
            row_start += nbrow_in;
        }
        gpuErrchk(cudaStreamSynchronize(state->stream));
        state->t_compute += SuperLU_timer_() - t;

        t = SuperLU_timer_();
        gpuErrchk(cudaMemcpyAsync(delta_send, state->d_delta,
                                  static_cast<size_t>(delta_count) * sizeof(double),
                                  cudaMemcpyDeviceToHost, state->stream));
        gpuErrchk(cudaStreamSynchronize(state->stream));
        state->t_d2h += SuperLU_timer_() - t;
        return 0;
    }

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


    // Single precision:
    sLUgpu_Handle sCreateLUgpuHandle(int_t nsupers, int_t ldt_, strf3Dpartition_t *trf3Dpartition,
                                     sLUstruct_t *LUstruct, gridinfo3d_t *grid3d,
                                     SCT_t *SCT_, superlu_dist_options_t *options_, SuperLUStat_t *stat,
                                     float thresh_, int *info_)
    {
        #if (DEBUGlevel >= 1)
        CHECK_MALLOC(grid3d->iam, "Enter createLUgpuHandle");
        #endif

        xLUstruct_t<float> *instance = new xLUstruct_t<float>(nsupers, ldt_, trf3Dpartition,
                                                                    LUstruct, grid3d,
                                                                    SCT_, options_, stat,
                                                                    thresh_, info_);
        
        return reinterpret_cast<sLUgpu_Handle>(instance);
    }

    void sDestroyLUgpuHandle(sLUgpu_Handle LuH)
    {
        //printf("\t... before delete luH\n"); fflush(stdout);
        delete reinterpret_cast<xLUstruct_t<float> *>(LuH);
	
	// printf("\t... after delete luH\n"); fflush(stdout);
    }

    // I think the following is not used
    int sGatherFactoredLU3Dto2D(sLUgpu_Handle LuH);

    int sCopyLUGPU2Host(sLUgpu_Handle LuH, sLUstruct_t *LUstruct)
    {
        
        xLUstruct_t<float> *LU_v1 = reinterpret_cast<xLUstruct_t<float> *>(LuH);
        double tXferGpu2Host = SuperLU_timer_();
        if (LU_v1->superlu_acc_offload)
        {
#ifdef HAVE_CUDA
            cudaStreamSynchronize(LU_v1->A_gpu.cuStreams[0]); // in theory I don't need it
            LU_v1->copyLUGPUtoHost();
#endif
        }

        if (!LU_v1->useSymV2Solve())
            LU_v1->packedU2skyline(LUstruct);
        tXferGpu2Host = SuperLU_timer_() - tXferGpu2Host;
#if ( PRNTlevel >= 1 )
        printf("Time to send data back= %g\n", tXferGpu2Host);
#endif
        return 0;
    }

    int psgstrf3d_LUv1(sLUgpu_Handle LUHand) // pdgstrf3d_Upacked 
    {
        
        xLUstruct_t<float> *LU_v1 = reinterpret_cast<xLUstruct_t<float> *>(LUHand);
        return LU_v1->pdgstrf3d();
        
    }


    //  Double Complex precision:
    zLUgpu_Handle zCreateLUgpuHandle(int_t nsupers, int_t ldt_, ztrf3Dpartition_t *trf3Dpartition,
                                     zLUstruct_t *LUstruct, gridinfo3d_t *grid3d,
                                     SCT_t *SCT_, superlu_dist_options_t *options_, SuperLUStat_t *stat,
                                     double thresh_, int *info_)
    {
        #if (DEBUGlevel >= 1)
        CHECK_MALLOC(grid3d->iam, "Enter createLUgpuHandle");
        #endif

        xLUstruct_t<doublecomplex> *instance = new xLUstruct_t<doublecomplex>(nsupers, ldt_, trf3Dpartition,
                                                                    LUstruct, grid3d,
                                                                    SCT_, options_, stat,
                                                                    thresh_, info_);
        
        return reinterpret_cast<zLUgpu_Handle>(instance);
    } 

    void zDestroyLUgpuHandle(zLUgpu_Handle LuH)
    {
        // printf("\t... before delete luH\n");  fflush(stdout);
	
        delete reinterpret_cast<xLUstruct_t<doublecomplex> *>(LuH);
	
        // printf("\t... after delete luH\n"); fflush(stdout);
    }

    // I think the following is not used
    int zGatherFactoredLU3Dto2D(zLUgpu_Handle LuH);

    int zCopyLUGPU2Host(zLUgpu_Handle LuH, zLUstruct_t *LUstruct)
    {
        
        xLUstruct_t<doublecomplex> *LU_v1 = reinterpret_cast<xLUstruct_t<doublecomplex> *>(LuH);
        double tXferGpu2Host = SuperLU_timer_();
        if (LU_v1->superlu_acc_offload)
        {
#ifdef HAVE_CUDA
            cudaStreamSynchronize(LU_v1->A_gpu.cuStreams[0]); // in theory I don't need it
            LU_v1->copyLUGPUtoHost();
#endif
        }

        if (!LU_v1->useSymV2Solve())
            LU_v1->packedU2skyline(LUstruct);
        tXferGpu2Host = SuperLU_timer_() - tXferGpu2Host;
#if ( PRNTlevel >= 1 )
        printf("Time to send data back= %g\n", tXferGpu2Host);
#endif
        return 0;
    }

    int pzgstrf3d_LUv1(zLUgpu_Handle LUHand) // pdgstrf3d_Upacked 
    {
        
        xLUstruct_t<doublecomplex> *LU_v1 = reinterpret_cast<xLUstruct_t<doublecomplex> *>(LUHand);
        return LU_v1->pdgstrf3d();
        
    }
}
