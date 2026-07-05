// This file is included inside the SymLDL GPU solve C API implementation.
// Keep declarations here limited to solve-local CUDA state, kernels, and helpers.

    struct dSymLDLSolveGPUState
    {
        int_t nsupers;
        int nrhs;
        double **d_lusup;
        int_t *lusup_count;
        int *owns_lusup;
        int **d_row_to_send_pos;
        int_t *row_to_send_count;
        int_t **d_block_luptr;
        int_t **d_block_nbrow;
        int_t **d_block_row_start;
        int_t *block_count;
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

    __global__ void symldl_forward_panel_kernel(
        const double *lusup, int_t nsupr, int_t ksupc, int nrhs,
        const int_t *block_luptr, const int_t *block_nbrow,
        const int_t *block_row_start, int_t nblocks,
        const int *row_to_send_pos, const double *xk, double *send_vals)
    {
        int_t block = static_cast<int_t>(blockIdx.x);
        int rhs = static_cast<int>(blockIdx.y);
        if (block >= nblocks || rhs >= nrhs)
            return;

        int_t nbrow = block_nbrow[block];
        int_t row_start = block_row_start[block];
        int_t luptr = block_luptr[block];
        for (int_t r = threadIdx.x; r < nbrow; r += blockDim.x) {
            double sum = 0.0;
            const double *a = lusup + luptr + r;
            const double *x = xk + static_cast<int_t>(rhs) * ksupc;
            for (int_t c = 0; c < ksupc; ++c)
                sum += a[c * nsupr] * x[c];
            int pos = row_to_send_pos[row_start + r];
            send_vals[static_cast<int_t>(pos) * nrhs + rhs] = -sum;
        }
    }

    __global__ void symldl_backward_panel_kernel(
        const double *lusup, int_t nsupr, int_t ksupc, int nrhs,
        const int_t *block_luptr, const int_t *block_nbrow,
        const int_t *block_row_start, int_t nblocks,
        const double *row_values, double *delta)
    {
        int_t block = static_cast<int_t>(blockIdx.x);
        int rhs = static_cast<int>(blockIdx.y);
        if (block >= nblocks || rhs >= nrhs)
            return;

        int_t row_start = block_row_start[block];
        int_t nbrow = block_nbrow[block];
        int_t luptr = block_luptr[block];
        for (int_t c = threadIdx.x; c < ksupc; c += blockDim.x) {
            double sum = 0.0;
            const double *a = lusup + luptr + c * nsupr;
            for (int_t r = 0; r < nbrow; ++r)
                sum += a[r] * row_values[(row_start + r) * nrhs + rhs];
            atomicAdd(&delta[c + static_cast<int_t>(rhs) * ksupc], -sum);
        }
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
