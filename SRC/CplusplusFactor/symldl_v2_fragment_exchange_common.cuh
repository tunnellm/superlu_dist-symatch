#pragma once

#include <climits>
#include <cstdio>
#include <limits>

#include "xlupanels.hpp"
#include "symldl_v2_config.hpp"

#ifdef HAVE_CUDA

static inline int symldl_v2_mpi_count(size_t count, const char *what)
{
    if (count > static_cast<size_t>(INT_MAX))
        ABORT(what);
    return static_cast<int>(count);
}

static inline int symldl_v2_mpi_count(int_t count, const char *what)
{
    if (count < 0)
        ABORT(what);
    return symldl_v2_mpi_count(static_cast<size_t>(count), what);
}

static inline size_t symldl_v2_square_count(int_t n, const char *what)
{
    if (n < 0)
        ABORT(what);
    size_t dim = static_cast<size_t>(n);
    if (dim != 0 && dim > static_cast<size_t>(-1) / dim)
        ABORT(what);
    return dim * dim;
}

static inline size_t symldl_v2_square_bytes(int_t n, size_t elem_size,
                                            const char *what)
{
    size_t count = symldl_v2_square_count(n, what);
    if (elem_size != 0 && count > static_cast<size_t>(-1) / elem_size)
        ABORT(what);
    return count * elem_size;
}

static __global__ void symldl_v2_lfrag_pack_kernel(const double *lpanel,
                                                   double *sendbuf,
                                                   const int_t *sendmap,
                                                   int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;

    int_t src = sendmap[idx];
    sendbuf[idx] = (src < 0) ? (double) (-src - 1) : lpanel[src];
}

static __global__ void symldl_v2_lfrag_pack_raw_kernel(
    const double *lpanel, double *sendbuf, const int_t *sendmap, int count,
    int_t panel_ld, const double *diag, int_t diag_ld)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;

    int_t src = sendmap[idx];
    if (src < 0)
    {
        sendbuf[idx] = (double) (-src - 1);
        return;
    }

    int_t row = src % panel_ld;
    int_t col = src / panel_ld;
    double sum = 0.0;
    for (int_t p = 0; p < diag_ld; ++p)
        sum += lpanel[row + p * panel_ld] * diag[p + col * diag_ld];
    sendbuf[idx] = sum;
}

static __global__ void symldl_v2_row_down_pack_segments_kernel(
    const double *lpanel,
    double *sendbuf,
    const int_t *base_sendmap,
    size_t sendmap_base,
    const SymV2RowDownSendSegmentGPU *segments,
    int nsegments,
    int_t ksupc,
    int_t dst_lda)
{
    int seg_id = blockIdx.x;
    if (seg_id >= nsegments)
        return;

    SymV2RowDownSendSegmentGPU seg = segments[seg_id];
    int_t nrows = seg.nrows;
    if (nrows <= 0 || ksupc <= 0 || dst_lda <= 0)
        return;

    int_t count = nrows * ksupc;
    for (int_t idx = threadIdx.x; idx < count; idx += blockDim.x)
    {
        int_t row = idx % nrows;
        int_t col = idx / nrows;
        size_t map_pos = (seg.map_offset - sendmap_base) +
                         static_cast<size_t>(row) +
                         static_cast<size_t>(col) *
                             static_cast<size_t>(nrows);
        int_t src = base_sendmap[map_pos];
        sendbuf[seg.dst_row_offset + row + col * dst_lda] =
            (src < 0) ? (double) (-src - 1) : lpanel[src];
    }
}

static inline void symldl_v2_trace_pcfrag_exchange(
    gridinfo3d_t *grid3d, int_t k, int stream_offset, int_t partner_rows,
    int partner_recv_total, int_t row_rows, int row_recv_total,
    int row_send_total)
{
    if (!superlu_sym_v2_trace_pcfrag())
        return;

    static int printed = 0;
    if (printed >= 64)
        return;

    std::fprintf(stderr,
                 "[symv2-pcfrag] rank %d exchange k %d stream %d partner_rows %lld partner_recv %d row_rows %lld row_recv %d row_send %d\n",
                 grid3d != NULL ? grid3d->iam : -1,
                 static_cast<int>(k), stream_offset,
                 static_cast<long long>(partner_rows), partner_recv_total,
                 static_cast<long long>(row_rows), row_recv_total,
                 row_send_total);
    std::fflush(stderr);
    ++printed;
}

static __global__ void symldl_v2_lfrag_assemble_kernel(
    const double *stage,
    double *frag,
    const int_t *recv_map,
    int pieces,
    int_t ksupc,
    int_t frag_lda)
{
    int piece = blockIdx.x;
    if (piece >= pieces)
        return;

    int_t dst_offset = recv_map[3 * piece];
    int_t nrows = recv_map[3 * piece + 1];
    int_t src_offset = recv_map[3 * piece + 2];
    int_t count = nrows * ksupc;
    for (int_t idx = threadIdx.x; idx < count; idx += blockDim.x)
    {
        int_t row = idx % nrows;
        int_t col = idx / nrows;
        frag[dst_offset + row + col * frag_lda] =
            stage[src_offset + row + col * nrows];
    }
}

#endif
