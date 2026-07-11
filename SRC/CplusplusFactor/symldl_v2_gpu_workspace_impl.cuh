#pragma once

#include "symldl_v2_grid_runtime.h"
#include "symldl_v2_l_panel_gpu_copy_impl.cuh"
#include "symldl_v2_gpu_fragment_stream_buffers_impl.cuh"
#include "symldl_v2_gpu_stream_workspace_impl.cuh"
#include "symldl_v2_gpu_gemm_workspace_impl.cuh"
#include "symldl_v2_gpu_panel_index_impl.cuh"

static inline int symldl_v2_limit_gpu_streams(int streams)
{
    int selected_cap = dSymLDLV2GetCurrentGridGPUStreamCap();
    return selected_cap > 0 ? SUPERLU_MIN(streams, selected_cap) : streams;
}
