#pragma once

#include <cstddef>

#include "superlu_ddefs.h"

struct SymV2RowDownSendSegmentGPU
{
    size_t map_offset;
    int_t nrows;
    int_t dst_row_offset;
};

struct SymV2RowDownSeg
{
    int_t gid;
    int_t chunk_pc;
    int_t nrows;
    int_t dst_row_offset;
    int_t value_count;
    size_t map_offset;
};
