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

enum SymV2RouteProfileCounter
{
    SYM_V2_ROUTE_PANEL_BCAST = 0,
    SYM_V2_ROUTE_PCFRAG_PANEL_BCAST,
    SYM_V2_ROUTE_LFRAG_EXCHANGE,
    SYM_V2_ROUTE_PCFRAG_EXCHANGE,
    SYM_V2_ROUTE_PARTNER_L_EXCHANGE,
    SYM_V2_ROUTE_ROWFRAG_EXCHANGE,
    SYM_V2_ROUTE_DUAL_FRAGMENT_LOOKAHEAD,
    SYM_V2_ROUTE_DUAL_FRAGMENT_EXCLUDE,
    SYM_V2_ROUTE_PROFILE_COUNTERS
};

struct SymV2RouteProfile
{
    long long counters[SYM_V2_ROUTE_PROFILE_COUNTERS] = {0};
};
