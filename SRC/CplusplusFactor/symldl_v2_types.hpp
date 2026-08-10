#pragma once

#include <cstddef>

#include "superlu_ddefs.h"

enum SymLDLV2FactorBackend
{
    SYM_LDL_V2_BACKEND_NONE = 0,
    SYM_LDL_V2_BACKEND_CPU,
    SYM_LDL_V2_BACKEND_GPU
};

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
    SYM_V2_ROUTE_GPU_PR1_FULL_PANEL,
    SYM_V2_ROUTE_GPU_PR1_DUAL_FRAGMENT,
    SYM_V2_ROUTE_PROFILE_COUNTERS
};

struct SymV2RouteProfile
{
    long long counters[SYM_V2_ROUTE_PROFILE_COUNTERS] = {0};
};

struct SymV2CommunicationProfile
{
    unsigned long long partner_send_messages = 0;
    unsigned long long partner_send_bytes = 0;
    unsigned long long partner_max_message_bytes = 0;
    unsigned long long row_send_messages = 0;
    unsigned long long row_send_bytes = 0;
    unsigned long long row_max_message_bytes = 0;
};
