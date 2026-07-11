#include "symldl_v2_grid_runtime.h"

static int dSymLDLV2CurrentGridGPUStreamCap;
static int dSymLDLV2CurrentGridGPUStreamsUsed;

void dSymLDLV2SetCurrentGridGPUStreamCap(int streams)
{
    dSymLDLV2CurrentGridGPUStreamCap = streams > 0 ? streams : 0;
}

int dSymLDLV2GetCurrentGridGPUStreamCap(void)
{
    return dSymLDLV2CurrentGridGPUStreamCap;
}

void dSymLDLV2SetCurrentGridGPUStreamsUsed(int streams)
{
    dSymLDLV2CurrentGridGPUStreamsUsed = streams > 0 ? streams : 0;
}

int dSymLDLV2GetCurrentGridGPUStreamsUsed(void)
{
    return dSymLDLV2CurrentGridGPUStreamsUsed;
}
