#ifndef SYMLDL_V2_GRID_RUNTIME_H
#define SYMLDL_V2_GRID_RUNTIME_H

#ifdef __cplusplus
extern "C" {
#endif

void dSymLDLV2SetCurrentGridGPUStreamCap(int streams);

int dSymLDLV2GetCurrentGridGPUStreamCap(void);

void dSymLDLV2SetCurrentGridGPUStreamsUsed(int streams);

int dSymLDLV2GetCurrentGridGPUStreamsUsed(void);

#ifdef __cplusplus
}
#endif

#endif
