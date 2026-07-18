// This file is included inside xLUstruct_t under HAVE_CUDA.
// Keep declarations here limited to SymLDL V2 Pc-fragment GPU
// materialization and runtime state.

int_t *symV2RowFragRecvMapPoolGPU = NULL;
SymV2RowDownSendSegmentGPU *symV2RowDownSendSegPoolGPU = NULL;
size_t symV2RowFragRecvMapPoolCount = 0;
size_t symV2RowDownSendSegPoolCount = 0;
std::vector<SymV2RowDownSendSegmentGPU *> symV2RowDownSendSegsGPU;

std::vector<SymV2RowDownSendSegmentGPU> symV2RowDownSendSegsHost;
std::vector<size_t> symV2RowDownSendSegOffsets;
std::vector<int> symV2RowDownSendSegCounts;

Ftype *symV2RowFragHostRecvPoolPinned = NULL;
Ftype *symV2RowFragHostSendPoolPinned = NULL;
size_t symV2RowFragHostRecvPoolPinnedCount = 0;
size_t symV2RowFragHostSendPoolPinnedCount = 0;
int symV2RowFragHostRecvPinned = 0;
int symV2RowFragHostSendPinned = 0;

std::vector<unsigned char> symV2RowFragSendActive;
std::vector<std::vector<int_t> > symV2RowFragRecvMap;
std::vector<size_t> symV2RowFragRecvMapOffsets;
std::vector<int_t *> symV2RowFragRecvMapsGPU;

std::vector<Ftype *> symV2RowFragHostRecvBufs;
std::vector<Ftype *> symV2RowFragHostSendBufs;
