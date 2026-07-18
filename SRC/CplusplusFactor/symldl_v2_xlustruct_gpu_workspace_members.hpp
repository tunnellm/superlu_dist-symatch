// This file is included inside xLUstruct_t under HAVE_CUDA.
// Keep declarations here limited to SymLDL V2 GPU workspace state.

std::vector<int> symPanelReadyEventIds;
std::vector<int_t> symV2RawPanelNodes;

void *symV2LPanelArenaGPU = NULL;
void *symV2StreamArenaGPU = NULL;
void *symV2GemmArenaGPU = NULL;
size_t symV2LPanelArenaBytes = 0;
size_t symV2StreamArenaBytes = 0;
size_t symV2GemmArenaBytes = 0;
