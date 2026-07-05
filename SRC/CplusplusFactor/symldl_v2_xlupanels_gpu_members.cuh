// This file is included inside xLUstructGPU_t.
// Keep declarations here limited to SymLDL V2 GPU-side struct members.

cudaEvent_t symV2PartnerLPackReadyEvents[MAX_CUDA_STREAMS];

Ftype* symPartnerLvalRecvBufs[MAX_CUDA_STREAMS];
Ftype* symPartnerLStageBufs[MAX_CUDA_STREAMS];
Ftype* symPartnerLSendStageBufs[MAX_CUDA_STREAMS];
Ftype* symV2RowFragStageBufs[MAX_CUDA_STREAMS];
Ftype* symV2RowFragValRecvBufs[MAX_CUDA_STREAMS];
int_t* symV2RowFragIdxRecvBufs[MAX_CUDA_STREAMS];
int_t* symV2RowFragSendMapStageBufs[MAX_CUDA_STREAMS];
Ftype* symV2RawPanelBufs[MAX_CUDA_STREAMS];
cudaEvent_t symV2RawPanelReadyEvents[MAX_CUDA_STREAMS];
int_t* symPartnerLidxRecvBufs[MAX_CUDA_STREAMS];

int useSymV2PanelIndex;
int_t *symV2PanelLocalIndex;
