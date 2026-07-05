// This file is included inside xLUstruct_t.
// Keep declarations here limited to SymLDL V2 state.

std::vector<Ftype *> symV2DiagBlocks;
#ifdef HAVE_CUDA
std::vector<Ftype *> symV2DiagBlocksGPU;
#endif

#ifdef HAVE_CUDA
std::vector<Ftype *> symV2PartnerLSendBufsGPU;
std::vector<int_t *> symL2LSendMapsGPU;
Ftype *symV2PartnerLSendBufPoolGPU = NULL;
int_t *symL2LSendMapPoolGPU = NULL;
int_t *symV2PartnerLRecvMapPoolGPU = NULL;
int_t *symV2RowFragRecvMapPoolGPU = NULL;
SymV2RowDownSendSegmentGPU *symV2RowDownSendSegPoolGPU = NULL;
size_t symV2PartnerLSendBufPoolCount = 0;
size_t symL2LSendMapPoolCount = 0;
size_t symV2PartnerLRecvMapPoolCount = 0;
size_t symV2RowFragRecvMapPoolCount = 0;
size_t symV2RowDownSendSegPoolCount = 0;
std::vector<SymV2RowDownSendSegmentGPU *> symV2RowDownSendSegsGPU;

std::vector<std::vector<int_t> > symL2LSendMeta;
std::vector<std::vector<Ftype> > symV2PartnerLHostSendBufs;
std::vector<Ftype *> symV2PartnerLHostSendBufsPinned;
std::vector<size_t> symV2PartnerLMapOffsets;
std::vector<int_t> symV2PartnerLPackedMaps;

std::vector<int> symV2RowDownSendSizes;
std::vector<SymV2RowDownSendSegmentGPU> symV2RowDownSendSegsHost;
std::vector<size_t> symV2RowDownSendSegOffsets;
std::vector<int> symV2RowDownSendSegCounts;
std::vector<size_t> symV2RowDownSegOffsets;
std::vector<SymV2RowDownSeg> symV2RowDownSegs;
std::vector<int> symV2RowDownRecvSizes;
std::vector<unsigned char> symV2RowDownPlanReady;

Ftype *symV2PartnerLHostSendPoolPinned = NULL;
Ftype *symV2PartnerLHostRecvPoolPinned = NULL;
Ftype *symV2RowFragHostRecvPoolPinned = NULL;
Ftype *symV2RowFragHostSendPoolPinned = NULL;
size_t symV2PartnerLHostSendPoolPinnedCount = 0;
size_t symV2PartnerLHostRecvPoolPinnedCount = 0;
size_t symV2RowFragHostRecvPoolPinnedCount = 0;
size_t symV2RowFragHostSendPoolPinnedCount = 0;
int symV2PartnerLHostRecvPinned = 0;
int symV2RowFragHostRecvPinned = 0;
int symV2RowFragHostSendPinned = 0;

std::vector<size_t> symV2PartnerLHostSendScratchOffsets;
std::vector<int> symV2ExchangeSendSizesScratch;
std::vector<int> symV2ExchangeRecvSizesScratch;
std::vector<int> symV2ExchangeRecvOffsetsScratch;
std::vector<MPI_Request> symV2ExchangeRecvReqsScratch;
std::vector<MPI_Request> symV2ExchangeSendReqsScratch;
std::vector<int> symV2ExchangeRecvPeersScratch;
std::vector<int> symV2ExchangeWaitIndicesScratch;
std::vector<MPI_Status> symV2ExchangeWaitStatusesScratch;
std::vector<int> symV2RowFragSendCountsScratch;
std::vector<int> symV2RowFragSendOffsetsScratch;
std::vector<MPI_Request> symV2RowFragSendReqsScratch;

std::vector<int> symV2PartnerLSendSizes;
std::vector<unsigned char> symV2PartnerLSendRowActive;
std::vector<unsigned char> symV2RowFragSendActive;
std::vector<unsigned char> symV2PartnerLPrepacked;
std::vector<int> symV2PartnerLRecvSizes;
std::vector<std::vector<int_t> > symV2PartnerLRecvIndex;
std::vector<std::vector<int_t> > symV2PartnerLRecvIndexBySrc;
std::vector<std::vector<int_t> > symV2PartnerLRecvMap;
std::vector<size_t> symV2PartnerLRecvMapOffsets;
std::vector<int_t *> symV2PartnerLRecvMapsGPU;
std::vector<int> symV2RowFragRecvSizes;
std::vector<std::vector<int_t> > symV2RowFragRecvIndex;
std::vector<std::vector<int_t> > symV2RowFragRecvMap;
std::vector<size_t> symV2RowFragRecvMapOffsets;
std::vector<int_t *> symV2RowFragRecvMapsGPU;
std::vector<int> symPanelReadyEventIds;
std::vector<unsigned char> symV2UsePcFragmentSchur;
std::vector<int_t> symV2RawPanelNodes;

void *symV2LPanelArenaGPU = NULL;
void *symV2StreamArenaGPU = NULL;
void *symV2GemmArenaGPU = NULL;
size_t symV2LPanelArenaBytes = 0;
size_t symV2StreamArenaBytes = 0;
size_t symV2GemmArenaBytes = 0;

int_t maxSymV2RowFragStageCount = 0;
int_t maxSymV2RowFragValRecvCount = 0;
int_t maxSymV2RowFragIdxRecvCount = 0;
int_t maxSymV2RowFragValSendCount = 0;
std::vector<Ftype *> symV2RowFragHostRecvBufs;
std::vector<Ftype *> symV2RowFragHostSendBufs;
#endif
