// This file is included inside xLUstruct_t under HAVE_CUDA.
// Keep declarations here limited to SymLDL V2 Pc-fragment partner-L state.

std::vector<Ftype *> symV2PartnerLSendBufsGPU;
std::vector<int_t *> symL2LSendMapsGPU;
Ftype *symV2PartnerLSendBufPoolGPU = NULL;
int_t *symL2LSendMapPoolGPU = NULL;
int_t *symV2PartnerLRecvMapPoolGPU = NULL;
size_t symV2PartnerLSendBufPoolCount = 0;
size_t symL2LSendMapPoolCount = 0;
size_t symV2PartnerLRecvMapPoolCount = 0;

std::vector<std::vector<int_t> > symL2LSendMeta;
std::vector<std::vector<Ftype> > symV2PartnerLHostSendBufs;
std::vector<Ftype *> symV2PartnerLHostSendBufsPinned;
std::vector<size_t> symV2PartnerLHostSendScratchOffsets;
std::vector<size_t> symV2PartnerLMapOffsets;
std::vector<int_t> symV2PartnerLPackedMaps;

Ftype *symV2PartnerLHostSendPoolPinned = NULL;
Ftype *symV2PartnerLHostRecvPoolPinned = NULL;
size_t symV2PartnerLHostSendPoolPinnedCount = 0;
size_t symV2PartnerLHostRecvPoolPinnedCount = 0;
int symV2PartnerLHostRecvPinned = 0;

std::vector<int> symV2PartnerLSendSizes;
std::vector<unsigned char> symV2PartnerLSendRowActive;
std::vector<unsigned char> symV2PartnerLPrepacked;
std::vector<int> symV2PartnerLRecvSizes;
std::vector<std::vector<int_t> > symV2PartnerLRecvIndex;
std::vector<std::vector<int_t> > symV2PartnerLRecvIndexBySrc;
std::vector<std::vector<int_t> > symV2PartnerLRecvMap;
std::vector<size_t> symV2PartnerLRecvMapOffsets;
std::vector<int_t *> symV2PartnerLRecvMapsGPU;
