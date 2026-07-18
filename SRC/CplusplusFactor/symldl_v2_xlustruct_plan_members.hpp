// This file is included inside xLUstruct_t.
// Keep declarations here limited to backend-neutral SymLDL V2 plan state.

std::vector<std::vector<int_t> > symL2LSendMeta;
std::vector<int> symV2PartnerLSendSizes;
std::vector<int> symV2PartnerLRecvSizes;
std::vector<std::vector<int_t> > symV2PartnerLRecvIndex;
std::vector<std::vector<int_t> > symV2PartnerLRecvIndexBySrc;

std::vector<int> symV2RowDownSendSizes;
std::vector<size_t> symV2RowDownSegOffsets;
std::vector<SymV2RowDownSeg> symV2RowDownSegs;
std::vector<int> symV2RowDownRecvSizes;
std::vector<unsigned char> symV2RowDownPlanReady;

std::vector<int> symV2RowFragRecvSizes;
std::vector<std::vector<int_t> > symV2RowFragRecvIndex;

int_t maxSymV2RowFragStageCount = 0;
int_t maxSymV2RowFragValRecvCount = 0;
int_t maxSymV2RowFragIdxRecvCount = 0;
int_t maxSymV2RowFragValSendCount = 0;
