// This file is included inside xLUstruct_t.
// Keep declarations here limited to backend-neutral SymLDL V2 plan state.

std::vector<unsigned char> symV2UsePcFragmentSchur;

std::vector<std::vector<int_t> > symL2LSendMeta;
std::vector<int> symV2PartnerLSendSizes;
std::vector<int> symV2PartnerLRecvSizes;
std::vector<std::vector<int_t> > symV2PartnerLRecvIndex;
std::vector<std::vector<int_t> > symV2PartnerLRecvIndexBySrc;

unsigned long long symV2PartnerMetadataGatherCalls = 0;
unsigned long long symV2PartnerMetadataLocalBytes = 0;
unsigned long long symV2PartnerMetadataReceivedBytes = 0;
double symV2PartnerMetadataGatherTime = 0.0;
unsigned long long symV2PartnerMetadataRowReceivedBytes = 0;
unsigned long long symV2PartnerMetadataColumnReceivedBytes = 0;
unsigned long long symV2PartnerMetadataScopedPeakBytes = 0;
double symV2PartnerMetadataRowGatherTime = 0.0;
double symV2PartnerMetadataColumnGatherTime = 0.0;

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
