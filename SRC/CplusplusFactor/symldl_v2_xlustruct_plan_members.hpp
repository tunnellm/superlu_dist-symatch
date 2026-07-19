// This file is included inside xLUstruct_t.
// Keep declarations here limited to backend-neutral SymLDL V2 plan state.

std::vector<unsigned char> symV2UsePcFragmentSchur;

std::vector<std::vector<int_t> > symL2LSendMeta;
std::vector<int> symV2PartnerLSendSizes;
std::vector<unsigned char> symV2PartnerLSendRowActive;
std::vector<int> symV2PartnerLRecvSizes;
std::vector<unsigned char> symV2PartnerLRecvActive;
std::vector<std::vector<int_t> > symV2PartnerLRecvIndex;
std::vector<std::vector<int_t> > symV2PartnerLRecvIndexBySrc;

uint64_t symV2PartnerCandidateRemoteRecipients = 0;
uint64_t symV2PartnerActiveRemoteRecipients = 0;
uint64_t symV2PartnerCandidateRemoteValues = 0;
uint64_t symV2PartnerActiveRemoteValues = 0;
uint64_t symV2PartnerActiveSelfRecipients = 0;
uint64_t symV2PartnerDemandRecords = 0;
uint64_t symV2PartnerDemandPayloadBytes = 0;
uint64_t symV2PartnerMetadataPayloadBytes = 0;
double symV2PartnerMetadataGatherTime = 0.0;
double symV2PartnerDemandPlanTime = 0.0;
double symV2PartnerDemandExchangeTime = 0.0;

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
