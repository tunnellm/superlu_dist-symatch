// This file is included inside xLUstruct_t under HAVE_CUDA.
// Keep declarations here limited to SymLDL V2 Pc-fragment exchange scratch state.

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
