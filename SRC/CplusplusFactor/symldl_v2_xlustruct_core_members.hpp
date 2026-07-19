// This file is included inside xLUstruct_t.
// Keep declarations here limited to non-GPU SymLDL V2 state.

int symGPU3DVersion = 0;
SymLDLV2FactorBackend symV2FactorBackendKind = SYM_LDL_V2_BACKEND_NONE;

std::vector<Ftype *> symV2DiagBlocks;

SymV2RouteProfile symV2RouteProfile;
SymV2CommunicationProfile symV2CommunicationProfile;
