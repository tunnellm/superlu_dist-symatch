// This file is included inside xLUstruct_t.
// Keep declarations here limited to SymLDL V2 methods.

int_t symV2PanelRoot(int_t k);
int_t symV2DiagRoot(int_t k);
int_t symV2DiagProc(int_t k);
int_t symV2PanelIndex(int_t k);
int_t symV2RowIndex(int_t k);
int_t symV2PanelCount();
int_t symV2RowCount();
int_t symV2PanelGid(int_t local_index);
int_t symV2RowGid(int_t local_index);
bool useSymV2Solve() const;
bool needsUPanelStorage() const;
bool symV2UsesCpuFactor() const;
bool symV2UsesGpuFactor() const;
const char *symV2FactorBackendName() const;
bool symV2IsCollapsedGrid() const;
bool symV2IsPc1Fastpath() const;
bool symV2IsPr1Fastpath() const;
bool symV2ScheduleActive() const;
int_t symV2ForestLevelCount() const;
void symV2RouteProfileReset();
void symV2RouteProfileNote(SymV2RouteProfileCounter counter);
void symV2RouteProfileNotePanelBcast(bool pc_fragment);
void symV2RouteProfileNoteLFragmentExchange();
void symV2RouteProfileNotePcFragmentExchange();
void symV2RouteProfileNoteDualFragmentLookahead();
void symV2RouteProfileNoteDualFragmentExclude();
void symV2RouteProfilePrint(const char *phase) const;
void symV2FreeDiagBlocks();
void symV2FreeCpuStorage();
void symV2FreeStreamHostBuffers(int stream);
int_t pdgstrf3dSymV2();
bool symV2UsePcFragmentSchurPanel(int_t k) const;

#ifdef HAVE_CUDA
void symV2FreeGpuStorage();

int_t dSymV2PanelBcastGPU(int_t k, int_t offset);
int_t dSymV2PrepackLFragmentsGPU(int_t k, int_t stream_offset);
int_t dSymV2LFragmentExchangeGPU(int_t k, int_t stream_offset);
int_t dSymV2LookAheadUpdateGPU(int streamId, int_t k, int_t laIdx,
                               xlpanel_t<Ftype> &lpanel);
int_t dSymV2SchurCompUpdateExcludeOneGPU(int streamId, int_t k, int_t ex,
                                         xlpanel_t<Ftype> &lpanel);
#endif
