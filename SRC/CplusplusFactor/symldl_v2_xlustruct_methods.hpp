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
bool symV2ScheduleActive() const;
int_t symV2ForestLevelCount() const;
void symV2FreeDiagBlocks();
void symV2FreeStreamHostBuffers(int stream);

#ifdef HAVE_CUDA
void symV2FreeGpuStorage();

int_t dSymV2PanelBcastGPU(int_t k, int_t offset);
int_t pdgstrf3dSymV2();
int_t dSymV2PrepackLFragmentsGPU(int_t k, int_t stream_offset);
int_t dSymV2LFragmentExchangeGPU(int_t k, int_t stream_offset);
int_t dSymV2LookAheadUpdateGPU(int streamId, int_t k, int_t laIdx,
                               xlpanel_t<Ftype> &lpanel);
int_t dSymV2SchurCompUpdateExcludeOneGPU(int streamId, int_t k, int_t ex,
                                         xlpanel_t<Ftype> &lpanel);
bool symV2UsePcFragmentSchurPanel(int_t k) const;
#endif
