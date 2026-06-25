# SymLDL V2 Pc>1 External Handoff

This handoff is for an external implementation review of the SymLDL V2 Pc>1
row-fragment / dual-fragment Schur path. Assume a fresh clone from GitHub.

```bash
git clone git@github.com:tunnellm/superlu_dist-symatch.git
cd superlu_dist-symatch
git checkout symldl-v2-dual-fragment-schur
```

Current pushed head:

```text
5d856f41 Add post-solve row-L transport option
```

The goal is to get implementation guidance for an optimized Pc>1 LDL-native
path without regressing the current Pc=1 fast path.

## Problem Statement

The Pc=1 SymLDL V2 fast path has been productive because it avoids unnecessary
U/LU facade movement. Pc>1 is harder because Schur updates need data that is
logically split across process rows and columns. The current Pc>1 candidate
tries to avoid full row L-panel broadcast by exchanging row and column L
fragments, then doing the update from two fragments.

The current implementation is correct on the `nlpkkt120` debug cases, but the
latest row-L transport variants have not improved performance. We need a better
implementation strategy for Pc>1 that avoids facade materialization without
spending the savings on metadata construction, packing, staging, and extra MPI
traffic.

## Primary Questions For Review

1. Is the current Pc>1 data movement structure fundamentally sound, or should it
   be reorganized around a different ownership/demand model?
2. How should destination-oriented row-L planning be represented so it is sparse
   and cheap to build?
3. Can the Pc=1 "send down after solve" idea be generalized to Pc>1 by sending
   down process rows and across process columns without materializing U-facing
   facade data?
4. Which row-L path should be kept: scratch-split, destination-packed, direct
   receive, post-solve aggregate, or a new hybrid?
5. What should the hybrid selector use as its cost model? The current simple
   value-count selector made poor choices.
6. What changes are required to restore async factorization and CUDA-aware MPI
   later without blocking the main Pc>1 optimization?

## Important Runtime Flags

The baseline optimized SymLDL V2 options used in recent tests:

```text
GPU3DVERSION=2
GPU3DCONTRACT=0
GPU3DV2_BATCH_SCHUR=1
GPU3DV2_LOWER_ENVELOPE=1
GPU3DV2_ASYNC_FACTOR=0
GPU3DV2_CTA_SCATTER=0
GPU3DV2_BATCH_ANCESTOR_REDUCE=0
GPU3DV2_PINNED_STAGING=1
GPU3DV2_PINNED_STAGING_POOL=1
GPU3DV2_PANEL_ARENA=1
GPU3DV2_WORKSPACE_ARENA=1
GPU3DV2_WPANEL_CACHE=1
GPU3DV2_OWNER_AFFINITY=0.05
GPU3DV2_FACTOR_TIMING=1
GPU3DV2_SYM_SOLVE_GPU=1
GPU3DV2_SYM_SOLVE_TIMING=1
GPU3DV2_PROFILE=1
GPU3DV2_FACTOR_PROFILE=0
SUPERLU_CUDA_AWARE_MPI=0
SUPERLU_LBS=GD
SUPERLU_ACC_OFFLOAD=1
SUPERLU_BIND_MPI_GPU=1
SUPERLU_MAXSUP=256
SUPERLU_RELAX=64
SUPERLU_MAX_BUFFER_SIZE=256000000
SUPERLU_N_GEMM=6000
SUPERLU_MPI_PROCESS_PER_GPU=1
MPI_PROCESS_PER_GPU=1
```

Pc>1 candidate switch:

```text
GPU3DV2_PC_FRAGMENT_SCHUR=1
```

Experimental row-fragment switches:

```text
GPU3DV2_ROWFRAG_DEST_PACK=1
GPU3DV2_ROW_L_SOURCE_PACK=1
GPU3DV2_ROW_L_DIRECT_RECV=1
GPU3DV2_ROW_L_POSTSOLVE_SEND=1
GPU3DV2_HYBRID_ROW_BCAST=1
GPU3DV2_EXACT_FRAGMENT_DEMAND=1
GPU3DV2_EXACT_ROW_FRAGMENT_DEMAND=1
GPU3DV2_EXACT_PARTNER_FRAGMENT_DEMAND=1
```

Most of those default to off. `GPU3DV2_ROW_L_POSTSOLVE_SEND=1` was the latest
attempt; it intentionally uses the aggregated destination row-fragment path and
suppresses the exact/direct/hybrid machinery.

## Code Map

Use these relative paths in a fresh clone.

- `SRC/CplusplusFactor/gpu_mpi_utils.hpp`
  - Environment flag helpers.
  - Key functions: `superlu_sym_v2_pc_fragment_schur()`,
    `superlu_sym_v2_hybrid_row_bcast()`,
    `superlu_sym_v2_row_l_source_pack()`,
    `superlu_sym_v2_row_l_direct_recv()`,
    `superlu_sym_v2_row_l_postsolve_send()`,
    `superlu_sym_v2_rowfrag_destination_path()`,
    `superlu_sym_v2_exact_fragment_demand()`,
    `superlu_sym_v2_exact_row_fragment_demand()`.

- `SRC/CplusplusFactor/dsparseTreeFactorGPU_impl.hpp`
  - GPU factor scheduler and panel broadcast path.
  - Key places to inspect: calls to `symV2UsePcFragmentSchurPanel(k)`,
    `dSymV2LFragmentExchangeGPU(k, offset)`, full L-panel row-broadcast skip,
    and dual-fragment Schur dispatch.

- `SRC/CplusplusFactor/lupanels_impl.hpp`
  - Host setup, workspace sizing, metadata construction, exact demand maps,
    direct row-L receive maps, and current hybrid selector.
  - Key functions/fields: `initSymFactWorkspace()`,
    `symV2UsePcFragmentSchur`, `symV2UsePcFragmentSchurPanel()`.

- `SRC/CplusplusFactor/lupanels_GPU_impl.hpp`
  - GPU row/column fragment prepack, exchange, receive staging, assembly, and
    row-fragment payload profile accounting.
  - Key functions: `dSymV2PrepackLFragmentsGPU()`,
    `dSymV2LFragmentExchangeGPU()`.

- `SRC/CplusplusFactor/schurCompUpdate_impl.cuh`
  - Dual-fragment Schur update path.
  - Key function: `dSymSchurCompUpdatePartDualFragmentsGPU()`.

- `SRC/CplusplusFactor/xlupanels.hpp`
  - Declarations and SymLDL V2 metadata/storage members.
  - Key declarations: `symV2UsePcFragmentSchur`,
    `dSymV2PrepackLFragmentsGPU()`, `dSymV2LFragmentExchangeGPU()`,
    `dSymSchurCompUpdatePartDualFragmentsGPU()`.

- `DOC/symldl-v2-nlpkkt120-comparison.md`
  - Benchmark history for `nlpkkt120`, including earlier Pc>1 runs and prior
    optimization stages.

- `DOC/symldl-v2-nlpkkt80-comparison.md`
  - Smaller `nlpkkt80` validation history.

## Current Known-Good Pc>1 Baseline

The most useful current baseline is the scratch-split Pc-fragment candidate:

```text
GPU3DV2_PC_FRAGMENT_SCHUR=1
```

Recorded in `DOC/symldl-v2-nlpkkt120-comparison.md`, it improved over
Pc-fragment disabled on `nlpkkt120`:

| Grid | Nodes | Candidate | FACTOR | Factorization_Time | SOLVE |
|---|---:|---:|---:|---:|---:|
| `2x2x2` | 2 | off | `33.742s` | `30.02s` | `1.974s` |
| `2x2x2` | 2 | on | `28.830s` | `24.46s` | `1.971s` |
| `4x2x1` | 2 | off | `55.242s` | `51.18s` | `2.693s` |
| `4x2x1` | 2 | on | `49.826s` | `45.08s` | `2.716s` |
| `2x1x4` | 2 | Pc=1 guard | `18.877s` | `14.64s` | `1.927s` |

Correctness was clean: `exit_code=0`, `info=0`, tiny pivots `0`, sytrf 2x2
pivots `0`, solution error `2.273737e-13`, inertia
`(1814400, 1728000, 0)`.

## Attempts That Did Not Help

Destination-packed row fragments:

```text
GPU3DV2_PC_FRAGMENT_SCHUR=1
GPU3DV2_ROWFRAG_DEST_PACK=1
```

This reduced message granularity but was slower than the scratch-split
candidate:

| Grid | FACTOR Delta vs Scratch-Split | Factorization_Time Delta |
|---|---:|---:|
| `2x2x2` | `+0.921s` | `+0.90s` |
| `4x2x1` | `+0.294s` | `+0.31s` |
| `2x1x4` | `-0.043s` | `0.00s` |

Direct row-L receive and exact row-demand variants were correctness-clean in
debug runs, but setup costs were too high. The main regressions came from
expensive exact/direct map construction and receive-map setup.

The hybrid selector also regressed because it chose the slower full-row-bcast
or non-dual path based on a simple value-count heuristic. That cost model was
too weak.

## Latest Post-Solve Row-L Attempt

Latest commit:

```text
5d856f41 Add post-solve row-L transport option
```

Files changed:

```text
SRC/CplusplusFactor/gpu_mpi_utils.hpp
SRC/CplusplusFactor/lupanels_GPU_impl.hpp
SRC/CplusplusFactor/lupanels_impl.hpp
```

New flag:

```text
GPU3DV2_ROW_L_POSTSOLVE_SEND=1
```

Intent:

- Use the existing aggregated destination row-fragment transport.
- Pack row-L after panel solve.
- Avoid exact row-demand maps.
- Avoid direct receive maps.
- Disable the current hybrid selector while this flag is active.

Perlmutter debug run on `nlpkkt120`, 2 nodes, 4 ranks per node, 8 threads,
rank order `Z`, MC80, lookahead 32:

| Grid | Baseline FACTOR | Postsolve FACTOR | Delta | Baseline FactTime | Postsolve FactTime |
|---|---:|---:|---:|---:|---:|
| `2x2x2` | `29.014s` | `29.309s` | `+1.0%` slower | `24.59s` | `24.88s` |
| `4x2x1` | `49.174s` | `50.024s` | `+1.7%` slower | `44.41s` | `45.23s` |
| `2x1x4` | `18.874s` | `18.817s` | noise | `14.62s` | `14.57s` |

Correctness was clean for all six cases: `exit_code=0`, `info=0`, tiny pivots
`0`, sytrf 2x2 pivots `0`, inertia `(1814400, 1728000, 0)`, solution error
`2.273737e-13`.

Interpretation: this option avoids the large direct/hybrid regression, but it
does not improve performance on `nlpkkt120`. Keep it opt-in.

## Performance Hypothesis

The current Pc>1 code is not losing because the math is wrong. It is losing
because row-fragment metadata and movement are too heavy:

- Demand metadata is still coarse in several paths.
- Destination packing adds extra packing/assembly without enough payload
  reduction.
- Exact/direct row maps can cost multiple seconds on `nlpkkt120`.
- The hybrid selector lacks a real cost model and can select slower modes.
- Current Pc>1 path still has too much facade-like movement compared with Pc=1.

The outside review should focus on the representation and schedule of row-L
movement before adding more kernels.

## Suggested Review Strategy

1. Start from `GPU3DV2_PC_FRAGMENT_SCHUR=1` with all row-experiment flags off.
2. Read `dSymV2LFragmentExchangeGPU()` in
   `SRC/CplusplusFactor/lupanels_GPU_impl.hpp` together with metadata setup in
   `initSymFactWorkspace()` in `SRC/CplusplusFactor/lupanels_impl.hpp`.
3. Identify what row data each destination actually needs for
   `dSymSchurCompUpdatePartDualFragmentsGPU()` in
   `SRC/CplusplusFactor/schurCompUpdate_impl.cuh`.
4. Propose a sparse row-local handshake that avoids global/per-panel dense
   maps and does not require full row L-panel broadcast.
5. Propose a hybrid selector based on measured/estimated communication plus
   pack/assemble cost, not just value count.
6. Keep Pc=1 behavior unchanged unless there is a clear, isolated benefit.

