# SymLDL v2 nlpkkt80 Comparison

Recorded: 2026-06-20

Locked SymLDL v2 code commit:

```text
439f810f Keep SymLDL v2 solve optimizations only
```

Branch and fork:

```text
symldl-v2-solve-optimization
git@github.com:tunnellm/superlu_dist-symatch.git
```

Scope: this file records `nlpkkt80` runs only. The saved `nlpkkt120` runs are
recorded separately in `DOC/symldl-v2-nlpkkt120-comparison.md`.

## Test Setup

Perlmutter run directory:

```text
/pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/results/nlpkkt80/20260619-214747-v2-noprint-4min-54737465
```

Case:

```text
matrix: nlpkkt80
nodes: 1 GPU node
ranks: 4 MPI ranks, 4 ranks per node
threads: 8 OMP threads per rank
grid: 2x1x2
lookahead: 32
GPU3DVERSION: 2
GPU3DCONTRACT: 0
GPU3DV2_SYM_SOLVE_GPU: 1
SUPERLU_CUDA_AWARE_MPI: 0
SUPERLU_RELAX: 64
SUPERLU_MAXSUP: 256
```

Clean V0 comparison log:

```text
/tmp/superlu-nlpkkt80-fixed-guard-bench/nlpkkt80_gpuv0_grid2x1x2_l32.log
```

## Top-Level Timing

| Phase | V0 clean baseline | V2 locked commit |
|---|---:|---:|
| COLPERM | 6.452 s | 6.685 s |
| SYMBFACT | 10.423 s | 10.208 s |
| DISTRIBUTE | 5.943 s | 3.706 s |
| FACTOR | 24.886 s | 21.668 s |
| SOLVE | 0.859 s | 0.972 s |

V2 factor time was 3.218 s lower than the clean V0 baseline, or about 1.15x faster.
V2 solve time was 0.113 s higher than the clean V0 baseline.

## Correctness

| Metric | V0 clean baseline | V2 locked commit |
|---|---:|---:|
| info | 0 | 0 |
| tiny pivots | 0 | 0 |
| solution error `||X-Xtrue||/||X||` | 5.212497e-13 | 1.989520e-13 |
| sytrf 2x2 pivots | 240792 | 0 |

The V2 log reports zero sytrf 2x2 pivots for this locked run. The V0 clean baseline reports 240792.

## V2 Detail

Top-level V2 factor time includes a native LDLt factor call reported as:

```text
factor call 15.590653 / 15.561574 / 1 (LDLt native)
```

Notable V2 max-rank factor timing components:

| Component | Max time |
|---|---:|
| factor_tree_wall | 15.408097 s |
| sched_factor_dispatch | 6.783560 s |
| sched_bcast_advance | 5.718435 s |
| sched_lookahead_dispatch | 1.729520 s |
| initial_factor_dispatch | 1.266371 s |
| lpanel_transform | 1.006321 s |
| diag_d2h | 0.441569 s |
| diag_bcast | 0.416856 s |
| panel_bcast | 0.001431 s |

V2 solve timing components:

| Component | Max time |
|---|---:|
| forward_xk | 0.061409 s |
| forward_compute | 0.279924 s |
| forward_values | 0.029633 s |
| forward_apply | 0.044365 s |
| diag_comm | 0.004107 s |
| diag_compute | 0.118433 s |
| x_cache_fill | 0.110228 s |
| backward_values | 0.000000 s |
| backward_compute | 0.298560 s |
| backward_delta | 0.068922 s |

## V2 Factor Communication Update: nlpkkt80 Smoke

Recorded: 2026-06-21

Updated SymLDL v2 code commit:

```text
fe7fd76b Optimize SymFact V2 factor communication
```

Perlmutter run directory:

```text
/pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/results/v2-perf-validate/20260621-163955-nlpkkt80-gpu-2x1x2-1n-54808768
```

Local copy:

```text
/tmp/20260621-163955-nlpkkt80-gpu-2x1x2-1n-54808768
```

Case:

```text
matrix: nlpkkt80
nodes: 1 GPU node
ranks: 4 MPI ranks, 4 ranks per node
threads: 16 OMP threads per rank
grid: 2x1x2
lookahead: 32
build: build-perlmutter-v2-perf
GPU3DVERSION: 2
GPU3DCONTRACT: 0
GPU3DV2_SYM_SOLVE_GPU: 1
SUPERLU_CUDA_AWARE_MPI: 0
SUPERLU_RELAX: 64
SUPERLU_MAXSUP: 256
```

Top-level timing:

| Phase | Time |
|---|---:|
| EQUIL | 0.077 s |
| ROWPERM | 0.810 s |
| COLPERM | 6.328 s |
| SYMBFACT | 10.208 s |
| DISTRIBUTE | 3.757 s |
| FACTOR | 15.701 s |
| SOLVE | 0.980 s |

Correctness:

| Metric | Value |
|---|---:|
| exit code | 0 |
| info | 0 |
| tiny pivots | 0 |
| solution error `||X-Xtrue||/||X||` | 1.847411e-13 |
| max component relative error | 1.705303e-13 |
| sytrf 2x2 pivots | 0 |
| inertia `(pos,neg,zero)` | `(550400, 512000, 0)` |

High-level factor timing from this perf build reported:

```text
Factorization_Time : 12.96
Communication_Time : 12.96
```

This is a one-node `nlpkkt80` smoke. It should not be read as the larger
`nlpkkt120` scaling result.

This perf-build smoke does not include the detailed `SLU_ENABLE_SYM_GPU3D_TIMING`
factor counters. The timing-enabled smoke of the same commit and same matrix/grid,
but using the instrumented build and 8 OMP threads instead of the 16-thread perf
configuration above, showed the targeted factor communication sections moving in
the intended direction relative to the recent good rerun:

| Metric | Recent good rerun | fe7fd76b timing smoke | Change |
|---|---:|---:|---:|
| FACTOR time | 17.736 s | 15.284 s | 1.16x faster |
| internal Factorization_Time | 14.930 s | 12.460 s | 1.20x faster |
| SOLVE time | 0.961 s | 0.978 s | 0.98x |
| lpanel_transform max rank | 1.003 s | 0.595 s | 1.69x faster |
| lfrag_exchange_total max rank | 5.086 s | 2.515 s | 2.02x faster |
| lfrag_mpi_recv_wait max rank | 2.527 s | 1.041 s | 2.43x faster |
| lfrag_h2d_stage_issue max rank | 2.168 s | 1.015 s | 2.14x faster |

## V2 Batched Schur Update: nlpkkt80 Smoke

Recorded: 2026-06-21

Updated SymLDL v2 code commit:

```text
84805b52 Batch SymLDL V2 Schur updates
```

Case:

```text
matrix: nlpkkt80
nodes: 1 GPU node
ranks: 4 MPI ranks, 4 ranks per node
threads: 16 OMP threads per rank
grid: 2x1x2
lookahead: 32
build: build-perlmutter-v2-perf
GPU3DVERSION: 2
GPU3DCONTRACT: 0
GPU3DV2_SYM_SOLVE_GPU: 1
SUPERLU_CUDA_AWARE_MPI: 0
```

Run directories:

```text
GPU3DV2_BATCH_SCHUR=0: /pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/results/v2-perf-validate/20260621-182242-nlpkkt80-gpu-2x1x2-1n-54810432
GPU3DV2_BATCH_SCHUR=1: /pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/results/v2-perf-validate/20260621-191403-nlpkkt80-gpu-2x1x2-1n-54811636
```

Local copies:

```text
GPU3DV2_BATCH_SCHUR=0: /tmp/20260621-182242-nlpkkt80-gpu-2x1x2-1n-54810432
GPU3DV2_BATCH_SCHUR=1: /tmp/20260621-191403-nlpkkt80-gpu-2x1x2-1n-54811636
```

Top-level timing:

| Metric | Batch Schur off | Batch Schur on | Change |
|---|---:|---:|---:|
| FACTOR | 15.814 s | 10.283 s | 1.54x faster |
| Factorization_Time | 13.02 s | 7.53 s | 1.73x faster |
| SOLVE | 0.986 s | 0.989 s | unchanged |
| Grid-0 Factor:Level-0 | 0.3829 s | 0.3009 s | 1.27x faster |
| Grid-0 Factor:Level-1 | 12.3405 s | 7.0350 s | 1.75x faster |

Correctness:

| Metric | Batch Schur off | Batch Schur on |
|---|---:|---:|
| exit code | 0 | 0 |
| info | 0 | 0 |
| tiny pivots | 0 | 0 |
| solution error `||X-Xtrue||/||X||` | 1.847411e-13 | 1.989520e-13 |
| inertia `(pos,neg,zero)` | `(550400, 512000, 0)` | `(550400, 512000, 0)` |

## V2 Async Factor Pipeline: nlpkkt80 Smoke

Recorded: 2026-06-21

Updated SymLDL v2 code commit:

```text
d8a85006 Add SymLDL V2 async factor pipeline
```

Case:

```text
matrix: nlpkkt80
nodes: 1 GPU node
ranks: 4 MPI ranks, 4 ranks per node
threads: 16 OMP threads per rank
grid: 2x1x2
lookahead: 32
build: build-perlmutter-v2-perf
GPU3DVERSION: 2
GPU3DCONTRACT: 0
GPU3DV2_BATCH_SCHUR: 1
GPU3DV2_SYM_SOLVE_GPU: 1
SUPERLU_CUDA_AWARE_MPI: 0
```

Run directory:

```text
/pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/results/nlpkkt80/20260621-205432-v2-asyncAB-grid2x1x2-1n-54814871
```

Local copy:

```text
/tmp/20260621-205432-v2-asyncAB-grid2x1x2-1n-54814871
```

Top-level timing:

| Async Factor | FACTOR | Factorization_Time | SOLVE | Grid-0 Level-0 | Grid-0 Level-1 |
|---:|---:|---:|---:|---:|---:|
| 0 | 10.227 s | 7.48 s | 0.983 s | 0.2736 s | 7.0577 s |
| 1 | 10.090 s | 7.37 s | 0.979 s | 0.2947 s | 6.6668 s |

Async factor speedup:

| Metric | Result |
|---|---:|
| FACTOR speedup | 1.014x |
| Factorization_Time speedup | 1.015x |
| FACTOR reduction | 1.34% |

Correctness:

| Async Factor | Exit | Info | Tiny pivots | Solution error | Inertia `(pos,neg,zero)` |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 1.989520e-13 | `(550400, 512000, 0)` |
| 1 | 0 | 0 | 0 | 1.989520e-13 | `(550400, 512000, 0)` |

The async-factor pipeline was correct on this smoke, but the performance
movement was small compared with the earlier batched Schur update.

## V2 CTA Scatter Path: nlpkkt80 Smoke

Recorded: 2026-06-22

Updated SymLDL v2 code commits:

```text
47136448 Add SymLDL V2 CTA scatter path
e93d9a09 Fix SymLDL V2 CTA scatter lookup
```

Case:

```text
matrix: nlpkkt80
nodes: 1 GPU node
ranks: 4 MPI ranks, 4 ranks per node
threads: 16 OMP threads per rank
grid: 2x1x2
lookahead: 32
build: build-perlmutter-v2-perf
GPU3DVERSION: 2
GPU3DCONTRACT: 0
GPU3DV2_BATCH_SCHUR: 1
GPU3DV2_ASYNC_FACTOR: 0
GPU3DV2_SYM_SOLVE_GPU: 1
SUPERLU_CUDA_AWARE_MPI: 0
```

Run directories:

```text
pre-fix CTA smoke: /pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/results/nlpkkt80/20260621-220413-v2-ctaAB-grid2x1x2-1n-54817310
fixed CTA smoke: /pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/results/nlpkkt80/20260621-222236-v2-ctaAB-grid2x1x2-1n-54817612
```

Local copies:

```text
pre-fix CTA smoke: /tmp/20260621-220413-v2-ctaAB-grid2x1x2-1n-54817310
fixed CTA smoke: /tmp/20260621-222236-v2-ctaAB-grid2x1x2-1n-54817612
```

The initial CTA implementation was invalid: `GPU3DV2_CTA_SCATTER=1` timed out
after 3D initialization because the CTA metadata path called the barrier-using
device `find()` routine from only thread 0. Commit `e93d9a09` fixed this by
having the whole CTA enter `find()` before thread 0 consumes the result.

Fixed top-level timing:

| CTA Scatter | FACTOR | Factorization_Time | SOLVE |
|---:|---:|---:|---:|
| 0 | 10.206 s | 7.44 s | 0.977 s |
| 1 | 10.323 s | 7.58 s | 0.970 s |

CTA scatter speed:

| Metric | Result |
|---|---:|
| FACTOR speedup | 0.989x |
| Factorization_Time speedup | 0.982x |
| FACTOR change | 1.15% slower |

Correctness:

| CTA Scatter | Exit | Info | Tiny pivots | Solution error | Inertia `(pos,neg,zero)` |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 1.989520e-13 | `(550400, 512000, 0)` |
| 1 | 0 | 0 | 0 | 1.989520e-13 | `(550400, 512000, 0)` |

The fixed CTA scatter path was correct on this smoke, but it was slightly slower
than the existing scatter kernel.

## V2 Lower-Envelope Schur Pruning: nlpkkt80 Smoke

Recorded: 2026-06-22

Updated SymLDL v2 code commit:

```text
1a5ea2b3 Add SymLDL V2 lower-envelope Schur pruning
```

Case:

```text
matrix: nlpkkt80
nodes: 1 GPU node
ranks: 4 MPI ranks, 4 ranks per node
threads: 16 OMP threads per rank
grid: 2x1x2
lookahead: 32
build: build-perlmutter-v2-perf
GPU3DVERSION: 2
GPU3DCONTRACT: 0
GPU3DV2_BATCH_SCHUR: 1
GPU3DV2_ASYNC_FACTOR: 0
GPU3DV2_CTA_SCATTER: 0
GPU3DV2_SYM_SOLVE_GPU: 1
SUPERLU_CUDA_AWARE_MPI: 0
```

Run directory:

```text
/pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/results/nlpkkt80/20260622-090715-v2-lowerAB-grid2x1x2-1n-54831681
```

Local copy:

```text
/tmp/superlu-stage5-lower-envelope/20260622-090715-v2-lowerAB-grid2x1x2-1n-54831681
```

Top-level timing:

| Lower Envelope | FACTOR | Factorization_Time | SOLVE |
|---:|---:|---:|---:|
| 0 | 10.336 s | 7.51 s | 0.987 s |
| 1 | 10.067 s | 7.35 s | 0.978 s |

Lower-envelope speed:

| Metric | Result |
|---|---:|
| FACTOR speedup | 1.027x |
| Factorization_Time speedup | 1.022x |
| FACTOR reduction | 2.60% |

Correctness:

| Lower Envelope | Exit | Info | Tiny pivots | sytrf 2x2 pivots | Solution error | Inertia `(pos,neg,zero)` |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 | 1.989520e-13 | `(550400, 512000, 0)` |
| 1 | 0 | 0 | 0 | 0 | 1.989520e-13 | `(550400, 512000, 0)` |

The lower-envelope path was correct on this smoke and gave a modest factor-time
improvement with the neutral A/B settings from the CTA test disabled.

## Notes

Do not use `/tmp/superlu-perlmutter-results/nlpkkt80/v0_2x1x2_1n4r_8t.log` as the correctness baseline for this comparison. That run reported `FACTOR time 27.151 s` and `SOLVE time 0.660 s`, but also had solution error `3.648858e-01` and zero sytrf 2x2 pivots, so it is not comparable to the clean V0 baseline above.
