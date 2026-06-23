# SymLDL v2 nlpkkt120 Comparison

Recorded: 2026-06-21

Scope: this file records saved `nlpkkt120` runs only. The one-node `nlpkkt80`
smokes are recorded separately in `DOC/symldl-v2-nlpkkt80-comparison.md`.

Branch and fork:

```text
symldl-v2-solve-optimization
git@github.com:tunnellm/superlu_dist-symatch.git
```

## Completed Perf Runs

These runs used the Perlmutter perf build:

```text
/pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/build-perlmutter-v2-perf/EXAMPLE/pddrive3d-sym
```

Common setup:

```text
matrix: nlpkkt120
matrix file: /pscratch/sd/m/mtunnell/matrices_large/nlpkkt120/nlpkkt120.i32.bin
nodes: 4 GPU nodes
ranks: 16 MPI ranks, 4 ranks per node
threads: 16 OMP threads per rank
grid: 4x2x2
lookahead: 32
GPU3DCONTRACT: 0
SUPERLU_CUDA_AWARE_MPI: 0
SUPERLU_RELAX: 64
SUPERLU_MAXSUP: 256
```

Run directories:

```text
V2 GPU: /pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/results/v2-perf-validate/20260620-234123-nlpkkt120-gpu-4x2x2-4n-54778971
V0 GPU: /pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/results/v0-gpu-perf-check/20260621-114804-nlpkkt120-v0-gpu-4x2x2-4n-16t-54798332
```

Local copies:

```text
V2 GPU: /tmp/superlu-v2-perf-validate/20260620-234123-nlpkkt120-gpu-4x2x2-4n-54778971
V0 GPU: /tmp/superlu-v0-gpu-perf-check/20260621-114804-nlpkkt120-v0-gpu-4x2x2-4n-16t-54798332
```

Commits reported by run metadata:

| Run | Commit | Origin commit |
|---|---|---|
| V2 GPU | `d12016bb` | `559f0551` |
| V0 GPU | `d12016bb` | `559f0551` |

Top-level timing:

| Phase | V0 GPU | V2 GPU |
|---|---:|---:|
| EQUIL | 0.170 s | 0.171 s |
| ROWPERM | 2.698 s | 2.700 s |
| COLPERM | 22.675 s | 22.922 s |
| SYMBFACT | 55.801 s | 56.426 s |
| DISTRIBUTE | 9.064 s | 5.400 s |
| FACTOR | 62.763 s | 50.698 s |
| SOLVE | 1.380 s | 1.843 s |

V2 factor time was 12.065 s lower than the V0 GPU run, or about 1.24x faster.
V2 solve time was 0.463 s higher than the V0 GPU run.

High-level factor timing:

| Metric | V0 GPU | V2 GPU |
|---|---:|---:|
| Factorization_Time | 57.72 s | 46.39 s |

Correctness:

| Metric | V0 GPU | V2 GPU |
|---|---:|---:|
| exit code | 0 | 0 |
| info | 0 | 0 |
| tiny pivots | 0 | 0 |
| solution error `||X-Xtrue||/||X||` | 2.000622e-13 | 2.131628e-13 |
| sytrf 2x2 pivots | 0 | 0 |
| inertia `(pos,neg,zero)` | `(926292, 882344, 0)` | `(1814400, 1728000, 0)` |

The V0 and V2 inertia lines are recorded as printed. The sums differ, so these
inertia values should not be used as a like-for-like algorithmic comparison
without first auditing the reporting path for each factorization mode.

## Earlier V2 Debug Runs

These were debug-queue runs from commit `fc3ef764`, using the timing build:

```text
/pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/build-perlmutter-v2/EXAMPLE/pddrive3d-sym
```

| Grid | Nodes | Exit | Factor | Solve | Factorization_Time | Solution error |
|---|---:|---:|---:|---:|---:|---:|
| 2x2x2 | 2 | 0 | 71.920 s | 1.993 s | 55.71 s | 2.131628e-13 |
| 4x2x2 | 4 | 0 | 58.536 s | 1.856 s | 50.35 s | 2.131628e-13 |
| 2x2x4 | 4 | 143 | incomplete | incomplete | 31.83 s before failure | not reported |

Run directories:

```text
2x2x2, 2 nodes: /pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/results/nlpkkt120/20260620-101612-v2-debug10-grid2x2x2-2n-54759847
4x2x2, 4 nodes: /pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/results/nlpkkt120/20260620-101614-v2-debug10-grid4x2x2-4n-54759848
2x2x4, 4 nodes: /pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/results/nlpkkt120/20260620-102715-v2-debug10-grid2x2x4-4n-54759849
```

## V2 Batched Schur Update

Recorded: 2026-06-21

Updated SymLDL v2 code commit:

```text
84805b52 Batch SymLDL V2 Schur updates
```

These runs used the Perlmutter perf build, with each A/B pair run inside the
same debug allocation:

```text
/pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/build-perlmutter-v2-perf/EXAMPLE/pddrive3d-sym
```

Common setup:

```text
matrix: nlpkkt120
matrix file: /pscratch/sd/m/mtunnell/matrices_large/nlpkkt120/nlpkkt120.i32.bin
ranks: 4 MPI ranks per node
threads: 16 OMP threads per rank
lookahead: 32
GPU3DVERSION: 2
GPU3DCONTRACT: 0
GPU3DV2_SYM_SOLVE_GPU: 1
SUPERLU_CUDA_AWARE_MPI: 0
SUPERLU_RELAX: 64
SUPERLU_MAXSUP: 256
```

Run directories:

```text
2 nodes, 2x2x2: /pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/results/nlpkkt120/20260621-191405-v2-batchAB-grid2x2x2-2n-54811758
4 nodes, 2x2x4: /pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/results/nlpkkt120/20260621-191646-v2-batchAB-grid2x2x4-4n-54811760
```

Local copies:

```text
2 nodes, 2x2x2: /tmp/20260621-191405-v2-batchAB-grid2x2x2-2n-54811758
4 nodes, 2x2x4: /tmp/20260621-191646-v2-batchAB-grid2x2x4-4n-54811760
```

Top-level timing:

| Grid | Nodes | Batch Schur | FACTOR | Factorization_Time | SOLVE |
|---|---:|---:|---:|---:|---:|
| 2x2x2 | 2 | 0 | 52.127 s | 45.77 s | 1.971 s |
| 2x2x2 | 2 | 1 | 38.863 s | 32.56 s | 1.953 s |
| 2x2x4 | 4 | 0 | 30.053 s | 26.24 s | 1.309 s |
| 2x2x4 | 4 | 1 | 22.910 s | 19.12 s | 1.311 s |

Batched Schur speedup:

| Grid | Nodes | FACTOR speedup | Factorization_Time speedup | FACTOR reduction |
|---|---:|---:|---:|---:|
| 2x2x2 | 2 | 1.34x | 1.41x | 25.45% |
| 2x2x4 | 4 | 1.31x | 1.37x | 23.77% |

Factor-tree timing:

| Grid | Nodes | Batch Schur | Grid-0 Level-0 | Grid-0 Level-1 | Grid-0 Level-2 |
|---|---:|---:|---:|---:|---:|
| 2x2x2 | 2 | 0 | 2.0572 s | 43.5668 s | - |
| 2x2x2 | 2 | 1 | 1.8319 s | 30.5885 s | - |
| 2x2x4 | 4 | 0 | 2.0553 s | 2.7222 s | 20.9713 s |
| 2x2x4 | 4 | 1 | 1.8509 s | 2.0330 s | 14.7576 s |

Correctness:

| Grid | Nodes | Batch Schur | Exit | Info | Tiny pivots | Solution error | Inertia `(pos,neg,zero)` |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2x2x2 | 2 | 0 | 0 | 0 | 0 | 2.273737e-13 | `(1814400, 1728000, 0)` |
| 2x2x2 | 2 | 1 | 0 | 0 | 0 | 2.131628e-13 | `(1814400, 1728000, 0)` |
| 2x2x4 | 4 | 0 | 0 | 0 | 0 | 2.415845e-13 | `(1814400, 1728000, 0)` |
| 2x2x4 | 4 | 1 | 0 | 0 | 0 | 2.131628e-13 | `(1814400, 1728000, 0)` |

## V2 Async Factor Pipeline

Recorded: 2026-06-21

Updated SymLDL v2 code commit:

```text
d8a85006 Add SymLDL V2 async factor pipeline
```

These runs used the Perlmutter perf build, with each A/B pair run inside the
same debug allocation:

```text
/pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/build-perlmutter-v2-perf/EXAMPLE/pddrive3d-sym
```

Common setup:

```text
matrix: nlpkkt120
matrix file: /pscratch/sd/m/mtunnell/matrices_large/nlpkkt120/nlpkkt120.i32.bin
ranks: 4 MPI ranks per node
threads: 16 OMP threads per rank
lookahead: 32
GPU3DVERSION: 2
GPU3DCONTRACT: 0
GPU3DV2_BATCH_SCHUR: 1
GPU3DV2_SYM_SOLVE_GPU: 1
SUPERLU_CUDA_AWARE_MPI: 0
SUPERLU_RELAX: 64
SUPERLU_MAXSUP: 256
```

Run directories:

```text
2 nodes, 2x2x2: /pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/results/nlpkkt120/20260621-204938-v2-asyncAB-grid2x2x2-2n-54814690
4 nodes, 2x2x4: /pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/results/nlpkkt120/20260621-204938-v2-asyncAB-grid2x2x4-4n-54814691
```

Local copies:

```text
2 nodes, 2x2x2: /tmp/20260621-204938-v2-asyncAB-grid2x2x2-2n-54814690
4 nodes, 2x2x4: /tmp/20260621-204938-v2-asyncAB-grid2x2x4-4n-54814691
```

Top-level timing:

| Grid | Nodes | Async Factor | FACTOR | Factorization_Time | SOLVE |
|---|---:|---:|---:|---:|---:|
| 2x2x2 | 2 | 0 | 38.922 s | 32.58 s | 1.948 s |
| 2x2x2 | 2 | 1 | 38.696 s | 32.34 s | 1.936 s |
| 2x2x4 | 4 | 0 | 22.943 s | 19.21 s | 1.287 s |
| 2x2x4 | 4 | 1 | 22.795 s | 18.99 s | 1.288 s |

Async factor speedup:

| Grid | Nodes | FACTOR speedup | Factorization_Time speedup | FACTOR reduction |
|---|---:|---:|---:|---:|
| 2x2x2 | 2 | 1.006x | 1.007x | 0.58% |
| 2x2x4 | 4 | 1.006x | 1.012x | 0.65% |

Factor-tree timing:

| Grid | Nodes | Async Factor | Grid-0 Level-0 | Grid-0 Level-1 | Grid-0 Level-2 |
|---|---:|---:|---:|---:|---:|
| 2x2x2 | 2 | 0 | 1.8470 s | 30.5691 s | - |
| 2x2x2 | 2 | 1 | 1.8214 s | 30.3973 s | - |
| 2x2x4 | 4 | 0 | 1.8192 s | 2.0472 s | 14.8550 s |
| 2x2x4 | 4 | 1 | 1.8455 s | 2.0411 s | 14.6055 s |

Correctness:

| Grid | Nodes | Async Factor | Exit | Info | Tiny pivots | Solution error | Inertia `(pos,neg,zero)` |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2x2x2 | 2 | 0 | 0 | 0 | 0 | 2.131628e-13 | `(1814400, 1728000, 0)` |
| 2x2x2 | 2 | 1 | 0 | 0 | 0 | 2.131628e-13 | `(1814400, 1728000, 0)` |
| 2x2x4 | 4 | 0 | 0 | 0 | 0 | 2.131628e-13 | `(1814400, 1728000, 0)` |
| 2x2x4 | 4 | 1 | 0 | 0 | 0 | 2.131628e-13 | `(1814400, 1728000, 0)` |

The async-factor pipeline was correct on these smokes, but the performance
movement was small compared with the earlier batched Schur update.

## V2 Lower-Envelope Schur Pruning

Recorded: 2026-06-22

Updated SymLDL v2 code commit:

```text
1a5ea2b3 Add SymLDL V2 lower-envelope Schur pruning
```

These runs used the Perlmutter perf build, with each A/B pair run inside the
same debug allocation:

```text
/pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/build-perlmutter-v2-perf/EXAMPLE/pddrive3d-sym
```

Common setup:

```text
matrix: nlpkkt120
matrix file: /pscratch/sd/m/mtunnell/matrices_large/nlpkkt120/nlpkkt120.i32.bin
ranks: 4 MPI ranks per node
threads: 16 OMP threads per rank
lookahead: 32
GPU3DVERSION: 2
GPU3DCONTRACT: 0
GPU3DV2_BATCH_SCHUR: 1
GPU3DV2_ASYNC_FACTOR: 0
GPU3DV2_CTA_SCATTER: 0
GPU3DV2_SYM_SOLVE_GPU: 1
SUPERLU_CUDA_AWARE_MPI: 0
SUPERLU_RELAX: 64
SUPERLU_MAXSUP: 256
```

Run directories:

```text
2 nodes, 2x2x2: /pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/results/nlpkkt120/20260622-090955-v2-lowerAB-grid2x2x2-2n-54831683
4 nodes, 2x2x4: /pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/results/nlpkkt120/20260622-090955-v2-lowerAB-grid2x2x4-4n-54831684
```

Local copies:

```text
2 nodes, 2x2x2: /tmp/superlu-stage5-lower-envelope/20260622-090955-v2-lowerAB-grid2x2x2-2n-54831683
4 nodes, 2x2x4: /tmp/superlu-stage5-lower-envelope/20260622-090955-v2-lowerAB-grid2x2x4-4n-54831684
```

Top-level timing:

| Grid | Nodes | Lower Envelope | FACTOR | Factorization_Time | SOLVE |
|---|---:|---:|---:|---:|---:|
| 2x2x2 | 2 | 0 | 39.114 s | 32.84 s | 1.951 s |
| 2x2x2 | 2 | 1 | 38.452 s | 32.12 s | 1.985 s |
| 2x2x4 | 4 | 0 | 23.154 s | 19.38 s | 1.346 s |
| 2x2x4 | 4 | 1 | 22.520 s | 18.78 s | 1.295 s |

Lower-envelope speedup:

| Grid | Nodes | FACTOR speedup | Factorization_Time speedup | FACTOR reduction |
|---|---:|---:|---:|---:|
| 2x2x2 | 2 | 1.017x | 1.022x | 1.69% |
| 2x2x4 | 4 | 1.028x | 1.032x | 2.74% |

Correctness:

| Grid | Nodes | Lower Envelope | Exit | Info | Tiny pivots | sytrf 2x2 pivots | Solution error | Inertia `(pos,neg,zero)` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2x2x2 | 2 | 0 | 0 | 0 | 0 | 0 | 2.131628e-13 | `(1814400, 1728000, 0)` |
| 2x2x2 | 2 | 1 | 0 | 0 | 0 | 0 | 2.131628e-13 | `(1814400, 1728000, 0)` |
| 2x2x4 | 4 | 0 | 0 | 0 | 0 | 0 | 2.131628e-13 | `(1814400, 1728000, 0)` |
| 2x2x4 | 4 | 1 | 0 | 0 | 0 | 0 | 2.131628e-13 | `(1814400, 1728000, 0)` |

The lower-envelope path was correct on these smokes and consistently faster,
but the gain was modest compared with the earlier batched Schur update.

## V2 Batched Ancestor Reduction

Recorded: 2026-06-22

Updated SymLDL v2 code commit:

```text
c3e59c71 Add SymLDL V2 batched ancestor reduction
```

These runs used the Perlmutter perf build, with each A/B pair run inside the
same debug allocation:

```text
/pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/build-perlmutter-v2-perf/EXAMPLE/pddrive3d-sym
```

Common setup:

```text
matrix: nlpkkt120
matrix file: /pscratch/sd/m/mtunnell/matrices_large/nlpkkt120/nlpkkt120.i32.bin
ranks: 4 MPI ranks per node
threads: 16 OMP threads per rank
lookahead: 32
GPU3DVERSION: 2
GPU3DCONTRACT: 0
GPU3DV2_BATCH_SCHUR: 1
GPU3DV2_LOWER_ENVELOPE: 1
GPU3DV2_ASYNC_FACTOR: 0
GPU3DV2_CTA_SCATTER: 0
GPU3DV2_SYM_SOLVE_GPU: 1
SUPERLU_CUDA_AWARE_MPI: 0
SUPERLU_RELAX: 64
SUPERLU_MAXSUP: 256
```

Run directories:

```text
2 nodes, 2x2x2: /pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/results/nlpkkt120/20260622-111520-v2-ancAB-grid2x2x2-2n-54837381
4 nodes, 2x2x4: /pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/results/nlpkkt120/20260622-111520-v2-ancAB-grid2x2x4-4n-54837383
```

Local copies:

```text
2 nodes, 2x2x2: /tmp/superlu-stage6-ancestor-reduce/20260622-111520-v2-ancAB-grid2x2x2-2n-54837381
4 nodes, 2x2x4: /tmp/superlu-stage6-ancestor-reduce/20260622-111520-v2-ancAB-grid2x2x4-4n-54837383
```

Top-level timing:

| Grid | Nodes | Batched Ancestor Reduce | FACTOR | Factorization_Time | SOLVE |
|---|---:|---:|---:|---:|---:|
| 2x2x2 | 2 | 0 | 38.616 s | 32.27 s | 1.974 s |
| 2x2x2 | 2 | 1 | 39.017 s | 32.71 s | 2.003 s |
| 2x2x4 | 4 | 0 | 22.662 s | 18.92 s | 1.313 s |
| 2x2x4 | 4 | 1 | 25.136 s | 21.38 s | 1.328 s |

Batched ancestor reduction speed:

| Grid | Nodes | FACTOR speedup | Factorization_Time speedup | FACTOR change |
|---|---:|---:|---:|---:|
| 2x2x2 | 2 | 0.990x | 0.987x | 1.04% slower |
| 2x2x4 | 4 | 0.902x | 0.885x | 10.92% slower |

Factor-tree timing:

| Grid | Nodes | Batched Ancestor Reduce | Grid-0 Level-0 | Grid-0 Level-1 | Grid-0 Level-2 |
|---|---:|---:|---:|---:|---:|
| 2x2x2 | 2 | 0 | 1.8438 s | 30.2790 s | - |
| 2x2x2 | 2 | 1 | 1.8499 s | 29.8869 s | - |
| 2x2x4 | 4 | 0 | 1.8062 s | 1.9521 s | 14.6483 s |
| 2x2x4 | 4 | 1 | 1.8238 s | 1.9356 s | 14.4818 s |

Correctness:

| Grid | Nodes | Batched Ancestor Reduce | Exit | Info | Tiny pivots | sytrf 2x2 pivots | Solution error | Inertia `(pos,neg,zero)` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2x2x2 | 2 | 0 | 0 | 0 | 0 | 0 | 2.131628e-13 | `(1814400, 1728000, 0)` |
| 2x2x2 | 2 | 1 | 0 | 0 | 0 | 0 | 2.131628e-13 | `(1814400, 1728000, 0)` |
| 2x2x4 | 4 | 0 | 0 | 0 | 0 | 0 | 2.131628e-13 | `(1814400, 1728000, 0)` |
| 2x2x4 | 4 | 1 | 0 | 0 | 0 | 0 | 2.131628e-13 | `(1814400, 1728000, 0)` |

The batched ancestor reduction was correct but slower, especially in the
targeted `Pz>1` four-node case. The factor-tree level timers moved slightly in
the favorable direction, but total `Factorization_Time` and top-level `FACTOR`
regressed. That points to overhead outside the existing level timers: per-chunk
`cudaMalloc`/`cudaFree`, per-chunk pinned host staging allocation, explicit
packing, `MPI_Sendrecv` signature checks, and extra stream synchronization are
the likely costs.

This patch may still be salvageable if rewritten to use persistent device and
pinned host scratch buffers, perform layout/signature validation once during
setup, and batch without adding allocation or synchronization inside each
ancestor reduction call. The current implementation should remain disabled for
independent benchmarks.

## V2 Pinned MPI Staging

Recorded: 2026-06-22

Updated SymLDL v2 code commits:

```text
9cfaefd3 Add SymLDL V2 pinned MPI staging
4030be83 Fix SymLDL V2 pinned staging teardown
```

The first pinned-staging run completed factorization and solve correctly but
crashed during teardown because pinned receive buffers could be released through
the normal `SUPERLU_FREE` path. Commit `4030be83` fixed that ownership lifetime
bug and the A/B below is from the clean rerun.

These runs used the Perlmutter perf build, with each A/B pair run inside the
same debug allocation:

```text
/pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/build-perlmutter-v2-perf/EXAMPLE/pddrive3d-sym
```

Common setup:

```text
matrix: nlpkkt120
matrix file: /pscratch/sd/m/mtunnell/matrices_large/nlpkkt120/nlpkkt120.i32.bin
ranks: 4 MPI ranks per node
threads: 16 OMP threads per rank
lookahead: 32
GPU3DVERSION: 2
GPU3DCONTRACT: 0
GPU3DV2_BATCH_SCHUR: 1
GPU3DV2_LOWER_ENVELOPE: 1
GPU3DV2_ASYNC_FACTOR: 0
GPU3DV2_CTA_SCATTER: 0
GPU3DV2_BATCH_ANCESTOR_REDUCE: 0
GPU3DV2_SYM_SOLVE_GPU: 1
SUPERLU_CUDA_AWARE_MPI: 0
SUPERLU_RELAX: 64
SUPERLU_MAXSUP: 256
```

Run directories:

```text
2 nodes, 2x2x2: /pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/results/nlpkkt120/20260622-142106-v2-pinAB-grid2x2x2-2n-54844472
4 nodes, 2x2x4: /pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/results/nlpkkt120/20260622-142329-v2-pinAB-grid2x2x4-4n-54844473
```

Local copies:

```text
2 nodes, 2x2x2: /tmp/superlu-stage7-pinned-staging-fixed/20260622-142106-v2-pinAB-grid2x2x2-2n-54844472
4 nodes, 2x2x4: /tmp/superlu-stage7-pinned-staging-fixed/20260622-142329-v2-pinAB-grid2x2x4-4n-54844473
```

Top-level timing:

| Grid | Nodes | Pinned Staging | FACTOR | Factorization_Time | SOLVE |
|---|---:|---:|---:|---:|---:|
| 2x2x2 | 2 | 0 | 37.828 s | 31.37 s | 1.944 s |
| 2x2x2 | 2 | 1 | 40.342 s | 29.55 s | 1.933 s |
| 2x2x4 | 4 | 0 | 22.614 s | 18.82 s | 1.293 s |
| 2x2x4 | 4 | 1 | 24.663 s | 17.85 s | 1.312 s |

Pinned staging speed:

| Grid | Nodes | FACTOR speedup | Factorization_Time speedup | FACTOR change |
|---|---:|---:|---:|---:|
| 2x2x2 | 2 | 0.938x | 1.062x | 6.64% slower |
| 2x2x4 | 4 | 0.917x | 1.054x | 9.06% slower |

Factor-tree timing:

| Grid | Nodes | Pinned Staging | Grid-0 Level-0 | Grid-0 Level-1 | Grid-0 Level-2 |
|---|---:|---:|---:|---:|---:|
| 2x2x2 | 2 | 0 | 1.8272 s | 29.4074 s | - |
| 2x2x2 | 2 | 1 | 1.6447 s | 27.7658 s | - |
| 2x2x4 | 4 | 0 | 1.8080 s | 1.9305 s | 14.5920 s |
| 2x2x4 | 4 | 1 | 1.6911 s | 1.9073 s | 13.7320 s |

Correctness:

| Grid | Nodes | Pinned Staging | Exit | Info | Tiny pivots | sytrf 2x2 pivots | Solution error | Inertia `(pos,neg,zero)` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2x2x2 | 2 | 0 | 0 | 0 | 0 | 0 | 2.131628e-13 | `(1814400, 1728000, 0)` |
| 2x2x2 | 2 | 1 | 0 | 0 | 0 | 0 | 2.131628e-13 | `(1814400, 1728000, 0)` |
| 2x2x4 | 4 | 0 | 0 | 0 | 0 | 0 | 2.131628e-13 | `(1814400, 1728000, 0)` |
| 2x2x4 | 4 | 1 | 0 | 0 | 0 | 0 | 2.131628e-13 | `(1814400, 1728000, 0)` |

Pinned staging was correct after the teardown fix and improved the internal
factor-tree timing by about 5-9%, but top-level `FACTOR` time regressed by
about 7-9%. This is useful timing movement: it suggests the patch moves cost
outside the measured factor-tree region or adds setup/teardown/allocation
overhead around the factor call, making those outer costs a cleaner next target.

## V2 Serial CTA Lookup

Recorded: 2026-06-22

Updated SymLDL v2 code commit:

```text
6687ac4b Add SymLDL V2 serial CTA lookup
```

This A/B re-tested the CTA scatter path after replacing the barrier-using
cooperative destination lookup with a serial thread-0 lookup. The test kept the
other independent restructuring candidates disabled.

These runs used the Perlmutter perf build, with each A/B pair run inside the
same debug allocation:

```text
/pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/build-perlmutter-v2-perf/EXAMPLE/pddrive3d-sym
```

Common setup:

```text
matrix: nlpkkt120
matrix file: /pscratch/sd/m/mtunnell/matrices_large/nlpkkt120/nlpkkt120.i32.bin
ranks: 4 MPI ranks per node
threads: 16 OMP threads per rank
lookahead: 32
GPU3DVERSION: 2
GPU3DCONTRACT: 0
GPU3DV2_BATCH_SCHUR: 1
GPU3DV2_LOWER_ENVELOPE: 1
GPU3DV2_ASYNC_FACTOR: 0
GPU3DV2_PINNED_STAGING: 0
GPU3DV2_BATCH_ANCESTOR_REDUCE: 0
GPU3DV2_SYM_SOLVE_GPU: 1
SUPERLU_CUDA_AWARE_MPI: 0
SUPERLU_RELAX: 64
SUPERLU_MAXSUP: 256
```

Run directories:

```text
2 nodes, 2x2x2: /pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/results/nlpkkt120/20260622-150241-v2-r1ctaAB-grid2x2x2-2n-54846797
4 nodes, 2x2x4: /pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/results/nlpkkt120/20260622-150942-v2-r1ctaAB-grid2x2x4-4n-54846798
```

Local copies:

```text
2 nodes, 2x2x2: /tmp/superlu-stage8-r1-cta-serial/20260622-150241-v2-r1ctaAB-grid2x2x2-2n-54846797
4 nodes, 2x2x4: /tmp/superlu-stage8-r1-cta-serial/20260622-150942-v2-r1ctaAB-grid2x2x4-4n-54846798
```

Top-level timing:

| Grid | Nodes | CTA Scatter | FACTOR | Factorization_Time | SOLVE |
|---|---:|---:|---:|---:|---:|
| 2x2x2 | 2 | 0 | 37.521 s | 31.37 s | 1.927 s |
| 2x2x2 | 2 | 1 | 37.544 s | 31.38 s | 1.911 s |
| 2x2x4 | 4 | 0 | 22.739 s | 18.89 s | 1.298 s |
| 2x2x4 | 4 | 1 | 22.901 s | 19.06 s | 1.278 s |

Serial CTA lookup speed:

| Grid | Nodes | FACTOR speedup | Factorization_Time speedup | FACTOR change |
|---|---:|---:|---:|---:|
| 2x2x2 | 2 | 0.999x | 1.000x | 0.06% slower |
| 2x2x4 | 4 | 0.993x | 0.991x | 0.71% slower |

Factor-tree timing:

| Grid | Nodes | CTA Scatter | 3D-AncestorReduce | Grid-0 Level-0 | Grid-0 Level-1 | Grid-0 Level-2 |
|---|---:|---:|---:|---:|---:|---:|
| 2x2x2 | 2 | 0 | 0.3673 s | 1.7935 s | 29.2180 s | - |
| 2x2x2 | 2 | 1 | 0.1350 s | 1.8328 s | 29.4165 s | - |
| 2x2x4 | 4 | 0 | 0.4163 s | 1.8260 s | 1.9137 s | 14.6731 s |
| 2x2x4 | 4 | 1 | 0.4145 s | 1.8227 s | 1.9754 s | 14.7783 s |

Correctness:

| Grid | Nodes | CTA Scatter | Exit | Info | Tiny pivots | sytrf 2x2 pivots | Solution error | Inertia `(pos,neg,zero)` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2x2x2 | 2 | 0 | 0 | 0 | 0 | 0 | 2.131628e-13 | `(1814400, 1728000, 0)` |
| 2x2x2 | 2 | 1 | 0 | 0 | 0 | 0 | 2.131628e-13 | `(1814400, 1728000, 0)` |
| 2x2x4 | 4 | 0 | 0 | 0 | 0 | 0 | 2.131628e-13 | `(1814400, 1728000, 0)` |
| 2x2x4 | 4 | 1 | 0 | 0 | 0 | 0 | 2.131628e-13 | `(1814400, 1728000, 0)` |

The serial CTA lookup was correct, but not a useful multi-node performance win.
It was neutral on two nodes and slightly slower on four nodes. The two-node
case moved `3D-AncestorReduce` down, but that gain was absorbed by slower level
factor timers; the four-node case did not improve the ancestor reduction path.
Keep this path opt-in/off by default unless a later rewrite changes the CTA
scatter work enough to make the lookup cost dominant.

## V2 Owner Affinity

Recorded: 2026-06-22

Updated SymLDL v2 code commit:

```text
547f79cd Add SymLDL V2 owner affinity policy
```

This A/B tested a parent/child owner-affinity penalty in the SymLDL v2
post-symbolic 3D owner selection. The test kept the other independent
restructuring candidates disabled.

These runs used the Perlmutter perf build, with each A/B pair run inside the
same debug allocation:

```text
/pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/build-perlmutter-v2-perf/EXAMPLE/pddrive3d-sym
```

Common setup:

```text
matrix: nlpkkt120
matrix file: /pscratch/sd/m/mtunnell/matrices_large/nlpkkt120/nlpkkt120.i32.bin
ranks: 4 MPI ranks per node
threads: 16 OMP threads per rank
lookahead: 32
GPU3DVERSION: 2
GPU3DCONTRACT: 0
GPU3DV2_BATCH_SCHUR: 1
GPU3DV2_LOWER_ENVELOPE: 1
GPU3DV2_ASYNC_FACTOR: 0
GPU3DV2_CTA_SCATTER: 0
GPU3DV2_PINNED_STAGING: 0
GPU3DV2_BATCH_ANCESTOR_REDUCE: 0
GPU3DV2_SYM_SOLVE_GPU: 1
SUPERLU_CUDA_AWARE_MPI: 0
SUPERLU_RELAX: 64
SUPERLU_MAXSUP: 256
```

Run directories:

```text
2 nodes, 2x2x2: /pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/results/nlpkkt120/20260622-153205-v2-ownerAB-grid2x2x2-2n-54848022
4 nodes, 2x2x4: /pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/results/nlpkkt120/20260622-153520-v2-ownerAB-grid2x2x4-4n-54848024
```

Local copies:

```text
2 nodes, 2x2x2: /tmp/superlu-stage9-owner-affinity/20260622-153205-v2-ownerAB-grid2x2x2-2n-54848022
4 nodes, 2x2x4: /tmp/superlu-stage9-owner-affinity/20260622-153520-v2-ownerAB-grid2x2x4-4n-54848024
```

Top-level timing:

| Grid | Nodes | Owner Affinity | FACTOR | Factorization_Time | SOLVE |
|---|---:|---:|---:|---:|---:|
| 2x2x2 | 2 | 0 | 38.534 s | 32.23 s | 1.987 s |
| 2x2x2 | 2 | 0.05 | 38.654 s | 32.08 s | 1.941 s |
| 2x2x4 | 4 | 0 | 22.791 s | 19.03 s | 1.287 s |
| 2x2x4 | 4 | 0.05 | 22.120 s | 18.40 s | 1.182 s |

Owner-affinity speed:

| Grid | Nodes | FACTOR speedup | Factorization_Time speedup | FACTOR change |
|---|---:|---:|---:|---:|
| 2x2x2 | 2 | 0.997x | 1.005x | 0.31% slower |
| 2x2x4 | 4 | 1.030x | 1.034x | 2.94% faster |

Factor-tree timing:

| Grid | Nodes | Owner Affinity | 3D-AncestorReduce | Grid-0 Level-0 | Grid-0 Level-1 | Grid-0 Level-2 |
|---|---:|---:|---:|---:|---:|---:|
| 2x2x2 | 2 | 0 | 0.1373 s | 1.8077 s | 30.2800 s | - |
| 2x2x2 | 2 | 0.05 | 0.1309 s | 1.8669 s | 30.0769 s | - |
| 2x2x4 | 4 | 0 | 0.4423 s | 1.8262 s | 1.9372 s | 14.7277 s |
| 2x2x4 | 4 | 0.05 | 0.4187 s | 1.8571 s | 1.9428 s | 14.1788 s |

Memory high-water:

| Grid | Nodes | Owner Affinity | Sum-of-all | Avg | Max |
|---|---:|---:|---:|---:|---:|
| 2x2x2 | 2 | 0 | 93484.01 MB | 11685.50 MB | 12040.09 MB |
| 2x2x2 | 2 | 0.05 | 93483.81 MB | 11685.48 MB | 11808.28 MB |
| 2x2x4 | 4 | 0 | 112028.71 MB | 7001.79 MB | 7456.77 MB |
| 2x2x4 | 4 | 0.05 | 112028.78 MB | 7001.80 MB | 7160.06 MB |

Correctness:

| Grid | Nodes | Owner Affinity | Exit | Info | Tiny pivots | sytrf 2x2 pivots | Solution error | Inertia `(pos,neg,zero)` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2x2x2 | 2 | 0 | 0 | 0 | 0 | 0 | 2.131628e-13 | `(1814400, 1728000, 0)` |
| 2x2x2 | 2 | 0.05 | 0 | 0 | 0 | 0 | 2.273737e-13 | `(1814400, 1728000, 0)` |
| 2x2x4 | 4 | 0 | 0 | 0 | 0 | 0 | 2.131628e-13 | `(1814400, 1728000, 0)` |
| 2x2x4 | 4 | 0.05 | 0 | 0 | 0 | 0 | 2.273737e-13 | `(1814400, 1728000, 0)` |

Owner affinity was correct and looks promising for the four-node `Pz=4` case:
top-level factor time improved by about 3%, internal factor time by about 3.4%,
solve time by about 8%, and max memory high-water dropped by about 297 MB. The
two-node case was effectively neutral. This candidate is worth keeping opt-in
and tuning with a small weight sweep after the remaining independent patches are
tested.

## V2 W-Panel Cache

Recorded: 2026-06-22

Updated SymLDL v2 code commit:

```text
89d17d23 Add SymLDL V2 W-panel cache
```

This A/B tested the explicit transient W-panel cache. The normal current-grid
cases completed cleanly on both node counts. The additional Pr=1 diagnostic
grids failed in the baseline `GPU3DV2_WPANEL_CACHE=0` case before the candidate
case ran, so those failures are recorded as a separate partner-L metadata/layout
issue rather than a W-panel cache regression.

These runs used the Perlmutter perf build, with each A/B pair run inside the
same debug allocation:

```text
/pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/build-perlmutter-v2-perf/EXAMPLE/pddrive3d-sym
```

Common setup:

```text
matrix: nlpkkt120
matrix file: /pscratch/sd/m/mtunnell/matrices_large/nlpkkt120/nlpkkt120.i32.bin
ranks: 4 MPI ranks per node
threads: 16 OMP threads per rank
lookahead: 32
GPU3DVERSION: 2
GPU3DCONTRACT: 0
GPU3DV2_BATCH_SCHUR: 1
GPU3DV2_LOWER_ENVELOPE: 1
GPU3DV2_ASYNC_FACTOR: 0
GPU3DV2_CTA_SCATTER: 0
GPU3DV2_PINNED_STAGING: 0
GPU3DV2_BATCH_ANCESTOR_REDUCE: 0
GPU3DV2_SYM_SOLVE_GPU: 1
SUPERLU_CUDA_AWARE_MPI: 0
SUPERLU_RELAX: 64
SUPERLU_MAXSUP: 256
```

Run directories:

```text
2 nodes, current 2x2x2 plus Pr=1 diagnostic 1x4x2: /pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/results/nlpkkt120/20260622-162448-v2-wpanelBundleAB-nlpkkt120_2n_bundle-2n-54848743
4 nodes, current 2x2x4 plus Pr=1 diagnostic 1x4x4: /pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/results/nlpkkt120/20260622-162448-v2-wpanelBundleAB-nlpkkt120_4n_bundle-4n-54848744
```

Local copies:

```text
2 nodes: /tmp/superlu-stage10-wpanel-bundle/20260622-162448-v2-wpanelBundleAB-nlpkkt120_2n_bundle-2n-54848743
4 nodes: /tmp/superlu-stage10-wpanel-bundle/20260622-162448-v2-wpanelBundleAB-nlpkkt120_4n_bundle-4n-54848744
```

Top-level timing for the current grids:

| Grid | Nodes | W-panel cache | FACTOR | Factorization_Time | SOLVE |
|---|---:|---:|---:|---:|---:|
| 2x2x2 | 2 | 0 | 37.913 s | 31.40 s | 1.928 s |
| 2x2x2 | 2 | 1 | 37.542 s | 31.24 s | 1.914 s |
| 2x2x4 | 4 | 0 | 22.431 s | 18.66 s | 1.294 s |
| 2x2x4 | 4 | 1 | 22.367 s | 18.54 s | 1.260 s |

W-panel cache speed:

| Grid | Nodes | FACTOR speedup | Factorization_Time speedup | FACTOR reduction |
|---|---:|---:|---:|---:|
| 2x2x2 | 2 | 1.010x | 1.005x | 0.98% |
| 2x2x4 | 4 | 1.003x | 1.006x | 0.29% |

Factor-tree timing:

| Grid | Nodes | W-panel cache | 3D-AncestorReduce | Grid-0 Level-0 | Grid-0 Level-1 | Grid-0 Level-2 |
|---|---:|---:|---:|---:|---:|---:|
| 2x2x2 | 2 | 0 | 0.1404 s | 1.8203 s | 29.4528 s | - |
| 2x2x2 | 2 | 1 | 0.1283 s | 1.8124 s | 29.2941 s | - |
| 2x2x4 | 4 | 0 | 0.4229 s | 1.8578 s | 1.9198 s | 14.3843 s |
| 2x2x4 | 4 | 1 | 0.4371 s | 1.8307 s | 1.9423 s | 14.2548 s |

Correctness for the current grids:

| Grid | Nodes | W-panel cache | Exit | Info | Tiny pivots | sytrf 2x2 pivots | Solution error | Inertia `(pos,neg,zero)` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2x2x2 | 2 | 0 | 0 | 0 | 0 | 0 | 2.131628e-13 | `(1814400, 1728000, 0)` |
| 2x2x2 | 2 | 1 | 0 | 0 | 0 | 0 | 2.131628e-13 | `(1814400, 1728000, 0)` |
| 2x2x4 | 4 | 0 | 0 | 0 | 0 | 0 | 2.131628e-13 | `(1814400, 1728000, 0)` |
| 2x2x4 | 4 | 1 | 0 | 0 | 0 | 0 | 2.131628e-13 | `(1814400, 1728000, 0)` |

The Pr=1 diagnostic baselines failed with:

```text
SymFact V2 partner-L cached index exceeds receive buffer.
```

The failure occurred in `SRC/CplusplusFactor/lupanels_impl.hpp` at the
partner-L cached receive-index size check. Since it occurred with
`GPU3DV2_WPANEL_CACHE=0`, this is an existing Pr=1 partner-L metadata/buffer
sizing issue and not a W-panel cache correctness failure.

## V2 Pooled Pinned Staging And GPU Arenas

Recorded: 2026-06-23

Relevant SymLDL v2 commits:

```text
6867bcea Add SymLDL V2 pooled pinned staging
2825bf0a Add SymLDL V2 GPU allocation arenas
14b6af0d Align SymLDL V2 GPU panel arenas
```

The R7 run compared pageable staging, the original per-buffer pinned staging,
and pooled pinned staging. The R8 run then tested GPU panel and workspace
arenas on top of pooled pinned staging. R8 initially exposed a device pointer
alignment bug in panel arenas; `14b6af0d` aligned the arena value payload after
the integer index payload.

Common setup:

```text
matrix: nlpkkt120
matrix file: /pscratch/sd/m/mtunnell/matrices_large/nlpkkt120/nlpkkt120.i32.bin
ranks: 4 MPI ranks per node
threads: 16 OMP threads per rank
lookahead: 32
GPU3DVERSION: 2
GPU3DCONTRACT: 0
GPU3DV2_BATCH_SCHUR: 1
GPU3DV2_LOWER_ENVELOPE: 1
GPU3DV2_ASYNC_FACTOR: 0
GPU3DV2_CTA_SCATTER: 0
GPU3DV2_BATCH_ANCESTOR_REDUCE: 0
GPU3DV2_SYM_SOLVE_GPU: 1
SUPERLU_CUDA_AWARE_MPI: 0
SUPERLU_RELAX: 64
SUPERLU_MAXSUP: 256
```

Run directories:

```text
R8 fixed, 2 nodes, 2x2x2: /pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/results/nlpkkt120/20260622-223221-v2-r8ComboFix-grid2x2x2-2n-54867465
R8 fixed, 4 nodes, 2x2x4: /pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/results/nlpkkt120/20260622-224915-v2-r8ComboFix-grid2x2x4-4n-54867467
```

Local copies:

```text
R8 fixed, 4 nodes: /tmp/superlu-r8-combo-fix/20260622-224915-v2-r8ComboFix-grid2x2x4-4n-54867467
```

R7 pooled staging summary:

| Grid | Nodes | Mode | FACTOR | Factorization_Time | SOLVE | create handle |
|---|---:|---|---:|---:|---:|---:|
| 2x2x2 | 2 | pageable baseline | 38.622 s | 32.33 s | 1.947 s | 6.285 s |
| 2x2x2 | 2 | original pinned staging | 41.283 s | 30.56 s | 1.954 s | 10.714 s |
| 2x2x2 | 2 | pooled pinned staging | 34.372 s | 30.34 s | 1.920 s | 4.027 s |
| 2x2x2 | 2 | pooled pinned staging plus prior winners | 35.174 s | 31.03 s | 1.967 s | 4.135 s |
| 2x2x4 | 4 | pageable baseline | 22.698 s | 18.75 s | 1.371 s | 3.941 s |
| 2x2x4 | 4 | original pinned staging | 24.604 s | 17.69 s | 1.299 s | 6.911 s |
| 2x2x4 | 4 | pooled pinned staging | 20.123 s | 17.53 s | 1.276 s | 2.584 s |
| 2x2x4 | 4 | pooled pinned staging plus prior winners | 19.699 s | 17.26 s | 1.203 s | 2.431 s |

R8 fixed arena comparison:

| Grid | Nodes | Mode | Panel arena | Workspace arena | Prior winners | FACTOR | Factorization_Time | SOLVE |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 2x2x2 | 2 | A R7 baseline | 0 | 0 | 0 | 34.518 s | 30.48 s | 2.032 s |
| 2x2x2 | 2 | B panel arena | 1 | 0 | 0 | 33.216 s | 29.73 s | 1.937 s |
| 2x2x2 | 2 | C workspace arena | 0 | 1 | 0 | 34.090 s | 30.10 s | 1.918 s |
| 2x2x2 | 2 | D both arenas | 1 | 1 | 0 | 33.289 s | 29.82 s | 1.908 s |
| 2x2x2 | 2 | E R7 winners | 0 | 0 | 1 | 34.414 s | 30.21 s | 1.943 s |
| 2x2x2 | 2 | F both arenas plus winners | 1 | 1 | 1 | 33.489 s | 29.88 s | 1.928 s |
| 2x2x4 | 4 | A R7 baseline | 0 | 0 | 0 | 20.190 s | 17.69 s | 1.295 s |
| 2x2x4 | 4 | B panel arena | 1 | 0 | 0 | 19.499 s | 17.31 s | 1.253 s |
| 2x2x4 | 4 | C workspace arena | 0 | 1 | 0 | 19.973 s | 17.44 s | 1.347 s |
| 2x2x4 | 4 | D both arenas | 1 | 1 | 0 | 19.534 s | 17.35 s | 1.258 s |
| 2x2x4 | 4 | E R7 winners | 0 | 0 | 1 | 19.561 s | 17.14 s | 1.184 s |
| 2x2x4 | 4 | F both arenas plus winners | 1 | 1 | 1 | 19.281 s | 17.18 s | 1.156 s |

R8 correctness for all fixed arena modes:

| Grid | Nodes | Exit | Info | Tiny pivots | sytrf 2x2 pivots | Solution error | Inertia `(pos,neg,zero)` |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2x2x2 | 2 | 0 | 0 | 0 | 0 | 2.13e-13 to 2.27e-13 | `(1814400, 1728000, 0)` |
| 2x2x4 | 4 | 0 | 0 | 0 | 0 | 2.13e-13 to 2.27e-13 | `(1814400, 1728000, 0)` |

The useful R7/R8 effect is now mostly setup-side: pooled pinned staging removes
the original pinned-staging setup penalty, and panel arenas reduce panel-copy
setup cost. The remaining factor-loop cost is still dominated by the V2
base-level work, so further tuning should target Schur/panel update scheduling
rather than additional allocation cleanup.

## Incomplete Or Failed Runs

The following saved runs are not valid timing comparisons:

| Run | Result |
|---|---|
| V2 CPU perf, 4x2x2, 4 nodes, job 54779146 | Timed out after 3D initialization before numerical factorization completed; no `status.txt` was copied locally. |
| V2 `GPU3DCONTRACT=1`, 4x2x2, 4 nodes, job 54798792 | Failed with exit code 143 after `inertia_from_dsytrf: malformed ipiv at end of array`; not comparable. |
| V2 CTA scatter pre-fix, 2x2x2, 2 nodes, job 54817311 | `GPU3DV2_CTA_SCATTER=0` completed with `FACTOR 39.264 s`, `Factorization_Time 32.98 s`, and solution error `2.131628e-13`; `GPU3DV2_CTA_SCATTER=1` timed out after 3D initialization because the first CTA implementation called the barrier-using device `find()` from only thread 0. |
| V2 CTA scatter pre-fix, 2x2x4, 4 nodes, job 54817312 | Cancelled after the CTA bug was identified; only partial `GPU3DV2_CTA_SCATTER=0` output was produced, so it is not a timing comparison. |

The CTA scatter lookup bug was fixed in:

```text
e93d9a09 Fix SymLDL V2 CTA scatter lookup
```

The later serial CTA lookup rerun is recorded above.
