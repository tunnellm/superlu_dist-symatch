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

Only the smaller `nlpkkt80` CTA A/B smoke has been rerun after that fix so far.
