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

## Incomplete Or Failed Runs

The following saved runs are not valid timing comparisons:

| Run | Result |
|---|---|
| V2 CPU perf, 4x2x2, 4 nodes, job 54779146 | Timed out after 3D initialization before numerical factorization completed; no `status.txt` was copied locally. |
| V2 `GPU3DCONTRACT=1`, 4x2x2, 4 nodes, job 54798792 | Failed with exit code 143 after `inertia_from_dsytrf: malformed ipiv at end of array`; not comparable. |

