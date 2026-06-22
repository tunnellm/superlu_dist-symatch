# nlpkkt120 Unsymmetric V1 MC64 Baseline

Recorded: 2026-06-21

Scope: this file records the unsymmetric `GPU3DVERSION=1` MC64 baseline used
to compare against the current SymLDL V2 path on `nlpkkt120`.

Branch and fork:

```text
symldl-v2-solve-optimization
git@github.com:tunnellm/superlu_dist-symatch.git
```

Run checkout:

```text
88a02fd8 Record SymLDL V2 batched Schur timings
```

Source change baseline for the current SymLDL V2 batched result:

```text
84805b52 Batch SymLDL V2 Schur updates
```

## Unsymmetric V1 Baseline

This run used the Perlmutter perf build:

```text
/pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/build-perlmutter-v2-perf/EXAMPLE/pddrive3d
```

Setup:

```text
matrix: nlpkkt120
matrix file: /pscratch/sd/m/mtunnell/matrices_large/nlpkkt120/nlpkkt120.i32.bin
factorization: unsymmetric
driver: pddrive3d
nodes: 4 GPU nodes
ranks: 16 MPI ranks, 4 ranks per node
threads: 16 OMP threads per rank
grid: 2x2x4
lookahead: 32
GPU3DVERSION: 1
SymFact: 0
RowPerm: 1, LargeDiag_MC64
MPI_PROCESS_PER_GPU: 1
SLURM_CPU_BIND: cores
```

Run directory:

```text
/pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/results/nlpkkt120/20260621-194627-unsym-v1-mc64-grid2x2x4-4n-54813541
```

Local copy:

```text
/tmp/20260621-194627-unsym-v1-mc64-grid2x2x4-4n-54813541
```

Correctness:

| Metric | Value |
|---|---:|
| exit code | 0 |
| info | 0 |
| tiny pivots | 0 |
| solution error `||X-Xtrue||/||X||` | 4.944933e-13 |
| max component relative error | 3.663734e-13 |
| sytrf 2x2 pivots | 0 |
| inertia `(pos,neg,zero)` | `(0, 0, 0)` |

The inertia line is recorded as printed. The unsymmetric path does not compute
or report LDL inertia, so the `(0, 0, 0)` value is not comparable to SymLDL V2
inertia.

Top-level timing:

| Phase | Time |
|---|---:|
| EQUIL | 0.082 s |
| ROWPERM | 1.125 s |
| COLPERM | 49.907 s |
| SYMBFACT | 32.436 s |
| DISTRIBUTE | 12.519 s |
| FACTOR | 27.515 s |
| SOLVE | 1.793 s |

Internal factor timing:

| Metric | Time |
|---|---:|
| Factorization_Time | 20.47 s |
| Grid-0 Level-0 | 1.8491 s |
| Grid-0 Level-1 | 2.7826 s |
| Grid-0 Level-2 | 15.0863 s |

Solve timing:

| Metric | Avg | Min | Max |
|---|---:|---:|---:|
| forwardSolve | 1.0418 s | 1.0133 s | 1.0867 s |
| forwardSolve-compute | 0.5809 s | 0.4212 s | 0.7773 s |
| forwardSolve-comm | 0.4292 s | 0.2212 s | 0.5999 s |
| backSolve | 0.7010 s | 0.6527 s | 0.7400 s |
| backSolve-compute | 0.3828 s | 0.3474 s | 0.4273 s |
| backSolve-comm | 0.2860 s | 0.2237 s | 0.3545 s |

Memory:

| Metric | Sum | Avg | Max |
|---|---:|---:|---:|
| Total highmark | 54446.03 MB | 13611.51 MB | 13844.43 MB |
| NUMfact L/U | 52519.82 MB | - | 13357.61 MB |

## Comparison To SymLDL V2 Batched Schur

The SymLDL V2 comparison point is the 4-node `2x2x4` batched Schur run from:

```text
/pscratch/sd/m/mtunnell/superlu_dist-symatch-v2/results/nlpkkt120/20260621-191646-v2-batchAB-grid2x2x4-4n-54811760
```

| Case | FACTOR | Factorization_Time | SOLVE | Solution error |
|---|---:|---:|---:|---:|
| Unsymmetric V1 MC64 | 27.515 s | 20.47 s | 1.793 s | 4.944933e-13 |
| SymLDL V2, batch Schur off | 30.053 s | 26.24 s | 1.309 s | 2.415845e-13 |
| SymLDL V2, batch Schur on | 22.910 s | 19.12 s | 1.311 s | 2.131628e-13 |

Relative to unsymmetric V1 MC64, SymLDL V2 with batched Schur is:

| Metric | Result |
|---|---:|
| FACTOR speedup | 1.20x |
| FACTOR reduction | 16.74% |
| Factorization_Time speedup | 1.07x |
| Factorization_Time reduction | 6.60% |
| SOLVE speedup | 1.37x |

For user-visible timing, compare `FACTOR`. For diagnosing the numeric
factorization body, compare `Factorization_Time`.
