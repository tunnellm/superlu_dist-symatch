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

## Notes

Do not use `/tmp/superlu-perlmutter-results/nlpkkt80/v0_2x1x2_1n4r_8t.log` as the correctness baseline for this comparison. That run reported `FACTOR time 27.151 s` and `SOLVE time 0.660 s`, but also had solution error `3.648858e-01` and zero sytrf 2x2 pivots, so it is not comparable to the clean V0 baseline above.
