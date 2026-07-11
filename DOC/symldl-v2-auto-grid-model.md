# SymLDL Automatic Process-Grid Model

The SymLDL process-grid selector chooses `(Pr, Pc, Pz)` after symbolic
factorization and before numerical distribution. It does not benchmark the
machine and does not require users to provide latency, bandwidth, or compute
rates. The selector first rejects grids that exceed the modeled host or GPU
memory budget, then compares feasible grids using structural work and
communication bounds.

## Structural model

For each supernode `k`, with width `s_k` and lower block heights `m_i`, the
model counts:

- dense diagonal work proportional to `s_k^3` on the CPU;
- panel transformations `2 m_i s_k^2`;
- lower-envelope Schur work `2 m_i m_j s_k` for real block pairs;
- local value traffic implied by those operations;
- GPU task launches and factorization synchronization points;
- partner-L, row-down, diagonal, and Z-layer communication payloads.

Each forest is numerically factored on one active depth layer. Other layers
hold partial updates for its ancestor panels; they do not repeat the panel or
Schur computation. The model follows the same binary reduction as
`ancestorReduction3dGPU()`: every process row on the panel-root process column
sends its local lower-panel segment toward the active layer, and the receiver
is charged for the accumulation. Thus mathematical factor FLOPs are invariant
with `Pz`, while depth communication and reduction work grow with the actual
forest-reduction edges.

The Schur count is accumulated with reverse row suffixes. It does not build a
dense Cartesian product or a task DAG.

The model follows the partition plan's selected diagonal roots, panel roots,
forest layers, and factor levels. A factor level is divided into scheduling
windows bounded by the configured lookahead and, for GPU execution, the
number of streams that fit in memory.

For every resource `r`, the model reports:

```text
total_r    = aggregate resource use over the factorization
critical_r = sum over windows of the largest rank or node exposure
waiting_r  = sum over windows of (rank_count * rank_max - rank_sum)
```

`waiting_r` is an imbalance diagnostic. It is not minimized directly because
removing useful work can increase this derived idle exposure.

## Communication topology

The selector discovers physical nodes with `MPI_Comm_split_type(...,
MPI_COMM_TYPE_SHARED, ...)` and uses the configured Z-major or XY-major rank
order to map `(pr, pc, pz)` coordinates to communicator ranks. Each modeled
message is therefore classified as intra-node or inter-node for the actual
allocation. The model records message counts, payload bytes, endpoint/node
critical loads, and host-staging bytes when CUDA-aware MPI is disabled.

In conventional alpha-beta notation, the communication portion of a machine
specific estimate could be written as:

```text
T_comm >= alpha_intra * critical_intra_messages
        + beta_intra  * critical_intra_bytes
        + alpha_inter * critical_inter_messages
        + beta_inter  * critical_inter_bytes
```

Compute, local-memory traffic, launches, synchronization, and host staging
have analogous positive coefficients. The automatic selector deliberately
does not guess those coefficients.

## Grid selection

A feasible candidate dominates another candidate only when it is no worse in
both total and critical exposure for every active resource and is strictly
better in at least one. Dominated candidates are removed.

If one candidate remains, it is selected with `dominant` confidence. For a
multi-candidate Pareto frontier, each total and critical resource exposure is
range-normalized over that frontier. The selector minimizes the largest
normalized regret (Chebyshev minimax), then summed regret. Memory pressure and
grid dimensions are deterministic tie breakers. This produces a defensible
best guess without fitting machine constants. The default report also shows
up to two nearby Pareto alternatives; detailed reporting prints the complete
frontier and every raw metric.

## Scope and limitations

- The objective covers factorization, not the current SymLDL solve.
- The schedule model aggregates levels and lookahead windows; it is not a
  cycle-accurate task simulation.
- Physical node placement is modeled, but NIC topology, routing congestion,
  GPU interconnect topology, and asynchronous progress rates are not.
- GPU and CPU paths activate different metric subsets. CPU selection is
  structurally supported, but initial performance validation focuses on GPU
  factorization.
- Historical timings are validation data only. They are not used to fit or
  tune the selector.

The validation target is that the selected grid is within 10 percent of the
fastest numerically valid historical `nlpkkt120` grid at each tested node
count, while the measured winner remains on the predicted Pareto frontier.
`nlpkkt80` is held out as a separate structural check.

## Validation snapshot

The July 2026 Perlmutter validation used A100 40 GB nodes, four MPI ranks per
node, 16 OpenMP threads per rank, MC80, rank order Z, and no fitted model
coefficients. For `nlpkkt120`, the selector reproduced the exact fastest grid
from the earlier exhaustive sweep at every tested node count:

| Nodes | Selected grid | Historical fastest grid | New factor time |
|---:|:---:|:---:|---:|
| 1 | `1x1x4` | `1x1x4` | 37.792 s |
| 2 | `2x1x4` | `2x1x4` | 20.732 s |
| 4 | `2x1x8` | `2x1x8` | 14.940 s |

The independently held-out `nlpkkt80` matrix was then evaluated against the
selected grid's two reported alternatives:

| Nodes | Selected grid/time | Alternative 1 | Alternative 2 |
|---:|:---|:---|:---|
| 1 | `1x1x4`, 6.309 s | `2x1x2`, 8.403 s | `1x2x2`, 13.741 s |
| 2 | `1x1x8`, 4.692 s | `2x1x4`, 4.989 s | `1x2x4`, 8.272 s |
| 4 | `2x1x8`, 3.621 s | `1x1x16`, 4.046 s | `4x2x2`, 7.870 s |

All runs completed with `info=0`, zero tiny pivots, unchanged inertia, and
solution error near `2e-13`. The model labeled these choices as ambiguous
ties rather than dominant results, so applications still receive alternatives
when the symbolic alpha-beta comparison cannot prove one grid universally
best.

A second holdout, SuiteSparse `PARSEC/Si41Ge41H72`, exposed the limitation of
using summed coordinate-wise regret to choose among an entirely nondominated
frontier. The 1- and 2-node choices were the fastest tested candidates, but
the 4-node choice was not:

| Nodes | Selected grid/time | Best tested grid/time | Result |
|---:|:---|:---|:---|
| 1 | `2x1x2`, 9.569 s | `2x1x2`, 9.569 s | selected winner |
| 2 | `2x1x4`, 7.195 s | `2x1x4`, 7.195 s | selected winner |
| 4 | `4x2x2`, 9.427 s | `2x1x8`, 5.736 s median | selection miss |

The missed `2x1x8` candidate remained on the Pareto frontier but ranked fourth
by summed regret, behind the two alternatives shown by the default report.
This is a reporting and decision-rule limitation, not a memory-filter or
numerical failure: every run completed with `info=0`, inertia
`(185321,318,0)`, and no-refinement solution error of order `1e-6`. Future
selection work should replace the summed-regret tie break with symbolic
coefficient-region analysis and, when desired, a short allocation-level
calibration of compute, launch, synchronization, and alpha-beta communication
costs.
