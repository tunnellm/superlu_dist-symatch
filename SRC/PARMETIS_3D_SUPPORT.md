# ParMETIS and parallel symbolic factorization in `pdgssvx3d`

## Purpose and scope

This note records the double-precision changes that allow the 3D driver,
`pdgssvx3d`, to use

```text
ColPerm = PARMETIS
ParSymbFact = YES
```

with three numerical factorization paths:

1. ordinary (unsymmetric) LU;
2. symmetric LU; and
3. symmetric-indefinite `LDL^T` (`GPU3DVERSION=2`).

The implementation is in the double-precision `SRC/double` path.  The 2D
driver already had a ParMETIS/parallel-symbolic route; the work described here
adapts its output to the 3D forest partition and the per-depth numerical
storage used by `pdgssvx3d`.

The main implementation commits are:

| Commit | Scope |
| --- | --- |
| `2d8d235a` | 3D ParMETIS and parallel symbolic support for `SymFact=NO` |
| `71b6415e` | matched ParMETIS ordering and parallel symbolic support for symmetric LU |
| `635b3c90` | distributed-symbolic adapters for symmetric `LDL^T` |

Related commits include `2ce867b5` (run MC80 only on rank 0 of layer 0),
`1aead757` (expose `ParSymbFact` as `-f` in
`pddrive3d-sym`), and `482dc32a` (use 64-bit symbolic nonzero counts and
printing).

## Supported combinations

| Path | Required options | Numerical representation | 3D numerical entry point |
| --- | --- | --- | --- |
| Ordinary LU | `SymFact=NO`, `ParSymbFact=YES`, `ColPerm=PARMETIS` | conventional distributed `L` and `U` | `pdgstrf3d` for version 0, or `pdgstrf3d_LUv1` for `GPU3DVERSION=1` |
| Symmetric LU | `SymFact=YES`, a symmetric matching row permutation, `ParSymbFact=YES`, `ColPerm=PARMETIS`, `GPU3DVERSION=0` | conventional distributed `L` and `U`; optional `CommL` schedule | `pdgstrf3d` |
| Symmetric `LDL^T` | `SymFact=YES`, a symmetric matching row permutation, `ParSymbFact=YES`, `ColPerm=PARMETIS`, `GPU3DVERSION=2` | V2 L-only panels plus diagonal `D` data | `pdgstrf3d_LUv2` |

At the API level, a fresh 3D ParMETIS run sets `Algo3d=YES`, `Fact=DOFACT`,
`ParSymbFact=YES`, and `ColPerm=PARMETIS`.  Ordinary LU keeps
`SymFact=NO`.  A symmetric run selects `SUITOR`, `SUMAC`, or `MC80` as
`RowPerm`; `pddrive3d-sym` then sets `SymFact=YES`.  In the current example
driver, `-f 1 -q 5` selects parallel symbolic factorization and ParMETIS,
while `-p 4`, `-p 5`, and `-p 6` select those three symmetric matchers.  Use
the named enumeration constants in library code rather than relying on these
numeric command-line values.

For V2 `LDL^T`, `GPU3DV2_MAPPING` must be unset, empty, or `CYCLIC`.
`ID`, `ID2D`, and `GREEDY` are intentionally rejected.  The compact
symbolic structures produced by `ddist_symbLU` use block-cyclic `PCOL` and
`PROW` ownership, so they cannot directly describe those alternate V2 owner
mappings.

The following limitations apply to the new 3D parallel-symbolic route:

- it currently supports a fresh `DOFACT` operation, not
  `SamePattern`, `SamePattern_SameRowPerm`, or other symbolic-pattern reuse;
- a ParMETIS-enabled build (`HAVE_PARMETIS`) is required;
- symmetric paths require `SLU_IS_SYMATCH_ROWPERM(options->RowPerm)`;
- symmetric parallel symbolic factorization is supported only for
  `GPU3DVERSION=0` and `GPU3DVERSION=2`; and
- this implementation has not automatically been propagated to the single,
  complex, or double-complex drivers.

## Process-grid terminology

Let the 3D process grid be `Pr x Pc x Pz`.

- `grid3d->grid2d` is the `Pr x Pc` grid within one depth layer.
- `grid3d->zscp.comm` connects ranks with the same `(prow,pcol)` coordinate
  across the `Pz` layers.
- "layer 0" means the 2D grid whose z coordinate is zero.
- A "z-line broadcast" sends data from the layer-0 rank `(prow,pcol,0)` to
  the matching `(prow,pcol,z)` rank in every other layer.

Ordering and parallel symbolic factorization run only on layer 0.  The
resulting compact symbolic data are then replicated along z-lines.  Every
depth layer independently constructs its 3D partition metadata and assembles
only the factor storage assigned to that layer.

## Common top-level call graph

The three paths share the following driver structure:

```text
pdgssvx3d                                      [all 3D ranks]
|
+-- dGatherNRformat_loc3d_allgrid
|     Assemble each z-line's contributions into matching 2D-local A/B data
|     on every depth.  Layer 0 subsequently preprocesses its copy.
|
+-- if grid3d->zscp.Iam == 0                  [layer 0 only]
|   |
|   +-- dperform_row_permutation
|   |
|   +-- ParMETIS column ordering
|   |     ordinary LU: get_perm_c_parmetis(A, perm_r, ...)
|   |     symmetric:   coarsen_graph_v3
|   |                  -> dDistributeGlobalNCByRows
|   |                  -> get_perm_c_parmetis(quotient graph, ...)
|   |                  -> dExpandMatchedParmetisOrder
|   |
|   +-- symbfact_dist
|   |     Produce process-local parallel-symbolic L/U data.
|   |
|   +-- ddist_symbLU
|   |     Reconstruct numerical supernodes and redistribute the symbolic
|   |     structure into compact block-cyclic L/U arrays.
|   |
|   +-- ddist_build_supno_tree
|         Build and validate a supernodal dependency tree from the
|         ParMETIS separator tree and compact L/U dependencies.
|
+-- dbcastDistSymbLU                           [one broadcast per z-line]
|     Replicate A, permutations/scaling, xsup/supno, the supernodal tree,
|     compact L/U arrays, and symbolic-memory accounting to every depth.
|
+-- apply the final column mapping to local A  [all 3D ranks]
|     ordinary LU: colind <- perm_c[colind]
|     symmetric:   colind <- perm_c[perm_r[colind]]
|
+-- construct the 3D forest partition          [all layers]
|     ordinary/symmetric LU: dnewTrfPartitionInitFromSetree
|     symmetric LDLT:       dSymV2TrfPartitionInitFromDistSymb
|
+-- assemble per-depth numerical storage       [all layers]
|     ordinary/symmetric LU: ddist_psymbtonum3d
|     symmetric LDLT:        dSymV2Distribute3dFromSymb
|
+-- initialize factorization-specific metadata
|
+-- numerical 3D factorization                 [all layers]
      ordinary LU:       pdgstrf3d or pdgstrf3d_LUv1
      symmetric LU:      pdgstrf3d
      symmetric LDLT:    pdgstrf3d_LUv2
```

Two details are important:

1. `dbcastDistSymbLU` broadcasts between matching ranks along z-lines, not
   from one global root to every rank.  Thus the block-cyclic 2D distribution
   created on layer 0 is preserved on every layer.
2. The partition must be initialized before numerical assembly.  Its
   `superGridMap` tells the assembly routine which supernodes are retained on
   the current depth.

## Shared ParMETIS and parallel-symbolic stages

### Layer-zero preprocessing

`pdgssvx3d` first calls `dGatherNRformat_loc3d_allgrid`, which assembles a
matching 2D-local A/B view on every depth.  Normally it gathers contributions
along each z-line; the natural-row-label/Z-major case gathers over the full 3D
communicator and extracts the appropriate natural contiguous block.  Layer 0
then performs equilibration, row permutation, column ordering, and symbolic
factorization on its copy.  The other layers do not repeat those operations;
`dbcastDistSymbLU` later propagates the modified layer-0 state along z-lines.

For `ParSymbFact=YES`, the driver creates `symb_comm` for the first
`noDomains` ranks of the layer-0 2D communicator.  `noDomains` is a power of
two because the separator metadata used by `symbfact_dist` is stored as a
complete binary tree.

For ordinary LU, the established domain-count rule is retained.  For a
matched symmetric quotient graph, the power of two is also limited by the
number of quotient vertices, preventing empty-domain configurations caused by
using more symbolic domains than matched vertices.

### Ordinary ParMETIS ordering

The ordinary path calls

```text
get_perm_c_parmetis(A, perm_r, perm_c, ...)
```

on the distributed input matrix.  ParMETIS returns:

- `perm_c`, the column permutation;
- `sizes`, the sizes of the nested-dissection domains and separators; and
- `fstVtxSep`, their starting vertices.

No matched-pair contraction or expansion is required.

### Matched symmetric ParMETIS ordering

Symmetric LU and symmetric `LDL^T` use the same ordering front end:

```text
dperform_row_permutation
|
+-- symmetric matcher on rank 0 of layer 0
|     SUITOR / SUMAC / MC80
|
+-- broadcast perm_r and crs_info across layer 0
|
+-- apply_perm_sym: B = Pr A Pr^T

coarsen_graph_v3(GA, GA_c, crs_info)
|
+-- dDistributeGlobalNCByRows(GA_c, GA_c_loc)
|
+-- get_perm_c_parmetis(GA_c_loc, identity, crs_perm_c, ...)
|
+-- dExpandMatchedParmetisOrder
      -> fine-column perm_c
      -> fine-column sizes/fstVtxSep
      -> options->indicator_2x2
```

The matcher describes the contracted graph in `crs_info`:

- `crs_vrts[c]` is 1 or 2, the number of fine vertices represented by
  quotient vertex `c`; and
- `ftoc` records the fine-to-coarse relationship.

`dDistributeGlobalNCByRows` converts the replicated compressed-column
quotient graph into the block-row `NR_loc` form expected by ParMETIS.
`dExpandMatchedParmetisOrder` then expands the quotient permutation and every
separator interval back to fine-column coordinates.  It also rebuilds
`options->indicator_2x2` in the final order, with the grammar

```text
1 : a 1x1 pivot
2 : the first column of a 2x2 pivot
0 : the second column of that 2x2 pivot
```

Consequently, a matched pair remains consecutive and no expanded separator
boundary cuts through a 2x2 pivot.

### Pair-safe parallel symbolic factorization

The symmetric changes to `symbfact_dist` are deliberately below the common
driver interface.  They pass `options->indicator_2x2` through the internal
symbolic call graph:

```text
symbfact_dist
|
+-- symbfact_mapVtcs(..., pair_indicator, ...)
|     Keep process-ownership blocks aligned to complete 1x1/2x2 pivots.
|
+-- symbfact_distributeMatrix(..., congruentCols=YES, ...)
|     Apply perm_r to both row and column indices before perm_c.
|
+-- domain_symbfact(..., pair_indicator)
|     `-- blk_symbfact(..., pair_indicator)
|
`-- intraLvl_symbfact(..., pair_indicator)
      `-- blk_symbfact(..., pair_indicator)
```

For the symmetric 3D path, these routines:

- validate the complete `1` or `2,0` indicator grammar;
- validate that separator and symbolic-ownership boundaries do not split a
  pair;
- force both columns of a 2x2 pivot into the same symbolic supernode;
- start the pair in a new supernode when the maximum-supernode limit lacks
  room for both columns;
- disable dense-separator shortcuts that do not preserve forced pairs; and
- use a congruent structural permutation rather than the ordinary LU column
  transformation.

`ddist_symbLU` has a second pair-safety check while it reconstructs the final
numerical supernodes.  Process-owner changes, symbolic-supernode changes, and
the maximum-supernode size may start a new numerical supernode, but never
between the two columns of a matched pair.

### Compact symbolic conversion

`symbfact_dist` produces a `Pslu_freeable_t` distributed according to the
parallel-symbolic ownership.  `ddist_symbLU` converts it to the numerical
block-cyclic layout and installs the final numerical `xsup` and `supno` in
`Glu_persist`.

Its outputs are:

```text
dist_xlsub, dist_lsub   compact symbolic L block columns
dist_xusub, dist_usub   compact symbolic U block rows
dist_symb_mem           persistent-memory accounting (negative on success)
```

For a local L panel, the compact data contain all scalar rows owned by the
current process row, plus one representative row for every other active
process row.  The owned rows define local factor storage; the representatives
describe which process rows participate in panel communication.  The compact
U data carry the analogous block-row information.

`ddist_symbLU` consumes and frees the process-local structures in
`Pslu_freeable`.

### Constructing a tree for the 3D forest

The parallel symbolic routine does not provide the replicated serial
`Glu_freeable` tree expected by the original 3D partition initializer.
`ddist_build_supno_tree` therefore constructs a conservative supernodal tree
from `sizes` and `fstVtxSep`:

1. supernodes within each separator are chained in increasing order;
2. the last supernode of a separator points to the first supernode in its
   nearest nonempty ancestor separator; and
3. the artificial node `nsupers` joins roots of a possible forest.

The routine validates that every compact L and U update endpoint is the source
itself or an ancestor of the source in this tree.  This guarantees that the
tree is safe for 3D layer assignment even though it can be more conservative
than the exact elimination tree.

The driver also synthesizes a scalar `LUstruct->etree` consistent with this
supernodal tree because later common code still expects that field.  The
ParMETIS 3D partition is built from `dist_setree`, not by recomputing the tree
from this synthesized scalar array.

### Broadcasting to all depths

`dbcastDistSymbLU` is called collectively by all 3D ranks.  Along each
`grid3d->zscp.comm`, it broadcasts:

- `Glu_persist->{xsup,supno}`;
- the synthesized scalar etree and `dist_setree`;
- `dist_{x,l}lsub` and `dist_{x,u}usub`;
- `dist_symb_mem`;
- permutations and scaling vectors; and
- the layer-zero contents of the corresponding local part of `A`.

Every nonzero depth allocates a private copy.  This is required because both
symbolic-to-numeric adapters reorder and free the compact arrays.

## Path 1: ordinary LU

### Call graph

```text
pdgssvx3d, layer 0
|
+-- get_perm_c_parmetis(A, perm_r, perm_c, ...)
+-- symbfact_dist                              [no pair indicator]
+-- ddist_symbLU
`-- ddist_build_supno_tree

pdgssvx3d, all layers
|
+-- dbcastDistSymbLU
+-- dnewTrfPartitionInitFromSetree
|     `-- dnewTrfPartitionInitWithOwnedSetree
|           -> setree2list / calcTreeWeight
|           -> getForests / createSuperGridMap
|
+-- ddist_psymbtonum3d
|     `-- ddist_psymbtonum_from_symb(..., superGridMap)
|
+-- pdflatten_LDATA
+-- dinit3DLUstructForest
`-- pdgstrf3d or pdgstrf3d_LUv1
```

`dnewTrfPartitionInitFromSetree` copies and validates `dist_setree`, then calls
the refactored common initializer `dnewTrfPartitionInitWithOwnedSetree`.
The copy becomes `trf3Dpart->gEtreeInfo.setree` and is owned by the partition.
This avoids deriving a new tree from factor structures that do not exist yet.

`ddist_psymbtonum3d` calls the common numerical assembly helper with the
partition's `superGridMap`.  The helper uses the complete symbolic structure
to compute communication and receive-buffer requirements, but allocates
persistent L/U panels only for supernodes present on the current depth.

## Path 2: symmetric LU

### Call graph

```text
pdgssvx3d, layer 0
|
+-- dperform_row_permutation                  [symmetric matching]
+-- coarsen_graph_v3
+-- dDistributeGlobalNCByRows
+-- get_perm_c_parmetis                       [quotient graph]
+-- dExpandMatchedParmetisOrder
+-- symbfact_dist                              [pair-aware]
+-- ddist_symbLU                               [pair-safe reconstruction]
`-- ddist_build_supno_tree

pdgssvx3d, all layers
|
+-- dbcastDistSymbLU
+-- dnewTrfPartitionInitFromSetree
+-- ddist_psymbtonum3d
|     +-- ddist_psymbtonum_from_symb
|     `-- dSetupCommL, when options->CommL == YES
|
+-- pdflatten_LDATA
+-- dinit3DLUstructForest
`-- pdgstrf3d                                  [GPU3DVERSION=0]
```

Symmetric LU retains conventional L and U factor structures.  When
`options->CommL==YES`, `dSetupCommL` derives the L-to-U redistribution plan
from the already assembled and depth-filtered L/U structures.  Building this
plan after `ddist_psymbtonum3d` is essential: each depth can retain a different
subset of the global forest.

`dDestroyCommL`, called from `dDestroy_LU`, releases the per-panel requests,
statuses, payload buffers, and send/receive descriptors.

## Path 3: symmetric `LDL^T`

The V2 backend cannot use the conventional partition and numerical assembly
unchanged.  It has different owner metadata, a different cost model, and no
stored conventional U factor.

### Call graph

```text
pdgssvx3d, layer 0
|
+-- the same matched quotient-graph ParMETIS path as symmetric LU
+-- the same pair-aware symbfact_dist path
+-- ddist_symbLU
`-- ddist_build_supno_tree

pdgssvx3d, all layers
|
+-- dbcastDistSymbLU
|
+-- dSymV2TrfPartitionInitFromDistSymb
|     +-- dSymV2CopySetree
|     +-- setree2list
|     +-- dSymV2CalcLDLTreeWeightFromDistSymb
|     |     `-- dSymV2SetLDLTreeWeightFromRows
|     `-- dSymV2FinishTrfPartitionInit
|           +-- fillEtreeInfo
|           +-- dSymV2InitLDLOwners
|           +-- dSymV2BuildLDLSchedule
|           `-- dSymV2InstallLDLForest
|
+-- dSymV2Distribute3dFromSymb
|     `-- dSymV2Distribute3d_LDL_impl
|           +-- dReDistribute_A
|           +-- dSymV2RedistributeLowerToLDL
|           +-- dSymV2OrderDistributedDiagonalRows
|           `-- construct retained V2 L/D panels and solve metadata
|
+-- dSymV2LluBufInit / identity supernode permutation
`-- dCreateLUgpuHandle -> pdgstrf3d_LUv2
```

### Partition adapter

`dSymV2TrfPartitionInitFromDistSymb` accepts `dist_setree` and the compact L
structure directly.  It requires cyclic ownership and performs the following
steps:

1. copy and validate the supplied supernodal tree;
2. build its `treeList`;
3. recover the global row count of every L panel;
4. evaluate the V2 LDL cost model and accumulated tree weights;
5. assign cyclic panel and diagonal roots;
6. build V2 factor levels and local panel/row indices; and
7. install the V2 forests and `superGridMap`.

Recovering a panel's row count requires care.  Each compact L panel contains
owned scalar rows and remote participation representatives.  On each process
row, `dSymV2CalcLDLTreeWeightFromDistSymb` counts only entries satisfying

```text
PROW(supno[row]) == myrow
```

and sums those counts across the layer's 2D communicator.  Thus every real
structural row is counted once and remote representatives are not mistaken for
additional factor rows.

`dSymV2InitLDLOwners` receives `Glu_freeable=NULL` in this route.  That is safe
for the cyclic mapping, whose roots are simply `PCOL(k)` and `PROW(k)`, but is
another reason not to enable the graph-dependent `ID`, `ID2D`, or `GREEDY`
mappings here.

### L-only distribution adapter

`dSymV2Distribute3dFromSymb` invokes a generalized
`dSymV2Distribute3d_LDL_impl`.  The implementation can now read either:

- serial symbolic panels from `Glu_freeable->{xlsub,lsub}`, indexed by the
  first scalar column `fsupc`; or
- compact parallel-symbolic panels from `dist_{x,l}lsub`, indexed by the local
  block column `LBj(jb,grid)`.

For the distributed input, it verifies that V2 roots still equal the
block-cyclic `PCOL(jb)` and `PROW(jb)`.  Owned compact rows become local L
blocks; remote representatives are used only to establish panel send
metadata.

The parallel symbolic routine does not promise that scalar rows of the
diagonal block appear first or in scalar-column order.
`dSymV2OrderDistributedDiagonalRows` performs this normalization on the
diagonal process row before the dense diagonal block is assembled.

The adapter then:

- redistributes numerical entries of `A`;
- maps the symmetric lower triangle to the V2 panel/diagonal owners;
- allocates only panels retained by the current layer's `superGridMap`;
- creates V2 L, diagonal inverse, solve, and communication metadata;
- leaves conventional U pointers null; and
- does not build `CommL`, because V2 reconstructs and exchanges partner
  fragments rather than maintaining a conventional U factor.

After assembly, `dSymV2Distribute3dFromSymb` frees all four compact symbolic
arrays.  The U arrays are accepted only to transfer their ownership to the
adapter and are then discarded; V2 does not inspect them while constructing
its L-only numerical storage.  Their symbolic information was already used
while constructing and validating the common supernodal tree.

## New functions and responsibilities

### General 3D parallel-symbolic support

| Function | File | Responsibility |
| --- | --- | --- |
| `dnewTrfPartitionInitWithOwnedSetree` | [`double/d3DPartition.c`](double/d3DPartition.c) | Refactored common 3D forest initialization when the caller transfers ownership of a supernodal tree. |
| `dnewTrfPartitionInitFromSetree` | [`double/d3DPartition.c`](double/d3DPartition.c) | Validate and copy a caller-supplied supernodal tree before common 3D partition initialization. |
| `dbcastDistSymbLU` | [`double/d3DPartition.c`](double/d3DPartition.c) | Broadcast compact symbolic state and matching local A data from layer 0 along each z-line. |
| `ddist_psymbtonum_from_symb` | [`double/pdsymbfact_distdata.c`](double/pdsymbfact_distdata.c) | Shared 2D/3D compact-symbolic-to-numerical L/U assembly; optionally filters persistent storage through `superGridMap`. |
| `ddist_psymbtonum3d` | [`double/pdsymbfact_distdata.c`](double/pdsymbfact_distdata.c) | 3D wrapper that requires an initialized partition and passes its `superGridMap` to the common assembler. |
| `ddist_build_supno_tree` | [`double/pdsymbfact_distdata.c`](double/pdsymbfact_distdata.c) | Build a conservative supernodal tree from ParMETIS separators and validate all compact L/U dependencies. |
| `ddist_compare_sep_intervals` | [`double/pdsymbfact_distdata.c`](double/pdsymbfact_distdata.c) | Private ordering helper used while validating separator coverage. |

`ddist_symbLU` was made externally callable rather than newly created.  The
2D function `ddist_psymbtonum` remains available and now delegates to
`ddist_symbLU` followed by `ddist_psymbtonum_from_symb` with no depth filter.

### Matched symmetric support

| Function | File | Responsibility |
| --- | --- | --- |
| `dDistributeGlobalNCByRows` | [`double/pdgssvx3d.c`](double/pdgssvx3d.c) | Convert a replicated matched quotient graph into distributed block-row storage for ParMETIS. |
| `dExpandMatchedParmetisOrder` | [`double/pdgssvx3d.c`](double/pdgssvx3d.c) | Expand the quotient permutation and separators to fine columns and rebuild `indicator_2x2`. |
| `dSetupCommL` | [`double/pddistribute3d.c`](double/pddistribute3d.c) | Build symmetric-LU L-to-U send/receive schedules from the retained numerical factors. |
| `dDestroyCommL` | [`double/pddistribute3d.c`](double/pddistribute3d.c) | Destroy every allocation owned by those schedules. |

### Statistics and reporting follow-up

| Function | File | Responsibility |
| --- | --- | --- |
| `countnz_dist64` | [`prec-independent/util.c`](prec-independent/util.c) | Count serial-symbolic L and U nonzeros in `int64_t`, avoiding overflow independently of the `int_t` build configuration. |

The legacy `countnz_dist` interface remains as a compatibility wrapper around
`countnz_dist64`.  The parallel `symbfact_dist` route likewise accumulates
local L/U counts as `int64_t`, performs an `MPI_Allreduce` over the complete
layer-0 numerical communicator (including ranks not in `symb_comm`), installs
the global total in `Pslu_freeable->nnzLU`, and prints the three factor counts
with `PRId64`.  This makes serial-METIS and parallel-ParMETIS statistics use
the same width and ensures every layer-0 rank receives valid accounting.

### V2 `LDL^T` distributed-symbolic support

| Function | File | Responsibility |
| --- | --- | --- |
| `dSymV2TrfPartitionInitFromDistSymb` | [`double/dsymldl_v2_partition.c`](double/dsymldl_v2_partition.c) | Public V2 partition adapter for a distributed supernodal tree and compact symbolic L. |
| `dSymV2CalcLDLTreeWeightFromDistSymb` | [`double/dsymldl_v2_partition.c`](double/dsymldl_v2_partition.c) | Recover each panel's true global row count without counting participation representatives. |
| `dSymV2SetLDLTreeWeightFromRows` | [`double/dsymldl_v2_partition.c`](double/dsymldl_v2_partition.c) | Apply the common V2 LDL cost model using supplied panel-row counts. |
| `dSymV2CopySetree` | [`double/dsymldl_v2_partition.c`](double/dsymldl_v2_partition.c) | Validate and copy the supplied tree for partition ownership. |
| `dSymV2PrepareTrfPartition` | [`double/dsymldl_v2_partition.c`](double/dsymldl_v2_partition.c) | Initialize common and V2 partition fields before construction. |
| `dSymV2FinishTrfPartitionInit` | [`double/dsymldl_v2_partition.c`](double/dsymldl_v2_partition.c) | Shared finalization for serial- and distributed-symbolic V2 partitions. |
| `dSymV2Distribute3dFromSymb` | [`double/dsymldl_v2_distribution.c`](double/dsymldl_v2_distribution.c) | Public compact-symbolic-to-V2-numerical adapter; consumes all symbolic L/U arrays. |
| `dSymV2OrderDistributedDiagonalRows` | [`double/dsymldl_v2_distribution.c`](double/dsymldl_v2_distribution.c) | Put diagonal scalar rows first and in order before dense diagonal-block assembly. |

The internal `dSymV2Distribute3d_LDL_impl` was generalized to serve both the
existing serial-symbolic entry point `dSymV2Distribute3d` and the new
distributed-symbolic entry point `dSymV2Distribute3dFromSymb`.

Public declarations were added to
[`include/pddistribute3d.h`](include/pddistribute3d.h),
[`include/dsymldl_v2_partition.h`](include/dsymldl_v2_partition.h), and
[`include/superlu_ddefs.h`](include/superlu_ddefs.h).  The precision-independent
`countnz_dist64` declaration is in
[`include/superlu_defs.h`](include/superlu_defs.h).

## Important modifications to existing functions

| Function | Change |
| --- | --- |
| `pdgssvx3d` | Added global option guards, layer-zero compact-symbolic conversion, z-line broadcast, branch-specific partition/distribution routing, and cleanup. |
| `dperform_row_permutation` | Allows `GA` to be absent for `NOROWPERM`; symmetric matchers execute on rank 0 and broadcast their permutation/contraction metadata across layer 0. |
| `symbfact_dist` and its internal helpers | Accept pair metadata, use congruent columns for symmetric structure, preserve 2x2 pivots across separators/owners/supernodes, and retain ordinary behavior when the pair indicator is null. |
| `ddist_symbLU` | Exposed for the driver, preserves 2x2 pairs during numerical-supernode reconstruction, and returns compact L/U state for later z-line broadcast. |
| `pddistribute3d_Yang` | Uses the extracted `dSetupCommL` helper instead of an embedded symmetric-LU setup block. |
| `dDestroy_LU` | Calls `dDestroyCommL` before destroying factor storage. |
| `dSymV2TrfPartitionInit` | Shares preparation and finalization helpers with the new distributed-symbolic partition route. |
| `dSymV2Distribute3d_LDL_impl` | Accepts either serial or compact distributed symbolic L input. |

## Ownership and lifetime rules

The cleanup order is part of the interface and should be preserved during
future refactoring.

| Object | Created by | Consumed or owned by |
| --- | --- | --- |
| `Pslu_freeable` internal arrays | `symbfact_dist` | consumed and freed by `ddist_symbLU` on layer 0 |
| `Glu_persist->{xsup,supno}` | `ddist_symbLU` | persistent `LUstruct` state; copied down z-lines |
| `dist_{x,l}lsub`, `dist_{x,u}usub` | `ddist_symbLU` | copied by `dbcastDistSymbLU`; each rank's copy is consumed and freed by `ddist_psymbtonum3d` or `dSymV2Distribute3dFromSymb` |
| `dist_setree` | `ddist_build_supno_tree` | copied by the chosen partition initializer, then freed by `pdgssvx3d` |
| `trf3Dpart->gEtreeInfo.setree` | partition initializer | owned and freed with `dtrf3Dpartition_t` |
| `sizes`, `fstVtxSep`, `symb_comm` | ParMETIS/front end | retained through `symbfact_dist` and tree construction, then freed before numerical assembly |
| `crs_info` temporary arrays | symmetric matcher | used through quotient expansion, then freed on layer 0 |
| `options->indicator_2x2` | matched-order expansion | used by pair-aware symbolic factorization and numerical-supernode reconstruction on layer 0; freed near the end of `pdgssvx3d` and reset to null |
| `Send_CommL`, `Recv_CommL` | `dSetupCommL` | owned by `dLocalLU_t`; freed by `dDestroyCommL` |

The compact parallel-symbolic interfaces `ddist_symbLU`,
`ddist_psymbtonum3d` (and its common helper), and
`dSymV2Distribute3dFromSymb` use the following memory-reporting convention:

- a nonpositive return value means success and represents persistent bytes;
- a positive value means allocation failure and represents approximate bytes
  allocated before failure.

The numerical adapters and the LU path's temporary `ddist_A` redistribution
accumulate byte counts in `double` and cast operands before multiplying
dimensions.  This is required even in a 32-bit `int_t` build: an individual
factor panel or redistributed portion of A can exceed 2 GiB although its
element count still fits in `int_t`.  Performing byte products in signed
32-bit accounting arithmetic can otherwise wrap negative and be mistaken for
a positive, out-of-memory return after applying the convention above.  This
is especially easy to miss because V2 LDLT uses its own A-redistribution path,
whereas ordinary and symmetric LU both pass through `ddist_A`.

For the V2 distributed-symbolic route,
`dSymV2Distribute3dFromSymb` returns the negative sum of the compact-symbolic
and V2 persistent numerical storage.  This matches the parallel-symbolic
memory interpretation in `dSymV2PrintFactorStats`.  The pre-existing
serial-symbolic `dSymV2Distribute3d` interface retains its older convention of
returning successful persistent memory as a positive value.

## Why the 3D design does not call the 2D converter before partitioning

The 2D `ddist_psymbtonum` creates a full conventional L/U representation on
every calling 2D grid.  Calling it on layer 0 and then copying or repartitioning
that representation would:

- materialize the complete numerical symbolic/factor layout before knowing
  the depth assignment;
- concentrate unnecessary memory on layer 0; and
- make the V2 L-only representation pass through an unused conventional U
  representation.

Instead, the 3D path keeps the compact output of `ddist_symbLU`, constructs the
forest first, broadcasts the compact data, and lets each depth assemble only
its retained panels.  This is the central difference between the 2D and 3D
routes.

## Validation and release checklist

The implementation has been syntax-checked with both normal and `_LONGINT`
index configurations.  CMake includes the V2 implementation files; the legacy
`SRC/Makefile` does not list them and should not be used to validate the V2
path without first updating that build description.

For a release or a backport, test at least the following:

1. ordinary LU with `ParSymbFact=YES`, `ColPerm=PARMETIS`, at `Pz=1` and
   `Pz>1`;
2. symmetric LU with both 1x1 and 2x2 matched pivots, at `Pz=1` and `Pz>1`;
3. CPU V2 `LDL^T` with `GPU3DV2_MAPPING=CYCLIC`, followed by the GPU V2 run;
4. residual/error checks against the corresponding serial-symbolic path;
5. factor nonzero counts and memory reporting in both normal and `_LONGINT`
   builds;
6. a negative V2 test with `GPU3DV2_MAPPING=ID`, `ID2D`, or `GREEDY`, which
   must abort before entering layer-specific collectives;
7. a negative symmetric test with `GPU3DVERSION=1`, which must also be
   rejected; and
8. repeated construction/destruction under a memory checker to cover compact
   symbolic arrays, copied trees, and `CommL` payloads.

When extending this work to another precision, port the driver routing,
partition/broadcast helpers, compact-symbolic numerical adapter, pair-aware
changes in the precision-independent symbolic code, declarations, and cleanup
as one unit.  Porting only `get_perm_c_parmetis` is insufficient: the 3D
partition and ownership/lifetime adaptations are what make the parallel
symbolic result usable at `Pz>1`.
