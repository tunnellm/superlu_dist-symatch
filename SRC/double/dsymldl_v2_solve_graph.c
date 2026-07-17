/*! @file
 * \brief Immutable retained-L graph shared by SymLDL solve backends.
 */

#include "dsymldl_v2_solve_graph.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int_t slot;
    int_t panel_gid;
    int_t edge;
} dSymLDLReverseEntry;

typedef struct {
    int_t order;
    int rank;
} dSymLDLRankOrder;

static size_t
symldl_graph_checked_product(size_t a, size_t b, const char *message)
{
    if (a != 0 && b > SIZE_MAX / a)
        ABORT(message);
    return a * b;
}

static void *
symldl_graph_alloc(size_t count, size_t size, const char *message)
{
    void *ptr;
    if (count == 0)
        return NULL;
    ptr = SUPERLU_MALLOC(symldl_graph_checked_product(count, size, message));
    if (ptr == NULL)
        ABORT(message);
    return ptr;
}

static void *
symldl_graph_copy(const void *source, size_t count, size_t size,
                  const char *message)
{
    void *copy = symldl_graph_alloc(count, size, message);
    if (count > 0)
        memcpy(copy, source, count * size);
    return copy;
}

static int
symldl_graph_reverse_cmp(const void *left, const void *right)
{
    const dSymLDLReverseEntry *a = (const dSymLDLReverseEntry *) left;
    const dSymLDLReverseEntry *b = (const dSymLDLReverseEntry *) right;
    if (a->slot != b->slot)
        return a->slot < b->slot ? -1 : 1;
    if (a->panel_gid != b->panel_gid)
        return a->panel_gid > b->panel_gid ? -1 : 1;
    return a->edge < b->edge ? -1 : a->edge != b->edge;
}

static int
symldl_graph_rank_asc(const void *left, const void *right)
{
    const dSymLDLRankOrder *a = (const dSymLDLRankOrder *) left;
    const dSymLDLRankOrder *b = (const dSymLDLRankOrder *) right;
    if (a->order != b->order)
        return a->order < b->order ? -1 : 1;
    return a->rank < b->rank ? -1 : a->rank != b->rank;
}

static int
symldl_graph_rank_desc(const void *left, const void *right)
{
    const dSymLDLRankOrder *a = (const dSymLDLRankOrder *) left;
    const dSymLDLRankOrder *b = (const dSymLDLRankOrder *) right;
    if (a->order != b->order)
        return a->order > b->order ? -1 : 1;
    return a->rank < b->rank ? -1 : a->rank != b->rank;
}

static void
symldl_graph_tree_null(dSymLDLTreeNode *node)
{
    node->root_rank = -1;
    node->parent_rank = -1;
    node->children[0] = -1;
    node->children[1] = -1;
    node->rank_index = -1;
    node->child_count = 0;
    node->active = 0;
}

static void
symldl_graph_tree_create(dSymLDLTreeNode *node, const int *ranks,
                         int rank_count, int my_rank)
{
    int index = -1;
    symldl_graph_tree_null(node);
    for (int i = 0; i < rank_count; ++i) {
        if (ranks[i] == my_rank) {
            index = i;
            break;
        }
    }
    if (index < 0)
        return;

    node->active = 1;
    node->root_rank = ranks[0];
    node->rank_index = index;
    if (index > 0)
        node->parent_rank = ranks[(index - 1) / 2];
    for (int child = 2 * index + 1;
         child < rank_count && node->child_count < 2; ++child)
        node->children[node->child_count++] = ranks[child];
}

static int
symldl_graph_panel_active(const dSymLDLSolveGraph *graph, int_t gid)
{
    int_t slot;
    if (gid < 0 || gid >= graph->nsupers)
        return 0;
    slot = graph->panel_local_index[gid];
    return slot >= 0 && slot < graph->panel_count &&
           graph->panels[slot].gid == gid && graph->panels[slot].active;
}

static void
symldl_graph_validate(dSymLDLSolveGraph *graph,
                      dtrf3Dpartition_t *partition, gridinfo3d_t *grid3d)
{
    gridinfo_t *grid = &grid3d->grid2d;
    if (graph->panel_count != partition->symV2LocalPanelCount ||
        graph->row_count != partition->symV2LocalRowCount)
        ABORT("SymLDL solve graph local dimensions are inconsistent.");
    if (graph->xsup[0] != 0 || graph->xsup[graph->nsupers] != graph->n)
        ABORT("SymLDL solve graph supernode bounds are inconsistent.");

    for (int_t k = 0; k < graph->nsupers; ++k) {
        int_t width = graph->xsup[k + 1] - graph->xsup[k];
        if (width <= 0 || width > INT_MAX ||
            graph->diag_roots[k] < 0 ||
            graph->diag_roots[k] >= grid->nprow ||
            graph->panel_roots[k] < 0 ||
            graph->panel_roots[k] >= grid->npcol)
            ABORT("SymLDL solve graph root or supernode metadata is invalid.");
    }

    for (int_t slot = 0; slot < graph->panel_count; ++slot) {
        const dSymLDLPanelDesc *panel = &graph->panels[slot];
        int_t gid = graph->panel_gids[slot];
        if (gid < 0 || gid >= graph->nsupers || panel->gid != gid ||
            graph->panel_local_index[gid] != slot || panel->width <= 0 ||
            panel->width != graph->xsup[gid + 1] - graph->xsup[gid] ||
            panel->block_begin < 0 || panel->block_count < 0 ||
            panel->block_begin > graph->block_count ||
            panel->block_count > graph->block_count - panel->block_begin)
            ABORT("SymLDL solve graph panel metadata is invalid.");
        if (!panel->active) {
            if (panel->block_count != 0 || panel->has_diag)
                ABORT("SymLDL solve graph inactive panel has retained blocks.");
            continue;
        }
        if (panel->nsupr <= 0 || panel->nsupr > INT_MAX ||
            panel->value_count < 0 || panel->values == NULL ||
            (size_t) panel->nsupr > SIZE_MAX / (size_t) panel->width ||
            (size_t) panel->value_count !=
                (size_t) panel->nsupr * (size_t) panel->width)
            ABORT("SymLDL solve graph active panel has no retained values.");
        if (panel->has_diag &&
            (panel->diag_luptr < 0 ||
             panel->diag_luptr > panel->nsupr - panel->width))
            ABORT("SymLDL solve graph diagonal block is invalid.");

        int_t previous_target = -1;
        for (int_t local = 0; local < panel->block_count; ++local) {
            int_t edge = panel->block_begin + local;
            const dSymLDLBlockDesc *block = &graph->blocks[edge];
            if (block->panel_id != slot || block->target_gid <= gid ||
                block->target_gid >= graph->nsupers ||
                block->target_gid <= previous_target || block->nbrow <= 0 ||
                block->row_begin < 0 ||
                block->row_begin > graph->factor_row_count ||
                block->nbrow > graph->factor_row_count - block->row_begin ||
                block->luptr < 0 || block->luptr > panel->nsupr ||
                block->nbrow > panel->nsupr - block->luptr)
                ABORT("SymLDL solve graph retained block is invalid.");
            previous_target = block->target_gid;
            for (int_t row = 0; row < block->nbrow; ++row) {
                int_t grow = graph->rows[block->row_begin + row];
                if (grow < graph->xsup[block->target_gid] ||
                    grow >= graph->xsup[block->target_gid + 1])
                    ABORT("SymLDL solve graph retained row is invalid.");
            }
        }
    }

    if (graph->ilsum[0] != 0)
        ABORT("SymLDL solve graph local solution offsets are invalid.");
    for (int_t slot = 0; slot < graph->row_count; ++slot) {
        int_t gid = graph->row_gids[slot];
        if (gid < 0 || gid >= graph->nsupers ||
            graph->row_local_index[gid] != slot ||
            (slot > 0 && graph->row_gids[slot - 1] >= gid) ||
            graph->ilsum[slot] < 0 ||
            graph->ilsum[slot + 1] < graph->ilsum[slot] ||
            graph->ilsum[slot + 1] - graph->ilsum[slot] !=
                graph->xsup[gid + 1] - graph->xsup[gid])
            ABORT("SymLDL solve graph local row order is invalid.");
    }
}

static void
symldl_graph_build_reverse(dSymLDLSolveGraph *graph)
{
    dSymLDLReverseEntry *entries = (dSymLDLReverseEntry *)
        symldl_graph_alloc((size_t) graph->block_count, sizeof(*entries),
                           "Malloc fails for SymLDL reverse edges.");
    graph->source_edge_offsets = (int_t *) symldl_graph_alloc(
        (size_t) graph->row_count + 1, sizeof(int_t),
        "Malloc fails for SymLDL reverse offsets.");
    graph->source_edge_ids = (int_t *) symldl_graph_alloc(
        (size_t) graph->block_count, sizeof(int_t),
        "Malloc fails for SymLDL reverse edge ids.");
    memset(graph->source_edge_offsets, 0,
           ((size_t) graph->row_count + 1) * sizeof(int_t));

    for (int_t edge = 0; edge < graph->block_count; ++edge) {
        int_t source = graph->blocks[edge].target_gid;
        int_t slot = graph->row_local_index[source];
        if (slot < 0 || slot >= graph->row_count)
            ABORT("SymLDL solve graph reverse edge has no local source row.");
        entries[edge].slot = slot;
        entries[edge].panel_gid =
            graph->panels[graph->blocks[edge].panel_id].gid;
        entries[edge].edge = edge;
        ++graph->source_edge_offsets[slot + 1];
    }
    for (int_t slot = 0; slot < graph->row_count; ++slot)
        graph->source_edge_offsets[slot + 1] +=
            graph->source_edge_offsets[slot];
    if (graph->block_count > 1)
        qsort(entries, (size_t) graph->block_count, sizeof(*entries),
              symldl_graph_reverse_cmp);
    for (int_t pos = 0; pos < graph->block_count; ++pos)
        graph->source_edge_ids[pos] = entries[pos].edge;
    if (entries != NULL)
        SUPERLU_FREE(entries);
}

static void
symldl_graph_build_forward_bcast(dSymLDLSolveGraph *graph,
                                  gridinfo3d_t *grid3d)
{
    gridinfo_t *grid = &grid3d->grid2d;
    int myrow = MYROW(grid->iam, grid);
    int mycol = MYCOL(grid->iam, grid);
    int Pr = grid->nprow;
    int_t missing = graph->nsupers;
    int count = (int) graph->panel_count;
    size_t summary_count = symldl_graph_checked_product(
        (size_t) Pr, (size_t) count,
        "SymLDL forward broadcast summary count overflows.");
    int_t *local = (int_t *) symldl_graph_alloc(
        (size_t) SUPERLU_MAX(1, count), sizeof(int_t),
        "Malloc fails for SymLDL forward broadcast summaries.");
    int_t *all = (int_t *) symldl_graph_alloc(
        SUPERLU_MAX((size_t) 1, summary_count), sizeof(int_t),
        "Malloc fails for SymLDL forward broadcast summaries.");
    dSymLDLRankOrder *ordered = (dSymLDLRankOrder *) symldl_graph_alloc(
        (size_t) SUPERLU_MAX(1, Pr), sizeof(*ordered),
        "Malloc fails for SymLDL forward broadcast ordering.");
    int *ranks = (int *) symldl_graph_alloc(
        (size_t) SUPERLU_MAX(1, Pr), sizeof(int),
        "Malloc fails for SymLDL forward broadcast ranks.");

    for (int_t slot = 0; slot < graph->panel_count; ++slot) {
        const dSymLDLPanelDesc *panel = &graph->panels[slot];
        local[slot] = missing;
        for (int_t block = 0; block < panel->block_count; ++block)
            local[slot] = SUPERLU_MIN(
                local[slot],
                graph->blocks[panel->block_begin + block].target_gid);
        if (panel->active && myrow == graph->diag_roots[panel->gid])
            local[slot] = SUPERLU_MIN(local[slot], panel->gid);
    }
    MPI_Allgather(local, count, mpi_int_t, all, count, mpi_int_t,
                  grid->cscp.comm);

    for (int_t slot = 0; slot < graph->panel_count; ++slot) {
        int_t gid = graph->panel_gids[slot];
        int root = graph->diag_roots[gid];
        int ordered_count = 0;
        int root_active = all[(size_t) root * count + slot] != missing;
        for (int pr = 0; pr < Pr; ++pr) {
            int_t first = all[(size_t) pr * count + slot];
            if (pr != root && first != missing) {
                ordered[ordered_count].order = first;
                ordered[ordered_count++].rank = pr;
            }
        }
        if (!root_active && ordered_count > 0)
            ABORT("SymLDL forward broadcast root is inactive.");
        qsort(ordered, (size_t) ordered_count, sizeof(*ordered),
              symldl_graph_rank_asc);
        int rank_count = 0;
        if (root_active) {
            ranks[rank_count++] = PNUM(root, mycol, grid);
            for (int i = 0; i < ordered_count; ++i)
                ranks[rank_count++] = PNUM(ordered[i].rank, mycol, grid);
        }
        symldl_graph_tree_create(&graph->forward_bcast[slot], ranks,
                                 rank_count, grid->iam);
    }
    SUPERLU_FREE(ranks);
    SUPERLU_FREE(ordered);
    SUPERLU_FREE(all);
    SUPERLU_FREE(local);
}

static void
symldl_graph_build_forward_reduce(dSymLDLSolveGraph *graph,
                                   gridinfo3d_t *grid3d)
{
    gridinfo_t *grid = &grid3d->grid2d;
    int myrow = MYROW(grid->iam, grid);
    int Pc = grid->npcol;
    int_t missing = -1;
    int count = (int) graph->row_count;
    size_t summary_count = symldl_graph_checked_product(
        (size_t) Pc, (size_t) count,
        "SymLDL forward reduction summary count overflows.");
    int_t *local = (int_t *) symldl_graph_alloc(
        (size_t) SUPERLU_MAX(1, count), sizeof(int_t),
        "Malloc fails for SymLDL forward reduction summaries.");
    int_t *all = (int_t *) symldl_graph_alloc(
        SUPERLU_MAX((size_t) 1, summary_count), sizeof(int_t),
        "Malloc fails for SymLDL forward reduction summaries.");
    dSymLDLRankOrder *ordered = (dSymLDLRankOrder *) symldl_graph_alloc(
        (size_t) SUPERLU_MAX(1, Pc), sizeof(*ordered),
        "Malloc fails for SymLDL forward reduction ordering.");
    int *ranks = (int *) symldl_graph_alloc(
        (size_t) SUPERLU_MAX(1, Pc), sizeof(int),
        "Malloc fails for SymLDL forward reduction ranks.");

    for (int_t slot = 0; slot < graph->row_count; ++slot) {
        local[slot] = missing;
        graph->forward_local_dependencies[slot] = 0;
    }
    for (int_t edge = 0; edge < graph->block_count; ++edge) {
        const dSymLDLBlockDesc *block = &graph->blocks[edge];
        int_t slot = graph->row_local_index[block->target_gid];
        int_t source = graph->panels[block->panel_id].gid;
        local[slot] = SUPERLU_MAX(local[slot], source);
        ++graph->forward_local_dependencies[slot];
    }
    MPI_Allgather(local, count, mpi_int_t, all, count, mpi_int_t,
                  grid->rscp.comm);

    for (int_t slot = 0; slot < graph->row_count; ++slot) {
        int_t gid = graph->row_gids[slot];
        int root = graph->panel_roots[gid];
        int total = 0;
        int ordered_count = 0;
        for (int pc = 0; pc < Pc; ++pc)
            total += all[(size_t) pc * count + slot] != missing;
        if (total == 0) {
            symldl_graph_tree_null(&graph->forward_reduce[slot]);
            continue;
        }
        for (int pc = 0; pc < Pc; ++pc) {
            int_t last = all[(size_t) pc * count + slot];
            if (pc == root)
                last = SUPERLU_MAX(last, gid);
            if (pc != root && last != missing) {
                ordered[ordered_count].order = last;
                ordered[ordered_count++].rank = pc;
            }
        }
        qsort(ordered, (size_t) ordered_count, sizeof(*ordered),
              symldl_graph_rank_desc);
        int rank_count = 0;
        ranks[rank_count++] = PNUM(myrow, root, grid);
        for (int i = 0; i < ordered_count; ++i)
            ranks[rank_count++] = PNUM(myrow, ordered[i].rank, grid);
        symldl_graph_tree_create(&graph->forward_reduce[slot], ranks,
                                 rank_count, grid->iam);
    }
    SUPERLU_FREE(ranks);
    SUPERLU_FREE(ordered);
    SUPERLU_FREE(all);
    SUPERLU_FREE(local);
}

static void
symldl_graph_build_backward_bcast(dSymLDLSolveGraph *graph,
                                   gridinfo3d_t *grid3d)
{
    gridinfo_t *grid = &grid3d->grid2d;
    int myrow = MYROW(grid->iam, grid);
    int mycol = MYCOL(grid->iam, grid);
    int Pc = grid->npcol;
    int_t missing = -1;
    int count = (int) graph->row_count;
    size_t summary_count = symldl_graph_checked_product(
        (size_t) Pc, (size_t) count,
        "SymLDL backward broadcast summary count overflows.");
    int_t *local = (int_t *) symldl_graph_alloc(
        (size_t) SUPERLU_MAX(1, count), sizeof(int_t),
        "Malloc fails for SymLDL backward broadcast summaries.");
    int_t *all = (int_t *) symldl_graph_alloc(
        SUPERLU_MAX((size_t) 1, summary_count), sizeof(int_t),
        "Malloc fails for SymLDL backward broadcast summaries.");
    dSymLDLRankOrder *ordered = (dSymLDLRankOrder *) symldl_graph_alloc(
        (size_t) SUPERLU_MAX(1, Pc), sizeof(*ordered),
        "Malloc fails for SymLDL backward broadcast ordering.");
    int *ranks = (int *) symldl_graph_alloc(
        (size_t) SUPERLU_MAX(1, Pc), sizeof(int),
        "Malloc fails for SymLDL backward broadcast ranks.");

    for (int_t slot = 0; slot < graph->row_count; ++slot) {
        int_t gid = graph->row_gids[slot];
        local[slot] = missing;
        if ((mycol == graph->panel_roots[gid] &&
             symldl_graph_panel_active(graph, gid)) ||
            graph->source_edge_offsets[slot + 1] >
                graph->source_edge_offsets[slot])
            local[slot] = gid;
    }
    MPI_Allgather(local, count, mpi_int_t, all, count, mpi_int_t,
                  grid->rscp.comm);

    for (int_t slot = 0; slot < graph->row_count; ++slot) {
        int_t gid = graph->row_gids[slot];
        int root = graph->panel_roots[gid];
        int ordered_count = 0;
        int root_active = all[(size_t) root * count + slot] != missing;
        for (int pc = 0; pc < Pc; ++pc) {
            int_t last = all[(size_t) pc * count + slot];
            if (pc != root && last != missing) {
                ordered[ordered_count].order = last;
                ordered[ordered_count++].rank = pc;
            }
        }
        if (!root_active && ordered_count > 0)
            ABORT("SymLDL backward broadcast root is inactive.");
        qsort(ordered, (size_t) ordered_count, sizeof(*ordered),
              symldl_graph_rank_desc);
        int rank_count = 0;
        if (root_active) {
            ranks[rank_count++] = PNUM(myrow, root, grid);
            for (int i = 0; i < ordered_count; ++i)
                ranks[rank_count++] = PNUM(myrow, ordered[i].rank, grid);
        }
        symldl_graph_tree_create(&graph->backward_bcast[slot], ranks,
                                 rank_count, grid->iam);
    }
    SUPERLU_FREE(ranks);
    SUPERLU_FREE(ordered);
    SUPERLU_FREE(all);
    SUPERLU_FREE(local);
}

static void
symldl_graph_build_backward_reduce(dSymLDLSolveGraph *graph,
                                    gridinfo3d_t *grid3d)
{
    gridinfo_t *grid = &grid3d->grid2d;
    int myrow = MYROW(grid->iam, grid);
    int mycol = MYCOL(grid->iam, grid);
    int Pr = grid->nprow;
    int_t missing = -1;
    int count = (int) graph->panel_count;
    size_t summary_count = symldl_graph_checked_product(
        (size_t) Pr, (size_t) count,
        "SymLDL backward reduction summary count overflows.");
    int_t *local = (int_t *) symldl_graph_alloc(
        (size_t) SUPERLU_MAX(1, count), sizeof(int_t),
        "Malloc fails for SymLDL backward reduction summaries.");
    int_t *all = (int_t *) symldl_graph_alloc(
        SUPERLU_MAX((size_t) 1, summary_count), sizeof(int_t),
        "Malloc fails for SymLDL backward reduction summaries.");
    dSymLDLRankOrder *ordered = (dSymLDLRankOrder *) symldl_graph_alloc(
        (size_t) SUPERLU_MAX(1, Pr), sizeof(*ordered),
        "Malloc fails for SymLDL backward reduction ordering.");
    int *ranks = (int *) symldl_graph_alloc(
        (size_t) SUPERLU_MAX(1, Pr), sizeof(int),
        "Malloc fails for SymLDL backward reduction ranks.");

    for (int_t slot = 0; slot < graph->panel_count; ++slot) {
        const dSymLDLPanelDesc *panel = &graph->panels[slot];
        local[slot] = missing;
        graph->backward_local_dependencies[slot] = (int) panel->block_count;
        if (panel->block_count > 0)
            local[slot] = panel->gid + 1;
        if (panel->active && myrow == graph->diag_roots[panel->gid])
            local[slot] = SUPERLU_MAX(local[slot], panel->gid);
    }
    MPI_Allgather(local, count, mpi_int_t, all, count, mpi_int_t,
                  grid->cscp.comm);

    for (int_t slot = 0; slot < graph->panel_count; ++slot) {
        int_t gid = graph->panel_gids[slot];
        int root = graph->diag_roots[gid];
        int ordered_count = 0;
        int root_active = all[(size_t) root * count + slot] != missing;
        for (int pr = 0; pr < Pr; ++pr) {
            int_t last = all[(size_t) pr * count + slot];
            if (pr != root && last != missing) {
                ordered[ordered_count].order = last;
                ordered[ordered_count++].rank = pr;
            }
        }
        if (!root_active && ordered_count > 0)
            ABORT("SymLDL backward reduction root is inactive.");
        qsort(ordered, (size_t) ordered_count, sizeof(*ordered),
              symldl_graph_rank_desc);
        int rank_count = 0;
        if (root_active) {
            ranks[rank_count++] = PNUM(root, mycol, grid);
            for (int i = 0; i < ordered_count; ++i)
                ranks[rank_count++] = PNUM(ordered[i].rank, mycol, grid);
        }
        symldl_graph_tree_create(&graph->backward_reduce[slot], ranks,
                                 rank_count, grid->iam);
    }
    SUPERLU_FREE(ranks);
    SUPERLU_FREE(ordered);
    SUPERLU_FREE(all);
    SUPERLU_FREE(local);
}

static unsigned long long
symldl_graph_z_hash(unsigned long long hash, int_t gid, int_t width)
{
    const unsigned long long prime = 1099511628211ULL;
    hash ^= (unsigned long long) gid;
    hash *= prime;
    hash ^= (unsigned long long) width;
    hash *= prime;
    return hash;
}

static void
symldl_graph_build_z_plan(dSymLDLSolveGraph *graph,
                          dtrf3Dpartition_t *partition,
                          gridinfo3d_t *grid3d)
{
    gridinfo_t *grid = &grid3d->grid2d;
    int_t *seen;
    int_t total_gids = 0;
    const int_t int_t_max = sizeof(int_t) == sizeof(int64_t)
                                ? (int_t) INT64_MAX : (int_t) INT32_MAX;
    int max_level;
    int myrow = MYROW(grid->iam, grid);
    int mycol = MYCOL(grid->iam, grid);
    unsigned long long *local_summary;
    unsigned long long *all_summary;

    graph->z_rank = grid3d->zscp.Iam;
    graph->z_size = grid3d->zscp.Np;
    if (graph->z_size <= 0)
        ABORT("SymLDL sparse Z plan has an invalid process count.");
    if (graph->z_size == 1)
        return;
    if ((graph->z_size & (graph->z_size - 1)) != 0 ||
        graph->z_size > INT_MAX / 2 + 1)
        ABORT("SymLDL sparse Z plan requires a power-of-two process count.");
    if (partition->myTreeIdxs == NULL || partition->myZeroTrIdxs == NULL ||
        partition->sForests == NULL)
        ABORT("SymLDL sparse Z plan is missing forest metadata.");

    max_level = (int) log2i(graph->z_size) + 1;
    const int forest_count = 2 * graph->z_size - 1;
    graph->z_stage_count = max_level - 1;
    graph->z_stages = (dSymLDLZStage *) symldl_graph_alloc(
        (size_t) graph->z_stage_count, sizeof(*graph->z_stages),
        "Malloc fails for SymLDL sparse Z stages.");
    memset(graph->z_stages, 0,
           (size_t) graph->z_stage_count * sizeof(*graph->z_stages));
    seen = (int_t *) symldl_graph_alloc(
        (size_t) graph->nsupers, sizeof(*seen),
        "Malloc fails for SymLDL sparse Z validation.");
    for (int_t k = 0; k < graph->nsupers; ++k)
        seen[k] = -1;

    for (int level = 1; level < max_level; ++level) {
        dSymLDLZStage *stage = &graph->z_stages[level - 1];
        int_t current;
        int_t count = 0;
        int_t values = 0;
        stage->gid_begin = total_gids;
        if (partition->myZeroTrIdxs[level - 1])
            continue;
        stage->active = 1;
        if ((graph->z_rank % (1 << level)) == 0) {
            stage->sender_z = graph->z_rank + (1 << (level - 1));
            stage->receiver_z = graph->z_rank;
        } else {
            stage->sender_z = graph->z_rank;
            stage->receiver_z = graph->z_rank - (1 << (level - 1));
        }
        if (stage->sender_z < 0 || stage->sender_z >= graph->z_size ||
            stage->receiver_z < 0 || stage->receiver_z >= graph->z_size)
            ABORT("SymLDL sparse Z stage has an invalid peer.");

        current = partition->myTreeIdxs[level];
        for (int ancestor = level; ancestor < max_level; ++ancestor) {
            sForest_t *forest;
            if (current < 0 || current >= forest_count)
                ABORT("SymLDL sparse Z stage has an invalid forest.");
            forest = partition->sForests[current];
            if (forest != NULL) {
                if (forest->nNodes < 0 ||
                    (forest->nNodes > 0 && forest->nodeList == NULL))
                    ABORT("SymLDL sparse Z forest metadata is invalid.");
                for (int_t pos = 0; pos < forest->nNodes; ++pos) {
                    int_t gid = forest->nodeList[pos];
                    int_t width;
                    if (gid < 0 || gid >= graph->nsupers)
                        ABORT("SymLDL sparse Z forest contains an invalid node.");
                    if (graph->diag_roots[gid] != myrow ||
                        graph->panel_roots[gid] != mycol)
                        continue;
                    if (seen[gid] == level)
                        ABORT("SymLDL sparse Z stage contains a duplicate node.");
                    seen[gid] = level;
                    if (graph->row_local_index[gid] < 0)
                        ABORT("SymLDL sparse Z node has no local solution row.");
                    width = graph->xsup[gid + 1] - graph->xsup[gid];
                    if (width <= 0 || values > int_t_max - width)
                        ABORT("SymLDL sparse Z stage value count overflows.");
                    values += width;
                    ++count;
                }
            }
            current = (current + 1) / 2 - 1;
        }
        if (count > int_t_max - total_gids)
            ABORT("SymLDL sparse Z gid count overflows.");
        stage->gid_count = count;
        stage->value_count = values;
        total_gids += count;
    }

    graph->z_gid_count = total_gids;
    graph->z_stage_gids = (int_t *) symldl_graph_alloc(
        (size_t) total_gids, sizeof(*graph->z_stage_gids),
        "Malloc fails for SymLDL sparse Z gids.");
    for (int_t k = 0; k < graph->nsupers; ++k)
        seen[k] = -1;
    for (int level = 1; level < max_level; ++level) {
        dSymLDLZStage *stage = &graph->z_stages[level - 1];
        int_t current;
        int_t write = stage->gid_begin;
        if (!stage->active)
            continue;
        current = partition->myTreeIdxs[level];
        for (int ancestor = level; ancestor < max_level; ++ancestor) {
            sForest_t *forest;
            if (current < 0 || current >= forest_count)
                ABORT("SymLDL sparse Z stage has an invalid forest.");
            forest = partition->sForests[current];
            if (forest != NULL) {
                if (forest->nNodes < 0 ||
                    (forest->nNodes > 0 && forest->nodeList == NULL))
                    ABORT("SymLDL sparse Z forest metadata is invalid.");
                for (int_t pos = 0; pos < forest->nNodes; ++pos) {
                    int_t gid = forest->nodeList[pos];
                    if (gid < 0 || gid >= graph->nsupers)
                        ABORT("SymLDL sparse Z forest contains an invalid node.");
                    if (graph->diag_roots[gid] != myrow ||
                        graph->panel_roots[gid] != mycol)
                        continue;
                    if (seen[gid] == level)
                        ABORT("SymLDL sparse Z stage contains a duplicate node.");
                    seen[gid] = level;
                    if (write >= stage->gid_begin + stage->gid_count)
                        ABORT("SymLDL sparse Z stage fill overflows.");
                    graph->z_stage_gids[write++] = gid;
                }
            }
            current = (current + 1) / 2 - 1;
        }
        if (write != stage->gid_begin + stage->gid_count)
            ABORT("SymLDL sparse Z stage fill is inconsistent.");
    }
    SUPERLU_FREE(seen);

    local_summary = (unsigned long long *) symldl_graph_alloc(
        (size_t) 3 * graph->z_stage_count, sizeof(*local_summary),
        "Malloc fails for SymLDL sparse Z summaries.");
    all_summary = (unsigned long long *) symldl_graph_alloc(
        (size_t) 3 * graph->z_stage_count * graph->z_size,
        sizeof(*all_summary), "Malloc fails for SymLDL sparse Z summaries.");
    for (int stage_id = 0; stage_id < graph->z_stage_count; ++stage_id) {
        const dSymLDLZStage *stage = &graph->z_stages[stage_id];
        unsigned long long hash = 1469598103934665603ULL;
        for (int_t pos = 0; pos < stage->gid_count; ++pos) {
            int_t gid = graph->z_stage_gids[stage->gid_begin + pos];
            hash = symldl_graph_z_hash(
                hash, gid, graph->xsup[gid + 1] - graph->xsup[gid]);
        }
        local_summary[3 * stage_id] = stage->active;
        local_summary[3 * stage_id + 1] = (unsigned long long) stage->gid_count;
        local_summary[3 * stage_id + 2] = hash;
    }
    MPI_Allgather(local_summary, 3 * graph->z_stage_count,
                  MPI_UNSIGNED_LONG_LONG, all_summary,
                  3 * graph->z_stage_count, MPI_UNSIGNED_LONG_LONG,
                  grid3d->zscp.comm);
    for (int stage_id = 0; stage_id < graph->z_stage_count; ++stage_id) {
        const dSymLDLZStage *stage = &graph->z_stages[stage_id];
        int peer;
        size_t peer_base;
        if (!stage->active)
            continue;
        peer = graph->z_rank == stage->sender_z
                   ? stage->receiver_z : stage->sender_z;
        peer_base = (size_t) peer * 3 * graph->z_stage_count +
                    (size_t) 3 * stage_id;
        if (all_summary[peer_base] != 1 ||
            all_summary[peer_base + 1] != local_summary[3 * stage_id + 1] ||
            all_summary[peer_base + 2] != local_summary[3 * stage_id + 2])
            ABORT("SymLDL sparse Z peer plans are inconsistent.");
    }
    SUPERLU_FREE(all_summary);
    SUPERLU_FREE(local_summary);
}

dSymLDLSolveGraph *
dSymLDLSolveGraphCreate(
    int_t n, int_t nsupers,
    int_t panel_count, const dSymLDLPanelDesc *panels,
    int_t block_count, const dSymLDLBlockDesc *blocks,
    int_t factor_row_count, const int_t *rows,
    const int_t *xsup, const int_t *ilsum,
    dtrf3Dpartition_t *partition, gridinfo3d_t *grid3d)
{
    dSymLDLSolveGraph *graph;
    if (n <= 0 || nsupers <= 0 || panel_count < 0 || block_count < 0 ||
        factor_row_count < 0 || (panel_count > 0 && panels == NULL) ||
        (block_count > 0 && blocks == NULL) ||
        (factor_row_count > 0 && rows == NULL) || xsup == NULL ||
        ilsum == NULL || partition == NULL || grid3d == NULL ||
        (panel_count > 0 && partition->symV2LocalPanelGids == NULL) ||
        (partition->symV2LocalRowCount > 0 &&
         partition->symV2LocalRowGids == NULL) ||
        partition->symV2PanelLocalIndex == NULL ||
        partition->symV2RowLocalIndex == NULL ||
        partition->symV2DiagRoot == NULL ||
        partition->symV2PanelRoot == NULL || panel_count > INT_MAX ||
        block_count > INT_MAX ||
        partition->symV2LocalRowCount > INT_MAX)
        return NULL;

    graph = (dSymLDLSolveGraph *)
        symldl_graph_alloc(1, sizeof(*graph),
                           "Malloc fails for SymLDL solve graph.");
    memset(graph, 0, sizeof(*graph));
    graph->n = n;
    graph->nsupers = nsupers;
    graph->panel_count = panel_count;
    graph->block_count = block_count;
    graph->factor_row_count = factor_row_count;
    graph->row_count = partition->symV2LocalRowCount;

    graph->panels = (dSymLDLPanelDesc *) symldl_graph_copy(
        panels, (size_t) panel_count, sizeof(*panels),
        "Malloc fails for SymLDL solve panels.");
    graph->blocks = (dSymLDLBlockDesc *) symldl_graph_copy(
        blocks, (size_t) block_count, sizeof(*blocks),
        "Malloc fails for SymLDL solve blocks.");
    graph->rows = (int_t *) symldl_graph_copy(
        rows, (size_t) factor_row_count, sizeof(*rows),
        "Malloc fails for SymLDL solve rows.");
    graph->panel_gids = (int_t *) symldl_graph_copy(
        partition->symV2LocalPanelGids, (size_t) panel_count,
        sizeof(int_t), "Malloc fails for SymLDL local panel gids.");
    graph->row_gids = (int_t *) symldl_graph_copy(
        partition->symV2LocalRowGids, (size_t) graph->row_count,
        sizeof(int_t), "Malloc fails for SymLDL local row gids.");
    graph->panel_local_index = (int_t *) symldl_graph_copy(
        partition->symV2PanelLocalIndex, (size_t) nsupers, sizeof(int_t),
        "Malloc fails for SymLDL panel lookup.");
    graph->row_local_index = (int_t *) symldl_graph_copy(
        partition->symV2RowLocalIndex, (size_t) nsupers, sizeof(int_t),
        "Malloc fails for SymLDL row lookup.");
    graph->diag_roots = (int *) symldl_graph_copy(
        partition->symV2DiagRoot, (size_t) nsupers, sizeof(int),
        "Malloc fails for SymLDL diagonal roots.");
    graph->panel_roots = (int *) symldl_graph_copy(
        partition->symV2PanelRoot, (size_t) nsupers, sizeof(int),
        "Malloc fails for SymLDL panel roots.");
    graph->xsup = (int_t *) symldl_graph_copy(
        xsup, (size_t) nsupers + 1, sizeof(int_t),
        "Malloc fails for SymLDL supernode offsets.");
    graph->ilsum = (int_t *) symldl_graph_copy(
        ilsum, (size_t) graph->row_count + 1, sizeof(int_t),
        "Malloc fails for SymLDL local solution offsets.");
    graph->target_ilsum = (int_t *) symldl_graph_alloc(
        (size_t) panel_count + 1, sizeof(int_t),
        "Malloc fails for SymLDL backward target offsets.");
    graph->forward_bcast = (dSymLDLTreeNode *) symldl_graph_alloc(
        (size_t) panel_count, sizeof(dSymLDLTreeNode),
        "Malloc fails for SymLDL forward broadcast trees.");
    graph->forward_reduce = (dSymLDLTreeNode *) symldl_graph_alloc(
        (size_t) graph->row_count, sizeof(dSymLDLTreeNode),
        "Malloc fails for SymLDL forward reduction trees.");
    graph->backward_bcast = (dSymLDLTreeNode *) symldl_graph_alloc(
        (size_t) graph->row_count, sizeof(dSymLDLTreeNode),
        "Malloc fails for SymLDL backward broadcast trees.");
    graph->backward_reduce = (dSymLDLTreeNode *) symldl_graph_alloc(
        (size_t) panel_count, sizeof(dSymLDLTreeNode),
        "Malloc fails for SymLDL backward reduction trees.");
    graph->forward_local_dependencies = (int *) symldl_graph_alloc(
        (size_t) graph->row_count, sizeof(int),
        "Malloc fails for SymLDL forward dependencies.");
    graph->backward_local_dependencies = (int *) symldl_graph_alloc(
        (size_t) panel_count, sizeof(int),
        "Malloc fails for SymLDL backward dependencies.");

    symldl_graph_validate(graph, partition, grid3d);
    graph->target_ilsum[0] = 0;
    for (int_t slot = 0; slot < panel_count; ++slot) {
        const int_t int_t_max = sizeof(int_t) == sizeof(int64_t)
                                    ? (int_t) INT64_MAX : (int_t) INT32_MAX;
        int_t gid = graph->panel_gids[slot];
        int_t width = graph->xsup[gid + 1] - graph->xsup[gid];
        if (graph->target_ilsum[slot] > int_t_max - width)
            ABORT("SymLDL backward target offsets overflow int_t.");
        graph->target_ilsum[slot + 1] =
            graph->target_ilsum[slot] + width;
    }
    graph->backward_lsum_rows = graph->target_ilsum[panel_count];
    for (int_t k = 0; k < nsupers; ++k)
        graph->maxsup = SUPERLU_MAX(
            graph->maxsup, (int) (graph->xsup[k + 1] - graph->xsup[k]));

    symldl_graph_build_reverse(graph);
    symldl_graph_build_forward_bcast(graph, grid3d);
    symldl_graph_build_forward_reduce(graph, grid3d);
    symldl_graph_build_backward_bcast(graph, grid3d);
    symldl_graph_build_backward_reduce(graph, grid3d);
    symldl_graph_build_z_plan(graph, partition, grid3d);
    return graph;
}

void
dSymLDLSolveGraphDestroy(dSymLDLSolveGraph *graph)
{
    if (graph == NULL)
        return;
    if (graph->z_stage_gids) SUPERLU_FREE(graph->z_stage_gids);
    if (graph->z_stages) SUPERLU_FREE(graph->z_stages);
    if (graph->backward_local_dependencies) SUPERLU_FREE(graph->backward_local_dependencies);
    if (graph->forward_local_dependencies) SUPERLU_FREE(graph->forward_local_dependencies);
    if (graph->backward_reduce) SUPERLU_FREE(graph->backward_reduce);
    if (graph->backward_bcast) SUPERLU_FREE(graph->backward_bcast);
    if (graph->forward_reduce) SUPERLU_FREE(graph->forward_reduce);
    if (graph->forward_bcast) SUPERLU_FREE(graph->forward_bcast);
    if (graph->target_ilsum) SUPERLU_FREE(graph->target_ilsum);
    if (graph->panel_roots) SUPERLU_FREE(graph->panel_roots);
    if (graph->diag_roots) SUPERLU_FREE(graph->diag_roots);
    if (graph->ilsum) SUPERLU_FREE(graph->ilsum);
    if (graph->xsup) SUPERLU_FREE(graph->xsup);
    if (graph->source_edge_ids) SUPERLU_FREE(graph->source_edge_ids);
    if (graph->source_edge_offsets) SUPERLU_FREE(graph->source_edge_offsets);
    if (graph->row_local_index) SUPERLU_FREE(graph->row_local_index);
    if (graph->panel_local_index) SUPERLU_FREE(graph->panel_local_index);
    if (graph->row_gids) SUPERLU_FREE(graph->row_gids);
    if (graph->panel_gids) SUPERLU_FREE(graph->panel_gids);
    if (graph->rows) SUPERLU_FREE(graph->rows);
    if (graph->blocks) SUPERLU_FREE(graph->blocks);
    if (graph->panels) SUPERLU_FREE(graph->panels);
    SUPERLU_FREE(graph);
}
