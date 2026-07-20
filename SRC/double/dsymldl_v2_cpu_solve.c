/*! @file
 * \brief Event-driven CPU runtime for the true-symmetric retained-L solve.
 */

#include "dsymldl_v2_cpu_solve.h"
#include "dsymldl_v2_cpu_kernels.h"

#include <limits.h>
#include <stdint.h>
#include <sched.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

enum {
    SYMLDL_CPU_PHASE_FORWARD = 1,
    SYMLDL_CPU_PHASE_BACKWARD = 2
};

enum {
    SYMLDL_CPU_MESSAGE_X = 1,
    SYMLDL_CPU_MESSAGE_PARTIAL = 2
};

enum {
    SYMLDL_CPU_FORWARD_TAG = 19131,
    SYMLDL_CPU_BACKWARD_TAG = 19132,
    SYMLDL_CPU_Z_REDUCE_TAG = 19133,
    SYMLDL_CPU_Z_BCAST_TAG = 19134
};

typedef struct {
    int kind;
    int generation;
    int count;
    int reserved;
    int_t gid;
    double payload_alignment;
} dSymLDLCPUMessageHeader;

struct dSymLDLCPUSolveHandle {
    const dSymLDLSolveGraph *graph;
    dtrf3Dpartition_t *partition;
    MPI_Comm layer_comm;
    MPI_Comm z_comm;
    int rank;
    int size;
    int global_rank;
    int z_rank;
    int z_size;
    int myrow;
    int mycol;
    int nrhs_capacity;
    int max_threads;
    int receive_depth;
    int send_capacity;
    int generation;

    size_t *panel_message_offsets;
    size_t *row_message_offsets;
    size_t panel_message_bytes;
    size_t row_message_bytes;
    unsigned char *x_messages;
    unsigned char *sum_messages;
    unsigned char *receive_messages;
    size_t receive_stride;

    MPI_Request *receive_requests;
    MPI_Request *send_requests;
    int *completed_receive_indices;
    MPI_Status *completed_receive_statuses;

    int *dependencies;
    unsigned char *target_locks;
    unsigned char *target_queued;
    unsigned char *target_terminal;
    unsigned char *received_child_mask;
    unsigned char *source_published;

    int_t *task_ids;
    int_t *completion_slots;
    unsigned char *completion_ready;
    double *worker_scratch;
    double *worker_compute_times;

    int_t rhs_count;
    int_t rhs_value_count;
    int_t *rhs_gids;
    int_t *rhs_offsets;
    double *rhs_values;
    int_t z_value_capacity;
    double *z_send_values;
    double *z_recv_values;

    volatile int stop_workers;
    volatile int_t tasks_published;
    volatile int_t tasks_claimed;
    volatile int_t tasks_completed;
    volatile int_t completions_reserved;
    int_t completions_consumed;

    int phase;
    int nrhs;
    double *x;
    int_t x_count;
    int target_count;
    int source_count;
    int terminal_expected;
    int terminal_completed;
    int receive_expected;
    int receive_posted;
    int receive_completed;
    int send_count;

    double setup_time;
    double forward_time;
    double diagonal_time;
    double backward_time;
    double mpi_progress_time;
    double numeric_compute_time;
    double rhs_distribution_time;
    double sparse_reduce_time;
    double sparse_broadcast_time;
    double panel_replication_time;
    dSymLDLCPUSolveCommStats comm_stats;
};

static size_t
symldl_cpu_checked_product(size_t a, size_t b, const char *message)
{
    if (a != 0 && b > SIZE_MAX / a)
        ABORT(message);
    return a * b;
}

static size_t
symldl_cpu_checked_sum(size_t a, size_t b, const char *message)
{
    if (b > SIZE_MAX - a)
        ABORT(message);
    return a + b;
}

static void *
symldl_cpu_alloc(size_t count, size_t size, const char *message)
{
    void *memory;
    if (count == 0)
        return NULL;
    memory = SUPERLU_MALLOC(symldl_cpu_checked_product(count, size, message));
    if (memory == NULL)
        ABORT(message);
    return memory;
}

static size_t
symldl_cpu_align(size_t value, size_t alignment, const char *message)
{
    size_t remainder = value % alignment;
    return remainder == 0 ? value :
        symldl_cpu_checked_sum(value, alignment - remainder, message);
}

static int
symldl_cpu_count_to_int(int_t count, const char *message)
{
    int value = (int) count;
    if (count < 0 || (int_t) value != count)
        ABORT(message);
    return value;
}

static int
symldl_cpu_size_to_int(size_t count, const char *message)
{
    if (count > INT_MAX)
        ABORT(message);
    return (int) count;
}

static int
symldl_cpu_product_to_int(int_t count, int multiplier,
                          const char *message)
{
    if (count < 0 || multiplier < 0)
        ABORT(message);
    return symldl_cpu_size_to_int(
        symldl_cpu_checked_product((size_t) count, (size_t) multiplier,
                                   message),
        message);
}

static int_t
symldl_cpu_size_to_int_t(size_t value, const char *message)
{
    int_t converted = (int_t) value;
    if (converted < 0 || (size_t) converted != value)
        ABORT(message);
    return converted;
}

static int_t
symldl_cpu_product_to_int_t(int_t count, int multiplier,
                            const char *message)
{
    if (count < 0 || multiplier < 0)
        ABORT(message);
    return symldl_cpu_size_to_int_t(
        symldl_cpu_checked_product((size_t) count, (size_t) multiplier,
                                   message),
        message);
}

static size_t *
symldl_cpu_build_message_offsets(const dSymLDLSolveGraph *graph,
                                 const int_t *gids, int_t count,
                                 int nrhs_capacity, size_t *total)
{
    size_t *offsets = (size_t *) symldl_cpu_alloc(
        (size_t) count + 1, sizeof(*offsets),
        "Malloc fails for SymLDL CPU message offsets.");
    size_t cursor = 0;
    for (int_t slot = 0; slot < count; ++slot) {
        int_t gid = gids[slot];
        int_t width = graph->xsup[gid + 1] - graph->xsup[gid];
        size_t values = symldl_cpu_checked_product(
            (size_t) width, (size_t) nrhs_capacity,
            "SymLDL CPU message value count overflows.");
        offsets[slot] = cursor;
        cursor = symldl_cpu_checked_sum(
            cursor, sizeof(dSymLDLCPUMessageHeader),
            "SymLDL CPU message workspace overflows.");
        cursor = symldl_cpu_checked_sum(
            cursor, symldl_cpu_checked_product(
                        values, sizeof(double),
                        "SymLDL CPU message workspace overflows."),
            "SymLDL CPU message workspace overflows.");
        cursor = symldl_cpu_align(
            cursor, 64, "SymLDL CPU message workspace overflows.");
    }
    offsets[count] = cursor;
    *total = cursor;
    return offsets;
}

static dSymLDLCPUMessageHeader *
symldl_cpu_message(unsigned char *storage, const size_t *offsets, int_t slot)
{
    return (dSymLDLCPUMessageHeader *) (storage + offsets[slot]);
}

static double *
symldl_cpu_payload(dSymLDLCPUMessageHeader *header)
{
    return (double *) (header + 1);
}

static size_t
symldl_cpu_message_bytes(int count)
{
    return sizeof(dSymLDLCPUMessageHeader) +
           (size_t) count * sizeof(double);
}

static int_t
symldl_cpu_x_offset(const dSymLDLCPUSolveHandle *handle, int_t row_slot)
{
    size_t offset_size = symldl_cpu_checked_sum(
        symldl_cpu_checked_product(
            (size_t) handle->graph->ilsum[row_slot],
            (size_t) handle->nrhs,
            "SymLDL CPU solve X offset overflows."),
        symldl_cpu_checked_product(
            (size_t) row_slot + 1, (size_t) XK_H,
            "SymLDL CPU solve X offset overflows."),
        "SymLDL CPU solve X offset overflows.");
    int_t offset = symldl_cpu_size_to_int_t(
        offset_size, "SymLDL CPU solve X offset overflows int_t.");
    int_t gid = handle->graph->row_gids[row_slot];
    int_t count = symldl_cpu_size_to_int_t(
        symldl_cpu_checked_product(
            (size_t) (handle->graph->xsup[gid + 1] -
                      handle->graph->xsup[gid]),
            (size_t) handle->nrhs,
            "SymLDL CPU solve X count overflows."),
        "SymLDL CPU solve X count overflows int_t.");
    if (offset < 0 || count < 0 || offset > handle->x_count ||
        count > handle->x_count - offset)
        ABORT("SymLDL CPU solve X offset is outside the workspace.");
    return offset;
}

static int
symldl_cpu_panel_active(const dSymLDLCPUSolveHandle *handle, int_t gid)
{
    int_t slot;
    if (gid < 0 || gid >= handle->graph->nsupers)
        return 0;
    slot = handle->graph->panel_local_index[gid];
    return slot >= 0 && slot < handle->graph->panel_count &&
           handle->graph->panels[slot].gid == gid &&
           handle->graph->panels[slot].active;
}

static void
symldl_cpu_lock(unsigned char *lock)
{
    int spins = 0;
    while (__atomic_test_and_set(lock, __ATOMIC_ACQUIRE)) {
        if ((++spins & 255) == 0)
            sched_yield();
    }
}

static void
symldl_cpu_unlock(unsigned char *lock)
{
    __atomic_clear(lock, __ATOMIC_RELEASE);
}

static void
symldl_cpu_queue_target(dSymLDLCPUSolveHandle *handle, int_t slot)
{
    unsigned char expected = 0;
    if (!__atomic_compare_exchange_n(
            &handle->target_queued[slot], &expected, 1, 0,
            __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
        ABORT("SymLDL CPU solve target became ready more than once.");
    int_t position = __atomic_fetch_add(
        &handle->completions_reserved, 1, __ATOMIC_ACQ_REL);
    if (position < 0 || position >= handle->target_count)
        ABORT("SymLDL CPU solve completion queue overflows.");
    handle->completion_slots[position] = slot;
    __atomic_store_n(&handle->completion_ready[position], 1,
                     __ATOMIC_RELEASE);
}

static void
symldl_cpu_decrement_target_locked(dSymLDLCPUSolveHandle *handle,
                                   int_t slot)
{
    int ready;
    --handle->dependencies[slot];
    if (handle->dependencies[slot] < 0)
        ABORT("SymLDL CPU solve target dependency became negative.");
    ready = handle->dependencies[slot] == 0;
    symldl_cpu_unlock(&handle->target_locks[slot]);
    if (ready)
        symldl_cpu_queue_target(handle, slot);
}

static void
symldl_cpu_enqueue_task(dSymLDLCPUSolveHandle *handle, int_t edge)
{
    int_t position = __atomic_load_n(
        &handle->tasks_published, __ATOMIC_RELAXED);
    if (position < 0 || position >= handle->graph->block_count)
        ABORT("SymLDL CPU solve task queue overflows.");
    handle->task_ids[position] = edge;
    __atomic_store_n(&handle->tasks_published, position + 1,
                     __ATOMIC_RELEASE);
}

static int
symldl_cpu_claim_task(dSymLDLCPUSolveHandle *handle, int_t *edge)
{
    for (;;) {
        int_t claimed = __atomic_load_n(
            &handle->tasks_claimed, __ATOMIC_RELAXED);
        int_t published = __atomic_load_n(
            &handle->tasks_published, __ATOMIC_ACQUIRE);
        if (claimed >= published)
            return 0;
        if (__atomic_compare_exchange_n(
                &handle->tasks_claimed, &claimed, claimed + 1, 0,
                __ATOMIC_ACQ_REL, __ATOMIC_RELAXED)) {
            *edge = handle->task_ids[claimed];
            return 1;
        }
    }
}

static void
symldl_cpu_merge_forward(dSymLDLCPUSolveHandle *handle, int_t edge,
                         const double *output)
{
    const dSymLDLSolveGraph *graph = handle->graph;
    const dSymLDLBlockDesc *block = &graph->blocks[edge];
    int_t target = block->target_gid;
    int_t target_slot = graph->row_local_index[target];
    int_t first = graph->xsup[target];
    int_t width = graph->xsup[target + 1] - first;
    dSymLDLCPUMessageHeader *sum = symldl_cpu_message(
        handle->sum_messages, handle->row_message_offsets, target_slot);
    double *values = symldl_cpu_payload(sum);

    symldl_cpu_lock(&handle->target_locks[target_slot]);
    for (int rhs = 0; rhs < handle->nrhs; ++rhs)
        for (int_t row = 0; row < block->nbrow; ++row) {
            int_t relative = graph->rows[block->row_begin + row] - first;
            values[relative + (int_t) rhs * width] +=
                output[row + (int_t) rhs * block->nbrow];
        }
    symldl_cpu_decrement_target_locked(handle, target_slot);
}

static void
symldl_cpu_merge_backward(dSymLDLCPUSolveHandle *handle, int_t edge,
                          const double *output)
{
    const dSymLDLBlockDesc *block = &handle->graph->blocks[edge];
    const dSymLDLPanelDesc *panel =
        &handle->graph->panels[block->panel_id];
    int_t target_slot = block->panel_id;
    int count = symldl_cpu_product_to_int(
        panel->width, handle->nrhs,
        "SymLDL CPU backward block count overflows int.");
    dSymLDLCPUMessageHeader *sum = symldl_cpu_message(
        handle->sum_messages, handle->panel_message_offsets, target_slot);
    double *values = symldl_cpu_payload(sum);

    symldl_cpu_lock(&handle->target_locks[target_slot]);
    for (int i = 0; i < count; ++i)
        values[i] += output[i];
    symldl_cpu_decrement_target_locked(handle, target_slot);
}

static void
symldl_cpu_process_task(dSymLDLCPUSolveHandle *handle, int thread_id,
                        int_t edge)
{
    const dSymLDLSolveGraph *graph = handle->graph;
    size_t stride = symldl_cpu_checked_product(
        (size_t) graph->maxsup, (size_t) handle->nrhs_capacity,
        "SymLDL CPU worker scratch stride overflows.");
    double *gather = handle->worker_scratch +
                     (size_t) (2 * thread_id) * stride;
    double *output = gather + stride;
    double start = SuperLU_timer_();

    if (handle->phase == SYMLDL_CPU_PHASE_FORWARD) {
        const dSymLDLBlockDesc *block = &graph->blocks[edge];
        dSymLDLCPUMessageHeader *source = symldl_cpu_message(
            handle->x_messages, handle->panel_message_offsets,
            block->panel_id);
        dSymLDLCPUForwardBlock(graph, edge, handle->nrhs,
                              symldl_cpu_payload(source), output);
        symldl_cpu_merge_forward(handle, edge, output);
    } else {
        const dSymLDLBlockDesc *block = &graph->blocks[edge];
        int_t source_slot = graph->row_local_index[block->target_gid];
        dSymLDLCPUMessageHeader *source = symldl_cpu_message(
            handle->x_messages, handle->row_message_offsets, source_slot);
        dSymLDLCPUBackwardBlock(graph, edge, handle->nrhs,
                               symldl_cpu_payload(source), gather, output);
        symldl_cpu_merge_backward(handle, edge, output);
    }
    handle->worker_compute_times[thread_id] += SuperLU_timer_() - start;
    __atomic_fetch_add(&handle->tasks_completed, 1, __ATOMIC_RELEASE);
}

static void
symldl_cpu_worker_loop(dSymLDLCPUSolveHandle *handle, int thread_id)
{
    int idle = 0;
    for (;;) {
        int_t edge;
        if (symldl_cpu_claim_task(handle, &edge)) {
            symldl_cpu_process_task(handle, thread_id, edge);
            idle = 0;
            continue;
        }
        if (__atomic_load_n(&handle->stop_workers, __ATOMIC_ACQUIRE))
            return;
        if ((++idle & 255) == 0)
            sched_yield();
    }
}

static void
symldl_cpu_initialize_header(dSymLDLCPUMessageHeader *header, int kind,
                             int generation, int_t gid, int count)
{
    header->kind = kind;
    header->generation = generation;
    header->count = count;
    header->reserved = 0;
    header->gid = gid;
    header->payload_alignment = 0.0;
}

static void
symldl_cpu_post_send(dSymLDLCPUSolveHandle *handle,
                     dSymLDLCPUMessageHeader *message, int peer)
{
    int tag = handle->phase == SYMLDL_CPU_PHASE_FORWARD
                  ? SYMLDL_CPU_FORWARD_TAG : SYMLDL_CPU_BACKWARD_TAG;
    size_t bytes = symldl_cpu_message_bytes(message->count);
    int mpi_bytes = symldl_cpu_size_to_int(
        bytes, "SymLDL CPU message exceeds an MPI count.");
    if (handle->send_count >= handle->send_capacity)
        ABORT("SymLDL CPU send request array overflows.");
    if (MPI_Isend(message, mpi_bytes, MPI_BYTE, peer, tag,
                  handle->layer_comm,
                  &handle->send_requests[handle->send_count++]) !=
        MPI_SUCCESS)
        ABORT("SymLDL CPU tree send failed.");
    if (handle->phase == SYMLDL_CPU_PHASE_FORWARD) {
        if (message->kind == SYMLDL_CPU_MESSAGE_X) {
            ++handle->comm_stats.forward_x_messages;
            handle->comm_stats.forward_x_bytes += bytes;
        } else if (message->kind == SYMLDL_CPU_MESSAGE_PARTIAL) {
            ++handle->comm_stats.forward_partial_messages;
            handle->comm_stats.forward_partial_bytes += bytes;
        } else
            ABORT("SymLDL CPU send has an unknown message kind.");
    } else {
        if (message->kind == SYMLDL_CPU_MESSAGE_X) {
            ++handle->comm_stats.backward_x_messages;
            handle->comm_stats.backward_x_bytes += bytes;
        } else if (message->kind == SYMLDL_CPU_MESSAGE_PARTIAL) {
            ++handle->comm_stats.backward_partial_messages;
            handle->comm_stats.backward_partial_bytes += bytes;
        } else
            ABORT("SymLDL CPU send has an unknown message kind.");
    }
}

static void
symldl_cpu_send_children(dSymLDLCPUSolveHandle *handle,
                         const dSymLDLTreeNode *tree,
                         dSymLDLCPUMessageHeader *message)
{
    for (int child = 0; child < tree->child_count; ++child)
        symldl_cpu_post_send(handle, message, tree->children[child]);
}

static void
symldl_cpu_publish_source(dSymLDLCPUSolveHandle *handle, int_t source_slot,
                          int copy_from_x)
{
    const dSymLDLSolveGraph *graph = handle->graph;
    dSymLDLCPUMessageHeader *message;
    const dSymLDLTreeNode *tree;
    int_t gid;
    int count;

    if (source_slot < 0 || source_slot >= handle->source_count)
        ABORT("SymLDL CPU source slot is invalid.");
    if (handle->source_published[source_slot])
        ABORT("SymLDL CPU source was published more than once.");
    handle->source_published[source_slot] = 1;

    if (handle->phase == SYMLDL_CPU_PHASE_FORWARD) {
        const dSymLDLPanelDesc *panel = &graph->panels[source_slot];
        int_t row_slot = graph->row_local_index[panel->gid];
        gid = panel->gid;
        count = symldl_cpu_product_to_int(
            panel->width, handle->nrhs,
            "SymLDL CPU forward source count overflows int.");
        message = symldl_cpu_message(
            handle->x_messages, handle->panel_message_offsets, source_slot);
        tree = &graph->forward_bcast[source_slot];
        if (copy_from_x) {
            if (row_slot < 0)
                ABORT("SymLDL CPU forward owner has no local row.");
            memcpy(symldl_cpu_payload(message),
                   handle->x + symldl_cpu_x_offset(handle, row_slot),
                   (size_t) count * sizeof(double));
        }
        symldl_cpu_send_children(handle, tree, message);
        for (int_t local = 0; local < panel->block_count; ++local)
            symldl_cpu_enqueue_task(handle, panel->block_begin + local);
    } else {
        gid = graph->row_gids[source_slot];
        int_t width = graph->xsup[gid + 1] - graph->xsup[gid];
        count = symldl_cpu_product_to_int(
            width, handle->nrhs,
            "SymLDL CPU backward source count overflows int.");
        message = symldl_cpu_message(
            handle->x_messages, handle->row_message_offsets, source_slot);
        tree = &graph->backward_bcast[source_slot];
        if (copy_from_x)
            memcpy(symldl_cpu_payload(message),
                   handle->x + symldl_cpu_x_offset(handle, source_slot),
                   (size_t) count * sizeof(double));
        symldl_cpu_send_children(handle, tree, message);
        for (int_t pos = graph->source_edge_offsets[source_slot];
             pos < graph->source_edge_offsets[source_slot + 1]; ++pos)
            symldl_cpu_enqueue_task(handle, graph->source_edge_ids[pos]);
    }
    if (!tree->active)
        ABORT("SymLDL CPU published source has no broadcast tree.");
    if (message->gid != gid || message->count != count)
        ABORT("SymLDL CPU source message metadata is inconsistent.");
}

static void
symldl_cpu_process_target(dSymLDLCPUSolveHandle *handle, int_t target_slot)
{
    const dSymLDLSolveGraph *graph = handle->graph;
    const dSymLDLTreeNode *tree;
    dSymLDLCPUMessageHeader *sum;
    int_t gid;

    if (target_slot < 0 || target_slot >= handle->target_count ||
        handle->target_terminal[target_slot])
        ABORT("SymLDL CPU terminal target is invalid.");
    handle->target_terminal[target_slot] = 1;
    ++handle->terminal_completed;

    if (handle->phase == SYMLDL_CPU_PHASE_FORWARD) {
        gid = graph->row_gids[target_slot];
        tree = &graph->forward_reduce[target_slot];
        sum = symldl_cpu_message(
            handle->sum_messages, handle->row_message_offsets, target_slot);
        if (tree->active && tree->parent_rank >= 0) {
            symldl_cpu_post_send(handle, sum, tree->parent_rank);
            return;
        }
        if (handle->mycol != graph->panel_roots[gid])
            ABORT("SymLDL CPU forward reduction terminated off the root.");
        int_t width = graph->xsup[gid + 1] - graph->xsup[gid];
        int count = symldl_cpu_product_to_int(
            width, handle->nrhs,
            "SymLDL CPU forward target count overflows int.");
        double *x = handle->x + symldl_cpu_x_offset(handle, target_slot);
        double *values = symldl_cpu_payload(sum);
        for (int i = 0; i < count; ++i)
            x[i] += values[i];
        int_t panel_slot = graph->panel_local_index[gid];
        if (panel_slot < 0)
            ABORT("SymLDL CPU forward root has no local panel.");
        symldl_cpu_publish_source(handle, panel_slot, 1);
    } else {
        const dSymLDLPanelDesc *panel = &graph->panels[target_slot];
        gid = panel->gid;
        tree = &graph->backward_reduce[target_slot];
        sum = symldl_cpu_message(
            handle->sum_messages, handle->panel_message_offsets, target_slot);
        if (tree->active && tree->parent_rank >= 0) {
            symldl_cpu_post_send(handle, sum, tree->parent_rank);
            return;
        }
        if (handle->myrow != graph->diag_roots[gid])
            ABORT("SymLDL CPU backward reduction terminated off the root.");
        int_t row_slot = graph->row_local_index[gid];
        if (row_slot < 0)
            ABORT("SymLDL CPU backward root has no local row.");
        int count = symldl_cpu_product_to_int(
            panel->width, handle->nrhs,
            "SymLDL CPU backward target count overflows int.");
        double *x = handle->x + symldl_cpu_x_offset(handle, row_slot);
        double *values = symldl_cpu_payload(sum);
        for (int i = 0; i < count; ++i)
            x[i] += values[i];
        symldl_cpu_publish_source(handle, row_slot, 1);
    }
}

static int
symldl_cpu_child_index(const dSymLDLTreeNode *tree, int source)
{
    for (int child = 0; child < tree->child_count; ++child)
        if (tree->children[child] == source)
            return child;
    return -1;
}

static void
symldl_cpu_merge_received_partial(dSymLDLCPUSolveHandle *handle,
                                  int_t target_slot, int source,
                                  const double *payload, int count)
{
    const dSymLDLTreeNode *tree =
        handle->phase == SYMLDL_CPU_PHASE_FORWARD
            ? &handle->graph->forward_reduce[target_slot]
            : &handle->graph->backward_reduce[target_slot];
    int child = symldl_cpu_child_index(tree, source);
    dSymLDLCPUMessageHeader *sum =
        handle->phase == SYMLDL_CPU_PHASE_FORWARD
            ? symldl_cpu_message(handle->sum_messages,
                                  handle->row_message_offsets, target_slot)
            : symldl_cpu_message(handle->sum_messages,
                                  handle->panel_message_offsets, target_slot);
    if (child < 0)
        ABORT("SymLDL CPU partial arrived from a non-child rank.");

    symldl_cpu_lock(&handle->target_locks[target_slot]);
    if (handle->received_child_mask[target_slot] & (1u << child))
        ABORT("SymLDL CPU partial arrived twice from one child.");
    handle->received_child_mask[target_slot] |= (unsigned char) (1u << child);
    double *values = symldl_cpu_payload(sum);
    for (int i = 0; i < count; ++i)
        values[i] += payload[i];
    symldl_cpu_decrement_target_locked(handle, target_slot);
}

static void
symldl_cpu_dispatch_message(dSymLDLCPUSolveHandle *handle, int receive_slot,
                            const MPI_Status *status)
{
    const dSymLDLSolveGraph *graph = handle->graph;
    dSymLDLCPUMessageHeader *message =
        (dSymLDLCPUMessageHeader *)
            (handle->receive_messages +
             (size_t) receive_slot * handle->receive_stride);
    int bytes = 0;
    MPI_Get_count((MPI_Status *) status, MPI_BYTE, &bytes);
    if (bytes < (int) sizeof(*message) ||
        message->generation != handle->generation ||
        message->gid < 0 || message->gid >= graph->nsupers ||
        message->count < 0 ||
        (size_t) bytes != symldl_cpu_message_bytes(message->count))
        ABORT("SymLDL CPU solve received an invalid message.");

    int_t gid = message->gid;
    int_t width = graph->xsup[gid + 1] - graph->xsup[gid];
    int expected_count = symldl_cpu_product_to_int(
        width, handle->nrhs,
        "SymLDL CPU receive count overflows int.");
    if (message->count != expected_count)
        ABORT("SymLDL CPU solve received a message with the wrong width.");

    if (message->kind == SYMLDL_CPU_MESSAGE_X) {
        int_t source_slot;
        const dSymLDLTreeNode *tree;
        dSymLDLCPUMessageHeader *destination;
        if (handle->phase == SYMLDL_CPU_PHASE_FORWARD) {
            source_slot = graph->panel_local_index[gid];
            if (source_slot < 0)
                ABORT("SymLDL CPU forward X has no local panel.");
            tree = &graph->forward_bcast[source_slot];
            destination = symldl_cpu_message(
                handle->x_messages, handle->panel_message_offsets,
                source_slot);
        } else {
            source_slot = graph->row_local_index[gid];
            if (source_slot < 0)
                ABORT("SymLDL CPU backward X has no local row.");
            tree = &graph->backward_bcast[source_slot];
            destination = symldl_cpu_message(
                handle->x_messages, handle->row_message_offsets,
                source_slot);
        }
        if (!tree->active || tree->parent_rank != status->MPI_SOURCE)
            ABORT("SymLDL CPU X arrived from the wrong tree parent.");
        memcpy(symldl_cpu_payload(destination), symldl_cpu_payload(message),
               (size_t) expected_count * sizeof(double));
        symldl_cpu_publish_source(handle, source_slot, 0);
        return;
    }

    if (message->kind == SYMLDL_CPU_MESSAGE_PARTIAL) {
        int_t target_slot = handle->phase == SYMLDL_CPU_PHASE_FORWARD
                                ? graph->row_local_index[gid]
                                : graph->panel_local_index[gid];
        if (target_slot < 0)
            ABORT("SymLDL CPU partial has no local target.");
        symldl_cpu_merge_received_partial(
            handle, target_slot, status->MPI_SOURCE,
            symldl_cpu_payload(message), expected_count);
        return;
    }
    ABORT("SymLDL CPU solve received an unknown message kind.");
}

static void
symldl_cpu_post_receive(dSymLDLCPUSolveHandle *handle, int slot)
{
    int tag = handle->phase == SYMLDL_CPU_PHASE_FORWARD
                  ? SYMLDL_CPU_FORWARD_TAG : SYMLDL_CPU_BACKWARD_TAG;
    int bytes = symldl_cpu_size_to_int(
        handle->receive_stride,
        "SymLDL CPU receive slot exceeds an MPI count.");
    void *buffer = handle->receive_messages +
                   (size_t) slot * handle->receive_stride;
    if (MPI_Irecv(buffer, bytes, MPI_BYTE, MPI_ANY_SOURCE, tag,
                  handle->layer_comm, &handle->receive_requests[slot]) !=
        MPI_SUCCESS)
        ABORT("SymLDL CPU receive posting failed.");
    ++handle->receive_posted;
}

static int
symldl_cpu_progress_receives(dSymLDLCPUSolveHandle *handle)
{
    if (handle->receive_completed >= handle->receive_expected)
        return 0;
    int completed = 0;
    double start = SuperLU_timer_();
    MPI_Testsome(handle->receive_depth, handle->receive_requests,
                 &completed, handle->completed_receive_indices,
                 handle->completed_receive_statuses);
    handle->mpi_progress_time += SuperLU_timer_() - start;
    if (completed == MPI_UNDEFINED || completed <= 0)
        return 0;
    for (int item = 0; item < completed; ++item) {
        int slot = handle->completed_receive_indices[item];
        symldl_cpu_dispatch_message(
            handle, slot, &handle->completed_receive_statuses[item]);
        ++handle->receive_completed;
        if (handle->receive_posted < handle->receive_expected)
            symldl_cpu_post_receive(handle, slot);
    }
    return completed;
}

static int
symldl_cpu_drain_completions(dSymLDLCPUSolveHandle *handle)
{
    int completed = 0;
    for (;;) {
        int_t reserved = __atomic_load_n(
            &handle->completions_reserved, __ATOMIC_ACQUIRE);
        if (handle->completions_consumed >= reserved)
            break;
        int_t position = handle->completions_consumed;
        if (!__atomic_load_n(&handle->completion_ready[position],
                             __ATOMIC_ACQUIRE))
            break;
        int_t slot = handle->completion_slots[position];
        ++handle->completions_consumed;
        symldl_cpu_process_target(handle, slot);
        ++completed;
    }
    return completed;
}

static int
symldl_cpu_phase_done(dSymLDLCPUSolveHandle *handle)
{
    return handle->terminal_completed == handle->terminal_expected &&
           handle->receive_completed == handle->receive_expected &&
           __atomic_load_n(&handle->tasks_published, __ATOMIC_ACQUIRE) ==
               handle->graph->block_count &&
           __atomic_load_n(&handle->tasks_completed, __ATOMIC_ACQUIRE) ==
               handle->graph->block_count;
}

static void
symldl_cpu_scheduler(dSymLDLCPUSolveHandle *handle, int team_size)
{
    int idle = 0;
    while (!symldl_cpu_phase_done(handle)) {
        int progress = symldl_cpu_progress_receives(handle);
        progress += symldl_cpu_drain_completions(handle);
        if (team_size == 1) {
            int_t edge;
            if (symldl_cpu_claim_task(handle, &edge)) {
                symldl_cpu_process_task(handle, 0, edge);
                ++progress;
            }
        }
        if (progress == 0 && (++idle & 255) == 0)
            sched_yield();
        else if (progress != 0)
            idle = 0;
    }
    __atomic_store_n(&handle->stop_workers, 1, __ATOMIC_RELEASE);
    if (handle->send_count > 0) {
        double start = SuperLU_timer_();
        MPI_Waitall(handle->send_count, handle->send_requests,
                    MPI_STATUSES_IGNORE);
        handle->mpi_progress_time += SuperLU_timer_() - start;
    }
}

static void
symldl_cpu_prepare_phase(dSymLDLCPUSolveHandle *handle, int phase,
                         double *x, int_t x_count, int nrhs)
{
    const dSymLDLSolveGraph *graph = handle->graph;
    const int_t *source_gids;
    const int_t *target_gids;
    const size_t *source_offsets;
    const size_t *target_offsets;

    if (nrhs <= 0 || nrhs > handle->nrhs_capacity || x == NULL ||
        x_count <= 0)
        ABORT("SymLDL CPU solve phase has invalid workspace dimensions.");
    handle->phase = phase;
    handle->nrhs = nrhs;
    handle->x = x;
    handle->x_count = x_count;
    handle->generation = handle->generation == INT_MAX
                             ? 1 : handle->generation + 1;
    handle->target_count = phase == SYMLDL_CPU_PHASE_FORWARD
                               ? (int) graph->row_count
                               : (int) graph->panel_count;
    handle->source_count = phase == SYMLDL_CPU_PHASE_FORWARD
                               ? (int) graph->panel_count
                               : (int) graph->row_count;
    source_gids = phase == SYMLDL_CPU_PHASE_FORWARD
                      ? graph->panel_gids : graph->row_gids;
    target_gids = phase == SYMLDL_CPU_PHASE_FORWARD
                      ? graph->row_gids : graph->panel_gids;
    source_offsets = phase == SYMLDL_CPU_PHASE_FORWARD
                         ? handle->panel_message_offsets
                         : handle->row_message_offsets;
    target_offsets = phase == SYMLDL_CPU_PHASE_FORWARD
                         ? handle->row_message_offsets
                         : handle->panel_message_offsets;

    memset(handle->target_locks, 0,
           (size_t) SUPERLU_MAX(1, handle->target_count));
    memset(handle->target_queued, 0,
           (size_t) SUPERLU_MAX(1, handle->target_count));
    memset(handle->target_terminal, 0,
           (size_t) SUPERLU_MAX(1, handle->target_count));
    memset(handle->received_child_mask, 0,
           (size_t) SUPERLU_MAX(1, handle->target_count));
    memset(handle->source_published, 0,
           (size_t) SUPERLU_MAX(1, handle->source_count));
    memset(handle->completion_ready, 0,
           (size_t) SUPERLU_MAX(1, handle->target_count));
    memset(handle->worker_compute_times, 0,
           (size_t) handle->max_threads * sizeof(double));
    for (int slot = 0; slot < handle->receive_depth; ++slot)
        handle->receive_requests[slot] = MPI_REQUEST_NULL;
    for (int slot = 0; slot < handle->send_capacity; ++slot)
        handle->send_requests[slot] = MPI_REQUEST_NULL;

    __atomic_store_n(&handle->stop_workers, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&handle->tasks_published, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&handle->tasks_claimed, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&handle->tasks_completed, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&handle->completions_reserved, 0, __ATOMIC_RELEASE);
    handle->completions_consumed = 0;
    handle->terminal_expected = 0;
    handle->terminal_completed = 0;
    handle->receive_expected = 0;
    handle->receive_posted = 0;
    handle->receive_completed = 0;
    handle->send_count = 0;

    for (int slot = 0; slot < handle->source_count; ++slot) {
        int_t gid = source_gids[slot];
        int_t width = graph->xsup[gid + 1] - graph->xsup[gid];
        int count = symldl_cpu_product_to_int(
            width, nrhs, "SymLDL CPU source count overflows int.");
        dSymLDLCPUMessageHeader *message = symldl_cpu_message(
            handle->x_messages, source_offsets, slot);
        symldl_cpu_initialize_header(
            message, SYMLDL_CPU_MESSAGE_X, handle->generation, gid, count);
    }

    for (int slot = 0; slot < handle->target_count; ++slot) {
        int_t gid = target_gids[slot];
        int_t width = graph->xsup[gid + 1] - graph->xsup[gid];
        int count = symldl_cpu_product_to_int(
            width, nrhs, "SymLDL CPU target count overflows int.");
        dSymLDLCPUMessageHeader *message = symldl_cpu_message(
            handle->sum_messages, target_offsets, slot);
        symldl_cpu_initialize_header(
            message, SYMLDL_CPU_MESSAGE_PARTIAL, handle->generation,
            gid, count);
        memset(symldl_cpu_payload(message), 0,
               (size_t) count * sizeof(double));

        if (phase == SYMLDL_CPU_PHASE_FORWARD) {
            const dSymLDLTreeNode *reduce = &graph->forward_reduce[slot];
            int owner = handle->mycol == graph->panel_roots[gid] &&
                        symldl_cpu_panel_active(handle, gid);
            int participant = reduce->active || owner;
            handle->dependencies[slot] =
                graph->forward_local_dependencies[slot] +
                (reduce->active ? reduce->child_count : 0);
            if (participant) {
                ++handle->terminal_expected;
                if (handle->dependencies[slot] == 0)
                    symldl_cpu_queue_target(handle, slot);
            } else if (handle->dependencies[slot] != 0) {
                ABORT("SymLDL CPU forward dependency has no reduction tree.");
            }
        } else {
            const dSymLDLTreeNode *reduce = &graph->backward_reduce[slot];
            handle->dependencies[slot] =
                graph->backward_local_dependencies[slot] +
                (reduce->active ? reduce->child_count : 0);
            if (reduce->active) {
                ++handle->terminal_expected;
                if (handle->dependencies[slot] == 0)
                    symldl_cpu_queue_target(handle, slot);
            } else if (handle->dependencies[slot] != 0) {
                ABORT("SymLDL CPU backward dependency has no reduction tree.");
            }
        }
    }

    if (phase == SYMLDL_CPU_PHASE_FORWARD) {
        for (int_t slot = 0; slot < graph->panel_count; ++slot)
            handle->receive_expected +=
                graph->forward_bcast[slot].parent_rank >= 0;
        for (int_t slot = 0; slot < graph->row_count; ++slot)
            handle->receive_expected +=
                graph->forward_reduce[slot].child_count;
    } else {
        for (int_t slot = 0; slot < graph->row_count; ++slot)
            handle->receive_expected +=
                graph->backward_bcast[slot].parent_rank >= 0;
        for (int_t slot = 0; slot < graph->panel_count; ++slot)
            handle->receive_expected +=
                graph->backward_reduce[slot].child_count;
    }
    for (int slot = 0;
         slot < handle->receive_depth &&
         handle->receive_posted < handle->receive_expected; ++slot)
        symldl_cpu_post_receive(handle, slot);
}

static int
symldl_cpu_run_phase(dSymLDLCPUSolveHandle *handle, int phase,
                     double *x, int_t x_count, int nrhs)
{
    double start = SuperLU_timer_();
    symldl_cpu_prepare_phase(handle, phase, x, x_count, nrhs);

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        int thread_id = 0;
        int team_size = 1;
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
        team_size = omp_get_num_threads();
#endif
        if (thread_id == 0)
            symldl_cpu_scheduler(handle, team_size);
        else
            symldl_cpu_worker_loop(handle, thread_id);
    }

    for (int thread = 0; thread < handle->max_threads; ++thread)
        handle->numeric_compute_time +=
            handle->worker_compute_times[thread];
    if (phase == SYMLDL_CPU_PHASE_FORWARD)
        handle->forward_time += SuperLU_timer_() - start;
    else
        handle->backward_time += SuperLU_timer_() - start;
    return 0;
}

static void
symldl_cpu_replicate_final_panels(dSymLDLCPUSolveHandle *handle)
{
    const dSymLDLSolveGraph *graph = handle->graph;
    const int gid_count = symldl_cpu_count_to_int(
        graph->nsupers, "SymLDL CPU panel replication count overflows int.");
    const int_t missing_count = sizeof(int_t) == sizeof(int64_t)
                                    ? (int_t) INT64_MAX : (int_t) INT32_MAX;
    int *local_source_count;
    int *source_count;
    int *local_source;
    int *source;
    int_t *local_min_count;
    int_t *min_count;
    int_t *local_max_count;
    int_t *value_count;
    int_t *segment_gids;
    int_t *segment_offsets;
    int_t *segment_counts;
    double *batch;
    const int_t batch_capacity = 1024 * 1024;
    double start;

    if (handle->z_size <= 1)
        return;
    start = SuperLU_timer_();
    local_source_count = (int *) symldl_cpu_alloc(
        (size_t) gid_count, sizeof(int),
        "Malloc fails for SymLDL CPU panel source counts.");
    source_count = (int *) symldl_cpu_alloc(
        (size_t) gid_count, sizeof(int),
        "Malloc fails for SymLDL CPU panel source counts.");
    local_source = (int *) symldl_cpu_alloc(
        (size_t) gid_count, sizeof(int),
        "Malloc fails for SymLDL CPU panel sources.");
    source = (int *) symldl_cpu_alloc(
        (size_t) gid_count, sizeof(int),
        "Malloc fails for SymLDL CPU panel sources.");
    local_min_count = (int_t *) symldl_cpu_alloc(
        (size_t) gid_count, sizeof(int_t),
        "Malloc fails for SymLDL CPU panel sizes.");
    min_count = (int_t *) symldl_cpu_alloc(
        (size_t) gid_count, sizeof(int_t),
        "Malloc fails for SymLDL CPU panel sizes.");
    local_max_count = (int_t *) symldl_cpu_alloc(
        (size_t) gid_count, sizeof(int_t),
        "Malloc fails for SymLDL CPU panel sizes.");
    value_count = (int_t *) symldl_cpu_alloc(
        (size_t) gid_count, sizeof(int_t),
        "Malloc fails for SymLDL CPU panel sizes.");

    for (int gid = 0; gid < gid_count; ++gid) {
        int_t slot = graph->panel_local_index[gid];
        const dSymLDLPanelDesc *panel =
            slot >= 0 && slot < graph->panel_count
                ? &graph->panels[slot] : NULL;
        int active = panel != NULL && panel->gid == gid && panel->active;
        int is_source = active &&
                        handle->partition->superGridMap[gid] == IN_GRID_AIJ;
        local_source_count[gid] = is_source;
        local_source[gid] = is_source ? handle->z_rank : INT_MAX;
        local_min_count[gid] = active ? panel->value_count : missing_count;
        local_max_count[gid] = active ? panel->value_count : -1;
    }
    MPI_Allreduce(local_source_count, source_count, gid_count, MPI_INT,
                  MPI_SUM, handle->z_comm);
    MPI_Allreduce(local_source, source, gid_count, MPI_INT, MPI_MIN,
                  handle->z_comm);
    MPI_Allreduce(local_min_count, min_count, gid_count, mpi_int_t, MPI_MIN,
                  handle->z_comm);
    MPI_Allreduce(local_max_count, value_count, gid_count, mpi_int_t,
                  MPI_MAX, handle->z_comm);
    for (int gid = 0; gid < gid_count; ++gid) {
        if (value_count[gid] < 0)
            continue;
        if (source_count[gid] != 1 || source[gid] == INT_MAX)
            ABORT("SymLDL CPU finalized panel source is not unique.");
        if (min_count[gid] != value_count[gid])
            ABORT("SymLDL CPU panel layouts differ across Z layers.");
    }

    segment_gids = (int_t *) symldl_cpu_alloc(
        (size_t) gid_count + 1, sizeof(int_t),
        "Malloc fails for SymLDL CPU panel replication segments.");
    segment_offsets = (int_t *) symldl_cpu_alloc(
        (size_t) gid_count + 1, sizeof(int_t),
        "Malloc fails for SymLDL CPU panel replication segments.");
    segment_counts = (int_t *) symldl_cpu_alloc(
        (size_t) gid_count + 1, sizeof(int_t),
        "Malloc fails for SymLDL CPU panel replication segments.");
    batch = (double *) symldl_cpu_alloc(
        (size_t) batch_capacity, sizeof(double),
        "Malloc fails for SymLDL CPU panel replication buffer.");

    for (int root = 0; root < handle->z_size; ++root) {
        int gid = 0;
        int_t panel_offset = 0;
        while (gid < gid_count) {
            int segments = 0;
            int_t used = 0;
            while (gid < gid_count && used < batch_capacity) {
                int_t remaining;
                int_t take;
                if (source[gid] != root || value_count[gid] < 0) {
                    ++gid;
                    panel_offset = 0;
                    continue;
                }
                remaining = value_count[gid] - panel_offset;
                if (remaining <= 0) {
                    ++gid;
                    panel_offset = 0;
                    continue;
                }
                take = SUPERLU_MIN(remaining, batch_capacity - used);
                segment_gids[segments] = gid;
                segment_offsets[segments] = panel_offset;
                segment_counts[segments] = take;
                ++segments;
                used += take;
                panel_offset += take;
                if (panel_offset == value_count[gid]) {
                    ++gid;
                    panel_offset = 0;
                }
            }
            if (used == 0)
                continue;
            if (handle->z_rank == root) {
                int_t write = 0;
                for (int segment = 0; segment < segments; ++segment) {
                    int_t segment_gid = segment_gids[segment];
                    int_t slot = graph->panel_local_index[segment_gid];
                    const dSymLDLPanelDesc *panel =
                        slot >= 0 && slot < graph->panel_count
                            ? &graph->panels[slot] : NULL;
                    if (panel == NULL || panel->gid != segment_gid ||
                        !panel->active || panel->values == NULL)
                        ABORT("SymLDL CPU source panel storage is unavailable.");
                    memcpy(batch + write,
                           panel->values + segment_offsets[segment],
                           (size_t) segment_counts[segment] * sizeof(double));
                    write += segment_counts[segment];
                }
            }
            MPI_Bcast(batch, (int) used, MPI_DOUBLE, root, handle->z_comm);
            {
                int_t read = 0;
                for (int segment = 0; segment < segments; ++segment) {
                    int_t panel_gid = segment_gids[segment];
                    int_t slot = graph->panel_local_index[panel_gid];
                    const dSymLDLPanelDesc *panel =
                        slot >= 0 && slot < graph->panel_count
                            ? &graph->panels[slot] : NULL;
                    if (panel != NULL && panel->gid == panel_gid &&
                        panel->active) {
                        if (panel->values == NULL ||
                            panel->value_count != value_count[panel_gid])
                            ABORT("SymLDL CPU target panel storage is unavailable.");
                        memcpy(panel->values + segment_offsets[segment],
                               batch + read,
                               (size_t) segment_counts[segment] *
                                   sizeof(double));
                    }
                    read += segment_counts[segment];
                }
            }
        }
    }

    SUPERLU_FREE(batch);
    SUPERLU_FREE(segment_counts);
    SUPERLU_FREE(segment_offsets);
    SUPERLU_FREE(segment_gids);
    SUPERLU_FREE(value_count);
    SUPERLU_FREE(local_max_count);
    SUPERLU_FREE(min_count);
    SUPERLU_FREE(local_min_count);
    SUPERLU_FREE(source);
    SUPERLU_FREE(local_source);
    SUPERLU_FREE(source_count);
    SUPERLU_FREE(local_source_count);
    handle->panel_replication_time += SuperLU_timer_() - start;
}

static void
symldl_cpu_build_z_workspace(dSymLDLCPUSolveHandle *handle)
{
    const dSymLDLSolveGraph *graph = handle->graph;
    int_t rhs_count = 0;
    size_t rhs_values = 0;
    size_t z_values = 0;

    if (handle->z_size <= 1)
        return;
    for (int_t k = 0; k < graph->nsupers; ++k)
        rhs_count += graph->diag_roots[k] == handle->myrow &&
                     graph->panel_roots[k] == handle->mycol;
    handle->rhs_count = rhs_count;
    handle->rhs_gids = (int_t *) symldl_cpu_alloc(
        (size_t) rhs_count, sizeof(int_t),
        "Malloc fails for SymLDL CPU RHS gids.");
    handle->rhs_offsets = (int_t *) symldl_cpu_alloc(
        (size_t) rhs_count + 1, sizeof(int_t),
        "Malloc fails for SymLDL CPU RHS offsets.");
    handle->rhs_offsets[0] = 0;
    rhs_count = 0;
    for (int_t k = 0; k < graph->nsupers; ++k) {
        size_t count;
        if (graph->diag_roots[k] != handle->myrow ||
            graph->panel_roots[k] != handle->mycol)
            continue;
        count = (size_t) XK_H + symldl_cpu_checked_product(
            (size_t) (graph->xsup[k + 1] - graph->xsup[k]),
            (size_t) handle->nrhs_capacity,
            "SymLDL CPU RHS workspace overflows.");
        rhs_values = symldl_cpu_checked_sum(
            rhs_values, count, "SymLDL CPU RHS workspace overflows.");
        handle->rhs_gids[rhs_count] = k;
        handle->rhs_offsets[rhs_count + 1] = symldl_cpu_size_to_int_t(
            rhs_values, "SymLDL CPU RHS workspace overflows int_t.");
        ++rhs_count;
    }
    handle->rhs_value_count = symldl_cpu_size_to_int_t(
        rhs_values, "SymLDL CPU RHS workspace overflows int_t.");
    handle->rhs_values = (double *) symldl_cpu_alloc(
        SUPERLU_MAX((size_t) 1, rhs_values), sizeof(double),
        "Malloc fails for SymLDL CPU RHS workspace.");

    for (int stage = 0; stage < graph->z_stage_count; ++stage) {
        size_t count = symldl_cpu_checked_product(
            (size_t) graph->z_stages[stage].value_count,
            (size_t) handle->nrhs_capacity,
            "SymLDL CPU sparse Z workspace overflows.");
        z_values = SUPERLU_MAX(z_values, count);
    }
    handle->z_value_capacity = symldl_cpu_size_to_int_t(
        z_values, "SymLDL CPU sparse Z workspace overflows int_t.");
    handle->z_send_values = (double *) symldl_cpu_alloc(
        SUPERLU_MAX((size_t) 1, z_values), sizeof(double),
        "Malloc fails for SymLDL CPU sparse Z send workspace.");
    handle->z_recv_values = (double *) symldl_cpu_alloc(
        SUPERLU_MAX((size_t) 1, z_values), sizeof(double),
        "Malloc fails for SymLDL CPU sparse Z receive workspace.");
}

static void
symldl_cpu_allreduce_in_place(double *values, int_t count, MPI_Comm comm)
{
    const int_t chunk_capacity = 8 * 1024 * 1024;
    for (int_t begin = 0; begin < count; begin += chunk_capacity) {
        int count_now = (int) SUPERLU_MIN(chunk_capacity, count - begin);
        MPI_Allreduce(MPI_IN_PLACE, values + begin, count_now, MPI_DOUBLE,
                      MPI_SUM, comm);
    }
}

static void
symldl_cpu_initialize_rhs(dSymLDLCPUSolveHandle *handle, double *x,
                          int_t x_count, int nrhs)
{
    const dSymLDLSolveGraph *graph = handle->graph;
    double start;
    if (handle->z_size <= 1)
        return;
    start = SuperLU_timer_();
    memset(handle->rhs_values, 0,
           (size_t) handle->rhs_value_count * sizeof(double));
    for (int_t pos = 0; pos < handle->rhs_count; ++pos) {
        int_t gid = handle->rhs_gids[pos];
        int_t row_slot = graph->row_local_index[gid];
        int_t width = graph->xsup[gid + 1] - graph->xsup[gid];
        int_t count = symldl_cpu_size_to_int_t(
            symldl_cpu_checked_sum(
                (size_t) XK_H,
                symldl_cpu_checked_product(
                    (size_t) width, (size_t) nrhs,
                    "SymLDL CPU RHS count overflows."),
                "SymLDL CPU RHS count overflows."),
            "SymLDL CPU RHS count overflows int_t.");
        if (handle->global_rank == handle->partition->symV2DiagOwner[gid]) {
            int_t x_offset;
            if (row_slot < 0)
                ABORT("SymLDL CPU RHS owner has no local solution row.");
            x_offset = symldl_cpu_x_offset(handle, row_slot) - XK_H;
            memcpy(handle->rhs_values + handle->rhs_offsets[pos],
                   x + x_offset, (size_t) count * sizeof(double));
        }
    }
    symldl_cpu_allreduce_in_place(
        handle->rhs_values, handle->rhs_value_count, handle->z_comm);

    memset(x, 0, (size_t) x_count * sizeof(double));
    for (int_t pos = 0; pos < handle->rhs_count; ++pos) {
        int_t gid = handle->rhs_gids[pos];
        int_t row_slot = graph->row_local_index[gid];
        int_t width = graph->xsup[gid + 1] - graph->xsup[gid];
        int_t x_offset;
        const double *source = handle->rhs_values + handle->rhs_offsets[pos];
        if (row_slot < 0)
            continue;
        x_offset = symldl_cpu_x_offset(handle, row_slot);
        memcpy(x + x_offset - XK_H, source,
               (size_t) XK_H * sizeof(double));
        if (handle->partition->superGridMap[gid] == IN_GRID_AIJ) {
            if (!symldl_cpu_panel_active(handle, gid))
                ABORT("SymLDL CPU active RHS replica has no retained panel.");
            size_t values = symldl_cpu_checked_product(
                (size_t) width, (size_t) nrhs,
                "SymLDL CPU RHS restore count overflows.");
            memcpy(x + x_offset, source + XK_H,
                   symldl_cpu_checked_product(
                       values, sizeof(double),
                       "SymLDL CPU RHS restore bytes overflow."));
        }
    }
    handle->rhs_distribution_time += SuperLU_timer_() - start;
}

static int_t
symldl_cpu_pack_z_stage(dSymLDLCPUSolveHandle *handle,
                        const dSymLDLZStage *stage, const double *x,
                        double *packed)
{
    const dSymLDLSolveGraph *graph = handle->graph;
    int_t write = 0;
    for (int_t pos = 0; pos < stage->gid_count; ++pos) {
        int_t gid = graph->z_stage_gids[stage->gid_begin + pos];
        int_t row_slot = graph->row_local_index[gid];
        int_t count = symldl_cpu_product_to_int_t(
            graph->xsup[gid + 1] - graph->xsup[gid], handle->nrhs,
            "SymLDL CPU sparse Z block count overflows int_t.");
        if (row_slot < 0)
            ABORT("SymLDL CPU sparse Z node has no local solution row.");
        if (write < 0 || count > handle->z_value_capacity - write)
            ABORT("SymLDL CPU sparse Z pack overflows its workspace.");
        memcpy(packed + write, x + symldl_cpu_x_offset(handle, row_slot),
               (size_t) count * sizeof(double));
        write += count;
    }
    if (write != symldl_cpu_product_to_int_t(
                     stage->value_count, handle->nrhs,
                     "SymLDL CPU sparse Z stage count overflows int_t."))
        ABORT("SymLDL CPU sparse Z pack is inconsistent.");
    return write;
}

static void
symldl_cpu_unpack_z_stage(dSymLDLCPUSolveHandle *handle,
                          const dSymLDLZStage *stage, double *x,
                          const double *packed, int accumulate)
{
    const dSymLDLSolveGraph *graph = handle->graph;
    int_t read = 0;
    for (int_t pos = 0; pos < stage->gid_count; ++pos) {
        int_t gid = graph->z_stage_gids[stage->gid_begin + pos];
        int_t row_slot = graph->row_local_index[gid];
        int_t count = symldl_cpu_product_to_int_t(
            graph->xsup[gid + 1] - graph->xsup[gid], handle->nrhs,
            "SymLDL CPU sparse Z block count overflows int_t.");
        double *target;
        if (row_slot < 0)
            ABORT("SymLDL CPU sparse Z node has no local solution row.");
        if (read < 0 || count > handle->z_value_capacity - read)
            ABORT("SymLDL CPU sparse Z unpack exceeds its workspace.");
        target = x + symldl_cpu_x_offset(handle, row_slot);
        if (accumulate) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int_t i = 0; i < count; ++i)
                target[i] += packed[read + i];
        } else {
            memcpy(target, packed + read, (size_t) count * sizeof(double));
        }
        read += count;
    }
    if (read != symldl_cpu_product_to_int_t(
                    stage->value_count, handle->nrhs,
                    "SymLDL CPU sparse Z stage count overflows int_t."))
        ABORT("SymLDL CPU sparse Z unpack is inconsistent.");
}

static void
symldl_cpu_zero_z_stage(dSymLDLCPUSolveHandle *handle,
                        const dSymLDLZStage *stage, double *x)
{
    const dSymLDLSolveGraph *graph = handle->graph;
    for (int_t pos = 0; pos < stage->gid_count; ++pos) {
        int_t gid = graph->z_stage_gids[stage->gid_begin + pos];
        int_t row_slot = graph->row_local_index[gid];
        int_t count = symldl_cpu_product_to_int_t(
            graph->xsup[gid + 1] - graph->xsup[gid], handle->nrhs,
            "SymLDL CPU sparse Z block count overflows int_t.");
        memset(x + symldl_cpu_x_offset(handle, row_slot), 0,
               (size_t) count * sizeof(double));
    }
}

static void
symldl_cpu_z_exchange(dSymLDLCPUSolveHandle *handle, int source,
                      int target, int tag, int_t count)
{
    const int_t chunk_capacity = 8 * 1024 * 1024;
    for (int_t begin = 0; begin < count; begin += chunk_capacity) {
        int count_now = (int) SUPERLU_MIN(chunk_capacity, count - begin);
        MPI_Request request = MPI_REQUEST_NULL;
        if (handle->z_rank == target)
            MPI_Irecv(handle->z_recv_values + begin, count_now, MPI_DOUBLE,
                      source, tag, handle->z_comm, &request);
        else if (handle->z_rank == source)
            MPI_Isend(handle->z_send_values + begin, count_now, MPI_DOUBLE,
                      target, tag, handle->z_comm, &request);
        else
            ABORT("SymLDL CPU sparse Z stage has an invalid local role.");
        MPI_Wait(&request, MPI_STATUS_IGNORE);
    }
}

static void
symldl_cpu_sparse_z(dSymLDLCPUSolveHandle *handle, double *x)
{
    const dSymLDLSolveGraph *graph = handle->graph;
    double start;
    if (handle->z_size <= 1)
        return;

    start = SuperLU_timer_();
    for (int stage_id = 0; stage_id < graph->z_stage_count; ++stage_id) {
        const dSymLDLZStage *stage = &graph->z_stages[stage_id];
        int_t count;
        if (!stage->active || stage->gid_count == 0)
            continue;
        count = symldl_cpu_product_to_int_t(
            stage->value_count, handle->nrhs,
            "SymLDL CPU sparse Z stage count overflows int_t.");
        if (handle->z_rank == stage->sender_z)
            symldl_cpu_pack_z_stage(
                handle, stage, x, handle->z_send_values);
        symldl_cpu_z_exchange(handle, stage->sender_z, stage->receiver_z,
                              SYMLDL_CPU_Z_REDUCE_TAG, count);
        if (handle->z_rank == stage->sender_z)
            symldl_cpu_zero_z_stage(handle, stage, x);
        else
            symldl_cpu_unpack_z_stage(
                handle, stage, x, handle->z_recv_values, 1);
    }
    handle->sparse_reduce_time += SuperLU_timer_() - start;

    start = SuperLU_timer_();
    for (int stage_id = graph->z_stage_count; stage_id > 0; --stage_id) {
        const dSymLDLZStage *stage = &graph->z_stages[stage_id - 1];
        int_t count;
        if (!stage->active || stage->gid_count == 0)
            continue;
        count = symldl_cpu_product_to_int_t(
            stage->value_count, handle->nrhs,
            "SymLDL CPU sparse Z stage count overflows int_t.");
        if (handle->z_rank == stage->receiver_z)
            symldl_cpu_pack_z_stage(
                handle, stage, x, handle->z_send_values);
        symldl_cpu_z_exchange(handle, stage->receiver_z, stage->sender_z,
                              SYMLDL_CPU_Z_BCAST_TAG, count);
        if (handle->z_rank == stage->sender_z)
            symldl_cpu_unpack_z_stage(
                handle, stage, x, handle->z_recv_values, 0);
    }
    handle->sparse_broadcast_time += SuperLU_timer_() - start;
}

dSymLDLCPUSolveHandle *
dSymLDLCPUSolveCreate(const dSymLDLSolveGraph *graph, int nrhs_capacity,
                      dtrf3Dpartition_t *partition, gridinfo3d_t *grid3d)
{
    dSymLDLCPUSolveHandle *handle;
    gridinfo_t *grid;
    double start = SuperLU_timer_();
    if (graph == NULL || partition == NULL || grid3d == NULL ||
        nrhs_capacity <= 0 || partition->symV2DiagOwner == NULL ||
        partition->superGridMap == NULL)
        return NULL;

    handle = (dSymLDLCPUSolveHandle *) symldl_cpu_alloc(
        1, sizeof(*handle), "Malloc fails for SymLDL CPU solve handle.");
    memset(handle, 0, sizeof(*handle));
    handle->layer_comm = MPI_COMM_NULL;
    handle->z_comm = MPI_COMM_NULL;
    grid = &grid3d->grid2d;
    handle->graph = graph;
    handle->partition = partition;
    handle->nrhs_capacity = nrhs_capacity;
    handle->global_rank = grid3d->iam;
    handle->myrow = MYROW(grid->iam, grid);
    handle->mycol = MYCOL(grid->iam, grid);
#ifdef _OPENMP
    handle->max_threads = SUPERLU_MAX(1, omp_get_max_threads());
#else
    handle->max_threads = 1;
#endif
    if (MPI_Comm_dup(grid->comm, &handle->layer_comm) != MPI_SUCCESS)
        ABORT("SymLDL CPU solve could not duplicate the layer communicator.");
    if (MPI_Comm_dup(grid3d->zscp.comm, &handle->z_comm) != MPI_SUCCESS)
        ABORT("SymLDL CPU solve could not duplicate the Z communicator.");
    MPI_Comm_rank(handle->layer_comm, &handle->rank);
    MPI_Comm_size(handle->layer_comm, &handle->size);
    MPI_Comm_rank(handle->z_comm, &handle->z_rank);
    MPI_Comm_size(handle->z_comm, &handle->z_size);
    if (handle->z_rank != graph->z_rank || handle->z_size != graph->z_size)
        ABORT("SymLDL CPU solve Z communicator is inconsistent with its plan.");

    handle->panel_message_offsets = symldl_cpu_build_message_offsets(
        graph, graph->panel_gids, graph->panel_count, nrhs_capacity,
        &handle->panel_message_bytes);
    handle->row_message_offsets = symldl_cpu_build_message_offsets(
        graph, graph->row_gids, graph->row_count, nrhs_capacity,
        &handle->row_message_bytes);
    size_t message_bytes = SUPERLU_MAX(handle->panel_message_bytes,
                                       handle->row_message_bytes);
    handle->x_messages = (unsigned char *) symldl_cpu_alloc(
        SUPERLU_MAX((size_t) 1, message_bytes), 1,
        "Malloc fails for SymLDL CPU X messages.");
    handle->sum_messages = (unsigned char *) symldl_cpu_alloc(
        SUPERLU_MAX((size_t) 1, message_bytes), 1,
        "Malloc fails for SymLDL CPU partial messages.");

    handle->receive_stride = symldl_cpu_align(
        symldl_cpu_checked_sum(
            sizeof(dSymLDLCPUMessageHeader),
            symldl_cpu_checked_product(
                symldl_cpu_checked_product(
                    (size_t) graph->maxsup, (size_t) nrhs_capacity,
                    "SymLDL CPU receive workspace overflows."),
                sizeof(double),
                "SymLDL CPU receive workspace overflows."),
            "SymLDL CPU receive workspace overflows."),
        64, "SymLDL CPU receive workspace overflows.");
    size_t max_forward_receives = 0;
    size_t max_backward_receives = 0;
    size_t max_forward_sends = 0;
    size_t max_backward_sends = 0;
    for (int_t slot = 0; slot < graph->panel_count; ++slot) {
        max_forward_receives = symldl_cpu_checked_sum(
            max_forward_receives,
            graph->forward_bcast[slot].parent_rank >= 0,
            "SymLDL CPU receive request count overflows.");
        max_forward_sends = symldl_cpu_checked_sum(
            max_forward_sends, graph->forward_bcast[slot].child_count,
            "SymLDL CPU send request count overflows.");
        max_backward_receives = symldl_cpu_checked_sum(
            max_backward_receives, graph->backward_reduce[slot].child_count,
            "SymLDL CPU receive request count overflows.");
        max_backward_sends = symldl_cpu_checked_sum(
            max_backward_sends,
            graph->backward_reduce[slot].parent_rank >= 0,
            "SymLDL CPU send request count overflows.");
    }
    for (int_t slot = 0; slot < graph->row_count; ++slot) {
        max_forward_receives = symldl_cpu_checked_sum(
            max_forward_receives, graph->forward_reduce[slot].child_count,
            "SymLDL CPU receive request count overflows.");
        max_forward_sends = symldl_cpu_checked_sum(
            max_forward_sends,
            graph->forward_reduce[slot].parent_rank >= 0,
            "SymLDL CPU send request count overflows.");
        max_backward_receives = symldl_cpu_checked_sum(
            max_backward_receives,
            graph->backward_bcast[slot].parent_rank >= 0,
            "SymLDL CPU receive request count overflows.");
        max_backward_sends = symldl_cpu_checked_sum(
            max_backward_sends, graph->backward_bcast[slot].child_count,
            "SymLDL CPU send request count overflows.");
    }
    int max_receives = symldl_cpu_size_to_int(
        SUPERLU_MAX(max_forward_receives, max_backward_receives),
        "SymLDL CPU receive request count overflows int.");
    int max_sends = symldl_cpu_size_to_int(
        SUPERLU_MAX(max_forward_sends, max_backward_sends),
        "SymLDL CPU send request count overflows int.");
    int threaded_receive_depth = handle->max_threads <= INT_MAX / 2
                                     ? 2 * handle->max_threads : INT_MAX;
    handle->receive_depth = SUPERLU_MAX(
        1, SUPERLU_MIN(max_receives,
                       SUPERLU_MAX(8, threaded_receive_depth)));
    handle->send_capacity = SUPERLU_MAX(1, max_sends);
    handle->receive_messages = (unsigned char *) symldl_cpu_alloc(
        (size_t) handle->receive_depth, handle->receive_stride,
        "Malloc fails for SymLDL CPU receive ring.");
    handle->receive_requests = (MPI_Request *) symldl_cpu_alloc(
        (size_t) handle->receive_depth, sizeof(MPI_Request),
        "Malloc fails for SymLDL CPU receive requests.");
    handle->send_requests = (MPI_Request *) symldl_cpu_alloc(
        (size_t) handle->send_capacity, sizeof(MPI_Request),
        "Malloc fails for SymLDL CPU send requests.");
    handle->completed_receive_indices = (int *) symldl_cpu_alloc(
        (size_t) handle->receive_depth, sizeof(int),
        "Malloc fails for SymLDL CPU receive indices.");
    handle->completed_receive_statuses = (MPI_Status *) symldl_cpu_alloc(
        (size_t) handle->receive_depth, sizeof(MPI_Status),
        "Malloc fails for SymLDL CPU receive statuses.");

    int max_targets = SUPERLU_MAX(1, (int) SUPERLU_MAX(
        graph->panel_count, graph->row_count));
    int max_sources = max_targets;
    handle->dependencies = (int *) symldl_cpu_alloc(
        (size_t) max_targets, sizeof(int),
        "Malloc fails for SymLDL CPU dependencies.");
    handle->target_locks = (unsigned char *) symldl_cpu_alloc(
        (size_t) max_targets, 1, "Malloc fails for SymLDL CPU target locks.");
    handle->target_queued = (unsigned char *) symldl_cpu_alloc(
        (size_t) max_targets, 1,
        "Malloc fails for SymLDL CPU target readiness.");
    handle->target_terminal = (unsigned char *) symldl_cpu_alloc(
        (size_t) max_targets, 1,
        "Malloc fails for SymLDL CPU terminal state.");
    handle->received_child_mask = (unsigned char *) symldl_cpu_alloc(
        (size_t) max_targets, 1,
        "Malloc fails for SymLDL CPU receive state.");
    handle->source_published = (unsigned char *) symldl_cpu_alloc(
        (size_t) max_sources, 1,
        "Malloc fails for SymLDL CPU source state.");
    handle->task_ids = (int_t *) symldl_cpu_alloc(
        (size_t) SUPERLU_MAX((int_t) 1, graph->block_count), sizeof(int_t),
        "Malloc fails for SymLDL CPU task queue.");
    handle->completion_slots = (int_t *) symldl_cpu_alloc(
        (size_t) max_targets, sizeof(int_t),
        "Malloc fails for SymLDL CPU completion queue.");
    handle->completion_ready = (unsigned char *) symldl_cpu_alloc(
        (size_t) max_targets, 1,
        "Malloc fails for SymLDL CPU completion state.");
    size_t scratch_values = symldl_cpu_checked_product(
        symldl_cpu_checked_product(
            2, (size_t) handle->max_threads,
            "SymLDL CPU worker scratch overflows."),
        symldl_cpu_checked_product(
            (size_t) graph->maxsup, (size_t) nrhs_capacity,
            "SymLDL CPU worker scratch overflows."),
        "SymLDL CPU worker scratch overflows.");
    handle->worker_scratch = (double *) symldl_cpu_alloc(
        SUPERLU_MAX((size_t) 1, scratch_values), sizeof(double),
        "Malloc fails for SymLDL CPU worker scratch.");
    handle->worker_compute_times = (double *) symldl_cpu_alloc(
        (size_t) handle->max_threads, sizeof(double),
        "Malloc fails for SymLDL CPU worker timers.");
    symldl_cpu_build_z_workspace(handle);
    symldl_cpu_replicate_final_panels(handle);
    handle->setup_time = SuperLU_timer_() - start;
    return handle;
}

int
dSymLDLCPUForward(dSymLDLCPUSolveHandle *handle, double *x,
                  int_t x_count, int nrhs)
{
    if (handle == NULL)
        return -1;
    handle->x = x;
    handle->x_count = x_count;
    handle->nrhs = nrhs;
    symldl_cpu_initialize_rhs(handle, x, x_count, nrhs);
    if (symldl_cpu_run_phase(
            handle, SYMLDL_CPU_PHASE_FORWARD, x, x_count, nrhs) != 0)
        return -1;
    symldl_cpu_sparse_z(handle, x);
    return 0;
}

int
dSymLDLCPUDiagonal(dSymLDLCPUSolveHandle *handle, double *x,
                   int_t x_count, int nrhs)
{
    if (handle == NULL || x == NULL || nrhs <= 0 ||
        nrhs > handle->nrhs_capacity)
        return -1;
    handle->x = x;
    handle->x_count = x_count;
    handle->nrhs = nrhs;
    double start = SuperLU_timer_();

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int_t slot = 0; slot < handle->graph->panel_count; ++slot) {
        const dSymLDLPanelDesc *panel = &handle->graph->panels[slot];
        if (handle->myrow != handle->graph->diag_roots[panel->gid] ||
            handle->mycol != handle->graph->panel_roots[panel->gid])
            continue;
        if (!panel->active || !panel->has_diag)
            ABORT("SymLDL CPU diagonal owner has no inverse diagonal block.");
        int_t row_slot = handle->graph->row_local_index[panel->gid];
        if (row_slot < 0)
            ABORT("SymLDL CPU diagonal owner has no local X row.");
        int thread_id = 0;
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
#endif
        size_t stride = (size_t) handle->graph->maxsup *
                        handle->nrhs_capacity;
        double *input = handle->worker_scratch +
                        (size_t) (2 * thread_id) * stride;
        double *output = input + stride;
        int count = symldl_cpu_product_to_int(
            panel->width, nrhs,
            "SymLDL CPU diagonal count overflows int.");
        double *xk = x + symldl_cpu_x_offset(handle, row_slot);
        memcpy(input, xk, (size_t) count * sizeof(double));
        dSymLDLCPUDiagonalBlock(panel, nrhs, input, output);
        memcpy(xk, output, (size_t) count * sizeof(double));
    }
    handle->diagonal_time += SuperLU_timer_() - start;
    return 0;
}

int
dSymLDLCPUBackward(dSymLDLCPUSolveHandle *handle, double *x,
                   int_t x_count, int nrhs)
{
    if (handle == NULL)
        return -1;
    return symldl_cpu_run_phase(
        handle, SYMLDL_CPU_PHASE_BACKWARD, x, x_count, nrhs);
}

void
dSymLDLCPUSolveTakeTimers(dSymLDLCPUSolveHandle *handle, double *setup,
                          double *forward, double *diagonal,
                          double *backward, double *mpi_progress,
                          double *numeric_compute)
{
    if (handle == NULL)
        return;
    if (setup) *setup = handle->setup_time;
    if (forward) *forward = handle->forward_time +
                            handle->rhs_distribution_time +
                            handle->sparse_reduce_time +
                            handle->sparse_broadcast_time;
    if (diagonal) *diagonal = handle->diagonal_time;
    if (backward) *backward = handle->backward_time;
    if (mpi_progress) *mpi_progress = handle->mpi_progress_time;
    if (numeric_compute) *numeric_compute = handle->numeric_compute_time;
    handle->setup_time = 0.0;
    handle->forward_time = 0.0;
    handle->diagonal_time = 0.0;
    handle->backward_time = 0.0;
    handle->mpi_progress_time = 0.0;
    handle->numeric_compute_time = 0.0;
    handle->rhs_distribution_time = 0.0;
    handle->sparse_reduce_time = 0.0;
    handle->sparse_broadcast_time = 0.0;
    handle->panel_replication_time = 0.0;
}

void
dSymLDLCPUSolveTakeCommStats(dSymLDLCPUSolveHandle *handle,
                             dSymLDLCPUSolveCommStats *stats)
{
    if (handle == NULL || stats == NULL)
        return;
    *stats = handle->comm_stats;
    memset(&handle->comm_stats, 0, sizeof(handle->comm_stats));
}

void
dSymLDLCPUSolveDestroy(dSymLDLCPUSolveHandle *handle)
{
    if (handle == NULL)
        return;
    if (handle->z_recv_values) SUPERLU_FREE(handle->z_recv_values);
    if (handle->z_send_values) SUPERLU_FREE(handle->z_send_values);
    if (handle->rhs_values) SUPERLU_FREE(handle->rhs_values);
    if (handle->rhs_offsets) SUPERLU_FREE(handle->rhs_offsets);
    if (handle->rhs_gids) SUPERLU_FREE(handle->rhs_gids);
    if (handle->z_comm != MPI_COMM_NULL)
        MPI_Comm_free(&handle->z_comm);
    if (handle->layer_comm != MPI_COMM_NULL)
        MPI_Comm_free(&handle->layer_comm);
    if (handle->worker_compute_times) SUPERLU_FREE(handle->worker_compute_times);
    if (handle->worker_scratch) SUPERLU_FREE(handle->worker_scratch);
    if (handle->completion_ready) SUPERLU_FREE(handle->completion_ready);
    if (handle->completion_slots) SUPERLU_FREE(handle->completion_slots);
    if (handle->task_ids) SUPERLU_FREE(handle->task_ids);
    if (handle->source_published) SUPERLU_FREE(handle->source_published);
    if (handle->received_child_mask) SUPERLU_FREE(handle->received_child_mask);
    if (handle->target_terminal) SUPERLU_FREE(handle->target_terminal);
    if (handle->target_queued) SUPERLU_FREE(handle->target_queued);
    if (handle->target_locks) SUPERLU_FREE(handle->target_locks);
    if (handle->dependencies) SUPERLU_FREE(handle->dependencies);
    if (handle->completed_receive_statuses) SUPERLU_FREE(handle->completed_receive_statuses);
    if (handle->completed_receive_indices) SUPERLU_FREE(handle->completed_receive_indices);
    if (handle->send_requests) SUPERLU_FREE(handle->send_requests);
    if (handle->receive_requests) SUPERLU_FREE(handle->receive_requests);
    if (handle->receive_messages) SUPERLU_FREE(handle->receive_messages);
    if (handle->sum_messages) SUPERLU_FREE(handle->sum_messages);
    if (handle->x_messages) SUPERLU_FREE(handle->x_messages);
    if (handle->row_message_offsets) SUPERLU_FREE(handle->row_message_offsets);
    if (handle->panel_message_offsets) SUPERLU_FREE(handle->panel_message_offsets);
    SUPERLU_FREE(handle);
}
