#include "dsymldl_v2_runtime_model.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The model intentionally has no machine-rate coefficients. For each factor
   level/lookahead window it accumulates structural resource exposure by rank
   (and by physical node for communication), then records:

     total    aggregate work or movement over the factorization,
     critical sum of per-window maximum endpoint/node exposure,
     waiting  rank-idle exposure relative to the per-window maximum.

   Selection treats total and critical as monotone costs. Waiting is reported
   as an imbalance diagnostic; it is not itself minimized because reducing
   useful work can increase this derived quantity. */
typedef struct {
    const dSymLDLV2GridModelInput *input;
    const dSymLDLV2PartitionPlan *plan;
    dSymLDLV2PerformanceEstimate *performance;
    size_t ranks;
    size_t nodes;
    double *rank_value;
    double *node_value;
} dSymLDLV2WindowAccumulator;

static void dSymLDLV2SetRuntimeError(char *error, size_t error_size,
                                     const char *message)
{
    if (error != NULL && error_size > 0)
    {
        snprintf(error, error_size, "%s", message);
        error[error_size - 1] = '\0';
    }
}

static int dSymLDLV2RuntimeAllocationFits(size_t count, size_t size)
{
    return size == 0 || count <= SIZE_MAX / size;
}

static int dSymLDLV2RuntimeForestState(
    const dSymLDLV2PartitionPlan *plan, int z, int_t target_tree)
{
    int_t tree = plan->pz - 1 + z;
    for (int_t level = 0; level < plan->max_z_levels; ++level)
    {
        if (tree == target_tree)
            return z % ((int) 1 << level) == 0 ? 1 : 2;
        if (level + 1 < plan->max_z_levels)
            tree = (tree - 1) / 2;
    }
    return 0;
}

int dSymLDLV2RuntimeRankForCoordinates(
    dSymLDLV2RankOrder rank_order, int pr_count, int pc_count, int pz_count,
    int pr, int pc, int pz)
{
    if (pr_count <= 0 || pc_count <= 0 || pz_count <= 0 ||
        pr < 0 || pr >= pr_count || pc < 0 || pc >= pc_count ||
        pz < 0 || pz >= pz_count)
        return -1;
    long long rank;
    if (rank_order == DSYMLDL_V2_RANK_ORDER_XY_MAJOR)
        rank = ((long long) pr * pc_count + pc) * pz_count + pz;
    else
        rank = (long long) pz * pr_count * pc_count +
               (long long) pr * pc_count + pc;
    long long count = (long long) pr_count * pc_count * pz_count;
    return rank >= 0 && rank < count && rank <= INT_MAX
               ? (int) rank
               : -1;
}

static int dSymLDLV2RuntimeRank(const dSymLDLV2GridModelInput *input,
                                const dSymLDLV2PartitionPlan *plan,
                                int pr, int pc, int pz)
{
    return dSymLDLV2RuntimeRankForCoordinates(
        input->runtime.rank_order, plan->pr, plan->pc, plan->pz,
        pr, pc, pz);
}

static int dSymLDLV2RuntimeNode(const dSymLDLV2WindowAccumulator *acc,
                                int rank)
{
    const dSymLDLV2GridTopology *topology = acc->input->topology;
    if (topology == NULL || !topology->topology_known ||
        topology->node_of_rank == NULL || rank < 0 ||
        rank >= topology->communicator_size)
        return -1;
    int node = topology->node_of_rank[rank];
    return node >= 0 && (size_t) node < acc->nodes ? node : -1;
}

static double *dSymLDLV2RuntimeRankValue(
    dSymLDLV2WindowAccumulator *acc, int metric, int rank)
{
    return &acc->rank_value[(size_t) metric * acc->ranks + (size_t) rank];
}

static double *dSymLDLV2RuntimeNodeValue(
    dSymLDLV2WindowAccumulator *acc, int metric, int node)
{
    return &acc->node_value[(size_t) metric * acc->nodes + (size_t) node];
}

static void dSymLDLV2RuntimeAddLocal(
    dSymLDLV2WindowAccumulator *acc,
    dSymLDLV2RuntimeMetricKind metric, int rank, double value)
{
    if (value <= 0.0 || rank < 0 || (size_t) rank >= acc->ranks)
        return;
    *dSymLDLV2RuntimeRankValue(acc, metric, rank) += value;
    acc->performance->metric[metric].total += value;
}

static void dSymLDLV2RuntimeAddComm(
    dSymLDLV2WindowAccumulator *acc, int source, int destination,
    double bytes)
{
    if (bytes <= 0.0 || source < 0 || destination < 0 ||
        source == destination || (size_t) source >= acc->ranks ||
        (size_t) destination >= acc->ranks)
        return;
    int source_node = dSymLDLV2RuntimeNode(acc, source);
    int destination_node = dSymLDLV2RuntimeNode(acc, destination);
    int intra = source_node >= 0 && source_node == destination_node;
    dSymLDLV2RuntimeMetricKind message_metric =
        intra ? DSYMLDL_V2_RUNTIME_INTRA_NODE_MESSAGES
              : DSYMLDL_V2_RUNTIME_INTER_NODE_MESSAGES;
    dSymLDLV2RuntimeMetricKind byte_metric =
        intra ? DSYMLDL_V2_RUNTIME_INTRA_NODE_BYTES
              : DSYMLDL_V2_RUNTIME_INTER_NODE_BYTES;

    *dSymLDLV2RuntimeRankValue(acc, message_metric, source) += 1.0;
    *dSymLDLV2RuntimeRankValue(acc, message_metric, destination) += 1.0;
    *dSymLDLV2RuntimeRankValue(acc, byte_metric, source) += bytes;
    *dSymLDLV2RuntimeRankValue(acc, byte_metric, destination) += bytes;
    acc->performance->metric[message_metric].total += 1.0;
    acc->performance->metric[byte_metric].total += bytes;
    uint64_t payload = bytes >= (double) UINT64_MAX
                           ? UINT64_MAX : (uint64_t) ceil(bytes);
    int payload_bin = 0;
    while (payload_bin + 1 < DSYMLDL_V2_COMM_SIZE_HISTOGRAM_BINS &&
           payload > (UINT64_C(1) << payload_bin))
        ++payload_bin;
    ++acc->performance->communication_size_histogram[payload_bin];

    if (source_node >= 0)
    {
        *dSymLDLV2RuntimeNodeValue(acc, message_metric, source_node) += 1.0;
        *dSymLDLV2RuntimeNodeValue(acc, byte_metric, source_node) += bytes;
    }
    if (destination_node >= 0 && destination_node != source_node)
    {
        *dSymLDLV2RuntimeNodeValue(acc, message_metric,
                                  destination_node) += 1.0;
        *dSymLDLV2RuntimeNodeValue(acc, byte_metric,
                                  destination_node) += bytes;
    }

    if (acc->input->runtime.gpu_offload &&
        !acc->input->runtime.cuda_aware_mpi)
    {
        dSymLDLV2RuntimeAddLocal(
            acc, DSYMLDL_V2_RUNTIME_DEVICE_TO_HOST_BYTES, source, bytes);
        dSymLDLV2RuntimeAddLocal(
            acc, DSYMLDL_V2_RUNTIME_HOST_TO_DEVICE_BYTES, destination, bytes);
    }
}

static void dSymLDLV2RuntimeFinishWindow(dSymLDLV2WindowAccumulator *acc)
{
    for (int metric = 0; metric < DSYMLDL_V2_RUNTIME_METRIC_COUNT;
         ++metric)
    {
        dSymLDLV2RuntimeMetric *result = &acc->performance->metric[metric];
        if (!result->active)
            continue;
        double rank_sum = 0.0;
        double rank_maximum = 0.0;
        for (size_t rank = 0; rank < acc->ranks; ++rank)
        {
            double value = *dSymLDLV2RuntimeRankValue(acc, metric, rank);
            rank_sum += value;
            rank_maximum = fmax(rank_maximum, value);
        }
        double critical = rank_maximum;
        if (metric == DSYMLDL_V2_RUNTIME_INTRA_NODE_MESSAGES ||
            metric == DSYMLDL_V2_RUNTIME_INTRA_NODE_BYTES ||
            metric == DSYMLDL_V2_RUNTIME_INTER_NODE_MESSAGES ||
            metric == DSYMLDL_V2_RUNTIME_INTER_NODE_BYTES)
        {
            for (size_t node = 0; node < acc->nodes; ++node)
                critical = fmax(
                    critical,
                    *dSymLDLV2RuntimeNodeValue(acc, metric, (int) node));
        }
        result->critical += critical;
        if (rank_maximum > 0.0)
            result->waiting += fmax(
                0.0, rank_maximum * (double) acc->ranks - rank_sum);
    }
    memset(acc->rank_value, 0,
           DSYMLDL_V2_RUNTIME_METRIC_COUNT * acc->ranks * sizeof(double));
    memset(acc->node_value, 0,
           DSYMLDL_V2_RUNTIME_METRIC_COUNT * acc->nodes * sizeof(double));
}

static int dSymLDLV2RuntimeAddPanel(
    dSymLDLV2WindowAccumulator *acc, int_t k, int z,
    double *partner_values, double *row_values,
    double *suffix_rows, double *suffix_blocks, unsigned char *prefix_pc,
    char *error, size_t error_size)
{
    const dSymLDLV2GridModelInput *input = acc->input;
    const dSymLDLV2StructuralSummary *structure = input->structure;
    const dSymLDLV2PartitionPlanInput *partition = input->partition_input;
    const dSymLDLV2PartitionPlan *plan = acc->plan;
    int_t begin = structure->panel_block_offsets[k];
    int_t end = structure->panel_block_offsets[k + 1];
    int_t columns = partition->xsup[k + 1] - partition->xsup[k];
    int panel_root = plan->panel_root[k];
    int diag_root = plan->diag_root[k];
    int diagonal_rank = dSymLDLV2RuntimeRank(
        input, plan, diag_root, panel_root, z);
    double value_bytes = sizeof(double);
    double diagonal_values = (double) columns * (double) columns;
    /* Dense symmetric diagonal factorization is cubic in the supernode. */
    double diagonal_flops = (double) columns * (double) columns *
                            (double) columns;
    dSymLDLV2RuntimeMetricKind compute_metric =
        input->runtime.gpu_offload ? DSYMLDL_V2_RUNTIME_GPU_FLOPS
                                   : DSYMLDL_V2_RUNTIME_CPU_FLOPS;
    dSymLDLV2RuntimeMetricKind compute_byte_metric =
        input->runtime.gpu_offload
            ? DSYMLDL_V2_RUNTIME_GPU_LOCAL_BYTES
            : DSYMLDL_V2_RUNTIME_CPU_LOCAL_BYTES;

    if (columns <= 0 || begin < 0 || end < begin ||
        end > structure->block_count || diagonal_rank < 0)
    {
        dSymLDLV2SetRuntimeError(error, error_size,
                                 "SymLDL runtime panel metadata is invalid.");
        return 0;
    }

    dSymLDLV2RuntimeAddLocal(acc, DSYMLDL_V2_RUNTIME_CPU_FLOPS,
                            diagonal_rank, diagonal_flops);
    dSymLDLV2RuntimeAddLocal(acc, DSYMLDL_V2_RUNTIME_CPU_LOCAL_BYTES,
                            diagonal_rank, 2.0 * diagonal_values * value_bytes);
    dSymLDLV2RuntimeAddLocal(acc, DSYMLDL_V2_RUNTIME_TASK_LAUNCHES,
                            diagonal_rank, input->runtime.gpu_offload ? 0.0 : 1.0);
    acc->performance->total_factor_flops += diagonal_flops;

    double diagonal_message_bytes = 2.0 * diagonal_values * value_bytes;
    for (int pr = 0; pr < plan->pr; ++pr)
    {
        int destination = dSymLDLV2RuntimeRank(
            input, plan, pr, panel_root, z);
        dSymLDLV2RuntimeAddComm(acc, diagonal_rank, destination,
                               diagonal_message_bytes);
    }
    if (plan->pr == 1 && plan->pc > 1)
    {
        for (int pc = 0; pc < plan->pc; ++pc)
        {
            int destination = dSymLDLV2RuntimeRank(input, plan, 0, pc, z);
            dSymLDLV2RuntimeAddComm(acc, diagonal_rank, destination,
                                   diagonal_values * value_bytes);
        }
    }

    memset(partner_values, 0,
           (size_t) plan->pr * (size_t) plan->pc * sizeof(double));
    memset(row_values, 0,
           (size_t) plan->pr * (size_t) plan->pc * sizeof(double));
    memset(prefix_pc, 0, (size_t) plan->pc * sizeof(unsigned char));

    for (int_t block = begin; block < end; ++block)
    {
        int_t row = structure->block_row_supernode[block];
        int_t rows = structure->block_row_count[block];
        if (row < k || row >= structure->nsupers || rows <= 0)
        {
            dSymLDLV2SetRuntimeError(error, error_size,
                                     "SymLDL runtime block metadata is invalid.");
            return 0;
        }
        if (row == k)
            continue;
        int pr = plan->diag_root[row];
        int pc = plan->panel_root[row];
        int owner = dSymLDLV2RuntimeRank(input, plan, pr, panel_root, z);
        double values = (double) rows * (double) columns;
        /* Apply the dense diagonal transform: 2 m_i s_k^2 FLOPs. */
        double transform_flops = 2.0 * values * (double) columns;
        dSymLDLV2RuntimeAddLocal(acc, compute_metric, owner,
                                transform_flops);
        dSymLDLV2RuntimeAddLocal(
            acc, compute_byte_metric, owner,
            (2.0 * values + diagonal_values) * value_bytes);
        dSymLDLV2RuntimeAddLocal(acc, DSYMLDL_V2_RUNTIME_TASK_LAUNCHES,
                                owner, input->runtime.gpu_offload ? 1.0 : 0.0);
        partner_values[(size_t) pr * (size_t) plan->pc + (size_t) pc] +=
            values;
        prefix_pc[pc] = 1;
        for (int destination_pc = 0; destination_pc < plan->pc;
             ++destination_pc)
            if (prefix_pc[destination_pc])
                row_values[(size_t) pr * (size_t) plan->pc +
                           (size_t) destination_pc] += values;
        acc->performance->total_factor_flops += transform_flops;
    }

    for (int source_pr = 0; source_pr < plan->pr; ++source_pr)
    {
        int source = dSymLDLV2RuntimeRank(
            input, plan, source_pr, panel_root, z);
        for (int destination_pc = 0; destination_pc < plan->pc;
             ++destination_pc)
        {
            double values = partner_values[
                (size_t) source_pr * (size_t) plan->pc +
                (size_t) destination_pc];
            if (values <= 0.0)
                continue;
            dSymLDLV2RuntimeAddLocal(
                acc, compute_byte_metric, source,
                2.0 * values * value_bytes);
            dSymLDLV2RuntimeAddLocal(
                acc, DSYMLDL_V2_RUNTIME_TASK_LAUNCHES, source,
                input->runtime.gpu_offload ? 1.0 : 0.0);
            for (int destination_pr = 0; destination_pr < plan->pr;
                 ++destination_pr)
            {
                int destination = dSymLDLV2RuntimeRank(
                    input, plan, destination_pr, destination_pc, z);
                dSymLDLV2RuntimeAddComm(
                    acc, source, destination, values * value_bytes);
                if (destination != source)
                    dSymLDLV2RuntimeAddLocal(
                        acc, DSYMLDL_V2_RUNTIME_TASK_LAUNCHES,
                        destination,
                        input->runtime.gpu_offload ? 1.0 : 0.0);
            }
        }
    }

    if (plan->pc > 1)
    {
        for (int pr = 0; pr < plan->pr; ++pr)
        {
            int source = dSymLDLV2RuntimeRank(input, plan, pr, panel_root, z);
            for (int pc = 0; pc < plan->pc; ++pc)
            {
                double values = row_values[
                    (size_t) pr * (size_t) plan->pc + (size_t) pc];
                if (values <= 0.0)
                    continue;
                int destination = dSymLDLV2RuntimeRank(input, plan, pr, pc, z);
                dSymLDLV2RuntimeAddLocal(
                    acc, compute_byte_metric, source,
                    2.0 * values * value_bytes);
                dSymLDLV2RuntimeAddLocal(
                    acc, DSYMLDL_V2_RUNTIME_TASK_LAUNCHES, source,
                    input->runtime.gpu_offload ? 1.0 : 0.0);
                dSymLDLV2RuntimeAddComm(acc, source, destination,
                                       values * value_bytes);
            }
        }
    }

    memset(suffix_rows, 0, (size_t) plan->pr * sizeof(double));
    memset(suffix_blocks, 0, (size_t) plan->pr * sizeof(double));
    for (int_t block = end; block-- > begin; )
    {
        int_t row = structure->block_row_supernode[block];
        int_t rows = structure->block_row_count[block];
        if (row == k)
            continue;
        int pr = plan->diag_root[row];
        int pc = plan->panel_root[row];
        suffix_rows[pr] += (double) rows;
        suffix_blocks[pr] += 1.0;
        for (int output_pr = 0; output_pr < plan->pr; ++output_pr)
        {
            if (suffix_blocks[output_pr] <= 0.0)
                continue;
            int owner = dSymLDLV2RuntimeRank(
                input, plan, output_pr, pc, z);
            /* Sum 2 m_i m_j s_k over the real lower-envelope block pairs.
               The reverse suffix avoids materializing a Cartesian task DAG. */
            double flops = 2.0 * (double) rows * (double) columns *
                           suffix_rows[output_pr];
            double bytes = value_bytes *
                ((double) columns * suffix_rows[output_pr] +
                 (double) rows * (double) columns * suffix_blocks[output_pr] +
                 2.0 * (double) rows * suffix_rows[output_pr]);
            dSymLDLV2RuntimeAddLocal(acc, compute_metric, owner, flops);
            dSymLDLV2RuntimeAddLocal(acc, compute_byte_metric,
                                    owner, bytes);
            dSymLDLV2RuntimeAddLocal(
                acc, DSYMLDL_V2_RUNTIME_TASK_LAUNCHES, owner,
                input->runtime.gpu_offload ? suffix_blocks[output_pr] : 0.0);
            acc->performance->total_factor_flops += flops;
        }
    }

    if (plan->pr * plan->pc > 1)
        dSymLDLV2RuntimeAddLocal(
            acc, input->runtime.gpu_offload
                     ? DSYMLDL_V2_RUNTIME_GPU_SYNCHRONIZATIONS
                     : DSYMLDL_V2_RUNTIME_PROCESS_SYNCHRONIZATIONS,
            diagonal_rank, 1.0);
    if (plan->pr > 1 || plan->pc > 1)
        dSymLDLV2RuntimeAddLocal(
            acc, DSYMLDL_V2_RUNTIME_PROCESS_SYNCHRONIZATIONS,
            diagonal_rank, 1.0);
    return 1;
}

static int dSymLDLV2RuntimeAddAncestorReduction(
    dSymLDLV2WindowAccumulator *acc, int_t k, int active_z,
    int participating_layers, const double *partner_values,
    char *error, size_t error_size)
{
    const dSymLDLV2GridModelInput *input = acc->input;
    const dSymLDLV2PartitionPlan *plan = acc->plan;
    int_t columns = input->partition_input->xsup[k + 1] -
                    input->partition_input->xsup[k];
    double value_bytes = sizeof(double);
    dSymLDLV2RuntimeMetricKind compute_metric =
        input->runtime.gpu_offload ? DSYMLDL_V2_RUNTIME_GPU_FLOPS
                                   : DSYMLDL_V2_RUNTIME_CPU_FLOPS;
    dSymLDLV2RuntimeMetricKind compute_byte_metric =
        input->runtime.gpu_offload
            ? DSYMLDL_V2_RUNTIME_GPU_LOCAL_BYTES
            : DSYMLDL_V2_RUNTIME_CPU_LOCAL_BYTES;

    if (participating_layers <= 1)
        return 1;
    if (active_z < 0 || active_z + participating_layers > plan->pz ||
        (participating_layers & (participating_layers - 1)) != 0 ||
        active_z % participating_layers != 0 || columns <= 0)
    {
        dSymLDLV2SetRuntimeError(
            error, error_size,
            "SymLDL runtime ancestor-reduction metadata is invalid.");
        return 0;
    }

    /* Each depth layer holds partial updates for ancestor panels, but only the
       group root factors the ancestor forest. Mirror ancestorReduction3dGPU:
       combine those partial lower panels along a binary tree. Every process
       row on the panel-root process column sends its local panel segment. */
    for (int stride = 1; stride < participating_layers; stride <<= 1)
    {
        for (int receiver_z = active_z;
             receiver_z < active_z + participating_layers;
             receiver_z += 2 * stride)
        {
            int sender_z = receiver_z + stride;
            for (int pr = 0; pr < plan->pr; ++pr)
            {
                double values = pr == plan->diag_root[k]
                                    ? (double) columns * (double) columns
                                    : 0.0;
                for (int pc = 0; pc < plan->pc; ++pc)
                    values += partner_values[
                        (size_t) pr * (size_t) plan->pc + (size_t) pc];
                if (values <= 0.0)
                    continue;

                int source = dSymLDLV2RuntimeRank(
                    input, plan, pr, plan->panel_root[k], sender_z);
                int destination = dSymLDLV2RuntimeRank(
                    input, plan, pr, plan->panel_root[k], receiver_z);
                dSymLDLV2RuntimeAddComm(
                    acc, source, destination, values * value_bytes);
                dSymLDLV2RuntimeAddLocal(
                    acc, compute_metric, destination, 2.0 * values);
                dSymLDLV2RuntimeAddLocal(
                    acc, compute_byte_metric, destination,
                    3.0 * values * value_bytes);
                dSymLDLV2RuntimeAddLocal(
                    acc, DSYMLDL_V2_RUNTIME_TASK_LAUNCHES, destination,
                    input->runtime.gpu_offload ? 2.0 : 0.0);
            }
        }
    }
    return 1;
}

int dSymLDLV2EvaluateRuntimeModel(
    const dSymLDLV2GridModelInput *input,
    const dSymLDLV2PartitionPlan *plan,
    int requested_streams, int estimated_streams,
    dSymLDLV2PerformanceEstimate *performance,
    char *error, size_t error_size)
{
    dSymLDLV2WindowAccumulator acc;
    double *partner_values = NULL;
    double *row_values = NULL;
    double *suffix_rows = NULL;
    double *suffix_blocks = NULL;
    unsigned char *prefix_pc = NULL;
    int success = 0;

    if (input == NULL || input->partition_input == NULL ||
        input->structure == NULL || plan == NULL || performance == NULL ||
        plan->pr <= 0 || plan->pc <= 0 || plan->pz <= 0 ||
        requested_streams <= 0 || estimated_streams <= 0)
    {
        dSymLDLV2SetRuntimeError(error, error_size,
                                 "SymLDL runtime model input is invalid.");
        return 0;
    }

    memset(performance, 0, sizeof(*performance));
    performance->requested_gpu_streams = requested_streams;
    performance->estimated_gpu_streams = estimated_streams;
    performance->pareto_dominator = -1;
    performance->metric[DSYMLDL_V2_RUNTIME_CPU_FLOPS].active = 1;
    performance->metric[DSYMLDL_V2_RUNTIME_CPU_LOCAL_BYTES].active = 1;
    performance->metric[
        DSYMLDL_V2_RUNTIME_PROCESS_SYNCHRONIZATIONS].active = 1;
    performance->metric[DSYMLDL_V2_RUNTIME_INTRA_NODE_MESSAGES].active = 1;
    performance->metric[DSYMLDL_V2_RUNTIME_INTRA_NODE_BYTES].active = 1;
    performance->metric[DSYMLDL_V2_RUNTIME_INTER_NODE_MESSAGES].active = 1;
    performance->metric[DSYMLDL_V2_RUNTIME_INTER_NODE_BYTES].active = 1;
    if (input->runtime.gpu_offload)
    {
        performance->metric[DSYMLDL_V2_RUNTIME_GPU_FLOPS].active = 1;
        performance->metric[DSYMLDL_V2_RUNTIME_GPU_LOCAL_BYTES].active = 1;
        performance->metric[DSYMLDL_V2_RUNTIME_TASK_LAUNCHES].active = 1;
        performance->metric[
            DSYMLDL_V2_RUNTIME_GPU_SYNCHRONIZATIONS].active = 1;
        if (!input->runtime.cuda_aware_mpi)
        {
            performance->metric[
                DSYMLDL_V2_RUNTIME_HOST_TO_DEVICE_BYTES].active = 1;
            performance->metric[
                DSYMLDL_V2_RUNTIME_DEVICE_TO_HOST_BYTES].active = 1;
        }
    }

    memset(&acc, 0, sizeof(acc));
    acc.input = input;
    acc.plan = plan;
    acc.performance = performance;
    if ((size_t) plan->pr > SIZE_MAX / (size_t) plan->pc ||
        (size_t) plan->pr * (size_t) plan->pc >
            SIZE_MAX / (size_t) plan->pz)
    {
        dSymLDLV2SetRuntimeError(error, error_size,
                                 "SymLDL runtime rank count overflows.");
        goto cleanup;
    }
    acc.ranks = (size_t) plan->pr * (size_t) plan->pc * (size_t) plan->pz;
    acc.nodes = input->topology != NULL &&
                        input->topology->topology_known &&
                        input->topology->node_count > 0
                    ? (size_t) input->topology->node_count
                    : 1;
    if (acc.ranks > SIZE_MAX / DSYMLDL_V2_RUNTIME_METRIC_COUNT ||
        acc.nodes > SIZE_MAX / DSYMLDL_V2_RUNTIME_METRIC_COUNT ||
        (input->topology != NULL && input->topology->topology_known &&
         (input->topology->communicator_size != (int) acc.ranks ||
          input->topology->rank_order != input->runtime.rank_order)))
    {
        dSymLDLV2SetRuntimeError(
            error, error_size,
            "SymLDL runtime topology does not match the process grid.");
        goto cleanup;
    }
    size_t rank_values = DSYMLDL_V2_RUNTIME_METRIC_COUNT * acc.ranks;
    size_t node_values = DSYMLDL_V2_RUNTIME_METRIC_COUNT * acc.nodes;
    size_t partner_slots = (size_t) plan->pr * (size_t) plan->pc;
    if (!dSymLDLV2RuntimeAllocationFits(rank_values, sizeof(double)) ||
        !dSymLDLV2RuntimeAllocationFits(node_values, sizeof(double)) ||
        !dSymLDLV2RuntimeAllocationFits(partner_slots, sizeof(double)) ||
        !dSymLDLV2RuntimeAllocationFits((size_t) plan->pr, sizeof(double)) ||
        !dSymLDLV2RuntimeAllocationFits((size_t) plan->pc,
                                        sizeof(unsigned char)))
    {
        dSymLDLV2SetRuntimeError(error, error_size,
                                 "SymLDL runtime model allocation overflows.");
        goto cleanup;
    }
    acc.rank_value = (double *) calloc(rank_values, sizeof(double));
    acc.node_value = (double *) calloc(node_values, sizeof(double));
    partner_values = (double *) calloc(partner_slots, sizeof(double));
    row_values = (double *) calloc(partner_slots, sizeof(double));
    suffix_rows = (double *) calloc((size_t) plan->pr, sizeof(double));
    suffix_blocks = (double *) calloc((size_t) plan->pr, sizeof(double));
    prefix_pc = (unsigned char *) calloc((size_t) plan->pc,
                                         sizeof(unsigned char));
    if (acc.rank_value == NULL || acc.node_value == NULL ||
        partner_values == NULL || row_values == NULL ||
        suffix_rows == NULL || suffix_blocks == NULL || prefix_pc == NULL)
    {
        dSymLDLV2SetRuntimeError(error, error_size,
                                 "SymLDL runtime model allocation failed.");
        goto cleanup;
    }

    int_t window_depth = (int_t) input->runtime.lookahead_depth;
    if (window_depth < 1)
        window_depth = 1;
    if (input->runtime.gpu_offload && window_depth > estimated_streams)
        window_depth = estimated_streams;

    for (int_t level = 0; level < plan->factor_level_count; ++level)
    {
        int_t level_begin = plan->factor_level_ptr[level];
        int_t level_end = plan->factor_level_ptr[level + 1];
        if (level_begin < 0 || level_end < level_begin ||
            level_end > input->structure->nsupers)
        {
            dSymLDLV2SetRuntimeError(error, error_size,
                                     "SymLDL runtime factor level is invalid.");
            goto cleanup;
        }
        for (int_t begin = level_begin; begin < level_end; )
        {
            int_t end = level_end - begin > window_depth
                            ? begin + window_depth
                            : level_end;
            for (int_t position = begin; position < end; ++position)
            {
                int_t k = plan->factor_nodes[position];
                int_t tree = plan->forest_of_supernode[k];
                int active_z = -1;
                int participating_layers = 0;
                for (int z = 0; z < plan->pz; ++z)
                {
                    int state = dSymLDLV2RuntimeForestState(plan, z, tree);
                    if (state != 0)
                        ++participating_layers;
                    if (state == 1)
                    {
                        if (active_z >= 0)
                        {
                            dSymLDLV2SetRuntimeError(
                                error, error_size,
                                "SymLDL runtime forest has multiple active layers.");
                            goto cleanup;
                        }
                        active_z = z;
                    }
                }
                if (active_z < 0)
                {
                    dSymLDLV2SetRuntimeError(
                        error, error_size,
                        "SymLDL runtime model cannot locate the active forest layer.");
                    goto cleanup;
                }
                if (!dSymLDLV2RuntimeAddPanel(
                        &acc, k, active_z, partner_values,
                        row_values, suffix_rows, suffix_blocks, prefix_pc,
                        error, error_size) ||
                    !dSymLDLV2RuntimeAddAncestorReduction(
                        &acc, k, active_z, participating_layers,
                        partner_values, error, error_size))
                    goto cleanup;
            }
            dSymLDLV2RuntimeFinishWindow(&acc);
            begin = end;
        }
    }
    success = 1;

cleanup:
    free(prefix_pc);
    free(suffix_blocks);
    free(suffix_rows);
    free(row_values);
    free(partner_values);
    free(acc.node_value);
    free(acc.rank_value);
    return success;
}
