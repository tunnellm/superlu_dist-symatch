#include "dsymldl_v2_grid_selector.h"

#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void dSymLDLV2SetGridError(char *error, size_t error_size,
                                  const char *message)
{
    if (error != NULL && error_size > 0)
    {
        snprintf(error, error_size, "%s", message);
        error[error_size - 1] = '\0';
    }
}

static int dSymLDLV2GridDimensionMatches(int requested, int candidate)
{
    return requested == SUPERLU_GRID_AUTO || requested == candidate;
}

static int dSymLDLV2WithinLimit(int value, int limit)
{
    return limit <= 0 || value <= limit;
}

static int dSymLDLV2IsPowerOfTwo(int value)
{
    return value > 0 && (value & (value - 1)) == 0;
}

static int dSymLDLV2RequestDimensionValid(int value)
{
    return value == SUPERLU_GRID_AUTO || value > 0;
}

static int dSymLDLV2CandidateCount(const dSymLDLV2GridRequest *request,
                                   int communicator_size, size_t *count)
{
    size_t found = 0;
    int pz = 1;

    while (pz <= communicator_size)
    {
        if (communicator_size % pz == 0 &&
            dSymLDLV2GridDimensionMatches(request->pz, pz) &&
            dSymLDLV2WithinLimit(pz, request->max_pz))
        {
            int plane_size = communicator_size / pz;
            for (int pr = 1; pr <= plane_size; )
            {
                if (plane_size % pr == 0)
                {
                    int pc = plane_size / pr;
                    if (dSymLDLV2GridDimensionMatches(request->pr, pr) &&
                        dSymLDLV2GridDimensionMatches(request->pc, pc) &&
                        dSymLDLV2WithinLimit(pr, request->max_pr) &&
                        dSymLDLV2WithinLimit(pc, request->max_pc))
                    {
                        if (found == SIZE_MAX)
                            return 0;
                        ++found;
                    }
                }
                if (pr == plane_size)
                    break;
                ++pr;
            }
        }

        if (pz > communicator_size / 2)
            break;
        pz *= 2;
    }

    *count = found;
    return 1;
}

static int dSymLDLV2GridDimensionsLess(const dSymLDLV2GridCandidate *left,
                                       const dSymLDLV2GridCandidate *right)
{
    if (left->pz != right->pz)
        return left->pz < right->pz;
    if (left->pr != right->pr)
        return left->pr < right->pr;
    return left->pc < right->pc;
}

static long double dSymLDLV2CandidateMemory(
    const dSymLDLV2GridCandidate *candidate)
{
    return (long double) candidate->memory.host_high_water_per_node +
           (long double) candidate->memory.gpu_high_water_per_rank;
}

static double dSymLDLV2CandidateMemoryPressure(
    const dSymLDLV2GridSelection *selection,
    const dSymLDLV2GridCandidate *candidate)
{
    double pressure = 0.0;
    if (selection->usable_gpu_bytes > 0)
        pressure = fmax(
            pressure,
            (double) candidate->memory.gpu_high_water_per_rank /
                (double) selection->usable_gpu_bytes);
    if (selection->usable_host_bytes_per_node > 0)
        pressure = fmax(
            pressure,
            (double) candidate->memory.host_high_water_per_node /
                (double) selection->usable_host_bytes_per_node);
    if (pressure == 0.0)
    {
        long double bytes = dSymLDLV2CandidateMemory(candidate);
        pressure = bytes > (long double) DBL_MAX
                       ? DBL_MAX
                       : (double) bytes;
    }
    return pressure;
}

static int dSymLDLV2RuntimeValueLess(double left, double right)
{
    double scale = fmax(1.0, fmax(fabs(left), fabs(right)));
    return left < right - 64.0 * DBL_EPSILON * scale;
}

static int dSymLDLV2RuntimeValueEqual(double left, double right)
{
    double scale = fmax(1.0, fmax(fabs(left), fabs(right)));
    return fabs(left - right) <= 64.0 * DBL_EPSILON * scale;
}

static int dSymLDLV2CandidateDominates(
    const dSymLDLV2GridCandidate *left,
    const dSymLDLV2GridCandidate *right)
{
    int strict = 0;
    for (int metric = 0; metric < DSYMLDL_V2_RUNTIME_METRIC_COUNT;
         ++metric)
    {
        const dSymLDLV2RuntimeMetric *l = &left->performance.metric[metric];
        const dSymLDLV2RuntimeMetric *r = &right->performance.metric[metric];
        if (!l->active && !r->active)
            continue;
        if (l->active != r->active)
            return 0;
        if (dSymLDLV2RuntimeValueLess(r->total, l->total))
            return 0;
        if (dSymLDLV2RuntimeValueLess(r->critical, l->critical))
            return 0;
        strict = strict || dSymLDLV2RuntimeValueLess(l->total, r->total) ||
                 dSymLDLV2RuntimeValueLess(l->critical, r->critical);
    }
    return strict;
}

void dSymLDLV2GridRequestInit(dSymLDLV2GridRequest *request)
{
    if (request == NULL)
        return;
    memset(request, 0, sizeof(*request));
}

void dSymLDLV2GridSelectionInit(dSymLDLV2GridSelection *selection)
{
    if (selection == NULL)
        return;
    memset(selection, 0, sizeof(*selection));
    selection->selected_index = DSYMLDL_V2_GRID_NO_SELECTION;
}

int dSymLDLV2ParseGridDimension(const char *text, int *dimension,
                                char *error, size_t error_size)
{
    char *end = NULL;
    long value;

    if (text == NULL || dimension == NULL)
    {
        dSymLDLV2SetGridError(error, error_size,
                              "Grid dimension is missing.");
        return 0;
    }
    if (strcmp(text, "auto") == 0)
    {
        *dimension = SUPERLU_GRID_AUTO;
        return 1;
    }

    errno = 0;
    value = strtol(text, &end, 10);
    if (errno == ERANGE || value > INT_MAX || value < 1 || end == text ||
        *end != '\0')
    {
        dSymLDLV2SetGridError(
            error, error_size,
            "Grid dimension must be a positive integer or 'auto'.");
        return 0;
    }

    *dimension = (int) value;
    return 1;
}

int dSymLDLV2EnumerateGridCandidates(
    const dSymLDLV2GridRequest *request, int communicator_size,
    dSymLDLV2GridSelection *selection, char *error, size_t error_size)
{
    size_t candidate_count;
    size_t candidate_index = 0;
    int pz = 1;

    if (request == NULL || selection == NULL)
    {
        dSymLDLV2SetGridError(error, error_size,
                              "Grid request is missing.");
        return 0;
    }
    if (communicator_size < 1)
    {
        dSymLDLV2SetGridError(error, error_size,
                              "Communicator size must be positive.");
        return 0;
    }
    if (!dSymLDLV2RequestDimensionValid(request->pr) ||
        !dSymLDLV2RequestDimensionValid(request->pc) ||
        !dSymLDLV2RequestDimensionValid(request->pz))
    {
        dSymLDLV2SetGridError(
            error, error_size,
            "Requested grid dimensions must be positive or auto.");
        return 0;
    }
    if (request->max_pr < 0 || request->max_pc < 0 ||
        request->max_pz < 0)
    {
        dSymLDLV2SetGridError(
            error, error_size,
            "Grid dimension limits must be nonnegative.");
        return 0;
    }
    if (request->pz != SUPERLU_GRID_AUTO &&
        !dSymLDLV2IsPowerOfTwo(request->pz))
    {
        dSymLDLV2SetGridError(error, error_size,
                              "The process Z dimension must be a power of two.");
        return 0;
    }
    if (!dSymLDLV2CandidateCount(request, communicator_size,
                                 &candidate_count))
    {
        dSymLDLV2SetGridError(error, error_size,
                              "Grid candidate count overflows.");
        return 0;
    }
    if (candidate_count == 0)
    {
        dSymLDLV2SetGridError(
            error, error_size,
            "No process grid matches the communicator and requested dimensions.");
        return 0;
    }
    if (candidate_count > SIZE_MAX / sizeof(dSymLDLV2GridCandidate))
    {
        dSymLDLV2SetGridError(error, error_size,
                              "Grid candidate allocation overflows.");
        return 0;
    }

    dSymLDLV2GridSelectionDestroy(selection);
    selection->candidates = (dSymLDLV2GridCandidate *) calloc(
        candidate_count, sizeof(dSymLDLV2GridCandidate));
    if (selection->candidates == NULL)
    {
        dSymLDLV2SetGridError(error, error_size,
                              "Grid candidate allocation failed.");
        return 0;
    }
    selection->total_ranks = communicator_size;
    selection->candidate_count = candidate_count;
    selection->selected_index = DSYMLDL_V2_GRID_NO_SELECTION;

    while (pz <= communicator_size)
    {
        if (communicator_size % pz == 0 &&
            dSymLDLV2GridDimensionMatches(request->pz, pz) &&
            dSymLDLV2WithinLimit(pz, request->max_pz))
        {
            int plane_size = communicator_size / pz;
            for (int pr = 1; pr <= plane_size; )
            {
                if (plane_size % pr == 0)
                {
                    int pc = plane_size / pr;
                    if (dSymLDLV2GridDimensionMatches(request->pr, pr) &&
                        dSymLDLV2GridDimensionMatches(request->pc, pc) &&
                        dSymLDLV2WithinLimit(pr, request->max_pr) &&
                        dSymLDLV2WithinLimit(pc, request->max_pc))
                    {
                        dSymLDLV2GridCandidate *candidate =
                            &selection->candidates[candidate_index++];
                        candidate->pr = pr;
                        candidate->pc = pc;
                        candidate->pz = pz;
                        candidate->status = DSYMLDL_V2_GRID_FEASIBLE;
                    }
                }
                if (pr == plane_size)
                    break;
                ++pr;
            }
        }

        if (pz > communicator_size / 2)
            break;
        pz *= 2;
    }

    if (candidate_index != candidate_count)
    {
        dSymLDLV2GridSelectionDestroy(selection);
        dSymLDLV2SetGridError(error, error_size,
                              "Grid candidate enumeration is inconsistent.");
        return 0;
    }
    return 1;
}

int dSymLDLV2SelectGrid(dSymLDLV2GridSelection *selection,
                        char *error, size_t error_size)
{
    size_t best = DSYMLDL_V2_GRID_NO_SELECTION;

    if (selection == NULL || selection->candidates == NULL ||
        selection->candidate_count == 0)
    {
        dSymLDLV2SetGridError(error, error_size,
                              "Grid candidate list is empty.");
        return 0;
    }
    selection->pareto_count = 0;
    selection->confidence = DSYMLDL_V2_GRID_CONFIDENCE_NONE;
    for (size_t i = 0; i < selection->candidate_count; ++i)
    {
        dSymLDLV2GridCandidate *candidate = &selection->candidates[i];
        candidate->performance.pareto_dominated = 0;
        candidate->performance.pareto_dominator = -1;
        candidate->performance.worst_case_regret = 0.0;
        candidate->performance.summed_regret = 0.0;
        candidate->performance.predicted_seconds = 0.0;
        candidate->performance.predicted_seconds_lower = 0.0;
        candidate->performance.predicted_seconds_upper = 0.0;
        candidate->performance.calibrated_rank = 0;
        candidate->performance.plausible_alternative = 0;
        candidate->performance.robust_winner = 0;
        for (int metric = 0; metric < DSYMLDL_V2_RUNTIME_METRIC_COUNT;
             ++metric)
            candidate->performance.metric[metric].normalized_regret = 0.0;
        if (candidate->status != DSYMLDL_V2_GRID_FEASIBLE)
            continue;
        for (int metric = 0; metric < DSYMLDL_V2_RUNTIME_METRIC_COUNT;
             ++metric)
        {
            dSymLDLV2RuntimeMetric *value =
                &candidate->performance.metric[metric];
            if (value->active &&
                (!isfinite(value->total) || value->total < 0.0 ||
                 !isfinite(value->critical) || value->critical < 0.0 ||
                 !isfinite(value->waiting) || value->waiting < 0.0))
            {
                candidate->performance.pareto_dominated = 1;
                candidate->performance.pareto_dominator = -1;
                break;
            }
        }
    }

    /* A candidate must improve at least one total/critical structural cost
       without worsening another to dominate. No machine coefficient is
       introduced here. */
    for (size_t i = 0; i < selection->candidate_count; ++i)
    {
        dSymLDLV2GridCandidate *candidate = &selection->candidates[i];
        if (candidate->status != DSYMLDL_V2_GRID_FEASIBLE ||
            candidate->performance.pareto_dominated)
            continue;
        for (size_t j = 0; j < selection->candidate_count; ++j)
        {
            if (i == j ||
                selection->candidates[j].status !=
                    DSYMLDL_V2_GRID_FEASIBLE ||
                selection->candidates[j].performance.pareto_dominated)
                continue;
            if (dSymLDLV2CandidateDominates(&selection->candidates[j],
                                            candidate))
            {
                candidate->performance.pareto_dominated = 1;
                candidate->performance.pareto_dominator = (int) j;
                break;
            }
        }
        if (!candidate->performance.pareto_dominated)
            ++selection->pareto_count;
    }

    if (selection->pareto_count == 0)
        goto no_candidate;

    /* Resolve a multi-point frontier by minimizing worst normalized regret.
       This is the robust choice over unknown positive machine coefficients;
       summed regret, memory pressure, and grid dimensions only break ties. */
    for (int metric = 0; metric < DSYMLDL_V2_RUNTIME_METRIC_COUNT;
         ++metric)
    {
        double minimum_total = DBL_MAX;
        double maximum_total = -DBL_MAX;
        double minimum_critical = DBL_MAX;
        double maximum_critical = -DBL_MAX;
        int active = 0;
        for (size_t i = 0; i < selection->candidate_count; ++i)
        {
            dSymLDLV2GridCandidate *candidate = &selection->candidates[i];
            dSymLDLV2RuntimeMetric *value =
                &candidate->performance.metric[metric];
            if (candidate->status != DSYMLDL_V2_GRID_FEASIBLE ||
                candidate->performance.pareto_dominated || !value->active)
                continue;
            active = 1;
            minimum_total = fmin(minimum_total, value->total);
            maximum_total = fmax(maximum_total, value->total);
            minimum_critical = fmin(minimum_critical, value->critical);
            maximum_critical = fmax(maximum_critical, value->critical);
        }
        if (!active)
            continue;
        double total_range = maximum_total - minimum_total;
        double critical_range = maximum_critical - minimum_critical;
        for (size_t i = 0; i < selection->candidate_count; ++i)
        {
            dSymLDLV2GridCandidate *candidate = &selection->candidates[i];
            dSymLDLV2RuntimeMetric *value =
                &candidate->performance.metric[metric];
            if (candidate->status != DSYMLDL_V2_GRID_FEASIBLE ||
                candidate->performance.pareto_dominated || !value->active)
                continue;
            double total_regret =
                dSymLDLV2RuntimeValueEqual(minimum_total, maximum_total)
                    ? 0.0
                    : (value->total - minimum_total) / total_range;
            double critical_regret =
                dSymLDLV2RuntimeValueEqual(minimum_critical,
                                           maximum_critical)
                    ? 0.0
                    : (value->critical - minimum_critical) /
                          critical_range;
            value->normalized_regret = fmax(total_regret,
                                             critical_regret);
            candidate->performance.worst_case_regret = fmax(
                candidate->performance.worst_case_regret,
                value->normalized_regret);
            candidate->performance.summed_regret +=
                total_regret + critical_regret;
        }
    }

    for (size_t i = 0; i < selection->candidate_count; ++i)
    {
        dSymLDLV2GridCandidate *candidate = &selection->candidates[i];
        if (candidate->status != DSYMLDL_V2_GRID_FEASIBLE ||
            candidate->performance.pareto_dominated)
            continue;
        if (best == DSYMLDL_V2_GRID_NO_SELECTION)
        {
            best = i;
            continue;
        }
        dSymLDLV2GridCandidate *current = &selection->candidates[best];
        int choose = dSymLDLV2RuntimeValueLess(
            candidate->performance.worst_case_regret,
            current->performance.worst_case_regret);
        if (!choose && dSymLDLV2RuntimeValueEqual(
                           candidate->performance.worst_case_regret,
                           current->performance.worst_case_regret))
            choose = dSymLDLV2RuntimeValueLess(
                candidate->performance.summed_regret,
                current->performance.summed_regret);
        if (!choose && dSymLDLV2RuntimeValueEqual(
                           candidate->performance.worst_case_regret,
                           current->performance.worst_case_regret) &&
            dSymLDLV2RuntimeValueEqual(
                candidate->performance.summed_regret,
                current->performance.summed_regret))
        {
            double candidate_pressure =
                dSymLDLV2CandidateMemoryPressure(selection, candidate);
            double current_pressure =
                dSymLDLV2CandidateMemoryPressure(selection, current);
            choose = dSymLDLV2RuntimeValueLess(candidate_pressure,
                                               current_pressure) ||
                     (dSymLDLV2RuntimeValueEqual(candidate_pressure,
                                                 current_pressure) &&
                      dSymLDLV2GridDimensionsLess(candidate, current));
        }
        if (choose)
            best = i;
    }

    if (best == DSYMLDL_V2_GRID_NO_SELECTION)
        goto no_candidate;
    selection->selected_index = best;
    if (selection->pareto_count == 1)
        selection->confidence = DSYMLDL_V2_GRID_CONFIDENCE_DOMINANT;
    else
    {
        int tied = 0;
        for (size_t i = 0; i < selection->candidate_count; ++i)
        {
            if (i == best || selection->candidates[i].status !=
                                 DSYMLDL_V2_GRID_FEASIBLE ||
                selection->candidates[i].performance.pareto_dominated)
                continue;
            tied = tied || dSymLDLV2RuntimeValueEqual(
                               selection->candidates[i]
                                   .performance.worst_case_regret,
                               selection->candidates[best]
                                   .performance.worst_case_regret);
        }
        selection->confidence =
            tied ? DSYMLDL_V2_GRID_CONFIDENCE_AMBIGUOUS_TIE
                 : DSYMLDL_V2_GRID_CONFIDENCE_ROBUST_COMPROMISE;
    }
    return 1;

no_candidate:
    selection->selected_index = DSYMLDL_V2_GRID_NO_SELECTION;
    dSymLDLV2SetGridError(error, error_size,
                          "No feasible process grid candidate remains.");
    return 0;
}

static double dSymLDLV2CalibratedTime(
    const dSymLDLV2GridCandidate *candidate,
    const dSymLDLV2CalibrationProfile *profile, int bound)
{
    double result = 0.0;
    for (int metric = 0; metric < DSYMLDL_V2_RUNTIME_METRIC_COUNT;
         ++metric)
    {
        const dSymLDLV2RuntimeMetric *exposure =
            &candidate->performance.metric[metric];
        const dSymLDLV2CalibrationCoefficient *coefficient =
            &profile->coefficient[metric];
        if (!exposure->active)
            continue;
        double seconds = bound < 0 ? coefficient->lower_seconds_per_unit
                         : bound > 0 ? coefficient->upper_seconds_per_unit
                                     : coefficient->seconds_per_unit;
        result += exposure->critical * seconds;
    }
    return result;
}

static void dSymLDLV2PairDifferenceBounds(
    const dSymLDLV2GridCandidate *left,
    const dSymLDLV2GridCandidate *right,
    const dSymLDLV2CalibrationProfile *profile,
    double *minimum, double *maximum)
{
    *minimum = 0.0;
    *maximum = 0.0;
    for (int metric = 0; metric < DSYMLDL_V2_RUNTIME_METRIC_COUNT;
         ++metric)
    {
        double delta = left->performance.metric[metric].critical -
                       right->performance.metric[metric].critical;
        const dSymLDLV2CalibrationCoefficient *coefficient =
            &profile->coefficient[metric];
        if (delta >= 0.0)
        {
            *minimum += delta * coefficient->lower_seconds_per_unit;
            *maximum += delta * coefficient->upper_seconds_per_unit;
        }
        else
        {
            *minimum += delta * coefficient->upper_seconds_per_unit;
            *maximum += delta * coefficient->lower_seconds_per_unit;
        }
    }
}

int dSymLDLV2SelectGridCalibrated(
    dSymLDLV2GridSelection *selection,
    const dSymLDLV2CalibrationProfile *profile,
    char *error, size_t error_size)
{
    if (selection == NULL || profile == NULL || !profile->complete)
    {
        dSymLDLV2SetGridError(error, error_size,
                              "SymLDL calibration profile is incomplete.");
        return 0;
    }
    if (!dSymLDLV2SelectGrid(selection, error, error_size))
        return 0;
    for (size_t i = 0; i < selection->candidate_count; ++i)
    {
        dSymLDLV2GridCandidate *candidate = &selection->candidates[i];
        if (candidate->status != DSYMLDL_V2_GRID_FEASIBLE ||
            candidate->performance.pareto_dominated)
            continue;
        for (int metric = 0; metric < DSYMLDL_V2_RUNTIME_METRIC_COUNT;
             ++metric)
            if (candidate->performance.metric[metric].active &&
                profile->coefficient[metric].source ==
                    DSYMLDL_V2_CALIBRATION_UNAVAILABLE)
            {
                dSymLDLV2SetGridError(
                    error, error_size,
                    "SymLDL calibration is missing an active runtime metric.");
                return 0;
            }
        candidate->performance.predicted_seconds =
            dSymLDLV2CalibratedTime(candidate, profile, 0);
        candidate->performance.predicted_seconds_lower =
            dSymLDLV2CalibratedTime(candidate, profile, -1);
        candidate->performance.predicted_seconds_upper =
            dSymLDLV2CalibratedTime(candidate, profile, 1);
    }

    size_t best = DSYMLDL_V2_GRID_NO_SELECTION;
    for (size_t i = 0; i < selection->candidate_count; ++i)
    {
        dSymLDLV2GridCandidate *candidate = &selection->candidates[i];
        if (candidate->status != DSYMLDL_V2_GRID_FEASIBLE ||
            candidate->performance.pareto_dominated)
            continue;
        if (best == DSYMLDL_V2_GRID_NO_SELECTION ||
            dSymLDLV2RuntimeValueLess(
                candidate->performance.predicted_seconds,
                selection->candidates[best].performance.predicted_seconds))
            best = i;
        else if (dSymLDLV2RuntimeValueEqual(
                     candidate->performance.predicted_seconds,
                     selection->candidates[best]
                         .performance.predicted_seconds))
        {
            double candidate_pressure =
                dSymLDLV2CandidateMemoryPressure(selection, candidate);
            double best_pressure = dSymLDLV2CandidateMemoryPressure(
                selection, &selection->candidates[best]);
            if (dSymLDLV2RuntimeValueLess(candidate_pressure,
                                          best_pressure) ||
                (dSymLDLV2RuntimeValueEqual(candidate_pressure,
                                             best_pressure) &&
                 dSymLDLV2GridDimensionsLess(
                     candidate, &selection->candidates[best])))
                best = i;
        }
    }
    if (best == DSYMLDL_V2_GRID_NO_SELECTION)
    {
        dSymLDLV2SetGridError(error, error_size,
                              "No calibrated process grid remains.");
        return 0;
    }

    int robust = 1;
    selection->selected_index = best;
    for (size_t i = 0; i < selection->candidate_count; ++i)
    {
        dSymLDLV2GridCandidate *candidate = &selection->candidates[i];
        if (candidate->status != DSYMLDL_V2_GRID_FEASIBLE ||
            candidate->performance.pareto_dominated)
            continue;
        int rank = 1;
        for (size_t j = 0; j < selection->candidate_count; ++j)
            if (selection->candidates[j].status ==
                    DSYMLDL_V2_GRID_FEASIBLE &&
                !selection->candidates[j].performance.pareto_dominated &&
                dSymLDLV2RuntimeValueLess(
                    selection->candidates[j].performance.predicted_seconds,
                    candidate->performance.predicted_seconds))
                ++rank;
        candidate->performance.calibrated_rank = rank;

        double minimum = 0.0;
        double maximum = 0.0;
        dSymLDLV2PairDifferenceBounds(
            candidate, &selection->candidates[best], profile,
            &minimum, &maximum);
        candidate->performance.plausible_alternative =
            i == best || !dSymLDLV2RuntimeValueLess(0.0, minimum);
        if (i != best)
        {
            dSymLDLV2PairDifferenceBounds(
                &selection->candidates[best], candidate, profile,
                &minimum, &maximum);
            if (dSymLDLV2RuntimeValueLess(0.0, maximum))
                robust = 0;
        }
    }
    selection->candidates[best].performance.robust_winner = robust;
    selection->confidence = robust
                                ? DSYMLDL_V2_GRID_CONFIDENCE_CALIBRATED_ROBUST
                                : DSYMLDL_V2_GRID_CONFIDENCE_CALIBRATED_ESTIMATE;
    selection->calibration = *profile;
    return 1;
}

const char *dSymLDLV2RuntimeMetricName(dSymLDLV2RuntimeMetricKind kind)
{
    static const char *names[DSYMLDL_V2_RUNTIME_METRIC_COUNT] = {
        "GPU FLOPs", "CPU FLOPs", "GPU-local bytes", "CPU-local bytes",
        "GPU task launches", "GPU synchronizations",
        "process synchronizations", "intra-node messages",
        "intra-node bytes", "inter-node messages", "inter-node bytes",
        "host-to-device bytes", "device-to-host bytes"
    };
    return kind >= 0 && kind < DSYMLDL_V2_RUNTIME_METRIC_COUNT
               ? names[kind]
               : "unknown metric";
}

const char *dSymLDLV2GridConfidenceString(
    dSymLDLV2GridSelectionConfidence confidence)
{
    switch (confidence)
    {
    case DSYMLDL_V2_GRID_CONFIDENCE_DOMINANT:
        return "dominant";
    case DSYMLDL_V2_GRID_CONFIDENCE_ROBUST_COMPROMISE:
        return "robust compromise";
    case DSYMLDL_V2_GRID_CONFIDENCE_AMBIGUOUS_TIE:
        return "ambiguous tie";
    case DSYMLDL_V2_GRID_CONFIDENCE_CALIBRATED_ROBUST:
        return "calibrated robust";
    case DSYMLDL_V2_GRID_CONFIDENCE_CALIBRATED_ESTIMATE:
        return "calibrated estimate";
    case DSYMLDL_V2_GRID_CONFIDENCE_UNCALIBRATED_FALLBACK:
        return "uncalibrated fallback";
    default:
        return "unavailable";
    }
}

const char *dSymLDLV2GridCandidateStatusString(
    dSymLDLV2GridCandidateStatus status)
{
    switch (status)
    {
    case DSYMLDL_V2_GRID_FEASIBLE:
        return "feasible";
    case DSYMLDL_V2_GRID_REJECT_PRODUCT:
        return "rejected: rank product";
    case DSYMLDL_V2_GRID_REJECT_PZ:
        return "rejected: Z dimension";
    case DSYMLDL_V2_GRID_REJECT_LIMIT:
        return "rejected: implementation limit";
    case DSYMLDL_V2_GRID_REJECT_HOST_MEMORY:
        return "rejected: host memory";
    case DSYMLDL_V2_GRID_REJECT_GPU_MEMORY:
        return "rejected: GPU memory";
    case DSYMLDL_V2_GRID_REJECT_IMPLEMENTATION:
        return "rejected: unsupported";
    default:
        return "rejected: unknown";
    }
}

void dSymLDLV2GridSelectionDestroy(dSymLDLV2GridSelection *selection)
{
    if (selection == NULL)
        return;
    free(selection->candidates);
    dSymLDLV2GridSelectionInit(selection);
}
