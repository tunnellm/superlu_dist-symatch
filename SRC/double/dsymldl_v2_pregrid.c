#include "dsymldl_v2_pregrid.h"

#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static void dSymLDLV2SetPreGridError(char *error, size_t error_size,
                                     const char *message)
{
    if (error != NULL && error_size > 0)
    {
        snprintf(error, error_size, "%s", message);
        error[error_size - 1] = '\0';
    }
}

static int dSymLDLV2OwnerAffinity(double *weight,
                                  char *error, size_t error_size)
{
    const char *text = getenv("GPU3DV2_OWNER_AFFINITY");
    char *end = NULL;
    double parsed;

    *weight = 0.0;
    if (text == NULL || text[0] == '\0')
        return 1;
    errno = 0;
    parsed = strtod(text, &end);
    if (errno == ERANGE || end == text || *end != '\0' ||
        !isfinite(parsed) || parsed < 0.0)
    {
        dSymLDLV2SetPreGridError(
            error, error_size,
            "GPU3DV2_OWNER_AFFINITY must be a nonnegative number.");
        return 0;
    }
    *weight = parsed;
    return 1;
}

int dSymLDLV2SelectGridFromSymbolic(
    int_t nsupers, const Glu_persist_t *persist,
    const Glu_freeable_t *symbolic, const int_t *etree,
    uint64_t input_nnz, MPI_Comm communicator,
    const dSymLDLV2GridRequest *request,
    const dSymLDLV2GridRuntimeConfig *runtime,
    dSymLDLV2GridSelection *selection,
    char *error, size_t error_size)
{
    int_t *setree = NULL;
    treeList_t *tree_list = NULL;
    dSymLDLV2StructuralSummary structure;
    dSymLDLV2PartitionPlanInput partition_input;
    double owner_affinity_weight;
    int success = 0;
    int local_ready = 0;
    int all_ready = 0;
    double minimum_owner_affinity;
    double maximum_owner_affinity;

    memset(&structure, 0, sizeof(structure));
    memset(&partition_input, 0, sizeof(partition_input));
    if (nsupers < 1 || persist == NULL || persist->xsup == NULL ||
        persist->supno == NULL || symbolic == NULL || etree == NULL ||
        request == NULL || runtime == NULL || selection == NULL)
    {
        dSymLDLV2SetPreGridError(
            error, error_size,
            "SymLDL pre-grid symbolic input is incomplete.");
        goto synchronize;
    }
    if (!dSymLDLV2OwnerAffinity(&owner_affinity_weight,
                                error, error_size))
        goto synchronize;

    setree = supernodal_etree(nsupers, (int_t *) etree,
                              persist->supno, persist->xsup);
    if (setree == NULL)
    {
        dSymLDLV2SetPreGridError(
            error, error_size,
            "SymLDL pre-grid supernodal tree construction failed.");
        goto synchronize;
    }
    tree_list = setree2list(nsupers, setree);
    if (tree_list == NULL)
    {
        dSymLDLV2SetPreGridError(
            error, error_size,
            "SymLDL pre-grid tree list construction failed.");
        goto synchronize;
    }
    if (!dSymLDLV2BuildStructuralSummary(
            nsupers, persist, symbolic, &structure, error, error_size))
        goto synchronize;
    structure.input_nnz = input_nnz;

    partition_input.nsupers = nsupers;
    partition_input.setree = setree;
    partition_input.xsup = persist->xsup;
    partition_input.lrows = structure.panel_rows;
    partition_input.tree_list = tree_list;
    partition_input.owner_affinity_weight = owner_affinity_weight;
    local_ready = 1;

synchronize:
    MPI_Allreduce(&local_ready, &all_ready, 1, MPI_INT, MPI_MIN,
                  communicator);
    if (!all_ready)
    {
        if (local_ready)
            dSymLDLV2SetPreGridError(
                error, error_size,
                "SymLDL pre-grid symbolic preparation failed on another MPI rank.");
        goto cleanup;
    }
    MPI_Allreduce(&owner_affinity_weight, &minimum_owner_affinity, 1,
                  MPI_DOUBLE, MPI_MIN, communicator);
    MPI_Allreduce(&owner_affinity_weight, &maximum_owner_affinity, 1,
                  MPI_DOUBLE, MPI_MAX, communicator);
    if (minimum_owner_affinity != maximum_owner_affinity)
    {
        dSymLDLV2SetPreGridError(
            error, error_size,
            "GPU3DV2_OWNER_AFFINITY differs between MPI ranks.");
        goto cleanup;
    }
    success = dSymLDLV2SelectGridForStructure(
        &partition_input, &structure, communicator, request, runtime,
        selection, error, error_size);

cleanup:
    dSymLDLV2StructuralSummaryDestroy(&structure);
    if (tree_list != NULL)
        free_treelist(nsupers, tree_list);
    SUPERLU_FREE(setree);
    return success;
}
