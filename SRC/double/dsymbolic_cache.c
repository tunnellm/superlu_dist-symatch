#include "dsymbolic_cache.h"

#include <stdio.h>
#include <string.h>

#define DSYMBOLIC_CACHE_VERSION 1U

typedef struct {
    char magic[16];
    uint32_t version;
    uint32_t int_t_size;
    int64_t n;
    int64_t nsupers;
    int64_t nzlmax;
    int64_t nzumax;
    int64_t nnz_lu;
    int32_t diagonal_scale;
    int32_t equil;
    int32_t row_permutation;
    int32_t column_permutation;
    int32_t symmetric_factorization;
    uint32_t has_row_scale;
    uint32_t has_column_scale;
} dSymbolicCacheHeader;

static const char dSymbolicCacheMagic[16] = "SLU-SYMBOLIC-v1";

static void dSetSymbolicCacheError(
    char *error, size_t error_size, const char *message)
{
    if (error != NULL && error_size > 0)
        snprintf(error, error_size, "%s", message);
}

static int dSymbolicCacheWriteArray(
    FILE *file, const void *values, size_t count, size_t element_size)
{
    return count == 0 ||
           (values != NULL && fwrite(values, element_size, count, file) == count);
}

static int dSymbolicCacheReadArray(
    FILE *file, void *values, size_t count, size_t element_size)
{
    return count == 0 ||
           (values != NULL && fread(values, element_size, count, file) == count);
}

static int dSymbolicCacheHeaderValid(
    const dSymbolicCacheHeader *header, int_t n,
    const superlu_dist_options_t *options)
{
    return header != NULL &&
           memcmp(header->magic, dSymbolicCacheMagic,
                  sizeof(header->magic)) == 0 &&
           header->version == DSYMBOLIC_CACHE_VERSION &&
           header->int_t_size == sizeof(int_t) &&
           header->n == (int64_t) n && header->nsupers > 0 &&
           header->nsupers <= header->n &&
           header->nzlmax >= 0 && header->nzumax >= 0 &&
           header->nnz_lu >= 0 &&
#ifdef _LONGINT
           header->nzlmax <= INT64_MAX && header->nzumax <= INT64_MAX &&
#else
           header->nzlmax <= INT32_MAX && header->nzumax <= INT32_MAX &&
#endif
           header->diagonal_scale >= NOEQUIL &&
           header->diagonal_scale <= BOTH &&
           header->equil == options->Equil &&
           header->row_permutation == options->RowPerm &&
           header->column_permutation == options->ColPerm &&
           header->symmetric_factorization == options->SymFact &&
           header->has_row_scale ==
               (header->diagonal_scale == ROW ||
                header->diagonal_scale == BOTH) &&
           header->has_column_scale ==
               (header->diagonal_scale == COL ||
                header->diagonal_scale == BOTH);
}

static void dSymbolicCacheFreeArrays(
    Glu_freeable_t *symbolic)
{
    if (symbolic == NULL)
        return;
    SUPERLU_FREE(symbolic->lsub);
    SUPERLU_FREE(symbolic->xlsub);
    SUPERLU_FREE(symbolic->usub);
    SUPERLU_FREE(symbolic->xusub);
    symbolic->lsub = NULL;
    symbolic->xlsub = NULL;
    symbolic->usub = NULL;
    symbolic->xusub = NULL;
}

int dSymbolicCacheWrite(
    const char *path, int_t n, const superlu_dist_options_t *options,
    const dScalePermstruct_t *scale_permutation, const dLUstruct_t *lu,
    const Glu_freeable_t *symbolic, gridinfo3d_t *grid3d,
    char *error, size_t error_size)
{
    int local_success = 1;
    int success = 0;

    if (error != NULL && error_size > 0)
        error[0] = '\0';
    if (path == NULL || path[0] == '\0' || n < 1 || options == NULL ||
        scale_permutation == NULL || lu == NULL || lu->Glu_persist == NULL ||
        symbolic == NULL || grid3d == NULL)
    {
        dSetSymbolicCacheError(
            error, error_size, "Symbolic cache write input is incomplete.");
        return 0;
    }

    if (grid3d->iam == 0)
    {
        const Glu_persist_t *persist = lu->Glu_persist;
        int_t nsupers = persist->supno != NULL ? persist->supno[n - 1] + 1 : 0;
        dSymbolicCacheHeader header;
        memset(&header, 0, sizeof(header));
        memcpy(header.magic, dSymbolicCacheMagic,
               sizeof(header.magic));
        header.version = DSYMBOLIC_CACHE_VERSION;
        header.int_t_size = sizeof(int_t);
        header.n = n;
        header.nsupers = nsupers;
        header.nzlmax = symbolic->nzlmax;
        header.nzumax = symbolic->nzumax;
        header.nnz_lu = symbolic->nnzLU;
        header.diagonal_scale = scale_permutation->DiagScale;
        header.equil = options->Equil;
        header.row_permutation = options->RowPerm;
        header.column_permutation = options->ColPerm;
        header.symmetric_factorization = options->SymFact;
        header.has_row_scale = scale_permutation->DiagScale == ROW ||
                               scale_permutation->DiagScale == BOTH;
        header.has_column_scale = scale_permutation->DiagScale == COL ||
                                  scale_permutation->DiagScale == BOTH;

        FILE *file = fopen(path, "wb");
        local_success = file != NULL && nsupers > 0 &&
            scale_permutation->perm_r != NULL &&
            scale_permutation->perm_c != NULL && lu->etree != NULL &&
            persist->xsup != NULL && persist->supno != NULL &&
            symbolic->xlsub != NULL && symbolic->xusub != NULL &&
            (!header.has_row_scale || scale_permutation->R != NULL) &&
            (!header.has_column_scale || scale_permutation->C != NULL);
        if (local_success)
            local_success = fwrite(&header, sizeof(header), 1, file) == 1 &&
                dSymbolicCacheWriteArray(file, scale_permutation->perm_r,
                                    (size_t) n, sizeof(int_t)) &&
                dSymbolicCacheWriteArray(file, scale_permutation->perm_c,
                                    (size_t) n, sizeof(int_t)) &&
                dSymbolicCacheWriteArray(file, scale_permutation->R,
                                    header.has_row_scale ? (size_t) n : 0,
                                    sizeof(double)) &&
                dSymbolicCacheWriteArray(file, scale_permutation->C,
                                    header.has_column_scale ? (size_t) n : 0,
                                    sizeof(double)) &&
                dSymbolicCacheWriteArray(file, lu->etree,
                                    (size_t) n, sizeof(int_t)) &&
                dSymbolicCacheWriteArray(file, persist->xsup,
                                    (size_t) nsupers + 1, sizeof(int_t)) &&
                dSymbolicCacheWriteArray(file, persist->supno,
                                    (size_t) n, sizeof(int_t)) &&
                dSymbolicCacheWriteArray(file, symbolic->lsub,
                                    (size_t) symbolic->nzlmax,
                                    sizeof(int_t)) &&
                dSymbolicCacheWriteArray(file, symbolic->xlsub,
                                    (size_t) n + 1, sizeof(int_t)) &&
                dSymbolicCacheWriteArray(file, symbolic->usub,
                                    (size_t) symbolic->nzumax,
                                    sizeof(int_t)) &&
                dSymbolicCacheWriteArray(file, symbolic->xusub,
                                    (size_t) n + 1, sizeof(int_t));
        if (file != NULL && fclose(file) != 0)
            local_success = 0;
        if (!local_success)
            remove(path);
    }

    MPI_Allreduce(&local_success, &success, 1, MPI_INT, MPI_MIN,
                  grid3d->comm);
    if (!success)
        dSetSymbolicCacheError(
            error, error_size, "Symbolic cache write failed.");
    return success;
}

int dSymbolicCacheRead(
    const char *path, int_t n, const superlu_dist_options_t *options,
    dScalePermstruct_t *scale_permutation, dLUstruct_t *lu,
    Glu_freeable_t **symbolic, gridinfo3d_t *grid3d,
    char *error, size_t error_size)
{
    int local_success = 1;
    int success = 0;

    if (error != NULL && error_size > 0)
        error[0] = '\0';
    if (path == NULL || path[0] == '\0' || n < 1 || options == NULL ||
        scale_permutation == NULL || lu == NULL || lu->Glu_persist == NULL ||
        symbolic == NULL || grid3d == NULL)
    {
        dSetSymbolicCacheError(
            error, error_size, "Symbolic cache read input is incomplete.");
        return 0;
    }

    if (grid3d->zscp.Iam == 0)
    {
        dSymbolicCacheHeader header;
        memset(&header, 0, sizeof(header));
        FILE *file = fopen(path, "rb");
        local_success = file != NULL &&
                        fread(&header, sizeof(header), 1, file) == 1 &&
                        dSymbolicCacheHeaderValid(&header, n, options);
        if (local_success)
        {
            Glu_persist_t *persist = lu->Glu_persist;
            Glu_freeable_t *loaded = (Glu_freeable_t *)
                SUPERLU_MALLOC(sizeof(Glu_freeable_t));
            if (loaded != NULL)
                memset(loaded, 0, sizeof(*loaded));
            persist->xsup = intMalloc_dist((int_t) header.nsupers + 1);
            persist->supno = intMalloc_dist(n);
            if (header.has_row_scale)
                scale_permutation->R = doubleMalloc_dist(n);
            if (header.has_column_scale)
                scale_permutation->C = doubleMalloc_dist(n);
            if (loaded != NULL)
            {
                loaded->nzlmax = (int_t) header.nzlmax;
                loaded->nzumax = (int_t) header.nzumax;
                loaded->nnzLU = header.nnz_lu;
                loaded->MemModel = SYSTEM;
                loaded->lsub = loaded->nzlmax > 0
                                   ? intMalloc_dist(loaded->nzlmax) : NULL;
                loaded->xlsub = intMalloc_dist(n + 1);
                loaded->usub = loaded->nzumax > 0
                                   ? intMalloc_dist(loaded->nzumax) : NULL;
                loaded->xusub = intMalloc_dist(n + 1);
            }
            local_success = loaded != NULL && persist->xsup != NULL &&
                persist->supno != NULL && loaded->xlsub != NULL &&
                loaded->xusub != NULL &&
                (loaded->nzlmax == 0 || loaded->lsub != NULL) &&
                (loaded->nzumax == 0 || loaded->usub != NULL) &&
                (!header.has_row_scale || scale_permutation->R != NULL) &&
                (!header.has_column_scale || scale_permutation->C != NULL);
            if (local_success)
            {
                scale_permutation->DiagScale =
                    (DiagScale_t) header.diagonal_scale;
                local_success =
                    dSymbolicCacheReadArray(file, scale_permutation->perm_r,
                                      (size_t) n, sizeof(int_t)) &&
                    dSymbolicCacheReadArray(file, scale_permutation->perm_c,
                                      (size_t) n, sizeof(int_t)) &&
                    dSymbolicCacheReadArray(file, scale_permutation->R,
                                      header.has_row_scale ? (size_t) n : 0,
                                      sizeof(double)) &&
                    dSymbolicCacheReadArray(file, scale_permutation->C,
                                      header.has_column_scale ? (size_t) n : 0,
                                      sizeof(double)) &&
                    dSymbolicCacheReadArray(file, lu->etree,
                                      (size_t) n, sizeof(int_t)) &&
                    dSymbolicCacheReadArray(file, persist->xsup,
                                      (size_t) header.nsupers + 1,
                                      sizeof(int_t)) &&
                    dSymbolicCacheReadArray(file, persist->supno,
                                      (size_t) n, sizeof(int_t)) &&
                    dSymbolicCacheReadArray(file, loaded->lsub,
                                      (size_t) loaded->nzlmax,
                                      sizeof(int_t)) &&
                    dSymbolicCacheReadArray(file, loaded->xlsub,
                                      (size_t) n + 1, sizeof(int_t)) &&
                    dSymbolicCacheReadArray(file, loaded->usub,
                                      (size_t) loaded->nzumax,
                                      sizeof(int_t)) &&
                    dSymbolicCacheReadArray(file, loaded->xusub,
                                      (size_t) n + 1, sizeof(int_t));
            }
            if (local_success)
                *symbolic = loaded;
            else
            {
                dSymbolicCacheFreeArrays(loaded);
                SUPERLU_FREE(loaded);
                SUPERLU_FREE(persist->xsup);
                SUPERLU_FREE(persist->supno);
                SUPERLU_FREE(scale_permutation->R);
                SUPERLU_FREE(scale_permutation->C);
                persist->xsup = NULL;
                persist->supno = NULL;
                scale_permutation->R = NULL;
                scale_permutation->C = NULL;
            }
        }
        if (file != NULL)
            fclose(file);
    }

    MPI_Allreduce(&local_success, &success, 1, MPI_INT, MPI_MIN,
                  grid3d->comm);
    if (!success)
        dSetSymbolicCacheError(
            error, error_size,
            "Symbolic cache read failed or does not match the run.");
    return success;
}
