#pragma once

// Diagonal factor buffer helpers for xLUstruct_t.

template <typename Ftype>
diagFactBufs_type<Ftype> **xLUstruct_t<Ftype>::initDiagFactBufsArr(
    int_t num_bufs, int_t ldt)
{
    diagFactBufs_type<Ftype> **dFBufs =
        (diagFactBufs_type<Ftype> **) SUPERLU_MALLOC(
            num_bufs * sizeof(diagFactBufs_type<Ftype> *));
    for (int i = 0; i < num_bufs; i++)
    {
        dFBufs[i] = (diagFactBufs_type<Ftype> *) SUPERLU_MALLOC(
            sizeof(diagFactBufs_type<Ftype>));
        dFBufs[i]->BlockUFactor = (Ftype *) SUPERLU_MALLOC(
            symldl_v2_checked_product(
                symldl_v2_checked_product((size_t) ldt, (size_t) ldt,
                                          "diagonal factor buffer allocation overflows."),
                sizeof(Ftype),
                "diagonal factor buffer allocation overflows."));
        dFBufs[i]->BlockLFactor = (Ftype *) SUPERLU_MALLOC(
            symldl_v2_checked_product(
                symldl_v2_checked_product((size_t) ldt, (size_t) ldt,
                                          "diagonal factor buffer allocation overflows."),
                sizeof(Ftype),
                "diagonal factor buffer allocation overflows."));
        if (dFBufs[i]->BlockUFactor == NULL || dFBufs[i]->BlockLFactor == NULL)
            ABORT("Malloc fails for diagonal factor buffers.");
    }
    return dFBufs;
}

template <typename Ftype>
int xLUstruct_t<Ftype>::freeDiagFactBufsArr(
    int_t num_bufs, diagFactBufs_type<Ftype> **dFBufs)
{
    for (int i = 0; i < num_bufs; i++)
    {
        SUPERLU_FREE(dFBufs[i]->BlockUFactor);
        SUPERLU_FREE(dFBufs[i]->BlockLFactor);
        SUPERLU_FREE(dFBufs[i]);
    }

    if (num_bufs)
        SUPERLU_FREE(dFBufs);

    return 0;
}
