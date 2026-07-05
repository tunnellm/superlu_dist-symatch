#pragma once

#include <cstdio>
#include <limits>
#include <vector>

#include "xlupanels.hpp"
#include "symldl_v2_config.hpp"

template <typename Ftype>
static size_t symldl_v2_alloc_bytes(int_t count, size_t elem_size,
                                    const char *what)
{
    if (count < 0)
        ABORT(what);
    size_t n = static_cast<size_t>(count);
    if (elem_size != 0 && n > std::numeric_limits<size_t>::max() / elem_size)
        ABORT(what);
    return n * elem_size;
}

template <typename Ftype>
static void symldl_v2_trace_pcfrag_plan(xLUstruct_t<Ftype> *lu,
                                        const char *where, int_t k,
                                        int_t needed_count)
{
    if (!superlu_sym_v2_trace_pcfrag())
        return;

    static int printed = 0;
    if (printed >= 64 || lu == NULL || lu->grid3d == NULL ||
        k < 0 || k >= lu->nsupers)
        return;

    const std::vector<int_t> &partner =
        lu->symV2PartnerLRecvIndex[static_cast<size_t>(k)];
    const std::vector<int_t> &row =
        lu->symV2RowFragRecvIndex[static_cast<size_t>(k)];
    int_t partner_blocks = partner.empty() ? 0 : partner[0];
    int_t partner_rows = partner.empty() ? 0 : partner[1];
    int_t row_blocks = row.empty() ? 0 : row[0];
    int_t row_rows = row.empty() ? 0 : row[1];

    long long partner_recv = 0;
    size_t partner_base =
        static_cast<size_t>(k) * static_cast<size_t>(lu->Pr);
    if (partner_base + static_cast<size_t>(lu->Pr) <=
        lu->symV2PartnerLRecvSizes.size())
    {
        for (int pr = 0; pr < lu->Pr; ++pr)
            partner_recv += lu->symV2PartnerLRecvSizes[
                partner_base + static_cast<size_t>(pr)];
    }

    long long row_recv = 0;
    size_t row_base = static_cast<size_t>(k) * static_cast<size_t>(lu->Pc);
    if (row_base + static_cast<size_t>(lu->Pc) <=
        lu->symV2RowFragRecvSizes.size())
    {
        for (int pc = 0; pc < lu->Pc; ++pc)
            row_recv += lu->symV2RowFragRecvSizes[
                row_base + static_cast<size_t>(pc)];
    }

    long long row_send = 0;
    int_t lk = lu->symV2PanelIndex(k);
    if (lk >= 0)
    {
        size_t send_base =
            static_cast<size_t>(lk) * static_cast<size_t>(lu->Pc);
        if (send_base + static_cast<size_t>(lu->Pc) <=
            lu->symV2RowDownSendSizes.size())
        {
            for (int pc = 0; pc < lu->Pc; ++pc)
                row_send += lu->symV2RowDownSendSizes[
                    send_base + static_cast<size_t>(pc)];
        }
    }

    if (needed_count == 0 && partner_blocks == 0 && row_blocks == 0 &&
        partner_recv == 0 && row_recv == 0 && row_send == 0)
        return;

    std::fprintf(stderr,
                 "[symv2-pcfrag] rank %d %s k %d panel_root %d diag_root %d needed %lld partner_blocks %lld partner_rows %lld partner_recv %lld row_blocks %lld row_rows %lld row_recv %lld row_send %lld\n",
                 lu->grid3d->iam, where, static_cast<int>(k),
                 static_cast<int>(lu->symV2PanelRoot(k)),
                 static_cast<int>(lu->symV2DiagRoot(k)),
                 static_cast<long long>(needed_count),
                 static_cast<long long>(partner_blocks),
                 static_cast<long long>(partner_rows), partner_recv,
                 static_cast<long long>(row_blocks),
                 static_cast<long long>(row_rows), row_recv, row_send);
    std::fflush(stderr);
    ++printed;
}
