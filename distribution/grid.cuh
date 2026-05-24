#pragma once

#include "../core/common.cuh"
#include "../core/types.cuh"

// Pick a near-square factorisation Pr * Pc = world.
inline void choose_grid(int world, int& Pr, int& Pc) {
    Pr = static_cast<int>(std::sqrt(static_cast<double>(world)));
    while (Pr > 1 && (world % Pr != 0)) --Pr;
    if (Pr < 1) Pr = 1;
    Pc = world / Pr;
}

// Even / uneven block split
inline void split_dim(int total, int parts,
                      std::vector<int>& counts,
                      std::vector<int>& offsets) {
    counts.assign(parts, 0);
    offsets.assign(parts, 0);
    int base = total / parts;
    int rem  = total % parts;
    int off  = 0;
    for (int i = 0; i < parts; ++i) {
        counts[i]  = base + (i < rem ? 1 : 0);
        offsets[i] = off;
        off       += counts[i];
    }
}

// Distribute A (M x K, row-major) from rank 0 to (pr, *) ranks.
inline void distribute_A(const std::vector<float>& A_full,   // valid only on rank 0
                         std::vector<float>&       A_stripe, // sized (M_b * K) on every rank
                         const std::vector<int>&   row_counts,
                         const std::vector<int>&   row_offsets,
                         int K, int M_b,
                         const Grid2D& g, int world_rank) {
    A_stripe.assign((size_t)M_b * K, 0.0f);

    // Step 1: rank 0 ships each A-stripe to (pr, 0).
    if (world_rank == 0) {
        for (int pr = 0; pr < g.Pr; ++pr) {
            int dst = pr * g.Pc + 0;
            size_t cnt = (size_t)row_counts[pr] * K;
            const float* src = A_full.data() + (size_t)row_offsets[pr] * K;
            if (dst == 0) {
                std::memcpy(A_stripe.data(), src, cnt * sizeof(float));
            } else {
                MPI_CHECK(MPI_Send(src, (int)cnt, MPI_FLOAT, dst, 100, MPI_COMM_WORLD));
            }
        }
    } else if (g.pc == 0) {
        MPI_CHECK(MPI_Recv(A_stripe.data(), (int)((size_t)M_b * K), MPI_FLOAT,
                           0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    }

    // Step 2: bcast A_stripe within each row-comm; root is local rank 0 (= pc 0).
    MPI_CHECK(MPI_Bcast(A_stripe.data(), (int)((size_t)M_b * K), MPI_FLOAT, 0, g.row_comm));
}

// Distribute B (K x N, row-major) from rank 0 to (*, pc) ranks.
inline void distribute_B(const std::vector<float>& B_full,   // valid only on rank 0
                         std::vector<float>&       B_stripe, // sized (K * N_b)
                         const std::vector<int>&   col_counts,
                         const std::vector<int>&   col_offsets,
                         int K, int N_full, int N_b,
                         const Grid2D& g, int world_rank) {
    B_stripe.assign((size_t)K * N_b, 0.0f);

    if (world_rank == 0) {
        for (int pc = 0; pc < g.Pc; ++pc) {
            int dst = 0 * g.Pc + pc;
            std::vector<float> packed((size_t)K * col_counts[pc]);
            for (int k = 0; k < K; ++k) {
                std::memcpy(packed.data() + (size_t)k * col_counts[pc],
                            B_full.data()  + (size_t)k * N_full + col_offsets[pc],
                            (size_t)col_counts[pc] * sizeof(float));
            }
            if (dst == 0) {
                B_stripe = std::move(packed);
            } else {
                MPI_CHECK(MPI_Send(packed.data(), (int)packed.size(), MPI_FLOAT,
                                   dst, 200, MPI_COMM_WORLD));
            }
        }
    } else if (g.pr == 0) {
        MPI_CHECK(MPI_Recv(B_stripe.data(), (int)((size_t)K * N_b), MPI_FLOAT,
                           0, 200, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    }

    MPI_CHECK(MPI_Bcast(B_stripe.data(), (int)((size_t)K * N_b), MPI_FLOAT, 0, g.col_comm));
}
