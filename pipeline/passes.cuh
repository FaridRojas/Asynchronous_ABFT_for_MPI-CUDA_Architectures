#pragma once

#include "../core/common.cuh"
#include "../core/types.cuh"
#include "../kernels/gemm_cublas.cuh"
#include "../kernels/abft_stepwise.cuh"
#include "../kernels/swifi.cuh"
#include "../metrics/metrics.cuh"
#include "buffers.cuh"

// ===================================================================
// Pass functions
//
//   pass_baseline   — F cuBLAS calls only. Reference for overhead.
//   pass_calibrate  — F cuBLAS calls + ABFT row checksums (no detection),
//                     records every |actualRow - expectedRow| sample.
//   pass_online_loop— pipelined online ABFT: verification concurrent with
//                     cuBLAS via two streams, AND localization of iter k
//                     overlaps with cuBLAS of iter k+1 via double-buffered
//                     dC.  Pending corrections are processed after the loop.
// ===================================================================

// ---------------------------------------------------------------------------
// pass_baseline — one timing TRIAL of `repeats` unprotected GEMM iters.
//
// Returns the total wall-clock time (ms) of the whole trial, measured between
// `MPI_Barrier` and the final `cudaStreamSynchronize`.  The caller divides by
// `repeats` to get the mean per-iter time.
//
// IMPORTANT: this matches `pass_online_loop`'s timing scope (whole-loop wall
// clock).  Earlier per-iter timing was apples-to-oranges vs. the online path,
// which inflated baseline numbers and made the protected path look free.
// ---------------------------------------------------------------------------
inline double pass_baseline(PipelineBuffers& b,
                            const float* dA, int lda,
                            const float* dB, int ldb,
                            float*       dC, int ldc,
                            int M_b, int K, int N_b,
                            int repeats) {
    (void)N_b;
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto t0 = clk::now();

    for (int it = 0; it < repeats; ++it) {
        for (int f = 0; f < b.F; ++f) {
            int N_frag = b.col_counts[f];
            int off    = b.col_offsets[f];
            gemm_cublas(b.handle, dA, lda, dB + off, ldb, dC + off, ldc,
                        M_b, N_frag, K);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(b.compute_stream));
    auto t1 = clk::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ---------------------------------------------------------------------------
// pass_calibrate — clean GEMM + checksums, no detection.
// ---------------------------------------------------------------------------
inline double pass_calibrate(PipelineBuffers& b,
                             const float* dA, int lda,
                             const float* dB, int ldb,
                             float*       dC, int ldc,
                             int M_b, int K, int N_b,
                             double& out_max_diff,
                             std::vector<double>& diffs_out) {
    (void)N_b;
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto t0 = clk::now();

    for (int f = 0; f < b.F; ++f) {
        int N_frag = b.col_counts[f];
        int off    = b.col_offsets[f];
        gemm_cublas(b.handle, dA, lda, dB + off, ldb, dC + off, ldc,
                    M_b, N_frag, K);
    }
    CUDA_CHECK(cudaStreamSynchronize(b.compute_stream));

    launch_col_checksum_A(dA, lda, b.dColSumA, M_b, K, b.verify_stream);
    CUDA_CHECK(cudaStreamSynchronize(b.verify_stream));

    // Calibration is OFFLINE (not the timed perf path), so the host
    // round-trip here is harmless; use local host temporaries since the
    // pinned per-fragment staging was removed with the device-resident
    // online refactor.
    std::vector<double> hExp(b.N_frag_max), hAct(b.N_frag_max);
    for (int f = 0; f < b.F; ++f) {
        int N_frag = b.col_counts[f];
        int off    = b.col_offsets[f];
        launch_expected_row(b.dColSumA, dB + off, ldb,
                            b.dExpectedRow[f], K, N_frag, b.verify_stream);
        launch_actual_row  (dC + off,   ldc,    b.dActualRow  [f],
                            M_b, N_frag, b.verify_stream);
        CUDA_CHECK(cudaMemcpyAsync(hExp.data(), b.dExpectedRow[f],
                                   sizeof(double) * N_frag,
                                   cudaMemcpyDeviceToHost, b.verify_stream));
        CUDA_CHECK(cudaMemcpyAsync(hAct.data(), b.dActualRow[f],
                                   sizeof(double) * N_frag,
                                   cudaMemcpyDeviceToHost, b.verify_stream));
        CUDA_CHECK(cudaStreamSynchronize(b.verify_stream));

        for (int j = 0; j < N_frag; ++j) {
            double d = std::abs(hAct[j] - hExp[j]);
            diffs_out.push_back(d);
            if (d > out_max_diff) out_max_diff = d;
        }
    }

    auto t1 = clk::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ===================================================================
// Pipelined online ABFT loop — FULLY DEVICE-RESIDENT
// -------------------------------------------------------------------
// Detection, localisation and correction are now GPU kernels.  Inside
// the timed loop there is ZERO host round-trip: no per-iteration DtoH
// of checksum vectors, no cudaEventSynchronize, no host compare.  The
// verify stream runs entirely concurrently with the compute stream;
// the host only reads four CM counters + n_restored ONCE after the loop.
//
//   - colSumA + expectedRow_f (A,B only) are precomputed ONCE pre-loop.
//   - Per iter, per fragment, on verify_stream (gated by compute_done):
//       actualRow_f  ->  k_detect_row  (writes dErrCol[f]/dRowDiff[f],
//                                        folds TP/TN/FP/FN into dCM)
//     When injection is enabled, the localise+correct chain
//     (rowSumB, expectedCol, actualCol, locate+correct) is also queued;
//     every one of those kernels is DEVICE-GATED by dErrCol[f] so it is
//     a launch-latency no-op on the (overwhelming) clean path.
//   - Two dC buffers alternated by parity; iter k+2 waits on
//     buf_verify_done[k%2] before overwriting that buffer.
//   - The whole detect chain evaluates EVERY fragment every iter (the
//     old host path broke after the first detection, skipping the rest);
//     this makes the confusion matrix strictly more complete.
// ===================================================================

inline void pass_online_loop(PipelineBuffers& b,
                             const float* dA, int lda,
                             const float* dB, int ldb,
                             float* dC_buf0, float* dC_buf1, int ldc,
                             int M_b, int K, int N_b,
                             const std::vector<double>& thresholds,
                             const std::string& inject_mode,
                             const std::string& inject_zone,
                             uint64_t base_seed, int world_rank,
                             const std::vector<float>& C_golden,
                             int repeats,
                             std::vector<double>& out_iter_ms,
                             ConfusionMatrix& cm,
                             int& n_restored,
                             double& out_total_ms) {
    float* dC_bufs[2] = { dC_buf0, dC_buf1 };
    // "swifi" = real bit-flip (accuracy);  "add" = large additive fault
    // that always trips the threshold (overhead studies).  Both exercise
    // the full detect+localize+correct path.
    const bool inject_on  = (inject_mode == "swifi" || inject_mode == "add");
    const bool do_localize = inject_on;

    // --- Optional device golden (only for the restore-success metric) ---
    if (do_localize && !C_golden.empty() && b.dGolden == nullptr) {
        CUDA_CHECK(cudaMalloc(&b.dGolden,
                              sizeof(float) * (size_t)M_b * (size_t)N_b));
        CUDA_CHECK(cudaMemcpy(b.dGolden, C_golden.data(),
                              sizeof(float) * (size_t)M_b * (size_t)N_b,
                              cudaMemcpyHostToDevice));
    }

    // --- Pre-loop: colSumA + all expectedRow_f (A,B only), no DtoH ---
    launch_col_checksum_A(dA, lda, b.dColSumA, M_b, K, b.verify_stream);
    for (int f = 0; f < b.F; ++f) {
        int N_frag = b.col_counts[f];
        int off    = b.col_offsets[f];
        launch_expected_row(b.dColSumA, dB + off, ldb,
                            b.dExpectedRow[f], K, N_frag, b.verify_stream);
    }
    // Zero the device aggregate counters.
    CUDA_CHECK(cudaMemsetAsync(b.dCM, 0, sizeof(int) * 4, b.verify_stream));
    CUDA_CHECK(cudaMemsetAsync(b.dNRestored, 0, sizeof(int), b.verify_stream));
    CUDA_CHECK(cudaStreamSynchronize(b.verify_stream));

    // --- Buffer-reuse synchronisation events: one per dC buffer ---
    cudaEvent_t buf_verify_done[2];
    bool buf_event_used[2] = { false, false };
    for (int i = 0; i < 2; ++i)
        CUDA_CHECK(cudaEventCreateWithFlags(&buf_verify_done[i],
                                            cudaEventDisableTiming));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto loop_t0 = clk::now();

    // ============================ MAIN LOOP ===========================
    for (int it = 0; it < repeats; ++it) {
        int buf_idx = it % 2;
        float* dC   = dC_bufs[buf_idx];

        if (buf_event_used[buf_idx]) {
            CUDA_CHECK(cudaStreamWaitEvent(b.compute_stream,
                                           buf_verify_done[buf_idx], 0));
        }

        // ---- Per-iter SWIFI configuration (host picks WHICH frag only) ----
        int      inject_frag = -1;
        uint64_t inject_seed = base_seed
                             + (uint64_t)world_rank * 7919ull
                             + (uint64_t)it          * 104729ull;
        if (inject_on) {
            std::mt19937_64 rng(inject_seed);
            std::uniform_int_distribution<int> d(0, b.F - 1);
            inject_frag = d(rng);
        }

        // ---- SGEMMs into dC_bufs[buf_idx] ----
        for (int f = 0; f < b.F; ++f) {
            int N_frag = b.col_counts[f];
            int off    = b.col_offsets[f];
            gemm_cublas(b.handle, dA, lda, dB + off, ldb, dC + off, ldc,
                        M_b, N_frag, K);
            if (f == inject_frag) {
                if (inject_mode == "add")
                    inject_add_constant(dC + off, ldc, M_b, N_frag, f,
                                        inject_seed, b.compute_stream);
                else
                    inject_single_bitflip(dC + off, ldc, M_b, N_frag, f,
                                          inject_seed, b.compute_stream,
                                          inject_zone);
            }
            CUDA_CHECK(cudaEventRecord(b.compute_done[f], b.compute_stream));
        }

        // ---- Device-resident verify+localize+correct on verify_stream ----
        for (int f = 0; f < b.F; ++f) {
            int N_frag = b.col_counts[f];
            int off    = b.col_offsets[f];
            int injected = (f == inject_frag) ? 1 : 0;

            CUDA_CHECK(cudaStreamWaitEvent(b.verify_stream,
                                           b.compute_done[f], 0));
            launch_actual_row(dC + off, ldc, b.dActualRow[f],
                              M_b, N_frag, b.verify_stream);
            launch_detect_row(b.dExpectedRow[f], b.dActualRow[f],
                              N_frag, thresholds[f],
                              b.dErrCol + f, b.dRowDiff + f,
                              injected, b.dCM, b.verify_stream);
            if (do_localize) {
                // Every kernel below early-returns on the device when
                // dErrCol[f] < 0 (clean) — launch latency only.
                launch_localize_correct(dA, lda, dB + off, ldb,
                                        dC + off, ldc,
                                        b.dRowSumB, b.dExpectedCol,
                                        b.dActualCol,
                                        M_b, K, N_frag, thresholds[f],
                                        b.dErrCol + f, b.dRowDiff + f,
                                        b.dGolden, N_b, off, injected,
                                        b.dNRestored, b.verify_stream);
            }
        }

        CUDA_CHECK(cudaEventRecord(buf_verify_done[buf_idx], b.verify_stream));
        buf_event_used[buf_idx] = true;
    }

    // ============================ POST-LOOP ===========================
    CUDA_CHECK(cudaStreamSynchronize(b.compute_stream));
    CUDA_CHECK(cudaStreamSynchronize(b.verify_stream));

    auto loop_t1 = clk::now();
    out_total_ms = std::chrono::duration<double, std::milli>(loop_t1 - loop_t0).count();

    // Single host read of the device aggregates (4 ints + 1 int).
    int hCM[4] = {0, 0, 0, 0};
    int hNR    = 0;
    CUDA_CHECK(cudaMemcpy(hCM, b.dCM, sizeof(int) * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&hNR, b.dNRestored, sizeof(int), cudaMemcpyDeviceToHost));
    cm.TP += hCM[0];
    cm.TN += hCM[1];
    cm.FP += hCM[2];
    cm.FN += hCM[3];
    n_restored += hNR;

    // out_iter_ms is diagnostic only (per-iter timing was a host sync we
    // removed); report the loop mean so the field stays meaningful.
    out_iter_ms.assign(repeats,
                       repeats > 0 ? out_total_ms / repeats : 0.0);

    for (int i = 0; i < 2; ++i) cudaEventDestroy(buf_verify_done[i]);
}
