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

    launch_col_checksum_A(dA, lda, b.dColSumA, M_b, K, b.verify_stream,
                          b.dEncPart);
    CUDA_CHECK(cudaStreamSynchronize(b.verify_stream));

    // Calibration
    std::vector<double> hExp(b.N_frag_max), hAct(b.N_frag_max);
    for (int f = 0; f < b.F; ++f) {
        int N_frag = b.col_counts[f];
        int off    = b.col_offsets[f];
        launch_expected_row(b.dColSumA, dB + off, ldb,
                            b.dExpectedRow[f], K, N_frag, b.verify_stream,
                            b.dEncPart);
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

// Pipelined online ABFT loop — FULLY DEVICE-RESIDENT

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
                             double& out_total_ms,
                             const std::string& encoding_mode = "amortized") {
    float* dC_bufs[2] = { dC_buf0, dC_buf1 };
    // The second buffer is only indexed when repeats > 1 (iteration parity),
    // so main.cu skips allocating it for repeats == 1.  Fail loudly rather
    // than dereference a null pointer if that invariant is ever broken.
    if (repeats > 1 && dC_buf1 == nullptr) {
        std::cerr << "[pass_online_loop] repeats=" << repeats
                  << " requires the second C buffer, but it was not allocated"
                  << " (see the repeats>1 guard in main.cu).\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // "swifi" = real bit-flip (accuracy);  "add" = large additive fault
    // that always trips the threshold (overhead studies).  Both exercise
    // the full detect+localize+correct path.
    const bool inject_on  = (inject_mode == "swifi" || inject_mode == "add");
    const bool do_localize = inject_on;

    // --- Optional device golden (only for the restore-success metric) ---
    // Allocated once, but re-uploaded on EVERY call: under
    // --reseed-per-trial the operands change each trial, so the golden C
    // changes with them.  Uploading only on first allocation left the
    // device holding trial 0's golden, which no later corrected value can
    // match, and the correction-precision metric collapsed to 0.
    // The copy sits outside the timed window and only runs for SWIFI
    // (C_golden is empty in the timing sweeps), so it costs nothing there.
    if (do_localize && !C_golden.empty()) {
        if (b.dGolden == nullptr)
            CUDA_CHECK(cudaMalloc(&b.dGolden,
                                  sizeof(float) * (size_t)M_b * (size_t)N_b));
        CUDA_CHECK(cudaMemcpy(b.dGolden, C_golden.data(),
                              sizeof(float) * (size_t)M_b * (size_t)N_b,
                              cudaMemcpyHostToDevice));
    }

    // --- Buffer-reuse synchronisation events: one per dC buffer.
    //     Host-side setup, created before the timer regardless of mode. ---
    cudaEvent_t buf_verify_done[2];
    bool buf_event_used[2] = { false, false };
    for (int i = 0; i < 2; ++i)
        CUDA_CHECK(cudaEventCreateWithFlags(&buf_verify_done[i],
                                            cudaEventDisableTiming));

    // The one-time input encoding (colSumA + all expectedRow_f) depends only
    // on A,B — the same data the GEMMs read and never write — so it can sit
    // in three places relative to the timed window (see ExperimentConfig):
    //   amortized  enqueue + sync BEFORE the window (excluded; operand reuse)
    //   timed      enqueue + sync INSIDE the window (charged sequentially)
    //   overlap    enqueue INSIDE the window, no sync: the encoding kernels
    //              run on verify_stream while iteration 0's GEMMs run on
    //              compute_stream.  The verify stream is in-order, so every
    //              expectedRow_f completes before the first k_detect_row that
    //              reads it — correctness needs no extra synchronization.
    auto enqueue_encode = [&]() {
        launch_col_checksum_A(dA, lda, b.dColSumA, M_b, K, b.verify_stream,
                              b.dEncPart);
        for (int f = 0; f < b.F; ++f) {
            int N_frag = b.col_counts[f];
            int off    = b.col_offsets[f];
            launch_expected_row(b.dColSumA, dB + off, ldb,
                                b.dExpectedRow[f], K, N_frag, b.verify_stream,
                                b.dEncPart);
        }
        CUDA_CHECK(cudaMemsetAsync(b.dCM, 0, sizeof(int) * 4, b.verify_stream));
        CUDA_CHECK(cudaMemsetAsync(b.dNRestored, 0, sizeof(int), b.verify_stream));
    };

    clk::time_point loop_t0;
    if (encoding_mode == "timed") {
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        loop_t0 = clk::now();
        enqueue_encode();
        CUDA_CHECK(cudaStreamSynchronize(b.verify_stream));
    } else if (encoding_mode == "overlap") {
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        loop_t0 = clk::now();
        enqueue_encode();
    } else {  // "amortized"
        enqueue_encode();
        CUDA_CHECK(cudaStreamSynchronize(b.verify_stream));
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        loop_t0 = clk::now();
    }

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

    // out_iter_ms is diagnostic only, report the loop mean so the field stays meaningful.
    out_iter_ms.assign(repeats,
                       repeats > 0 ? out_total_ms / repeats : 0.0);

    for (int i = 0; i < 2; ++i) cudaEventDestroy(buf_verify_done[i]);
}
