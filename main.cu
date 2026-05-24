// ===================================================================
// main.cu — ABFT GEMM (cuBLAS, online pipelined) — orchestrator
// -------------------------------------------------------------------
// Only TWO modes are now supported:
//   --baseline                 unprotected cuBLAS only
//   (default, no --baseline)   online ABFT (concurrent + cross-iter pipelined)
//   --calibrate                special calibration pass (noise histogram)
// Post-hoc has been removed (not state-of-the-art for the comparison).
// ===================================================================

#include "core/common.cuh"
#include "core/types.cuh"
#include "core/cli.cuh"
#include "distribution/grid.cuh"
#include "kernels/gemm_cublas.cuh"
#include "kernels/abft_stepwise.cuh"
#include "kernels/swifi.cuh"
#include "metrics/metrics.cuh"
#include "pipeline/buffers.cuh"
#include "pipeline/passes.cuh"

int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));

    int world_rank = 0, world_size = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

    ExperimentConfig cfg = parse_args(argc, argv);

    // -----------------------------------------------------------
    // Grid setup
    // -----------------------------------------------------------
    if (cfg.Pr <= 0 || cfg.Pc <= 0) choose_grid(world_size, cfg.Pr, cfg.Pc);
    if (cfg.Pr * cfg.Pc != world_size) {
        if (world_rank == 0) {
            std::cerr << "ERROR: --grid " << cfg.Pr << " " << cfg.Pc
                      << " does not match world_size " << world_size << "\n";
        }
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    Grid2D g{};
    g.Pr = cfg.Pr; g.Pc = cfg.Pc;
    g.pr = world_rank / g.Pc;
    g.pc = world_rank % g.Pc;
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, g.pr, g.pc, &g.row_comm));
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, g.pc, g.pr, &g.col_comm));

    // -----------------------------------------------------------
    // GPU device
    // -----------------------------------------------------------
    int num_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    if (num_gpus <= 0) MPI_Abort(MPI_COMM_WORLD, -1);
    int local_rank = 0;
    if (const char* e  = std::getenv("SLURM_LOCALID"))                 local_rank = std::atoi(e);
    else if (const char* e2 = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK")) local_rank = std::atoi(e2);
    int dev = local_rank % num_gpus;
    CUDA_CHECK(cudaSetDevice(dev));

    // -----------------------------------------------------------
    // Block dims
    // -----------------------------------------------------------
    std::vector<int> row_counts, row_offsets, col_counts, col_offsets;
    split_dim(cfg.M, g.Pr, row_counts, row_offsets);
    split_dim(cfg.N, g.Pc, col_counts, col_offsets);
    int M_b = row_counts[g.pr];
    int N_b = col_counts[g.pc];
    int F   = std::min(cfg.frags_per_rank, N_b);

    // -----------------------------------------------------------
    // Small-matrix optimization (problems with max(M,K,N) < 1024)
    // -----------------------------------------------------------
    // Fragmenting C_local into F column pieces only pays off when the
    // per-fragment SGEMM is large enough that the next fragment's GEMM
    // can hide this fragment's ABFT verification.  For small matrices
    // the SGEMM is so cheap that fragmentation just MULTIPLIES fixed
    // overhead F-fold:  F tiny cuBLAS launches (launch latency dominates,
    // poor SM occupancy) + F actualRow kernels + F host-side
    // cudaEventSynchronize calls — verifying many sub-fragments that each
    // carry only a 1/F slice of the (already low) bit-flip probability.
    //
    // Collapsing to a single fragment keeps protection coverage IDENTICAL
    // (one checksum still spans all of C_local — every element is still
    // verified) while removing the F-fold overhead and giving cuBLAS one
    // properly-sized GEMM.  Bigger matrices are untouched.
    constexpr int SMALL_MATRIX_THRESHOLD = 1024;
    int max_dim = std::max(cfg.M, std::max(cfg.K, cfg.N));
    bool small_matrix_opt = (max_dim < SMALL_MATRIX_THRESHOLD);
    if (small_matrix_opt) F = 1;

    // -----------------------------------------------------------
    // RESILIENCE adaptive-F (only when --frag-cap E > 0; default off)
    // -----------------------------------------------------------
    // ABFT row/col checksums localize+correct at most ONE fault per
    // fragment.  A fixed F=8 is fine for the sizes evaluated here, but a
    // very large stripe (>10k/20k/100k) packs so many elements per
    // fragment that >1 fault becomes likely, breaking that assumption.
    // --frag-cap E bounds elements/fragment by raising F to
    // ceil(M_b*N_b / E) (clamped to [F, N_b]).  This ADDS fragments with
    // size — the opposite of a throughput knob.
    bool resilience_f = false;
    if (!small_matrix_opt && cfg.frag_cap > 0) {
        long long elems = (long long)M_b * (long long)N_b;
        int f_need = (int)((elems + cfg.frag_cap - 1) / cfg.frag_cap);
        if (f_need > F) { F = std::min(f_need, N_b); resilience_f = true; }
    }

    // Pick an internal scheme label so metrics CSV stays meaningful.
    std::string mode_label =
        cfg.calibrate     ? "calibrate" :
        cfg.baseline_only ? "baseline"  :
                            "online";

    if (world_rank == 0) {
        std::cout << "=== ABFT GEMM (cuBLAS, online pipelined) ===\n"
                  << "Mode           : " << mode_label
                  << "  (inject=" << cfg.inject << ")\n"
                  << "M x K x N      : " << cfg.M << " x " << cfg.K
                  << " x " << cfg.N << "\n"
                  << "Process grid   : " << g.Pr << " x " << g.Pc
                  << "  (world=" << world_size << ")\n"
                  << "Frags per rank : " << F
                  << (small_matrix_opt
                        ? "  (small-matrix opt: forced to 1)"
                        : resilience_f
                            ? "  (resilience adaptive-F: raised by --frag-cap)"
                            : "")
                  << "\n"
                  << "Repeats        : " << cfg.repeats << "\n"
                  << "Samples        : "
                  << (cfg.num_samples > 0 ? cfg.num_samples : 5)
                  << (cfg.reseed_per_trial
                        ? "  (fresh A,B per trial)" : "")
                  << "\n";
        if (cfg.threshold_override > 0)
            std::cout << "Threshold      : " << cfg.threshold_override << " (override)\n";
        else
            std::cout << "Threshold      : (per-fragment, formula)\n";
        std::cout << "GPUs visible   : " << num_gpus << "\n\n";
    }

    // -----------------------------------------------------------
    // Device buffers — A, B, and TWO dC buffers for double-buffering.
    // Allocated once; A and B are (re)uploaded by regen_inputs() below.
    // -----------------------------------------------------------
    float *dA = nullptr, *dB = nullptr;
    float *dC_buf0 = nullptr, *dC_buf1 = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(float) * (size_t)M_b * cfg.K));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(float) * (size_t)cfg.K * N_b));
    CUDA_CHECK(cudaMalloc(&dC_buf0, sizeof(float) * (size_t)M_b * N_b));
    CUDA_CHECK(cudaMalloc(&dC_buf1, sizeof(float) * (size_t)M_b * N_b));

    int lda = cfg.K;
    int ldb = N_b;
    int ldc = N_b;

    // -----------------------------------------------------------
    // Pipeline buffers
    // -----------------------------------------------------------
    PipelineBuffers buf{};
    buffers_init(buf, F, M_b, N_b, cfg.K, cfg.repeats);

    // -----------------------------------------------------------
    // (Re)generate A,B for the CURRENT seed, upload to dA/dB, and
    // recompute the per-fragment thresholds from the new B.  Called once
    // before the trial loop in legacy mode, and per-trial when the
    // `--reseed-per-trial` flag is set.
    // -----------------------------------------------------------
    std::vector<float> A_stripe(static_cast<size_t>(M_b) * cfg.K);
    std::vector<float> B_stripe(static_cast<size_t>(cfg.K) * N_b);
    std::vector<double> thresholds(F, 0.0);

    auto regen_inputs = [&](uint64_t s_a, uint64_t s_b) {
        std::vector<float> A_full, B_full;
        if (world_rank == 0) {
            A_full.resize((size_t)cfg.M * cfg.K);
            B_full.resize((size_t)cfg.K * cfg.N);
            fill_random(A_full, s_a);
            fill_random(B_full, s_b);
        }
        A_stripe.assign((size_t)M_b * cfg.K, 0.0f);
        B_stripe.assign((size_t)cfg.K * N_b, 0.0f);
        distribute_A(A_full, A_stripe, row_counts, row_offsets,
                     cfg.K, M_b, g, world_rank);
        distribute_B(B_full, B_stripe, col_counts, col_offsets,
                     cfg.K, cfg.N, N_b, g, world_rank);
        CUDA_CHECK(cudaMemcpy(dA, A_stripe.data(),
                              sizeof(float) * (size_t)M_b * cfg.K,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, B_stripe.data(),
                              sizeof(float) * (size_t)cfg.K * N_b,
                              cudaMemcpyHostToDevice));

        // Thresholds (formula or override) — recomputed because B changed.
        double norm_A = matrix_inf_norm(A_stripe.data(), M_b, cfg.K, cfg.K);
        for (int f = 0; f < F; ++f) {
            if (cfg.threshold_override > 0.0) {
                thresholds[f] = cfg.threshold_override;
            } else {
                double mx = 0.0;
                for (int k = 0; k < cfg.K; ++k) {
                    double rs = 0.0;
                    for (int j = 0; j < buf.col_counts[f]; ++j) {
                        rs += std::abs((double)B_stripe[(size_t)k * N_b
                                                        + buf.col_offsets[f] + j]);
                    }
                    if (rs > mx) mx = rs;
                }
                thresholds[f] = compute_threshold_formula(cfg.K, norm_A, mx);
            }
        }
    };

    // Initial generation (also the only call in legacy / fixed-seed mode).
    regen_inputs(cfg.seed_a, cfg.seed_b);

    // -----------------------------------------------------------
    // Measurement methodology
    // -----------------------------------------------------------
    // Each "sample" is one full TRIAL: a `cfg.repeats`-iter loop timed
    // end-to-end (MPI_Barrier → loop → final sync), divided by `cfg.repeats`.
    // Both baseline and online use the same timing scope, so the comparison
    // is apples-to-apples and the online time includes all ABFT work.
    //
    // The first NUM_WARMUP_TRIALS are DISCARDED — at the real problem shape
    // (not the toy gemm_warmup), so cuBLAS picks its heuristic algorithm
    // and GPU clocks settle before any sample is recorded.
    // -----------------------------------------------------------
    const int NUM_TRIALS        = cfg.num_samples > 0 ? cfg.num_samples : 5;
    const int NUM_WARMUP_TRIALS = cfg.warmups;            // --warmups (default 2)
    const int TOTAL_TRIALS      = NUM_TRIALS + NUM_WARMUP_TRIALS;

    // Quick cuBLAS library-load nudge (does its own 64x64 mini-GEMM).
    // Real warmup comes from the discarded trials below.
    gemm_warmup(buf.handle, buf.compute_stream);

    std::vector<double> baseline_samples;
    std::vector<double> protected_samples;
    baseline_samples.reserve(NUM_TRIALS);
    protected_samples.reserve(NUM_TRIALS);
    ConfusionMatrix cm_local{};
    int    n_restored_local = 0;
    double calib_max_local  = 0.0;
    std::vector<double> calib_diffs_local;
    if (cfg.calibrate) calib_diffs_local.reserve((size_t)N_b * (size_t)cfg.repeats);

    std::vector<float> C_golden;
    const bool need_golden = (!cfg.baseline_only && !cfg.calibrate
                              && cfg.inject == "swifi");

    if (cfg.calibrate) {
        // Calibration is not a timing comparison — collect noise samples only.
        for (int it = 0; it < cfg.repeats; ++it) {
            double t = pass_calibrate(buf, dA, lda, dB, ldb, dC_buf0, ldc,
                                      M_b, cfg.K, N_b,
                                      calib_max_local, calib_diffs_local);
            protected_samples.push_back(t);
        }
    } else {
        // --------------------------------------------------------
        // Trial loop.  Each trial measures BOTH baseline and protected
        // back-to-back on the SAME (A,B); with --reseed-per-trial those
        // matrices change every trial so every sample is independent.
        // First NUM_WARMUP_TRIALS are discarded.
        // --------------------------------------------------------
        for (int t = 0; t < TOTAL_TRIALS; ++t) {
            const bool record = (t >= NUM_WARMUP_TRIALS);

            if (cfg.reseed_per_trial) {
                // Per-trial deterministic but distinct seeds — keeps runs
                // reproducible while giving every sample fresh A,B.
                uint64_t s_a = cfg.seed_a + (uint64_t)t * 1000003ull;
                uint64_t s_b = cfg.seed_b + (uint64_t)t * 1000033ull;
                regen_inputs(s_a, s_b);
            }

            // ---- Baseline trial (also produces golden in dC_buf0) ----
            double trial_ms_b = pass_baseline(buf, dA, lda, dB, ldb,
                                              dC_buf0, ldc,
                                              M_b, cfg.K, N_b, cfg.repeats);
            if (record) baseline_samples.push_back(trial_ms_b / cfg.repeats);

            // Capture golden (only needed for SWIFI accuracy validation).
            // In legacy mode capture once on the last warmup; in reseed mode
            // we need it every trial because A,B just changed.
            if (need_golden && (cfg.reseed_per_trial
                                || t == NUM_WARMUP_TRIALS - 1
                                || (NUM_WARMUP_TRIALS == 0 && t == 0))) {
                if (C_golden.size() != (size_t)M_b * (size_t)N_b)
                    C_golden.resize((size_t)M_b * (size_t)N_b);
                CUDA_CHECK(cudaMemcpy(C_golden.data(), dC_buf0,
                                      sizeof(float) * (size_t)M_b * N_b,
                                      cudaMemcpyDeviceToHost));
            }

            if (cfg.baseline_only) {
                if (record) {
                    protected_samples.push_back(trial_ms_b / cfg.repeats);
                    cm_local.TN += F;
                }
                continue;
            }

            // ---- Protected trial ----
            std::vector<double> iter_ms_unused;
            double total_ms = 0.0;
            ConfusionMatrix cm_trial{};
            int n_restored_trial = 0;
            pass_online_loop(buf, dA, lda, dB, ldb, dC_buf0, dC_buf1, ldc,
                             M_b, cfg.K, N_b,
                             thresholds, cfg.inject, cfg.swifi_zone,
                             cfg.seed_a + cfg.seed_b
                                 + (uint64_t)t * 31337ull,
                             world_rank,
                             C_golden,
                             cfg.repeats,
                             iter_ms_unused, cm_trial,
                             n_restored_trial, total_ms);
            if (record) {
                protected_samples.push_back(total_ms / cfg.repeats);
                cm_local.TP += cm_trial.TP;
                cm_local.TN += cm_trial.TN;
                cm_local.FP += cm_trial.FP;
                cm_local.FN += cm_trial.FN;
                n_restored_local += n_restored_trial;
            }
        }

        if (world_rank == 0 && !cfg.baseline_only) {
            std::cout << "[Rank 0] " << NUM_TRIALS << " timed trials × "
                      << cfg.repeats << " iters/trial"
                      << (cfg.reseed_per_trial
                            ? "  (fresh A,B per trial)" : "")
                      << "\n";
        }
    }
    TimingStats baseline_local  = stats_of(baseline_samples);
    TimingStats protected_local = stats_of(protected_samples);

    // -----------------------------------------------------------
    // MPI aggregation onto rank 0
    // -----------------------------------------------------------
    ConfusionMatrix cm_global   = mpi_reduce_cm(cm_local, MPI_COMM_WORLD);
    int    n_restored_global    = mpi_reduce_int_sum(n_restored_local, MPI_COMM_WORLD);
    TimingStats baseline_global = mpi_reduce_timing_max(baseline_local, MPI_COMM_WORLD);
    TimingStats protected_global= mpi_reduce_timing_max(protected_local, MPI_COMM_WORLD);
    double calib_max_global     = mpi_reduce_double_max(calib_max_local, MPI_COMM_WORLD);

    // -----------------------------------------------------------
    // Calibration diff dump (only on rank 0)
    // -----------------------------------------------------------
    if (cfg.calibrate) {
        int local_count = static_cast<int>(calib_diffs_local.size());
        std::vector<int> counts(world_size, 0), displs(world_size, 0);
        MPI_CHECK(MPI_Gather(&local_count, 1, MPI_INT,
                             counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD));
        std::vector<double> all_diffs;
        if (world_rank == 0) {
            int total = 0;
            for (int r = 0; r < world_size; ++r) {
                displs[r] = total;
                total    += counts[r];
            }
            all_diffs.resize((size_t)total);
        }
        MPI_CHECK(MPI_Gatherv(calib_diffs_local.data(), local_count, MPI_DOUBLE,
                              world_rank == 0 ? all_diffs.data() : nullptr,
                              counts.data(), displs.data(), MPI_DOUBLE,
                              0, MPI_COMM_WORLD));
        if (world_rank == 0) {
            std::ofstream ofs(cfg.calib_diffs_path);
            if (ofs.is_open()) {
                ofs << "abs_diff\n";
                ofs << std::scientific << std::setprecision(8);
                for (double d : all_diffs) ofs << d << "\n";
                std::cout << "Calibration diffs written : "
                          << cfg.calib_diffs_path
                          << "  (" << all_diffs.size() << " samples)\n";
            }
        }
    }

    // -----------------------------------------------------------
    // Build & emit metrics on rank 0
    // -----------------------------------------------------------
    if (world_rank == 0) {
        ExperimentMetrics m{};
        m.cm                       = cm_global;
        m.baseline                 = baseline_global;
        m.protected_               = protected_global;
        m.n_successfully_restored  = n_restored_global;
        m.M = cfg.M; m.K = cfg.K; m.N = cfg.N;
        m.Pr = g.Pr; m.Pc = g.Pc;
        m.frags_per_rank           = F;
        m.num_fragments_total      = g.Pr * g.Pc * F;
        m.repeats                  = cfg.repeats;
        m.scheme                   = mode_label;
        m.inject                   = cfg.inject;
        m.swifi_zone               = cfg.swifi_zone;
        m.baseline_only            = cfg.baseline_only;
        m.calibrate                = cfg.calibrate;
        m.threshold_used           = (cfg.threshold_override > 0.0)
                                     ? cfg.threshold_override
                                     : (thresholds.empty() ? 0.0 : thresholds[0]);
        if (cfg.calibrate) {
            m.calib_max_diff       = calib_max_global;
            m.calib_safety_factor  = cfg.calibration_safety_factor;
            m.calib_suggested_tau  = calib_max_global * cfg.calibration_safety_factor;
        }
        finalize_metrics(m);
        print_metrics(m);
        log_metrics_csv(m, cfg.csv_path);
    }

    // -----------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------
    buffers_free(buf);
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC_buf0));
    CUDA_CHECK(cudaFree(dC_buf1));
    MPI_Comm_free(&g.row_comm);
    MPI_Comm_free(&g.col_comm);
    MPI_CHECK(MPI_Finalize());
    return 0;
}
