#pragma once

#include "../core/common.cuh"
#include "../core/types.cuh"

// Metrics Module — confusion matrix, derived metrics, MPI aggregation,

struct ConfusionMatrix {
    int TP = 0;   // injected AND detected
    int TN = 0;   // not injected AND not detected
    int FP = 0;   // not injected BUT detected
    int FN = 0;   // injected BUT not detected
};

struct ExperimentMetrics {
    ConfusionMatrix cm;          // global (rank-aggregated) confusion matrix

    TimingStats baseline;        // timing stats for the unprotected GEMM
    TimingStats protected_;      // timing stats for the chosen scheme
    double      overhead_pct = 0.0;

    // GFLOPS throughput (whole-problem 2*M*N*K, worst-rank time).
    double baseline_gflops_med = 0.0;
    double baseline_gflops_min = 0.0;
    double baseline_gflops_max = 0.0;
    double baseline_gflops_mean = 0.0;   // GFLOPS from the mean time
    double protected_gflops_med = 0.0;
    double protected_gflops_min = 0.0;
    double protected_gflops_max = 0.0;
    double protected_gflops_mean = 0.0;  // GFLOPS from the mean time

    double recall                  = 0.0;
    double precision               = 0.0;
    double correction_precision_pct = 0.0;
    int    n_successfully_restored = 0;

    int M = 0, K = 0, N = 0;
    int Pr = 0, Pc = 0;
    int frags_per_rank      = 0;
    int num_fragments_total = 0;
    int repeats             = 1;

    double threshold_used   = 0.0;   // single number printed in summary

    std::string scheme;          // "post" or "in-situ"
    std::string inject;          // "none" or "swifi"
    std::string swifi_zone = "any";  // any|sign|exponent|sig_high|sig_low
    bool        baseline_only = false;
    bool        calibrate     = false;

    // Calibration outputs (only meaningful when calibrate=true)
    double calib_max_diff       = 0.0;
    double calib_safety_factor  = 0.0;
    double calib_suggested_tau  = 0.0;
};

// Confusion-matrix bookkeeping
inline void update_confusion_matrix(ConfusionMatrix& cm,
                                    bool fault_injected,
                                    bool fault_detected) {
    if      ( fault_injected &&  fault_detected) cm.TP++;
    else if (!fault_injected && !fault_detected) cm.TN++;
    else if (!fault_injected &&  fault_detected) cm.FP++;
    else                                         cm.FN++;
}

inline ConfusionMatrix mpi_reduce_cm(const ConfusionMatrix& local, MPI_Comm comm) {
    int local_arr [4] = { local.TP,  local.TN,  local.FP,  local.FN  };
    int global_arr[4] = { 0, 0, 0, 0 };
    MPI_CHECK(MPI_Reduce(local_arr, global_arr, 4, MPI_INT, MPI_SUM, 0, comm));
    ConfusionMatrix g;
    g.TP = global_arr[0]; g.TN = global_arr[1];
    g.FP = global_arr[2]; g.FN = global_arr[3];
    return g;
}

inline int mpi_reduce_int_sum(int local, MPI_Comm comm) {
    int g = 0;
    MPI_CHECK(MPI_Reduce(&local, &g, 1, MPI_INT, MPI_SUM, 0, comm));
    return g;
}

inline double mpi_reduce_double_max(double local, MPI_Comm comm) {
    double g = 0.0;
    MPI_CHECK(MPI_Reduce(&local, &g, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
    return g;
}

inline TimingStats mpi_reduce_timing_max(const TimingStats& local, MPI_Comm comm) {
    // For wall-clock comparisons we take the slowest rank for each statistic.
    double in_arr [4] = { local.min_ms, local.median_ms, local.mean_ms,
                          local.max_ms };
    double out_arr[4] = { 0, 0, 0, 0 };
    MPI_CHECK(MPI_Reduce(in_arr, out_arr, 4, MPI_DOUBLE, MPI_MAX, 0, comm));
    TimingStats r;
    r.min_ms    = out_arr[0];
    r.median_ms = out_arr[1];
    r.mean_ms   = out_arr[2];
    r.max_ms    = out_arr[3];
    r.n_samples = local.n_samples;
    return r;
}

// Derived metrics
inline double compute_recall(const ConfusionMatrix& cm) {
    int d = cm.TP + cm.FN;
    return (d > 0) ? static_cast<double>(cm.TP) / d : 0.0;
}

inline double compute_precision(const ConfusionMatrix& cm) {
    int d = cm.TP + cm.FP;
    return (d > 0) ? static_cast<double>(cm.TP) / d : 0.0;
}

inline double compute_correction_precision(int restored, int TP) {
    return (TP > 0) ? (static_cast<double>(restored) / TP) * 100.0 : 0.0;
}

inline double compute_runtime_overhead(double t_protected_ms, double t_baseline_ms) {
    return (t_baseline_ms > 0.0)
        ? ((t_protected_ms - t_baseline_ms) / t_baseline_ms) * 100.0
        : 0.0;
}

// GFLOPS/s for a GEMM of size M×N×K given the per-iter wall time (ms).
inline double compute_gflops(double time_ms, int M, int N, int K) {
    if (time_ms <= 0.0) return 0.0;
    double flops = 2.0 * (double)M * (double)N * (double)K;
    return flops / (time_ms * 1e6);   // (flops / 1e9) / (time_ms / 1e3)
}

inline void finalize_metrics(ExperimentMetrics& m) {
    m.overhead_pct           = compute_runtime_overhead(m.protected_.median_ms,
                                                        m.baseline.median_ms);
    m.recall                 = compute_recall(m.cm);
    m.precision              = compute_precision(m.cm);
    m.correction_precision_pct = compute_correction_precision(m.n_successfully_restored,
                                                              m.cm.TP);

    // GFLOPS — note: fastest time → max GFLOPS, slowest time → min GFLOPS.
    m.baseline_gflops_med  = compute_gflops(m.baseline.median_ms, m.M, m.N, m.K);
    m.baseline_gflops_max  = compute_gflops(m.baseline.min_ms,    m.M, m.N, m.K);
    m.baseline_gflops_min  = compute_gflops(m.baseline.max_ms,    m.M, m.N, m.K);
    m.baseline_gflops_mean = compute_gflops(m.baseline.mean_ms,   m.M, m.N, m.K);
    m.protected_gflops_med = compute_gflops(m.protected_.median_ms, m.M, m.N, m.K);
    m.protected_gflops_max = compute_gflops(m.protected_.min_ms,    m.M, m.N, m.K);
    m.protected_gflops_min = compute_gflops(m.protected_.max_ms,    m.M, m.N, m.K);
    m.protected_gflops_mean = compute_gflops(m.protected_.mean_ms,  m.M, m.N, m.K);
}

// Pretty-print (rank 0 only — caller's responsibility)
inline void print_metrics(const ExperimentMetrics& m) {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n========== EXPERIMENT METRICS (global) ==========\n";
    if (m.calibrate) {
        std::cout << "Mode              : CALIBRATION\n";
    } else if (m.baseline_only) {
        std::cout << "Mode              : BASELINE\n";
    } else {
        std::cout << "Mode              : " << m.scheme
                  << " / inject=" << m.inject;
        if (m.inject == "swifi") std::cout << " (zone=" << m.swifi_zone << ")";
        std::cout << "\n";
    }
    std::cout << "M x K x N         : " << m.M << " x " << m.K << " x " << m.N << "\n";
    std::cout << "Process grid      : " << m.Pr << " x " << m.Pc << "\n";
    std::cout << "Frags per rank    : " << m.frags_per_rank << "\n";
    std::cout << "Total fragments   : " << m.num_fragments_total << "\n";
    std::cout << "Repeats           : " << m.repeats << "\n";
    if (!m.calibrate) {
        std::cout << "Threshold used    : " << m.threshold_used << "\n";
    }

    std::cout << "--- Timing (worst rank) ---\n";
    std::cout << "Baseline   min/med/max : "
              << m.baseline.min_ms  << " / "
              << m.baseline.median_ms << " / "
              << m.baseline.max_ms << "  ms\n";
    std::cout << "Baseline   GFLOPS (med): "
              << m.baseline_gflops_med
              << "  (min " << m.baseline_gflops_min
              << ", max " << m.baseline_gflops_max << ")\n";
    if (!m.calibrate) {
        std::cout << "Protected  min/med/max : "
                  << m.protected_.min_ms  << " / "
                  << m.protected_.median_ms << " / "
                  << m.protected_.max_ms << "  ms\n";
        std::cout << "Protected  GFLOPS (med): "
                  << m.protected_gflops_med
                  << "  (min " << m.protected_gflops_min
                  << ", max " << m.protected_gflops_max << ")\n";
        std::cout << "Overhead (median)      : " << m.overhead_pct << " %\n";
    }

    if (m.calibrate) {
        std::cout << "--- Calibration ---\n";
        std::cout << "Max |actual-expected|  : " << m.calib_max_diff       << "\n";
        std::cout << "Safety factor          : " << m.calib_safety_factor  << "\n";
        std::cout << "Suggested threshold    : " << m.calib_suggested_tau  << "\n";
        std::cout << "\n=> Re-run with:  --threshold " << m.calib_suggested_tau << "\n";
    } else {
        std::cout << "--- Confusion Matrix (global) ---\n";
        std::cout << "TP / TN / FP / FN     : "
                  << m.cm.TP << " / " << m.cm.TN << " / "
                  << m.cm.FP << " / " << m.cm.FN << "\n";
        std::cout << "Recall                : " << m.recall                  << "\n";
        std::cout << "Detection precision   : " << m.precision               << "\n";
        std::cout << "Correction precision  : " << m.correction_precision_pct << " %\n";
    }
    std::cout << "==================================================\n\n";
}

// CSV logging — one row per experiment (only rank 0 should call)
inline void log_metrics_csv(const ExperimentMetrics& m, const std::string& filename) {
    bool write_header = false;
    {
        std::ifstream test(filename);
        if (!test.good()) write_header = true;
    }
    std::ofstream ofs(filename, std::ios::app);
    if (!ofs.is_open()) {
        std::cerr << "WARNING: cannot open " << filename << " for CSV logging\n";
        return;
    }
    if (write_header) {
        ofs << "scheme,inject,swifi_zone,baseline_only,calibrate,"
               "M,K,N,Pr,Pc,frags_per_rank,num_fragments_total,repeats,"
               "threshold_used,"
               "baseline_min_ms,baseline_median_ms,baseline_mean_ms,baseline_max_ms,"
               "baseline_gflops_min,baseline_gflops_median,baseline_gflops_mean,baseline_gflops_max,"
               "protected_min_ms,protected_median_ms,protected_mean_ms,protected_max_ms,"
               "protected_gflops_min,protected_gflops_median,protected_gflops_mean,protected_gflops_max,"
               "overhead_pct,"
               "TP,TN,FP,FN,recall,precision,correction_precision_pct,"
               "calib_max_diff,calib_safety_factor,calib_suggested_tau\n";
    }
    ofs << std::fixed << std::setprecision(6);
    ofs << m.scheme << "," << m.inject << "," << m.swifi_zone << ","
        << (m.baseline_only ? 1 : 0) << ","
        << (m.calibrate     ? 1 : 0) << ","
        << m.M << "," << m.K << "," << m.N << ","
        << m.Pr << "," << m.Pc << "," << m.frags_per_rank << ","
        << m.num_fragments_total << "," << m.repeats << ","
        << m.threshold_used << ","
        << m.baseline.min_ms  << "," << m.baseline.median_ms  << "," << m.baseline.mean_ms  << "," << m.baseline.max_ms  << ","
        << m.baseline_gflops_min << "," << m.baseline_gflops_med << "," << m.baseline_gflops_mean << "," << m.baseline_gflops_max << ","
        << m.protected_.min_ms << "," << m.protected_.median_ms << "," << m.protected_.mean_ms << "," << m.protected_.max_ms << ","
        << m.protected_gflops_min << "," << m.protected_gflops_med << "," << m.protected_gflops_mean << "," << m.protected_gflops_max << ","
        << m.overhead_pct << ","
        << m.cm.TP << "," << m.cm.TN << "," << m.cm.FP << "," << m.cm.FN << ","
        << m.recall << "," << m.precision << "," << m.correction_precision_pct << ","
        << m.calib_max_diff << "," << m.calib_safety_factor << "," << m.calib_suggested_tau
        << "\n";
}
