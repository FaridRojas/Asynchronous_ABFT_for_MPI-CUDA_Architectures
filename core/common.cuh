#pragma once

#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <random>
#include <string>
#include <iomanip>
#include <fstream>
#include <cfloat>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <chrono>

// ---------------------------------------------------------------------------
// Error-checking macros
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call) do {                                       \
    cudaError_t err = (call);                                       \
    if (err != cudaSuccess) {                                       \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)      \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        MPI_Abort(MPI_COMM_WORLD, -1);                              \
    }                                                               \
} while(0)

#define MPI_CHECK(call) do {                                        \
    int err = (call);                                               \
    if (err != MPI_SUCCESS) {                                       \
        char errstr[MPI_MAX_ERROR_STRING];                          \
        int sz = 0;                                                 \
        MPI_Error_string(err, errstr, &sz);                         \
        std::cerr << "MPI error: " << std::string(errstr, sz)       \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        MPI_Abort(MPI_COMM_WORLD, -1);                              \
    }                                                               \
} while(0)

#define CUBLAS_CHECK(call) do {                                     \
    cublasStatus_t st = (call);                                     \
    if (st != CUBLAS_STATUS_SUCCESS) {                              \
        std::cerr << "cuBLAS error code " << (int)st                \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        MPI_Abort(MPI_COMM_WORLD, -1);                              \
    }                                                               \
} while(0)

using clk = std::chrono::high_resolution_clock;

// ---------------------------------------------------------------------------
// Math utilities
// ---------------------------------------------------------------------------

/// Fill a vector with pseudo-random values in [-1, 1]
inline void fill_random(std::vector<float>& M, uint64_t seed) {
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : M) x = dist(gen);
}

/// Infinity norm of a row-major matrix (rows x cols) with leading dim ld.
inline double matrix_inf_norm(const float* M, int rows, int cols, int ld) {
    double max_row_sum = 0.0;
    for (int i = 0; i < rows; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j < cols; ++j) {
            row_sum += std::abs(static_cast<double>(M[(size_t)i * ld + j]));
        }
        max_row_sum = std::max(max_row_sum, row_sum);
    }
    return max_row_sum;
}

/// Stepwise-checksum theoretical detection threshold (worst-case rounding bound):
///   tau = gamma_k * ||A||_inf * ||B||_inf,  gamma_k = k * eps / (1 - k * eps)
/// In practice this can be much larger than the natural noise floor;
/// run `--calibrate` to obtain an empirically tuned threshold.
inline double compute_threshold_formula(int k, double norm_A_inf, double norm_B_inf) {
    const double eps = static_cast<double>(FLT_EPSILON);
    double denom = 1.0 - static_cast<double>(k) * eps;
    if (denom <= 0.0) denom = 1.0;
    double gamma_k = (static_cast<double>(k) * eps) / denom;
    return gamma_k * norm_A_inf * norm_B_inf;
}

// ---------------------------------------------------------------------------
// Statistics over timing samples
// ---------------------------------------------------------------------------
struct TimingStats {
    double min_ms    = 0.0;
    double median_ms = 0.0;
    double mean_ms   = 0.0;   // arithmetic mean (apples-to-apples with theirs)
    double max_ms    = 0.0;
    int    n_samples = 0;
};

inline TimingStats stats_of(std::vector<double> samples) {
    TimingStats s{};
    s.n_samples = static_cast<int>(samples.size());
    if (samples.empty()) return s;
    double sum = 0.0;
    for (double v : samples) sum += v;
    s.mean_ms = sum / samples.size();
    std::sort(samples.begin(), samples.end());
    s.min_ms    = samples.front();
    s.max_ms    = samples.back();
    s.median_ms = samples[samples.size() / 2];
    return s;
}
