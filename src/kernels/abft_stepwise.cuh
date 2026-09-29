#pragma once

#include "../core/common.cuh"

// Kernels
// ===================================================================
__global__ inline void k_col_checksum_A(const float* __restrict__ A, int lda,
                                        double* __restrict__ colSumA,
                                        int M, int K) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < K) {
        double s = 0.0;
        for (int i = 0; i < M; ++i) s += static_cast<double>(A[i * lda + k]);
        colSumA[k] = s;
    }
}

// Chunked variants of the two input-encoding kernels.  The plain versions
// above expose only K (resp. N_frag) threads with a serial loop over the
// other dimension — enough when the encoding ran once outside the timed
// window, but far too little parallelism to hide behind a fast GEMM once the
// encoding is charged/overlapped inside it (a fraction of one SM on an A100).
// The chunked versions split the serial loop across ENC_CHUNKS grid slices
// writing double partials, and a tiny reduce kernel folds the partials.
constexpr int ENC_CHUNKS = 32;

__global__ inline void k_col_checksum_A_part(const float* __restrict__ A, int lda,
                                             double* __restrict__ part,
                                             int M, int K) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int p = blockIdx.y;
    if (k < K) {
        int chunk = (M + gridDim.y - 1) / gridDim.y;
        int i0 = p * chunk;
        int i1 = i0 + chunk; if (i1 > M) i1 = M;
        double s = 0.0;
        for (int i = i0; i < i1; ++i) s += static_cast<double>(A[(size_t)i * lda + k]);
        part[(size_t)p * K + k] = s;
    }
}

__global__ inline void k_expected_row_part(const double* __restrict__ colSumA,
                                           const float*  __restrict__ B_frag, int ldb,
                                           double*       __restrict__ part,
                                           int K, int N_frag) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int p = blockIdx.y;
    if (j < N_frag) {
        int chunk = (K + gridDim.y - 1) / gridDim.y;
        int k0 = p * chunk;
        int k1 = k0 + chunk; if (k1 > K) k1 = K;
        double s = 0.0;
        for (int k = k0; k < k1; ++k)
            s += colSumA[k] * static_cast<double>(B_frag[(size_t)k * ldb + j]);
        part[(size_t)p * N_frag + j] = s;
    }
}

__global__ inline void k_reduce_enc_part(const double* __restrict__ part,
                                         double* __restrict__ out,
                                         int n, int chunks) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        double s = 0.0;
        for (int p = 0; p < chunks; ++p) s += part[(size_t)p * n + j];
        out[j] = s;
    }
}

__global__ inline void k_row_checksum_B(const float* __restrict__ B_frag, int ldb,
                                        double* __restrict__ rowSumB,
                                        int K, int N_frag) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < K) {
        double s = 0.0;
        for (int j = 0; j < N_frag; ++j) s += static_cast<double>(B_frag[k * ldb + j]);
        rowSumB[k] = s;
    }
}

__global__ inline void k_expected_row(const double* __restrict__ colSumA,
                                      const float*  __restrict__ B_frag, int ldb,
                                      double*       __restrict__ expectedRow,
                                      int K, int N_frag) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < N_frag) {
        double s = 0.0;
        for (int k = 0; k < K; ++k) s += colSumA[k] * static_cast<double>(B_frag[k * ldb + j]);
        expectedRow[j] = s;
    }
}

__global__ inline void k_actual_row(const float* __restrict__ C_frag, int ldc,
                                    double*      __restrict__ actualRow,
                                    int M, int N_frag) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < N_frag) {
        double s = 0.0;
        for (int i = 0; i < M; ++i) s += static_cast<double>(C_frag[i * ldc + j]);
        actualRow[j] = s;
    }
}

__global__ inline void k_expected_col(const float*  __restrict__ A, int lda,
                                      const double* __restrict__ rowSumB,
                                      double*       __restrict__ expectedCol,
                                      int M, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) {
        double s = 0.0;
        for (int k = 0; k < K; ++k) s += static_cast<double>(A[i * lda + k]) * rowSumB[k];
        expectedCol[i] = s;
    }
}

__global__ inline void k_actual_col(const float* __restrict__ C_frag, int ldc,
                                    double*      __restrict__ actualCol,
                                    int M, int N_frag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) {
        double s = 0.0;
        for (int j = 0; j < N_frag; ++j) s += static_cast<double>(C_frag[i * ldc + j]);
        actualCol[i] = s;
    }
}

__global__ inline void k_correct_element(float* C_frag, int ldc,
                                         int row, int col, float value) {
    C_frag[row * ldc + col] = value;
}
// ===================================================================

// ===================================================================
// Device-resident detection / localisation / correction
// ===================================================================

__global__ inline void k_detect_row(const double* __restrict__ expectedRow,
                                    const double* __restrict__ actualRow,
                                    int N_frag, double threshold,
                                    int* __restrict__ out_errcol,
                                    double* __restrict__ out_rowdiff,
                                    int injected, int* __restrict__ cm) {
    __shared__ double s_d[256];
    __shared__ int    s_j[256];
    int tid = threadIdx.x;
    double best = 0.0;
    int    bestj = -1;
    for (int j = tid; j < N_frag; j += blockDim.x) {
        double d = fabs(actualRow[j] - expectedRow[j]);
        if (d > threshold && d > best) { best = d; bestj = j; }
    }
    s_d[tid] = best; s_j[tid] = bestj;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && s_d[tid + s] > s_d[tid]) {
            s_d[tid] = s_d[tid + s];
            s_j[tid] = s_j[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        int ec = s_j[0];
        *out_errcol  = ec;
        *out_rowdiff = (ec >= 0) ? (actualRow[ec] - expectedRow[ec]) : 0.0;
        int detected = (ec >= 0);
        int idx = injected ? (detected ? 0 : 3)   // TP : FN
                           : (detected ? 2 : 1);  // FP : TN
        atomicAdd(&cm[idx], 1);
    }
}

// Localisation triad — each is a no-op (early return, launch latency
// only) when the gate (this fragment's err_col) says "no detection".
__global__ inline void k_row_checksum_B_g(const float* __restrict__ B_frag, int ldb,
                                          double* __restrict__ rowSumB,
                                          int K, int N_frag,
                                          const int* __restrict__ gate) {
    if (*gate < 0) return;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < K) {
        double s = 0.0;
        for (int j = 0; j < N_frag; ++j) s += static_cast<double>(B_frag[k * ldb + j]);
        rowSumB[k] = s;
    }
}

__global__ inline void k_expected_col_g(const float* __restrict__ A, int lda,
                                        const double* __restrict__ rowSumB,
                                        double* __restrict__ expectedCol,
                                        int M, int K,
                                        const int* __restrict__ gate) {
    if (*gate < 0) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) {
        double s = 0.0;
        for (int k = 0; k < K; ++k) s += static_cast<double>(A[i * lda + k]) * rowSumB[k];
        expectedCol[i] = s;
    }
}

__global__ inline void k_actual_col_g(const float* __restrict__ C_frag, int ldc,
                                      double* __restrict__ actualCol,
                                      int M, int N_frag,
                                      const int* __restrict__ gate) {
    if (*gate < 0) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) {
        double s = 0.0;
        for (int j = 0; j < N_frag; ++j) s += static_cast<double>(C_frag[i * ldc + j]);
        actualCol[i] = s;
    }
}

// Single block.  Gated by err_col.  Finds the corrupt row, subtracts the
// row checksum delta in place, and folds a restore-success count.
__global__ inline void k_locate_correct(const double* __restrict__ expectedCol,
                                        const double* __restrict__ actualCol,
                                        int M, double threshold,
                                        const int*    __restrict__ errcol_ptr,
                                        const double* __restrict__ rowdiff_ptr,
                                        float* __restrict__ C_frag, int ldc,
                                        const float* __restrict__ golden, int ldg,
                                        int frag_col_offset, int injected,
                                        int* __restrict__ n_restored) {
    if (*errcol_ptr < 0) return;
    __shared__ double s_d[256];
    __shared__ int    s_i[256];
    int tid = threadIdx.x;
    double best = 0.0;
    int    besti = -1;
    for (int i = tid; i < M; i += blockDim.x) {
        double d = fabs(actualCol[i] - expectedCol[i]);
        if (d > threshold && d > best) { best = d; besti = i; }
    }
    s_d[tid] = best; s_i[tid] = besti;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && s_d[tid + s] > s_d[tid]) {
            s_d[tid] = s_d[tid + s];
            s_i[tid] = s_i[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        int er = s_i[0];
        if (er >= 0) {
            int    ec = *errcol_ptr;
            double rd = *rowdiff_ptr;
            float  c  = static_cast<float>(
                            static_cast<double>(C_frag[er * ldc + ec]) - rd);
            C_frag[er * ldc + ec] = c;
            if (injected && golden != nullptr) {
                float g = golden[(size_t)er * ldg + frag_col_offset + ec];
                if (fabs(static_cast<double>(c) - static_cast<double>(g))
                        <= threshold)
                    atomicAdd(n_restored, 1);
            }
        }
    }
}

inline void launch_detect_row(const double* dExpectedRow,
                              const double* dActualRow,
                              int N_frag, double threshold,
                              int* dErrCol, double* dRowDiff,
                              int injected, int* dCM, cudaStream_t stream) {
    k_detect_row<<<1, 256, 0, stream>>>(dExpectedRow, dActualRow, N_frag,
                                        threshold, dErrCol, dRowDiff,
                                        injected, dCM);
}

inline void launch_localize_correct(const float* dA, int lda,
                                    const float* dB_frag, int ldb,
                                    float* dC_frag, int ldc,
                                    double* dRowSumB,
                                    double* dExpectedCol, double* dActualCol,
                                    int M, int K, int N_frag,
                                    double threshold,
                                    const int* dErrCol, const double* dRowDiff,
                                    const float* dGolden, int ldg,
                                    int frag_col_offset, int injected,
                                    int* dNRestored, cudaStream_t stream) {
    int t = 256;
    k_row_checksum_B_g<<<(K + t - 1) / t, t, 0, stream>>>(
        dB_frag, ldb, dRowSumB, K, N_frag, dErrCol);
    k_expected_col_g<<<(M + t - 1) / t, t, 0, stream>>>(
        dA, lda, dRowSumB, dExpectedCol, M, K, dErrCol);
    k_actual_col_g<<<(M + t - 1) / t, t, 0, stream>>>(
        dC_frag, ldc, dActualCol, M, N_frag, dErrCol);
    k_locate_correct<<<1, 256, 0, stream>>>(
        dExpectedCol, dActualCol, M, threshold, dErrCol, dRowDiff,
        dC_frag, ldc, dGolden, ldg, frag_col_offset, injected, dNRestored);
}

// Per-stage launch helpers (enqueue, do not synchronise)
// ---------------------------------------------------------------------------

inline void launch_col_checksum_A(const float* dA, int lda,
                                  double* dColSumA,
                                  int M, int K, cudaStream_t stream,
                                  double* dEncPart = nullptr) {
    int t = 256, b = (K + t - 1) / t;
    if (dEncPart == nullptr) {   // legacy single-pass path
        k_col_checksum_A<<<b, t, 0, stream>>>(dA, lda, dColSumA, M, K);
        return;
    }
    dim3 grid(b, ENC_CHUNKS);
    k_col_checksum_A_part<<<grid, t, 0, stream>>>(dA, lda, dEncPart, M, K);
    k_reduce_enc_part<<<b, t, 0, stream>>>(dEncPart, dColSumA, K, ENC_CHUNKS);
}

inline void launch_expected_row(const double* dColSumA,
                                const float* dB_frag, int ldb,
                                double* dExpectedRow,
                                int K, int N_frag, cudaStream_t stream,
                                double* dEncPart = nullptr) {
    int t = 256, b = (N_frag + t - 1) / t;
    if (dEncPart == nullptr) {   // legacy single-pass path
        k_expected_row<<<b, t, 0, stream>>>(dColSumA, dB_frag, ldb, dExpectedRow, K, N_frag);
        return;
    }
    dim3 grid(b, ENC_CHUNKS);
    k_expected_row_part<<<grid, t, 0, stream>>>(dColSumA, dB_frag, ldb, dEncPart, K, N_frag);
    k_reduce_enc_part<<<b, t, 0, stream>>>(dEncPart, dExpectedRow, N_frag, ENC_CHUNKS);
}

inline void launch_actual_row(const float* dC_frag, int ldc,
                              double* dActualRow,
                              int M, int N_frag, cudaStream_t stream) {
    int t = 256, b = (N_frag + t - 1) / t;
    k_actual_row<<<b, t, 0, stream>>>(dC_frag, ldc, dActualRow, M, N_frag);
}

inline void launch_row_checksum_B(const float* dB_frag, int ldb,
                                  double* dRowSumB,
                                  int K, int N_frag, cudaStream_t stream) {
    int t = 256, b = (K + t - 1) / t;
    k_row_checksum_B<<<b, t, 0, stream>>>(dB_frag, ldb, dRowSumB, K, N_frag);
}

inline void launch_expected_col(const float* dA, int lda,
                                const double* dRowSumB,
                                double* dExpectedCol,
                                int M, int K, cudaStream_t stream) {
    int t = 256, b = (M + t - 1) / t;
    k_expected_col<<<b, t, 0, stream>>>(dA, lda, dRowSumB, dExpectedCol, M, K);
}

inline void launch_actual_col(const float* dC_frag, int ldc,
                              double* dActualCol,
                              int M, int N_frag, cudaStream_t stream) {
    int t = 256, b = (M + t - 1) / t;
    k_actual_col<<<b, t, 0, stream>>>(dC_frag, ldc, dActualCol, M, N_frag);
}

inline void launch_correct_element(float* dC_frag, int ldc,
                                   int row, int col, float value,
                                   cudaStream_t stream) {
    k_correct_element<<<1, 1, 0, stream>>>(dC_frag, ldc, row, col, value);
}
// ---------------------------------------------------------------------------

// Host-side anomaly searches
// ---------------------------------------------------------------------------
inline int find_row_anomaly(const double* hExpectedRow,
                            const double* hActualRow,
                            int N_frag, double threshold,
                            double* out_diff = nullptr) {
    int worst_col = -1;
    double worst_diff = 0.0;
    for (int j = 0; j < N_frag; ++j) {
        double d = std::abs(hActualRow[j] - hExpectedRow[j]);
        if (d > threshold && d > worst_diff) {
            worst_diff = d;
            worst_col  = j;
        }
    }
    if (out_diff) *out_diff = worst_diff;
    return worst_col;
}

inline int find_col_anomaly(const double* hExpectedCol,
                            const double* hActualCol,
                            int M, double threshold) {
    int worst_row = -1;
    double worst_diff = 0.0;
    for (int i = 0; i < M; ++i) {
        double d = std::abs(hActualCol[i] - hExpectedCol[i]);
        if (d > threshold && d > worst_diff) {
            worst_diff = d;
            worst_row  = i;
        }
    }
    return worst_row;
}

// Maximum |actual - expected| over a fragment (used by calibration).
inline double max_abs_row_diff(const double* hExpectedRow,
                               const double* hActualRow,
                               int N_frag) {
    double mx = 0.0;
    for (int j = 0; j < N_frag; ++j) {
        double d = std::abs(hActualRow[j] - hExpectedRow[j]);
        if (d > mx) mx = d;
    }
    return mx;
// ---------------------------------------------------------------------------

}
