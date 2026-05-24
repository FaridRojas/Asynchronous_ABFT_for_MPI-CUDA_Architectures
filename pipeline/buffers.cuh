#pragma once

#include "../core/common.cuh"
#include "../distribution/grid.cuh"

// Per-rank pipeline buffers — pre-allocated once for the full run.
struct PipelineBuffers {
    int F;
    int N_frag_max;
    int M_b;
    int N_b;
    int K;
    int R;                  // number of repeats

    // Device per-fragment row checksums (shared across iters)
    std::vector<double*> dExpectedRow;   // [F][N_frag_max]
    std::vector<double*> dActualRow;     // [F][N_frag_max]
    double*              dColSumA = nullptr;   // [K]

    // Device detection records, one per fragment, overwritten each iter
    int*    dErrCol   = nullptr;   // [F]  (-1 = clean)
    double* dRowDiff  = nullptr;   // [F]

    // Device localisation scratch (single instance — verify stream is
    double* dRowSumB     = nullptr;   // [K]
    double* dExpectedCol = nullptr;   // [M_b]
    double* dActualCol   = nullptr;   // [M_b]

    // Device aggregate counters, host-read ONCE after the timed loop
    int* dCM        = nullptr;   // [4] = {TP,TN,FP,FN}
    int* dNRestored = nullptr;   // [1]

    // Optional device copy of the golden C (only allocated in SWIFI runs)
    float* dGolden = nullptr;    // [M_b * N_b] or nullptr

    // Per-fragment events: gate the verify stream behind the compute stream
    std::vector<cudaEvent_t> compute_done;

    cudaStream_t   compute_stream;
    cudaStream_t   verify_stream;
    cublasHandle_t handle;

    // Per-fragment column metadata inside the rank's C_local
    std::vector<int> col_counts;
    std::vector<int> col_offsets;
};

inline void buffers_init(PipelineBuffers& b, int F, int M_b, int N_b, int K, int R) {
    b.F   = F;
    b.M_b = M_b;
    b.N_b = N_b;
    b.K   = K;
    b.R   = R;

    split_dim(N_b, F, b.col_counts, b.col_offsets);
    b.N_frag_max = *std::max_element(b.col_counts.begin(), b.col_counts.end());

    b.dExpectedRow.resize(F);
    b.dActualRow  .resize(F);
    b.compute_done.resize(F);

    for (int f = 0; f < F; ++f) {
        CUDA_CHECK(cudaMalloc(&b.dExpectedRow[f], sizeof(double) * b.N_frag_max));
        CUDA_CHECK(cudaMalloc(&b.dActualRow  [f], sizeof(double) * b.N_frag_max));
        CUDA_CHECK(cudaEventCreateWithFlags(&b.compute_done[f], cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaMalloc(&b.dColSumA, sizeof(double) * K));

    CUDA_CHECK(cudaMalloc(&b.dErrCol,      sizeof(int)    * F));
    CUDA_CHECK(cudaMalloc(&b.dRowDiff,     sizeof(double) * F));
    CUDA_CHECK(cudaMalloc(&b.dRowSumB,     sizeof(double) * K));
    CUDA_CHECK(cudaMalloc(&b.dExpectedCol, sizeof(double) * M_b));
    CUDA_CHECK(cudaMalloc(&b.dActualCol,   sizeof(double) * M_b));
    CUDA_CHECK(cudaMalloc(&b.dCM,          sizeof(int)    * 4));
    CUDA_CHECK(cudaMalloc(&b.dNRestored,   sizeof(int)));
    b.dGolden = nullptr;   // lazily allocated by pass_online_loop in SWIFI runs

    CUDA_CHECK(cudaStreamCreate(&b.compute_stream));
    CUDA_CHECK(cudaStreamCreate(&b.verify_stream));

    CUBLAS_CHECK(cublasCreate(&b.handle));
    CUBLAS_CHECK(cublasSetStream(b.handle, b.compute_stream));
}

inline void buffers_free(PipelineBuffers& b) {
    for (int f = 0; f < b.F; ++f) {
        cudaFree(b.dExpectedRow[f]);
        cudaFree(b.dActualRow[f]);
        cudaEventDestroy(b.compute_done[f]);
    }
    cudaFree(b.dColSumA);
    cudaFree(b.dErrCol);
    cudaFree(b.dRowDiff);
    cudaFree(b.dRowSumB);
    cudaFree(b.dExpectedCol);
    cudaFree(b.dActualCol);
    cudaFree(b.dCM);
    cudaFree(b.dNRestored);
    if (b.dGolden) cudaFree(b.dGolden);

    cudaStreamDestroy(b.compute_stream);
    cudaStreamDestroy(b.verify_stream);
    cublasDestroy(b.handle);
}
