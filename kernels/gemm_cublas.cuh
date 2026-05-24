#pragma once

#include "../core/common.cuh"

// cuBLAS-based GEMM (replaces hand-written kernels)

inline void gemm_cublas(cublasHandle_t handle,
                        const float* dA, int lda_row,
                        const float* dB, int ldb_row,
                        float*       dC, int ldc_row,
                        int M, int N, int K,
                        float alpha = 1.0f,
                        float beta  = 0.0f) {
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             dB, ldb_row,
                             dA, lda_row,
                             &beta,
                             dC, ldc_row));
}

// / Force cuBLAS module load + JIT + heuristic-cache to warm *before* timed
inline void gemm_warmup(cublasHandle_t handle, cudaStream_t stream) {
    constexpr int W           = 64;
    constexpr int WARMUP_REPS = 3;
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(float) * W * W));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(float) * W * W));
    CUDA_CHECK(cudaMalloc(&dC, sizeof(float) * W * W));
    CUDA_CHECK(cudaMemsetAsync(dA, 0, sizeof(float) * W * W, stream));
    CUDA_CHECK(cudaMemsetAsync(dB, 0, sizeof(float) * W * W, stream));
    CUDA_CHECK(cudaMemsetAsync(dC, 0, sizeof(float) * W * W, stream));
    for (int i = 0; i < WARMUP_REPS; ++i) {
        gemm_cublas(handle, dA, W, dB, W, dC, W, W, W, W);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
}
