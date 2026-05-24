#pragma once

#include "../core/common.cuh"

// ===================================================================
// cuBLAS-based GEMM (replaces hand-written kernels)
// ===================================================================
// Why cuBLAS:
//   cuBLAS implements highly tuned single-GPU GEMM that internally uses
//   SUMMA-like 2D tiling on the GPU. Since the focus of this project is
//   the ABFT scheme, delegating the multiplication to cuBLAS removes
//   GEMM implementation as a confounding variable.
//
// Layout:
//   Our matrices are row-major. cuBLAS is column-major, so we compute
//       (column-major)  C^T(N x M) = B^T(N x K) * A^T(K x M)
//   which is the same memory as
//       (row-major)     C(M x N)   = A(M x K)   * B(K x N).
//   "Swap A/B, swap m/n" — no transposes, no copies.
// ===================================================================

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

/// Force cuBLAS module load + JIT + heuristic-cache to warm *before* timed
/// measurements. Submits several SGEMMs with throwaway inputs and synchronizes.
/// Three iterations: first triggers JIT, the next two stabilise the kernel
/// heuristic cache so the first timed measurement isn't an outlier.
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
