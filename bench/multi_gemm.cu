// ===================================================================
// multi_gemm — traza de una secuencia de multiplicaciones protegidas
//              independientes, emitidas sin espera intermedia.
//
// Las P operaciones se emiten seguidas sobre un mismo par de flujos,
// uno de cómputo y otro de verificación.  Las multiplicaciones se
// serializan entre sí porque comparten flujo, pero la comprobación de
// la salida de una operación puede transcurrir mientras se multiplica
// la siguiente.  Con un solo fragmento, lo que la oculta no puede ser
// un fragmento posterior de la misma matriz, de modo que la traza
// muestra el solapamiento entre operaciones distintas.
//
// Se ejecuta sobre un solo dispositivo y un solo proceso.  MPI se
// inicializa porque las cabeceras del proyecto lo requieren, y se usa
// la implementación real del esquema y no una copia.
//
//   multi_gemm M K N [--ops P] [--frags F] [--rounds R]
//                    [--warmup-rounds W] [--csv FILE] [--tag NAME]
// ===================================================================

#include "../src/core/common.cuh"
#include "../src/core/types.cuh"
#include "../src/distribution/grid.cuh"
#include "../src/kernels/gemm_cublas.cuh"
#include "../src/pipeline/buffers.cuh"

#include <numeric>
#include <random>
#include <sstream>

// Una operación protegida completa: sus operandos, su salida y el
// espacio de trabajo de la verificación.
struct Op {
    float*           dA = nullptr;
    float*           dB = nullptr;
    float*           dC = nullptr;
    PipelineBuffers  buf{};
};

// Emite una operación sobre el par de flujos indicado.  No sincroniza:
// la secuencia completa se espera al final.
static void enqueue_op(Op& op, int M, int K, int N, int F,
                       cudaStream_t cs, cudaStream_t vs, double tau) {
    CUBLAS_CHECK(cublasSetStream(op.buf.handle, cs));

    // Depende solo de A y B, que la multiplicación lee y nunca escribe,
    // así que no necesita esperar a nada.
    launch_col_checksum_A(op.dA, K, op.buf.dColSumA, M, K, vs,
                          op.buf.dEncPart);
    for (int f = 0; f < F; ++f) {
        launch_expected_row(op.buf.dColSumA, op.dB + op.buf.col_offsets[f],
                            N, op.buf.dExpectedRow[f], K,
                            op.buf.col_counts[f], vs, op.buf.dEncPart);
    }

    for (int f = 0; f < F; ++f) {
        const int nf  = op.buf.col_counts[f];
        const int off = op.buf.col_offsets[f];
        gemm_cublas(op.buf.handle, op.dA, K, op.dB + off, N,
                    op.dC + off, N, M, nf, K);
        CUDA_CHECK(cudaEventRecord(op.buf.compute_done[f], cs));
    }

    for (int f = 0; f < F; ++f) {
        const int nf  = op.buf.col_counts[f];
        const int off = op.buf.col_offsets[f];
        CUDA_CHECK(cudaStreamWaitEvent(vs, op.buf.compute_done[f], 0));
        launch_actual_row(op.dC + off, N, op.buf.dActualRow[f], M, nf, vs);
        launch_detect_row(op.buf.dExpectedRow[f], op.buf.dActualRow[f], nf,
                          tau, op.buf.dErrCol + f, op.buf.dRowDiff + f,
                          0, op.buf.dCM, vs);
    }
}

int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));
    int world = 1;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world));
    if (world != 1) {
        std::cerr << "multi_gemm se ejecuta sobre un solo dispositivo; "
                     "ejecútese con un solo proceso\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (argc < 4) {
        std::cerr << "uso: multi_gemm M K N [--ops P] [--frags F] "
                     "[--rounds R] [--warmup-rounds W] [--csv F] [--tag T]\n";
        MPI_Finalize();
        return 2;
    }

    const int M = std::atoi(argv[1]);
    const int K = std::atoi(argv[2]);
    const int N = std::atoi(argv[3]);
    int         P = 4, F = 1, rounds = 10, warm = 2;
    double      tau = 1.0;
    uint64_t    seed = 12345;
    std::string csv = "multi_gemm.csv", tag;

    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--ops"           && i + 1 < argc) P      = std::atoi(argv[++i]);
        else if (a == "--frags"         && i + 1 < argc) F      = std::atoi(argv[++i]);
        else if (a == "--rounds"        && i + 1 < argc) rounds = std::atoi(argv[++i]);
        else if (a == "--warmup-rounds" && i + 1 < argc) warm   = std::atoi(argv[++i]);
        else if (a == "--csv"           && i + 1 < argc) csv    = argv[++i];
        else if (a == "--tag"           && i + 1 < argc) tag    = argv[++i];
        else std::cerr << "[aviso] se ignora '" << a << "'\n";
    }
    if (tag.empty())
        tag = std::to_string(M) + "x" + std::to_string(K) + "x" + std::to_string(N);

    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // Cada operación lleva operandos propios: reutilizarlos permitiría
    // reaprovechar la codificación de entrada entre operaciones, que es
    // precisamente lo que una secuencia real no puede hacer.
    std::vector<Op> ops(P);
    std::vector<float> h((size_t)M * K > (size_t)K * N ? (size_t)M * K : (size_t)K * N);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    for (int k = 0; k < P; ++k) {
        buffers_init(ops[k].buf, F, M, N, K, 1);
        CUDA_CHECK(cudaMalloc(&ops[k].dA, sizeof(float) * (size_t)M * K));
        CUDA_CHECK(cudaMalloc(&ops[k].dB, sizeof(float) * (size_t)K * N));
        CUDA_CHECK(cudaMalloc(&ops[k].dC, sizeof(float) * (size_t)M * N));
        for (size_t i = 0; i < (size_t)M * K; ++i) h[i] = u(rng);
        CUDA_CHECK(cudaMemcpy(ops[k].dA, h.data(), sizeof(float) * (size_t)M * K,
                              cudaMemcpyHostToDevice));
        for (size_t i = 0; i < (size_t)K * N; ++i) h[i] = u(rng);
        CUDA_CHECK(cudaMemcpy(ops[k].dB, h.data(), sizeof(float) * (size_t)K * N,
                              cudaMemcpyHostToDevice));
    }

    // Un único par de flujos para toda la secuencia.
    cudaStream_t cs, vs;
    CUDA_CHECK(cudaStreamCreate(&cs));
    CUDA_CHECK(cudaStreamCreate(&vs));

    auto correr = [&]() {
        auto t0 = clk::now();
        for (int k = 0; k < P; ++k)
            enqueue_op(ops[k], M, K, N, F, cs, vs, tau);
        CUDA_CHECK(cudaStreamSynchronize(cs));
        CUDA_CHECK(cudaStreamSynchronize(vs));
        return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    };

    const bool fresh = !std::ifstream(csv).good();
    std::ofstream out(csv, std::ios::app);
    if (fresh)
        out << "tag,M,K,N,ops,frags,ronda,ms\n";
    out << std::fixed << std::setprecision(6);

    std::cout << "=== multi_gemm " << tag << " ===\n"
              << prop.name << ", " << prop.multiProcessorCount << " SM   "
              << P << " operaciones independientes, F=" << F
              << ", " << rounds << " rondas (+" << warm << " descartadas)\n";

    for (int r = 0; r < warm + rounds; ++r) {
        double ms = correr();
        if (r >= warm) {
            out << tag << ',' << M << ',' << K << ',' << N << ',' << P << ','
                << F << ',' << (r - warm) << ',' << ms << '\n';
            out.flush();
        }
    }
    std::cout << "escrito " << csv << "\n";

    for (int k = 0; k < P; ++k) {
        buffers_free(ops[k].buf);
        CUDA_CHECK(cudaFree(ops[k].dA));
        CUDA_CHECK(cudaFree(ops[k].dB));
        CUDA_CHECK(cudaFree(ops[k].dC));
    }
    CUDA_CHECK(cudaStreamDestroy(cs));
    CUDA_CHECK(cudaStreamDestroy(vs));
    MPI_CHECK(MPI_Finalize());
    return 0;
}
