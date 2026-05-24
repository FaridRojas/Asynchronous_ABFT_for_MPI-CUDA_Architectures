# Asynchronous ABFT for Fail-Continue Error Mitigation in GEMM (MPI + CUDA)

Source code accompanying the paper *Fail-Continue Error Mitigation in
GEMM Operations: An Asynchronous ABFT Approach for MPI-CUDA
Architectures*.

The framework wraps an unmodified vendor GEMM (cuBLAS) in an online
Algorithm-Based Fault Tolerance (ABFT) layer whose checksum
verification runs on a second CUDA stream, concurrent with the main
compute stream, so that the localisation work of iteration *k*
overlaps with the SGEMMs of iteration *k+1*. The contribution is the
asynchronous scheduling: the compute path is never blocked by the
verification path, and the underlying matrix multiplication is left
to the closed-source vendor library.

The binary exposes two execution modes used throughout this release:

| Mode                            | What runs                                                                                                                                                                                                       |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--baseline`                    | Unprotected cuBLAS GEMM. Reference for the overhead measurement.                                                                                                                                                |
| `--inject {none\|add}`          | Online ABFT with pre-computed `expectedRow` and per-fragment `actualRow + compare` on the verify stream, double-buffered `dC`, async localisation, post-loop correction drain. `none` runs detection only; `add` forces the correction path on every iteration
The head-to-head comparison against the fused-kernel reference of Wu
et al., 2023 is not a separate binary mode; it is orchestrated by
[scripts/example_comparison.sh](scripts/example_comparison.sh), which
calls the two modes above alongside the reference binary.

## Prerequisites

- **CUDA toolkit** (`nvcc`, cuBLAS). Tested with 11.8 and 12.x.
- **OpenMPI** (`mpicc`, `mpirun`). Tested with 4.1.6.
- **GCC 11** or any C++17-compatible host compiler.
- A CUDA-capable GPU. Tested on Maxwell (sm_52, GTX Titan X) and
  Ampere (sm_80, A100). For a different architecture, override the
  compute capability via `-arch=sm_XX` to `nvcc`.

For the head-to-head comparison (optional), additionally:

- `git`, `cmake`, and `make`
- The `cuda-samples` Common headers (auto-fetched by the comparison
  script if not already on disk)

## Quick start

### Throughput study (paper Sect. 4.1)

```bash
bash scripts/example_run.sh
```

Builds `abft_gemm` and runs a three-size square sweep (`1024^3`,
`2048^3`, `4096^3` by default) covering the unprotected baseline, the
protected execution without a fault, and the protected execution with
a guaranteed additive fault. Writes a single CSV
(`abft_metrics.csv`); no plots are produced.

Knobs are environment-overridable, for example:

```bash
NP=2 SIZES="2048 4096 8192" SAMPLES=20 bash scripts/example_run.sh
```

See the header of [scripts/example_run.sh](scripts/example_run.sh)
for the full list.

### Head-to-head comparison (paper Sect. 4.2)

```bash
bash scripts/example_comparison.sh
```

Reproduces the comparison against the fused-kernel ABFT reference of
Wu et al., 2023 (see [Acknowledgements](#acknowledgements)). The
script:

1. Clones the reference project into `Fault-Tolerant-SGEMM-on-NVIDIA-GPUs/`
   if it is not already present.
2. Provisions the `cuda-samples` Common headers it depends on.
3. Patches its driver in two places: (a) to accept non-square
   dimensions through `FT_M` / `FT_N` / `FT_K` environment variables,
   and (b) to discard a fixed warm-up window before the timed
   measurement, so both sides start at boosted GPU clocks.
4. Builds the reference twice (with and without an always-on
   additive fault) and our `abft_gemm` once.
5. Sweeps the four shape regimes (`small_sq`, `big_sq`, `small_ns`,
   `big_ns`) and writes a single long-format CSV (`compare_all.csv`)
   with six rows per shape: `ours_baseline`, `ours_online`,
   `ours_online_inj`, `theirs_cublas`, `theirs_fused`, `theirs_fused_inj`.

Defaults are intentionally modest so the script finishes in minutes
on a single GPU. The exact paper-scale sweep is one env override
away:

```bash
SMALL_N=20 BIG_N=40 BIG_STEP=256 OUR_SAMPLES=15 THEIRS_SAMPLES=3 \
    bash scripts/example_comparison.sh
```

See the header of [scripts/example_comparison.sh](scripts/example_comparison.sh)
for the full knob list (regimes, sample counts, compute capability,
reference URL, etc.).

## Manual build

```bash
nvcc -O3 -std=c++17 -ccbin g++ \
     -I"$(dirname $(dirname $(which mpicc)))/include" \
     main.cu -o abft_gemm \
     -L"$(dirname $(dirname $(which mpicc)))/lib" -lmpi -lcublas
```

## Repository structure

```
.
├── README.md
├── main.cu                            orchestrator
├── core/                              types, CLI, math helpers
├── distribution/grid.cuh              2D process grid + data decomposition
├── kernels/
│   ├── gemm_cublas.cuh                cuBLAS wrapper + warm-up
│   ├── abft_stepwise.cuh              ABFT kernels + per-stage launchers
│   └── swifi.cuh                      single-bit-flip injection
├── metrics/metrics.cuh                confusion matrix, MPI aggregation
├── pipeline/
│   ├── buffers.cuh                    PipelineBuffers, per-iter OnlineSlot
│   └── passes.cuh                     pass_baseline / pass_online_loop
├── scripts/
│   ├── example_run.sh                 portable throughput sweep
│   └── example_comparison.sh          portable comparison vs the fused-kernel reference
└── Fault-Tolerant-SGEMM-on-NVIDIA-GPUs/   reference project (or auto-cloned)
```

## Distribution and ABFT model

Each MPI rank owns a tile of the result matrix on a 2D process grid
`Pr × Pc`, holding the corresponding row-stripe of `A`, column-stripe
of `B`, and local `C_local`. `C_local` is split into `F` column
fragments, and each iteration of the pipelined loop issues one cuBLAS
GEMM and one ABFT verification per fragment.

The Huang–Abraham checksum is computed stepwise:

```
colSumA[k]     = Σ_i A_stripe[i,k]                    A only, computed once
expectedRow[j] = Σ_k colSumA[k] · B_frag[k,j]         A,B only, computed once
actualRow[j]   = Σ_i C_frag[i,j]                      per iter, verify stream
detect          : |actualRow[j] − expectedRow[j]| > τ
localise        : column-checksum sweep finds row i*
correct         : C_frag[i*,j*] -= (actualRow[j*] − expectedRow[j*])
```

### Cross-iteration pipelining

```
verify_stream  : [actR_k | cmp][localise_k | async]  [actR_(k+1) | cmp] ...
compute_stream :                                     [SGEMMs of iter k+1 on dC[(k+1)%2]]
```

- `dC` is double-buffered: iter *k* uses `dC[k%2]`, iter *k+2* reuses it.
- Iter *k+2*'s SGEMMs `cudaStreamWaitEvent` on iter *k*'s verify-stream
  tail, so the buffer is only overwritten after iter *k*'s localisation
  has completed.
- Pending corrections are drained after the main loop: each carries
  an event the host syncs on before launching a single correction
  kernel and validating against the recomputed golden value.

## Outputs

### `abft_metrics.csv` (throughput study)

One row per (size, configuration). Columns of interest:

| Column                  | Meaning                                                              |
| ----------------------- | -------------------------------------------------------------------- |
| `scheme`                | `baseline` or `online`                                                |
| `inject`                | `none`, `add` (always-on additive fault), or `swifi` (single bit-flip) |
| `M`, `K`, `N`           | Problem dimensions                                                   |
| `baseline_gflops_mean`  | Mean GFLOPS of the unprotected cuBLAS reference                       |
| `protected_gflops_mean` | Mean GFLOPS of the protected execution                                |
| `overhead_pct`          | `100 × (T_protected − T_baseline) / T_baseline`                       |

### `compare_all.csv` (comparison)

Long format. Six rows per shape:

| Column   | Values                                                                                    |
| -------- | ----------------------------------------------------------------------------------------- |
| `regime` | `small_sq`, `big_sq`, `small_ns`, `big_ns`                                                |
| `M,K,N`  | Problem dimensions                                                                        |
| `who`    | `ours_baseline`, `ours_online`, `ours_online_inj`, `theirs_cublas`, `theirs_fused`, `theirs_fused_inj` |
| `gflops` | Mean GFLOPS                                                                               |
| `time_ms`| Mean wall-clock time per GEMM, derived as `2·M·K·N / (gflops·1e9) · 1e3`                  |

Throughput is `GFLOPS = 2·M·K·N / t` throughout.

## CLI reference

```
abft_gemm <M> <K> <N>         rectangular form
abft_gemm <N>                 square shorthand (M = K = N)

  --baseline                  unprotected cuBLAS only
  --inject {none|add|swifi}   none = detection only;
                              add  = guaranteed additive fault (forces correction);
                              swifi = single random bit-flip (accuracy study)
  --swifi-zone {any|sign|exponent|sig_high|sig_low}
                              restrict the SWIFI bit-flip to an IEEE-754 region
  --threshold T               detection threshold τ (required with --inject)
  --frags-per-rank F          pipeline depth per rank (default 4)
  --grid Pr Pc                2D process grid (default: auto near-square)
  --repeats N                 iterations per timed sample (default 20)
  --samples N                 outer trial count (default 5)
  --warmups N                 discarded warm-up samples (default 2)
  --reseed-per-trial          regenerate A, B (and golden C) before each sample
  --seed-a S, --seed-b S      RNG seeds
  --csv PATH                  metrics CSV output path (default abft_metrics.csv)
  --help                      this list
```

`--calibrate` and the SWIFI sub-flags exist in the binary but cover
threshold tuning, which is out of scope for the throughput study
presented in the paper.

## Acknowledgements

The comparison in Sect. 4.2 of the paper is run against the
fused-kernel ABFT reference of Wu et al., *Anatomy of
High-Performance GEMM with Online Fault Tolerance on GPUs*, ICS
2023. The reference implementation is publicly available at:

> https://github.com/shixun404/Fault-Tolerant-SGEMM-on-NVIDIA-GPUs

`scripts/example_comparison.sh` clones it on demand, applies two
minimal patches (a non-square dimension override and a warm-up
window for an apples-to-apples timing protocol), and links its
results into `compare_all.csv`. All credit for the fused-kernel
reference belongs to its original authors.

## Citing

If you use this code, please cite the accompanying paper. A BibTeX
entry will be added here once the proceedings are published.

## License

See `LICENSE` (to be added at release time).
