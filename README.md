# Asynchronous ABFT for Fail-Continue Error Mitigation in GEMM (MPI + CUDA)

Source code, campaign scripts and raw data of the undergraduate thesis
*Diseño e Implementación de un Esquema ABFT Asíncrono para la Mitigación de
Errores tipo fail-continue en Multiplicación de Matrices bajo el Modelo
MPI+CUDA* (Universidad Industrial de Santander, 2026). The same framework
underlies the paper *Fail-Continue Error Mitigation in GEMM Operations: An
Asynchronous ABFT Approach for MPI-CUDA Architectures*.

The framework wraps an unmodified vendor GEMM (cuBLAS) in an online
Algorithm-Based Fault Tolerance (ABFT) layer. Each process splits its local
output block into fragments; the checksum verification of a fragment runs on
a second CUDA stream, gated by a per-fragment event, while the next fragment
is still being multiplied on the compute stream. Detection, localisation and
correction run entirely on the device. The detection threshold is derived
from the measured distribution of the verification residual instead of a
fixed constant.

## Repository layout

```
.
├── src/
│   ├── main.cu                  driver: CLI, operand generation, phases, CSV output
│   ├── core/                    run configuration, CLI parsing, common helpers
│   ├── distribution/grid.cuh    2D process grid and operand decomposition
│   ├── kernels/                 cuBLAS wrapper, ABFT checksum kernels, SWIFI injector
│   ├── pipeline/                two-stream pipeline: buffers and timed passes
│   └── metrics/metrics.cuh      confusion matrix and MPI aggregation
├── bench/multi_gemm.cu         sequence of independent protected GEMMs (trace)
├── scripts/                     Slurm launchers (*.sbatch), analysis and plotting (*.py)
└── docs/                        raw data and figures, one folder per campaign
```

## Requirements

- CUDA toolkit (`nvcc`, cuBLAS). Tested with 11.8.
- An MPI implementation (`mpicc`). Tested with OpenMPI 4.1.6.
- A C++17 host compiler. Tested with GCC 11.
- Python 3 with `numpy`, `pandas` and `matplotlib`, for analysis and plots.
- Slurm, for the launchers in `scripts/`. They target the SC3-UIS cluster:
  node `felix` (2 × GeForce GTX Titan X, Maxwell) and `pacca` (A100, Ampere).
  Adjust the `#SBATCH` headers and `module load` lines for another system.

## Build

```bash
nvcc -O3 -std=c++17 -ccbin g++ \
     -I"$(dirname $(dirname $(which mpicc)))/include" \
     src/main.cu -o abft_gemm \
     -L"$(dirname $(dirname $(which mpicc)))/lib" -lmpi -lcublas
```

Every launcher compiles the binary itself before running, so a manual build is
only needed for interactive use. `abft_gemm --help` lists all options.

## From results to data and scripts

| Result | Data (`docs/`) | Scripts (`scripts/`) |
| --- | --- | --- |
| Overhead vs size, square | `abft_metrics.csv`, `campaign/` | `run_abft_felix.sbatch`, `plot_metrics.py` |
| Overhead vs size, K = 1024 | `abft_metrics_ns.csv`, `campaign_ns/` | `run_abft_felix.sbatch`, `plot_metrics.py` |
| Cost of fragmenting the unprotected GEMM | `fragsweep/abft_baseline_frag.csv` | `run_baseline_frag_felix.sbatch`, `emit_baseline_frag_tables.py` |
| Comparison with the fused-kernel reference | `comparison/` (Titan X), `comparison_pacca/` (A100) | `run_comparison_felix.sbatch`, `run_comparison_pacca.sbatch`, `plot_comparison.py` |
| Execution traces | `profile_pacca/` (A100), `multigemm/` (Titan X) | `run_profile_pacca.sbatch`, `run_multitrace_felix.sbatch`, `plot_multigemm_traza.py` |
| Noise characterisation and threshold | `calibration/` | `run_calibrate_felix.sbatch`, `analyze_calibration.py` |
| Detection and correction by IEEE-754 region | `abft_metrics.csv` (`swifi` rows), `campaign/` | `run_abft_felix.sbatch`, `plot_metrics.py` |
| Sensitivity to the threshold factor | not included | `run_tau_sweep_felix.sbatch` |
| Methodology figures | none | `plot_metodologia_figs.py` |

`abft_metrics.csv` holds one row per measured configuration: the 40-size
square sweep in three conditions (unprotected, protected without a fault,
protected with an additive fault in every operation) and the single-bit-flip
campaign over six sizes and five regions of the IEEE-754 word. Overheads are
computed against the unprotected phase of the same campaign.

`export_eps.py` re-renders the data plots as EPS for LaTeX.

## Notes on reproducibility

- The detection thresholds are stored in `scripts/tau_table.generated.sh` and
  reused by the campaigns instead of being recalibrated on every run.
  `analyze_calibration.py` writes that table from the calibration data.
- The factor that multiplies the observed noise maximum is post-processing:
  the calibration data store the measured maximum, so changing the criterion
  does not require running the calibration again.
- The threshold analysis and every figure can be regenerated from the data in
  `docs/` without GPU access.
- The per-trial output of the threshold-factor sweep is not included in this
  repository; `run_tau_sweep_felix.sbatch` regenerates it.

## Reference implementation

The comparison runs against the fused-kernel ABFT of Wu et al., *Anatomy of
High-Performance GEMM with Online Fault Tolerance on GPUs*, ICS 2023, publicly
available at <https://github.com/shixun404/Fault-Tolerant-SGEMM-on-NVIDIA-GPUs>.
It is not redistributed here: `scripts/example_comparison.sh` clones it on
demand and applies two minimal patches (a non-square dimension override and a
warm-up window) so both sides follow the same timing protocol. All credit for
the reference implementation belongs to its authors.
