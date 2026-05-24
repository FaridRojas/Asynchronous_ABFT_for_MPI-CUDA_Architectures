#!/usr/bin/env bash
# example_run.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Build
echo "=== Building abft_gemm ==="

if ! command -v nvcc  >/dev/null 2>&1; then echo "ERROR: nvcc not in PATH";  exit 1; fi
if ! command -v mpicc >/dev/null 2>&1; then echo "ERROR: mpicc not in PATH"; exit 1; fi

MPICC_PATH="$(command -v mpicc)"
MPI_PREFIX="$(dirname "$(dirname "${MPICC_PATH}")")"

nvcc -O3 -std=c++17 -ccbin g++ \
  -I"${MPI_PREFIX}/include" \
  src/main.cu -o abft_gemm \
  -L"${MPI_PREFIX}/lib" -lmpi -lcublas

echo "Build OK: $(pwd)/abft_gemm"
echo

# Knobs
NP=${NP:-1}
SIZES=( ${SIZES:-1024 2048 4096} )
F=${F:-8}
SAMPLES=${SAMPLES:-10}
WARMUPS=${WARMUPS:-2}
R=${R:-1}
OUT_CSV=${OUT_CSV:-abft_metrics.csv}

# Threshold is irrelevant for the perf sweep (we are not measuring
TAU=1.0

rm -f "${OUT_CSV}"

COMMON=( --frags-per-rank "${F}"
         --samples         "${SAMPLES}"
         --warmups         "${WARMUPS}"
         --repeats         "${R}"
         --csv             "${OUT_CSV}" )

run() {
  echo "--- mpirun -np ${NP} ./abft_gemm $* ---"
  mpirun -np "${NP}" ./abft_gemm "$@"
  echo
}

# Sweep
echo "=== Throughput sweep (${#SIZES[@]} sizes, ${SAMPLES} samples/phase) ==="
echo "Sizes: ${SIZES[*]}"
echo

for S in "${SIZES[@]}"; do
  echo "############ S = ${S} ############"
  run "${S}" "${S}" "${S}" --baseline                            "${COMMON[@]}"
  run "${S}" "${S}" "${S}" --inject none --threshold "${TAU}"    "${COMMON[@]}"
  run "${S}" "${S}" "${S}" --inject add  --threshold "${TAU}"    "${COMMON[@]}"
done

echo
echo "=== Done ==="
echo "CSV: $(pwd)/${OUT_CSV}"
echo
echo "Rows of interest (mean GFLOPS):"
echo "  scheme=baseline               -> baseline_gflops_mean"
echo "  scheme=online,  inject=none   -> protected_gflops_mean   (no-fault)"
echo "  scheme=online,  inject=add    -> protected_gflops_mean   (with fault)"
echo
echo "Overhead vs baseline (%) is reported directly in column 'overhead_pct'."
