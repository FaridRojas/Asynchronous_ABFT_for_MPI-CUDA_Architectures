#!/bin/bash
#SBATCH --job-name=abft_tau
#SBATCH --partition=gpu_titan
#SBATCH --nodelist=felix
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --mem=16G
#SBATCH --output=scripts/abft_tau_%j.out
#SBATCH --error=scripts/abft_tau_%j.err

# ===================================================================
# Threshold-sensitivity sweep.
#
# The CM campaign measured detection at tau = 3 x the observed noise
# maximum, and an earlier zone sweep happened to use 10 x.  Those two
# cannot be compared: they differ in operand diversity as well as in
# tau.  This job re-measures the whole thing under ONE protocol, the
# campaign's, varying only the factor.
#
# Two quantities per (size, factor):
#   recall      — from --inject swifi, one bit flip per protected GEMM,
#                 five IEEE-754 zones.
#   false pos.  — from --inject none.  A clean run still evaluates the
#                 detector on every fragment, so any fragment it flags
#                 is a false alarm.  This is what tests the extreme-value
#                 model directly: the fit predicts a few percent per
#                 protected GEMM at factor 1 and nothing measurable at 3.
#
# Nothing here writes the tau table or the campaign CSVs, so the
# published results are untouched.
# ===================================================================

set -euo pipefail

module purge
module load cuda/11.8
module load mpi/openmpi-gcc11/4.1.6

MPICC_PATH="$(which mpicc)"
MPI_PREFIX="$(dirname "$(dirname "$MPICC_PATH")")"

echo "=== Compiling ==="
nvcc -O3 -std=c++17 -ccbin g++ \
  -I"${MPI_PREFIX}/include" \
  src/main.cu -o abft_gemm \
  -L"${MPI_PREFIX}/lib" -lmpi -lcublas

# ---------- Knobs ----------
SIZES=( ${SIZES:-1024 2048 4096 8192} )
FACTORS=( ${FACTORS:-1 2 3 10} )
ZONES=( ${ZONES:-any sign exponent sig_high sig_low} )
F=${F:-8}
SAMPLES=${SAMPLES:-100}
R=${R:-1}
WARMUPS=${WARMUPS:-2}
OUT=${OUT:-abft_tau_sweep.csv}

# Same protocol as the CM campaign: one timed GEMM per sample, fresh
# operands per sample.  Only --threshold changes across runs.
COMMON=(--frags-per-rank ${F} --samples ${SAMPLES} --repeats ${R}
        --warmups ${WARMUPS} --reseed-per-trial --encoding-mode overlap
        --csv "${OUT}")

rm -f "${OUT}"

# Observed noise maxima, from scripts/tau_table.generated.sh (tau = 3 x max).
# Reading the table and dividing keeps this job consistent with the
# calibration that produced the campaign, instead of re-measuring it.
declare -A MAXDIFF=()
if [[ -f scripts/tau_table.generated.sh ]]; then
  # shellcheck disable=SC1091
  source scripts/tau_table.generated.sh
  for S in "${SIZES[@]}"; do
    t="${TAU_TABLE[$S]:-}"
    [[ -n "$t" ]] && MAXDIFF[$S]="$(python3 -c "print('%.10g' % (${t}/3.0))")"
  done
fi

echo "############################################"
echo "###  TAU SWEEP  ${#SIZES[@]} sizes x ${#FACTORS[@]} factors x $(( ${#ZONES[@]} + 1 )) modes"
echo "############################################"
for S in "${SIZES[@]}"; do
  echo "  max|a-e|(${S}) = ${MAXDIFF[$S]:-<missing>}"
done

run() {
  local label="$1"; shift
  echo "============================================"
  echo "=== ${label} ==="
  echo "============================================"
  srun --mpi=pmix ./abft_gemm "$@"
  echo
}

for S in "${SIZES[@]}"; do
  mx="${MAXDIFF[$S]:-}"
  if [[ -z "${mx}" ]]; then
    echo "  SKIP ${S}: no stored maximum"; continue
  fi
  for c in "${FACTORS[@]}"; do
    tau="$(python3 -c "print('%.10g' % (${mx} * ${c}))")"
    echo "### ${S}^3  factor=${c}  tau=${tau}"
    # Clean run: every flagged fragment is a false positive.
    run "FP  ${S}^3 c=${c}" ${S} ${S} ${S} \
        --inject none --threshold "${tau}" "${COMMON[@]}"
    for z in "${ZONES[@]}"; do
      run "SWIFI ${S}^3 c=${c} zone=${z}" ${S} ${S} ${S} \
          --inject swifi --swifi-zone "${z}" --threshold "${tau}" "${COMMON[@]}"
    done
  done
done

echo "=== Done ==="
echo "CSV: $(pwd)/${OUT}"
