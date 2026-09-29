#!/bin/bash
#SBATCH --job-name=abft_calib
#SBATCH --partition=gpu_titan
#SBATCH --nodelist=felix
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --output=scripts/abft_calib_%j.out
#SBATCH --error=scripts/abft_calib_%j.err

# ===================================================================
# Two phases:
#
#  PHASE A — Clean noise calibration over SEVERAL matrix sizes.
#    For each size: clean GEMM + ABFT, no detection; dump every
#    |actualRow-expectedRow|.  Larger K accumulates more rounding, so
#    the noise floor (hence tau) grows with size — that is exactly why
#    we sweep sizes instead of trusting a single point.
#    The sizes swept here are the CM-campaign sizes, so the resulting
#    thresholds can be stored once as the TAU_TABLE of
#    scripts/run_abft_felix.sbatch and reused by every later campaign
#    instead of paying a calibration pass per run.
#    Outputs: abft_calibration.csv (one row/size), per-size diff dumps,
#             one histogram+CDF PNG per size,
#             scripts/tau_table.generated.sh (paste-ready TAU_TABLE).
#
#  PHASE B — Per-ZONE SWIFI recall sweep (sizes x bit-zones).
#    For each size x zone in {sign,exponent,sig_high,sig_low,any}:
#    inject a single bit-flip restricted to that IEEE-754 region and
#    measure recall.  This quantifies WHY false negatives happen:
#    sig_low flips produce |Δ| below tau and are physically
#    undetectable, while sign/exponent/sig_high are caught.
#    Output: abft_zone_recall.csv  + a recall-vs-size per-zone plot.
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
# CALIB_SIZES mirrors CM_SIZES in scripts/run_abft_felix.sbatch so the
# generated table covers every size the campaign will ask for.
CALIB_SIZES=(512 1024 2048 4096 8192 10240)   # square, clean noise calibration
ZONE_SIZES=(1024 2048 4096)         # square, per-zone SWIFI sweep
ZONES=(any sign exponent sig_high sig_low)
F=8
R=20
# Detection-threshold safety factor: tau = SAFETY x observed max |a-e|.
# NOT a free parameter.  scripts/analyze_calibration.py fits a Gumbel to the
# R per-pass maxima and reports the smallest integer factor whose false-
# positive probability per protected GEMM stays under 1e-12 at every
# calibrated size; on the 2026-08-29 data that is 3 (worst case 1.4e-15,
# against 7e-2 at factor 1 — the noise floor really is that sharp).  The old
# value of 10 was arbitrary and cost recall: every fault smaller than tau is
# invisible, and at 4096 and above a factor of 10 hides roughly 20 more
# percentage points of sig_high bit flips than a factor of 3 does.
# Re-derive with:  python3 scripts/analyze_calibration.py --recall
SAFETY=3
# Fallback tau for the zone-recall sweep.  PHASE B normally uses the tau
# PHASE A just measured FOR THAT SIZE (see TAU_OF below); this value is
# only used for a size PHASE A did not cover.  A single tau shared across
# sizes is misleading here: it is ~9x the noise floor at 4096 but ~200x
# at 1024, so it would suppress detections at the small end and make the
# recall look size-dependent when the threshold is what changed.
ZONE_TAU="0.05"

# Which phases to run.  PHASE=AB (default) runs both; PHASE=B runs only the
# per-zone sweep and takes tau from the STORED table instead of measuring it
# again.  That matters: re-running PHASE A draws new operands, so its maxima
# (and therefore the table) would shift, and the campaign already executed
# against the current table.  PHASE=B re-measures recall at the stored
# threshold without disturbing anything the campaign depends on.
PHASE=${PHASE:-AB}

declare -A TAU_OF=()      # size -> suggested tau, filled by PHASE A

mkdir -p scripts/plots/calibration

rm -f abft_calibration.csv abft_zone_recall.csv

# =========================================================
# PHASE A — clean noise calibration, multiple sizes
# =========================================================
if [[ "${PHASE}" == *A* ]]; then
for SZ in "${CALIB_SIZES[@]}"; do
  echo
  echo "============================================"
  echo "=== PHASE A : calibration  N=${SZ}  F=${F} R=${R} ==="
  echo "============================================"
  srun --mpi=pmix ./abft_gemm ${SZ} ${SZ} ${SZ} \
       --calibrate \
       --frags-per-rank ${F} \
       --repeats ${R} \
       --calibration-safety ${SAFETY} \
       --csv         abft_calibration.csv \
       --calib-diffs abft_calib_diffs_${SZ}.csv

  TAU_OF[$SZ]="$(python3 - abft_calibration.csv "${SZ}" <<'PY'
import csv, sys
tau = ""
try:
    for row in csv.DictReader(open(sys.argv[1])):
        if row.get("M") == sys.argv[2]:
            tau = row.get("calib_suggested_tau", "")
except Exception:
    pass
print(tau or "")
PY
)"
  echo "=== tau(${SZ}) = ${TAU_OF[$SZ]:-<none>} ==="

  python3 scripts/plot_metrics.py \
          --metrics abft_calibration.csv \
          --diffs   abft_calib_diffs_${SZ}.csv \
          --out     scripts/plots/calibration \
      && mv scripts/plots/calibration/calibration_distribution.png \
            scripts/plots/calibration/calib_dist_${SZ}.png 2>/dev/null \
      || echo "(plot skipped for ${SZ}: install pandas matplotlib numpy)"
done
else
  echo "=== PHASE A skipped (PHASE=${PHASE}); tau comes from the stored table ==="
  if [[ -f scripts/tau_table.generated.sh ]]; then
    # shellcheck disable=SC1091
    source scripts/tau_table.generated.sh
    for SZ in "${ZONE_SIZES[@]}"; do
      TAU_OF[$SZ]="${TAU_TABLE[$SZ]:-}"
      echo "    tau(${SZ}) = ${TAU_OF[$SZ]:-<none>}  (tabla almacenada)"
    done
  else
    echo "    WARNING: scripts/tau_table.generated.sh not found; using ZONE_TAU"
  fi
fi

# =========================================================
# PHASE B — per-zone SWIFI recall sweep
# =========================================================
for SZ in "${ZONE_SIZES[@]}"; do
  ZT="${TAU_OF[$SZ]:-${ZONE_TAU}}"
  for Z in "${ZONES[@]}"; do
    echo
    echo "--- PHASE B : N=${SZ}  zone=${Z}  tau=${ZT} ---"
    srun --mpi=pmix ./abft_gemm ${SZ} ${SZ} ${SZ} \
         --inject swifi --swifi-zone ${Z} \
         --threshold ${ZT} \
         --frags-per-rank ${F} --repeats ${R} \
         --csv abft_zone_recall.csv
  done
done

echo
echo "============================================"
echo "=== Per-zone recall plot ==="
echo "============================================"
python3 scripts/plot_zone_recall.py \
        --csv abft_zone_recall.csv \
        --out scripts/plots/calibration/zone_recall.png \
    || echo "(zone-recall plot skipped: install pandas matplotlib)"

# =========================================================
# Paste-ready tau table — the factor is DERIVED, not assumed
# =========================================================
TAU_TABLE_FILE=scripts/tau_table.generated.sh
TAU_TEX_FILE=scripts/tau_table.tex
if [[ "${PHASE}" != *A* ]]; then
  echo "=== tau table left untouched (PHASE A did not run) ==="
else
python3 scripts/analyze_calibration.py \
        --diffs-dir  . \
        --metrics    abft_calibration.csv \
        --repeats    ${R} \
        --emit-table "${TAU_TABLE_FILE}" \
        --emit-tex   "${TAU_TEX_FILE}" \
        --recall \
    || echo "(tau table skipped: install numpy on the cluster)"

echo
echo "============================================"
echo "=== TAU_TABLE — paste into run_abft_felix.sbatch ==="
echo "============================================"
cat "${TAU_TABLE_FILE}" 2>/dev/null || echo "(not generated)"
fi

echo
echo "=== Done ==="
echo "  abft_calibration.csv         (per-size noise max + suggested tau)"
echo "  abft_calib_diffs_<N>.csv     (raw |actual-expected| per size)"
echo "  abft_zone_recall.csv         (recall per size x bit-zone)"
echo "  ${TAU_TABLE_FILE}            (paste-ready TAU_TABLE block)"
echo "  ${TAU_TEX_FILE}              (same numbers as LaTeX table rows)"
echo "  scripts/plots/calibration/calib_dist_<N>.png"
echo "  scripts/plots/calibration/zone_recall.png"
echo
echo "Paste the TAU_TABLE block above into scripts/run_abft_felix.sbatch and"
echo "run the campaign with TAU=table — no calibration pass per campaign."
