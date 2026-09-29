#!/bin/bash
#SBATCH --job-name=abft_gemm
#SBATCH --partition=gpu_titan
#SBATCH --nodelist=felix
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --output=scripts/abft_felix_%j.out
#SBATCH --error=scripts/abft_felix_%j.err

set -euo pipefail

# =========================================================
# Environment
# =========================================================
module purge
module load cuda/11.8
module load mpi/openmpi-gcc11/4.1.6

MPICC_PATH="$(which mpicc)"
MPI_PREFIX="$(dirname "$(dirname "$MPICC_PATH")")"

echo "=== Compiling ABFT GEMM (cuBLAS, online pipelined) ==="
nvcc -O3 -std=c++17 -ccbin g++ \
  -I"${MPI_PREFIX}/include" \
  src/main.cu -o abft_gemm \
  -L"${MPI_PREFIX}/lib" -lmpi -lcublas
echo "Compilation OK"
echo

echo "=== GPU inventory ==="
nvidia-smi -L || true
echo

# =========================================================
# Guard: do not start until BOTH GPUs are idle.
#
# --gres=gpu:2 --exclusive already makes Slurm hold the job until two GPUs
# (and the whole node) are free, so this loop only matters when someone is
# driving a GPU OUTSIDE Slurm, which Slurm cannot see.  The campaign is a
# timing benchmark: a neighbour process would both slow it down and skew
# every number, so it is worth refusing to start.
#
#   WAIT_GPUS=0        disable the wait entirely
#   WAIT_TIMEOUT=7200  give up (non-zero exit) after this many seconds
#   WAIT_POLL=60       seconds between probes
# =========================================================
WAIT_GPUS=${WAIT_GPUS:-1}
WAIT_TIMEOUT=${WAIT_TIMEOUT:-7200}
WAIT_POLL=${WAIT_POLL:-60}

busy_gpu_count() {
  # Distinct GPUs that currently host at least one compute process.
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null \
    | sed '/^\s*$/d' | sort -u | wc -l
}

if [[ "${WAIT_GPUS}" != "0" ]]; then
  echo "=== Waiting for both GPUs to be idle ==="
  waited=0
  while :; do
    busy="$(busy_gpu_count)"
    if [[ "${busy}" -eq 0 ]]; then
      echo "    Both GPUs idle after ${waited}s — starting campaign."
      break
    fi
    if (( waited >= WAIT_TIMEOUT )); then
      echo "    TIMEOUT after ${WAIT_TIMEOUT}s with ${busy} GPU(s) still busy."
      echo "    Aborting so the benchmark is not polluted by a neighbour."
      nvidia-smi || true
      exit 1
    fi
    echo "    ${busy} GPU(s) busy; retrying in ${WAIT_POLL}s (waited ${waited}s)"
    sleep "${WAIT_POLL}"
    waited=$(( waited + WAIT_POLL ))
  done
  echo
fi

# =========================================================
# Knobs
# =========================================================
F=${F:-8}
R=${R:-1}                       # cfg.repeats — one timed GEMM per sample
WARMUPS=${WARMUPS:-2}            # discarded warm-up samples per phase
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
SAFETY=${SAFETY:-3}

# Encoding placement for every ABFT (online) run: the input encoding is
# issued inside the timed window on the verification stream and overlapped
# with the first-fragment multiplications, so its full cost is charged to
# each sample — the same protocol used by the comparison experiments.
ENC_FLAG=(--encoding-mode overlap)

# ---- Perf sweep (drives time_vs_size / gflops_vs_size / *_overhead) ----
# 40 sizes 256..10240 (256-aligned, like the comparison Fig.21).
# Per size we run only 3 fast phases (baseline, ABFT no-fault, ABFT additive
# fault) with SWEEP_SAMPLES back-to-back timed GEMMs on FIXED A,B (no reseed:
# dense-GEMM timing is operand-value-independent, and reseeding would insert
# idle gaps that cool the GPU clocks and bias the throughput downward).
# No SWIFI zones here — zones are CM territory and dominate wallclock.
SWEEP_N=${SWEEP_N:-40}
SWEEP_SAMPLES=${SWEEP_SAMPLES:-20}
sweep_grid() { local i; for ((i=0;i<SWEEP_N;i++)); do echo $(( 256*(i+1) )); done; }
mapfile -t SWEEP_SIZES < <( sweep_grid | awk '!s[$0]++' )

# Non-square perf sweep: M=N varies (SWEEP_SIZES), K is fixed.  Same
# 40 sizes, so the resulting gflops_vs_size_ns plot is directly
# comparable to the square one.  K=1024 mirrors the comparison job's
# big_ns regime so cross-experiment numbers stay consistent.
SWEEP_K_NS=${SWEEP_K_NS:-1024}

# ---- CM campaign (drives confusion_matrix / detection_metrics) ---------
# Independent of the perf sweep: a few sizes, every IEEE-754 zone, many
# samples on fresh (A,B) so the CM aggregates over many operand contexts.
CM_SIZES=( ${CM_SIZES:-512 1024 2048 4096 8192 10240} )
CM_SAMPLES=${CM_SAMPLES:-100}
ZONES=( ${ZONES:-any sign exponent sig_high sig_low} )

# Detection threshold mode — single knob, four values:
#   TAU=<number>     MANUAL: use exactly this tau for every CM size
#   TAU=calibrate    (default) per-size --calibrate pass picks tau
#   TAU=table        read tau from the stored TAU_TABLE below
#   TAU=formula      use the binary's conservative analytical formula
TAU=${TAU:-calibrate}   # per-size calibration (what produced the
                        # published results).  A fixed small tau
                        # would sit BELOW the noise floor at large
                        # N and flood the run with false positives.

# ---- Stored tau table (TAU=table) -------------------------------------
# A calibration pass is itself a full GEMM + checksum reduction per size,
# so re-deriving tau at the start of every campaign repeats exactly the
# work the stored-checksum scheme exists to avoid.  Calibrate ONCE with
# scripts/run_calibrate_felix.sbatch, paste its "TAU_TABLE" block here,
# and every later campaign reads the numbers instead of recomputing them.
#
# Keys are UPPER BOUNDS of a size zone: size S uses the entry of the
# SMALLEST listed size >= S.  Rounding up is the safe direction.  The
# noise floor grows monotonically with the reduction length, so a tau
# borrowed from a larger size sits ABOVE the local noise: it can only
# miss faults whose magnitude is below that noise — perturbations too
# small to change the result they contaminate — whereas a tau borrowed
# from a smaller size sits BELOW the local noise floor and turns every
# fragment into a false positive.  Sizes above the largest entry are not
# covered and fall back to a live calibration.
#
# Format:  [size]="tau"    # observed max |a-e| x SAFETY factor
# Fill from a calibration run made with the CURRENT binary: the earlier
# numbers came from a calibrate loop that reused one operand pair for all
# R passes, so their maxima are single-draw and can under-estimate tau.
declare -A TAU_TABLE=(   # tau = 3 x max observado, calibrado 2026-08-29
  [512]="0.000319"
  [1024]="0.000802"
  [2048]="0.002312"
  [4096]="0.021641"
  [8192]="0.059241"
  [10240]="0.088163"
)

# Calibration uses a separate, larger R because it needs many noise samples
# regardless of the per-sample R used by the rest of the campaign.
CALIB_R=${CALIB_R:-20}

# NOTE: no --frag-width here.  Adaptive-F was removed; the binary keeps the
# fixed F-fragment two-stream pipeline this campaign measures.

rm -f abft_metrics.csv abft_metrics_ns.csv

run() {
  local label="$1"; shift
  echo "============================================"
  echo "=== ${label} ==="
  echo "============================================"
  srun --mpi=pmix ./abft_gemm "$@"
  echo
}

# Convenience: a fresh calibration run drops scripts/tau_table.generated.sh
# next to this script.  Load it ONLY while the table above is still empty,
# so pasting the numbers in (the committed, citable version) always wins
# over a leftover generated file.
if (( ${#TAU_TABLE[@]} == 0 )) && [[ -f scripts/tau_table.generated.sh ]]; then
  # shellcheck disable=SC1091
  source scripts/tau_table.generated.sh
  echo "### TAU_TABLE loaded from scripts/tau_table.generated.sh"
fi

tau_from_table() {  # $1 = size  -> echoes tau ("" if no zone covers it)
  local S="$1" k best=""
  (( ${#TAU_TABLE[@]} == 0 )) && { echo ""; return; }
  if [[ -n "${TAU_TABLE[$S]:-}" ]]; then echo "${TAU_TABLE[$S]}"; return; fi
  for k in "${!TAU_TABLE[@]}"; do          # smallest POPULATED size >= S
    [[ -n "${TAU_TABLE[$k]}" ]] || continue   # blank entry == not calibrated
    (( k >= S )) || continue
    if [[ -z "${best}" ]] || (( k < best )); then best="${k}"; fi
  done
  [[ -n "${best}" ]] && echo "${TAU_TABLE[$best]}" || echo ""
}

calibrate_tau() {   # $1 = size  -> echoes tau ("" on failure)
  local S="$1" cf="_calib_${S}.csv"
  rm -f "${cf}"
  srun --mpi=pmix ./abft_gemm "${S}" "${S}" "${S}" --calibrate \
       --frags-per-rank ${F} --repeats ${CALIB_R} \
       --calibration-safety ${SAFETY} --csv "${cf}" >/dev/null 2>&1 || true
  python3 - "${cf}" <<'PY'
import csv,sys
tau=""
try:
    for row in csv.DictReader(open(sys.argv[1])): tau=row.get("calib_suggested_tau","")
except Exception: pass
print(tau or "")
PY
}

# =========================================================
# Phase 1 — 40-size perf sweep
#   For every size: baseline -> ABFT(no fault) -> ABFT(additive fault).
#   No SWIFI, no calibration (perf-only); tau=1.0 since accuracy is not
#   measured here.  Drives the four "vs size" plots.
# =========================================================
echo "############################################"
echo "###  PERF SWEEP  (${#SWEEP_SIZES[@]} sizes, ${SWEEP_SAMPLES} samples)"
echo "############################################"

SWEEP_COMMON=(--frags-per-rank ${F} --samples ${SWEEP_SAMPLES}
              --repeats ${R} --warmups ${WARMUPS})
SWEEP_TAU="1.0"   # perf-only, threshold value irrelevant

for S in "${SWEEP_SIZES[@]}"; do
  echo "###  sweep ${S}^3"
  run "SWEEP baseline ${S}^3" ${S} ${S} ${S} --baseline                     "${SWEEP_COMMON[@]}"
  run "SWEEP online  ${S}^3"  ${S} ${S} ${S} --inject none --threshold ${SWEEP_TAU} "${ENC_FLAG[@]}" "${SWEEP_COMMON[@]}"
  run "SWEEP online+add ${S}^3" ${S} ${S} ${S} --inject add  --threshold ${SWEEP_TAU} "${ENC_FLAG[@]}" "${SWEEP_COMMON[@]}"
done

# =========================================================
# Phase 1b — Non-square perf sweep: M=N varies, K=SWEEP_K_NS fixed.
#   Same SWEEP_SIZES, same 3 phases, separate CSV (abft_metrics_ns.csv)
#   so plot_metrics renders a parallel set of "vs size" plots — in
#   particular gflops_vs_size_ns.png — for the K-fixed regime.
# =========================================================
echo "############################################"
echo "###  PERF SWEEP NS  (${#SWEEP_SIZES[@]} sizes, K=${SWEEP_K_NS} fixed, ${SWEEP_SAMPLES} samples)"
echo "############################################"

SWEEP_COMMON_NS=("${SWEEP_COMMON[@]}" --csv abft_metrics_ns.csv)

for S in "${SWEEP_SIZES[@]}"; do
  echo "###  sweep_ns ${S}x${SWEEP_K_NS}x${S}"
  run "SWEEP_NS baseline ${S}x${SWEEP_K_NS}x${S}"   ${S} ${SWEEP_K_NS} ${S} --baseline                     "${SWEEP_COMMON_NS[@]}"
  run "SWEEP_NS online   ${S}x${SWEEP_K_NS}x${S}"   ${S} ${SWEEP_K_NS} ${S} --inject none --threshold ${SWEEP_TAU} "${ENC_FLAG[@]}" "${SWEEP_COMMON_NS[@]}"
  run "SWEEP_NS online+add ${S}x${SWEEP_K_NS}x${S}" ${S} ${SWEEP_K_NS} ${S} --inject add  --threshold ${SWEEP_TAU} "${ENC_FLAG[@]}" "${SWEEP_COMMON_NS[@]}"
done

# =========================================================
# Phase 2 — CM campaign (3 sizes × 5 zones × 100 samples)
#   Per size: (calibrate) -> online+SWIFI for every IEEE-754 zone.
#   Every SWIFI run carries M,K,N + swifi_zone; only these rows feed the
#   confusion_matrix / detection_metrics plots.
#
#   --reseed-per-trial IS used here (unlike the perf sweep): different
#   operand values change the checksum noise floor and a fault's
#   detectability, so fresh (A,B) per sample makes the CM aggregate over
#   many operand contexts.  Each SWIFI run internally does a clean baseline
#   pass per trial to capture the golden C, so no separate baseline phase
#   is needed — and dropping baseline/none/add here also avoids emitting
#   duplicate timing rows at sizes the perf sweep already covers.
# =========================================================
echo "############################################"
echo "###  CM CAMPAIGN  (${#CM_SIZES[@]} sizes, ${CM_SAMPLES} samples × ${#ZONES[@]} zones)"
echo "############################################"

CM_COMMON=(--frags-per-rank ${F} --samples ${CM_SAMPLES}
           --reseed-per-trial --repeats ${R} --warmups ${WARMUPS})

for S in "${CM_SIZES[@]}"; do
  case "${TAU}" in
    formula)
      TH=(); tau="<formula>" ;;
    table)
      tau="$(tau_from_table "${S}")"
      if [[ -n "${tau}" ]]; then
        TH=(--threshold "${tau}")
        echo "### tau for ${S}^3 read from TAU_TABLE (no calibration run)"
      else
        echo "  NOTE: no TAU_TABLE zone covers ${S}^3; calibrating it live"
        tau="$(calibrate_tau "${S}")"
        if [[ -z "${tau}" || "${tau}" == "0.000000" ]]; then
          echo "  WARNING: calibration gave no tau for ${S}^3; using formula"
          TH=(); tau="<formula>"
        else
          TH=(--threshold "${tau}")
        fi
      fi ;;
    calibrate|"")
      echo "### Calibrating tau for ${S}^3 ..."
      tau="$(calibrate_tau "${S}")"
      if [[ -z "${tau}" || "${tau}" == "0.000000" ]]; then
        echo "  WARNING: calibration gave no tau for ${S}^3; using formula"
        TH=(); tau="<formula>"
      else
        TH=(--threshold "${tau}")
      fi ;;
    *)                                  # MANUAL: TAU is a number
      tau="${TAU}"; TH=(--threshold "${TAU}") ;;
  esac
  echo "### CM ${S}^3  tau=${tau}"

  for z in "${ZONES[@]}"; do
    run "CM online/swifi[${z}] ${S}^3" ${S} ${S} ${S} \
        --inject swifi --swifi-zone "${z}" "${TH[@]}" "${ENC_FLAG[@]}" "${CM_COMMON[@]}"
  done
done

echo "=== All runs complete ==="
echo "CSV: $(pwd)/abft_metrics.csv"

# =========================================================
# Render plots in-cluster so deploy_and_run.sh can scp them home.
# =========================================================
echo
echo "=== Rendering plots ==="
mkdir -p scripts/plots/campaign scripts/plots/campaign_ns
python3 scripts/plot_metrics.py \
        --metrics abft_metrics.csv \
        --out     scripts/plots/campaign \
    || echo "(skipped: install pandas matplotlib numpy on the cluster)"
# Non-square sweep plots — parallel set with gflops_vs_size_ns etc.
if [[ -s abft_metrics_ns.csv ]]; then
    python3 scripts/plot_metrics.py \
            --metrics abft_metrics_ns.csv \
            --out     scripts/plots/campaign_ns \
        || echo "(ns plots skipped)"
fi
