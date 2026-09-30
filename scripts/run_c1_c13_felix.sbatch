#!/bin/bash
#SBATCH --job-name=abft_c1c13
#SBATCH --partition=gpu_titan
#SBATCH --nodelist=felix
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --output=scripts/abft_c1c13_%j.out
#SBATCH --error=scripts/abft_c1c13_%j.err

# Two measurements taken after the thesis campaigns, on the same node, with
# the same modules and the same flags as those campaigns:
#
#   C13  Which cuBLAS kernel runs at each calibrated size (nvprof), and
#        whether the current environment reproduces the published
#        calibration residuals (same seeds, same flags as
#        run_calibrate_felix.sbatch; outputs to c13/).
#
#   C1   What launching the localisation kernels on EVERY fragment costs
#        (--localize-always, which a deployment with online recovery
#        needs) against launching them only on injection, as the
#        campaigns did.  Per size the two fault-free variants run twice in
#        A-B-B-A order next to a standalone baseline and the additive-fault
#        condition; each run also measures its own interleaved baseline, so
#        the paired overheads are immune to clock drift between runs.
#        Output: c1_metrics.csv (scheme online = A, online_loc = B).

set -euo pipefail
module purge
module load cuda/11.8
module load mpi/openmpi-gcc11/4.1.6
MPI_PREFIX="$(dirname "$(dirname "$(which mpicc)")")"

echo "=== Compiling ==="
nvcc -O3 -std=c++17 -ccbin g++ -I"${MPI_PREFIX}/include" src/main.cu -o abft_gemm \
     -L"${MPI_PREFIX}/lib" -lmpi -lcublas
echo "Compilation OK"
echo

echo "=== Environment ==="
date -Is
uname -r
nvidia-smi --query-gpu=index,name,driver_version --format=csv
nvcc --version | tail -1
echo

# Same guard as run_abft_felix.sbatch: refuse to time anything while a
# process outside Slurm is using either GPU.
WAIT_TIMEOUT=${WAIT_TIMEOUT:-7200}
waited=0
while :; do
  busy="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null \
          | sed '/^\s*$/d' | sort -u | wc -l)"
  (( busy == 0 )) && break
  if (( waited >= WAIT_TIMEOUT )); then
    echo "GPUs still busy after ${WAIT_TIMEOUT}s; aborting"; nvidia-smi || true; exit 1
  fi
  echo "${busy} GPU(s) busy; waiting"; sleep 60; waited=$(( waited + 60 ))
done

F=8
run() { local label="$1"; shift; echo "=== ${label}"; srun --mpi=pmix ./abft_gemm "$@"; echo; }

# =========================================================================
# C13 — kernel names and calibration reproduction
# =========================================================================
mkdir -p c13
PROF_SIZES=(512 1024 1536 2048 2304 2560 2816 3072 3328 3584 3840 4096 5120 6144 8192 10240)
for S in "${PROF_SIZES[@]}"; do
  echo "=== C13 kernels ${S}^3"
  # Two calibration passes suffice: cuBLAS picks its kernel per shape.
  if ! srun --mpi=pmix nvprof --unified-memory-profiling off --print-gpu-summary \
         --log-file "c13/nvprof_${S}_rank%q{SLURM_PROCID}.txt" \
         ./abft_gemm ${S} ${S} ${S} --calibrate --frags-per-rank ${F} --repeats 2 \
         --csv c13/_scratch.csv --calib-diffs c13/_scratch_diffs.csv > /dev/null; then
    echo "    nvprof failed for ${S}"
  fi
done
rm -f c13/_scratch.csv c13/_scratch_diffs.csv

CALIB_SIZES=(512 1024 2048 4096 8192 10240)
rm -f c13/abft_calibration.csv
for S in "${CALIB_SIZES[@]}"; do
  run "C13 calibration ${S}^3" ${S} ${S} ${S} --calibrate --frags-per-rank ${F} \
      --repeats 20 --calibration-safety 3 \
      --csv c13/abft_calibration.csv --calib-diffs "c13/abft_calib_diffs_${S}.csv"
done

# =========================================================================
# C1 — cost of launching localisation on every fragment
# =========================================================================
COMMON=(--frags-per-rank ${F} --samples 20 --repeats 1 --warmups 2)
PROT=(--threshold 1.0 --encoding-mode overlap "${COMMON[@]}")   # tau irrelevant for timing
OUT=c1_metrics.csv
rm -f "${OUT}"
for S in $(seq 256 256 10240); do
  run "C1 baseline ${S}^3"      ${S} ${S} ${S} --baseline "${COMMON[@]}" --csv "${OUT}"
  run "C1 A1 detect ${S}^3"     ${S} ${S} ${S} --inject none                   "${PROT[@]}" --csv "${OUT}"
  run "C1 B1 always ${S}^3"     ${S} ${S} ${S} --inject none --localize-always "${PROT[@]}" --csv "${OUT}"
  run "C1 B2 always ${S}^3"     ${S} ${S} ${S} --inject none --localize-always "${PROT[@]}" --csv "${OUT}"
  run "C1 A2 detect ${S}^3"     ${S} ${S} ${S} --inject none                   "${PROT[@]}" --csv "${OUT}"
  run "C1 fault ${S}^3"         ${S} ${S} ${S} --inject add                    "${PROT[@]}" --csv "${OUT}"
done

# Quick look in the log; the analysis proper runs off-cluster.
python3 - "${OUT}" <<'PY'
import csv, sys
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[1])))
by = defaultdict(lambda: defaultdict(list))
for r in rows:
    key = "baseline" if r["baseline_only"] == "1" else \
          ("fault" if r["inject"] == "add" else r["scheme"])
    by[int(r["M"])][key].append(r)
def paired(r):
    return 100.0 * (float(r["protected_mean_ms"]) / float(r["baseline_mean_ms"]) - 1.0)
print("size  detect%  always%  fault%   (paired, mean of runs)")
for S in sorted(by):
    d = by[S]
    m = lambda k: sum(paired(r) for r in d[k]) / len(d[k]) if d[k] else float("nan")
    print(f"{S:5d}  {m('online'):7.2f}  {m('online_loc'):7.2f}  {m('fault'):7.2f}")
PY
echo "=== All runs complete ==="
