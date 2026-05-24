#!/usr/bin/env bash
# =====================================================================
# example_comparison.sh
#
# Portable reproducibility recipe for the head-to-head comparison
# against the fused-kernel ABFT reference of Wu et al., 2023:
#
#     Fault-Tolerant-SGEMM-on-NVIDIA-GPUs
#     https://github.com/shixun404/Fault-Tolerant-SGEMM-on-NVIDIA-GPUs.git
#
# For every shape (M, K, N) in the requested regimes the script emits
# six rows into compare_all.csv:
#
#     ours_baseline       our cuBLAS, unprotected
#     ours_online         our online ABFT, no fault
#     ours_online_inj     our online ABFT with an always-on additive fault
#     theirs_cublas       their cuBLAS reference
#     theirs_fused        their fused-kernel ABFT, no fault
#     theirs_fused_inj    their fused-kernel ABFT with a fault
#
# Output: compare_all.csv (long format). No plots are produced.
#
# Requirements (in addition to the abft_gemm prerequisites):
#   - git, cmake (to build the reference project)
#   - the cuda-samples Common headers (auto-fetched if missing)
#
# Usage:
#   bash scripts/example_comparison.sh
#
# Overridable knobs (environment variables):
#   REGIMES          space-separated subset of  small_sq big_sq small_ns big_ns
#                    default: "small_sq big_sq small_ns big_ns"
#   SMALL_N          shapes per small regime           (default 4;   paper: 20)
#   BIG_N            shapes per big   regime           (default 6;   paper: 40)
#   BIG_STEP         stride for the big regime sweep   (default 1024; paper: 256)
#   SMALL_K_NS       fixed K for small non-square      (default 256)
#   BIG_K_NS         fixed K for big   non-square      (default 1024)
#   OUR_SAMPLES      back-to-back timed GEMMs of ours  (default 10;  paper: 15)
#   THEIRS_SAMPLES   re-invocations of theirs/shape    (default 2;   paper: 3)
#   WARMUPS          discarded warm-up samples         (default 5)
#   F                fragments per rank (ours)         (default 8)
#   CUDA_ARCH        compute capability for the ref    (default 52)
#   THEIRS_DIR       reference project directory       (default Fault-Tolerant-SGEMM-on-NVIDIA-GPUs)
#   THEIRS_URL       reference project clone URL       (default upstream above)
#   OUT_CSV          output CSV path                   (default compare_all.csv)
#
# Quick smoke run (one regime, three shapes):
#   REGIMES=small_sq SMALL_N=3 bash scripts/example_comparison.sh
# Reproduce paper-scale sweep (slow):
#   SMALL_N=20 BIG_N=40 BIG_STEP=256 OUR_SAMPLES=15 THEIRS_SAMPLES=3 \
#       bash scripts/example_comparison.sh
# =====================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

declare -gA _THEIRS_DUMPED=()

# --------------------------------------------------------------------
# Tool checks
# --------------------------------------------------------------------
for tool in nvcc mpicc mpirun git cmake make python3; do
  command -v "${tool}" >/dev/null 2>&1 || { echo "ERROR: ${tool} not in PATH"; exit 1; }
done

MPICC_PATH="$(command -v mpicc)"
MPI_PREFIX="$(dirname "$(dirname "${MPICC_PATH}")")"

# --------------------------------------------------------------------
# Knobs
# --------------------------------------------------------------------
REGIMES="${REGIMES:-small_sq big_sq small_ns big_ns}"
SMALL_N=${SMALL_N:-4}
BIG_N=${BIG_N:-6}
BIG_STEP=${BIG_STEP:-1024}
SMALL_K_NS=${SMALL_K_NS:-256}
BIG_K_NS=${BIG_K_NS:-1024}
OUR_SAMPLES=${OUR_SAMPLES:-10}
THEIRS_SAMPLES=${THEIRS_SAMPLES:-2}
WARMUPS=${WARMUPS:-5}
F=${F:-8}
CUDA_ARCH=${CUDA_ARCH:-52}
THEIRS_DIR=${THEIRS_DIR:-Fault-Tolerant-SGEMM-on-NVIDIA-GPUs}
THEIRS_URL=${THEIRS_URL:-https://github.com/shixun404/Fault-Tolerant-SGEMM-on-NVIDIA-GPUs.git}
OUT_CSV=${OUT_CSV:-compare_all.csv}

OUR_THRESHOLD="1.0"   # perf-only comparison; detection accuracy irrelevant
OUR_R=1               # one timed GEMM per sample (back-to-back)

# Shape grids.
# small_grid: SMALL_N points uniformly spaced in (0, 1024], rounded to 16.
# big_grid:   BIG_N points stepped by BIG_STEP starting at BIG_STEP.
small_grid() {
  local i s
  for ((i=0; i<SMALL_N; i++)); do
    s=$(python3 -c "print(int(round(($i+1)*1024.0/${SMALL_N}/16))*16)")
    echo "${s}"
  done
}
big_grid() {
  local i
  for ((i=0; i<BIG_N; i++)); do
    echo $(( BIG_STEP*(i+1) ))
  done
}

mapfile -t SMALL_SQ < <(small_grid | awk '!s[$0]++ {print $1, $1, $1}')
mapfile -t BIG_SQ   < <(big_grid   | awk '!s[$0]++ {print $1, $1, $1}')
mapfile -t SMALL_NS < <(small_grid | awk -v K=${SMALL_K_NS} '!s[$0]++ {print $1, K, $1}')
mapfile -t BIG_NS   < <(big_grid   | awk -v K=${BIG_K_NS}   '!s[$0]++ {print $1, K, $1}')

# --------------------------------------------------------------------
# 1. Provision the reference project
# --------------------------------------------------------------------
if [[ ! -d "${THEIRS_DIR}" ]]; then
  echo "=== Cloning ${THEIRS_URL} into ${THEIRS_DIR}/ ==="
  git clone --depth 1 "${THEIRS_URL}" "${THEIRS_DIR}"
fi

THEIRS_COMMON="${THEIRS_DIR}/cuda-samples/Common"
if [[ ! -f "${THEIRS_COMMON}/helper_functions.h" ]]; then
  echo "=== Fetching cuda-samples/Common/ ==="
  FOUND_INC=""
  for p in "${CUDA_HOME:-}/samples/common/inc" \
           "/usr/local/cuda/samples/common/inc"; do
    [[ -n "$p" && -f "$p/helper_functions.h" ]] && { FOUND_INC="$p"; break; }
  done
  if [[ -n "${FOUND_INC}" ]]; then
    mkdir -p "${THEIRS_COMMON}"
    cp -r "${FOUND_INC}/." "${THEIRS_COMMON}/"
  else
    rm -rf "${THEIRS_DIR}/cuda-samples"
    git clone --depth 1 https://github.com/NVIDIA/cuda-samples.git \
        "${THEIRS_DIR}/cuda-samples"
  fi
  [[ -f "${THEIRS_COMMON}/helper_functions.h" ]] \
      || { echo "ERROR: could not provision cuda-samples"; exit 1; }
fi

# Patch the reference CMakeLists to compile for CUDA_ARCH.
sed -i -E "s/set\\(CMAKE_CUDA_ARCHITECTURES[[:space:]]+[0-9]+\\)/set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})/" \
    "${THEIRS_DIR}/CMakeLists.txt"

# Patch their driver to accept non-square dims through FT_M/FT_N/FT_K.
SGEMM_SRC="${THEIRS_DIR}/kernel/ft_sgemm/sgemm.cu"
if ! grep -q "FT_M override" "${SGEMM_SRC}"; then
  echo "=== Patching ${SGEMM_SRC} for non-square (FT_M/FT_N/FT_K) ==="
  sed -i 's@N = K = M = max_size;@N = K = M = max_size; /*FT_M override*/ { char*_em=getenv("FT_M"); if(_em)M=atoi(_em); char*_en=getenv("FT_N"); if(_en)N=atoi(_en); char*_ek=getenv("FT_K"); if(_ek)K=atoi(_ek); }@' \
      "${SGEMM_SRC}"
  grep -q "FT_M override" "${SGEMM_SRC}" \
      || { echo "ERROR: driver non-square patch failed"; exit 1; }
fi

# Inject a warm-up window in their timing harness so both sides start at
# boosted clocks.  gflops still divides by num_tests (the timed-iter
# count), so the formula is unchanged.
if ! grep -q "FT_WARMUP" "${SGEMM_SRC}"; then
  echo "=== Patching ${SGEMM_SRC} for warm-up (FT_WARMUP) ==="
  sed -i 's@#define multi 20@#define multi 20\n#define FT_WARMUP 5  /* discarded warm-up iters before timed num_tests */@' \
      "${SGEMM_SRC}"
  sed -i 's@for(int ii = 0; ii < num_tests; ++ii){@for(int ii = 0; ii < num_tests + FT_WARMUP; ++ii){ if(ii==FT_WARMUP) cudaEventRecord(beg);@g' \
      "${SGEMM_SRC}"
  grep -q "FT_WARMUP" "${SGEMM_SRC}" \
      || { echo "ERROR: warm-up patch failed"; exit 1; }
fi

# Their fault injection is hard-coded as a compile-time constant in every
# ft_sgemm_*.cuh, so we build twice: once with error_inject=0.0 (clean)
# and once with error_inject=10000.0 (always-on additive fault), the
# analogue of OURS --inject add.
FT_HDRS=( "${THEIRS_DIR}"/kernel/ft_sgemm/include_code_gen/ft_sgemm_*.cuh )
[[ -e "${FT_HDRS[0]}" ]] \
    || { echo "ERROR: no ft_sgemm_*.cuh headers in ${THEIRS_DIR}"; exit 1; }

build_theirs() {  # $1 = clean|inj
  local tag="$1" val="10000.0" h
  [[ "${tag}" == clean ]] && val="0.0"
  for h in "${FT_HDRS[@]}"; do
    grep -q "error_inject" "$h" || continue
    sed -i -E "s/(float[[:space:]]+error_inject[[:space:]]*=[[:space:]]*)[0-9.]+f?;/\\1${val};/" "$h"
  done
  echo "=== Building reference ft_sgemm [${tag}] ==="
  rm -rf "${THEIRS_DIR}/build"
  mkdir -p "${THEIRS_DIR}/build"
  ( cd "${THEIRS_DIR}/build" \
      && cmake .. -DCMAKE_BUILD_TYPE=Release >/dev/null \
      && make -j4 >/dev/null )
  local b
  b="$(find "${THEIRS_DIR}/build" -name ft_sgemm -type f -executable | head -1)"
  [[ -n "${b}" ]] || { echo "ERROR: ft_sgemm [${tag}] not built"; exit 1; }
  cp "${b}" "${THEIRS_DIR}/ft_sgemm_${tag}"
}
build_theirs clean
build_theirs inj
THEIRS_CLEAN="${THEIRS_DIR}/ft_sgemm_clean"
THEIRS_INJ="${THEIRS_DIR}/ft_sgemm_inj"

# --------------------------------------------------------------------
# 2. Build our project
# --------------------------------------------------------------------
echo "=== Building our abft_gemm ==="
nvcc -O3 -std=c++17 -ccbin g++ \
  -I"${MPI_PREFIX}/include" \
  main.cu -o abft_gemm \
  -L"${MPI_PREFIX}/lib" -lmpi -lcublas

# --------------------------------------------------------------------
# 3. Helpers
# --------------------------------------------------------------------
echo "regime,M,K,N,who,gflops,time_ms" > "${OUT_CSV}"

their_gflops() {  # $1=logfile  $2=row-name (cublas|abft_kernel_huge|...)
  awk -F'|' -v name="$2" '
    $0 ~ "^"name" " { v=$2; gsub(/ /,"",v); print v; exit }' "$1"
}

gflops_to_ms() {  # $1=gflops $2=M $3=K $4=N
  python3 -c "
g=float('$1'); M=int('$2'); K=int('$3'); N=int('$4')
print('' if g<=0 else 2.0*M*N*K/(g*1e9)*1e3)" 2>/dev/null
}

mean_lines() { awk '$0!="" {s+=$1; n++} END {if(n>0) printf "%.6f",s/n}'; }

last_protected() {  # $1 = abft csv -> "base_g base_t prot_g prot_t"
  python3 - "$1" <<'PY'
import csv, sys
row=None
for row in csv.DictReader(open(sys.argv[1])): pass
print(row["baseline_gflops_mean"], row["baseline_mean_ms"],
      row["protected_gflops_mean"], row["protected_mean_ms"])
PY
}

# --------------------------------------------------------------------
# 4. Per-shape comparison
# --------------------------------------------------------------------
run_shape() {  # $1=regime  $2=M $3=K $4=N
  local regime="$1" M="$2" K="$3" N="$4"
  local smax=$(( M>N ? (M>K?M:K) : (N>K?N:K) ))
  echo
  echo "### ${regime}  M=${M} K=${K} N=${N}  samples=${OUR_SAMPLES}  warmups=${WARMUPS}"

  # ----- ours, no fault (also yields the cuBLAS baseline row) -----
  rm -f abft_metrics_cmp.csv _ours_launch.log
  if ! mpirun -np 1 ./abft_gemm ${M} ${K} ${N} \
       --inject none --threshold ${OUR_THRESHOLD} --grid 1 1 \
       --frags-per-rank ${F} \
       --samples ${OUR_SAMPLES} --repeats ${OUR_R} \
       --warmups ${WARMUPS} --csv abft_metrics_cmp.csv \
       > /dev/null 2> _ours_launch.log; then
    echo "  WARNING: ours(none) launch failed ${M}x${K}x${N}; skipping shape"
    sed 's/^/    | /' _ours_launch.log 2>/dev/null | head -10
    return 0
  fi
  local ob_g ob_t op_g op_t
  if ! read ob_g ob_t op_g op_t < <(last_protected abft_metrics_cmp.csv 2>/dev/null); then
    echo "  WARNING: ours(none) ${M}x${K}x${N}: bad CSV"
    return 0
  fi
  echo "${regime},${M},${K},${N},ours_baseline,${ob_g},${ob_t}" >> "${OUT_CSV}"
  echo "${regime},${M},${K},${N},ours_online,${op_g},${op_t}"   >> "${OUT_CSV}"

  # ----- ours, with fault -----
  rm -f abft_metrics_inj.csv
  if mpirun -np 1 ./abft_gemm ${M} ${K} ${N} \
       --inject add --threshold ${OUR_THRESHOLD} --grid 1 1 \
       --frags-per-rank ${F} \
       --samples ${OUR_SAMPLES} --repeats ${OUR_R} \
       --warmups ${WARMUPS} --csv abft_metrics_inj.csv \
       > /dev/null 2>> _ours_launch.log; then
    local _b _b2 oi_g oi_t
    if read _b _b2 oi_g oi_t < <(last_protected abft_metrics_inj.csv 2>/dev/null); then
      echo "${regime},${M},${K},${N},ours_online_inj,${oi_g},${oi_t}" >> "${OUT_CSV}"
    fi
  fi

  # ----- theirs: shape-appropriate kernel dispatch -----
  # (dispatch tracks the reference's kernel_number mapping, sgemm.cu L240)
  local kidx kname
  local out_max=$(( M>N ? M : N ))
  if   (( M >= 2*N ));            then kidx=14; kname="abft_kernel_tall"
  elif (( N >= 2*M ));            then kidx=15; kname="abft_kernel_wide"
  elif (( out_max <= 128 ));      then kidx=11; kname="abft_kernel_small"
  elif (( out_max <= 256 ));      then kidx=12; kname="abft_kernel_medium"
  elif (( out_max <= 512 ));      then kidx=13; kname="abft_kernel_large"
  else                                 kidx=16; kname="abft_kernel_huge"
  fi
  local _tc_acc="" _tfc_acc="" _tfi_acc="" s v
  for ((s=0; s<THEIRS_SAMPLES; s++)); do
    FT_M=${M} FT_N=${N} FT_K=${K} \
        "${THEIRS_CLEAN}" ${smax} ${smax} 1 0 0             > _tc.log  2>&1 || true
    FT_M=${M} FT_N=${N} FT_K=${K} \
        "${THEIRS_CLEAN}" ${smax} ${smax} 1 ${kidx} ${kidx} > _tfc.log 2>&1 || true
    FT_M=${M} FT_N=${N} FT_K=${K} \
        "${THEIRS_INJ}"   ${smax} ${smax} 1 ${kidx} ${kidx} > _tfi.log 2>&1 || true
    v=$(their_gflops  _tc.log  cublas);     [[ -n "${v}" ]] && _tc_acc+="${v}"$'\n'
    v=$(their_gflops _tfc.log "${kname}"); [[ -n "${v}" ]] && _tfc_acc+="${v}"$'\n'
    v=$(their_gflops _tfi.log "${kname}"); [[ -n "${v}" ]] && _tfi_acc+="${v}"$'\n'
  done
  local tc_g tfc_g tfi_g tc_t tfc_t tfi_t
  tc_g=$(printf  '%s' "${_tc_acc}"  | mean_lines)
  tfc_g=$(printf '%s' "${_tfc_acc}" | mean_lines)
  tfi_g=$(printf '%s' "${_tfi_acc}" | mean_lines)
  tc_t=""; tfc_t=""; tfi_t=""
  [[ -n "${tc_g}"  ]] && tc_t=$(gflops_to_ms  "${tc_g}"  ${M} ${K} ${N})
  [[ -n "${tfc_g}" ]] && tfc_t=$(gflops_to_ms "${tfc_g}" ${M} ${K} ${N})
  [[ -n "${tfi_g}" ]] && tfi_t=$(gflops_to_ms "${tfi_g}" ${M} ${K} ${N})
  [[ -n "${tc_g}"  ]] && echo "${regime},${M},${K},${N},theirs_cublas,${tc_g},${tc_t}"      >> "${OUT_CSV}"
  [[ -n "${tfc_g}" ]] && echo "${regime},${M},${K},${N},theirs_fused,${tfc_g},${tfc_t}"     >> "${OUT_CSV}"
  [[ -n "${tfi_g}" ]] && echo "${regime},${M},${K},${N},theirs_fused_inj,${tfi_g},${tfi_t}" >> "${OUT_CSV}"

  if [[ -z "${tfc_g}" || -z "${tfi_g}" ]]; then
    echo "  NOTE: theirs_fused empty for ${M}x${K}x${N} (kname=${kname}, kidx=${kidx})"
    if [[ -z "${_THEIRS_DUMPED[${kname}]:-}" ]]; then
      _THEIRS_DUMPED[${kname}]=1
      echo "  --- _tfc.log (first dump for ${kname}) ---"
      head -20 _tfc.log 2>/dev/null | sed 's/^/      /'
      echo "  --- _tfi.log (first dump for ${kname}) ---"
      head -20 _tfi.log 2>/dev/null | sed 's/^/      /'
      echo "  --- end ---"
    fi
  fi
}

# --------------------------------------------------------------------
# 5. Sweep
# --------------------------------------------------------------------
for reg in ${REGIMES}; do
  case "${reg}" in
    small_sq) for s in "${SMALL_SQ[@]}"; do run_shape small_sq $s; done ;;
    big_sq)   for s in "${BIG_SQ[@]}";   do run_shape big_sq   $s; done ;;
    small_ns) for s in "${SMALL_NS[@]}"; do run_shape small_ns $s; done ;;
    big_ns)   for s in "${BIG_NS[@]}";   do run_shape big_ns   $s; done ;;
    *)        echo "WARNING: unknown regime '${reg}' (skipped)" ;;
  esac
done

echo
echo "=== Done ==="
echo "CSV: $(pwd)/${OUT_CSV}  (regime,M,K,N,who,gflops,time_ms)"
