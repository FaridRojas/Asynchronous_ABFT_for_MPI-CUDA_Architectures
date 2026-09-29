#!/bin/bash
#SBATCH --job-name=abft_compare
#SBATCH --partition=gpu_titan
#SBATCH --nodelist=felix
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --output=scripts/abft_compare_%j.out
#SBATCH --error=scripts/abft_compare_%j.err

# =====================================================================
# 4-regime comparison vs Fault-Tolerant-SGEMM-on-NVIDIA-GPUs, one GPU:
#
#   small_sq   square,     ~256 .. 1024
#   big_sq     square,     ~2048 .. 10240
#   small_ns   non-square, small dims
#   big_ns     non-square, big dims
#
# Per shape we collect, into compare_all.csv (long format):
#   ours_baseline / ours_online      (our cuBLAS + our online ABFT)
#   theirs_cublas / theirs_fused     (their cuBLAS + their fused ft_sgemm_huge)
# as GFLOPS; temporal overhead is derived from GFLOPS in the plotter
# (t = 2MNK/GFLOPS).  Their driver is square-only, so we patch it to read
# FT_M/FT_N/FT_K env vars and feed non-square dims to their kernels.
# =====================================================================

set -euo pipefail
# Track which kernel names have already had their theirs-log dumped so
# we don't flood the .out — one dump per failing kname is enough.
declare -gA _THEIRS_DUMPED=()

module purge
module load cuda/11.8
module load mpi/openmpi-gcc11/4.1.6

MPICC_PATH="$(which mpicc)"
MPI_PREFIX="$(dirname "$(dirname "$MPICC_PATH")")"

# ---------- shape sweeps: 20 small / 40 big, every regime ----------
# SMALL: fine grid in (0,1024] like Fig.19 of the reference paper, NOT
#   forced to 128 (16-aligned; their small 16x16 kernel + our cuBLAS are
#   fine).  BIG: 128-aligned (their HUGE kernel has 128x128 tiles, no
#   partial-tile guard).  NON-SQUARE = fixed K, swept M=N (Fig.19(b)).
# Env-overridable for quick single-graph smoke runs, e.g.:
#   REGIMES=small_sq SMALL_N=6 sbatch scripts/run_comparison_felix.sbatch
# Full-run defaults (env-overridable for quick iterations).
# Total shapes = 2 small regimes * SMALL_N + 2 big regimes * BIG_N.
# 20+40+20+40 = 120 shapes.  Use SMALL_N=8 BIG_N=10 for fast smoke runs.
SMALL_N=${SMALL_N:-20}; BIG_N=${BIG_N:-40}
REGIMES="${REGIMES:-small_sq big_sq small_ns big_ns}"
SMALL_K_NS=256                 # fixed K for the small non-square regime
BIG_K_NS=1024                  # fixed K for the big   non-square regime
small_grid() { local i s; for ((i=0;i<SMALL_N;i++)); do
    s=$(python3 -c "print(int(round(($i+1)*1024.0/${SMALL_N}/16))*16)")
    echo "$s"; done; }
# 40 sizes spanning ~0..10k (Fig.21 style): 256,512,...,10240.
# 256-aligned => their huge kernel's M,N%128 / K%8 constraints hold.
big_grid()   { local i; for ((i=0;i<BIG_N;i++)); do echo $(( 256*(i+1) )); done; }
mapfile -t SMALL_SQ < <(small_grid | awk '!s[$0]++ {print $1, $1, $1}')
mapfile -t BIG_SQ   < <(big_grid   | awk '!s[$0]++ {print $1, $1, $1}')
mapfile -t SMALL_NS < <(small_grid | awk -v K=${SMALL_K_NS} '!s[$0]++ {print $1, K, $1}')
mapfile -t BIG_NS   < <(big_grid   | awk -v K=${BIG_K_NS}   '!s[$0]++ {print $1, K, $1}')

OUR_THRESHOLD="1.0"     # perf-only comparison; detection accuracy irrelevant
OUR_F=8
# Methodology (PERF comparison): timing of a dense cuBLAS GEMM is operand-
# VALUE-independent (same M,N,K => same FLOPs/access pattern), so we do NOT
# reseed mid-measurement.  Fix one random (A,B), discard a few warm-up trials
# to boost the GPU clocks, then time many back-to-back GEMMs at steady clocks
# — exactly THEIR harness (srand once, then num_tests back-to-back iters).
#
# IMPORTANT: do NOT add --reseed-per-trial here.  Its per-sample H2D re-upload
# inserts an idle gap that lets clocks ramp DOWN, making each single timed
# GEMM run cold (even our plain baseline then looks slower than theirs).
# Reseeding belongs only in the CM campaign (operand diversity => detection).
OUR_R=1
OUR_SAMPLES=${OUR_SAMPLES:-15}      # back-to-back timed GEMMs on fixed A,B
# Apples-to-apples timed-iteration count.  Their binary times num_tests=5
# iters per process invocation (hardcoded in sgemm.cu), so to match OUR
# 5 timed GEMMs we launch theirs 5/5 = 1 time.  The launch reuses their
# stock fixed (A,B) (srand(10)); since dense-GEMM timing is value-
# independent, that single launch still captures run-to-run clock jitter
# exactly like ours' 5 back-to-back iters.
#   THEIRS_SAMPLES × 5  ==  OUR_SAMPLES   (keep this invariant if you tune)
THEIRS_SAMPLES=${THEIRS_SAMPLES:-3}
WARMUPS=${WARMUPS:-5}    # discarded warm-up samples, all regimes (boost clocks)
# Where the one-time input encoding (colSumA + expectedRow) sits relative to
# the timed window:
#   overlap    (default) inside the window, enqueued on verify_stream so it
#              runs concurrently with iteration 0's GEMMs — the encoding is
#              charged to the measurement but hidden by the overlap
#   timed      inside the window, completed before the loop (sequential charge)
#   amortized  before the window (excluded; the original protocol)
ENCODING=${ENCODING:-overlap}
OUR_ENC_FLAG="--encoding-mode ${ENCODING}"
CUDA_ARCH="52"
THEIRS_DIR="Fault-Tolerant-SGEMM-on-NVIDIA-GPUs"
OUT_CSV="compare_all.csv"

# =========================================================
# 1. Provision cuda-samples + patch + build their project
# =========================================================
[[ -d "${THEIRS_DIR}" ]] || { echo "ERROR: ${THEIRS_DIR}/ missing (run upload.sh)"; exit 1; }

THEIRS_COMMON="${THEIRS_DIR}/cuda-samples/Common"
if [[ ! -f "${THEIRS_COMMON}/helper_functions.h" ]]; then
    echo "=== Fetching cuda-samples/Common/ ==="
    FOUND_INC=""
    for p in "${CUDA_HOME:-}/samples/common/inc" \
             "/usr/local/cuda-11.8/samples/common/inc" \
             "/usr/local/cuda/samples/common/inc"; do
        [[ -n "$p" && -f "$p/helper_functions.h" ]] && { FOUND_INC="$p"; break; }
    done
    if [[ -n "${FOUND_INC}" ]]; then
        mkdir -p "${THEIRS_COMMON}"; cp -r "${FOUND_INC}/." "${THEIRS_COMMON}/"
    else
        rm -rf "${THEIRS_DIR}/cuda-samples"
        git clone --depth 1 https://github.com/NVIDIA/cuda-samples.git \
            "${THEIRS_DIR}/cuda-samples"
    fi
    [[ -f "${THEIRS_COMMON}/helper_functions.h" ]] || { echo "ERROR: no cuda-samples"; exit 1; }
fi

# Patch arch (their CMakeLists hardcodes sm_80; felix is sm_52).
sed -i -E "s/set\\(CMAKE_CUDA_ARCHITECTURES[[:space:]]+[0-9]+\\)/set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})/" \
    "${THEIRS_DIR}/CMakeLists.txt"

# Patch their driver: their perf loop forces N=K=M=max_size (square only).
# Append an env-var override so we can feed non-square M,K,N to their
# kernels (which already accept arbitrary M,N,K).  Idempotent: only patch
# if the marker isn't already there.
SGEMM_SRC="${THEIRS_DIR}/kernel/ft_sgemm/sgemm.cu"
if ! grep -q "FT_M override" "${SGEMM_SRC}"; then
    echo "=== Patching ${SGEMM_SRC} for non-square (FT_M/FT_N/FT_K) ==="
    sed -i 's@N = K = M = max_size;@N = K = M = max_size; /*FT_M override*/ { char*_em=getenv("FT_M"); if(_em)M=atoi(_em); char*_en=getenv("FT_N"); if(_en)N=atoi(_en); char*_ek=getenv("FT_K"); if(_ek)K=atoi(_ek); }@' \
        "${SGEMM_SRC}"
    grep -q "FT_M override" "${SGEMM_SRC}" || { echo "ERROR: driver patch failed"; exit 1; }
fi
# Apples-to-apples timing: their perf harness times num_tests=5 iters with
# NO warm-up (cold clocks) — the source of theirs-side negative/jittery
# overhead.  Inject FT_WARMUP discarded iters before the timed window and
# (re)record `beg` at ii==FT_WARMUP; gflops still divides by num_tests
# (= count of timed iters), so the formula is unchanged.
if ! grep -q "FT_WARMUP" "${SGEMM_SRC}"; then
    sed -i 's@#define multi 20@#define multi 20\n#define FT_WARMUP 5  /* discarded warm-up iters before timed num_tests */@' \
        "${SGEMM_SRC}"
    sed -i 's@for(int ii = 0; ii < num_tests; ++ii){@for(int ii = 0; ii < num_tests + FT_WARMUP; ++ii){ if(ii==FT_WARMUP) cudaEventRecord(beg);@g' \
        "${SGEMM_SRC}"
    grep -q "FT_WARMUP" "${SGEMM_SRC}" || { echo "ERROR: their warm-up patch failed"; exit 1; }
fi

# Their fault injection is hardcoded in EVERY ft_sgemm_*.cuh and cannot be
# toggled at runtime, so build their kernels TWICE — inject ON and
# error_inject forced 0.0 — to fairly compare their ABFT with vs without a
# fault (analog of OURS --inject swifi vs none).  Patch ALL headers (small
# regimes use ft_sgemm_small, big regimes ft_sgemm_huge — see run_shape).
FT_HDRS=( "${THEIRS_DIR}"/kernel/ft_sgemm/include_code_gen/ft_sgemm_*.cuh )
[[ -e "${FT_HDRS[0]}" ]] || { echo "ERROR: no ft_sgemm_*.cuh headers"; exit 1; }
build_theirs() {                       # $1 = clean|inj
    local tag="$1" val="10000.0" h
    [[ "${tag}" == clean ]] && val="0.0"
    for h in "${FT_HDRS[@]}"; do
        grep -q "error_inject" "$h" || continue
        sed -i -E "s/(float[[:space:]]+error_inject[[:space:]]*=[[:space:]]*)[0-9.]+f?;/\\1${val};/" "$h"
        grep -q "error_inject *= *${val};" "$h" \
            || { echo "ERROR: error_inject patch (${tag}) failed in $h"; exit 1; }
    done
    echo "=== Building their ft_sgemm [${tag}] ==="
    rm -rf "${THEIRS_DIR}/build"; mkdir -p "${THEIRS_DIR}/build"
    ( cd "${THEIRS_DIR}/build" && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4 )
    local b; b="$(find ${THEIRS_DIR}/build -name ft_sgemm -type f -executable | head -1)"
    [[ -n "${b}" ]] || { echo "ERROR: ft_sgemm [${tag}] not built"; exit 1; }
    cp "${b}" "${THEIRS_DIR}/ft_sgemm_${tag}"
}
build_theirs clean
build_theirs inj
THEIRS_CLEAN="${THEIRS_DIR}/ft_sgemm_clean"
THEIRS_INJ="${THEIRS_DIR}/ft_sgemm_inj"

echo "=== Building our abft_gemm ==="
nvcc -O3 -std=c++17 -ccbin g++ -I"${MPI_PREFIX}/include" \
  src/main.cu -o abft_gemm -L"${MPI_PREFIX}/lib" -lmpi -lcublas

# =========================================================
# 2. Helpers
# =========================================================
echo "regime,M,K,N,who,gflops,time_ms" > "${OUT_CSV}"

# Parse one GFLOPS value from their stdout for a given row-name.
their_gflops() {  # $1=logfile  $2=row-name (cublas|abft_kernel_huge)
    awk -F'|' -v name="$2" '
        $0 ~ "^"name" " { v=$2; gsub(/ /,"",v); print v; exit }' "$1"
}

# Convert GFLOPS -> ms for a given (M,K,N): t_ms = 2*M*N*K / (gflops*1e9) * 1e3.
gflops_to_ms() {  # $1=gflops $2=M $3=K $4=N
    python3 -c "
g=float('$1'); M=int('$2'); K=int('$3'); N=int('$4')
print('' if g<=0 else 2.0*M*N*K/(g*1e9)*1e3)" 2>/dev/null
}

mean_lines() { awk '$0!="" {s+=$1; n++} END {if(n>0) printf "%.6f",s/n}'; }

last_protected() {   # $1 = abft csv
    # echoes: base_gflops_mean  base_time_ms_mean  prot_gflops_mean  prot_time_ms_mean
    # MEAN (not median) to match theirs, whose GFLOPS = 2*num_tests*MNK /
    # total_time is intrinsically a mean over its timed iters.
    python3 - "$1" <<'PY'
import csv, sys
row=None
for row in csv.DictReader(open(sys.argv[1])): pass
print(row["baseline_gflops_mean"], row["baseline_mean_ms"],
      row["protected_gflops_mean"], row["protected_mean_ms"])
PY
}

# Six series so every comparison is apples-to-apples (see plot_comparison.py):
#   ours_baseline / ours_online / ours_online_inj
#   theirs_cublas / theirs_fused / theirs_fused_inj
run_shape() {  # $1=regime  $2=M $3=K $4=N
    local regime="$1" M="$2" K="$3" N="$4"
    local smax=$(( M>N ? (M>K?M:K) : (N>K?N:K) ))
    local WU="${WARMUPS}"
    echo
    echo "### ${regime}  M=${M} K=${K} N=${N}  (Smax=${smax})  samples=${OUR_SAMPLES}  warmups=${WU}"

    # ----- ours, NO fault (also yields the cuBLAS baseline) -----
    # One invocation; the binary warms up ${WU} trials then times
    # OUR_SAMPLES back-to-back GEMMs on FIXED A,B (no reseed — timing is
    # operand-value-independent, keeps the GPU clocks boosted).
    rm -f abft_metrics_cmp.csv _ours_launch.log
    if ! srun -n 1 --gres=gpu:1 --mpi=pmix ./abft_gemm ${M} ${K} ${N} \
         --inject none --threshold ${OUR_THRESHOLD} --grid 1 1 \
         --frags-per-rank ${OUR_F} ${OUR_ENC_FLAG} \
         --samples ${OUR_SAMPLES} --repeats ${OUR_R} \
         --warmups ${WU} --csv abft_metrics_cmp.csv > /dev/null 2> _ours_launch.log; then
        echo "  WARNING: ours(none) launch failed ${M}x${K}x${N} (stderr below):"
        sed 's/^/    | /' _ours_launch.log 2>/dev/null | head -20
        return 0
    fi
    local ob_g ob_t op_g op_t
    if ! read ob_g ob_t op_g op_t < <(last_protected abft_metrics_cmp.csv 2>&1); then
        echo "  WARNING: ours(none) ${M}x${K}x${N}: empty/bad CSV (no *_mean columns?):"
        ls -l abft_metrics_cmp.csv 2>&1 | sed 's/^/    | /'
        head -3 abft_metrics_cmp.csv 2>/dev/null | sed 's/^/    | /'
        return 0
    fi
    echo "${regime},${M},${K},${N},ours_baseline,${ob_g},${ob_t}" >> "${OUT_CSV}"
    echo "${regime},${M},${K},${N},ours_online,${op_g},${op_t}"   >> "${OUT_CSV}"

    # ----- ours, WITH a fault -----
    rm -f abft_metrics_inj.csv
    if srun -n 1 --gres=gpu:1 --mpi=pmix ./abft_gemm ${M} ${K} ${N} \
         --inject add --threshold ${OUR_THRESHOLD} --grid 1 1 \
         --frags-per-rank ${OUR_F} ${OUR_ENC_FLAG} \
         --samples ${OUR_SAMPLES} --repeats ${OUR_R} \
         --warmups ${WU} --csv abft_metrics_inj.csv > /dev/null 2>> _ours_launch.log; then
        local oi_g oi_t
        if read _ _ oi_g oi_t < <(last_protected abft_metrics_inj.csv 2>&1); then
            echo "${regime},${M},${K},${N},ours_online_inj,${oi_g},${oi_t}" >> "${OUT_CSV}"
        else
            echo "  WARNING: ours(inj) ${M}x${K}x${N}: empty/bad CSV"
        fi
    else
        echo "  WARNING: ours(inj) launch failed ${M}x${K}x${N}"
    fi

    # ----- theirs: external THEIRS_SAMPLES-call loop (stock fixed A,B) -----
    # Pick the size/shape-appropriate kernel.  Dispatch criterion is
    # output-tile shape (driven by M,N), NOT max(M,N,K) — K only affects
    # flops per tile, not tile geometry.  See sgemm.cu line 240 for the
    # non-contiguous kernel_number mapping (11..16).
    local kidx kname
    local out_max=$(( M>N ? M : N ))
    if   (( M >= 2*N )); then kidx=14; kname="abft_kernel_tall"
    elif (( N >= 2*M )); then kidx=15; kname="abft_kernel_wide"
    elif (( out_max <= 128 )); then kidx=11; kname="abft_kernel_small"
    elif (( out_max <= 256 )); then kidx=12; kname="abft_kernel_medium"
    elif (( out_max <= 512 )); then kidx=13; kname="abft_kernel_large"
    else                            kidx=16; kname="abft_kernel_huge"
    fi
    local _tc_acc="" _tfc_acc="" _tfi_acc=""
    local s
    for ((s=0; s<THEIRS_SAMPLES; s++)); do
        FT_M=${M} FT_N=${N} FT_K=${K} \
            "${THEIRS_CLEAN}" ${smax} ${smax} 1 0 0             > _tc.log  2>&1 || true
        FT_M=${M} FT_N=${N} FT_K=${K} \
            "${THEIRS_CLEAN}" ${smax} ${smax} 1 ${kidx} ${kidx} > _tfc.log 2>&1 || true
        FT_M=${M} FT_N=${N} FT_K=${K} \
            "${THEIRS_INJ}"   ${smax} ${smax} 1 ${kidx} ${kidx} > _tfi.log 2>&1 || true
        local v
        v=$(their_gflops  _tc.log  cublas);    [[ -n "${v}" ]] && _tc_acc+="${v}"$'\n'
        v=$(their_gflops _tfc.log "${kname}"); [[ -n "${v}" ]] && _tfc_acc+="${v}"$'\n'
        v=$(their_gflops _tfi.log "${kname}"); [[ -n "${v}" ]] && _tfi_acc+="${v}"$'\n'
    done
    local tc_g tfc_g tfi_g
    tc_g=$(printf '%s' "${_tc_acc}"  | mean_lines)
    tfc_g=$(printf '%s' "${_tfc_acc}" | mean_lines)
    tfi_g=$(printf '%s' "${_tfi_acc}" | mean_lines)
    local tc_t tfc_t tfi_t
    # Convert each non-empty GFLOPS to ms and emit a CSV row.  Use plain
    # if/fi so set -e does NOT exit on the harmless `[[ -n "" ]]` failure
    # when a kernel produced no parseable output.
    tc_t=""; tfc_t=""; tfi_t=""
    if [[ -n "${tc_g}"  ]]; then tc_t=$(gflops_to_ms  "${tc_g}"  ${M} ${K} ${N}); fi
    if [[ -n "${tfc_g}" ]]; then tfc_t=$(gflops_to_ms "${tfc_g}" ${M} ${K} ${N}); fi
    if [[ -n "${tfi_g}" ]]; then tfi_t=$(gflops_to_ms "${tfi_g}" ${M} ${K} ${N}); fi
    if [[ -n "${tc_g}"  ]]; then echo "${regime},${M},${K},${N},theirs_cublas,${tc_g},${tc_t}"      >> "${OUT_CSV}"; fi
    if [[ -n "${tfc_g}" ]]; then echo "${regime},${M},${K},${N},theirs_fused,${tfc_g},${tfc_t}"     >> "${OUT_CSV}"; fi
    if [[ -n "${tfi_g}" ]]; then echo "${regime},${M},${K},${N},theirs_fused_inj,${tfi_g},${tfi_t}" >> "${OUT_CSV}"; fi
    # Dump the theirs logs once per kname (first failing shape only) so
    # we can see what the FT kernel actually printed.
    if [[ -z "${tfc_g}" || -z "${tfi_g}" ]]; then
        echo "  NOTE: theirs_fused empty for ${M}x${K}x${N} (kname=${kname}, kidx=${kidx})"
        if [[ -z "${_THEIRS_DUMPED[${kname}]:-}" ]]; then
            _THEIRS_DUMPED[${kname}]=1
            echo "  --- _tfc.log (CLEAN, first dump for ${kname}) head ---"
            head -30 _tfc.log 2>/dev/null | sed 's/^/      /'
            echo "  --- _tfi.log (INJ,   first dump for ${kname}) head ---"
            head -30 _tfi.log 2>/dev/null | sed 's/^/      /'
            echo "  --- end ---"
        fi
    fi
}

# =========================================================
# 3. Run all four regimes
# =========================================================
for reg in ${REGIMES}; do
  case "${reg}" in
    small_sq) for s in "${SMALL_SQ[@]}"; do run_shape small_sq $s; done ;;
    big_sq)   for s in "${BIG_SQ[@]}";   do run_shape big_sq   $s; done ;;
    small_ns) for s in "${SMALL_NS[@]}"; do run_shape small_ns $s; done ;;
    big_ns)   for s in "${BIG_NS[@]}";   do run_shape big_ns   $s; done ;;
    *) echo "WARNING: unknown regime '${reg}' (skipped)" ;;
  esac
done

# =========================================================
# 4. Plots — one GFLOPS + one overhead figure per regime
# =========================================================
mkdir -p scripts/plots/comparison
python3 scripts/plot_comparison.py \
        --csv "${OUT_CSV}" \
        --outdir scripts/plots/comparison \
    || echo "(plots skipped: install pandas matplotlib)"

echo
echo "=== Done ==="
echo "  ${OUT_CSV}  (regime,M,K,N,who,gflops)"
ls scripts/plots/comparison/ | sed 's/^/    /'
