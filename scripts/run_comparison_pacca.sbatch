#!/bin/bash
#SBATCH --job-name=abft_cmp_pacca
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=scripts/abft_compare_pacca_%j.out
#SBATCH --error=scripts/abft_compare_pacca_%j.err

# =====================================================================
# 4-regime comparison on the PACCA A100 (sm_80), 20 shapes/regime, vs
# Fault-Tolerant-SGEMM-on-NVIDIA-GPUs.  Same logic as the felix job but
# A100 arch and defensive module loading (adjust module names if your
# site differs — everything also works if the tools are already on PATH).
# =====================================================================
set -euo pipefail
# --- diagnostic traps: if the script dies for any reason, print what
# command was running on which line.  ERR fires on any non-zero return
# under set -e; EXIT always fires last.  Both go to stderr so they end
# up in the SLURM .err file even if the .out is mid-buffer when killed.
trap 'rc=$?; echo "TRAP ERR rc=${rc} line=${LINENO} cmd=[${BASH_COMMAND}]" >&2' ERR
trap 'rc=$?; echo "TRAP EXIT rc=${rc} line=${LINENO} cmd=[${BASH_COMMAND}]" >&2' EXIT
# Track which kernel names have already had their theirs-log dumped so
# we don't flood the .out — one dump per failing kname is enough.
declare -gA _THEIRS_DUMPED=()

# --- pacca toolchain (OpenHPC/Lmod): CUDA ships INSIDE the NVIDIA HPC SDK
#     (module nvhpc/23.1, loaded by default).  Do NOT 'module purge' here —
#     that would drop the default gnu12 + nvhpc compilers.  Just load what
#     we need on top; missing names are tolerated. ---
module load devtools/nvidia/hpc_sdk/nvhpc/23.1            2>/dev/null || true
module load openmpi4/4.1.6        2>/dev/null \
  || module load devtools/mpi/openmpi4-gnu12 2>/dev/null || true
module load cmake/3.24.2          2>/dev/null || module load cmake 2>/dev/null || true
module load pmix/4.2.9            2>/dev/null || true
module load prun                  2>/dev/null || true

# nvcc lives in the HPC SDK's cuda/bin, which is NOT auto-added to PATH
# (only nvc++ is).  Locate it from nvc++ / NVHPC env / the ohpc tree.
if ! command -v nvcc >/dev/null 2>&1; then
    CAND=""
    if command -v nvc++ >/dev/null 2>&1; then
        VERDIR="$(dirname "$(dirname "$(dirname "$(readlink -f "$(command -v nvc++)")")")")"
        for c in "$VERDIR"/cuda/bin/nvcc "$VERDIR"/cuda/*/bin/nvcc; do
            [[ -x "$c" ]] && { CAND="$(dirname "$c")"; break; }
        done
    fi
    if [[ -z "$CAND" ]]; then
        for v in "${NVHPC_ROOT:-}" "${NVHPC:-}" "${CUDA_HOME:-}" "${CUDADIR:-}"; do
            [[ -z "$v" ]] && continue
            for c in "$v"/cuda/bin/nvcc "$v"/cuda/*/bin/nvcc "$v"/bin/nvcc; do
                [[ -x "$c" ]] && { CAND="$(dirname "$c")"; break; }
            done
            [[ -n "$CAND" ]] && break
        done
    fi
    if [[ -z "$CAND" ]]; then
        for c in /opt/ohpc/pub/*/nvhpc*/Linux_x86_64/*/cuda/bin/nvcc \
                 /opt/ohpc/pub/*/nvhpc*/Linux_x86_64/*/cuda/*/bin/nvcc; do
            [[ -x "$c" ]] && { CAND="$(dirname "$c")"; break; }
        done
    fi
    [[ -n "$CAND" ]] && export PATH="$CAND:$PATH"
fi
command -v nvcc >/dev/null 2>&1 || {
    echo "ERROR: nvcc still not found.  On pacca CUDA is bundled in nvhpc/23.1."
    echo "Diagnose with:   which nvc++ ;  ls \$(dirname \$(dirname \$(which nvc++)))/../cuda*/bin"
    echo "then prepend that dir to PATH at the top of this sbatch."
    exit 1
}
echo "Using nvcc: $(command -v nvcc)"; nvcc --version | tail -1 || true

# CUDA runtime libs: their cmake-built ft_sgemm dynamically links
# libcudart.so.12 / libcublas.  NVHPC nests these deeply (e.g.
# .../<ver>/cuda/<cuver>/targets/x86_64-linux/lib), so GUESS-by-path is
# unreliable -> actually FIND the file (first match, -quit = fast).
CUDA_BIN="$(dirname "$(readlink -f "$(command -v nvcc)")")"
CUDA_ROOT="$(dirname "${CUDA_BIN}")"
NVHPC_TOP="$(dirname "$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")")"
# NVHPC keeps cudart under cuda/<v>/targets/.../lib but cublas under a
# SEPARATE math_libs/<v>/targets/.../lib — locate BOTH.
CUDA_SEARCH_ROOTS=( "${CUDA_ROOT}" "${CUDA_ROOT%/*}" "${CUDA_HOME:-}" \
    "${NVHPC_ROOT:-}" "${NVHPC_TOP}" \
    /opt/ohpc/pub/devtools/nvidia/hpc_sdk /usr/local/cuda )
find_libdir() {  # $1 = libname glob -> echoes dir of first match
    local g="$1" r f
    for r in "${CUDA_SEARCH_ROOTS[@]}"; do
        [[ -n "$r" && -d "$r" ]] || continue
        f="$(find "$r" -maxdepth 9 -name "$g" -print -quit 2>/dev/null)"
        [[ -n "$f" ]] && { dirname "$f"; return 0; }
    done
}
CUDART_DIR="$(find_libdir 'libcudart.so.12*')"
CUBLAS_DIR="$(find_libdir 'libcublas.so.12*')"
CUDA_LD=""
[[ -n "${CUDART_DIR}" ]] && CUDA_LD="${CUDART_DIR}"
[[ -n "${CUBLAS_DIR}" && "${CUBLAS_DIR}" != "${CUDART_DIR}" ]] \
    && CUDA_LD="${CUDA_LD:+${CUDA_LD}:}${CUBLAS_DIR}"
[[ -n "${CUDA_LD}" ]] && export LD_LIBRARY_PATH="${CUDA_LD}:${LD_LIBRARY_PATH:-}"
echo "libcudart dir: ${CUDART_DIR:-<NOT FOUND>}"
echo "libcublas dir: ${CUBLAS_DIR:-<NOT FOUND>}"
export OMPI_MCA_btl_openib_warn_no_device_params_found=0

MPICC="$(command -v mpicc || true)"
if [[ -n "${MPICC}" ]]; then
    MPI_PREFIX="$(dirname "$(dirname "$(readlink -f "${MPICC}")")")"
else
    MPI_PREFIX="${MPI_PREFIX:-/usr}"
    echo "WARNING: mpicc not found; assuming MPI under ${MPI_PREFIX}"
fi
# Derive MPI flags from mpicc --showme.  CRUCIAL on pacca: the NVHPC SDK
# hijacks mpicc so --showme reports the SDK's bundled OpenMPI (HPCX), even
# though `which mpicc` lives under openmpi4-gnu12.  The binary links the
# SDK MPI, so OPAL_PREFIX / mpirun / LD_LIBRARY_PATH must ALL point at the
# SDK MPI (= dirname of the --showme libdir), not at `which mpicc`s prefix.
# Mismatch here is exactly what caused opal_shmem_base_open / -np failures.
MPI_INCD="$(mpicc --showme:incdirs 2>/dev/null || true)"
MPI_LIBD="$(mpicc --showme:libdirs 2>/dev/null || true)"
MPI_LIBS="$(mpicc --showme:libs    2>/dev/null || true)"
[[ -z "${MPI_INCD}" ]] && MPI_INCD="${MPI_PREFIX}/include"
if [[ -z "${MPI_LIBD}" ]]; then
    for d in "${MPI_PREFIX}/lib" "${MPI_PREFIX}/lib64"; do
        [[ -e "$d/libmpi.so" ]] && MPI_LIBD="$d"
    done
fi
[[ -z "${MPI_LIBS}" ]] && MPI_LIBS="mpi"
MPI_INC_FLAGS=""; for d in ${MPI_INCD}; do MPI_INC_FLAGS+=" -I$d"; done
MPI_LNK_FLAGS=""; for d in ${MPI_LIBD}; do MPI_LNK_FLAGS+=" -L$d"; done
for l in ${MPI_LIBS}; do MPI_LNK_FLAGS+=" -l$l"; done

# Runtime MPI prefix = parent of the (first) --showme libdir => the SAME
# OpenMPI the binary links against.
MPI_RT_LIBD="$(echo ${MPI_LIBD} | awk '{print $1}')"
MPI_RT_PREFIX="$(dirname "${MPI_RT_LIBD}")"
export OPAL_PREFIX="${MPI_RT_PREFIX}"
export LD_LIBRARY_PATH="${MPI_LIBD// /:}:${LD_LIBRARY_PATH:-}"
MPIRUN="${MPI_RT_PREFIX}/bin/mpirun"
[[ -x "${MPIRUN}" ]] || MPIRUN="${MPI_PREFIX}/bin/mpirun"
[[ -x "${MPIRUN}" ]] || MPIRUN="$(command -v mpirun || echo mpirun)"
echo "MPI (which mpicc): ${MPI_PREFIX}"
echo "MPI (linked,RT)  : ${MPI_RT_PREFIX}"
echo "OPAL_PREFIX      : ${OPAL_PREFIX}"
echo "MPI build flags  : ${MPI_INC_FLAGS} |${MPI_LNK_FLAGS}"
echo "MPIRUN           : ${MPIRUN}"

# ---------- shape sweeps: 20 small / 40 big, every regime ----------
# SMALL: a fine grid in (0,1024] like Fig.19 of the reference paper — NOT
#   forced to 128.  16-aligned only (their small 16x16-tiled kernel +
#   our cuBLAS are both fine with that; big sizes still use 128 because
#   their HUGE kernel has 128x128 tiles with no partial-tile guard).
# NON-SQUARE: like Fig.19(b) — K is FIXED, M=N is swept.
# Env-overridable for quick single-graph smoke runs, e.g.:
#   REGIMES=small_sq SMALL_N=6 sbatch scripts/run_comparison_pacca.sbatch
# plot_comparison.py only emits graphs for regimes present in the CSV.
# Full-run defaults (env-overridable for quick iterations).
# Total shapes = 2 small regimes * SMALL_N + 2 big regimes * BIG_N.
# 20+40+20+40 = 120 shapes.  Use SMALL_N=8 BIG_N=10 for fast smoke runs.
SMALL_N=${SMALL_N:-20}; BIG_N=${BIG_N:-40}
REGIMES="${REGIMES:-small_sq big_sq small_ns big_ns}"
SMALL_K_NS=${SMALL_K_NS:-256}  # fixed K for the small non-square regime
BIG_K_NS=${BIG_K_NS:-1024}     # fixed K for the big   non-square regime
SMALL_UPPER=${SMALL_UPPER:-1024}  # upper bound for the small auto-grid
# Paper-replication mode: if EXPLICIT_SIZES is set (space-separated 1D
# values, e.g. "64 80 96 112"), use it verbatim as the size list for
# BOTH small_sq and small_ns regimes — overrides the auto-grid.  Useful
# for replicating Wu et al. Fig.20 (12 specific sizes) and Fig.19 (20
# sizes from 0..SMALL_UPPER).
EXPLICIT_SIZES="${EXPLICIT_SIZES:-}"
# Output suffix so paper-replication runs don't clobber the main sweep's
# compare_all.csv / scripts/plots/comparison/ outputs.  Empty = no suffix.
OUT_TAG="${OUT_TAG:-}"
# small grid: SMALL_N sizes, 16-aligned, ~ (SMALL_UPPER/SMALL_N .. SMALL_UPPER]
small_grid() {
    if [[ -n "${EXPLICIT_SIZES}" ]]; then
        printf '%s\n' ${EXPLICIT_SIZES}
        return
    fi
    local i s; for ((i=0;i<SMALL_N;i++)); do
    s=$(python3 -c "print(int(round(($i+1)*${SMALL_UPPER}.0/${SMALL_N}/16))*16)")
    echo "$s"; done
}
# big grid: 40 sizes, 128-aligned, 3072 .. 13056 (huge-kernel safe)
# 40 sizes spanning ~0..10k (Fig.21 style): 256,512,...,10240.
# 256-aligned => their huge kernel's M,N%128 / K%8 constraints hold.
big_grid()   { local i; for ((i=0;i<BIG_N;i++)); do echo $(( 256*(i+1) )); done; }
mapfile -t SMALL_SQ < <(small_grid | awk '!s[$0]++ {print $1, $1, $1}')
mapfile -t BIG_SQ   < <(big_grid   | awk '!s[$0]++ {print $1, $1, $1}')
mapfile -t SMALL_NS < <(small_grid | awk -v K=${SMALL_K_NS} '!s[$0]++ {print $1, K, $1}')
mapfile -t BIG_NS   < <(big_grid   | awk -v K=${BIG_K_NS}   '!s[$0]++ {print $1, K, $1}')

# Methodology (comparison): 20 independent samples per (M,K,N) per phase,
# Methodology (PERF comparison): timing of a dense cuBLAS GEMM is operand-
# VALUE-independent (same M,N,K => same FLOPs, same access pattern), so we do
# NOT reseed mid-measurement.  We fix one random (A,B), discard a few warm-up
# trials to boost the GPU clocks, then time many back-to-back GEMMs at steady
# clocks.  This is exactly THEIR harness's methodology (srand once, then
# num_tests back-to-back iters) — apples-to-apples.
#
# IMPORTANT: do NOT add --reseed-per-trial here.  It re-uploads A,B (a large
# H2D copy) before every sample; that idle gap lets the A100 clocks ramp DOWN,
# so each single timed GEMM runs cold and even our plain cuBLAS baseline looks
# slower than theirs.  Reseeding belongs only in the CM campaign, where
# operand diversity actually changes detection accuracy.
OUR_THRESHOLD="1.0"; OUR_F=8; OUR_R=1
OUR_SAMPLES=${OUR_SAMPLES:-15}      # back-to-back timed GEMMs on fixed A,B
# Apples-to-apples timed-iteration count.  Their binary times num_tests=5
# iters per process invocation (hardcoded in sgemm.cu), so to match OUR
# 5 timed GEMMs we launch theirs 5/5 = 1 time.  The launch reuses their
# stock fixed (A,B) (srand(10)); since dense-GEMM timing is value-
# independent, that single launch still captures run-to-run clock jitter
# exactly like ours' 5 back-to-back iters.
#   THEIRS_SAMPLES × 5  ==  OUR_SAMPLES   (keep this invariant if you tune)
THEIRS_SAMPLES=${THEIRS_SAMPLES:-3}
WARMUPS=${WARMUPS:-5}          # discarded warm-up samples, all regimes
                               # (boost clocks before the timed window)
# Where the one-time input encoding (colSumA + expectedRow) sits relative to
# the timed window: overlap (default; charged but hidden behind iteration 0's
# GEMMs), timed (charged sequentially), amortized (excluded, original protocol).
ENCODING=${ENCODING:-overlap}
OUR_ENC_FLAG="--encoding-mode ${ENCODING}"
CUDA_ARCH="80"                 # A100 = Ampere sm_80
THEIRS_DIR="Fault-Tolerant-SGEMM-on-NVIDIA-GPUs"
# OUT_TAG (e.g. "fig19", "fig20") suffixes the CSV + plot dir so paper-
# replication runs don't clobber the main sweep's output.
_SUFFIX="${OUT_TAG:+_${OUT_TAG}}"
OUT_CSV="compare_all${_SUFFIX}.csv"
PLOT_DIR="scripts/plots/comparison${_SUFFIX}"

# --- OpenMPI runtime env (mirrors the proven-working interactive recipe:
#     OPAL_PREFIX + a writable TMPDIR only; do NOT force OMPI_MCA_shmem —
#     with the correct OPAL_PREFIX, OpenMPI selects shmem itself, and
#     forcing a component was what caused "shmem mmap not found"). --------
export TMPDIR="${TMPDIR:-/tmp}"
[[ -d "${TMPDIR}" && -w "${TMPDIR}" ]] || export TMPDIR=/tmp
mkdir -p "${TMPDIR}" 2>/dev/null || true
# If MPI_Init ever fails again: try `export OMPI_MCA_shmem=posix` or switch
# to module mpich/3.4.3-ofi (rebuild; launch with srun --mpi=pmi2).

# run_ours: launch our single-rank binary with the SAME OpenMPI it links.
# ${MPIRUN} is now the NVHPC-SDK HPCX mpirun (accepts -np); that is the
# launcher proven to work interactively.  Override with OURS_LAUNCH if
# needed (e.g. OURS_LAUNCH="srun --mpi=pmix -n 1 --gres=gpu:1").
run_ours() {
    # Disable errexit inside the fallback chain: each launcher is expected
    # to fail on pacca's mixed PMI/PMIx world, and a signal-killed mpirun
    # (e.g. HPCX SIGTERM propagation) would otherwise tear down the whole
    # script even from an `if !` context.  We collect rc explicitly.
    set +e
    : > _ours_launch.log
    local rc=1 tag=""
    if [[ -n "${OURS_LAUNCH:-}" ]]; then
        echo "[run_ours] trying OURS_LAUNCH" >> _ours_launch.log
        ${OURS_LAUNCH} ./abft_gemm "$@" 2>>_ours_launch.log
        rc=$?; tag="OURS_LAUNCH"
    else
        for try in \
            "${MPIRUN} -np 1" \
            "${MPIRUN} -n  1" \
            "srun --mpi=pmix -n 1 --gres=gpu:1" \
            "srun --mpi=pmi2 -n 1 --gres=gpu:1" \
            "prun" \
            "env -u SLURM_PROCID -u SLURM_JOBID -u SLURM_NODELIST -u SLURM_NTASKS -u SLURM_STEP_ID -u SLURM_LOCALID -u PMIX_RANK"; do
            echo "[run_ours] trying: ${try}" >> _ours_launch.log
            ${try} ./abft_gemm "$@" 2>>_ours_launch.log
            rc=$?
            tag="${try}"
            if (( rc == 0 )); then break; fi
            echo "[run_ours] -> rc=${rc}" >> _ours_launch.log
        done
    fi
    echo "[run_ours] final rc=${rc} via [${tag}]" >> _ours_launch.log
    set -e
    return ${rc}
}

# ---------- provision + patch + build their project ----------
[[ -d "${THEIRS_DIR}" ]] || { echo "ERROR: ${THEIRS_DIR}/ missing (run upload_pacca.sh)"; exit 1; }
THEIRS_COMMON="${THEIRS_DIR}/cuda-samples/Common"
if [[ ! -f "${THEIRS_COMMON}/helper_functions.h" ]]; then
    FOUND_INC=""
    for p in "${CUDA_HOME:-}/samples/common/inc" \
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
sed -i -E "s/set\\(CMAKE_CUDA_ARCHITECTURES[[:space:]]+[0-9]+\\)/set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})/" \
    "${THEIRS_DIR}/CMakeLists.txt"
SGEMM_SRC="${THEIRS_DIR}/kernel/ft_sgemm/sgemm.cu"
if ! grep -q "FT_M override" "${SGEMM_SRC}"; then
    sed -i 's@N = K = M = max_size;@N = K = M = max_size; /*FT_M override*/ { char*_em=getenv("FT_M"); if(_em)M=atoi(_em); char*_en=getenv("FT_N"); if(_en)N=atoi(_en); char*_ek=getenv("FT_K"); if(_ek)K=atoi(_ek); }@' \
        "${SGEMM_SRC}"
    grep -q "FT_M override" "${SGEMM_SRC}" || { echo "ERROR: driver patch failed"; exit 1; }
fi
# Apples-to-apples timing: their perf harness times num_tests=5 iters with
# NO warm-up (cold clocks/heuristics) — that is the source of the
# theirs-side negative/jittery overhead.  Inject FT_WARMUP discarded iters
# before the timed window: extend each timed loop to num_tests+FT_WARMUP
# and (re)record `beg` at ii==FT_WARMUP (the earlier standalone
# cudaEventRecord(beg) is harmlessly superseded).  gflops uses num_tests,
# which still equals the count of timed iters, so the formula is unchanged.
if ! grep -q "FT_WARMUP" "${SGEMM_SRC}"; then
    sed -i 's@#define multi 20@#define multi 20\n#define FT_WARMUP 5  /* discarded warm-up iters before timed num_tests */@' \
        "${SGEMM_SRC}"
    sed -i 's@for(int ii = 0; ii < num_tests; ++ii){@for(int ii = 0; ii < num_tests + FT_WARMUP; ++ii){ if(ii==FT_WARMUP) cudaEventRecord(beg);@g' \
        "${SGEMM_SRC}"
    grep -q "FT_WARMUP" "${SGEMM_SRC}" || { echo "ERROR: their warm-up patch failed"; exit 1; }
fi
# Their ABFT fault injection is HARDCODED in EVERY ft_sgemm_*.cuh kernel
# (float error_inject = 10000.0; ... res[0] += error_inject;).  It cannot
# be toggled at runtime, so for a FAIR comparison we build their kernels
# TWICE: upstream (inject ON) and with error_inject forced 0.0 (ABFT
# detect/correct path still runs, no fault) — the analog of OURS with
# --inject none vs swifi.  Patch ALL headers (small regimes use
# ft_sgemm_small, big regimes ft_sgemm_huge — see run_shape).
FT_HDRS=( "${THEIRS_DIR}"/kernel/ft_sgemm/include_code_gen/ft_sgemm_*.cuh )
[[ -e "${FT_HDRS[0]}" ]] || { echo "ERROR: no ft_sgemm_*.cuh headers"; exit 1; }
build_theirs() {                       # $1 = clean|inj
    local tag="$1" val="10000.0" h
    [[ "${tag}" == clean ]] && val="0.0"
    for h in "${FT_HDRS[@]}"; do
        grep -q "error_inject" "$h" || continue
        # idempotent: matches whatever numeric literal is currently there
        sed -i -E "s/(float[[:space:]]+error_inject[[:space:]]*=[[:space:]]*)[0-9.]+f?;/\\1${val};/" "$h"
        grep -q "error_inject *= *${val};" "$h" \
            || { echo "ERROR: error_inject patch (${tag}) failed in $h"; exit 1; }
    done
    echo "=== Building their ft_sgemm [${tag}] (sm_${CUDA_ARCH}) ==="
    rm -rf "${THEIRS_DIR}/build"; mkdir -p "${THEIRS_DIR}/build"
    ( cd "${THEIRS_DIR}/build" && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j16 )
    local b; b="$(find ${THEIRS_DIR}/build -name ft_sgemm -type f -executable | head -1)"
    [[ -n "${b}" ]] || { echo "ERROR: ft_sgemm [${tag}] not built"; exit 1; }
    cp "${b}" "${THEIRS_DIR}/ft_sgemm_${tag}"
}
build_theirs clean
build_theirs inj
THEIRS_CLEAN="${THEIRS_DIR}/ft_sgemm_clean"   # ABFT, NO injected fault
THEIRS_INJ="${THEIRS_DIR}/ft_sgemm_inj"       # ABFT, injected fault (upstream)

echo "=== Building our abft_gemm (sm_${CUDA_ARCH}) ==="
nvcc -O3 -std=c++17 -arch=sm_${CUDA_ARCH} -ccbin g++ \
  ${MPI_INC_FLAGS} \
  src/main.cu -o abft_gemm ${MPI_LNK_FLAGS} -lcublas

echo "regime,M,K,N,who,gflops,time_ms" > "${OUT_CSV}"

# Extract GFLOPS from their per-kernel report line, format:
#   "<kname>      | <gflops> | ..." — gflops is field 2 with whitespace.
their_gflops() { awk -F'|' -v name="$2" '
    $0 ~ "^"name" " { v=$2; gsub(/ /,"",v); print v; exit }' "$1"; }

# Convert GFLOPS -> ms for a given (M,K,N): t_ms = 2*M*N*K / gflops * 1e-6.
gflops_to_ms() {  # $1=gflops $2=M $3=K $4=N
    python3 -c "
g=float('$1'); M=int('$2'); K=int('$3'); N=int('$4')
print('' if g<=0 else 2.0*M*N*K/(g*1e9)*1e3)" 2>/dev/null
}

# Mean of stdin (one number per line, blanks ignored).
mean_lines() { awk '$0!="" {s+=$1; n++} END {if(n>0) printf "%.6f",s/n}'; }

last_protected() {   # $1 = abft csv
    # echoes: "base_gflops_mean  base_time_ms_mean  prot_gflops_mean  prot_time_ms_mean"
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

# Six series per shape so EVERY comparison is apples-to-apples:
#   ours_baseline     our cuBLAS, no ABFT          (ours' reference)
#   ours_online       our ABFT, no fault           } compare these two
#   ours_online_inj   our ABFT, fault (swifi)       }  vs each other
#   theirs_cublas     their cuBLAS, no ABFT        (theirs' reference)
#   theirs_fused      their ABFT, no fault          } compare these two
#   theirs_fused_inj  their ABFT, fault (upstream)   }  vs each other
# clean-overhead  = (online vs baseline)        & (fused      vs cublas)
# fault-overhead  = (online_inj vs baseline)    & (fused_inj  vs cublas)
# Our fault model is a single-bit flip (--inject swifi); theirs is their
# hardcoded +10000.0 — both are one corrupted output element, so the ABFT
# detect/correct work (hence the overhead) is comparable.
run_shape() {
    local regime="$1" M="$2" K="$3" N="$4"
    local smax=$(( M>N ? (M>K?M:K) : (N>K?N:K) ))
    local WU="${WARMUPS}"
    echo "### ${regime}  ${M}x${K}x${N}  (Smax=${smax})  samples=${OUR_SAMPLES}  warmups=${WU}"

    # ----------------------------------------------------------
    # OURS — one invocation per phase, INTERNAL --samples loop
    # ----------------------------------------------------------
    # Fixed (A,B): warm up ${WU} trials to boost clocks, then time
    # OUR_SAMPLES back-to-back cuBLAS GEMMs at steady clocks (NO reseed —
    # GEMM timing is operand-value-independent).  The reported baseline_/
    # protected_ median + ms columns are over those timed samples.
    rm -f abft_metrics_cmp.csv
    echo "  [stage] calling run_ours(none) for ${M}x${K}x${N}"
    set +e
    run_ours ${M} ${K} ${N} --inject none --threshold ${OUR_THRESHOLD} \
         --grid 1 1 --frags-per-rank ${OUR_F} ${OUR_ENC_FLAG} \
         --samples ${OUR_SAMPLES} --repeats ${OUR_R} \
         --warmups ${WU} --csv abft_metrics_cmp.csv > /dev/null
    local _ru_rc=$?
    set -e
    echo "  [stage] run_ours(none) returned rc=${_ru_rc}"
    if (( _ru_rc != 0 )); then
        echo "  WARNING: ours(none) launch failed ${M}x${K}x${N} (launch log tail):"
        sed 's/^/    | /' _ours_launch.log 2>/dev/null | tail -25
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

    # --- ours, WITH a fault (ABFT detect+localize+correct active) ---
    rm -f abft_metrics_inj.csv
    if run_ours ${M} ${K} ${N} --inject add --threshold ${OUR_THRESHOLD} \
         --grid 1 1 --frags-per-rank ${OUR_F} ${OUR_ENC_FLAG} \
         --samples ${OUR_SAMPLES} --repeats ${OUR_R} \
         --warmups ${WU} --csv abft_metrics_inj.csv > /dev/null; then
        local oi_g oi_t
        if read _ _ oi_g oi_t < <(last_protected abft_metrics_inj.csv 2>&1); then
            echo "${regime},${M},${K},${N},ours_online_inj,${oi_g},${oi_t}" >> "${OUT_CSV}"
        else
            echo "  WARNING: ours(inj) ${M}x${K}x${N}: empty/bad CSV"
        fi
    else
        echo "  WARNING: ours(inj) launch failed ${M}x${K}x${N}"
    fi

    # ----------------------------------------------------------
    # THEIRS — external THEIRS_SAMPLES-call loop
    # ----------------------------------------------------------
    # Their binary cannot iterate internally; each call generates A,B once
    # (stock srand(10)) and averages num_tests=5 timed iters with
    # FT_WARMUP=5 warmups.  We average their reported GFLOPS over the
    # THEIRS_SAMPLES independent calls (run-to-run clock jitter) — the
    # value-independence of GEMM timing means the fixed (A,B) is fine.
    # Pick the size/shape-appropriate kernel as their codegen would.
    # Their dispatch uses NON-CONTIGUOUS kernel_number values — see
    # sgemm.cu line 240:
    #     int list[14] = {0,1,2,3,4,5,6,10,11,12,13,14,15,16};
    # Real mapping:
    #   abft_baseline      = 10
    #   abft_kernel_small  = 11   (16x16  output tile)
    #   abft_kernel_medium = 12   (32x32  output tile)
    #   abft_kernel_large  = 13   (64x64  output tile)
    #   abft_kernel_tall   = 14   (128x32 output tile, for M >> N)
    #   abft_kernel_wide   = 15   (32x128 output tile, for N >> M)
    #   abft_kernel_huge   = 16   (128x128 output tile)
    #
    # Dispatch criterion is the OUTPUT-tile shape (driven by M,N), not
    # max(M,N,K).  K only affects flops per tile, not tile geometry.
    # tall/wide only apply when output is strongly rectangular.
    local kidx kname
    local out_max=$(( M>N ? M : N ))
    if   (( M >= 2*N )); then kidx=14; kname="abft_kernel_tall"
    elif (( N >= 2*M )); then kidx=15; kname="abft_kernel_wide"
    elif (( out_max <= 128 )); then kidx=11; kname="abft_kernel_small"
    elif (( out_max <= 256 )); then kidx=12; kname="abft_kernel_medium"
    elif (( out_max <= 512 )); then kidx=13; kname="abft_kernel_large"
    else                            kidx=16; kname="abft_kernel_huge"
    fi
    local tc_g="" tfc_g="" tfi_g=""
    local _tc_acc="" _tfc_acc="" _tfi_acc=""
    local s
    for ((s=0; s<THEIRS_SAMPLES; s++)); do
        FT_M=${M} FT_N=${N} FT_K=${K} \
            "${THEIRS_CLEAN}" ${smax} ${smax} 1 0 0          > _tc.log  2>&1 || true
        FT_M=${M} FT_N=${N} FT_K=${K} \
            "${THEIRS_CLEAN}" ${smax} ${smax} 1 ${kidx} ${kidx} > _tfc.log 2>&1 || true
        FT_M=${M} FT_N=${N} FT_K=${K} \
            "${THEIRS_INJ}"   ${smax} ${smax} 1 ${kidx} ${kidx} > _tfi.log 2>&1 || true
        local v
        v=$(their_gflops  _tc.log  cublas);       [[ -n "${v}" ]] && _tc_acc+="${v}"$'\n'
        v=$(their_gflops _tfc.log "${kname}");    [[ -n "${v}" ]] && _tfc_acc+="${v}"$'\n'
        v=$(their_gflops _tfi.log "${kname}");    [[ -n "${v}" ]] && _tfi_acc+="${v}"$'\n'
    done
    tc_g=$(printf '%s' "${_tc_acc}"  | mean_lines)
    tfc_g=$(printf '%s' "${_tfc_acc}" | mean_lines)
    tfi_g=$(printf '%s' "${_tfi_acc}" | mean_lines)
    local tc_t tfc_t tfi_t
    # Convert each non-empty GFLOPS to ms and emit a CSV row.  Use plain
    # if/fi so set -e does NOT exit on the harmless `[[ -n "" ]]` failure
    # when a kernel produced no parseable output (`abft_kernel_*` may be
    # legitimately missing for some shapes — we just skip those rows).
    tc_t=""; tfc_t=""; tfi_t=""
    if [[ -n "${tc_g}"  ]]; then tc_t=$(gflops_to_ms  "${tc_g}"  ${M} ${K} ${N}); fi
    if [[ -n "${tfc_g}" ]]; then tfc_t=$(gflops_to_ms "${tfc_g}" ${M} ${K} ${N}); fi
    if [[ -n "${tfi_g}" ]]; then tfi_t=$(gflops_to_ms "${tfi_g}" ${M} ${K} ${N}); fi
    if [[ -n "${tc_g}"  ]]; then echo "${regime},${M},${K},${N},theirs_cublas,${tc_g},${tc_t}"       >> "${OUT_CSV}"; fi
    if [[ -n "${tfc_g}" ]]; then echo "${regime},${M},${K},${N},theirs_fused,${tfc_g},${tfc_t}"     >> "${OUT_CSV}"; fi
    if [[ -n "${tfi_g}" ]]; then echo "${regime},${M},${K},${N},theirs_fused_inj,${tfi_g},${tfi_t}" >> "${OUT_CSV}"; fi
    # Diagnostic: if EITHER fused kernel produced no parseable GFLOPS,
    # dump the first 30 lines of theirs' log so we can see WHAT was
    # actually printed (kernel skipped due to shape constraint?
    # different table format? compile mismatch?).  Only printed once per
    # shape that fails — we don't want to flood the log.
    # Dump the theirs logs once per kname (first failing shape only) so
    # we can see what the FT kernel actually printed.  The empty-row NOTE
    # still prints for every failing shape so we know which ones lost data.
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

for reg in ${REGIMES}; do
  case "${reg}" in
    small_sq) for s in "${SMALL_SQ[@]}"; do run_shape small_sq $s; done ;;
    big_sq)   for s in "${BIG_SQ[@]}";   do run_shape big_sq   $s; done ;;
    small_ns) for s in "${SMALL_NS[@]}"; do run_shape small_ns $s; done ;;
    big_ns)   for s in "${BIG_NS[@]}";   do run_shape big_ns   $s; done ;;
    *) echo "WARNING: unknown regime '${reg}' (skipped)" ;;
  esac
done

mkdir -p "${PLOT_DIR}"
python3 scripts/plot_comparison.py --csv "${OUT_CSV}" \
        --outdir "${PLOT_DIR}" \
    || echo "(plots skipped: pip install pandas matplotlib)"

echo "=== Done ==="
echo "  CSV : ${OUT_CSV}"
echo "  plot: ${PLOT_DIR}/"
ls "${PLOT_DIR}/" | sed 's/^/    /'
