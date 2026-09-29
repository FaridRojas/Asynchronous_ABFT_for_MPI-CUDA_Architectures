#!/bin/bash
#SBATCH --job-name=abft_prof_pacca
#SBATCH --partition=GPU
#SBATCH --nodelist=paccaA100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --output=scripts/abft_profile_pacca_%j.out
#SBATCH --error=scripts/abft_profile_pacca_%j.err

# =====================================================================
# Profiling on PACCA A100.  Two complementary tools, two distinct outputs:
#
#   Nsight Compute (ncu)   -> MEASURED FP32 throughput = real GFLOPS
#                             (vs the theoretical 2*M*N*K/walltime).  This
#                             is the AUTHORITATIVE source of real-GFLOPS on
#                             pacca — admin perf-counter permissions are
#                             now fixed, so there is no fallback path.
#   Nsight Systems (nsys)  -> TIMELINE for visualizing the two-stream
#                             pipeline (open in nsys-ui).  Timeline only —
#                             nsys is NEVER used as a GFLOPS source.
#
# One representative shape per regime, two profiles each:
#   <regime>_theirs_fused.ncu-rep  / .nsys-rep
#   <regime>_ours_online.ncu-rep   / .nsys-rep
#
# Launching:
#   theirs (non-MPI): env <profiler> ./ft_sgemm ...
#   ours   (MPI):     env <profiler> mpirun -np 1 ./abft_gemm ...
#     (pacca OpenMPI singleton fails inside Slurm; only mpirun -np 1 works.
#      ncu --target-processes all / nsys descendant-tracing catch the child)
# =====================================================================
set -euo pipefail

# --- pacca toolchain: CUDA is inside nvhpc/23.1 (loaded by default). Do NOT
#     'module purge'.  Load extras on top; tolerate missing names. ---
module load devtools/nvidia/hpc_sdk/nvhpc/23.1            2>/dev/null || true
module load openmpi4/4.1.6        2>/dev/null \
  || module load devtools/mpi/openmpi4-gnu12 2>/dev/null || true
module load cmake/3.24.2          2>/dev/null || module load cmake 2>/dev/null || true
module load pmix/4.2.9            2>/dev/null || true
module load prun                  2>/dev/null || true
module load nsight-systems        2>/dev/null \
  || module load nsight-systems/2026.1.1 2>/dev/null || true

# Locate nvcc (bundled in the HPC SDK; only nvc++ is on PATH by default).
if ! command -v nvcc >/dev/null 2>&1; then
    CAND=""
    if command -v nvc++ >/dev/null 2>&1; then
        VERDIR="$(dirname "$(dirname "$(dirname "$(readlink -f "$(command -v nvc++)")")")")"
        for c in "$VERDIR"/cuda/bin/nvcc "$VERDIR"/cuda/*/bin/nvcc; do
            [[ -x "$c" ]] && { CAND="$(dirname "$c")"; break; }
        done
    fi
    [[ -z "$CAND" ]] && for v in "${NVHPC_ROOT:-}" "${NVHPC:-}" "${CUDA_HOME:-}"; do
        [[ -z "$v" ]] && continue
        for c in "$v"/cuda/bin/nvcc "$v"/cuda/*/bin/nvcc "$v"/bin/nvcc; do
            [[ -x "$c" ]] && { CAND="$(dirname "$c")"; break; }
        done; [[ -n "$CAND" ]] && break
    done
    [[ -z "$CAND" ]] && for c in \
        /opt/ohpc/pub/*/nvhpc*/Linux_x86_64/*/cuda/bin/nvcc \
        /opt/ohpc/pub/*/nvhpc*/Linux_x86_64/*/cuda/*/bin/nvcc; do
        [[ -x "$c" ]] && { CAND="$(dirname "$c")"; break; }
    done
    [[ -n "$CAND" ]] && export PATH="$CAND:$PATH"
fi
command -v nvcc >/dev/null 2>&1 || {
    echo "ERROR: nvcc not found.  CUDA is bundled in nvhpc/23.1; diagnose:"
    echo "  which nvc++ ; ls \$(dirname \$(dirname \$(which nvc++)))/../cuda*/bin"
    exit 1; }
echo "Using nvcc: $(command -v nvcc)"

# CUDA runtime libs for the cmake-built ft_sgemm: FIND libcudart.so.12
# (NVHPC nests it deeply; -quit = first match, fast).
CUDA_BIN="$(dirname "$(readlink -f "$(command -v nvcc)")")"
CUDA_ROOT="$(dirname "${CUDA_BIN}")"
NVHPC_TOP="$(dirname "$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")")"
CUDA_SEARCH_ROOTS=( "${CUDA_ROOT}" "${CUDA_ROOT%/*}" "${CUDA_HOME:-}" \
    "${NVHPC_ROOT:-}" "${NVHPC_TOP}" \
    /opt/ohpc/pub/devtools/nvidia/hpc_sdk /usr/local/cuda )
find_libdir() {
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

# --- Nsight discovery: we run BOTH tools ------------------------------
#   ncu  (Nsight Compute) -> MEASURED FP throughput = real GFLOPS
#                            (vs the theoretical 2*M*N*K/walltime)
#   nsys (Nsight Systems) -> timeline to inspect both pipelines
# sbatch jobs do NOT source ~/.bashrc, so prepend both install dirs here.
# Override with NCU_DIR / NSYS_DIR.
NCU_DIR="${NCU_DIR:-/home/latorresn/fcrojasv/nsight_compute}"
NSYS_DIR="${NSYS_DIR:-/home/latorresn/fcrojasv/nsight_systems}"
for d in "${NCU_DIR}" "${NSYS_DIR}"; do
    [[ -d "$d" ]] && export PATH="$d:$d/bin:${PATH}"
done

NCU="$(command -v ncu 2>/dev/null || command -v nv-nsight-cu-cli 2>/dev/null || true)"
NSYS="$(command -v nsys 2>/dev/null || true)"
if [[ -z "${NCU}" ]]; then
    for c in "${NCU_DIR}"/ncu "${NCU_DIR}"/bin/ncu \
             /opt/ohpc/pub/*/nvhpc*/Linux_x86_64/*/profilers/Nsight_Compute/ncu; do
        [[ -x "$c" ]] && { NCU="$c"; break; }
    done
fi
if [[ -z "${NSYS}" ]]; then
    for c in "${NSYS_DIR}"/nsys "${NSYS_DIR}"/bin/nsys \
             /opt/ohpc/pub/*/nvhpc*/Linux_x86_64/*/profilers/Nsight_Systems/bin/nsys; do
        [[ -x "$c" ]] && { NSYS="$c"; break; }
    done
fi
[[ -n "${NCU}" ]] || {
    echo "ERROR: ncu not found.  ncu is REQUIRED for real-GFLOPS on pacca"
    echo "       (no nsys fallback).  Set NCU_DIR to its install directory."
    exit 1; }
echo "Nsight Compute -> ${NCU}"; "${NCU}" --version | head -1 || true
[[ -n "${NSYS}" ]] && { echo "Nsight Systems -> ${NSYS}"; "${NSYS}" --version | head -1 || true; } \
                   || echo "WARNING: nsys not found -> no timeline this run."

# Nsight Compute FP32 instruction + duration metrics (roofline-standard,
# stable since NCU 2019).  real_flops = fadd + fmul + 2*ffma .
NCU_METRICS="sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
gpu__time_duration.sum"

MPICC="$(command -v mpicc || true)"
MPI_PREFIX="$([[ -n "${MPICC}" ]] && dirname "$(dirname "$(readlink -f "${MPICC}")")" || echo "${MPI_PREFIX:-/usr}")"
# Flags from mpicc --showme (NVHPC SDK hijacks mpicc -> SDK's HPCX OpenMPI;
# OPAL_PREFIX/mpirun/LD must match THAT, = dirname of the --showme libdir).
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
MPI_RT_LIBD="$(echo ${MPI_LIBD} | awk '{print $1}')"
MPI_RT_PREFIX="$(dirname "${MPI_RT_LIBD}")"
export OPAL_PREFIX="${MPI_RT_PREFIX}"   # fixes opal_shmem_base_open
export LD_LIBRARY_PATH="${MPI_LIBD// /:}:${LD_LIBRARY_PATH:-}"
MPIRUN="${MPI_RT_PREFIX}/bin/mpirun"
[[ -x "${MPIRUN}" ]] || MPIRUN="${MPI_PREFIX}/bin/mpirun"
[[ -x "${MPIRUN}" ]] || MPIRUN="$(command -v mpirun || echo mpirun)"
# Launcher for the profiled ours run.  Default to the linked OpenMPI's
# mpirun (proven to work).  ncu --target-processes all / nsys descendant
# tracing capture the abft_gemm child.  Override with OURS_LAUNCH.
OURS_LAUNCH="${OURS_LAUNCH:-${MPIRUN} -np 1}"
echo "MPI (linked,RT): ${MPI_RT_PREFIX}"
echo "OPAL_PREFIX    : ${OPAL_PREFIX}"
echo "MPIRUN         : ${MPIRUN}"
echo "OURS_LAUNCH    : ${OURS_LAUNCH}"

# OpenMPI runtime env (mirrors the proven interactive recipe: OPAL_PREFIX
# + writable TMPDIR only; no forced OMPI_MCA_shmem).
export TMPDIR="${TMPDIR:-/tmp}"
[[ -d "${TMPDIR}" && -w "${TMPDIR}" ]] || export TMPDIR=/tmp
mkdir -p "${TMPDIR}" 2>/dev/null || true

# regime : "M K N" (one representative shape each)
REGIMES=( "small_sq:1024 1024 1024"
          "big_sq:8192 8192 8192"
          "small_ns:1024 512 384"
          "big_ns:8192 4096 2048" )
OUR_THRESHOLD="1.0"; OUR_F=8; OUR_R=1
# Encoding placement for ours (see --encoding-mode in abft_gemm --help):
#   overlap (default) = colSumA + expectedRow enqueued on verify_stream inside
#   the timed window, overlapping iteration 0's GEMMs — THIS is what the nsys
#   timeline should show; amortized = encoding completes before the loop.
ENCODING=${ENCODING:-overlap}
OUR_ENC_FLAG="--encoding-mode ${ENCODING}"
THEIR_KERNEL=16; CUDA_ARCH="80"
THEIRS_DIR="Fault-Tolerant-SGEMM-on-NVIDIA-GPUs"
OUTDIR="scripts/profile"; mkdir -p "${OUTDIR}"

# ---------- provision + patch + build their project ----------
[[ -d "${THEIRS_DIR}" ]] || { echo "ERROR: ${THEIRS_DIR}/ missing"; exit 1; }
THEIRS_COMMON="${THEIRS_DIR}/cuda-samples/Common"
if [[ ! -f "${THEIRS_COMMON}/helper_functions.h" ]]; then
    FOUND_INC=""
    for p in "${CUDA_HOME:-}/samples/common/inc" "/usr/local/cuda/samples/common/inc"; do
        [[ -n "$p" && -f "$p/helper_functions.h" ]] && { FOUND_INC="$p"; break; }
    done
    if [[ -n "${FOUND_INC}" ]]; then
        mkdir -p "${THEIRS_COMMON}"; cp -r "${FOUND_INC}/." "${THEIRS_COMMON}/"
    else
        rm -rf "${THEIRS_DIR}/cuda-samples"
        git clone --depth 1 https://github.com/NVIDIA/cuda-samples.git "${THEIRS_DIR}/cuda-samples"
    fi
    [[ -f "${THEIRS_COMMON}/helper_functions.h" ]] || { echo "ERROR: no cuda-samples"; exit 1; }
fi
sed -i -E "s/set\\(CMAKE_CUDA_ARCHITECTURES[[:space:]]+[0-9]+\\)/set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})/" \
    "${THEIRS_DIR}/CMakeLists.txt"
SGEMM_SRC="${THEIRS_DIR}/kernel/ft_sgemm/sgemm.cu"
if ! grep -q "FT_M override" "${SGEMM_SRC}"; then
    sed -i 's@N = K = M = max_size;@N = K = M = max_size; /*FT_M override*/ { char*_em=getenv("FT_M"); if(_em)M=atoi(_em); char*_en=getenv("FT_N"); if(_en)N=atoi(_en); char*_ek=getenv("FT_K"); if(_ek)K=atoi(_ek); }@' \
        "${SGEMM_SRC}"
fi
echo "=== Building their ft_sgemm ==="
rm -rf "${THEIRS_DIR}/build"; mkdir -p "${THEIRS_DIR}/build"
( cd "${THEIRS_DIR}/build" && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j16 )
THEIRS_BIN="$(find ${THEIRS_DIR}/build -name ft_sgemm -type f -executable | head -1)"
[[ -n "${THEIRS_BIN}" ]] || { echo "ERROR: ft_sgemm not found"; exit 1; }

echo "=== Building our abft_gemm ==="
nvcc -O3 -std=c++17 -arch=sm_${CUDA_ARCH} -ccbin g++ ${MPI_INC_FLAGS} \
  src/main.cu -o abft_gemm ${MPI_LNK_FLAGS} -lcublas

profile_one() {   # global ${LAUNCH[@]}, ${NCU}, ${NSYS};  $1=label  $2..=cmd
    local label="$1"; shift

    # --- Nsight Compute: measured FP throughput (-> real GFLOPS) ---------
    # Targeted metric list (one replay pass, fast).  --target-processes all
    # is the ncu analog of nvprof --profile-child-processes (needed because
    # mpirun is the launcher for ours).  --launch-count caps profiled
    # kernels so big shapes don't run for hours.  --csv --log-file writes
    # the metric table plot_real_gflops.py parses; -o keeps the .ncu-rep
    # for ncu-ui.
    echo "### ncu  ${label}  (launcher: ${LAUNCH[*]})  cmd: $*"
    "${LAUNCH[@]}" "${NCU}" --target-processes all \
        --metrics "${NCU_METRICS}" --launch-count 24 \
        --csv --log-file "${OUTDIR}/${label}_metrics.csv" \
        -f -o "${OUTDIR}/${label}" \
        "$@" > "${OUTDIR}/${label}_ncu_run.log" 2>&1 \
        || echo "  (ncu rc!=0; see ${label}_ncu_run.log)"

    # --- Nsight Systems: TIMELINE ONLY (no GFLOPS fallback) -------------
    if [[ -n "${NSYS}" ]]; then
        echo "### nsys ${label}  (launcher: ${LAUNCH[*]})  cmd: $*"
        "${LAUNCH[@]}" "${NSYS}" profile --force-overwrite true \
            -o "${OUTDIR}/${label}" -t cuda,nvtx,osrt --stats=false \
            "$@" > "${OUTDIR}/${label}_nsys_run.log" 2>&1 \
            || echo "  (nsys rc!=0; see ${label}_nsys_run.log)"
    fi
}

for entry in "${REGIMES[@]}"; do
    regime="${entry%%:*}"; dims="${entry#*:}"
    set -- ${dims}; M=$1; K=$2; N=$3
    smax=$(( M>N ? (M>K?M:K) : (N>K?N:K) ))
    echo "############ REGIME ${regime}  ${M}x${K}x${N} ############"
    LAUNCH=(env FT_M=${M} FT_N=${N} FT_K=${K})
    profile_one "${regime}_theirs_fused" \
        "${THEIRS_BIN}" ${smax} ${smax} 1 ${THEIR_KERNEL} ${THEIR_KERNEL}
    # On pacca, OpenMPI singleton fails inside a Slurm allocation (OMPI not
    # built with Slurm PMI).  Only `mpirun -np 1` initializes MPI.  The
    # profiler runs mpirun, mpirun forks abft_gemm; ncu --target-processes
    # all and nsys descendant-tracing both capture the abft_gemm child.
    LAUNCH=(env)
    profile_one "${regime}_ours_online" \
        ${OURS_LAUNCH} ./abft_gemm ${M} ${K} ${N} --inject none \
            --threshold ${OUR_THRESHOLD} --grid 1 1 \
            --frags-per-rank ${OUR_F} ${OUR_ENC_FLAG} --repeats ${OUR_R}
done

# --- Real (measured) GFLOPS comparison plot, ncu ONLY -----------------
mkdir -p scripts/plots/profile
python3 scripts/plot_real_gflops.py \
    --indir  "${OUTDIR}" \
    --tool   ncu \
    --out    scripts/plots/profile/real_gflops_pacca.png \
    || echo "(real-GFLOPS plot skipped: pip install matplotlib)"

# =========================================================
# Real-GFLOPS SIZE SWEEP (ncu) — square sizes, for the GFLOPS-vs-size line
# plot (the measured analog of the theoretical gflops_vs_size figure).
# Unlike the cheap wall-clock sweep, ncu replays each kernel, so we keep the
# size list modest and cap --launch-count.  One ncu pass per (size, who),
# exported to a .ncu-rep then `ncu --import`-ed to a CSV table (the -o + CSV
# combo only writes a banner, so import is the reliable way to get metrics).
# =========================================================
PROFILE_SWEEP=${PROFILE_SWEEP:-1}
if [[ "${PROFILE_SWEEP}" == 1 ]]; then
  SWEEP_SIZES=( ${SWEEP_SIZES:-256 1024 2048 3072 4096 5120 6144 7168 8192 9216 10240} )
  SWEEP_LAUNCH=${SWEEP_LAUNCH:-16}     # profiled kernels per point (replay cost)
  SWEEP_DIR="${OUTDIR}/sweep"; mkdir -p "${SWEEP_DIR}"
  SWEEP_CSV="${OUTDIR}/real_gflops_sweep.csv"
  echo "M,K,N,who,real_gflops" > "${SWEEP_CSV}"

  # Extract real GFLOPS from an ncu metrics CSV via plot_real_gflops's parser.
  real_gflops() {   # $1 = metrics csv  $2 = kernel filter substr ("" = none)
    python3 - "$1" "$2" <<'PY'
import sys
sys.path.insert(0, "scripts")
from plot_real_gflops import parse_ncu_csv
g = parse_ncu_csv(sys.argv[1], kernel_substr=(sys.argv[2] or None))
print("" if g is None else f"{g:.3f}")
PY
  }

  # Profile -> .ncu-rep, then import to a CSV metric table.  Uses ${LAUNCH[@]}.
  ncu_sweep_csv() {   # $1 = report stem  $2 = out csv  $3.. = cmd
    local rep="$1" out="$2"; shift 2
    "${LAUNCH[@]}" "${NCU}" --target-processes all \
        --metrics "${NCU_METRICS}" --launch-count ${SWEEP_LAUNCH} \
        -f -o "${rep}" "$@" >/dev/null 2>&1 || true
    "${NCU}" --import "${rep}.ncu-rep" --csv > "${out}" 2>/dev/null || true
  }

  for S in "${SWEEP_SIZES[@]}"; do
    echo "### SWEEP ${S}^3"
    # theirs (fused ABFT); count only ft_sgemm_* (skip their cutlass ref GEMM)
    LAUNCH=(env FT_M=${S} FT_N=${S} FT_K=${S})
    ncu_sweep_csv "${SWEEP_DIR}/sweep_${S}_theirs" \
                  "${SWEEP_DIR}/sweep_${S}_theirs.csv" \
                  "${THEIRS_BIN}" ${S} ${S} 1 ${THEIR_KERNEL} ${THEIR_KERNEL}
    tg=$(real_gflops "${SWEEP_DIR}/sweep_${S}_theirs.csv" "ft_sgemm")
    [[ -n "${tg}" ]] && echo "${S},${S},${S},theirs_fused,${tg}" >> "${SWEEP_CSV}"
    # ours (online ABFT)
    LAUNCH=(env)
    ncu_sweep_csv "${SWEEP_DIR}/sweep_${S}_ours" \
                  "${SWEEP_DIR}/sweep_${S}_ours.csv" \
                  ${OURS_LAUNCH} ./abft_gemm ${S} ${S} ${S} --inject none \
                  --threshold ${OUR_THRESHOLD} --grid 1 1 \
                  --frags-per-rank ${OUR_F} ${OUR_ENC_FLAG} --repeats ${OUR_R}
    og=$(real_gflops "${SWEEP_DIR}/sweep_${S}_ours.csv" "")
    [[ -n "${og}" ]] && echo "${S},${S},${S},ours_online,${og}" >> "${SWEEP_CSV}"
  done

  echo "Real-GFLOPS sweep CSV: ${SWEEP_CSV}"
  python3 scripts/plot_real_gflops.py \
      --sweep-csv "${SWEEP_CSV}" \
      --out scripts/plots/profile/real_gflops_vs_size_pacca.png \
      || echo "(sweep plot skipped)"
fi

echo "=== Done. Artifacts in ${OUTDIR}/ ==="
ls -lh "${OUTDIR}/" | sed 's/^/    /'
echo
echo "Real (measured) GFLOPS bar chart: scripts/plots/profile/real_gflops_pacca.png"
echo "Nsight Compute per-kernel reports: ncu-ui ${OUTDIR}/<regime>_<who>.ncu-rep"
echo "Nsight Systems timelines (pipelines): nsys-ui ${OUTDIR}/<regime>_<who>.nsys-rep"
