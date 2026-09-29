#!/bin/bash
#SBATCH --job-name=abft_mtrace
#SBATCH --partition=gpu_titan
#SBATCH --nodelist=felix
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --output=scripts/abft_mtrace_%j.out
#SBATCH --error=scripts/abft_mtrace_%j.err

# ===================================================================
# Traza de ejecución de una secuencia de multiplicaciones protegidas
# independientes, emitidas sin espera intermedia sobre un mismo par de
# flujos.
#
# La captura registra el instante de inicio y la duración de cada
# núcleo, junto con el flujo que lo ejecuta, de modo que la coincidencia
# temporal entre la comprobación de una operación y la multiplicación de
# la siguiente pueda leerse directamente.
#
# Se emplea nvprof porque la salida tabular de --print-gpu-trace da
# exactamente los cuatro campos que la figura necesita (inicio,
# duración, flujo y nombre) y se convierte en una figura del libro sin
# depender de una captura de pantalla de la interfaz. Nsight Systems se
# ejecuta además, por si conviene inspeccionar la traza de forma
# interactiva.
#
# Advertencia de método: el perfilador altera los tiempos absolutos, de
# modo que esta captura sirve para mostrar QUE hay solapamiento y no
# para cuantificarlo.
# ===================================================================

set -euo pipefail

module purge
module load cuda/11.8
module load mpi/openmpi-gcc11/4.1.6

MPICC_PATH="$(which mpicc)"
MPI_PREFIX="$(dirname "$(dirname "$MPICC_PATH")")"

OUTDIR="scripts/multitrace"
mkdir -p "${OUTDIR}"

echo "=== Compilando bench/multi_gemm ==="
nvcc -O3 -std=c++17 -ccbin g++ \
  -I"${MPI_PREFIX}/include" \
  bench/multi_gemm.cu -o multi_gemm \
  -L"${MPI_PREFIX}/lib" -lmpi -lcublas

NVPROF="$(which nvprof)"
NSYS="$(which nsys || true)"
OPS=${OPS:-4}
FRAGS=${FRAGS:-1}

# Dos tamaños: en 1024 la verificación es proporcionalmente grande y el
# solapamiento se aprecia sin ampliar; en 2048 la multiplicación domina
# y la figura se parece más al régimen de interés práctico.
for SZ in 1024 2048; do
  LAB="${SZ}_encadenado"
  echo "### traza ${LAB}"
  # Una ronda descartada y una registrada: la segunda es la que se
  # dibuja, ya con la biblioteca y los relojes en régimen.
  srun --gres=gpu:1 --mpi=pmix "${NVPROF}" --print-gpu-trace --csv \
      --log-file "${OUTDIR}/traza_${LAB}.csv" \
      ./multi_gemm ${SZ} ${SZ} ${SZ} --ops ${OPS} --frags ${FRAGS} \
      --rounds 1 --warmup-rounds 1 \
      --csv "${OUTDIR}/scratch.csv" --tag "${LAB}" \
      > "${OUTDIR}/traza_${LAB}.log" 2>&1 \
    || echo "  (nvprof rc!=0; ver traza_${LAB}.log)"

  if [[ -n "${NSYS}" ]]; then
    srun --gres=gpu:1 --mpi=pmix "${NSYS}" profile -t cuda,nvtx -f true \
        -o "${OUTDIR}/nsys_${LAB}" \
        ./multi_gemm ${SZ} ${SZ} ${SZ} --ops ${OPS} --frags ${FRAGS} \
        --rounds 1 --warmup-rounds 1 \
        --csv "${OUTDIR}/scratch.csv" --tag "${LAB}" \
        > "${OUTDIR}/nsys_${LAB}.log" 2>&1 \
      || echo "  (nsys rc!=0; ver nsys_${LAB}.log)"
  fi
done

echo "=== Listo ==="
ls -la "${OUTDIR}" | head -30
