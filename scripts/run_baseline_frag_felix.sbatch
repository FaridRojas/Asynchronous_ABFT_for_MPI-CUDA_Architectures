#!/bin/bash
#SBATCH --job-name=abft_basefrag
#SBATCH --partition=gpu_titan
#SBATCH --nodelist=felix
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --output=scripts/abft_basefrag_%j.out
#SBATCH --error=scripts/abft_basefrag_%j.err

# ===================================================================
# Costo de fragmentar la multiplicación, medido SIN protección alguna.
#
# Todas las mediciones del libro fijan F = 8 sin mostrar lo que esa
# elección cuesta por sí sola.  Fragmentar el bloque local reparte la
# multiplicación en F llamadas más estrechas a cuBLAS, y la biblioteca
# escoge su núcleo según la geometría del bloque de salida, de modo que
# el ancho del fragmento condiciona su rendimiento antes de que exista
# ninguna suma de comprobación.  Esa penalización es la que este barrido
# aísla: solo la fase --baseline, sin verificación de ningún tipo.
#
# El protocolo replica el del barrido de sobrecosto del libro para que
# las cifras sean directamente comparables con las de sus figuras:
# dos dispositivos (la malla queda en Pr=1, Pc=2), veinte muestras
# cronometradas, dos descartadas como calentamiento, y una pareja fija
# de operandos por ejecución.  No se regenera entre muestras, porque el
# tiempo de un GEMM denso no depende de los valores y regenerar dejaría
# el acelerador ocioso el tiempo suficiente para que bajen los relojes.
#
# Los tamaños son los cuarenta que recorren las figuras del libro, en
# los dos regímenes de forma: cuadrado, y con la dimensión de
# contracción fijada en 1024.
#
# Nota sobre los tamaños pequeños: por debajo de max(M,K,N) = 1024 la
# política consciente del tamaño colapsa F a uno, así que las columnas
# de esas filas repiten el mismo valor.  Es el comportamiento de la
# implementación, no un defecto del barrido.
# ===================================================================

set -euo pipefail

module purge
module load cuda/11.8
module load mpi/openmpi-gcc11/4.1.6

MPICC_PATH="$(which mpicc)"
MPI_PREFIX="$(dirname "$(dirname "$MPICC_PATH")")"

echo "=== Compilando ==="
nvcc -O3 -std=c++17 -ccbin g++ \
  -I"${MPI_PREFIX}/include" \
  src/main.cu -o abft_gemm \
  -L"${MPI_PREFIX}/lib" -lmpi -lcublas

# Rejilla idéntica a la de run_abft_felix.sbatch
SWEEP_N=${SWEEP_N:-40}
K_NS=${K_NS:-1024}
FRAGS=( ${FRAGS:-1 2 4 8 16 32} )
SAMPLES=${SAMPLES:-20}
WARMUPS=${WARMUPS:-2}
R=${R:-1}
OUT=${OUT:-abft_baseline_frag.csv}

sweep_grid() { local i; for ((i=0;i<SWEEP_N;i++)); do echo $(( 256*(i+1) )); done; }
mapfile -t SIZES < <( sweep_grid )

COMMON=(--baseline --samples ${SAMPLES} --repeats ${R} --warmups ${WARMUPS}
        --csv "${OUT}")

rm -f "${OUT}"

run() {
  local label="$1"; shift
  echo "--- ${label} ---"
  srun --mpi=pmix ./abft_gemm "$@"
}

echo "############################################"
echo "###  CUADRADAS  ${#SIZES[@]} tamaños x ${#FRAGS[@]} valores de F"
echo "############################################"
for S in "${SIZES[@]}"; do
  for F in "${FRAGS[@]}"; do
    run "sq ${S}^3 F=${F}" ${S} ${S} ${S} --frags-per-rank ${F} "${COMMON[@]}"
  done
done

echo "############################################"
echo "###  NO CUADRADAS  K=${K_NS} fijo"
echo "############################################"
for S in "${SIZES[@]}"; do
  for F in "${FRAGS[@]}"; do
    run "ns ${S}x${K_NS}x${S} F=${F}" ${S} ${K_NS} ${S} --frags-per-rank ${F} "${COMMON[@]}"
  done
done

echo "=== Listo ==="
echo "CSV: $(pwd)/${OUT}"
