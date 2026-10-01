#!/bin/bash
#SBATCH --job-name=abft_reloj
#SBATCH --partition=gpu_titan
#SBATCH --nodelist=felix
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --output=scripts/abft_reloj_%j.out
#SBATCH --error=scripts/abft_reloj_%j.err

# Why is the baseline measured inside a protected run slower than the
# standalone baseline run (thesis director's review, R1)?  The GPU clocks
# are not locked, so this repeats the campaign protocol on a few shapes while
# nvidia-smi logs SM clock, power and the active clock-limiting reasons every
# 100 ms.  Phase boundaries go to reloj/fases.csv.  Order per shape:
# standalone baseline, protected, standalone baseline, protected.

set -euo pipefail
module purge
module load cuda/11.8
module load mpi/openmpi-gcc11/4.1.6
MPI_PREFIX="$(dirname "$(dirname "$(which mpicc)")")"
nvcc -O3 -std=c++17 -ccbin g++ -I"${MPI_PREFIX}/include" src/main.cu -o abft_gemm \
     -L"${MPI_PREFIX}/lib" -lmpi -lcublas
nvidia-smi --query-gpu=index,name,driver_version,power.limit,clocks.max.sm --format=csv

mkdir -p reloj
rm -f reloj/*.csv
Q="timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active"
nvidia-smi --query-gpu=${Q} --format=csv,noheader -lms 100 > /dev/null 2>&1 & p=$!; sleep 1
if kill -0 $p 2>/dev/null; then kill $p; else Q="timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu"; fi
echo "campos: ${Q}"
nvidia-smi --query-gpu=${Q} --format=csv -lms 100 > reloj/reloj.csv &
SMI=$!
sleep 5

echo "fase,M,K,N,inicio,fin" > reloj/fases.csv
COMMON=(--frags-per-rank 8 --samples 20 --repeats 1 --warmups 2)
fase() {  # $1 = etiqueta, $2..$4 = M K N, resto = opciones
  local tag=$1 M=$2 K=$3 N=$4; shift 4
  local t0; t0=$(date +%s.%N)
  srun --mpi=pmix ./abft_gemm ${M} ${K} ${N} "$@" "${COMMON[@]}" --csv reloj/reloj_metrics.csv > /dev/null
  echo "${tag},${M},${K},${N},${t0},$(date +%s.%N)" >> reloj/fases.csv
  sleep 3
}
for shape in "5120 5120 5120" "10240 10240 10240" "10240 1024 10240"; do
  set -- ${shape}
  for rep in 1 2; do
    fase base "$1" "$2" "$3" --baseline
    fase prot "$1" "$2" "$3" --inject none --threshold 1.0 --encoding-mode overlap
  done
done
sleep 3
kill ${SMI}
echo "=== All runs complete ==="
