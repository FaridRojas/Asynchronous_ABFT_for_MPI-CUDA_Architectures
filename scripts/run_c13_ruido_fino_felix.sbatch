#!/bin/bash
#SBATCH --job-name=abft_c13fino
#SBATCH --partition=gpu_titan
#SBATCH --nodelist=felix
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --output=scripts/abft_c13fino_%j.out
#SBATCH --error=scripts/abft_c13fino_%j.err

# Follow-up to run_c1_c13_felix.sbatch.  nvprof showed cuBLAS switching from
# sgemm_32x32x32_NN_vec to sgemm_128x128x8_NN_vec between 2560 and 2816.  If
# that switch causes the jump of the noise floor between 2048 and 4096, the
# jump must happen exactly there and not spread over the interval.  Same
# calibration flags as run_calibrate_felix.sbatch, every 256 in [2048, 4096].

set -euo pipefail
module purge
module load cuda/11.8
module load mpi/openmpi-gcc11/4.1.6
MPI_PREFIX="$(dirname "$(dirname "$(which mpicc)")")"
nvcc -O3 -std=c++17 -ccbin g++ -I"${MPI_PREFIX}/include" src/main.cu -o abft_gemm \
     -L"${MPI_PREFIX}/lib" -lmpi -lcublas
date -Is
nvidia-smi --query-gpu=index,name,driver_version --format=csv

mkdir -p c13_fino
rm -f c13_fino/abft_calibration.csv
for S in $(seq 2048 256 4096); do
  echo "=== calibration ${S}^3"
  srun --mpi=pmix ./abft_gemm ${S} ${S} ${S} --calibrate --frags-per-rank 8 \
       --repeats 20 --calibration-safety 3 \
       --csv c13_fino/abft_calibration.csv --calib-diffs "c13_fino/abft_calib_diffs_${S}.csv"
done
echo "=== All runs complete ==="
