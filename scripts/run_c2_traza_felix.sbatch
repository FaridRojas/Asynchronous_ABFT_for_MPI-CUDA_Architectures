#!/bin/bash
#SBATCH --job-name=abft_c2traza
#SBATCH --partition=gpu_titan
#SBATCH --nodelist=felix
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --output=scripts/abft_c2traza_%j.out
#SBATCH --error=scripts/abft_c2traza_%j.err

# Where does the exposed cost of the large regime come from?  The thesis
# director's review (C2) asks whether the multiplications slow down when the
# verification kernels share the device with them.  Every protected run times,
# in each trial, the unprotected pass and then the protected pass on the same
# operands, so one nvprof trace per rank contains both: the first F
# multiplications of a trial run alone, the next F run next to the
# verification stream.  The trace also gives the tail after the last
# multiplication and the head spent before the first one.  Outputs to c2/.

set -euo pipefail
module purge
module load cuda/11.8
module load mpi/openmpi-gcc11/4.1.6
MPI_PREFIX="$(dirname "$(dirname "$(which mpicc)")")"
nvcc -O3 -std=c++17 -ccbin g++ -I"${MPI_PREFIX}/include" src/main.cu -o abft_gemm \
     -L"${MPI_PREFIX}/lib" -lmpi -lcublas
date -Is
nvidia-smi --query-gpu=index,name,driver_version --format=csv

mkdir -p c2
rm -f c2/c2_metrics.csv
for S in 5120 8192 10240; do
  echo "=== C2 trace ${S}^3"
  srun --mpi=pmix nvprof --concurrent-kernels on --print-gpu-trace --csv \
       --normalized-time-unit us \
       --log-file "c2/traza_${S}_rank%q{SLURM_PROCID}.csv" \
       ./abft_gemm ${S} ${S} ${S} --inject none --threshold 1.0 \
       --encoding-mode overlap --frags-per-rank 8 --samples 5 --repeats 1 \
       --warmups 2 --csv c2/c2_metrics.csv
done
echo "=== All runs complete ==="
