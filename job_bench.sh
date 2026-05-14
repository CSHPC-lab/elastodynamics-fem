#!/bin/bash -l

#SBATCH --nodes=2
#SBATCH --partition=40g
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --time=00:05:00
#SBATCH -o ./cpp_log/bench.%j.out
#SBATCH -e ./cpp_log/bench.%j.err

. /etc/profile.d/modules.sh
module load nvhpc/25.1

export UCX_TLS=rc,rc_mlx5,shm,self
export UCX_WARN_UNUSED_ENV_VARS=n
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# コンパイル
mpicxx -O2 -o bench_allreduce allreduce_bench.cpp

# 実行（GPU 不要なので CUDA_VISIBLE_DEVICES 不要）
mpirun -np ${SLURM_NTASKS} \
    --map-by ppr:4:node \
    --bind-to numa \
    ./bench_allreduce

echo "Done at $(date)"
