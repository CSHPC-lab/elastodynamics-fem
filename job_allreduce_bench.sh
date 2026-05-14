#!/bin/bash -l

#SBATCH --nodes=2
#SBATCH --partition=40g
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --time=00:20:00
#SBATCH -o ./cpp_log/allreduce_bench.%j.out
#SBATCH -e ./cpp_log/allreduce_bench.%j.err

. /etc/profile.d/modules.sh
module load nvhpc/25.1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DYNAMIC=false
export OMP_WAIT_POLICY=active
export CUDA_DEVICE_ORDER=PCI_BUS_ID

mpicxx -O3 -std=c++11 mpi_allreduce_bench.cpp -o mpi_allreduce_bench

# BENCH_ARGS can be overridden at submit time, e.g.
#   sbatch --export=ALL,BENCH_ARGS="--iters 1000 --warmup 100 --skew-us 500 --work-us 1000" job_allreduce_bench.sh
: "${BENCH_ARGS:=--iters 1000 --warmup 100 --skew-us 0 --work-us 0}"

mpirun -np ${SLURM_NTASKS} \
  --map-by ppr:4:node \
  --bind-to numa \
  bash -c '
    export CUDA_VISIBLE_DEVICES=${OMPI_COMM_WORLD_LOCAL_RANK}
    exec ./mpi_allreduce_bench '"${BENCH_ARGS}"'
  '

echo "Job finished at $(date)"
