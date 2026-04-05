#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --partition=80g
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=120:00:00
#SBATCH -o ./cpp_log/slurm.%j.out
#SBATCH -e ./cpp_log/slurm.%j.err

. /etc/profile.d/modules.sh
module load nvhpc-openmpi3/24.7

# export OMP_NUM_THREADS=1
# export NV_ACC_TIME=1

./a.out

echo "Job finished at $(date)"