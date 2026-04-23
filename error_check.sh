#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --partition=40g
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=120:00:00
#SBATCH -o ./cpp_log/slurm.%j.out
#SBATCH -e ./cpp_log/slurm.%j.err

. /etc/profile.d/modules.sh
module load nvhpc/25.1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export NV_ACC_TIME=1

mpirun -n $SLURM_NTASKS compute-sanitizer ./a.out

echo "Job finished at $(date)"