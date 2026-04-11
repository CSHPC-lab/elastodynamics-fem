#!/bin/bash -l

#SBATCH --nodes=4
#SBATCH --partition=40g
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --time=120:00:00
#SBATCH -o ./cpp_log/slurm.%j.out
#SBATCH -e ./cpp_log/slurm.%j.err

. /etc/profile.d/modules.sh
module load nvhpc/25.1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export NV_ACC_TIME=1

mpirun -n $SLURM_NTASKS ./a.out

echo "Job finished at $(date)"