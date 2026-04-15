#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --nodelist=lynx02
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=00:30:00
#SBATCH -o ./cpp_log/slurm.%j.out
#SBATCH -e ./cpp_log/slurm.%j.err

. /etc/profile.d/modules.sh
module load nvhpc/25.1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mpirun -n 1 ncu -o my_profile_${SLURM_JOB_ID} -f --set full -s 100 -c 20 ./a.out

echo "Job finished at $(date)"