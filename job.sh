#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --partition=40g
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=120:00:00
#SBATCH -o ./cpp_log/slurm.%j.out
#SBATCH -e ./cpp_log/slurm.%j.err

. /etc/profile.d/modules.sh
module load nvhpc/25.1

export OMP_NUM_THREADS=6        # 物理コア数に合わせる（SMT は使わない）
export OMP_PROC_BIND=close
export OMP_PLACES=cores          # "cores" 指定で各物理コアに 1 スレッドずつ
export OMP_DYNAMIC=false
export OMP_WAIT_POLICY=active

# export NV_ACC_TIME=1

mpirun -n $SLURM_NTASKS --map-by numa --bind-to numa ./a.out

echo "Job finished at $(date)"