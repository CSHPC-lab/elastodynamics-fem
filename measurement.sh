#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --nodelist=lynx02
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=00:30:00
#SBATCH -o ./cpp_log/slurm.%j.out
#SBATCH -e ./cpp_log/slurm.%j.err

. /etc/profile.d/modules.sh

export OMP_NUM_THREADS=24        # 物理コア数に合わせる（SMT は使わない）
export OMP_PROC_BIND=close
export OMP_PLACES=cores          # "cores" 指定で各物理コアに 1 スレッドずつ
export OMP_DYNAMIC=false
export OMP_WAIT_POLICY=active

# mpirun -n 1 ncu -o my_profile_${SLURM_JOB_ID} -f --set full -s 100 -c 20 ./a.out
./a.out

echo "Job finished at $(date)"