#!/bin/bash -l

#SBATCH --nodes=2
#SBATCH --partition=40g
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
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

# GDR (gdr_copy) 非対応環境: MPI に GPU ptr を渡さず pinned host staging を使う
# → UCX の暗黙 GPU sync を全 MPI コレクティブ前に抑止する
export UCX_TLS=rc,rc_mlx5,self
export UCX_WARN_UNUSED_ENV_VARS=n

# mpirun -n $SLURM_NTASKS --map-by numa --bind-to numa ./a.out
export CUDA_DEVICE_ORDER=PCI_BUS_ID

mpirun -np ${SLURM_NTASKS} \
    --map-by ppr:4:node \
    --bind-to numa \
    bash -c '
      export CUDA_VISIBLE_DEVICES=${OMPI_COMM_WORLD_LOCAL_RANK}
      echo rank=${OMPI_COMM_WORLD_RANK} local=${OMPI_COMM_WORLD_LOCAL_RANK} host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
      exec ./a.out
    '

echo "Job finished at $(date)"