#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --partition=40g
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=120:00:00
#SBATCH -o ./cpp_log/slurm.%j.out
#SBATCH -e ./cpp_log/slurm.%j.err

. /etc/profile.d/modules.sh
export LD_LIBRARY_PATH=$(pwd)/lib:$LD_LIBRARY_PATH

g++ -Iinclude column.cpp -Llib -lgmsh
export LD_LIBRARY_PATH=$(pwd)/lib:$LD_LIBRARY_PATH
./a.out

echo "Job finished at $(date)"