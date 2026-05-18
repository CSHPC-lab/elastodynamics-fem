#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --partition=40g
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=120:00:00
#SBATCH -o ../cpp_log/slurm.%j.out
#SBATCH -e ../cpp_log/slurm.%j.err

cd /data3/kusumoto/elastodynamics-fem/explicit

# gfortran -O2 -ffixed-line-length-none -mcmodel=medium main.f keme.f pointsource.f /usr/lib/x86_64-linux-gnu/lapack/liblapack.so.3

./a.out

echo "Job finished at $(date)"
