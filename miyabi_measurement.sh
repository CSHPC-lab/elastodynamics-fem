#!/bin/bash -l
#PBS -W group_list=gb25
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -N my_gpu_job
#PBS -o job_out.log
#PBS -e job_err.log

module unload nvidia
module load nvidia/25.11

cd ${PBS_O_WORKDIR}

export OMP_NUM_THREADS=12

mpirun -n 1 ncu -o my_profile -f --set full -s 100 -c 20 ./a.out

echo "Job finished at $(date)"