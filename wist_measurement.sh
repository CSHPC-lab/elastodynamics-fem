#!/bin/bash -l
#PJM -g gb25
#PJM -L rscgrp=short-a
#PJM -L node=1
#PJM --mpi proc=1
#PJM --omp thread=48
#PJM -L elapse=00:30:00
#PJM -j
#PJM -o ./logs/%j.out
#PJM -e ./logs/%j.err
#PJM -s --spath ./logs/%j.stats

. /etc/profile.d/modules.sh
module load nvidia/25.9
module load nvmpi/25.9

export OMP_NUM_THREADS=$PJM_OMP_THREAD

mpirun -n 1 ncu -o my_profile_${PJM_JOBID} -f --set full -s 100 -c 20 ./a.out

echo "Job finished at $(date)"