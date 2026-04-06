#!/bin/sh
#PJM -g gb25
#PJM -L rscgrp=short-o
#PJM -L node=4
#PJM --mpi proc=16
#PJM --omp thread=12
#PJM -L elapse=01:00:00
#PJM -j
#PJM -o ./logs/%j.out
#PJM -e ./logs/%j.err
#PJM -s --spath ./logs/%j.stats

module load odyssey
module load frontistr/5.5
module load metis/5.1.0
module load parmetis/4.0.3
module load trilinos/13.0.1
module load hdf5/1.12.0

cd ${PJM_O_WORKDIR}
mpirun -n 1 hecmw_part1
mpirun -n 16 fistr1

rm column_fistr_4.*
rm column_fistr.res.*
rm FSTR.*
rm hecmw_part.log
rm part.inp