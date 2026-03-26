#!/bin/sh
#PJM -g gb25
#PJM -L rscgrp=short-o
#PJM -L node=1
#PJM --mpi proc=4
#PJM --omp thread=1
#PJM -L elapse=01:00:00
#PJM -N fistr_nonlinear
#PJM -j
#PJM -o logs/nonlinear_%j.log

module load odyssey
module load frontistr/5.5
module load metis/5.1.0
module load parmetis/4.0.3
module load trilinos/13.0.1
module load hdf5/1.12.0

NPROC=${PJM_MPI_PROC}
WORKDIR=${PJM_O_WORKDIR}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUNDIR=${WORKDIR}/results/nonlinear_${TIMESTAMP}

echo "=== Nonlinear Dynamic Analysis (Finite Deformation) ==="
echo "MPI processes: ${NPROC}"
echo "Run directory: ${RUNDIR}"
echo "Start: $(date)"

mkdir -p ${RUNDIR}
cd ${RUNDIR}

# hecmw_ctrl.dat（参考ファイルの形式に準拠）
cat > hecmw_ctrl.dat << EOF
#
# for partitioner
#
!MESH, NAME=part_in,TYPE=HECMW-ENTIRE
 ../../column_fistr.msh
!MESH, NAME=part_out,TYPE=HECMW-DIST
 column_${NPROC}
#
# for solver
#
!MESH, NAME=fstrMSH, TYPE=HECMW-DIST
 column_${NPROC}
!CONTROL, NAME=fstrCNT
 ../../column_nonlinear.cnt
!RESULT, NAME=fstrRES, IO=OUT
 column_nonlinear.res
!RESULT, NAME=vis_out, IO=OUT
 column_nonlinear_vis
EOF

cat > hecmw_part_ctrl.dat << EOF
!PARTITION,TYPE=NODE-BASED,METHOD=PMETIS,DOMAIN=${NPROC}
EOF

echo "--- Domain Decomposition (${NPROC} domains) ---"
mpirun -n 1 hecmw_part1

echo "--- FrontISTR Solver ---"
mpirun -n ${NPROC} fistr1

echo "End: $(date)"
echo "Results in: ${RUNDIR}/"