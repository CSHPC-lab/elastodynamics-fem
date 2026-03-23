#!/bin/sh
#PJM -g gb25
#PJM -L rscgrp=short-o
#PJM -L node=1
#PJM --mpi proc=4
#PJM --omp thread=1
#PJM -L elapse=01:00:00
#PJM -N fistr_nonlinear
#PJM -j
 
module load odyssey
module load frontistr/5.5
module load metis/5.1.0
module load parmetis/4.0.3
module load trilinos/13.0.1
module load hdf5/1.12.0
 
cd ${PJM_O_WORKDIR}
 
NPROC=${PJM_MPI_PROC}
 
echo "=== Nonlinear Dynamic Analysis (Finite Deformation) ==="
echo "MPI processes: ${NPROC}"
echo "Start: $(date)"
 
# 制御ファイルを設定
cp hecmw_ctrl_nonlinear.dat hecmw_ctrl.dat
 
# ドメイン分割制御ファイルをプロセス数から自動生成
cat > hecmw_part_ctrl.dat << EOF
!PARTITION, TYPE=NODE-BASED, METHOD=PMETIS, DOMAIN=${NPROC}
EOF
 
# ドメイン分割
echo "--- Domain Decomposition (${NPROC} domains) ---"
mpirun -n 1 hecmw_part1
 
# 解析実行
echo "--- FrontISTR Solver ---"
mpirun -n ${NPROC} fistr1
 
echo "End: $(date)"