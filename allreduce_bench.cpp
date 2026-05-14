#include <mpi.h>
#include <stdio.h>

// 全ランクが同期した状態での AllReduce 素の遅延を測定する
// 用途: FEM コードで見える 40-60ms が「本当のネットワーク遅延」か
//       「ランク間の負荷不均衡による待ち時間」かを切り分ける

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N_WARMUP = 10;
    const int N_MEASURE = 200;

    double x1 = 1.0;
    double x3[3] = {1.0, 2.0, 3.0};

    // ウォームアップ（MPI 接続確立 + JIT コンパイル等）
    for (int i = 0; i < N_WARMUP; i++)
    {
        MPI_Allreduce(MPI_IN_PLACE, &x1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, x3, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }

    // ---- 測定 1: 1 double ----
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < N_MEASURE; i++)
        MPI_Allreduce(MPI_IN_PLACE, &x1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double lat1 = (MPI_Wtime() - t0) / N_MEASURE * 1e6; // μs per call

    // ---- 測定 2: 3 doubles（PCG の pAp/rnorm/rznew をまとめた場合の参考）----
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < N_MEASURE; i++)
        MPI_Allreduce(MPI_IN_PLACE, x3, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double lat3 = (MPI_Wtime() - t0) / N_MEASURE * 1e6;

    // ---- 測定 3: 連続 3 回（現在の PCG と同じパターン）----
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < N_MEASURE; i++)
    {
        MPI_Allreduce(MPI_IN_PLACE, &x1,   1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, x3,    1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, x3+1,  1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    double lat3seq = (MPI_Wtime() - t0) / N_MEASURE * 1e6;

    if (rank == 0)
    {
        printf("=== AllReduce latency (ranks=%d, N=%d, all ranks synchronized) ===\n", size, N_MEASURE);
        printf("  1 double  (single call)         : %8.1f us/call\n", lat1);
        printf("  3 doubles (single call)         : %8.1f us/call\n", lat3);
        printf("  3x 1 double (sequential, PCG型) : %8.1f us/round\n", lat3seq);
        printf("\n");
        printf("FEM コードでの AllReduce 計測値（参考）:\n");
        printf("  pAp allreduce (ランク不均衡込み) : 57000 us/call\n");
        printf("  → 純遅延 vs FEM の差が負荷不均衡の大きさ\n");
    }

    MPI_Finalize();
    return 0;
}
