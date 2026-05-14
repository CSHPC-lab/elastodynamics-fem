#include <mpi.h>
#include <stdio.h>
#include <vector>

// 指定ペア (src, dst) 間のピンポンレイテンシを測定する
// 非参加ランクも Barrier には参加するためデッドロックしない
static double pingpong_pair(int rank, int src, int dst, int n_bytes, int n_rep)
{
    std::vector<char> buf(n_bytes, 0);
    MPI_Status st;

    // ウォームアップ（Barrier なし、src/dst だけ通信）
    for (int i = 0; i < 10; i++)
    {
        if (rank == src)
        {
            MPI_Send(buf.data(), n_bytes, MPI_BYTE, dst, 99, MPI_COMM_WORLD);
            MPI_Recv(buf.data(), n_bytes, MPI_BYTE, dst, 99, MPI_COMM_WORLD, &st);
        }
        else if (rank == dst)
        {
            MPI_Recv(buf.data(), n_bytes, MPI_BYTE, src, 99, MPI_COMM_WORLD, &st);
            MPI_Send(buf.data(), n_bytes, MPI_BYTE, src, 99, MPI_COMM_WORLD);
        }
        // 他ランク: 何もしない（blocking send/recv はペア間のみなので問題なし）
    }

    // 全ランク同期してから計測開始
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int i = 0; i < n_rep; i++)
    {
        if (rank == src)
        {
            MPI_Send(buf.data(), n_bytes, MPI_BYTE, dst, 99, MPI_COMM_WORLD);
            MPI_Recv(buf.data(), n_bytes, MPI_BYTE, dst, 99, MPI_COMM_WORLD, &st);
        }
        else if (rank == dst)
        {
            MPI_Recv(buf.data(), n_bytes, MPI_BYTE, src, 99, MPI_COMM_WORLD, &st);
            MPI_Send(buf.data(), n_bytes, MPI_BYTE, src, 99, MPI_COMM_WORLD);
        }
    }

    double elapsed = MPI_Wtime() - t0;

    // 計測後も全ランク同期
    MPI_Barrier(MPI_COMM_WORLD);

    // src/dst どちらかが計測値を持つ（同じ値）、他は 0
    if (rank == src)
        return elapsed / n_rep / 2.0 * 1e6; // 片道 μs
    return 0.0;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ---- AllReduce ウォームアップ ----
    double x = 1.0;
    for (int i = 0; i < 20; i++)
        MPI_Allreduce(MPI_IN_PLACE, &x, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // ---- AllReduce: 1 double ----
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < 500; i++)
        MPI_Allreduce(MPI_IN_PLACE, &x, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double lat_ar1 = (MPI_Wtime() - t0) / 500 * 1e6;

    // ---- AllReduce: 2 doubles（統合後 PCG パターン）----
    double y[2] = {1.0, 2.0};
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < 500; i++)
    {
        MPI_Allreduce(MPI_IN_PLACE, &x,   1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, y,    2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    double lat_ar2 = (MPI_Wtime() - t0) / 500 * 1e6;

    // ---- 点対点 ping-pong ----
    // ハロー交換に相当するサイズ:
    //   Partition1 ↔ Partition2: send 6112 nodes × 3 comp × 8B = 147KB（1成分 49KB）
    //   Partition3 ↔ Partition8: cross-node, 12594 nodes × 3 comp = 302KB
    const int sizes[]  = {8, 1024, 8*1024, 50*1024, 150*1024, 300*1024};
    const char* labels[] = {"8B", "1KB", "8KB", "50KB(1成分)", "150KB(3成分intra)", "300KB(3成分cross)"};
    const int n_sizes = 6;
    const int N_PP = 500;

    // rank0 ↔ rank1: intra-node (lynx05)
    double lat_intra[n_sizes], lat_inter[n_sizes];
    for (int s = 0; s < n_sizes; s++)
        lat_intra[s] = pingpong_pair(rank, 0, 1, sizes[s], N_PP);

    // rank0 ↔ rank4: inter-node (lynx05 ↔ lynx06)
    for (int s = 0; s < n_sizes; s++)
        lat_inter[s] = pingpong_pair(rank, 0, 4, sizes[s], N_PP);

    if (rank == 0)
    {
        printf("=== Network latency (ranks=%d, all synchronized) ===\n\n", size);

        printf("[AllReduce]\n");
        printf("  1 double  (1 call)                 : %8.1f us\n", lat_ar1);
        printf("  2x allreduce/iter (現在の PCG パターン): %8.1f us/round\n", lat_ar2);
        printf("\n");

        printf("[Ping-pong: rank0 <-> rank1 (intra-node, lynx05 内)]\n");
        printf("  %-22s  %10s\n", "size", "us (one-way)");
        for (int s = 0; s < n_sizes; s++)
            printf("  %-22s  %10.1f\n", labels[s], lat_intra[s]);
        printf("\n");

        printf("[Ping-pong: rank0 <-> rank4 (inter-node, lynx05 <-> lynx06)]\n");
        printf("  %-22s  %10s\n", "size", "us (one-way)");
        for (int s = 0; s < n_sizes; s++)
            printf("  %-22s  %10.1f\n", labels[s], lat_inter[s]);
        printf("\n");

        printf("[FEM コード実測値（参考、負荷不均衡込み）]\n");
        printf("  AllReduce pAp  : 57000 us → 純遅延 %.1f us → 不均衡 %.0f us\n", lat_ar1, 57000.0 - lat_ar1);
        printf("  HaloWait       : 29000 us → 点対点レイテンシと比較して判断\n");
    }

    MPI_Finalize();
    return 0;
}
