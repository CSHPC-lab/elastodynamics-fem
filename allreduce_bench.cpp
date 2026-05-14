#include <mpi.h>
#include <stdio.h>
#include <vector>
#include <algorithm>

// ランク間の純粋なレイテンシを測定する
// AllReduce と点対点（ping-pong）の両方を計測して
// FEM コードの HaloWait / AllReduce の遅さがネットワーク由来か負荷不均衡由来かを切り分ける

// 全ランクが同期した状態でのピンポンレイテンシ（送信元 → 送信先 → 送信元）
static double pingpong(int rank, int peer, int n_bytes, int n_rep)
{
    std::vector<char> buf(n_bytes, 0);
    MPI_Status st;

    // ウォームアップ
    for (int i = 0; i < 5; i++)
    {
        if (rank < peer)
        {
            MPI_Send(buf.data(), n_bytes, MPI_BYTE, peer, 0, MPI_COMM_WORLD);
            MPI_Recv(buf.data(), n_bytes, MPI_BYTE, peer, 0, MPI_COMM_WORLD, &st);
        }
        else
        {
            MPI_Recv(buf.data(), n_bytes, MPI_BYTE, peer, 0, MPI_COMM_WORLD, &st);
            MPI_Send(buf.data(), n_bytes, MPI_BYTE, peer, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < n_rep; i++)
    {
        if (rank < peer)
        {
            MPI_Send(buf.data(), n_bytes, MPI_BYTE, peer, 0, MPI_COMM_WORLD);
            MPI_Recv(buf.data(), n_bytes, MPI_BYTE, peer, 0, MPI_COMM_WORLD, &st);
        }
        else
        {
            MPI_Recv(buf.data(), n_bytes, MPI_BYTE, peer, 0, MPI_COMM_WORLD, &st);
            MPI_Send(buf.data(), n_bytes, MPI_BYTE, peer, 0, MPI_COMM_WORLD);
        }
    }
    return (MPI_Wtime() - t0) / n_rep / 2.0 * 1e6; // 片道 μs
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N_WARMUP  = 10;
    const int N_MEASURE = 200;

    double x1 = 1.0;
    double x3[3] = {1.0, 2.0, 3.0};

    // ---- AllReduce ウォームアップ ----
    for (int i = 0; i < N_WARMUP; i++)
    {
        MPI_Allreduce(MPI_IN_PLACE, &x1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, x3, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }

    // ---- 測定 1: AllReduce 1 double ----
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < N_MEASURE; i++)
        MPI_Allreduce(MPI_IN_PLACE, &x1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double lat_ar1 = (MPI_Wtime() - t0) / N_MEASURE * 1e6;

    // ---- 測定 2: AllReduce 2 doubles（統合後の PCG パターン）----
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < N_MEASURE; i++)
    {
        MPI_Allreduce(MPI_IN_PLACE, &x1,  1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); // pAp
        MPI_Allreduce(MPI_IN_PLACE, x3,   2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); // rnorm+rznew
    }
    double lat_ar2seq = (MPI_Wtime() - t0) / N_MEASURE * 1e6;

    // ---- 測定 3: ping-pong（同一ノード内、intra-node）----
    // rank 0 ↔ rank 1 (lynx05 内)
    double lat_intra = 0, lat_inter = 0;
    const int sizes[] = {8, 1024, 8*1024, 50*1024, 150*1024}; // 8B, 1KB, 8KB, 50KB, 150KB
    const int n_sizes = sizeof(sizes)/sizeof(sizes[0]);
    double lat_pp_intra[n_sizes], lat_pp_inter[n_sizes];

    for (int s = 0; s < n_sizes; s++)
    {
        // intra-node: rank 0 ↔ rank 1
        if (rank == 0 || rank == 1)
            lat_pp_intra[s] = pingpong(rank, 1 - rank, sizes[s], 200);
        else
        {
            // other ranks just participate in MPI_Barrier inside pingpong
            MPI_Barrier(MPI_COMM_WORLD); // warmup barrier
            MPI_Barrier(MPI_COMM_WORLD); // measure barrier
            lat_pp_intra[s] = 0;
        }

        // inter-node: rank 0 (lynx05) ↔ rank 4 (lynx06)
        if (rank == 0 || rank == 4)
            lat_pp_inter[s] = pingpong(rank, 4 - rank, sizes[s], 200);
        else
        {
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            lat_pp_inter[s] = 0;
        }
    }

    if (rank == 0)
    {
        printf("=== Network latency (ranks=%d, all synchronized) ===\n\n", size);

        printf("[AllReduce]\n");
        printf("  1 double  (1 call)             : %8.1f us/call\n", lat_ar1);
        printf("  2x allreduce per iter (現在PCG): %8.1f us/round\n", lat_ar2seq);
        printf("\n");

        printf("[Ping-pong latency: rank0 <-> rank1 (intra-node, lynx05)]\n");
        printf("  %8s  %12s\n", "bytes", "us (one-way)");
        for (int s = 0; s < n_sizes; s++)
            printf("  %8d  %12.1f\n", sizes[s], lat_pp_intra[s]);
        printf("\n");

        printf("[Ping-pong latency: rank0 <-> rank4 (inter-node, lynx05<->lynx06)]\n");
        printf("  %8s  %12s\n", "bytes", "us (one-way)");
        for (int s = 0; s < n_sizes; s++)
            printf("  %8d  %12.1f\n", sizes[s], lat_pp_inter[s]);
        printf("\n");

        printf("[FEM コード実測値（参考）]\n");
        printf("  AllReduce pAp (負荷不均衡込み) : 57000 us/call → 差=負荷不均衡\n");
        printf("  HaloWait                       : 29000 us/iter → 点対点レイテンシ比で判断\n");
    }

    MPI_Finalize();
    return 0;
}
