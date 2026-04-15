/*実行コマンド
cd /data3/kusumoto/elastodynamics-fem/
module load nvhpc/25.1
mpicxx main_oacc.cpp msh_reader.cpp -fopenmp -acc -Minfo=accel
./a.out
*/

#include "msh_reader.hpp"
#include "config.hpp"
#include <iostream>
#include <cmath>
#include <ctime>
#include <sys/stat.h>
#include <omp.h>
#include <openacc.h>
#include <mpi.h>

void calculate_dN(double dN[30], double r, double s, double t);
void gauss_integrate(double dN0[30], double dN1[30], double dN2[30], double dN3[30]);
double inverse_3_3_mat(double mat[9], double inv_mat[9]);
void construct_mat(
    double *node_coords,
    int *ele_nodes,
    int num_elements,
    double dN0[30],
    double dN1[30],
    double dN2[30],
    double dN3[30],
    double lambda,
    double mu,
    double rho,
    double dt,
    double *kmat_coo_val,
    double *mmat_coo_val,
    int *coo_row,
    int *coo_col);
int sort_and_merge_bcoo(
    int nnz_coo,
    int num_nodes,
    int *coo_row,
    int *coo_col,
    double *kmat_coo_val,
    double *mmat_coo_val);
void build_bcrs(
    int *coo_row,
    int *coo_col,
    double *kmat_coo_val,
    double *mmat_coo_val,
    int nnz_bcoo,
    int num_nodes,
    int *bcrs_row_ptr,
    int *bcrs_col_ind,
    double *bcrs_kval_00,
    double *bcrs_kval_01,
    double *bcrs_kval_02,
    double *bcrs_kval_10,
    double *bcrs_kval_11,
    double *bcrs_kval_12,
    double *bcrs_kval_20,
    double *bcrs_kval_21,
    double *bcrs_kval_22,
    double *bcrs_mval);
void extract_bc_correction(
    int num_nodes,
    int *bcrs_row_ptr,
    int *bcrs_col_ind,
    double *kval_00,
    double *kval_01,
    double *kval_02,
    double *kval_10,
    double *kval_11,
    double *kval_12,
    double *kval_20,
    double *kval_21,
    double *kval_22,
    int *bc_flag,
    double *bc_corr_00,
    double *bc_corr_01,
    double *bc_corr_02,
    double *bc_corr_10,
    double *bc_corr_11,
    double *bc_corr_12,
    double *bc_corr_20,
    double *bc_corr_21,
    double *bc_corr_22);
void apply_bc_to_lhs(
    int num_nodes,
    int *bcrs_row_ptr,
    int *bcrs_col_ind,
    double *kval_00,
    double *kval_01,
    double *kval_02,
    double *kval_10,
    double *kval_11,
    double *kval_12,
    double *kval_20,
    double *kval_21,
    double *kval_22,
    int *bc_flag);
void apply_bc_to_rhs(
    int num_nodes,
    int *bc_flag,
    double *bc_val,
    double *bc_corr_00,
    double *bc_corr_01,
    double *bc_corr_02,
    double *bc_corr_10,
    double *bc_corr_11,
    double *bc_corr_12,
    double *bc_corr_20,
    double *bc_corr_21,
    double *bc_corr_22,
    double *rhs_0,
    double *rhs_1,
    double *rhs_2);
void build_block_jacobi(
    int num_nodes,
    int *row_ptr,
    int *col_ind,
    double *kval_00,
    double *kval_01,
    double *kval_02,
    double *kval_10,
    double *kval_11,
    double *kval_12,
    double *kval_20,
    double *kval_21,
    double *kval_22,
    double *inv_diag_00,
    double *inv_diag_01,
    double *inv_diag_02,
    double *inv_diag_10,
    double *inv_diag_11,
    double *inv_diag_12,
    double *inv_diag_20,
    double *inv_diag_21,
    double *inv_diag_22);
void bcrs_spmv_m(
    int num_nodes,
    int *row_ptr,
    int *col_ind,
    double *val,
    double *x_0,
    double *x_1,
    double *x_2,
    double *y_0,
    double *y_1,
    double *y_2);
int pcg_solve(
    MPI_Request *request,
    int num_neighbors,
    int *neighbor_ranks,
    int *recv_starts,
    int *recv_counts,
    int *send_starts,
    int *send_counts,
    int *send_nodes,
    double *send_buffer_0,
    double *send_buffer_1,
    double *send_buffer_2,
    int num_inner,
    int num_owned,
    int num_nodes,
    int *row_ptr,
    int *col_ind,
    double *kval_00,
    double *kval_01,
    double *kval_02,
    double *kval_10,
    double *kval_11,
    double *kval_12,
    double *kval_20,
    double *kval_21,
    double *kval_22,
    double *inv_diag_00,
    double *inv_diag_01,
    double *inv_diag_02,
    double *inv_diag_10,
    double *inv_diag_11,
    double *inv_diag_12,
    double *inv_diag_20,
    double *inv_diag_21,
    double *inv_diag_22,
    double *b_0,
    double *b_1,
    double *b_2,
    double *x_0,
    double *x_1,
    double *x_2,
    double tol,
    int max_iter,
    double *r_0,
    double *r_1,
    double *r_2,
    double *z_0,
    double *z_1,
    double *z_2,
    double *p_0,
    double *p_1,
    double *p_2,
    double *Ap_0,
    double *Ap_1,
    double *Ap_2);
void write_vtk_displacement(
    const char *filename,
    double *node_coords,
    int num_nodes,
    int *ele_nodes,
    int num_elements,
    double *displacement,
    double total_time);
void write_node_disp_csv(
    const char *filename,
    double *u,
    int num_steps,
    int sample_freq,
    double dt);

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Request requests[1000];

    double start_time = MPI_Wtime();

    FEMMesh mesh = read_msh("column_4.msh", rank + 1);
    print_mesh_info(mesh);

    Config cfg;
    cfg.load("config.txt");

    double duration = cfg.get_double("duration");
    int num_steps = cfg.get_int("num_steps");
    int sample_freq = cfg.get_int("sample_freq");
    double c1 = cfg.get_double("c1");
    double c2 = cfg.get_double("c2");
    double rho = cfg.get_double("rho");
    int target_node = get_local_id(mesh, cfg.get_int("target_node"));
    int force_node = get_local_id(mesh, cfg.get_int("force_node"));
    int force_dof = cfg.get_int("force_dof");
    double force_magnitude = cfg.get_double("force_magnitude");

    printf("c1: %.2f m/s, c2: %.2f m/s\n, rho: %.2e kg/m^3\n", c1, c2, rho);

    double lambda = rho * (c1 * c1 - 2 * c2 * c2);
    double mu = rho * c2 * c2;
    double dt = duration / num_steps;

    printf("使用可能な最大スレッド数：%d\n", omp_get_max_threads());
    printf("使用可能な最大GPU数：%d\n", acc_get_num_devices(acc_device_nvidia));

    acc_set_device_num(rank % acc_get_num_devices(acc_device_nvidia), acc_device_nvidia); // GPUをランクに応じて割り当て
    acc_init(acc_device_nvidia);

    double *node_coords = mesh.coords_ptr();
    int num_nodes = mesh.num_total;
    int num_inner = mesh.num_inner;
    int num_bdr = mesh.num_bdr;
    int num_owned = mesh.num_owned;
    int num_ghost = mesh.num_ghost;
    int *ele_nodes = mesh.elem_ptr();
    int num_elements = mesh.num_total_elems;

    int num_neighbors = mesh.num_neighbors();
    int neighbor_ranks[num_neighbors];
    int recv_starts[num_neighbors];     // 受信節点の開始位置（ローカルID配列中の位置）
    int recv_counts[num_neighbors];     // 受信節点数（受信節点のメモリは隣接プロセス単位で連続）
    std::vector<int> send_nodes;        // 送信節点のローカルIDを格納する配列（隣接プロセス単位で連続、ローカルID配列は連続ではない）
    int send_starts[num_neighbors + 1]; // 送信節点の開始位置（send_nodes配列中の位置、最後の要素は全送信節点数）
    int send_counts[num_neighbors];     // 送信節点数（send_nodes配列中の数）
    for (int i = 0; i < num_neighbors; i++)
    {
        neighbor_ranks[i] = mesh.neighbors[i].partition_id - 1; // 0-indexed に変換
        recv_starts[i] = mesh.neighbors[i].recv_start;
        recv_counts[i] = mesh.neighbors[i].recv_count;
        send_starts[i] = 0;
        send_counts[i] = mesh.neighbors[i].send_size();
        if (i > 0)
        {
            send_starts[i] = send_starts[i - 1] + send_counts[i - 1];
        }
        send_nodes.insert(send_nodes.end(), mesh.neighbors[i].send_nodes.begin(), mesh.neighbors[i].send_nodes.end());
    }
    send_starts[num_neighbors] = send_starts[num_neighbors - 1] + send_counts[num_neighbors - 1];
    int *send_nodes_ptr = send_nodes.data();
    double send_buffer_0[send_starts[num_neighbors]];
    double send_buffer_1[send_starts[num_neighbors]];
    double send_buffer_2[send_starts[num_neighbors]];

    double dN0[30] = {0.0};                                      // ガウス点での形状関数の微分の配列
    double dN1[30] = {0.0};                                      // ガウス点での形状関数の微分の配列
    double dN2[30] = {0.0};                                      // ガウス点での形状関数の微分の配列
    double dN3[30] = {0.0};                                      // ガウス点での形状関数の微分の配列
    double *kmat_coo_val = new double[100 * num_elements * 9](); // 要素剛性行列の配列
    double *mmat_coo_val = new double[100 * num_elements]();     // 要素質量行列の配列
    int *coo_row = new int[100 * num_elements]();                // 要素行列の行インデックスの配列
    int *coo_col = new int[100 * num_elements]();                // 要素行列の列インデックスの配列

    if (target_node >= 0)
    {
        printf("Target node local ID: %d\n", target_node);
        printf("Target node global ID: %d\n", mesh.local_to_global[target_node]);
        printf("Target node coordinates: (%.3f, %.3f, %.3f)\n",
               node_coords[target_node * 3],
               node_coords[target_node * 3 + 1],
               node_coords[target_node * 3 + 2]);
    }

    gauss_integrate(dN0, dN1, dN2, dN3);

    construct_mat(node_coords,
                  ele_nodes,
                  num_elements,
                  dN0,
                  dN1,
                  dN2,
                  dN3,
                  lambda,
                  mu,
                  rho,
                  dt,
                  kmat_coo_val,
                  mmat_coo_val,
                  coo_row,
                  coo_col);

    int nnz_bcrs = sort_and_merge_bcoo(100 * num_elements, num_nodes, coo_row, coo_col, kmat_coo_val, mmat_coo_val);

    double *bcrs_kval_00 = new double[nnz_bcrs];
    double *bcrs_kval_01 = new double[nnz_bcrs];
    double *bcrs_kval_02 = new double[nnz_bcrs];
    double *bcrs_kval_10 = new double[nnz_bcrs];
    double *bcrs_kval_11 = new double[nnz_bcrs];
    double *bcrs_kval_12 = new double[nnz_bcrs];
    double *bcrs_kval_20 = new double[nnz_bcrs];
    double *bcrs_kval_21 = new double[nnz_bcrs];
    double *bcrs_kval_22 = new double[nnz_bcrs];
    double *bcrs_mval = new double[nnz_bcrs];
    int *bcrs_row_ptr = new int[num_nodes + 1];
    int *bcrs_col_ind = new int[nnz_bcrs];

    build_bcrs(coo_row, coo_col, kmat_coo_val, mmat_coo_val, nnz_bcrs, num_nodes, bcrs_row_ptr, bcrs_col_ind, bcrs_kval_00, bcrs_kval_01, bcrs_kval_02,
               bcrs_kval_10, bcrs_kval_11, bcrs_kval_12, bcrs_kval_20, bcrs_kval_21, bcrs_kval_22, bcrs_mval);

    delete[] coo_row;
    delete[] coo_col;
    delete[] kmat_coo_val;
    delete[] mmat_coo_val;

    std::cout << "Number of non-zero blocks in bcrs: " << nnz_bcrs << std::endl;

    int *bc_flag = new int[num_nodes]();
    double *bc_corr_00 = new double[num_nodes]();
    double *bc_corr_01 = new double[num_nodes]();
    double *bc_corr_02 = new double[num_nodes]();
    double *bc_corr_10 = new double[num_nodes]();
    double *bc_corr_11 = new double[num_nodes]();
    double *bc_corr_12 = new double[num_nodes]();
    double *bc_corr_20 = new double[num_nodes]();
    double *bc_corr_21 = new double[num_nodes]();
    double *bc_corr_22 = new double[num_nodes]();
    double *inv_diag_00 = new double[num_nodes]();
    double *inv_diag_01 = new double[num_nodes]();
    double *inv_diag_02 = new double[num_nodes]();
    double *inv_diag_10 = new double[num_nodes]();
    double *inv_diag_11 = new double[num_nodes]();
    double *inv_diag_12 = new double[num_nodes]();
    double *inv_diag_20 = new double[num_nodes]();
    double *inv_diag_21 = new double[num_nodes]();
    double *inv_diag_22 = new double[num_nodes]();
    double *u = new double[(num_steps / sample_freq + 1) * 3](); // 変位の記録 (target_nodeのみ)
    double *u_tmp_0 = new double[num_nodes]();                   // タイムステップごとの変位の一時保存用
    double *u_tmp_1 = new double[num_nodes]();                   // タイムステップごとの変位の一時保存用
    double *u_tmp_2 = new double[num_nodes]();                   // タイムステップごとの変位の一時保存用
    double *v_tmp_0 = new double[num_nodes]();                   // タイムステップごとの速度の一時保存用
    double *v_tmp_1 = new double[num_nodes]();                   // タイムステップごとの速度の一時保存用
    double *v_tmp_2 = new double[num_nodes]();                   // タイムステップごとの速度の一時保存用
    double *a_tmp_0 = new double[num_nodes]();                   // タイムステップごとの加速度の一時保存用
    double *a_tmp_1 = new double[num_nodes]();                   // タイムステップごとの加速度の一時保存用
    double *a_tmp_2 = new double[num_nodes]();                   // タイムステップごとの加速度の一時保存用
    double *u_prv_0 = new double[num_nodes]();                   // タイムステップごとの過去の変位の一時保存用
    double *u_prv_1 = new double[num_nodes]();                   // タイムステップごとの過去の変位の一時保存用
    double *u_prv_2 = new double[num_nodes]();                   // タイムステップごとの過去の変位の一時保存用
    double *rhs_0 = new double[num_nodes]();
    double *rhs_1 = new double[num_nodes]();
    double *rhs_2 = new double[num_nodes]();
    double *tmp_0 = new double[num_nodes]();
    double *tmp_1 = new double[num_nodes]();
    double *tmp_2 = new double[num_nodes]();

    // PCGソルバー用のベクトル
    double *r_0 = new double[num_nodes]();
    double *r_1 = new double[num_nodes]();
    double *r_2 = new double[num_nodes]();
    double *z_0 = new double[num_nodes]();
    double *z_1 = new double[num_nodes]();
    double *z_2 = new double[num_nodes]();
    double *p_0 = new double[num_nodes]();
    double *p_1 = new double[num_nodes]();
    double *p_2 = new double[num_nodes]();
    double *Ap_0 = new double[num_nodes]();
    double *Ap_1 = new double[num_nodes]();
    double *Ap_2 = new double[num_nodes]();

// 境界条件
#pragma omp parallel for
    for (int i = 0; i < num_nodes; i++)
    {
        if (node_coords[i * 3 + 2] < 1e-6)
        {
            bc_flag[i] = 1;
            // u_tmp[i * 3 + 0] = 0.0;
            // u_tmp[i * 3 + 1] = sin(0.0);
            // u_tmp[i * 3 + 2] = 0.0;
            // v_tmp[i * 3 + 0] = 0.0;
            // v_tmp[i * 3 + 1] = cos(0.0);
            // v_tmp[i * 3 + 2] = 0.0;
            // a_tmp[i * 3 + 0] = 0.0;
            // a_tmp[i * 3 + 1] = -sin(0.0);
            // a_tmp[i * 3 + 2] = 0.0;
        }
    }

    extract_bc_correction(num_nodes, bcrs_row_ptr, bcrs_col_ind, bcrs_kval_00, bcrs_kval_01, bcrs_kval_02, bcrs_kval_10, bcrs_kval_11, bcrs_kval_12, bcrs_kval_20,
                          bcrs_kval_21, bcrs_kval_22, bc_flag, bc_corr_00, bc_corr_01, bc_corr_02, bc_corr_10, bc_corr_11, bc_corr_12, bc_corr_20, bc_corr_21, bc_corr_22);

    apply_bc_to_lhs(num_nodes, bcrs_row_ptr, bcrs_col_ind, bcrs_kval_00, bcrs_kval_01, bcrs_kval_02, bcrs_kval_10, bcrs_kval_11, bcrs_kval_12, bcrs_kval_20, bcrs_kval_21,
                    bcrs_kval_22, bc_flag);

    build_block_jacobi(num_nodes, bcrs_row_ptr, bcrs_col_ind, bcrs_kval_00, bcrs_kval_01, bcrs_kval_02, bcrs_kval_10, bcrs_kval_11, bcrs_kval_12, bcrs_kval_20, bcrs_kval_21,
                       bcrs_kval_22, inv_diag_00, inv_diag_01, inv_diag_02, inv_diag_10, inv_diag_11, inv_diag_12, inv_diag_20, inv_diag_21, inv_diag_22);

    double data_transfer_start = MPI_Wtime();
#pragma acc enter data copyin(recv_starts[0 : num_neighbors], recv_counts[0 : num_neighbors], send_starts[0 : num_neighbors + 1], send_counts[0 : num_neighbors],             \
                              send_nodes_ptr[0 : send_starts[num_neighbors]], send_buffer_0[0 : send_starts[num_neighbors]], send_buffer_1[0 : send_starts[num_neighbors]],   \
                              send_buffer_2[0 : send_starts[num_neighbors]],                                                                                                  \
                              tmp_0[0 : num_nodes], tmp_1[0 : num_nodes], tmp_2[0 : num_nodes], u_tmp_0[0 : num_nodes], u_tmp_1[0 : num_nodes], u_tmp_2[0 : num_nodes],       \
                              v_tmp_0[0 : num_nodes], v_tmp_1[0 : num_nodes], v_tmp_2[0 : num_nodes], a_tmp_0[0 : num_nodes], a_tmp_1[0 : num_nodes], a_tmp_2[0 : num_nodes], \
                              bcrs_row_ptr[0 : num_nodes + 1], bcrs_col_ind[0 : nnz_bcrs], bcrs_mval[0 : nnz_bcrs], rhs_0[0 : num_nodes], rhs_1[0 : num_nodes],               \
                              rhs_2[0 : num_nodes], bc_flag[0 : num_nodes],                                                                                                   \
                              bcrs_kval_00[0 : nnz_bcrs], bcrs_kval_01[0 : nnz_bcrs], bcrs_kval_02[0 : nnz_bcrs], bcrs_kval_10[0 : nnz_bcrs],                                 \
                              bcrs_kval_11[0 : nnz_bcrs], bcrs_kval_12[0 : nnz_bcrs], bcrs_kval_20[0 : nnz_bcrs], bcrs_kval_21[0 : nnz_bcrs], bcrs_kval_22[0 : nnz_bcrs],     \
                              bc_corr_00[0 : num_nodes], bc_corr_01[0 : num_nodes], bc_corr_02[0 : num_nodes], bc_corr_10[0 : num_nodes], bc_corr_11[0 : num_nodes],          \
                              bc_corr_12[0 : num_nodes], bc_corr_20[0 : num_nodes], bc_corr_21[0 : num_nodes], bc_corr_22[0 : num_nodes],                                     \
                              inv_diag_00[0 : num_nodes], inv_diag_01[0 : num_nodes], inv_diag_02[0 : num_nodes], inv_diag_10[0 : num_nodes], inv_diag_11[0 : num_nodes],     \
                              inv_diag_12[0 : num_nodes], inv_diag_20[0 : num_nodes], inv_diag_21[0 : num_nodes], inv_diag_22[0 : num_nodes],                                 \
                              r_0[0 : num_nodes], r_1[0 : num_nodes], r_2[0 : num_nodes], z_0[0 : num_nodes], z_1[0 : num_nodes], z_2[0 : num_nodes],                         \
                              p_0[0 : num_nodes], p_1[0 : num_nodes], p_2[0 : num_nodes], Ap_0[0 : num_nodes], Ap_1[0 : num_nodes], Ap_2[0 : num_nodes],                      \
                              u_prv_0[0 : num_nodes], u_prv_1[0 : num_nodes], u_prv_2[0 : num_nodes])
    double data_transfer_end = MPI_Wtime();
    printf("Data transfer to GPU time: %.2f seconds\n", data_transfer_end - data_transfer_start);

    double bc_val_u[3] = {0.0};
    double bc_val_v[3] = {0.0};
    double bc_val_a[3] = {0.0};
#pragma acc enter data copyin(bc_val_u[0 : 3], bc_val_v[0 : 3], bc_val_a[0 : 3])

    // タイムステップループ
    for (int step = 1; step <= num_steps; step++)
    {
        double t = step * dt;
        bc_val_u[0] = 0.0;
        bc_val_u[1] = sin(t);
        bc_val_u[2] = 0.0;
        bc_val_v[0] = 0.0;
        bc_val_v[1] = cos(t);
        bc_val_v[2] = 0.0;
        bc_val_a[0] = 0.0;
        bc_val_a[1] = -sin(t);
        bc_val_a[2] = 0.0;
#pragma acc update device(bc_val_u[0 : 3], bc_val_v[0 : 3], bc_val_a[0 : 3]) // 境界条件の値をGPUに転送

        // ここでrhsを構築（外力の寄与なども加える） u, v, aは全節点について、前のタイムステップから正しい値を引き継いでいる
#pragma acc parallel loop present(tmp_0, tmp_1, tmp_2, u_tmp_0, u_tmp_1, u_tmp_2, v_tmp_0, v_tmp_1, v_tmp_2, a_tmp_0, a_tmp_1, a_tmp_2)
        for (int i = 0; i < num_nodes; i++)
        {
            // Newmark-β法の右辺の構築
            tmp_0[i] = u_tmp_0[i] * 4.0 / dt / dt + v_tmp_0[i] * 4.0 / dt + a_tmp_0[i];
            tmp_1[i] = u_tmp_1[i] * 4.0 / dt / dt + v_tmp_1[i] * 4.0 / dt + a_tmp_1[i];
            tmp_2[i] = u_tmp_2[i] * 4.0 / dt / dt + v_tmp_2[i] * 4.0 / dt + a_tmp_2[i];
        }
        bcrs_spmv_m(num_nodes, bcrs_row_ptr, bcrs_col_ind, bcrs_mval, tmp_0, tmp_1, tmp_2, rhs_0, rhs_1, rhs_2); // 質量行列の寄与 rhsは所有節点について正しい値を持つ
        if (force_node >= 0)
        {
            if (force_dof == 0)
            {
#pragma acc update self(rhs_0[force_node : 1])
                rhs_0[force_node] += force_magnitude;
#pragma acc update device(rhs_0[force_node : 1])
            }
            else if (force_dof == 1)
            {
#pragma acc update self(rhs_1[force_node : 1])
                rhs_1[force_node] += force_magnitude;
#pragma acc update device(rhs_1[force_node : 1])
            }
            else if (force_dof == 2)
            {
#pragma acc update self(rhs_2[force_node : 1])
                rhs_2[force_node] += force_magnitude;
#pragma acc update device(rhs_2[force_node : 1])
            }
        }

        apply_bc_to_rhs(num_nodes, bc_flag, bc_val_u, bc_corr_00, bc_corr_01, bc_corr_02, bc_corr_10, bc_corr_11, bc_corr_12, bc_corr_20, bc_corr_21, bc_corr_22, rhs_0, rhs_1, rhs_2);

        int iter = pcg_solve(requests, num_neighbors, neighbor_ranks, recv_starts, recv_counts, send_starts, send_counts, send_nodes_ptr, send_buffer_0, send_buffer_1, send_buffer_2,
                             num_inner, num_owned, num_nodes, bcrs_row_ptr, bcrs_col_ind, bcrs_kval_00, bcrs_kval_01, bcrs_kval_02, bcrs_kval_10, bcrs_kval_11, bcrs_kval_12, bcrs_kval_20,
                             bcrs_kval_21, bcrs_kval_22, inv_diag_00, inv_diag_01, inv_diag_02, inv_diag_10, inv_diag_11, inv_diag_12, inv_diag_20, inv_diag_21, inv_diag_22,
                             rhs_0, rhs_1, rhs_2, u_tmp_0, u_tmp_1, u_tmp_2, 1e-8, num_nodes * 3, r_0, r_1, r_2, z_0, z_1, z_2, p_0, p_1, p_2, Ap_0, Ap_1, Ap_2);
        if (rank == 0)
        {
            std::cout << "Step " << step << ", PCG iterations: " << iter << std::endl;
        }

        // 速度と加速度の更新（Newmark-β法） u,v,aは全節点について正しい値をもつ
#pragma acc parallel loop present(u_tmp_0, u_tmp_1, u_tmp_2, u_prv_0, u_prv_1, u_prv_2, v_tmp_0, v_tmp_1, v_tmp_2, a_tmp_0, a_tmp_1, a_tmp_2, bc_flag, bc_val_u, bc_val_v, bc_val_a)
        for (int i = 0; i < num_nodes; i++)
        {
            double u_new_0 = u_tmp_0[i];
            double u_new_1 = u_tmp_1[i];
            double u_new_2 = u_tmp_2[i];
            double u_old_0 = u_prv_0[i];
            double u_old_1 = u_prv_1[i];
            double u_old_2 = u_prv_2[i];
            double v_old_0 = v_tmp_0[i];
            double v_old_1 = v_tmp_1[i];
            double v_old_2 = v_tmp_2[i];
            double a_old_0 = a_tmp_0[i];
            double a_old_1 = a_tmp_1[i];
            double a_old_2 = a_tmp_2[i];

            double a_new_0 = (u_new_0 - u_old_0) * 4.0 / dt / dt - v_old_0 * 4.0 / dt - a_old_0;
            double a_new_1 = (u_new_1 - u_old_1) * 4.0 / dt / dt - v_old_1 * 4.0 / dt - a_old_1;
            double a_new_2 = (u_new_2 - u_old_2) * 4.0 / dt / dt - v_old_2 * 4.0 / dt - a_old_2;
            double v_new_0 = v_old_0 + (a_new_0 + a_old_0) * dt / 2.0;
            double v_new_1 = v_old_1 + (a_new_1 + a_old_1) * dt / 2.0;
            double v_new_2 = v_old_2 + (a_new_2 + a_old_2) * dt / 2.0;

            if (bc_flag[i])
            {
                u_new_0 = bc_val_u[0];
                u_new_1 = bc_val_u[1];
                u_new_2 = bc_val_u[2];
                v_new_0 = bc_val_v[0];
                v_new_1 = bc_val_v[1];
                v_new_2 = bc_val_v[2];
                a_new_0 = bc_val_a[0];
                a_new_1 = bc_val_a[1];
                a_new_2 = bc_val_a[2];
            }

            u_prv_0[i] = u_new_0;
            u_prv_1[i] = u_new_1;
            u_prv_2[i] = u_new_2;
            v_tmp_0[i] = v_new_0;
            v_tmp_1[i] = v_new_1;
            v_tmp_2[i] = v_new_2;
            a_tmp_0[i] = a_new_0;
            a_tmp_1[i] = a_new_1;
            a_tmp_2[i] = a_new_2;
        }

        // 解xを変位uに保存
        if (step % sample_freq == 0)
        {
            if (target_node >= 0) // 自プロセスに対象ノードがある場合のみ保存
            {
#pragma acc update self(u_tmp_0[target_node : 1], u_tmp_1[target_node : 1], u_tmp_2[target_node : 1]) // 対象ノードの変位をCPUに転送
                u[(step / sample_freq) * 3 + 0] = u_tmp_0[target_node];
                u[(step / sample_freq) * 3 + 1] = u_tmp_1[target_node];
                u[(step / sample_freq) * 3 + 2] = u_tmp_2[target_node];
            }
        }
    }

    data_transfer_start = MPI_Wtime();
    data_transfer_end = MPI_Wtime();
    printf("Data transfer from GPU time: %.2f seconds\n", data_transfer_end - data_transfer_start);

    if (target_node >= 0)
    {
        // --- 出力前にディレクトリ作成 ---
        // タイムスタンプ生成
        time_t now = time(nullptr);
        struct tm *lt = localtime(&now);
        char timestamp[64];
        strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", lt);

        // results/20260323_143025/ のようなパスを作成
        char output_dir[256];
        sprintf(output_dir, "results/%s", timestamp);

        mkdir("results", 0755);  // 親ディレクトリ
        mkdir(output_dir, 0755); // タイムスタンプディレクトリ

        // 各ステップのVTKを出力
        // for (int step = 0; step <= num_steps; step += sample_freq)
        // {
        //     char filename[512];
        //     sprintf(filename, "%s/result_%04d.vtk", output_dir, step / sample_freq);
        //     write_vtk_displacement(filename, node_coords, num_nodes,
        //                            ele_nodes, num_elements, u[step / sample_freq],
        //                            step * dt);
        // }

        char csv_filename[512];
        sprintf(csv_filename, "%s/target_disp.csv", output_dir);
        write_node_disp_csv(csv_filename, u, num_steps, sample_freq, dt);

        printf("Output: %s/\n", output_dir);
        printf("CSV: %s\n", csv_filename);
    }

    double end_time = MPI_Wtime();

    double elapsed_time = end_time - start_time;
    double max_elapsed_time;
    MPI_Reduce(&elapsed_time, &max_elapsed_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("Total elapsed time: %.2f seconds\n", max_elapsed_time);
    }

    MPI_Finalize();

    return 0;
}

void calculate_dN(double dN[30], double r, double s, double t)
{
    // 形状関数の微分 dN をガウス点で計算する
    // dN は各節点の形状関数の x,y,z に対する微分を格納
    dN[0] = 4.0 * (r + s + t) - 3.0;
    dN[1] = 4.0 * r - 1.0;
    dN[2] = 0.0;
    dN[3] = 0.0;
    dN[4] = -4.0 * (2 * r + s + t) + 4.0;
    dN[5] = 4.0 * s;
    dN[6] = -4.0 * s;
    dN[7] = -4.0 * t;
    dN[8] = 4.0 * t;
    dN[9] = 0.0;
    dN[10] = 4.0 * (r + s + t) - 3.0;
    dN[11] = 0.0;
    dN[12] = 4.0 * s - 1.0;
    dN[13] = 0.0;
    dN[14] = -4.0 * r;
    dN[15] = 4.0 * r;
    dN[16] = -4.0 * (r + 2 * s + t) + 4.0;
    dN[17] = -4.0 * t;
    dN[18] = 0.0;
    dN[19] = 4.0 * t;
    dN[20] = 4.0 * (r + s + t) - 3.0;
    dN[21] = 0.0;
    dN[22] = 0.0;
    dN[23] = 4.0 * t - 1.0;
    dN[24] = -4.0 * r;
    dN[25] = 0.0;
    dN[26] = -4.0 * s;
    dN[27] = -4.0 * (r + s + 2 * t) + 4.0;
    dN[28] = 4.0 * r;
    dN[29] = 4.0 * s;

    return;
}

void gauss_integrate(double dN0[30], double dN1[30], double dN2[30], double dN3[30])
{
    // ここにガウス積分を実装して、dN0,dN1,dN2,dN3 にガウス点での形状関数の微分を格納するコードを実装
    // dN0,dN1,dN2,dN3 は10節点の各節点の形状関数の x,y,z に対する微分を格納
    double a = (5.0 - std::sqrt(5.0)) / 20.0;
    double b = (5.0 + 3.0 * std::sqrt(5.0)) / 20.0;
    double gauss_points[4][3] = {
        {a, a, a},
        {b, a, a},
        {a, b, a},
        {a, a, b}};

    calculate_dN(dN0, gauss_points[0][0], gauss_points[0][1], gauss_points[0][2]);
    calculate_dN(dN1, gauss_points[1][0], gauss_points[1][1], gauss_points[1][2]);
    calculate_dN(dN2, gauss_points[2][0], gauss_points[2][1], gauss_points[2][2]);
    calculate_dN(dN3, gauss_points[3][0], gauss_points[3][1], gauss_points[3][2]);

    return;
}

double inverse_3_3_mat(double mat[9], double inv_mat[9])
{
    // 3x3行列の逆行列を計算する
    double det_mat = mat[0 * 3 + 0] * (mat[1 * 3 + 1] * mat[2 * 3 + 2] - mat[1 * 3 + 2] * mat[2 * 3 + 1]) +
                     mat[0 * 3 + 1] * (mat[1 * 3 + 2] * mat[2 * 3 + 0] - mat[1 * 3 + 0] * mat[2 * 3 + 2]) +
                     mat[0 * 3 + 2] * (mat[1 * 3 + 0] * mat[2 * 3 + 1] - mat[1 * 3 + 1] * mat[2 * 3 + 0]);
    double inv_det = 1.0 / det_mat;

    inv_mat[0 * 3 + 0] = (mat[1 * 3 + 1] * mat[2 * 3 + 2] - mat[1 * 3 + 2] * mat[2 * 3 + 1]) * inv_det;
    inv_mat[0 * 3 + 1] = (mat[0 * 3 + 2] * mat[2 * 3 + 1] - mat[0 * 3 + 1] * mat[2 * 3 + 2]) * inv_det;
    inv_mat[0 * 3 + 2] = (mat[0 * 3 + 1] * mat[1 * 3 + 2] - mat[0 * 3 + 2] * mat[1 * 3 + 1]) * inv_det;
    inv_mat[1 * 3 + 0] = (mat[1 * 3 + 2] * mat[2 * 3 + 0] - mat[1 * 3 + 0] * mat[2 * 3 + 2]) * inv_det;
    inv_mat[1 * 3 + 1] = (mat[0 * 3 + 0] * mat[2 * 3 + 2] - mat[0 * 3 + 2] * mat[2 * 3 + 0]) * inv_det;
    inv_mat[1 * 3 + 2] = (mat[0 * 3 + 2] * mat[1 * 3 + 0] - mat[0 * 3 + 0] * mat[1 * 3 + 2]) * inv_det;
    inv_mat[2 * 3 + 0] = (mat[1 * 3 + 0] * mat[2 * 3 + 1] - mat[1 * 3 + 1] * mat[2 * 3 + 0]) * inv_det;
    inv_mat[2 * 3 + 1] = (mat[0 * 3 + 1] * mat[2 * 3 + 0] - mat[0 * 3 + 0] * mat[2 * 3 + 1]) * inv_det;
    inv_mat[2 * 3 + 2] = (mat[0 * 3 + 0] * mat[1 * 3 + 1] - mat[0 * 3 + 1] * mat[1 * 3 + 0]) * inv_det;

    return det_mat;
}

void construct_mat(
    double *node_coords,
    int *ele_nodes,
    int num_elements,
    double dN0[30],
    double dN1[30],
    double dN2[30],
    double dN3[30],
    double lambda,
    double mu,
    double rho,
    double dt,
    double *kmat_coo_val,
    double *mmat_coo_val,
    int *coo_row,
    int *coo_col)
{
    // ここに要素剛性行列 kmat_coo を構築するコードを実装
    // node_coords: [num_nodes * 3] 連続配置: x0,y0,z0,x1,y1,z1,...
    // ele_nodes: [num_elements * 10] 連続配置: 要素0の10節点, 要素1の10節点,...
    // kmat_coo は [100 * num_elements][3][3] の配列で、各要素の行列を連続配置で格納

#pragma omp parallel for
    for (int elem = 0; elem < num_elements; elem++)
    {
        int local_node_indices[10]; // 要素の10節点のインデックスを格納する配列
        double local_coords[4 * 3]; // 4頂点の座標を格納する配列
        double jacobian[9];         // ヤコビアン行列を格納する配列
        double det_jacobian;        // ヤコビアンの行列式を格納する変数
        double inv_jacobian[9];     // ヤコビアンの逆行列を格納する配列
        double local_dN0[30] = {0}; // ガウス点での形状関数の微分の配列
        double local_dN1[30] = {0}; // ガウス点での形状関数の微分の配列
        double local_dN2[30] = {0}; // ガウス点での形状関数の微分の配列
        double local_dN3[30] = {0}; // ガウス点での形状関数の微分の配列
        double local_mmat[100] = {
            6.0 / 2520.0, 1.0 / 2520.0, 1.0 / 2520.0, 1.0 / 2520.0, -4.0 / 2520.0, -6.0 / 2520.0, -4.0 / 2520.0, -4.0 / 2520.0, -6.0 / 2520.0, -6.0 / 2520.0,
            1.0 / 2520.0, 6.0 / 2520.0, 1.0 / 2520.0, 1.0 / 2520.0, -4.0 / 2520.0, -4.0 / 2520.0, -6.0 / 2520.0, -6.0 / 2520.0, -4.0 / 2520.0, -6.0 / 2520.0,
            1.0 / 2520.0, 1.0 / 2520.0, 6.0 / 2520.0, 1.0 / 2520.0, -6.0 / 2520.0, -4.0 / 2520.0, -4.0 / 2520.0, -6.0 / 2520.0, -6.0 / 2520.0, -4.0 / 2520.0,
            1.0 / 2520.0, 1.0 / 2520.0, 1.0 / 2520.0, 6.0 / 2520.0, -6.0 / 2520.0, -6.0 / 2520.0, -6.0 / 2520.0, -4.0 / 2520.0, -4.0 / 2520.0, -4.0 / 2520.0,
            -4.0 / 2520.0, -4.0 / 2520.0, -6.0 / 2520.0, -6.0 / 2520.0, 32.0 / 2520.0, 16.0 / 2520.0, 16.0 / 2520.0, 16.0 / 2520.0, 16.0 / 2520.0, 8.0 / 2520.0,
            -6.0 / 2520.0, -4.0 / 2520.0, -4.0 / 2520.0, -6.0 / 2520.0, 16.0 / 2520.0, 32.0 / 2520.0, 16.0 / 2520.0, 8.0 / 2520.0, 16.0 / 2520.0, 16.0 / 2520.0,
            -4.0 / 2520.0, -6.0 / 2520.0, -4.0 / 2520.0, -6.0 / 2520.0, 16.0 / 2520.0, 16.0 / 2520.0, 32.0 / 2520.0, 16.0 / 2520.0, 8.0 / 2520.0, 16.0 / 2520.0,
            -4.0 / 2520.0, -6.0 / 2520.0, -6.0 / 2520.0, -4.0 / 2520.0, 16.0 / 2520.0, 8.0 / 2520.0, 16.0 / 2520.0, 32.0 / 2520.0, 16.0 / 2520.0, 16.0 / 2520.0,
            -6.0 / 2520.0, -4.0 / 2520.0, -6.0 / 2520.0, -4.0 / 2520.0, 16.0 / 2520.0, 16.0 / 2520.0, 8.0 / 2520.0, 16.0 / 2520.0, 32.0 / 2520.0, 16.0 / 2520.0,
            -6.0 / 2520.0, -6.0 / 2520.0, -4.0 / 2520.0, -4.0 / 2520.0, 8.0 / 2520.0, 16.0 / 2520.0, 16.0 / 2520.0, 16.0 / 2520.0, 16.0 / 2520.0, 32.0 / 2520.0};

        for (int i = 0; i < 10; i++)
        {
            local_node_indices[i] = ele_nodes[elem * 10 + i];
        }
        for (int i = 0; i < 4; i++)
        {
            local_coords[3 * i + 0] = node_coords[local_node_indices[i] * 3 + 0];
            local_coords[3 * i + 1] = node_coords[local_node_indices[i] * 3 + 1];
            local_coords[3 * i + 2] = node_coords[local_node_indices[i] * 3 + 2];
        }

        for (int i = 1; i < 4; i++)
        {
            jacobian[3 * (i - 1) + 0] = local_coords[3 * i + 0] - local_coords[0];
            jacobian[3 * (i - 1) + 1] = local_coords[3 * i + 1] - local_coords[1];
            jacobian[3 * (i - 1) + 2] = local_coords[3 * i + 2] - local_coords[2];
        }
        det_jacobian = inverse_3_3_mat(jacobian, inv_jacobian);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 10; j++)
            {
                local_dN0[10 * i + j] = inv_jacobian[3 * i + 0] * dN0[j] + inv_jacobian[3 * i + 1] * dN0[10 + j] + inv_jacobian[3 * i + 2] * dN0[20 + j];
                local_dN1[10 * i + j] = inv_jacobian[3 * i + 0] * dN1[j] + inv_jacobian[3 * i + 1] * dN1[10 + j] + inv_jacobian[3 * i + 2] * dN1[20 + j];
                local_dN2[10 * i + j] = inv_jacobian[3 * i + 0] * dN2[j] + inv_jacobian[3 * i + 1] * dN2[10 + j] + inv_jacobian[3 * i + 2] * dN2[20 + j];
                local_dN3[10 * i + j] = inv_jacobian[3 * i + 0] * dN3[j] + inv_jacobian[3 * i + 1] * dN3[10 + j] + inv_jacobian[3 * i + 2] * dN3[20 + j];
            }
        }

        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < 10; j++)
            {
                double local_kmat[3][3] = {0};
                for (int a = 0; a < 3; a++)
                {
                    for (int b = 0; b < 3; b++)
                    {
                        local_kmat[a][b] = lambda * local_dN0[10 * a + i] * local_dN0[10 * b + j] + mu * local_dN0[10 * a + j] * local_dN0[10 * b + i];
                        local_kmat[a][b] += lambda * local_dN1[10 * a + i] * local_dN1[10 * b + j] + mu * local_dN1[10 * a + j] * local_dN1[10 * b + i];
                        local_kmat[a][b] += lambda * local_dN2[10 * a + i] * local_dN2[10 * b + j] + mu * local_dN2[10 * a + j] * local_dN2[10 * b + i];
                        local_kmat[a][b] += lambda * local_dN3[10 * a + i] * local_dN3[10 * b + j] + mu * local_dN3[10 * a + j] * local_dN3[10 * b + i];
                    }
                }
                double dot_product0 = local_dN0[10 * 0 + i] * local_dN0[10 * 0 + j] + local_dN0[10 * 1 + i] * local_dN0[10 * 1 + j] + local_dN0[10 * 2 + i] * local_dN0[10 * 2 + j];
                double dot_product1 = local_dN1[10 * 0 + i] * local_dN1[10 * 0 + j] + local_dN1[10 * 1 + i] * local_dN1[10 * 1 + j] + local_dN1[10 * 2 + i] * local_dN1[10 * 2 + j];
                double dot_product2 = local_dN2[10 * 0 + i] * local_dN2[10 * 0 + j] + local_dN2[10 * 1 + i] * local_dN2[10 * 1 + j] + local_dN2[10 * 2 + i] * local_dN2[10 * 2 + j];
                double dot_product3 = local_dN3[10 * 0 + i] * local_dN3[10 * 0 + j] + local_dN3[10 * 1 + i] * local_dN3[10 * 1 + j] + local_dN3[10 * 2 + i] * local_dN3[10 * 2 + j];
                for (int a = 0; a < 3; a++)
                {
                    local_kmat[a][a] += mu * dot_product0;
                    local_kmat[a][a] += mu * dot_product1;
                    local_kmat[a][a] += mu * dot_product2;
                    local_kmat[a][a] += mu * dot_product3;
                    local_kmat[a][a] += 24.0 * rho * local_mmat[10 * i + j] * 4.0 / dt / dt; // 質量行列の寄与を剛性行列に加算（Newmark-β法のため）
                }
                for (int a = 0; a < 3; a++)
                {
                    for (int b = 0; b < 3; b++)
                    {
                        kmat_coo_val[9 * (elem * 100 + i * 10 + j) + 3 * a + b] = local_kmat[a][b] * det_jacobian / 24.0; // 24.0は4点のガウス積分の重み
                    }
                }
                mmat_coo_val[elem * 100 + i * 10 + j] = rho * local_mmat[10 * i + j] * det_jacobian; // 質量行列の値を格納
                coo_row[elem * 100 + i * 10 + j] = local_node_indices[i];
                coo_col[elem * 100 + i * 10 + j] = local_node_indices[j];
            }
        }
    }
}

int sort_and_merge_bcoo(
    int nnz_coo,
    int num_nodes,
    int *coo_row,
    int *coo_col,
    double *kmat_coo_val,
    double *mmat_coo_val)
{
    // ブロックCOOを並び替え＋重複足し合わせ
    // 行ごとのエントリ数カウント → 累積和 → バケットソート → 行内で列ソート（挿入ソート）＋重複足し合わせ

    int *offset = new int[num_nodes + 1]();
    for (int k = 0; k < nnz_coo; k++)
    {
        offset[coo_row[k] + 1]++;
    }
    for (int i = 1; i <= num_nodes; i++)
    {
        offset[i] += offset[i - 1];
    }

    int *work_col = new int[nnz_coo];
    double *work_kval = new double[nnz_coo * 9];
    double *work_mval = new double[nnz_coo];

    int *pos = new int[num_nodes];
#pragma omp parallel for
    for (int i = 0; i < num_nodes; i++)
    {
        pos[i] = offset[i];
    }

    for (int k = 0; k < nnz_coo; k++)
    {
        int r = coo_row[k];
        int p = pos[r]++;
        work_col[p] = coo_col[k];
        for (int a = 0; a < 9; a++)
            work_kval[9 * p + a] = kmat_coo_val[9 * k + a];
        work_mval[p] = mmat_coo_val[k];
    }
    delete[] pos;

    int nnz_bcrs = 0;

    for (int i = 0; i < num_nodes; i++)
    {
        int start = offset[i];
        int end = offset[i + 1];

        if (start == end)
            continue;

        /* 挿入ソート（列番号で昇順） */
        for (int a = start + 1; a < end; a++)
        {
            int tmp_col = work_col[a];
            double tmp_kval[9];
            double tmp_mval = work_mval[a];
            for (int p = 0; p < 9; p++)
                tmp_kval[p] = work_kval[9 * a + p];

            int b = a - 1;
            while (b >= start && work_col[b] > tmp_col)
            {
                work_col[b + 1] = work_col[b];
                for (int p = 0; p < 9; p++)
                    work_kval[9 * (b + 1) + p] = work_kval[9 * b + p];
                work_mval[b + 1] = work_mval[b];
                b--;
            }
            work_col[b + 1] = tmp_col;
            for (int p = 0; p < 9; p++)
                work_kval[9 * (b + 1) + p] = tmp_kval[p];
            work_mval[b + 1] = tmp_mval;
        }

        /* 重複列を足し合わせながらCOO配列の先頭に詰める */
        int write = nnz_bcrs;
        coo_row[write] = i;
        coo_col[write] = work_col[start];
        for (int a = 0; a < 9; a++)
            kmat_coo_val[9 * write + a] = work_kval[9 * start + a];
        mmat_coo_val[write] = work_mval[start];

        for (int k = start + 1; k < end; k++)
        {
            if (work_col[k] == coo_col[write])
            {
                for (int a = 0; a < 9; a++)
                    kmat_coo_val[9 * write + a] += work_kval[9 * k + a];
                mmat_coo_val[write] += work_mval[k];
            }
            else
            {
                write++;
                coo_row[write] = i;
                coo_col[write] = work_col[k];
                for (int a = 0; a < 9; a++)
                    kmat_coo_val[9 * write + a] = work_kval[9 * k + a];
                mmat_coo_val[write] = work_mval[k];
            }
        }
        write++;
        nnz_bcrs = write;
    }

    delete[] work_col;
    delete[] work_kval;
    delete[] work_mval;
    delete[] offset;

    return nnz_bcrs;
}

void build_bcrs(
    int *coo_row,
    int *coo_col,
    double *kmat_coo_val,
    double *mmat_coo_val,
    int nnz_bcrs,
    int num_nodes,
    int *bcrs_row_ptr,
    int *bcrs_col_ind,
    double *bcrs_kval_00,
    double *bcrs_kval_01,
    double *bcrs_kval_02,
    double *bcrs_kval_10,
    double *bcrs_kval_11,
    double *bcrs_kval_12,
    double *bcrs_kval_20,
    double *bcrs_kval_21,
    double *bcrs_kval_22,
    double *bcrs_mval)
{
    // 並び替え済みCOOからbcrsを構築 00成分が0~nnz_bcrs-1、01成分がnnz_bcrs~2*nnz_bcrs-1、02成分が2*nnz_bcrs~3*nnz_bcrs-1、...という配置で、bcrs_kvalに9成分ずつ格納されているとする。
    // col_ind, kval, mval をコピー
#pragma omp parallel for
    for (int k = 0; k < nnz_bcrs; k++)
    {
        bcrs_col_ind[k] = coo_col[k];
        bcrs_kval_00[k] = kmat_coo_val[9 * k + 0];
        bcrs_kval_01[k] = kmat_coo_val[9 * k + 1];
        bcrs_kval_02[k] = kmat_coo_val[9 * k + 2];
        bcrs_kval_10[k] = kmat_coo_val[9 * k + 3];
        bcrs_kval_11[k] = kmat_coo_val[9 * k + 4];
        bcrs_kval_12[k] = kmat_coo_val[9 * k + 5];
        bcrs_kval_20[k] = kmat_coo_val[9 * k + 6];
        bcrs_kval_21[k] = kmat_coo_val[9 * k + 7];
        bcrs_kval_22[k] = kmat_coo_val[9 * k + 8];
        bcrs_mval[k] = mmat_coo_val[k];
    }

    // row_ptr を構築
#pragma omp parallel for
    for (int i = 0; i <= num_nodes; i++)
    {
        bcrs_row_ptr[i] = 0;
    }
    for (int k = 0; k < nnz_bcrs; k++)
    {
        bcrs_row_ptr[coo_row[k] + 1]++;
    }
    for (int i = 1; i <= num_nodes; i++)
    {
        bcrs_row_ptr[i] += bcrs_row_ptr[i - 1];
    }
}

void bcrs_spmv_m(
    int num_nodes,
    int *row_ptr,
    int *col_ind,
    double *val,
    double *x_0,
    double *x_1,
    double *x_2,
    double *y_0,
    double *y_1,
    double *y_2)
{
    // bcrs 行列ベクトル積: y = A * x
#pragma acc parallel loop present(row_ptr, col_ind, val, x_0, x_1, x_2, y_0, y_1, y_2)
    for (int i = 0; i < num_nodes; i++)
    {
        double y0 = 0.0, y1 = 0.0, y2 = 0.0;
#pragma acc loop seq
        for (int p = row_ptr[i]; p < row_ptr[i + 1]; p++)
        {
            int j = col_ind[p];
            y0 += val[p] * x_0[j];
            y1 += val[p] * x_1[j];
            y2 += val[p] * x_2[j];
        }
        y_0[i] = y0;
        y_1[i] = y1;
        y_2[i] = y2;
    }
}

void extract_bc_correction(
    int num_nodes,
    int *row_ptr,
    int *col_ind,
    double *kval_00,
    double *kval_01,
    double *kval_02,
    double *kval_10,
    double *kval_11,
    double *kval_12,
    double *kval_20,
    double *kval_21,
    double *kval_22,
    int *bc_flag,
    double *bc_corr_00,
    double *bc_corr_01,
    double *bc_corr_02,
    double *bc_corr_10,
    double *bc_corr_11,
    double *bc_corr_12,
    double *bc_corr_20,
    double *bc_corr_21,
    double *bc_corr_22)
{
    // 境界条件の補正ベクトルを抽出
    // 自由節点iと拘束節点jの間のカップリング A[i][j] を抽出する。
    // 毎タイムステップの右辺ベクトル補正に使う。
    // bc_corr[num_nodes * 3][3]: bc_corr[i*3+a][b] = Σ_{j∈constrained} A_original[i][j][a][b]
#pragma omp parallel for
    for (int i = 0; i < num_nodes; i++)
        bc_corr_00[i] = bc_corr_01[i] = bc_corr_02[i] = bc_corr_10[i] = bc_corr_11[i] = bc_corr_12[i] = bc_corr_20[i] = bc_corr_21[i] = bc_corr_22[i] = 0.0;

#pragma omp parallel for
    for (int i = 0; i < num_nodes; i++)
    {
        if (bc_flag[i])
            continue;

        for (int p = row_ptr[i]; p < row_ptr[i + 1]; p++)
        {
            int j = col_ind[p];
            if (!bc_flag[j])
                continue;

            bc_corr_00[i] += kval_00[p];
            bc_corr_01[i] += kval_01[p];
            bc_corr_02[i] += kval_02[p];
            bc_corr_10[i] += kval_10[p];
            bc_corr_11[i] += kval_11[p];
            bc_corr_12[i] += kval_12[p];
            bc_corr_20[i] += kval_20[p];
            bc_corr_21[i] += kval_21[p];
            bc_corr_22[i] += kval_22[p];
        }
    }
}

void apply_bc_to_lhs(
    int num_nodes,
    int *row_ptr,
    int *col_ind,
    double *kval_00,
    double *kval_01,
    double *kval_02,
    double *kval_10,
    double *kval_11,
    double *kval_12,
    double *kval_20,
    double *kval_21,
    double *kval_22,
    int *bc_flag)
{
    // 境界条件を左辺行列に適用（1回だけ呼ぶ）
    // bc_flag[num_nodes]: 0=自由, 1=拘束（節点単位、3方向同時拘束）
#pragma omp parallel for
    for (int i = 0; i < num_nodes; i++)
    {
        for (int p = row_ptr[i]; p < row_ptr[i + 1]; p++)
        {
            int j = col_ind[p];

            if (bc_flag[i] || bc_flag[j])
            {
                kval_00[p] = kval_01[p] = kval_02[p] = kval_10[p] = kval_11[p] = kval_12[p] = kval_20[p] = kval_21[p] = kval_22[p] = 0.0;
            }

            if (bc_flag[i] && i == j)
            {
                kval_00[p] = kval_11[p] = kval_22[p] = 1.0;
            }
        }
    }
}

void apply_bc_to_rhs(
    int num_nodes,
    int *bc_flag,
    double *bc_val,
    double *bc_corr_00,
    double *bc_corr_01,
    double *bc_corr_02,
    double *bc_corr_10,
    double *bc_corr_11,
    double *bc_corr_12,
    double *bc_corr_20,
    double *bc_corr_21,
    double *bc_corr_22,
    double *rhs_0,
    double *rhs_1,
    double *rhs_2)
{
    // 境界条件を右辺ベクトルに適用（毎タイムステップ呼ぶ）
    // bc_val[3]: 拘束変位値（全拘束節点で共通、例: {0, sin(t), 0}）
    // 自由節点の右辺を補正: rhs[i] -= Σ_b bc_corr[i][b] * bc_val[b]
#pragma acc parallel loop present(bc_val, bc_corr_00, bc_corr_01, bc_corr_02, bc_corr_10, bc_corr_11, bc_corr_12, bc_corr_20, bc_corr_21, bc_corr_22, rhs_0, rhs_1, rhs_2)
    for (int i = 0; i < num_nodes; i++)
    {
        rhs_0[i] -= bc_corr_00[i] * bc_val[0] + bc_corr_01[i] * bc_val[1] + bc_corr_02[i] * bc_val[2];
        rhs_1[i] -= bc_corr_10[i] * bc_val[0] + bc_corr_11[i] * bc_val[1] + bc_corr_12[i] * bc_val[2];
        rhs_2[i] -= bc_corr_20[i] * bc_val[0] + bc_corr_21[i] * bc_val[1] + bc_corr_22[i] * bc_val[2];
    }

    // 拘束節点の右辺を拘束値に設定
#pragma acc parallel loop present(bc_flag, bc_val, rhs_0, rhs_1, rhs_2)
    for (int i = 0; i < num_nodes; i++)
    {
        if (bc_flag[i])
        {
            rhs_0[i] = bc_val[0];
            rhs_1[i] = bc_val[1];
            rhs_2[i] = bc_val[2];
        }
    }
}

void build_block_jacobi(
    int num_nodes,
    int *row_ptr,
    int *col_ind,
    double *kval_00,
    double *kval_01,
    double *kval_02,
    double *kval_10,
    double *kval_11,
    double *kval_12,
    double *kval_20,
    double *kval_21,
    double *kval_22,
    double *inv_diag_00,
    double *inv_diag_01,
    double *inv_diag_02,
    double *inv_diag_10,
    double *inv_diag_11,
    double *inv_diag_12,
    double *inv_diag_20,
    double *inv_diag_21,
    double *inv_diag_22)
{
    // ブロックヤコビ前処理の構築（対角ブロックの逆行列を計算）
#pragma omp parallel for
    for (int i = 0; i < num_nodes; i++)
    {
        /* 対角ブロックを探す */
        double d[9] = {};
        double inv_d[9] = {};
        for (int p = row_ptr[i]; p < row_ptr[i + 1]; p++)
        {
            if (col_ind[p] == i)
            {
                d[0] = kval_00[p];
                d[1] = kval_01[p];
                d[2] = kval_02[p];
                d[3] = kval_10[p];
                d[4] = kval_11[p];
                d[5] = kval_12[p];
                d[6] = kval_20[p];
                d[7] = kval_21[p];
                d[8] = kval_22[p];
                break;
            }
        }
        inverse_3_3_mat(d, inv_d);
        inv_diag_00[i] = inv_d[0];
        inv_diag_01[i] = inv_d[1];
        inv_diag_02[i] = inv_d[2];
        inv_diag_10[i] = inv_d[3];
        inv_diag_11[i] = inv_d[4];
        inv_diag_12[i] = inv_d[5];
        inv_diag_20[i] = inv_d[6];
        inv_diag_21[i] = inv_d[7];
        inv_diag_22[i] = inv_d[8];
    }
}

double dot(int n, double *a, double *b)
{
    // ベクトル内積
    double s = 0.0;
#pragma omp parallel for reduction(+ : s)
    for (int i = 0; i < n; i++)
    {
        s += a[i] * b[i];
    }
    return s;
}

int pcg_solve(
    MPI_Request *request,
    int num_neighbors,
    int *neighbor_ranks,
    int *recv_starts,
    int *recv_counts,
    int *send_starts,
    int *send_counts,
    int *send_nodes,
    double *send_buffer_0,
    double *send_buffer_1,
    double *send_buffer_2,
    int num_inner,
    int num_owned,
    int num_nodes,
    int *row_ptr,
    int *col_ind,
    double *kval_00,
    double *kval_01,
    double *kval_02,
    double *kval_10,
    double *kval_11,
    double *kval_12,
    double *kval_20,
    double *kval_21,
    double *kval_22,
    double *inv_diag_00,
    double *inv_diag_01,
    double *inv_diag_02,
    double *inv_diag_10,
    double *inv_diag_11,
    double *inv_diag_12,
    double *inv_diag_20,
    double *inv_diag_21,
    double *inv_diag_22,
    double *b_0,
    double *b_1,
    double *b_2,
    double *x_0,
    double *x_1,
    double *x_2,
    double tol,
    int max_iter,
    double *r_0,
    double *r_1,
    double *r_2,
    double *z_0,
    double *z_1,
    double *z_2,
    double *p_0,
    double *p_1,
    double *p_2,
    double *Ap_0,
    double *Ap_1,
    double *Ap_2)
{
    // Ax = b を前処理付き共役勾配法で解く。
    // x は初期解を入れて呼ぶ（ゼロでもよい）。解が上書きされる。
    double rz = 0.0;
    double b_norm = 0.0;
    double r_norm = 0.0;

#pragma acc parallel loop reduction(+ : rz, b_norm, r_norm) present(row_ptr, col_ind, kval_00, kval_01, kval_02, kval_10, kval_11, kval_12, kval_20, kval_21, kval_22,      \
                                                                    inv_diag_00, inv_diag_01, inv_diag_02, inv_diag_10, inv_diag_11, inv_diag_12, inv_diag_20, inv_diag_21, \
                                                                    inv_diag_22, b_0, b_1, b_2, x_0, x_1, x_2, r_0, r_1, r_2,                                               \
                                                                    z_0, z_1, z_2, p_0, p_1, p_2, Ap_0, Ap_1, Ap_2)
    for (int i = 0; i < num_owned; i++)
    {
        // r = b - A*x　r,bは所有節点について、xは全節点について正しい値が入っている
        double y0 = 0.0, y1 = 0.0, y2 = 0.0;
#pragma acc loop seq
        for (int p_ = row_ptr[i]; p_ < row_ptr[i + 1]; p_++)
        {
            int j = col_ind[p_];
            double x0 = x_0[j];
            double x1 = x_1[j];
            double x2 = x_2[j];
            y0 += kval_00[p_] * x0 + kval_01[p_] * x1 + kval_02[p_] * x2;
            y1 += kval_10[p_] * x0 + kval_11[p_] * x1 + kval_12[p_] * x2;
            y2 += kval_20[p_] * x0 + kval_21[p_] * x1 + kval_22[p_] * x2;
        }
        Ap_0[i] = y0;
        Ap_1[i] = y1;
        Ap_2[i] = y2;
        double b0 = b_0[i];
        double b1 = b_1[i];
        double b2 = b_2[i];
        double r0 = b0 - y0;
        double r1 = b1 - y1;
        double r2 = b2 - y2;
        r_0[i] = r0;
        r_1[i] = r1;
        r_2[i] = r2;

        // z = C⁻¹r zは所有節点について正しい値が入る
        double z0 = inv_diag_00[i] * r0 + inv_diag_01[i] * r1 + inv_diag_02[i] * r2;
        double z1 = inv_diag_10[i] * r0 + inv_diag_11[i] * r1 + inv_diag_12[i] * r2;
        double z2 = inv_diag_20[i] * r0 + inv_diag_21[i] * r1 + inv_diag_22[i] * r2;
        z_0[i] = z0;
        z_1[i] = z1;
        z_2[i] = z2;

        // p = z pは所有節点について正しい値が入る
        p_0[i] = z0;
        p_1[i] = z1;
        p_2[i] = z2;

        // rz = r · z
        rz += r0 * z0 + r1 * z1 + r2 * z2;

        // bのノルム
        b_norm += b0 * b0 + b1 * b1 + b2 * b2;

        // rのノルム
        r_norm += r0 * r0 + r1 * r1 + r2 * r2;
    }

    MPI_Iallreduce(MPI_IN_PLACE, &rz, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request[0]);
    MPI_Iallreduce(MPI_IN_PLACE, &b_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request[1]);
    MPI_Iallreduce(MPI_IN_PLACE, &r_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request[2]);
    MPI_Waitall(3, request, MPI_STATUSES_IGNORE);

    if (b_norm == 0.0)
        b_norm = 1.0;

    if (r_norm / b_norm < tol * tol)
    {
        return 0;
    }

    int iter;
    for (iter = 0; iter < max_iter; iter++)
    {
        // pは全節点について正しい必要があるので、GPU間でゴースト節点を送受信する。
        for (int n = 0; n < num_neighbors; n++)
        {
            int send_start = send_starts[n];
            // 送信用にまずはバッファにコピー
#pragma acc parallel loop present(p_0, p_1, p_2, send_buffer_0, send_buffer_1, send_buffer_2, send_nodes)
            for (int i = 0; i < send_counts[n]; i++)
            {
                int node = send_nodes[send_start + i];
                send_buffer_0[send_start + i] = p_0[node];
                send_buffer_1[send_start + i] = p_1[node];
                send_buffer_2[send_start + i] = p_2[node];
            }

#pragma acc host_data use_device(send_buffer_0, send_buffer_1, send_buffer_2, p_0, p_1, p_2)
            {
                MPI_Isend(&send_buffer_0[send_start], send_counts[n], MPI_DOUBLE, neighbor_ranks[n], 0, MPI_COMM_WORLD, &request[n]);
                MPI_Isend(&send_buffer_1[send_start], send_counts[n], MPI_DOUBLE, neighbor_ranks[n], 0, MPI_COMM_WORLD, &request[num_neighbors + n]);
                MPI_Isend(&send_buffer_2[send_start], send_counts[n], MPI_DOUBLE, neighbor_ranks[n], 0, MPI_COMM_WORLD, &request[2 * num_neighbors + n]);
                MPI_Irecv(&p_0[recv_starts[n]], recv_counts[n], MPI_DOUBLE, neighbor_ranks[n], 0, MPI_COMM_WORLD, &request[num_neighbors + n]);
                MPI_Irecv(&p_1[recv_starts[n]], recv_counts[n], MPI_DOUBLE, neighbor_ranks[n], 0, MPI_COMM_WORLD, &request[2 * num_neighbors + n]);
                MPI_Irecv(&p_2[recv_starts[n]], recv_counts[n], MPI_DOUBLE, neighbor_ranks[n], 0, MPI_COMM_WORLD, &request[3 * num_neighbors + n]);
            }
        }

        double pAp = 0.0;
        // 通信が不要な内側の節点については先に計算しておく
#pragma acc parallel loop present(row_ptr, col_ind, kval_00, kval_01, kval_02, kval_10, kval_11, kval_12, kval_20, kval_21, kval_22, p_0, p_1, p_2, Ap_0, Ap_1, Ap_2) gang
        for (int i = 0; i < num_inner; i++)
        {
            // Ap = A * p Apは所有節点について正しい値が入る
            double y0 = 0.0, y1 = 0.0, y2 = 0.0;
#pragma acc loop vector reduction(+ : y0, y1, y2)
            for (int p_ = row_ptr[i]; p_ < row_ptr[i + 1]; p_++)
            {
                int j = col_ind[p_];
                y0 += kval_00[p_] * p_0[j] + kval_01[p_] * p_1[j] + kval_02[p_] * p_2[j];
                y1 += kval_10[p_] * p_0[j] + kval_11[p_] * p_1[j] + kval_12[p_] * p_2[j];
                y2 += kval_20[p_] * p_0[j] + kval_21[p_] * p_1[j] + kval_22[p_] * p_2[j];
            }
            Ap_0[i] = y0;
            Ap_1[i] = y1;
            Ap_2[i] = y2;
        }

#pragma acc parallel loop reduction(+ : pAp) present(p_0, p_1, p_2, Ap_0, Ap_1, Ap_2)
        for (int i = 0; i < num_inner; i++)
        {
            // pAp = p · Ap
            pAp += p_0[i] * Ap_0[i] + p_1[i] * Ap_1[i] + p_2[i] * Ap_2[i];
        }

        // ゴースト節点を待つ。pは全節点について正しい値が入る
        MPI_Waitall(6 * num_neighbors, request, MPI_STATUSES_IGNORE);

        // 通信が必要な外側の節点について計算
#pragma acc parallel loop reduction(+ : pAp) present(row_ptr, col_ind, kval_00, kval_01, kval_02, kval_10, kval_11, kval_12, kval_20, kval_21, kval_22, p_0, p_1, p_2, Ap_0, Ap_1, Ap_2)
        for (int i = num_inner; i < num_owned; i++)
        {
            // Ap = A * p
            double y0 = 0.0, y1 = 0.0, y2 = 0.0;
#pragma acc loop seq
            for (int p_ = row_ptr[i]; p_ < row_ptr[i + 1]; p_++)
            {
                int j = col_ind[p_];
                y0 += kval_00[p_] * p_0[j] + kval_01[p_] * p_1[j] + kval_02[p_] * p_2[j];
                y1 += kval_10[p_] * p_0[j] + kval_11[p_] * p_1[j] + kval_12[p_] * p_2[j];
                y2 += kval_20[p_] * p_0[j] + kval_21[p_] * p_1[j] + kval_22[p_] * p_2[j];
            }
            Ap_0[i] = y0;
            Ap_1[i] = y1;
            Ap_2[i] = y2;

            // pAp = p · Ap
            pAp += p_0[i] * y0 + p_1[i] * y1 + p_2[i] * y2;
        }

        MPI_Allreduce(MPI_IN_PLACE, &pAp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // alpha = rz / (p · Ap)
        double alpha = rz / pAp;

#pragma acc parallel loop present(x_0, x_1, x_2, p_0, p_1, p_2)
        for (int i = 0; i < num_nodes; i++)
        {
            // xの更新
            // x += alpha * p xは全節点について正しい値が入る
            x_0[i] += alpha * p_0[i];
            x_1[i] += alpha * p_1[i];
            x_2[i] += alpha * p_2[i];
        }

        r_norm = 0.0;
#pragma acc parallel loop reduction(+ : r_norm) present(r_0, r_1, r_2, Ap_0, Ap_1, Ap_2)
        for (int i = 0; i < num_owned; i++)
        {
            // r -= alpha * Ap rは所有節点について正しい値が入る
            double r_i_0 = r_0[i] - alpha * Ap_0[i];
            double r_i_1 = r_1[i] - alpha * Ap_1[i];
            double r_i_2 = r_2[i] - alpha * Ap_2[i];
            r_0[i] = r_i_0;
            r_1[i] = r_i_1;
            r_2[i] = r_i_2;

            // rのノルム
            r_norm += r_i_0 * r_i_0 + r_i_1 * r_i_1 + r_i_2 * r_i_2;
        }

        MPI_Allreduce(MPI_IN_PLACE, &r_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // 収束判定
        if (r_norm / b_norm < tol * tol)
        {
            iter++;
            break;
        }

        double rz_new = 0.0;
#pragma acc parallel loop reduction(+ : rz_new) present(r_0, r_1, r_2, z_0, z_1, z_2, \
                                                        inv_diag_00, inv_diag_01, inv_diag_02, inv_diag_10, inv_diag_11, inv_diag_12, inv_diag_20, inv_diag_21, inv_diag_22)
        for (int i = 0; i < num_owned; i++)
        {
            // z = C⁻¹r zは所有節点について正しい値が入る
            double r0 = r_0[i];
            double r1 = r_1[i];
            double r2 = r_2[i];
            double z0 = inv_diag_00[i] * r0 + inv_diag_01[i] * r1 + inv_diag_02[i] * r2;
            double z1 = inv_diag_10[i] * r0 + inv_diag_11[i] * r1 + inv_diag_12[i] * r2;
            double z2 = inv_diag_20[i] * r0 + inv_diag_21[i] * r1 + inv_diag_22[i] * r2;
            z_0[i] = z0;
            z_1[i] = z1;
            z_2[i] = z2;

            // rz_new = r · z
            rz_new += r0 * z0 + r1 * z1 + r2 * z2;
        }

        MPI_Allreduce(MPI_IN_PLACE, &rz_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // beta = rz_new / rz
        double beta = rz_new / rz;

        // p = z + beta * p pは所有節点について正しい値が入る
#pragma acc parallel loop present(p_0, p_1, p_2, z_0, z_1, z_2)
        for (int i = 0; i < num_owned; i++)
        {
            p_0[i] = z_0[i] + beta * p_0[i];
            p_1[i] = z_1[i] + beta * p_1[i];
            p_2[i] = z_2[i] + beta * p_2[i];
        }
        rz = rz_new;
    }

    return iter;
}

void write_vtk_displacement(
    const char *filename,
    double *node_coords,
    int num_nodes,
    int *ele_nodes,
    int num_elements,
    double *displacement,
    double total_time)
{
    // 1タイムステップ分のVTKファイルを出力
    FILE *fp = fopen(filename, "w");

    /* ヘッダ */
    fprintf(fp, "# vtk DataFile Version 2.0\n");
    fprintf(fp, "FEM displacement result\n");
    fprintf(fp, "ASCII\n");
    fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");

    /* 節点座標 */
    fprintf(fp, "\nFIELD FieldData 1\n");
    fprintf(fp, "TOTALTIME 1 1 double\n");
    fprintf(fp, "%.10e\n", total_time);
    fprintf(fp, "POINTS %d double\n", num_nodes);
    for (int i = 0; i < num_nodes; i++)
    {
        fprintf(fp, "%.15e %.15e %.15e\n",
                node_coords[i * 3 + 0],
                node_coords[i * 3 + 1],
                node_coords[i * 3 + 2]);
    }

    /* セル（10節点四面体） */
    int cells_size = num_elements * 11; /* 各行: ノード数(10) + 10個の節点番号 */
    fprintf(fp, "CELLS %d %d\n", num_elements, cells_size);
    for (int e = 0; e < num_elements; e++)
    {
        fprintf(fp, "10");
        for (int i = 0; i < 10; i++)
        {
            fprintf(fp, " %d", ele_nodes[e * 10 + i]);
        }
        fprintf(fp, "\n");
    }

    /* セルタイプ（24 = VTK_QUADRATIC_TETRA） */
    fprintf(fp, "CELL_TYPES %d\n", num_elements);
    for (int e = 0; e < num_elements; e++)
    {
        fprintf(fp, "24\n");
    }

    /* 変位データ */
    fprintf(fp, "POINT_DATA %d\n", num_nodes);
    fprintf(fp, "SCALARS NODE_ID int 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int i = 0; i < num_nodes; i++)
    {
        fprintf(fp, "%d\n", i + 1); // 1始まりのノードIDを出力
    }
    fprintf(fp, "VECTORS DISPLACEMENT double\n");
    for (int i = 0; i < num_nodes; i++)
    {
        fprintf(fp, "%.15e %.15e %.15e\n",
                displacement[i * 3 + 0],
                displacement[i * 3 + 1],
                displacement[i * 3 + 2]);
    }

    fclose(fp);
}

void write_node_disp_csv(
    const char *filename,
    double *u,
    int num_steps,
    int sample_freq,
    double dt)
{
    FILE *csv_fp = fopen(filename, "w");
    fprintf(csv_fp, "time,disp_x,disp_y,disp_z,disp_mag\n");
    for (int step = 0; step <= num_steps; step += sample_freq)
    {
        double t = step * dt;
        int idx = step / sample_freq;
        // 配列にはターゲットノードのデータしかないので直接アクセスする
        double ux = u[idx * 3 + 0];
        double uy = u[idx * 3 + 1];
        double uz = u[idx * 3 + 2];
        double mag = sqrt(ux * ux + uy * uy + uz * uz);
        fprintf(csv_fp, "%.10e,%.10e,%.10e,%.10e,%.10e\n", t, ux, uy, uz, mag);
    }
    fclose(csv_fp);
}