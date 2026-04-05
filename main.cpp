/*実行コマンド
cd /data3/kusumoto/elastodynamics-fem/
g++ main.cpp vtk_reader.cpp -fopenmp
./a.out
*/

#include "vtk_reader.hpp"
#include <iostream>
#include <cmath>
#include <ctime>
#include <sys/stat.h>
#include <omp.h>

void calculate_dN(double dN[3][10], double r, double s, double t);
void gauss_integrate(double dN0[3][10], double dN1[3][10], double dN2[3][10], double dN3[3][10]);
double inverse_3_3_mat(double mat[3][3], double inv_mat[3][3]);
void construct_mat(
    double *node_coords,
    int *ele_nodes,
    int num_elements,
    double dN0[3][10],
    double dN1[3][10],
    double dN2[3][10],
    double dN3[3][10],
    double lambda,
    double mu,
    double rho,
    double dt,
    double (*kmat_coo_val)[3][3],
    double *mmat_coo_val,
    int *coo_row,
    int *coo_col);
int sort_and_merge_bcoo(
    int nnz_coo,
    int num_nodes,
    int *coo_row,
    int *coo_col,
    double (*kmat_coo_val)[3][3],
    double *mmat_coo_val);
void build_bcrs(
    int *coo_row,
    int *coo_col,
    double (*kmat_coo_val)[3][3],
    double *mmat_coo_val,
    int nnz_bcoo,
    int num_nodes,
    int *bcrs_row_ptr,
    int *bcrs_col_ind,
    double (*bcrs_kval)[3][3],
    double *bcrs_mval);
void extract_bc_correction(
    int num_nodes,
    int *bcrs_row_ptr,
    int *bcrs_col_ind,
    double (*bcrs_kval)[3][3],
    int *bc_flag,
    double (*bc_corr)[3]);
void apply_bc_to_lhs(
    int num_nodes,
    int *bcrs_row_ptr,
    int *bcrs_col_ind,
    double (*bcrs_kval)[3][3],
    int *bc_flag);
void apply_bc_to_rhs(
    int num_nodes,
    int *bc_flag,
    double *bc_val,
    double (*bc_corr)[3],
    double *rhs);
void build_block_jacobi(
    int num_nodes,
    int *bcrs_row_ptr,
    int *bcrs_col_ind,
    double (*bcrs_kval)[3][3],
    double (*inv_diag)[3][3]);
void bcrs_spmv_m(
    int num_nodes,
    int *bcrs_row_ptr,
    int *bcrs_col_ind,
    double *bcrs_mval,
    double *x,
    double *y);
int pcg_solve(
    int num_nodes,
    int *bcrs_row_ptr,
    int *bcrs_col_ind,
    double (*bcrs_kval)[3][3],
    double (*inv_diag)[3][3],
    double *rhs,
    double *x,
    double tol,
    int max_iter);
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
    int target_node,
    double **u,
    int num_steps,
    int sample_freq,
    double dt);

int main()
{
    double start_time = omp_get_wtime();

    char filepath[256] = "column.vtk";
    double duration = 0.005;
    int num_steps = 500000;
    int sample_freq = 500; // nステップごとにVTK出力
    double c1 = std::sqrt(4000.0 * 0.7 / 1.3 / 0.4 * 1.0e9);
    double c2 = std::sqrt(4000.0 / 2.0 / 1.3 * 1.0e9);
    double rho = 1.0e-9;
    int target_node = 0;
    int force_node = 0;
    int force_dof = 2; // z方向に力を加える
    double force_magnitude = -1.0;

    printf("c1: %.2f m/s, c2: %.2f m/s\n, rho: %.2e kg/m^3\n", c1, c2, rho);

    double lambda = rho * (c1 * c1 - 2 * c2 * c2);
    double mu = rho * c2 * c2;
    double dt = duration / num_steps;

    FEMMesh mesh = read_vtk(filepath);
    print_mesh_info(mesh);

    double *node_coords = mesh.coords_ptr();
    int num_nodes = mesh.num_nodes;
    int *ele_nodes = mesh.tet_ptr();
    int num_elements = mesh.num_tets;
    double dN0[3][10] = {0.0};                                             // ガウス点での形状関数の微分の配列
    double dN1[3][10] = {0.0};                                             // ガウス点での形状関数の微分の配列
    double dN2[3][10] = {0.0};                                             // ガウス点での形状関数の微分の配列
    double dN3[3][10] = {0.0};                                             // ガウス点での形状関数の微分の配列
    double (*kmat_coo_val)[3][3] = new double[100 * num_elements][3][3](); // 要素剛性行列の配列
    double *mmat_coo_val = new double[100 * num_elements]();               // 要素質量行列の配列
    int *coo_row = new int[100 * num_elements]();                          // 要素行列の行インデックスの配列
    int *coo_col = new int[100 * num_elements]();                          // 要素行列の列インデックスの配列

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

    double (*bcrs_kval)[3][3] = new double[nnz_bcrs][3][3];
    double *bcrs_mval = new double[nnz_bcrs];
    int *bcrs_row_ptr = new int[num_nodes + 1];
    int *bcrs_col_ind = new int[nnz_bcrs];

    build_bcrs(coo_row, coo_col, kmat_coo_val, mmat_coo_val, nnz_bcrs, num_nodes, bcrs_row_ptr, bcrs_col_ind, bcrs_kval, bcrs_mval);

    delete[] coo_row;
    delete[] coo_col;
    delete[] kmat_coo_val;
    delete[] mmat_coo_val;

    std::cout << "Number of non-zero blocks in bcrs: " << nnz_bcrs << std::endl;

    int *bc_flag = new int[num_nodes]();
    double (*bc_corr)[3] = new double[num_nodes * 3][3];
    double (*inv_diag)[3][3] = new double[num_nodes][3][3];
    double **u = new double *[num_steps / sample_freq + 1]; // 変位の記録
    for (int i = 0; i < num_steps / sample_freq + 1; i++)
        u[i] = new double[num_nodes * 3]();
    double *u_tmp = new double[num_nodes * 3](); // タイムステップごとの変位の一時保存用
    double *v_tmp = new double[num_nodes * 3](); // タイムステップごとの速度の一時保存用
    double *a_tmp = new double[num_nodes * 3](); // タイムステップごとの加速度の一時保存用
    double *u_prv = new double[num_nodes * 3](); // タイムステップごとの過去の変位の一時保存用
    double *rhs = new double[num_nodes * 3]();
    double *tmp = new double[num_nodes * 3]();

    // 境界条件
    for (int i = 0; i < num_nodes; i++)
    {
        if (node_coords[i * 3] < 1e-6)
            bc_flag[i] = 1;
    }

    extract_bc_correction(num_nodes, bcrs_row_ptr, bcrs_col_ind, bcrs_kval, bc_flag, bc_corr);

    apply_bc_to_lhs(num_nodes, bcrs_row_ptr, bcrs_col_ind, bcrs_kval, bc_flag);

    build_block_jacobi(num_nodes, bcrs_row_ptr, bcrs_col_ind, bcrs_kval, inv_diag);

    // タイムステップループ
    for (int step = 1; step <= num_steps; step++)
    {
        double t = step * dt;
        double bc_val[3] = {0.0, 0.0, 0.0};

        // ここでrhsを構築（外力の寄与なども加える）
        for (int i = 0; i < num_nodes * 3; i++)
        {
            // Newmark-β法の右辺の構築
            tmp[i] = u_tmp[i] * 4.0 / dt / dt + v_tmp[i] * 4.0 / dt + a_tmp[i];
        }
        bcrs_spmv_m(num_nodes, bcrs_row_ptr, bcrs_col_ind, bcrs_mval, tmp, rhs); // 質量行列の寄与
        rhs[force_node * 3 + force_dof] += force_magnitude;                      // 外力の寄与

        apply_bc_to_rhs(num_nodes, bc_flag, bc_val, bc_corr, rhs);

        int iter = pcg_solve(num_nodes, bcrs_row_ptr, bcrs_col_ind, bcrs_kval, inv_diag, rhs, u_tmp, 1e-8, num_nodes * 3);
        std::cout << "Step " << step << ", PCG iterations: " << iter << std::endl;

        // 速度と加速度の更新（Newmark-β法）
        for (int i = 0; i < num_nodes * 3; i++)
        {
            double u_new = u_tmp[i];
            double u_old = u_prv[i];
            double v_old = v_tmp[i];
            double a_old = a_tmp[i];

            double a_new = (u_new - u_old) * 4.0 / dt / dt - v_old * 4.0 / dt - a_old;
            double v_new = v_old + (a_new + a_old) * dt / 2.0;

            u_prv[i] = u_new;
            v_tmp[i] = v_new;
            a_tmp[i] = a_new;
        }

        // 解xを変位uに保存
        if (step % sample_freq == 0)
        {
            for (int i = 0; i < num_nodes * 3; i++)
            {
                u[step / sample_freq][i] = u_tmp[i];
            }
        }
    }

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
    sprintf(csv_filename, "%s/node%d_disp.csv", output_dir, target_node + 1);
    write_node_disp_csv(csv_filename, target_node, u, num_steps, sample_freq, dt);

    printf("Output: %s/\n", output_dir);
    printf("CSV: %s\n", csv_filename);

    double end_time = omp_get_wtime();
    printf("Total simulation time: %.2f seconds\n", end_time - start_time);

    return 0;
}

void calculate_dN(double dN[3][10], double r, double s, double t)
{
    // 形状関数の微分 dN をガウス点で計算する
    // dN は各節点の形状関数の x,y,z に対する微分を格納
    dN[0][0] = 4.0 * (r + s + t) - 3.0;
    dN[0][1] = 4.0 * r - 1.0;
    dN[0][2] = 0.0;
    dN[0][3] = 0.0;
    dN[0][4] = -4.0 * (2 * r + s + t) + 4.0;
    dN[0][5] = 4.0 * s;
    dN[0][6] = -4.0 * s;
    dN[0][7] = -4.0 * t;
    dN[0][8] = 4.0 * t;
    dN[0][9] = 0.0;
    dN[1][0] = 4.0 * (r + s + t) - 3.0;
    dN[1][1] = 0.0;
    dN[1][2] = 4.0 * s - 1.0;
    dN[1][3] = 0.0;
    dN[1][4] = -4.0 * r;
    dN[1][5] = 4.0 * r;
    dN[1][6] = -4.0 * (r + 2 * s + t) + 4.0;
    dN[1][7] = -4.0 * t;
    dN[1][8] = 0.0;
    dN[1][9] = 4.0 * t;
    dN[2][0] = 4.0 * (r + s + t) - 3.0;
    dN[2][1] = 0.0;
    dN[2][2] = 0.0;
    dN[2][3] = 4.0 * t - 1.0;
    dN[2][4] = -4.0 * r;
    dN[2][5] = 0.0;
    dN[2][6] = -4.0 * s;
    dN[2][7] = -4.0 * (r + s + 2 * t) + 4.0;
    dN[2][8] = 4.0 * r;
    dN[2][9] = 4.0 * s;

    return;
}

void gauss_integrate(double dN0[3][10], double dN1[3][10], double dN2[3][10], double dN3[3][10])
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

double inverse_3_3_mat(double mat[3][3], double inv_mat[3][3])
{
    // 3x3行列の逆行列を計算する
    double det_mat = mat[0][0] * (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1]) +
                     mat[0][1] * (mat[1][2] * mat[2][0] - mat[1][0] * mat[2][2]) +
                     mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]);
    double inv_det = 1.0 / det_mat;

    inv_mat[0][0] = (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1]) * inv_det;
    inv_mat[0][1] = (mat[0][2] * mat[2][1] - mat[0][1] * mat[2][2]) * inv_det;
    inv_mat[0][2] = (mat[0][1] * mat[1][2] - mat[0][2] * mat[1][1]) * inv_det;
    inv_mat[1][0] = (mat[1][2] * mat[2][0] - mat[1][0] * mat[2][2]) * inv_det;
    inv_mat[1][1] = (mat[0][0] * mat[2][2] - mat[0][2] * mat[2][0]) * inv_det;
    inv_mat[1][2] = (mat[0][2] * mat[1][0] - mat[0][0] * mat[1][2]) * inv_det;
    inv_mat[2][0] = (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]) * inv_det;
    inv_mat[2][1] = (mat[0][1] * mat[2][0] - mat[0][0] * mat[2][1]) * inv_det;
    inv_mat[2][2] = (mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]) * inv_det;

    return det_mat;
}

void construct_mat(
    double *node_coords,
    int *ele_nodes,
    int num_elements,
    double dN0[3][10],
    double dN1[3][10],
    double dN2[3][10],
    double dN3[3][10],
    double lambda,
    double mu,
    double rho,
    double dt,
    double (*kmat_coo_val)[3][3],
    double *mmat_coo_val,
    int *coo_row,
    int *coo_col)
{
    // ここに要素剛性行列 kmat_coo を構築するコードを実装
    // node_coords: [num_nodes * 3] 連続配置: x0,y0,z0,x1,y1,z1,...
    // ele_nodes: [num_elements * 10] 連続配置: 要素0の10節点, 要素1の10節点,...
    // kmat_coo は [100 * num_elements][3][3] の配列で、各要素の行列を連続配置で格納

    for (int elem = 0; elem < num_elements; elem++)
    {
        int local_node_indices[10];    // 要素の10節点のインデックスを格納する配列
        double local_coords[4][3];     // 4頂点の座標を格納する配列
        double jacobian[3][3];         // ヤコビアン行列を格納する配列
        double det_jacobian;           // ヤコビアンの行列式を格納する変数
        double inv_jacobian[3][3];     // ヤコビアンの逆行列を格納する配列
        double local_dN0[3][10] = {0}; // ガウス点での形状関数の微分の配列
        double local_dN1[3][10] = {0}; // ガウス点での形状関数の微分の配列
        double local_dN2[3][10] = {0}; // ガウス点での形状関数の微分の配列
        double local_dN3[3][10] = {0}; // ガウス点での形状関数の微分の配列
        double local_mmat[10][10] = {
            {6.0 / 2520.0, 1.0 / 2520.0, 1.0 / 2520.0, 1.0 / 2520.0, -4.0 / 2520.0, -6.0 / 2520.0, -4.0 / 2520.0, -4.0 / 2520.0, -6.0 / 2520.0, -6.0 / 2520.0},
            {1.0 / 2520.0, 6.0 / 2520.0, 1.0 / 2520.0, 1.0 / 2520.0, -4.0 / 2520.0, -4.0 / 2520.0, -6.0 / 2520.0, -6.0 / 2520.0, -4.0 / 2520.0, -6.0 / 2520.0},
            {1.0 / 2520.0, 1.0 / 2520.0, 6.0 / 2520.0, 1.0 / 2520.0, -6.0 / 2520.0, -4.0 / 2520.0, -4.0 / 2520.0, -6.0 / 2520.0, -6.0 / 2520.0, -4.0 / 2520.0},
            {1.0 / 2520.0, 1.0 / 2520.0, 1.0 / 2520.0, 6.0 / 2520.0, -6.0 / 2520.0, -6.0 / 2520.0, -6.0 / 2520.0, -4.0 / 2520.0, -4.0 / 2520.0, -4.0 / 2520.0},
            {-4.0 / 2520.0, -4.0 / 2520.0, -6.0 / 2520.0, -6.0 / 2520.0, 32.0 / 2520.0, 16.0 / 2520.0, 16.0 / 2520.0, 16.0 / 2520.0, 16.0 / 2520.0, 8.0 / 2520.0},
            {-6.0 / 2520.0, -4.0 / 2520.0, -4.0 / 2520.0, -6.0 / 2520.0, 16.0 / 2520.0, 32.0 / 2520.0, 16.0 / 2520.0, 8.0 / 2520.0, 16.0 / 2520.0, 16.0 / 2520.0},
            {-4.0 / 2520.0, -6.0 / 2520.0, -4.0 / 2520.0, -6.0 / 2520.0, 16.0 / 2520.0, 16.0 / 2520.0, 32.0 / 2520.0, 16.0 / 2520.0, 8.0 / 2520.0, 16.0 / 2520.0},
            {-4.0 / 2520.0, -6.0 / 2520.0, -6.0 / 2520.0, -4.0 / 2520.0, 16.0 / 2520.0, 8.0 / 2520.0, 16.0 / 2520.0, 32.0 / 2520.0, 16.0 / 2520.0, 16.0 / 2520.0},
            {-6.0 / 2520.0, -4.0 / 2520.0, -6.0 / 2520.0, -4.0 / 2520.0, 16.0 / 2520.0, 16.0 / 2520.0, 8.0 / 2520.0, 16.0 / 2520.0, 32.0 / 2520.0, 16.0 / 2520.0},
            {-6.0 / 2520.0, -6.0 / 2520.0, -4.0 / 2520.0, -4.0 / 2520.0, 8.0 / 2520.0, 16.0 / 2520.0, 16.0 / 2520.0, 16.0 / 2520.0, 16.0 / 2520.0, 32.0 / 2520.0}};

        for (int i = 0; i < 10; i++)
        {
            local_node_indices[i] = ele_nodes[elem * 10 + i];
        }
        for (int i = 0; i < 4; i++)
        {
            local_coords[i][0] = node_coords[local_node_indices[i] * 3 + 0];
            local_coords[i][1] = node_coords[local_node_indices[i] * 3 + 1];
            local_coords[i][2] = node_coords[local_node_indices[i] * 3 + 2];
        }

        for (int i = 1; i < 4; i++)
        {
            jacobian[i - 1][0] = local_coords[i][0] - local_coords[0][0];
            jacobian[i - 1][1] = local_coords[i][1] - local_coords[0][1];
            jacobian[i - 1][2] = local_coords[i][2] - local_coords[0][2];
        }
        det_jacobian = inverse_3_3_mat(jacobian, inv_jacobian);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 10; j++)
            {
                local_dN0[i][j] = inv_jacobian[i][0] * dN0[0][j] + inv_jacobian[i][1] * dN0[1][j] + inv_jacobian[i][2] * dN0[2][j];
                local_dN1[i][j] = inv_jacobian[i][0] * dN1[0][j] + inv_jacobian[i][1] * dN1[1][j] + inv_jacobian[i][2] * dN1[2][j];
                local_dN2[i][j] = inv_jacobian[i][0] * dN2[0][j] + inv_jacobian[i][1] * dN2[1][j] + inv_jacobian[i][2] * dN2[2][j];
                local_dN3[i][j] = inv_jacobian[i][0] * dN3[0][j] + inv_jacobian[i][1] * dN3[1][j] + inv_jacobian[i][2] * dN3[2][j];
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
                        local_kmat[a][b] = lambda * local_dN0[a][i] * local_dN0[b][j] + mu * local_dN0[a][j] * local_dN0[b][i];
                        local_kmat[a][b] += lambda * local_dN1[a][i] * local_dN1[b][j] + mu * local_dN1[a][j] * local_dN1[b][i];
                        local_kmat[a][b] += lambda * local_dN2[a][i] * local_dN2[b][j] + mu * local_dN2[a][j] * local_dN2[b][i];
                        local_kmat[a][b] += lambda * local_dN3[a][i] * local_dN3[b][j] + mu * local_dN3[a][j] * local_dN3[b][i];
                    }
                }
                double dot_product0 = local_dN0[0][i] * local_dN0[0][j] + local_dN0[1][i] * local_dN0[1][j] + local_dN0[2][i] * local_dN0[2][j];
                double dot_product1 = local_dN1[0][i] * local_dN1[0][j] + local_dN1[1][i] * local_dN1[1][j] + local_dN1[2][i] * local_dN1[2][j];
                double dot_product2 = local_dN2[0][i] * local_dN2[0][j] + local_dN2[1][i] * local_dN2[1][j] + local_dN2[2][i] * local_dN2[2][j];
                double dot_product3 = local_dN3[0][i] * local_dN3[0][j] + local_dN3[1][i] * local_dN3[1][j] + local_dN3[2][i] * local_dN3[2][j];
                for (int a = 0; a < 3; a++)
                {
                    local_kmat[a][a] += mu * dot_product0;
                    local_kmat[a][a] += mu * dot_product1;
                    local_kmat[a][a] += mu * dot_product2;
                    local_kmat[a][a] += mu * dot_product3;
                    local_kmat[a][a] += 24.0 * rho * local_mmat[i][j] * 4.0 / dt / dt; // 質量行列の寄与を剛性行列に加算（Newmark-β法のため）
                }
                for (int a = 0; a < 3; a++)
                {
                    for (int b = 0; b < 3; b++)
                    {
                        kmat_coo_val[elem * 100 + i * 10 + j][a][b] = local_kmat[a][b] * det_jacobian / 24.0; // 24.0は4点のガウス積分の重み
                    }
                }
                mmat_coo_val[elem * 100 + i * 10 + j] = rho * local_mmat[i][j] * det_jacobian; // 質量行列の値を格納
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
    double (*kmat_coo_val)[3][3],
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
    double (*work_kval)[3][3] = new double[nnz_coo][3][3];
    double *work_mval = new double[nnz_coo];

    int *pos = new int[num_nodes];
    for (int i = 0; i < num_nodes; i++)
    {
        pos[i] = offset[i];
    }

    for (int k = 0; k < nnz_coo; k++)
    {
        int r = coo_row[k];
        int p = pos[r]++;
        work_col[p] = coo_col[k];
        for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3; b++)
                work_kval[p][a][b] = kmat_coo_val[k][a][b];
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
            double tmp_kval[3][3];
            double tmp_mval = work_mval[a];
            for (int p = 0; p < 3; p++)
                for (int q = 0; q < 3; q++)
                    tmp_kval[p][q] = work_kval[a][p][q];

            int b = a - 1;
            while (b >= start && work_col[b] > tmp_col)
            {
                work_col[b + 1] = work_col[b];
                for (int p = 0; p < 3; p++)
                    for (int q = 0; q < 3; q++)
                        work_kval[b + 1][p][q] = work_kval[b][p][q];
                work_mval[b + 1] = work_mval[b];
                b--;
            }
            work_col[b + 1] = tmp_col;
            for (int p = 0; p < 3; p++)
                for (int q = 0; q < 3; q++)
                    work_kval[b + 1][p][q] = tmp_kval[p][q];
            work_mval[b + 1] = tmp_mval;
        }

        /* 重複列を足し合わせながらCOO配列の先頭に詰める */
        int write = nnz_bcrs;
        coo_row[write] = i;
        coo_col[write] = work_col[start];
        for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3; b++)
                kmat_coo_val[write][a][b] = work_kval[start][a][b];
        mmat_coo_val[write] = work_mval[start];

        for (int k = start + 1; k < end; k++)
        {
            if (work_col[k] == coo_col[write])
            {
                for (int a = 0; a < 3; a++)
                    for (int b = 0; b < 3; b++)
                        kmat_coo_val[write][a][b] += work_kval[k][a][b];
                mmat_coo_val[write] += work_mval[k];
            }
            else
            {
                write++;
                coo_row[write] = i;
                coo_col[write] = work_col[k];
                for (int a = 0; a < 3; a++)
                    for (int b = 0; b < 3; b++)
                        kmat_coo_val[write][a][b] = work_kval[k][a][b];
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
    double (*kmat_coo_val)[3][3],
    double *mmat_coo_val,
    int nnz_bcrs,
    int num_nodes,
    int *bcrs_row_ptr,
    int *bcrs_col_ind,
    double (*bcrs_kval)[3][3],
    double *bcrs_mval)
{
    // 並び替え済みCOOからbcrsを構築
    // col_ind, kval, mval をコピー
    for (int k = 0; k < nnz_bcrs; k++)
    {
        bcrs_col_ind[k] = coo_col[k];
        for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3; b++)
                bcrs_kval[k][a][b] = kmat_coo_val[k][a][b];
        bcrs_mval[k] = mmat_coo_val[k];
    }

    // row_ptr を構築
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

void bcrs_spmv(
    int num_nodes,
    int *row_ptr,
    int *col_ind,
    double (*val)[3][3],
    double *x,
    double *y)
{
    // bcrs 行列ベクトル積: y = A * x
    for (int i = 0; i < num_nodes; i++)
    {
        double y0 = 0.0, y1 = 0.0, y2 = 0.0;
        for (int p = row_ptr[i]; p < row_ptr[i + 1]; p++)
        {
            int j = col_ind[p];
            y0 += val[p][0][0] * x[j * 3 + 0] + val[p][0][1] * x[j * 3 + 1] + val[p][0][2] * x[j * 3 + 2];
            y1 += val[p][1][0] * x[j * 3 + 0] + val[p][1][1] * x[j * 3 + 1] + val[p][1][2] * x[j * 3 + 2];
            y2 += val[p][2][0] * x[j * 3 + 0] + val[p][2][1] * x[j * 3 + 1] + val[p][2][2] * x[j * 3 + 2];
        }
        y[i * 3 + 0] = y0;
        y[i * 3 + 1] = y1;
        y[i * 3 + 2] = y2;
    }
}

void bcrs_spmv_m(
    int num_nodes,
    int *row_ptr,
    int *col_ind,
    double(*val),
    double *x,
    double *y)
{
    // bcrs 行列ベクトル積: y = A * x
    for (int i = 0; i < num_nodes; i++)
    {
        double y0 = 0.0, y1 = 0.0, y2 = 0.0;
        for (int p = row_ptr[i]; p < row_ptr[i + 1]; p++)
        {
            int j = col_ind[p];
            y0 += val[p] * x[j * 3 + 0];
            y1 += val[p] * x[j * 3 + 1];
            y2 += val[p] * x[j * 3 + 2];
        }
        y[i * 3 + 0] = y0;
        y[i * 3 + 1] = y1;
        y[i * 3 + 2] = y2;
    }
}

void extract_bc_correction(
    int num_nodes,
    int *row_ptr,
    int *col_ind,
    double (*kval)[3][3],
    int *bc_flag,
    double (*bc_corr)[3])
{
    // 境界条件の補正ベクトルを抽出
    // 自由節点iと拘束節点jの間のカップリング A[i][j] を抽出する。
    // 毎タイムステップの右辺ベクトル補正に使う。
    // bc_corr[num_nodes * 3][3]: bc_corr[i*3+a][b] = Σ_{j∈constrained} A_original[i][j][a][b]
    int ndof = num_nodes * 3;
    for (int i = 0; i < ndof; i++)
        for (int b = 0; b < 3; b++)
            bc_corr[i][b] = 0.0;

    for (int i = 0; i < num_nodes; i++)
    {
        if (bc_flag[i])
            continue;

        for (int p = row_ptr[i]; p < row_ptr[i + 1]; p++)
        {
            int j = col_ind[p];
            if (!bc_flag[j])
                continue;

            for (int a = 0; a < 3; a++)
                for (int b = 0; b < 3; b++)
                    bc_corr[i * 3 + a][b] += kval[p][a][b];
        }
    }
}

void apply_bc_to_lhs(
    int num_nodes,
    int *row_ptr,
    int *col_ind,
    double (*kval)[3][3],
    int *bc_flag)
{
    // 境界条件を左辺行列に適用（1回だけ呼ぶ）
    // bc_flag[num_nodes]: 0=自由, 1=拘束（節点単位、3方向同時拘束）
    for (int i = 0; i < num_nodes; i++)
    {
        for (int p = row_ptr[i]; p < row_ptr[i + 1]; p++)
        {
            int j = col_ind[p];

            if (bc_flag[i] || bc_flag[j])
            {
                for (int a = 0; a < 3; a++)
                    for (int b = 0; b < 3; b++)
                        kval[p][a][b] = 0.0;
            }

            if (bc_flag[i] && i == j)
            {
                kval[p][0][0] = 1.0;
                kval[p][1][1] = 1.0;
                kval[p][2][2] = 1.0;
            }
        }
    }
}

void apply_bc_to_rhs(
    int num_nodes,
    int *bc_flag,
    double *bc_val,
    double (*bc_corr)[3],
    double *rhs)
{
    // 境界条件を右辺ベクトルに適用（毎タイムステップ呼ぶ）
    // bc_val[3]: 拘束変位値（全拘束節点で共通、例: {0, sin(t), 0}）
    // 自由節点の右辺を補正: rhs[i] -= Σ_b bc_corr[i][b] * bc_val[b]
    int ndof = num_nodes * 3;
    for (int i = 0; i < ndof; i++)
    {
        rhs[i] -= bc_corr[i][0] * bc_val[0] + bc_corr[i][1] * bc_val[1] + bc_corr[i][2] * bc_val[2];
    }

    // 拘束節点の右辺を拘束値に設定
    for (int i = 0; i < num_nodes; i++)
    {
        if (bc_flag[i])
        {
            rhs[i * 3 + 0] = bc_val[0];
            rhs[i * 3 + 1] = bc_val[1];
            rhs[i * 3 + 2] = bc_val[2];
        }
    }
}

void build_block_jacobi(
    int num_nodes,
    int *row_ptr,
    int *col_ind,
    double (*kval)[3][3],
    double (*inv_diag)[3][3])
{
    // ブロックヤコビ前処理の構築（対角ブロックの逆行列を計算）
    for (int i = 0; i < num_nodes; i++)
    {
        /* 対角ブロックを探す */
        double d[3][3] = {};
        for (int p = row_ptr[i]; p < row_ptr[i + 1]; p++)
        {
            if (col_ind[p] == i)
            {
                for (int a = 0; a < 3; a++)
                    for (int b = 0; b < 3; b++)
                        d[a][b] = kval[p][a][b];
                break;
            }
        }
        inverse_3_3_mat(d, inv_diag[i]);
    }
}

void apply_block_jacobi(
    int num_nodes,
    double (*inv_diag)[3][3],
    double *r,
    double *z)
{
    // ブロックヤコビ前処理の適用: z = M⁻¹ r
    for (int i = 0; i < num_nodes; i++)
    {
        double r0 = r[i * 3 + 0];
        double r1 = r[i * 3 + 1];
        double r2 = r[i * 3 + 2];
        z[i * 3 + 0] = inv_diag[i][0][0] * r0 + inv_diag[i][0][1] * r1 + inv_diag[i][0][2] * r2;
        z[i * 3 + 1] = inv_diag[i][1][0] * r0 + inv_diag[i][1][1] * r1 + inv_diag[i][1][2] * r2;
        z[i * 3 + 2] = inv_diag[i][2][0] * r0 + inv_diag[i][2][1] * r1 + inv_diag[i][2][2] * r2;
    }
}

double dot(int n, double *a, double *b)
{
    // ベクトル内積
    double s = 0.0;
    for (int i = 0; i < n; i++)
    {
        s += a[i] * b[i];
    }
    return s;
}

int pcg_solve(
    int num_nodes,
    int *row_ptr,
    int *col_ind,
    double (*kval)[3][3],
    double (*inv_diag)[3][3],
    double *b,
    double *x,
    double tol,
    int max_iter)
{
    // Ax = b を前処理付き共役勾配法で解く。
    // x は初期解を入れて呼ぶ（ゼロでもよい）。解が上書きされる。
    int ndof = num_nodes * 3;

    double *r = new double[ndof];
    double *z = new double[ndof];
    double *p = new double[ndof];
    double *Ap = new double[ndof];

    // r = b - A*x
    bcrs_spmv(num_nodes, row_ptr, col_ind, kval, x, Ap);
    for (int i = 0; i < ndof; i++)
    {
        r[i] = b[i] - Ap[i];
    }

    // z = M⁻¹r
    apply_block_jacobi(num_nodes, inv_diag, r, z);

    // p = z
    for (int i = 0; i < ndof; i++)
    {
        p[i] = z[i];
    }

    double rz = dot(ndof, r, z);
    double b_norm = sqrt(dot(ndof, b, b));
    if (b_norm == 0.0)
        b_norm = 1.0;

    double r_norm_init = sqrt(dot(ndof, r, r));
    if (r_norm_init / b_norm < tol)
    {
        delete[] r;
        delete[] z;
        delete[] p;
        delete[] Ap;
        return 0;
    }

    int iter;
    for (iter = 0; iter < max_iter; iter++)
    {
        // Ap = A * p
        bcrs_spmv(num_nodes, row_ptr, col_ind, kval, p, Ap);

        // alpha = rz / (p · Ap)
        double pAp = dot(ndof, p, Ap);
        double alpha = rz / pAp;

        // x += alpha * p, r -= alpha * Ap
        for (int i = 0; i < ndof; i++)
        {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        // 収束判定
        double r_norm = sqrt(dot(ndof, r, r));
        if (r_norm / b_norm < tol)
        {
            iter++;
            break;
        }

        // z = M⁻¹r
        apply_block_jacobi(num_nodes, inv_diag, r, z);

        // beta = rz_new / rz
        double rz_new = dot(ndof, r, z);
        double beta = rz_new / rz;

        // p = z + beta * p
        for (int i = 0; i < ndof; i++)
        {
            p[i] = z[i] + beta * p[i];
        }

        rz = rz_new;
    }

    delete[] r;
    delete[] z;
    delete[] p;
    delete[] Ap;

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
    int target_node,
    double **u,
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
        double ux = u[idx][target_node * 3 + 0];
        double uy = u[idx][target_node * 3 + 1];
        double uz = u[idx][target_node * 3 + 2];
        double mag = sqrt(ux * ux + uy * uy + uz * uz);
        fprintf(csv_fp, "%.10e,%.10e,%.10e,%.10e,%.10e\n", t, ux, uy, uz, mag);
    }
    fclose(csv_fp);
}