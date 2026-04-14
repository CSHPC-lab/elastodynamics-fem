/*実行コマンド
cd /data3/kusumoto/elastodynamics-fem/
module load nvhpc/25.1
nvcc main_cusparse_bsr.cu msh_reader.cpp -Xcompiler -fopenmp -ccbin mpicxx -lcusparse -lcublas -arch=sm_80
mpirun -np 4 ./a.out
*/

#include "msh_reader.hpp"
#include "config.hpp"
#include <iostream>
#include <cmath>
#include <ctime>
#include <sys/stat.h>
#include <omp.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

// ============================================================
// エラーチェックマクロ
// ============================================================
#define CUDA_CHECK(call)                                                     \
    do                                                                       \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess)                                              \
        {                                                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            MPI_Abort(MPI_COMM_WORLD, 1);                                    \
        }                                                                    \
    } while (0)

#define CUSPARSE_CHECK(call)                                                     \
    do                                                                           \
    {                                                                            \
        cusparseStatus_t err = call;                                             \
        if (err != CUSPARSE_STATUS_SUCCESS)                                      \
        {                                                                        \
            fprintf(stderr, "cuSPARSE error at %s:%d: %d\n", __FILE__, __LINE__, \
                    (int)err);                                                   \
            MPI_Abort(MPI_COMM_WORLD, 1);                                        \
        }                                                                        \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do                                                                         \
    {                                                                          \
        cublasStatus_t err = call;                                             \
        if (err != CUBLAS_STATUS_SUCCESS)                                      \
        {                                                                      \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, \
                    (int)err);                                                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    } while (0)

static const int BLOCK_SIZE = 256;

// ============================================================
// CPU側の関数宣言（手書き版と同一）
// ============================================================
void calculate_dN(double dN[30], double r, double s, double t);
void gauss_integrate(double dN0[30], double dN1[30], double dN2[30], double dN3[30]);
double inverse_3_3_mat(double mat[9], double inv_mat[9]);
void construct_mat(double *node_coords, int *ele_nodes, int num_elements,
                   double dN0[30], double dN1[30], double dN2[30], double dN3[30],
                   double lambda, double mu, double rho, double dt,
                   double *kmat_coo_val, double *mmat_coo_val, int *coo_row, int *coo_col);
int sort_and_merge_bcoo(int nnz_coo, int num_nodes, int *coo_row, int *coo_col,
                        double *kmat_coo_val, double *mmat_coo_val);
void build_bcrs(int *coo_row, int *coo_col, double *kmat_coo_val, double *mmat_coo_val,
                int nnz_bcoo, int num_nodes, int *bcrs_row_ptr, int *bcrs_col_ind,
                double *k00, double *k01, double *k02, double *k10, double *k11, double *k12,
                double *k20, double *k21, double *k22, double *mval);
void extract_bc_correction(int num_nodes, int *rp, int *ci,
                           double *k00, double *k01, double *k02, double *k10, double *k11, double *k12,
                           double *k20, double *k21, double *k22, int *bc_flag,
                           double *c00, double *c01, double *c02, double *c10, double *c11, double *c12,
                           double *c20, double *c21, double *c22);
void apply_bc_to_lhs(int num_nodes, int *rp, int *ci,
                     double *k00, double *k01, double *k02, double *k10, double *k11, double *k12,
                     double *k20, double *k21, double *k22, int *bc_flag);
void build_block_jacobi(int num_nodes, int *rp, int *ci,
                        double *k00, double *k01, double *k02, double *k10, double *k11, double *k12,
                        double *k20, double *k21, double *k22,
                        double *inv00, double *inv01, double *inv02, double *inv10, double *inv11, double *inv12,
                        double *inv20, double *inv21, double *inv22);
void write_vtk_displacement(const char *fn, double *coords, int nn, int *en, int ne, double *disp, double t);
void write_node_disp_csv(const char *fn, double *u, int ns, int sf, double dt);

// ============================================================
// BCRS → スカラーCSR 変換 (CPU上で1回だけ実行)
// ============================================================
// K: 各3×3ブロック → 9エントリ   nnz_scalar = 9 * nnz_bcrs
// M: 各ブロックは m*I → 3エントリ nnz_scalar = 3 * nnz_bcrs
void build_scalar_csr_K(
    int num_nodes, int nnz_bcrs,
    const int *brp, const int *bci,
    const double *k00, const double *k01, const double *k02,
    const double *k10, const double *k11, const double *k12,
    const double *k20, const double *k21, const double *k22,
    int *csr_rp, int *csr_ci, double *csr_val)
{
    // csr_rp: size 3*num_nodes + 1
    // csr_ci, csr_val: size 9*nnz_bcrs
    int N3 = 3 * num_nodes;

    // 各スカラー行の非ゼロ数を計算
    // ブロック行 i → スカラー行 3i, 3i+1, 3i+2 それぞれ 3*nnz_in_block_row 個
    for (int i = 0; i <= N3; i++)
        csr_rp[i] = 0;
    for (int i = 0; i < num_nodes; i++)
    {
        int bnnz = brp[i + 1] - brp[i]; // ブロック行iの非ゼロブロック数
        csr_rp[3 * i + 1] = 3 * bnnz;
        csr_rp[3 * i + 2] = 3 * bnnz;
        csr_rp[3 * i + 3] = 3 * bnnz;
    }
    for (int i = 1; i <= N3; i++)
        csr_rp[i] += csr_rp[i - 1];

    // 値を埋める
#pragma omp parallel for
    for (int i = 0; i < num_nodes; i++)
    {
        for (int r = 0; r < 3; r++)
        {
            int scalar_row = 3 * i + r;
            int write_pos = csr_rp[scalar_row];
            for (int bp = brp[i]; bp < brp[i + 1]; bp++)
            {
                int j = bci[bp];
                for (int c = 0; c < 3; c++)
                {
                    csr_ci[write_pos] = 3 * j + c;
                    // kval[r][c] at block position bp
                    double v = 0.0;
                    if (r == 0 && c == 0)
                        v = k00[bp];
                    else if (r == 0 && c == 1)
                        v = k01[bp];
                    else if (r == 0 && c == 2)
                        v = k02[bp];
                    else if (r == 1 && c == 0)
                        v = k10[bp];
                    else if (r == 1 && c == 1)
                        v = k11[bp];
                    else if (r == 1 && c == 2)
                        v = k12[bp];
                    else if (r == 2 && c == 0)
                        v = k20[bp];
                    else if (r == 2 && c == 1)
                        v = k21[bp];
                    else if (r == 2 && c == 2)
                        v = k22[bp];
                    csr_val[write_pos] = v;
                    write_pos++;
                }
            }
        }
    }
}

void build_scalar_csr_M(
    int num_nodes, int nnz_bcrs,
    const int *brp, const int *bci,
    const double *mval,
    int *csr_rp, int *csr_ci, double *csr_val)
{
    // M の各ブロック (i,j) は mval[bp] * I_3
    // スカラー行 3i+r には列 3j+r にだけ非ゼロ (r=0,1,2)
    // 各スカラー行の非ゼロ数 = ブロック行の非ゼロブロック数
    int N3 = 3 * num_nodes;
    for (int i = 0; i <= N3; i++)
        csr_rp[i] = 0;
    for (int i = 0; i < num_nodes; i++)
    {
        int bnnz = brp[i + 1] - brp[i];
        csr_rp[3 * i + 1] = bnnz;
        csr_rp[3 * i + 2] = bnnz;
        csr_rp[3 * i + 3] = bnnz;
    }
    for (int i = 1; i <= N3; i++)
        csr_rp[i] += csr_rp[i - 1];

#pragma omp parallel for
    for (int i = 0; i < num_nodes; i++)
    {
        for (int r = 0; r < 3; r++)
        {
            int scalar_row = 3 * i + r;
            int write_pos = csr_rp[scalar_row];
            for (int bp = brp[i]; bp < brp[i + 1]; bp++)
            {
                int j = bci[bp];
                csr_ci[write_pos] = 3 * j + r;
                csr_val[write_pos] = mval[bp];
                write_pos++;
            }
        }
    }
}

// ============================================================
// CUDAカーネル（ライブラリで置き換えられない問題固有の処理）
// ============================================================

// Newmark-β RHS一時ベクトル: インターリーブ、成分独立なので3N並列
__global__ void kernel_build_newmark_tmp(
    int n3, double dt_inv2, double dt_inv,
    const double *u, const double *v, const double *a, double *tmp)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n3)
        return;
    tmp[i] = u[i] * 4.0 * dt_inv2 + v[i] * 4.0 * dt_inv + a[i];
}

// 境界条件RHS補正（自由節点）: bc_corrはSoA、rhsはインターリーブ
__global__ void kernel_apply_bc_corr(
    int num_nodes, const double *bc_val,
    const double *c00, const double *c01, const double *c02,
    const double *c10, const double *c11, const double *c12,
    const double *c20, const double *c21, const double *c22,
    double *rhs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes)
        return;
    double bv0 = bc_val[0], bv1 = bc_val[1], bv2 = bc_val[2];
    rhs[3 * i + 0] -= c00[i] * bv0 + c01[i] * bv1 + c02[i] * bv2;
    rhs[3 * i + 1] -= c10[i] * bv0 + c11[i] * bv1 + c12[i] * bv2;
    rhs[3 * i + 2] -= c20[i] * bv0 + c21[i] * bv1 + c22[i] * bv2;
}

// 境界条件RHS設定（拘束節点）
__global__ void kernel_apply_bc_constrained(
    int num_nodes, const int *bc_flag, const double *bc_val, double *rhs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes)
        return;
    if (bc_flag[i])
    {
        rhs[3 * i + 0] = bc_val[0];
        rhs[3 * i + 1] = bc_val[1];
        rhs[3 * i + 2] = bc_val[2];
    }
}

// ブロックヤコビ前処理: z = C^{-1} r (inv_diagはSoA、r/zはインターリーブ)
__global__ void kernel_block_jacobi_precond(
    int num_owned,
    const double *inv00, const double *inv01, const double *inv02,
    const double *inv10, const double *inv11, const double *inv12,
    const double *inv20, const double *inv21, const double *inv22,
    const double *r, double *z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_owned)
        return;
    double r0 = r[3 * i], r1 = r[3 * i + 1], r2 = r[3 * i + 2];
    z[3 * i + 0] = inv00[i] * r0 + inv01[i] * r1 + inv02[i] * r2;
    z[3 * i + 1] = inv10[i] * r0 + inv11[i] * r1 + inv12[i] * r2;
    z[3 * i + 2] = inv20[i] * r0 + inv21[i] * r1 + inv22[i] * r2;
}

// Newmark-β更新
__global__ void kernel_newmark_update(
    int num_nodes, double dt, double dt_inv, double dt_inv2,
    const int *bc_flag,
    const double *bc_val_u, const double *bc_val_v, const double *bc_val_a,
    double *u, double *u_prv, double *v, double *a)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes)
        return;

    double un0 = u[3 * i], un1 = u[3 * i + 1], un2 = u[3 * i + 2];
    double uo0 = u_prv[3 * i], uo1 = u_prv[3 * i + 1], uo2 = u_prv[3 * i + 2];
    double vo0 = v[3 * i], vo1 = v[3 * i + 1], vo2 = v[3 * i + 2];
    double ao0 = a[3 * i], ao1 = a[3 * i + 1], ao2 = a[3 * i + 2];

    double an0 = (un0 - uo0) * 4.0 * dt_inv2 - vo0 * 4.0 * dt_inv - ao0;
    double an1 = (un1 - uo1) * 4.0 * dt_inv2 - vo1 * 4.0 * dt_inv - ao1;
    double an2 = (un2 - uo2) * 4.0 * dt_inv2 - vo2 * 4.0 * dt_inv - ao2;
    double vn0 = vo0 + (an0 + ao0) * dt / 2.0;
    double vn1 = vo1 + (an1 + ao1) * dt / 2.0;
    double vn2 = vo2 + (an2 + ao2) * dt / 2.0;

    if (bc_flag[i])
    {
        un0 = bc_val_u[0];
        un1 = bc_val_u[1];
        un2 = bc_val_u[2];
        vn0 = bc_val_v[0];
        vn1 = bc_val_v[1];
        vn2 = bc_val_v[2];
        an0 = bc_val_a[0];
        an1 = bc_val_a[1];
        an2 = bc_val_a[2];
    }

    u_prv[3 * i] = un0;
    u_prv[3 * i + 1] = un1;
    u_prv[3 * i + 2] = un2;
    v[3 * i] = vn0;
    v[3 * i + 1] = vn1;
    v[3 * i + 2] = vn2;
    a[3 * i] = an0;
    a[3 * i + 1] = an1;
    a[3 * i + 2] = an2;
}

// MPI送信バッファパック（インターリーブ）
__global__ void kernel_pack_send(
    int count, int offset, const int *send_nodes,
    const double *vec, double *sbuf)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    int node = send_nodes[offset + i];
    sbuf[3 * (offset + i) + 0] = vec[3 * node + 0];
    sbuf[3 * (offset + i) + 1] = vec[3 * node + 1];
    sbuf[3 * (offset + i) + 2] = vec[3 * node + 2];
}

// ============================================================
// ヘルパー
// ============================================================
template <typename T>
T *d_alloc(int n)
{
    T *p;
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
    return p;
}
template <typename T>
T *d_alloc_copy(const T *h, int n)
{
    T *p;
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(p, h, n * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
template <typename T>
T *d_alloc_zero(int n)
{
    T *p;
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
    CUDA_CHECK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}

#define GRID(n) (((n) + BLOCK_SIZE - 1) / BLOCK_SIZE)

// ============================================================
// PCGソルバー (cuSPARSE / cuBLAS 版)
// ============================================================
// ※ 通信と計算のオーバーラップは行わない（ライブラリ版の制約）
int pcg_solve_cusparse(
    cusparseHandle_t cusparse_h,
    cublasHandle_t cublas_h,
    cusparseSpMatDescr_t K_descr,
    cusparseDnVecDescr_t vecX_descr, cusparseDnVecDescr_t vecY_descr,
    void *spmv_buffer,
    // MPI
    MPI_Request *request,
    int num_neighbors, int *neighbor_ranks,
    int *recv_starts, int *recv_counts,
    int *send_starts, int *send_counts,
    int *d_send_nodes, double *d_send_buf,
    // サイズ
    int num_inner, int num_owned, int num_nodes,
    // 前処理
    double *d_inv00, double *d_inv01, double *d_inv02,
    double *d_inv10, double *d_inv11, double *d_inv12,
    double *d_inv20, double *d_inv21, double *d_inv22,
    // ベクトル（インターリーブ, size 3*num_nodes）
    double *d_b, double *d_x, double *d_r, double *d_z, double *d_p, double *d_Ap,
    double tol, int max_iter)
{
    int N3 = 3 * num_nodes;
    int owned3 = 3 * num_owned;
    double alpha_sp = 1.0, beta_sp = 0.0, neg_one = -1.0, one = 1.0;

    // --- r = b - K*x ---
    // Ap = K * x
    cusparseDnVecSetValues(vecX_descr, d_x);
    cusparseDnVecSetValues(vecY_descr, d_Ap);
    CUSPARSE_CHECK(cusparseSpMV(cusparse_h, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha_sp, K_descr, vecX_descr, &beta_sp, vecY_descr,
                                CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, spmv_buffer));

    // r = b
    CUDA_CHECK(cudaMemcpy(d_r, d_b, owned3 * sizeof(double), cudaMemcpyDeviceToDevice));
    // r -= Ap  (owned部分のみ)
    CUBLAS_CHECK(cublasDaxpy(cublas_h, owned3, &neg_one, d_Ap, 1, d_r, 1));

    // z = C^{-1} r
    kernel_block_jacobi_precond<<<GRID(num_owned), BLOCK_SIZE>>>(
        num_owned, d_inv00, d_inv01, d_inv02, d_inv10, d_inv11, d_inv12,
        d_inv20, d_inv21, d_inv22, d_r, d_z);

    // p = z (owned部分)
    CUDA_CHECK(cudaMemcpy(d_p, d_z, owned3 * sizeof(double), cudaMemcpyDeviceToDevice));

    // rz = r · z, b_norm = b · b, r_norm = r · r  (owned部分)
    double rz, b_norm, r_norm;
    CUBLAS_CHECK(cublasDdot(cublas_h, owned3, d_r, 1, d_z, 1, &rz));
    CUBLAS_CHECK(cublasDdot(cublas_h, owned3, d_b, 1, d_b, 1, &b_norm));
    CUBLAS_CHECK(cublasDdot(cublas_h, owned3, d_r, 1, d_r, 1, &r_norm));

    MPI_Allreduce(MPI_IN_PLACE, &rz, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &b_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &r_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (b_norm == 0.0)
        b_norm = 1.0;
    if (r_norm / b_norm < tol * tol)
        return 0;

    int iter;
    for (iter = 0; iter < max_iter; iter++)
    {
        // --- ゴースト節点通信 (p) ---
        for (int n = 0; n < num_neighbors; n++)
        {
            int ss = send_starts[n], sc = send_counts[n];
            if (sc > 0)
            {
                kernel_pack_send<<<GRID(sc), BLOCK_SIZE>>>(
                    sc, ss, d_send_nodes, d_p, d_send_buf);
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int n = 0; n < num_neighbors; n++)
        {
            int ss = send_starts[n], sc = send_counts[n];
            // 送受信: インターリーブなので3倍のデータ量、1回のsend/recvで済む
            MPI_Isend(&d_send_buf[3 * ss], 3 * sc, MPI_DOUBLE,
                      neighbor_ranks[n], 0, MPI_COMM_WORLD, &request[n]);
            MPI_Irecv(&d_p[3 * recv_starts[n]], 3 * recv_counts[n], MPI_DOUBLE,
                      neighbor_ranks[n], 0, MPI_COMM_WORLD, &request[num_neighbors + n]);
        }
        MPI_Waitall(2 * num_neighbors, request, MPI_STATUSES_IGNORE);

        // --- Ap = K * p （通信完了後に全体SpMV） ---
        cusparseDnVecSetValues(vecX_descr, d_p);
        cusparseDnVecSetValues(vecY_descr, d_Ap);
        CUSPARSE_CHECK(cusparseSpMV(cusparse_h, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha_sp, K_descr, vecX_descr, &beta_sp, vecY_descr,
                                    CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, spmv_buffer));

        // pAp = p · Ap (owned部分)
        double pAp;
        CUBLAS_CHECK(cublasDdot(cublas_h, owned3, d_p, 1, d_Ap, 1, &pAp));
        MPI_Allreduce(MPI_IN_PLACE, &pAp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double alpha = rz / pAp;

        // x += alpha * p (全節点)
        CUBLAS_CHECK(cublasDaxpy(cublas_h, N3, &alpha, d_p, 1, d_x, 1));

        // r -= alpha * Ap (owned部分)
        double neg_alpha = -alpha;
        CUBLAS_CHECK(cublasDaxpy(cublas_h, owned3, &neg_alpha, d_Ap, 1, d_r, 1));

        // r_norm (owned部分)
        CUBLAS_CHECK(cublasDdot(cublas_h, owned3, d_r, 1, d_r, 1, &r_norm));
        MPI_Allreduce(MPI_IN_PLACE, &r_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if (r_norm / b_norm < tol * tol)
        {
            iter++;
            break;
        }

        // z = C^{-1} r
        kernel_block_jacobi_precond<<<GRID(num_owned), BLOCK_SIZE>>>(
            num_owned, d_inv00, d_inv01, d_inv02, d_inv10, d_inv11, d_inv12,
            d_inv20, d_inv21, d_inv22, d_r, d_z);

        // rz_new = r · z (owned部分)
        double rz_new;
        CUBLAS_CHECK(cublasDdot(cublas_h, owned3, d_r, 1, d_z, 1, &rz_new));
        MPI_Allreduce(MPI_IN_PLACE, &rz_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double beta = rz_new / rz;

        // p = z + beta * p  → p *= beta; p += z (owned部分)
        CUBLAS_CHECK(cublasDscal(cublas_h, owned3, &beta, d_p, 1));
        CUBLAS_CHECK(cublasDaxpy(cublas_h, owned3, &one, d_z, 1, d_p, 1));

        rz = rz_new;
    }
    return iter;
}

// ============================================================
// main
// ============================================================
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
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

    int num_gpus;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    printf("使用可能な最大GPU数：%d\n", num_gpus);
    CUDA_CHECK(cudaSetDevice(rank % num_gpus));

    double *node_coords = mesh.coords_ptr();
    int num_nodes = mesh.num_total;
    int num_inner = mesh.num_inner;
    int num_owned = mesh.num_owned;
    int *ele_nodes = mesh.elem_ptr();
    int num_elements = mesh.num_total_elems;
    int N3 = 3 * num_nodes;

    // MPI通信情報
    int num_neighbors = mesh.num_neighbors();
    int *neighbor_ranks = new int[num_neighbors];
    int *recv_starts = new int[num_neighbors];
    int *recv_counts = new int[num_neighbors];
    std::vector<int> send_nodes_vec;
    int *send_starts = new int[num_neighbors + 1];
    int *send_counts = new int[num_neighbors];
    for (int i = 0; i < num_neighbors; i++)
    {
        neighbor_ranks[i] = mesh.neighbors[i].partition_id - 1;
        recv_starts[i] = mesh.neighbors[i].recv_start;
        recv_counts[i] = mesh.neighbors[i].recv_count;
        send_starts[i] = (i > 0) ? send_starts[i - 1] + send_counts[i - 1] : 0;
        send_counts[i] = mesh.neighbors[i].send_size();
        send_nodes_vec.insert(send_nodes_vec.end(),
                              mesh.neighbors[i].send_nodes.begin(), mesh.neighbors[i].send_nodes.end());
    }
    send_starts[num_neighbors] = send_starts[num_neighbors - 1] + send_counts[num_neighbors - 1];
    int total_send = send_starts[num_neighbors];

    // --- CPU上で行列構築（手書き版と同一） ---
    double dN0[30] = {}, dN1[30] = {}, dN2[30] = {}, dN3_[30] = {};
    double *kmat_coo_val = new double[100 * num_elements * 9]();
    double *mmat_coo_val = new double[100 * num_elements]();
    int *coo_row = new int[100 * num_elements]();
    int *coo_col = new int[100 * num_elements]();

    if (target_node >= 0)
    {
        printf("Target node local ID: %d\n", target_node);
        printf("Target node coordinates: (%.3f, %.3f, %.3f)\n",
               node_coords[target_node * 3], node_coords[target_node * 3 + 1], node_coords[target_node * 3 + 2]);
    }

    gauss_integrate(dN0, dN1, dN2, dN3_);
    construct_mat(node_coords, ele_nodes, num_elements, dN0, dN1, dN2, dN3_,
                  lambda, mu, rho, dt, kmat_coo_val, mmat_coo_val, coo_row, coo_col);
    int nnz_bcrs = sort_and_merge_bcoo(100 * num_elements, num_nodes, coo_row, coo_col, kmat_coo_val, mmat_coo_val);

    double *bk00 = new double[nnz_bcrs], *bk01 = new double[nnz_bcrs], *bk02 = new double[nnz_bcrs];
    double *bk10 = new double[nnz_bcrs], *bk11 = new double[nnz_bcrs], *bk12 = new double[nnz_bcrs];
    double *bk20 = new double[nnz_bcrs], *bk21 = new double[nnz_bcrs], *bk22 = new double[nnz_bcrs];
    double *bmval = new double[nnz_bcrs];
    int *brp = new int[num_nodes + 1], *bci = new int[nnz_bcrs];

    build_bcrs(coo_row, coo_col, kmat_coo_val, mmat_coo_val, nnz_bcrs, num_nodes, brp, bci,
               bk00, bk01, bk02, bk10, bk11, bk12, bk20, bk21, bk22, bmval);
    delete[] coo_row;
    delete[] coo_col;
    delete[] kmat_coo_val;
    delete[] mmat_coo_val;
    std::cout << "Number of non-zero blocks in bcrs: " << nnz_bcrs << std::endl;

    // 境界条件（CPU上）
    int *bc_flag = new int[num_nodes]();
    double *bc_corr_00 = new double[num_nodes](), *bc_corr_01 = new double[num_nodes](), *bc_corr_02 = new double[num_nodes]();
    double *bc_corr_10 = new double[num_nodes](), *bc_corr_11 = new double[num_nodes](), *bc_corr_12 = new double[num_nodes]();
    double *bc_corr_20 = new double[num_nodes](), *bc_corr_21 = new double[num_nodes](), *bc_corr_22 = new double[num_nodes]();
    double *inv00 = new double[num_nodes](), *inv01 = new double[num_nodes](), *inv02 = new double[num_nodes]();
    double *inv10 = new double[num_nodes](), *inv11 = new double[num_nodes](), *inv12 = new double[num_nodes]();
    double *inv20 = new double[num_nodes](), *inv21 = new double[num_nodes](), *inv22 = new double[num_nodes]();

#pragma omp parallel for
    for (int i = 0; i < num_nodes; i++)
        if (node_coords[i * 3 + 2] < 1e-6)
            bc_flag[i] = 1;

    // M用のBSR値配列（3x3ブロックの対角成分のみ）
    double *M_bsr_val = new double[9 * nnz_bcrs](); // 0埋め初期化
#pragma omp parallel for
    for (int k = 0; k < nnz_bcrs; k++)
    {
        M_bsr_val[9 * k + 0] = bmval[k];
        M_bsr_val[9 * k + 4] = bmval[k];
        M_bsr_val[9 * k + 8] = bmval[k];
    }

    extract_bc_correction(num_nodes, brp, bci, bk00, bk01, bk02, bk10, bk11, bk12, bk20, bk21, bk22,
                          bc_flag, bc_corr_00, bc_corr_01, bc_corr_02, bc_corr_10, bc_corr_11, bc_corr_12,
                          bc_corr_20, bc_corr_21, bc_corr_22);
    apply_bc_to_lhs(num_nodes, brp, bci, bk00, bk01, bk02, bk10, bk11, bk12, bk20, bk21, bk22, bc_flag);
    build_block_jacobi(num_nodes, brp, bci, bk00, bk01, bk02, bk10, bk11, bk12, bk20, bk21, bk22,
                       inv00, inv01, inv02, inv10, inv11, inv12, inv20, inv21, inv22);

    // K用のBSR値配列（Row-major順）
    double *K_bsr_val = new double[9 * nnz_bcrs];
#pragma omp parallel for
    for (int k = 0; k < nnz_bcrs; k++)
    {
        K_bsr_val[9 * k + 0] = bk00[k];
        K_bsr_val[9 * k + 1] = bk01[k];
        K_bsr_val[9 * k + 2] = bk02[k];
        K_bsr_val[9 * k + 3] = bk10[k];
        K_bsr_val[9 * k + 4] = bk11[k];
        K_bsr_val[9 * k + 5] = bk12[k];
        K_bsr_val[9 * k + 6] = bk20[k];
        K_bsr_val[9 * k + 7] = bk21[k];
        K_bsr_val[9 * k + 8] = bk22[k];
    }

    // ============================================================
    // GPUへ転送
    // ============================================================
    double xfer_start = MPI_Wtime();

    // BSR行列データ
    int *d_brp = d_alloc_copy(brp, num_nodes + 1);
    int *d_bci = d_alloc_copy(bci, nnz_bcrs);
    double *d_K_bsr_val = d_alloc_copy(K_bsr_val, 9 * nnz_bcrs);
    double *d_M_bsr_val = d_alloc_copy(M_bsr_val, 9 * nnz_bcrs);

    // 境界条件
    int *d_bc_flag = d_alloc_copy(bc_flag, num_nodes);
    double *d_bc_corr_00 = d_alloc_copy(bc_corr_00, num_nodes);
    double *d_bc_corr_01 = d_alloc_copy(bc_corr_01, num_nodes);
    double *d_bc_corr_02 = d_alloc_copy(bc_corr_02, num_nodes);
    double *d_bc_corr_10 = d_alloc_copy(bc_corr_10, num_nodes);
    double *d_bc_corr_11 = d_alloc_copy(bc_corr_11, num_nodes);
    double *d_bc_corr_12 = d_alloc_copy(bc_corr_12, num_nodes);
    double *d_bc_corr_20 = d_alloc_copy(bc_corr_20, num_nodes);
    double *d_bc_corr_21 = d_alloc_copy(bc_corr_21, num_nodes);
    double *d_bc_corr_22 = d_alloc_copy(bc_corr_22, num_nodes);

    // 前処理
    double *d_inv00 = d_alloc_copy(inv00, num_nodes), *d_inv01 = d_alloc_copy(inv01, num_nodes), *d_inv02 = d_alloc_copy(inv02, num_nodes);
    double *d_inv10 = d_alloc_copy(inv10, num_nodes), *d_inv11 = d_alloc_copy(inv11, num_nodes), *d_inv12 = d_alloc_copy(inv12, num_nodes);
    double *d_inv20 = d_alloc_copy(inv20, num_nodes), *d_inv21 = d_alloc_copy(inv21, num_nodes), *d_inv22 = d_alloc_copy(inv22, num_nodes);

    // ベクトル（インターリーブ, size 3*num_nodes）
    double *d_u = d_alloc_zero<double>(N3);    // 変位
    double *d_v = d_alloc_zero<double>(N3);    // 速度
    double *d_a = d_alloc_zero<double>(N3);    // 加速度
    double *d_uprv = d_alloc_zero<double>(N3); // 前ステップ変位
    double *d_rhs = d_alloc_zero<double>(N3);  // 右辺
    double *d_tmp = d_alloc_zero<double>(N3);  // 一時
    double *d_r = d_alloc_zero<double>(N3);    // PCG残差
    double *d_z = d_alloc_zero<double>(N3);
    double *d_p = d_alloc_zero<double>(N3);
    double *d_Ap = d_alloc_zero<double>(N3);

    // 境界条件の値
    double *d_bc_val_u = d_alloc_zero<double>(3);
    double *d_bc_val_v = d_alloc_zero<double>(3);
    double *d_bc_val_a = d_alloc_zero<double>(3);

    // MPI通信
    int *d_send_nodes = d_alloc_copy(send_nodes_vec.data(), total_send);
    double *d_send_buf = d_alloc_zero<double>(3 * total_send); // インターリーブ

    CUDA_CHECK(cudaDeviceSynchronize());
    double xfer_end = MPI_Wtime();
    printf("Data transfer to GPU time: %.2f seconds\n", xfer_end - xfer_start);

    // ============================================================
    // cuSPARSE / cuBLAS ハンドル設定
    // ============================================================
    cusparseHandle_t cusparse_h;
    cublasHandle_t cublas_h;
    CUSPARSE_CHECK(cusparseCreate(&cusparse_h));
    CUBLAS_CHECK(cublasCreate(&cublas_h));

    // --- バージョン確認用コード ---
    int cuda_version;
    cudaRuntimeGetVersion(&cuda_version);
    int cusparse_version;
    cusparseGetVersion(cusparse_h, &cusparse_version);

    if (rank == 0)
    {
        printf("CUDA Runtime Version: %d (Major: %d, Minor: %d)\n",
               cuda_version, cuda_version / 1000, (cuda_version % 100) / 10);
        printf("cuSPARSE Version: %d\n", cusparse_version);
    }
    // ------------------------------

    // K行列ディスクリプタ (BSR: ブロックサイズ3x3, Row-major)
    cusparseSpMatDescr_t K_descr;
    CUSPARSE_CHECK(cusparseCreateBsr(&K_descr, num_nodes, num_nodes, nnz_bcrs, 3, 3,
                                     d_brp, d_bci, d_K_bsr_val,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F,
                                     CUSPARSE_ORDER_ROW));

    // M行列ディスクリプタ
    cusparseSpMatDescr_t M_descr;
    CUSPARSE_CHECK(cusparseCreateBsr(&M_descr, num_nodes, num_nodes, nnz_bcrs, 3, 3,
                                     d_brp, d_bci, d_M_bsr_val,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F,
                                     CUSPARSE_ORDER_ROW));

    // 密ベクトルディスクリプタ（ポインタは後で更新可能）
    cusparseDnVecDescr_t vecX_descr, vecY_descr;
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecX_descr, N3, d_tmp, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecY_descr, N3, d_rhs, CUDA_R_64F));

    // SpMVバッファ確保 (K用)
    double alpha_sp = 1.0, beta_sp = 0.0;
    size_t K_buf_size = 0, M_buf_size = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(cusparse_h, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha_sp, K_descr, vecX_descr, &beta_sp, vecY_descr,
                                           CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &K_buf_size));
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(cusparse_h, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha_sp, M_descr, vecX_descr, &beta_sp, vecY_descr,
                                           CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &M_buf_size));
    size_t buf_size = (K_buf_size > M_buf_size) ? K_buf_size : M_buf_size;
    void *d_spmv_buffer;
    CUDA_CHECK(cudaMalloc(&d_spmv_buffer, buf_size));

    printf("cuSPARSE SpMV buffer size: %zu bytes\n", buf_size);

    // ============================================================
    // タイムステップループ
    // ============================================================
    double *u_record = new double[(num_steps / sample_freq + 1) * 3]();
    double dt_inv = 1.0 / dt, dt_inv2 = dt_inv * dt_inv;
    double h_bc_val_u[3], h_bc_val_v[3], h_bc_val_a[3];

    for (int step = 1; step <= num_steps; step++)
    {
        double t = step * dt;
        h_bc_val_u[0] = 0;
        h_bc_val_u[1] = sin(t);
        h_bc_val_u[2] = 0;
        h_bc_val_v[0] = 0;
        h_bc_val_v[1] = cos(t);
        h_bc_val_v[2] = 0;
        h_bc_val_a[0] = 0;
        h_bc_val_a[1] = -sin(t);
        h_bc_val_a[2] = 0;
        CUDA_CHECK(cudaMemcpy(d_bc_val_u, h_bc_val_u, 3 * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bc_val_v, h_bc_val_v, 3 * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bc_val_a, h_bc_val_a, 3 * sizeof(double), cudaMemcpyHostToDevice));

        // tmp = 4/dt^2 * u + 4/dt * v + a (インターリーブ, 成分独立)
        kernel_build_newmark_tmp<<<GRID(N3), BLOCK_SIZE>>>(
            N3, dt_inv2, dt_inv, d_u, d_v, d_a, d_tmp);

        // rhs = M * tmp (cuSPARSE)
        cusparseDnVecSetValues(vecX_descr, d_tmp);
        cusparseDnVecSetValues(vecY_descr, d_rhs);
        CUSPARSE_CHECK(cusparseSpMV(cusparse_h, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha_sp, M_descr, vecX_descr, &beta_sp, vecY_descr,
                                    CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_spmv_buffer));

        // 外力
        if (force_node >= 0)
        {
            int idx = 3 * force_node + force_dof;
            double h_val;
            CUDA_CHECK(cudaMemcpy(&h_val, &d_rhs[idx], sizeof(double), cudaMemcpyDeviceToHost));
            h_val += force_magnitude;
            CUDA_CHECK(cudaMemcpy(&d_rhs[idx], &h_val, sizeof(double), cudaMemcpyHostToDevice));
        }

        // BC適用
        kernel_apply_bc_corr<<<GRID(num_nodes), BLOCK_SIZE>>>(
            num_nodes, d_bc_val_u,
            d_bc_corr_00, d_bc_corr_01, d_bc_corr_02,
            d_bc_corr_10, d_bc_corr_11, d_bc_corr_12,
            d_bc_corr_20, d_bc_corr_21, d_bc_corr_22, d_rhs);
        kernel_apply_bc_constrained<<<GRID(num_nodes), BLOCK_SIZE>>>(
            num_nodes, d_bc_flag, d_bc_val_u, d_rhs);
        CUDA_CHECK(cudaDeviceSynchronize());

        // PCGソルバー
        int iter = pcg_solve_cusparse(
            cusparse_h, cublas_h, K_descr, vecX_descr, vecY_descr, d_spmv_buffer,
            requests, num_neighbors, neighbor_ranks, recv_starts, recv_counts,
            send_starts, send_counts, d_send_nodes, d_send_buf,
            num_inner, num_owned, num_nodes,
            d_inv00, d_inv01, d_inv02, d_inv10, d_inv11, d_inv12, d_inv20, d_inv21, d_inv22,
            d_rhs, d_u, d_r, d_z, d_p, d_Ap, 1e-8, N3);

        if (rank == 0)
            std::cout << "Step " << step << ", PCG iterations: " << iter << std::endl;

        // Newmark-β更新
        kernel_newmark_update<<<GRID(num_nodes), BLOCK_SIZE>>>(
            num_nodes, dt, dt_inv, dt_inv2,
            d_bc_flag, d_bc_val_u, d_bc_val_v, d_bc_val_a,
            d_u, d_uprv, d_v, d_a);

        // サンプリング
        if (step % sample_freq == 0 && target_node >= 0)
        {
            CUDA_CHECK(cudaMemcpy(&u_record[(step / sample_freq) * 3],
                                  &d_u[3 * target_node], 3 * sizeof(double), cudaMemcpyDeviceToHost));
        }
    }

    // ============================================================
    // 出力
    // ============================================================
    if (target_node >= 0)
    {
        time_t now = time(nullptr);
        struct tm *lt = localtime(&now);
        char ts[64];
        strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", lt);
        char odir[256];
        sprintf(odir, "results/%s", ts);
        mkdir("results", 0755);
        mkdir(odir, 0755);
        char csv[512];
        sprintf(csv, "%s/target_disp.csv", odir);
        write_node_disp_csv(csv, u_record, num_steps, sample_freq, dt);
        printf("Output: %s/\nCSV: %s\n", odir, csv);
    }

    // ============================================================
    // 後片付け
    // ============================================================
    cusparseDestroySpMat(K_descr);
    cusparseDestroySpMat(M_descr);
    cusparseDestroyDnVec(vecX_descr);
    cusparseDestroyDnVec(vecY_descr);
    cusparseDestroy(cusparse_h);
    cublasDestroy(cublas_h);
    cudaFree(d_spmv_buffer);
    cudaFree(d_brp);
    cudaFree(d_bci);
    cudaFree(d_K_bsr_val);
    cudaFree(d_M_bsr_val);
    cudaFree(d_bc_flag);
    cudaFree(d_bc_corr_00);
    cudaFree(d_bc_corr_01);
    cudaFree(d_bc_corr_02);
    cudaFree(d_bc_corr_10);
    cudaFree(d_bc_corr_11);
    cudaFree(d_bc_corr_12);
    cudaFree(d_bc_corr_20);
    cudaFree(d_bc_corr_21);
    cudaFree(d_bc_corr_22);
    cudaFree(d_inv00);
    cudaFree(d_inv01);
    cudaFree(d_inv02);
    cudaFree(d_inv10);
    cudaFree(d_inv11);
    cudaFree(d_inv12);
    cudaFree(d_inv20);
    cudaFree(d_inv21);
    cudaFree(d_inv22);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_a);
    cudaFree(d_uprv);
    cudaFree(d_rhs);
    cudaFree(d_tmp);
    cudaFree(d_r);
    cudaFree(d_z);
    cudaFree(d_p);
    cudaFree(d_Ap);
    cudaFree(d_bc_val_u);
    cudaFree(d_bc_val_v);
    cudaFree(d_bc_val_a);
    cudaFree(d_send_nodes);
    cudaFree(d_send_buf);

    delete[] K_bsr_val;
    delete[] M_bsr_val;
    delete[] bk00;
    delete[] bk01;
    delete[] bk02;
    delete[] bk10;
    delete[] bk11;
    delete[] bk12;
    delete[] bk20;
    delete[] bk21;
    delete[] bk22;
    delete[] bmval;
    delete[] brp;
    delete[] bci;
    delete[] bc_flag;
    delete[] bc_corr_00;
    delete[] bc_corr_01;
    delete[] bc_corr_02;
    delete[] bc_corr_10;
    delete[] bc_corr_11;
    delete[] bc_corr_12;
    delete[] bc_corr_20;
    delete[] bc_corr_21;
    delete[] bc_corr_22;
    delete[] inv00;
    delete[] inv01;
    delete[] inv02;
    delete[] inv10;
    delete[] inv11;
    delete[] inv12;
    delete[] inv20;
    delete[] inv21;
    delete[] inv22;
    delete[] u_record;
    delete[] neighbor_ranks;
    delete[] recv_starts;
    delete[] recv_counts;
    delete[] send_starts;
    delete[] send_counts;

    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time, max_elapsed;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
        printf("Total elapsed time: %.2f seconds\n", max_elapsed);

    MPI_Finalize();
    return 0;
}

// ============================================================
// CPU側の関数（手書き版と同一）
// ============================================================
void calculate_dN(double dN[30], double r, double s, double t)
{
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
}

void gauss_integrate(double dN0[30], double dN1[30], double dN2[30], double dN3[30])
{
    double a = (5.0 - std::sqrt(5.0)) / 20.0, b = (5.0 + 3.0 * std::sqrt(5.0)) / 20.0;
    double gp[4][3] = {{a, a, a}, {b, a, a}, {a, b, a}, {a, a, b}};
    calculate_dN(dN0, gp[0][0], gp[0][1], gp[0][2]);
    calculate_dN(dN1, gp[1][0], gp[1][1], gp[1][2]);
    calculate_dN(dN2, gp[2][0], gp[2][1], gp[2][2]);
    calculate_dN(dN3, gp[3][0], gp[3][1], gp[3][2]);
}

double inverse_3_3_mat(double m[9], double inv[9])
{
    double det = m[0] * (m[4] * m[8] - m[5] * m[7]) + m[1] * (m[5] * m[6] - m[3] * m[8]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
    double id = 1.0 / det;
    inv[0] = (m[4] * m[8] - m[5] * m[7]) * id;
    inv[1] = (m[2] * m[7] - m[1] * m[8]) * id;
    inv[2] = (m[1] * m[5] - m[2] * m[4]) * id;
    inv[3] = (m[5] * m[6] - m[3] * m[8]) * id;
    inv[4] = (m[0] * m[8] - m[2] * m[6]) * id;
    inv[5] = (m[2] * m[3] - m[0] * m[5]) * id;
    inv[6] = (m[3] * m[7] - m[4] * m[6]) * id;
    inv[7] = (m[1] * m[6] - m[0] * m[7]) * id;
    inv[8] = (m[0] * m[4] - m[1] * m[3]) * id;
    return det;
}

void construct_mat(double *nc, int *en, int ne, double dN0[30], double dN1[30], double dN2[30], double dN3[30],
                   double lam, double mu, double rho, double dt, double *kv, double *mv, int *cr, int *cc)
{
#pragma omp parallel for
    for (int elem = 0; elem < ne; elem++)
    {
        int ln[10];
        double lc[12];
        double jac[9], det, ijac[9];
        double ld0[30] = {}, ld1[30] = {}, ld2[30] = {}, ld3[30] = {};
        double lm[100] = {
            6.0 / 2520, 1.0 / 2520, 1.0 / 2520, 1.0 / 2520, -4.0 / 2520, -6.0 / 2520, -4.0 / 2520, -4.0 / 2520, -6.0 / 2520, -6.0 / 2520,
            1.0 / 2520, 6.0 / 2520, 1.0 / 2520, 1.0 / 2520, -4.0 / 2520, -4.0 / 2520, -6.0 / 2520, -6.0 / 2520, -4.0 / 2520, -6.0 / 2520,
            1.0 / 2520, 1.0 / 2520, 6.0 / 2520, 1.0 / 2520, -6.0 / 2520, -4.0 / 2520, -4.0 / 2520, -6.0 / 2520, -6.0 / 2520, -4.0 / 2520,
            1.0 / 2520, 1.0 / 2520, 1.0 / 2520, 6.0 / 2520, -6.0 / 2520, -6.0 / 2520, -6.0 / 2520, -4.0 / 2520, -4.0 / 2520, -4.0 / 2520,
            -4.0 / 2520, -4.0 / 2520, -6.0 / 2520, -6.0 / 2520, 32.0 / 2520, 16.0 / 2520, 16.0 / 2520, 16.0 / 2520, 16.0 / 2520, 8.0 / 2520,
            -6.0 / 2520, -4.0 / 2520, -4.0 / 2520, -6.0 / 2520, 16.0 / 2520, 32.0 / 2520, 16.0 / 2520, 8.0 / 2520, 16.0 / 2520, 16.0 / 2520,
            -4.0 / 2520, -6.0 / 2520, -4.0 / 2520, -6.0 / 2520, 16.0 / 2520, 16.0 / 2520, 32.0 / 2520, 16.0 / 2520, 8.0 / 2520, 16.0 / 2520,
            -4.0 / 2520, -6.0 / 2520, -6.0 / 2520, -4.0 / 2520, 16.0 / 2520, 8.0 / 2520, 16.0 / 2520, 32.0 / 2520, 16.0 / 2520, 16.0 / 2520,
            -6.0 / 2520, -4.0 / 2520, -6.0 / 2520, -4.0 / 2520, 16.0 / 2520, 16.0 / 2520, 8.0 / 2520, 16.0 / 2520, 32.0 / 2520, 16.0 / 2520,
            -6.0 / 2520, -6.0 / 2520, -4.0 / 2520, -4.0 / 2520, 8.0 / 2520, 16.0 / 2520, 16.0 / 2520, 16.0 / 2520, 16.0 / 2520, 32.0 / 2520};
        for (int i = 0; i < 10; i++)
            ln[i] = en[elem * 10 + i];
        for (int i = 0; i < 4; i++)
        {
            lc[3 * i] = nc[ln[i] * 3];
            lc[3 * i + 1] = nc[ln[i] * 3 + 1];
            lc[3 * i + 2] = nc[ln[i] * 3 + 2];
        }
        for (int i = 1; i < 4; i++)
        {
            jac[3 * (i - 1)] = lc[3 * i] - lc[0];
            jac[3 * (i - 1) + 1] = lc[3 * i + 1] - lc[1];
            jac[3 * (i - 1) + 2] = lc[3 * i + 2] - lc[2];
        }
        det = inverse_3_3_mat(jac, ijac);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 10; j++)
            {
                ld0[10 * i + j] = ijac[3 * i] * dN0[j] + ijac[3 * i + 1] * dN0[10 + j] + ijac[3 * i + 2] * dN0[20 + j];
                ld1[10 * i + j] = ijac[3 * i] * dN1[j] + ijac[3 * i + 1] * dN1[10 + j] + ijac[3 * i + 2] * dN1[20 + j];
                ld2[10 * i + j] = ijac[3 * i] * dN2[j] + ijac[3 * i + 1] * dN2[10 + j] + ijac[3 * i + 2] * dN2[20 + j];
                ld3[10 * i + j] = ijac[3 * i] * dN3[j] + ijac[3 * i + 1] * dN3[10 + j] + ijac[3 * i + 2] * dN3[20 + j];
            }
        for (int i = 0; i < 10; i++)
            for (int j = 0; j < 10; j++)
            {
                double lk[3][3] = {};
                for (int a = 0; a < 3; a++)
                    for (int b = 0; b < 3; b++)
                    {
                        lk[a][b] = lam * ld0[10 * a + i] * ld0[10 * b + j] + mu * ld0[10 * a + j] * ld0[10 * b + i];
                        lk[a][b] += lam * ld1[10 * a + i] * ld1[10 * b + j] + mu * ld1[10 * a + j] * ld1[10 * b + i];
                        lk[a][b] += lam * ld2[10 * a + i] * ld2[10 * b + j] + mu * ld2[10 * a + j] * ld2[10 * b + i];
                        lk[a][b] += lam * ld3[10 * a + i] * ld3[10 * b + j] + mu * ld3[10 * a + j] * ld3[10 * b + i];
                    }
                double dp0 = ld0[i] * ld0[j] + ld0[10 + i] * ld0[10 + j] + ld0[20 + i] * ld0[20 + j];
                double dp1 = ld1[i] * ld1[j] + ld1[10 + i] * ld1[10 + j] + ld1[20 + i] * ld1[20 + j];
                double dp2 = ld2[i] * ld2[j] + ld2[10 + i] * ld2[10 + j] + ld2[20 + i] * ld2[20 + j];
                double dp3 = ld3[i] * ld3[j] + ld3[10 + i] * ld3[10 + j] + ld3[20 + i] * ld3[20 + j];
                for (int a = 0; a < 3; a++)
                {
                    lk[a][a] += mu * (dp0 + dp1 + dp2 + dp3);
                    lk[a][a] += 24.0 * rho * lm[10 * i + j] * 4.0 / dt / dt;
                }
                for (int a = 0; a < 3; a++)
                    for (int b = 0; b < 3; b++)
                        kv[9 * (elem * 100 + i * 10 + j) + 3 * a + b] = lk[a][b] * det / 24.0;
                mv[elem * 100 + i * 10 + j] = rho * lm[10 * i + j] * det;
                cr[elem * 100 + i * 10 + j] = ln[i];
                cc[elem * 100 + i * 10 + j] = ln[j];
            }
    }
}

int sort_and_merge_bcoo(int nnz_coo, int nn, int *cr, int *cc, double *kv, double *mv)
{
    int *off = new int[nn + 1]();
    for (int k = 0; k < nnz_coo; k++)
        off[cr[k] + 1]++;
    for (int i = 1; i <= nn; i++)
        off[i] += off[i - 1];
    int *wc = new int[nnz_coo];
    double *wk = new double[nnz_coo * 9];
    double *wm = new double[nnz_coo];
    int *pos = new int[nn];
#pragma omp parallel for
    for (int i = 0; i < nn; i++)
        pos[i] = off[i];
    for (int k = 0; k < nnz_coo; k++)
    {
        int r = cr[k];
        int p = pos[r]++;
        wc[p] = cc[k];
        for (int a = 0; a < 9; a++)
            wk[9 * p + a] = kv[9 * k + a];
        wm[p] = mv[k];
    }
    delete[] pos;
    int nnz = 0;
    for (int i = 0; i < nn; i++)
    {
        int s = off[i], e = off[i + 1];
        if (s == e)
            continue;
        for (int a = s + 1; a < e; a++)
        {
            int tc = wc[a];
            double tk[9];
            double tm = wm[a];
            for (int p = 0; p < 9; p++)
                tk[p] = wk[9 * a + p];
            int b = a - 1;
            while (b >= s && wc[b] > tc)
            {
                wc[b + 1] = wc[b];
                for (int p = 0; p < 9; p++)
                    wk[9 * (b + 1) + p] = wk[9 * b + p];
                wm[b + 1] = wm[b];
                b--;
            }
            wc[b + 1] = tc;
            for (int p = 0; p < 9; p++)
                wk[9 * (b + 1) + p] = tk[p];
            wm[b + 1] = tm;
        }
        int w = nnz;
        cr[w] = i;
        cc[w] = wc[s];
        for (int a = 0; a < 9; a++)
            kv[9 * w + a] = wk[9 * s + a];
        mv[w] = wm[s];
        for (int k = s + 1; k < e; k++)
        {
            if (wc[k] == cc[w])
            {
                for (int a = 0; a < 9; a++)
                    kv[9 * w + a] += wk[9 * k + a];
                mv[w] += wm[k];
            }
            else
            {
                w++;
                cr[w] = i;
                cc[w] = wc[k];
                for (int a = 0; a < 9; a++)
                    kv[9 * w + a] = wk[9 * k + a];
                mv[w] = wm[k];
            }
        }
        w++;
        nnz = w;
    }
    delete[] wc;
    delete[] wk;
    delete[] wm;
    delete[] off;
    return nnz;
}

void build_bcrs(int *cr, int *cc, double *kv, double *mv, int nnz, int nn, int *rp, int *ci,
                double *k00, double *k01, double *k02, double *k10, double *k11, double *k12,
                double *k20, double *k21, double *k22, double *mval)
{
#pragma omp parallel for
    for (int k = 0; k < nnz; k++)
    {
        ci[k] = cc[k];
        k00[k] = kv[9 * k];
        k01[k] = kv[9 * k + 1];
        k02[k] = kv[9 * k + 2];
        k10[k] = kv[9 * k + 3];
        k11[k] = kv[9 * k + 4];
        k12[k] = kv[9 * k + 5];
        k20[k] = kv[9 * k + 6];
        k21[k] = kv[9 * k + 7];
        k22[k] = kv[9 * k + 8];
        mval[k] = mv[k];
    }
#pragma omp parallel for
    for (int i = 0; i <= nn; i++)
        rp[i] = 0;
    for (int k = 0; k < nnz; k++)
        rp[cr[k] + 1]++;
    for (int i = 1; i <= nn; i++)
        rp[i] += rp[i - 1];
}

void extract_bc_correction(int nn, int *rp, int *ci, double *k00, double *k01, double *k02,
                           double *k10, double *k11, double *k12, double *k20, double *k21, double *k22, int *bf,
                           double *c00, double *c01, double *c02, double *c10, double *c11, double *c12, double *c20, double *c21, double *c22)
{
#pragma omp parallel for
    for (int i = 0; i < nn; i++)
        c00[i] = c01[i] = c02[i] = c10[i] = c11[i] = c12[i] = c20[i] = c21[i] = c22[i] = 0;
#pragma omp parallel for
    for (int i = 0; i < nn; i++)
    {
        if (bf[i])
            continue;
        for (int p = rp[i]; p < rp[i + 1]; p++)
        {
            int j = ci[p];
            if (!bf[j])
                continue;
            c00[i] += k00[p];
            c01[i] += k01[p];
            c02[i] += k02[p];
            c10[i] += k10[p];
            c11[i] += k11[p];
            c12[i] += k12[p];
            c20[i] += k20[p];
            c21[i] += k21[p];
            c22[i] += k22[p];
        }
    }
}

void apply_bc_to_lhs(int nn, int *rp, int *ci, double *k00, double *k01, double *k02,
                     double *k10, double *k11, double *k12, double *k20, double *k21, double *k22, int *bf)
{
#pragma omp parallel for
    for (int i = 0; i < nn; i++)
        for (int p = rp[i]; p < rp[i + 1]; p++)
        {
            int j = ci[p];
            if (bf[i] || bf[j])
                k00[p] = k01[p] = k02[p] = k10[p] = k11[p] = k12[p] = k20[p] = k21[p] = k22[p] = 0;
            if (bf[i] && i == j)
                k00[p] = k11[p] = k22[p] = 1.0;
        }
}

void build_block_jacobi(int nn, int *rp, int *ci, double *k00, double *k01, double *k02,
                        double *k10, double *k11, double *k12, double *k20, double *k21, double *k22,
                        double *i00, double *i01, double *i02, double *i10, double *i11, double *i12, double *i20, double *i21, double *i22)
{
#pragma omp parallel for
    for (int i = 0; i < nn; i++)
    {
        double d[9] = {}, id[9] = {};
        for (int p = rp[i]; p < rp[i + 1]; p++)
            if (ci[p] == i)
            {
                d[0] = k00[p];
                d[1] = k01[p];
                d[2] = k02[p];
                d[3] = k10[p];
                d[4] = k11[p];
                d[5] = k12[p];
                d[6] = k20[p];
                d[7] = k21[p];
                d[8] = k22[p];
                break;
            }
        inverse_3_3_mat(d, id);
        i00[i] = id[0];
        i01[i] = id[1];
        i02[i] = id[2];
        i10[i] = id[3];
        i11[i] = id[4];
        i12[i] = id[5];
        i20[i] = id[6];
        i21[i] = id[7];
        i22[i] = id[8];
    }
}

void write_vtk_displacement(const char *fn, double *nc, int nn, int *en, int ne, double *d, double t)
{
    FILE *fp = fopen(fn, "w");
    fprintf(fp, "# vtk DataFile Version 2.0\nFEM displacement result\nASCII\nDATASET UNSTRUCTURED_GRID\n");
    fprintf(fp, "\nFIELD FieldData 1\nTOTALTIME 1 1 double\n%.10e\nPOINTS %d double\n", t, nn);
    for (int i = 0; i < nn; i++)
        fprintf(fp, "%.15e %.15e %.15e\n", nc[i * 3], nc[i * 3 + 1], nc[i * 3 + 2]);
    fprintf(fp, "CELLS %d %d\n", ne, ne * 11);
    for (int e = 0; e < ne; e++)
    {
        fprintf(fp, "10");
        for (int i = 0; i < 10; i++)
            fprintf(fp, " %d", en[e * 10 + i]);
        fprintf(fp, "\n");
    }
    fprintf(fp, "CELL_TYPES %d\n", ne);
    for (int e = 0; e < ne; e++)
        fprintf(fp, "24\n");
    fprintf(fp, "POINT_DATA %d\nSCALARS NODE_ID int 1\nLOOKUP_TABLE default\n", nn);
    for (int i = 0; i < nn; i++)
        fprintf(fp, "%d\n", i + 1);
    fprintf(fp, "VECTORS DISPLACEMENT double\n");
    for (int i = 0; i < nn; i++)
        fprintf(fp, "%.15e %.15e %.15e\n", d[i * 3], d[i * 3 + 1], d[i * 3 + 2]);
    fclose(fp);
}

void write_node_disp_csv(const char *fn, double *u, int ns, int sf, double dt)
{
    FILE *fp = fopen(fn, "w");
    fprintf(fp, "time,disp_x,disp_y,disp_z,disp_mag\n");
    for (int s = 0; s <= ns; s += sf)
    {
        double t = s * dt;
        int i = s / sf;
        double x = u[i * 3], y = u[i * 3 + 1], z = u[i * 3 + 2];
        fprintf(fp, "%.10e,%.10e,%.10e,%.10e,%.10e\n", t, x, y, z, sqrt(x * x + y * y + z * z));
    }
    fclose(fp);
}
