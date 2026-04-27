/*実行コマンド
cd /data3/kusumoto/elastodynamics-fem/
module load nvhpc/25.1
nvcc main_recursive.cu msh_reader.cpp -Xcompiler -fopenmp -ccbin mpicxx -arch=sm_80 -lineinfo
sm_80はA100向け。H100、GH200ならsm_90。
mpirun -np 4 ./a.out
増分解析バージョン
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
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>

// ============================================================
// CUDA エラーチェックマクロ
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

// ============================================================
// CUDA カーネル用の定数
// ============================================================
static const int BLOCK_SIZE = 256;

// ============================================================
// デバイスポインタをまとめる構造体
// ============================================================
struct DeviceData
{
    // パラメタ
    double *node_coords;
    double *ref_node_coords;
    int *ele_nodes;

    // COO行列
    int *coo_row, *coo_col;
    double *kmat_coo_val, *mmat_coo_val, *kemat_coo_val;

    // BCRS 行列
    int *row_ptr, *col_ind;
    double *kval_00, *kval_01, *kval_02;
    double *kval_10, *kval_11, *kval_12;
    double *kval_20, *kval_21, *kval_22;
    double *mval;
    double *keval_00, *keval_01, *keval_02;
    double *keval_10, *keval_11, *keval_12;
    double *keval_20, *keval_21, *keval_22;

    // 境界条件
    int *bc_flag;
    double *bc_corr_00, *bc_corr_01, *bc_corr_02;
    double *bc_corr_10, *bc_corr_11, *bc_corr_12;
    double *bc_corr_20, *bc_corr_21, *bc_corr_22;

    // ブロックヤコビ前処理
    double *inv_diag_00, *inv_diag_01, *inv_diag_02;
    double *inv_diag_10, *inv_diag_11, *inv_diag_12;
    double *inv_diag_20, *inv_diag_21, *inv_diag_22;

    // 変位・速度・加速度
    double *u_tmp_0, *u_tmp_1, *u_tmp_2;
    double *v_tmp_0, *v_tmp_1, *v_tmp_2;
    double *a_tmp_0, *a_tmp_1, *a_tmp_2;
    double *u_prv_0, *u_prv_1, *u_prv_2;
    double *delta_u_0, *delta_u_1, *delta_u_2;

    // 右辺・一時ベクトル
    double *rhs_0, *rhs_1, *rhs_2;
    double *tmp_0, *tmp_1, *tmp_2;
    double *rhs_original_0, *rhs_original_1, *rhs_original_2;
    double *f_int_0, *f_int_1, *f_int_2;

    // PCG ベクトル
    double *r_0, *r_1, *r_2;
    double *z_0, *z_1, *z_2;
    double *p_0, *p_1, *p_2;
    double *Ap_0, *Ap_1, *Ap_2;

    // 境界条件の値
    double *bc_val_u;

    // MPI 通信用
    int *send_nodes;
    double *send_buffer_0, *send_buffer_1, *send_buffer_2;

    // リダクション用
    double *d_reduce; // 小さいバッファ（3要素程度）
};

// ============================================================
// 定数をGPUのconstantメモリに乗せる
// ============================================================

__constant__ double c_mmat[100] = {
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
__constant__ double c_dN0[30];
__constant__ double c_dN1[30];
__constant__ double c_dN2[30];
__constant__ double c_dN3[30];
__constant__ double c_N0[10];
__constant__ double c_N1[10];
__constant__ double c_N2[10];
__constant__ double c_N3[10];

// ============================================================
// CUDA カーネル
// ============================================================

// x^(i+1)=x^(i)+ratio*u^(i)を基準にKを計算し、初期配置でMを計算するカーネル
// node_coordsはx^(i)、ref_node_coordsはx^(0)、delta_u_0、delta_u_1、delta_u_2がdu^(i)を表す。
// つまり、変形前のref_node_coordsを入れないといけない。
__global__ void construct_mat(
    const double *__restrict__ node_coords,
    const double *__restrict__ ref_node_coords,
    const double *__restrict__ delta_u_0,
    const double *__restrict__ delta_u_1,
    const double *__restrict__ delta_u_2,
    const double ratio,
    const int *__restrict__ ele_nodes,
    const int num_elements,
    const double lambda,
    const double mu,
    const double rho,
    const double dt,
    double *__restrict__ kmat_coo_val,
    double *__restrict__ mmat_coo_val,
    double *__restrict__ kemat_coo_val,
    int *__restrict__ coo_row,
    int *__restrict__ coo_col)
{
    const int elem = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem >= num_elements)
        return;

    // ノードインデックス
    int nidx[10];
#pragma unroll
    for (int i = 0; i < 10; i++)
        nidx[i] = ele_nodes[elem * 10 + i];

    // ヤコビアン
    double invJ[9], detJ, detJ_m;
    {
        double J[9], x0[3];
        x0[0] = node_coords[nidx[0] * 3 + 0] + ratio * delta_u_0[nidx[0]];
        x0[1] = node_coords[nidx[0] * 3 + 1] + ratio * delta_u_1[nidx[0]];
        x0[2] = node_coords[nidx[0] * 3 + 2] + ratio * delta_u_2[nidx[0]];
        for (int i = 0; i < 3; i++)
        {
            J[3 * i + 0] = node_coords[nidx[i + 1] * 3 + 0] + ratio * delta_u_0[nidx[i + 1]] - x0[0];
            J[3 * i + 1] = node_coords[nidx[i + 1] * 3 + 1] + ratio * delta_u_1[nidx[i + 1]] - x0[1];
            J[3 * i + 2] = node_coords[nidx[i + 1] * 3 + 2] + ratio * delta_u_2[nidx[i + 1]] - x0[2];
        }
        detJ = J[0] * (J[4] * J[8] - J[5] * J[7]) +
               J[1] * (J[5] * J[6] - J[3] * J[8]) +
               J[2] * (J[3] * J[7] - J[4] * J[6]);
        invJ[0] = (J[4] * J[8] - J[5] * J[7]) / detJ;
        invJ[1] = (J[2] * J[7] - J[1] * J[8]) / detJ;
        invJ[2] = (J[1] * J[5] - J[2] * J[4]) / detJ;
        invJ[3] = (J[5] * J[6] - J[3] * J[8]) / detJ;
        invJ[4] = (J[0] * J[8] - J[2] * J[6]) / detJ;
        invJ[5] = (J[2] * J[3] - J[0] * J[5]) / detJ;
        invJ[6] = (J[3] * J[7] - J[4] * J[6]) / detJ;
        invJ[7] = (J[1] * J[6] - J[0] * J[7]) / detJ;
        invJ[8] = (J[0] * J[4] - J[1] * J[3]) / detJ;

        for (int d = 0; d < 3; d++)
            x0[d] = ref_node_coords[nidx[0] * 3 + d];
        for (int i = 0; i < 3; i++)
            for (int d = 0; d < 3; d++)
                J[3 * i + d] = ref_node_coords[nidx[i + 1] * 3 + d] - x0[d];
        detJ_m = J[0] * (J[4] * J[8] - J[5] * J[7]) +
                 J[1] * (J[5] * J[6] - J[3] * J[8]) +
                 J[2] * (J[3] * J[7] - J[4] * J[6]);
    }
    const double detJ_24 = detJ / 24.0;
    const double mass_coeff = 4.0 * rho / (dt * dt) * detJ_m;

    // 形状関数勾配の物理座標変換
    double ldN0[30], ldN1[30], ldN2[30], ldN3[30];
    for (int d = 0; d < 3; d++)
    {
        const double c0 = invJ[3 * d], c1 = invJ[3 * d + 1], c2 = invJ[3 * d + 2];
#pragma unroll
        for (int n = 0; n < 10; n++)
        {
            ldN0[d * 10 + n] = c0 * c_dN0[n] + c1 * c_dN0[10 + n] + c2 * c_dN0[20 + n];
            ldN1[d * 10 + n] = c0 * c_dN1[n] + c1 * c_dN1[10 + n] + c2 * c_dN1[20 + n];
            ldN2[d * 10 + n] = c0 * c_dN2[n] + c1 * c_dN2[10 + n] + c2 * c_dN2[20 + n];
            ldN3[d * 10 + n] = c0 * c_dN3[n] + c1 * c_dN3[10 + n] + c2 * c_dN3[20 + n];
        }
    }

    // 要素行列の組み立て
    const int base = elem * 100;

    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            const int ij = i * 10 + j;
            double kab[9] = {0.0};

// 4ガウス点の寄与を加算 (手動展開)
#pragma unroll
            for (int gp = 0; gp < 4; gp++)
            {
                const double *ldn = (gp == 0) ? ldN0 : (gp == 1) ? ldN1
                                                   : (gp == 2)   ? ldN2
                                                                 : ldN3;
                const double gi[3] = {ldn[i], ldn[10 + i], ldn[20 + i]};
                const double gj[3] = {ldn[j], ldn[10 + j], ldn[20 + j]};
                const double dot = gi[0] * gj[0] + gi[1] * gj[1] + gi[2] * gj[2];

#pragma unroll
                for (int a = 0; a < 3; a++)
                {
#pragma unroll
                    for (int b = 0; b < 3; b++)
                    {
                        double val = lambda * gi[a] * gj[b] + mu * gi[b] * gj[a];
                        if (a == b)
                            val += mu * dot;
                        kab[3 * a + b] += val;
                    }
                }
            }

// 書き出し
#pragma unroll
            for (int k = 0; k < 9; k++)
            {
                kmat_coo_val[9 * (base + ij) + k] = kab[k] * detJ_24;
                kemat_coo_val[9 * (base + ij) + k] = kab[k] * detJ_24;
            }

            // 質量行列の対角寄与
            const double mij = c_mmat[ij];
#pragma unroll
            for (int a = 0; a < 3; a++)
                kemat_coo_val[9 * (base + ij) + 3 * a + a] += mass_coeff * mij;

            mmat_coo_val[base + ij] = rho * mij * detJ_m;
            coo_row[base + ij] = nidx[i];
            coo_col[base + ij] = nidx[j];
        }
    }
}

// HRZ法による集中質量行列バージョン。
// mmat_coo_valは対角成分(i==j)のみ非ゼロ、kemat_coo_valの質量寄与も対角ブロックのみ。
// 署名はconstruct_matと同一なので呼び出し側の変更は最小限。
//
// HRZスケール係数の導出:
//   reference tet体積 = 1/6
//   sum(diag(c_mmat)) = (4*6 + 6*32)/2520 = 216/2520
//   alpha = (1/6) / (216/2520) = 35/18
__global__ void construct_mat_lumped(
    const double *__restrict__ node_coords,
    const double *__restrict__ ref_node_coords,
    const double *__restrict__ delta_u_0,
    const double *__restrict__ delta_u_1,
    const double *__restrict__ delta_u_2,
    const double ratio,
    const int *__restrict__ ele_nodes,
    const int num_elements,
    const double lambda,
    const double mu,
    const double rho,
    const double dt,
    double *__restrict__ kmat_coo_val,
    double *__restrict__ mmat_coo_val,
    double *__restrict__ kemat_coo_val,
    int *__restrict__ coo_row,
    int *__restrict__ coo_col)
{
    const int elem = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem >= num_elements)
        return;

    int nidx[10];
#pragma unroll
    for (int i = 0; i < 10; i++)
        nidx[i] = ele_nodes[elem * 10 + i];

    double invJ[9], detJ, detJ_m;
    {
        double J[9], x0[3];
        x0[0] = node_coords[nidx[0] * 3 + 0] + ratio * delta_u_0[nidx[0]];
        x0[1] = node_coords[nidx[0] * 3 + 1] + ratio * delta_u_1[nidx[0]];
        x0[2] = node_coords[nidx[0] * 3 + 2] + ratio * delta_u_2[nidx[0]];
        for (int i = 0; i < 3; i++)
        {
            J[3 * i + 0] = node_coords[nidx[i + 1] * 3 + 0] + ratio * delta_u_0[nidx[i + 1]] - x0[0];
            J[3 * i + 1] = node_coords[nidx[i + 1] * 3 + 1] + ratio * delta_u_1[nidx[i + 1]] - x0[1];
            J[3 * i + 2] = node_coords[nidx[i + 1] * 3 + 2] + ratio * delta_u_2[nidx[i + 1]] - x0[2];
        }
        detJ = J[0] * (J[4] * J[8] - J[5] * J[7]) +
               J[1] * (J[5] * J[6] - J[3] * J[8]) +
               J[2] * (J[3] * J[7] - J[4] * J[6]);
        invJ[0] = (J[4] * J[8] - J[5] * J[7]) / detJ;
        invJ[1] = (J[2] * J[7] - J[1] * J[8]) / detJ;
        invJ[2] = (J[1] * J[5] - J[2] * J[4]) / detJ;
        invJ[3] = (J[5] * J[6] - J[3] * J[8]) / detJ;
        invJ[4] = (J[0] * J[8] - J[2] * J[6]) / detJ;
        invJ[5] = (J[2] * J[3] - J[0] * J[5]) / detJ;
        invJ[6] = (J[3] * J[7] - J[4] * J[6]) / detJ;
        invJ[7] = (J[1] * J[6] - J[0] * J[7]) / detJ;
        invJ[8] = (J[0] * J[4] - J[1] * J[3]) / detJ;

        for (int d = 0; d < 3; d++)
            x0[d] = ref_node_coords[nidx[0] * 3 + d];
        for (int i = 0; i < 3; i++)
            for (int d = 0; d < 3; d++)
                J[3 * i + d] = ref_node_coords[nidx[i + 1] * 3 + d] - x0[d];
        detJ_m = J[0] * (J[4] * J[8] - J[5] * J[7]) +
                 J[1] * (J[5] * J[6] - J[3] * J[8]) +
                 J[2] * (J[3] * J[7] - J[4] * J[6]);
    }
    const double detJ_24 = detJ / 24.0;
    // HRZスケール係数 alpha = (1/6) / (216/2520) = 35/18
    const double mass_coeff = 4.0 * rho / (dt * dt) * detJ_m * (35.0 / 18.0);

    double ldN0[30], ldN1[30], ldN2[30], ldN3[30];
    for (int d = 0; d < 3; d++)
    {
        const double c0 = invJ[3 * d], c1 = invJ[3 * d + 1], c2 = invJ[3 * d + 2];
#pragma unroll
        for (int n = 0; n < 10; n++)
        {
            ldN0[d * 10 + n] = c0 * c_dN0[n] + c1 * c_dN0[10 + n] + c2 * c_dN0[20 + n];
            ldN1[d * 10 + n] = c0 * c_dN1[n] + c1 * c_dN1[10 + n] + c2 * c_dN1[20 + n];
            ldN2[d * 10 + n] = c0 * c_dN2[n] + c1 * c_dN2[10 + n] + c2 * c_dN2[20 + n];
            ldN3[d * 10 + n] = c0 * c_dN3[n] + c1 * c_dN3[10 + n] + c2 * c_dN3[20 + n];
        }
    }

    const int base = elem * 100;

    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            const int ij = i * 10 + j;
            double kab[9] = {0.0};

#pragma unroll
            for (int gp = 0; gp < 4; gp++)
            {
                const double *ldn = (gp == 0) ? ldN0 : (gp == 1) ? ldN1
                                                   : (gp == 2)   ? ldN2
                                                                 : ldN3;
                const double gi[3] = {ldn[i], ldn[10 + i], ldn[20 + i]};
                const double gj[3] = {ldn[j], ldn[10 + j], ldn[20 + j]};
                const double dot = gi[0] * gj[0] + gi[1] * gj[1] + gi[2] * gj[2];

#pragma unroll
                for (int a = 0; a < 3; a++)
                {
#pragma unroll
                    for (int b = 0; b < 3; b++)
                    {
                        double val = lambda * gi[a] * gj[b] + mu * gi[b] * gj[a];
                        if (a == b)
                            val += mu * dot;
                        kab[3 * a + b] += val;
                    }
                }
            }

#pragma unroll
            for (int k = 0; k < 9; k++)
            {
                kmat_coo_val[9 * (base + ij) + k] = kab[k] * detJ_24;
                kemat_coo_val[9 * (base + ij) + k] = kab[k] * detJ_24;
            }

            coo_row[base + ij] = nidx[i];
            coo_col[base + ij] = nidx[j];

            if (i == j)
            {
                // HRZ集中質量: 対角ノードのみ、スケール係数(35/18)*c_mmat[i,i]を使用
                const double mii = c_mmat[ij]; // c_mmat[i*10+i]
#pragma unroll
                for (int a = 0; a < 3; a++)
                    kemat_coo_val[9 * (base + ij) + 3 * a + a] += mass_coeff * mii;
                mmat_coo_val[base + ij] = rho * (35.0 / 18.0) * mii * detJ_m;
            }
            else
            {
                mmat_coo_val[base + ij] = 0.0;
            }
        }
    }
}

// FiSTR mass_C3 + get_lumped_mass 相当: 4点ガウス積分でconsistent massを評価してHRZ集中化。
// 剛性行列はconstruct_mat_lumpedと同一。質量行列のみ異なる。
__global__ void construct_mat_second(
    const double *__restrict__ node_coords,
    const double *__restrict__ ref_node_coords,
    const double *__restrict__ delta_u_0,
    const double *__restrict__ delta_u_1,
    const double *__restrict__ delta_u_2,
    const double ratio,
    const int *__restrict__ ele_nodes,
    const int num_elements,
    const double lambda,
    const double mu,
    const double rho,
    const double dt,
    double *__restrict__ kmat_coo_val,
    double *__restrict__ mmat_coo_val,
    double *__restrict__ kemat_coo_val,
    int *__restrict__ coo_row,
    int *__restrict__ coo_col)
{
    const int elem = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem >= num_elements)
        return;

    int nidx[10];
#pragma unroll
    for (int i = 0; i < 10; i++)
        nidx[i] = ele_nodes[elem * 10 + i];

    double invJ[9], detJ, detJ_m;
    {
        double J[9], x0[3];
        x0[0] = node_coords[nidx[0] * 3 + 0] + ratio * delta_u_0[nidx[0]];
        x0[1] = node_coords[nidx[0] * 3 + 1] + ratio * delta_u_1[nidx[0]];
        x0[2] = node_coords[nidx[0] * 3 + 2] + ratio * delta_u_2[nidx[0]];
        for (int i = 0; i < 3; i++)
        {
            J[3 * i + 0] = node_coords[nidx[i + 1] * 3 + 0] + ratio * delta_u_0[nidx[i + 1]] - x0[0];
            J[3 * i + 1] = node_coords[nidx[i + 1] * 3 + 1] + ratio * delta_u_1[nidx[i + 1]] - x0[1];
            J[3 * i + 2] = node_coords[nidx[i + 1] * 3 + 2] + ratio * delta_u_2[nidx[i + 1]] - x0[2];
        }
        detJ = J[0] * (J[4] * J[8] - J[5] * J[7]) +
               J[1] * (J[5] * J[6] - J[3] * J[8]) +
               J[2] * (J[3] * J[7] - J[4] * J[6]);
        invJ[0] = (J[4] * J[8] - J[5] * J[7]) / detJ;
        invJ[1] = (J[2] * J[7] - J[1] * J[8]) / detJ;
        invJ[2] = (J[1] * J[5] - J[2] * J[4]) / detJ;
        invJ[3] = (J[5] * J[6] - J[3] * J[8]) / detJ;
        invJ[4] = (J[0] * J[8] - J[2] * J[6]) / detJ;
        invJ[5] = (J[2] * J[3] - J[0] * J[5]) / detJ;
        invJ[6] = (J[3] * J[7] - J[4] * J[6]) / detJ;
        invJ[7] = (J[1] * J[6] - J[0] * J[7]) / detJ;
        invJ[8] = (J[0] * J[4] - J[1] * J[3]) / detJ;

        for (int d = 0; d < 3; d++)
            x0[d] = ref_node_coords[nidx[0] * 3 + d];
        for (int i = 0; i < 3; i++)
            for (int d = 0; d < 3; d++)
                J[3 * i + d] = ref_node_coords[nidx[i + 1] * 3 + d] - x0[d];
        detJ_m = J[0] * (J[4] * J[8] - J[5] * J[7]) +
                 J[1] * (J[5] * J[6] - J[3] * J[8]) +
                 J[2] * (J[3] * J[7] - J[4] * J[6]);
    }
    const double detJ_24 = detJ / 24.0;

    double ldN0[30], ldN1[30], ldN2[30], ldN3[30];
    for (int d = 0; d < 3; d++)
    {
        const double c0 = invJ[3 * d], c1 = invJ[3 * d + 1], c2 = invJ[3 * d + 2];
#pragma unroll
        for (int n = 0; n < 10; n++)
        {
            ldN0[d * 10 + n] = c0 * c_dN0[n] + c1 * c_dN0[10 + n] + c2 * c_dN0[20 + n];
            ldN1[d * 10 + n] = c0 * c_dN1[n] + c1 * c_dN1[10 + n] + c2 * c_dN1[20 + n];
            ldN2[d * 10 + n] = c0 * c_dN2[n] + c1 * c_dN2[10 + n] + c2 * c_dN2[20 + n];
            ldN3[d * 10 + n] = c0 * c_dN3[n] + c1 * c_dN3[10 + n] + c2 * c_dN3[20 + n];
        }
    }

    // FiSTR get_lumped_mass: M_ii = (1/24)*detJ_m * sum_gp N_gp[i]^2
    // total_mass = detJ_m/6 (パーティション・オブ・ユニティにより厳密)
    // scale = total_mass / sum_i(M_ii)
    double diag_Mii[10] = {0.0};
    double diag_sum = 0.0;
#pragma unroll
    for (int i = 0; i < 10; i++)
    {
        const double ni_sq = c_N0[i] * c_N0[i] + c_N1[i] * c_N1[i] +
                             c_N2[i] * c_N2[i] + c_N3[i] * c_N3[i];
        diag_Mii[i] = (1.0 / 24.0) * detJ_m * ni_sq;
        diag_sum += diag_Mii[i];
    }
    const double total_mass = detJ_m / 6.0;
    const double scale = total_mass / diag_sum;
    const double mass_coeff = 4.0 * rho / (dt * dt) * scale;

    const int base = elem * 100;

    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            const int ij = i * 10 + j;
            double kab[9] = {0.0};

#pragma unroll
            for (int gp = 0; gp < 4; gp++)
            {
                const double *ldn = (gp == 0) ? ldN0 : (gp == 1) ? ldN1
                                                   : (gp == 2)   ? ldN2
                                                                 : ldN3;
                const double gi[3] = {ldn[i], ldn[10 + i], ldn[20 + i]};
                const double gj[3] = {ldn[j], ldn[10 + j], ldn[20 + j]};
                const double dot = gi[0] * gj[0] + gi[1] * gj[1] + gi[2] * gj[2];

#pragma unroll
                for (int a = 0; a < 3; a++)
                {
#pragma unroll
                    for (int b = 0; b < 3; b++)
                    {
                        double val = lambda * gi[a] * gj[b] + mu * gi[b] * gj[a];
                        if (a == b)
                            val += mu * dot;
                        kab[3 * a + b] += val;
                    }
                }
            }

#pragma unroll
            for (int k = 0; k < 9; k++)
            {
                kmat_coo_val[9 * (base + ij) + k] = kab[k] * detJ_24;
                kemat_coo_val[9 * (base + ij) + k] = kab[k] * detJ_24;
            }

            coo_row[base + ij] = nidx[i];
            coo_col[base + ij] = nidx[j];

            if (i == j)
            {
                const double lumped_i = rho * diag_Mii[i] * scale;
#pragma unroll
                for (int a = 0; a < 3; a++)
                    kemat_coo_val[9 * (base + ij) + 3 * a + a] += mass_coeff * diag_Mii[i];
                mmat_coo_val[base + ij] = lumped_i;
            }
            else
            {
                mmat_coo_val[base + ij] = 0.0;
            }
        }
    }
}

// ---- kmat(9要素) + mmat(1要素) をまとめた構造体 ----
struct BlockVal
{
    double k[9];
    double m;
    double ke[9];
};

__host__ __device__ BlockVal operator+(const BlockVal &a, const BlockVal &b)
{
    BlockVal c;
    for (int i = 0; i < 9; i++)
    {
        c.k[i] = a.k[i] + b.k[i];
        c.ke[i] = a.ke[i] + b.ke[i];
    }
    c.m = a.m + b.m;
    return c;
}

__host__ __device__ bool operator==(const BlockVal &, const BlockVal &)
{
    return false; // reduce_by_key用、実際には呼ばれない
}

// ---- COO → ソート済み構造体配列に変換するカーネル ----
__global__ void pack_coo(
    const int *__restrict__ coo_row,
    const int *__restrict__ coo_col,
    const double *__restrict__ kmat_val,
    const double *__restrict__ mmat_val,
    const double *__restrict__ kemat_val,
    int64_t *__restrict__ keys,
    BlockVal *__restrict__ vals,
    int nnz, int num_nodes)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nnz)
        return;
    keys[k] = (int64_t)coo_row[k] * num_nodes + coo_col[k];
    for (int i = 0; i < 9; i++)
    {
        vals[k].k[i] = kmat_val[9 * k + i];
        vals[k].ke[i] = kemat_val[9 * k + i];
    }
    vals[k].m = mmat_val[k];
}

// ---- 結果を元の配列に展開するカーネル ----
__global__ void unpack_bcrs(
    const int64_t *__restrict__ keys,
    const BlockVal *__restrict__ vals,
    int *__restrict__ out_row,
    int *__restrict__ out_col,
    double *__restrict__ out_kmat,
    double *__restrict__ out_mmat,
    double *__restrict__ out_kemat,
    int nnz_bcrs, int num_nodes)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nnz_bcrs)
        return;
    out_row[k] = (int)(keys[k] / num_nodes);
    out_col[k] = (int)(keys[k] % num_nodes);
    for (int i = 0; i < 9; i++)
    {
        out_kmat[9 * k + i] = vals[k].k[i];
        out_kemat[9 * k + i] = vals[k].ke[i];
    }
    out_mmat[k] = vals[k].m;
}

__global__ void unpack_bcrs_kernel(
    const int *__restrict__ coo_col,
    const double *__restrict__ kmat_coo_val,
    const double *__restrict__ mmat_coo_val,
    const double *__restrict__ kemat_coo_val,
    int *__restrict__ bcrs_col_ind,
    double *__restrict__ kval_00, double *__restrict__ kval_01, double *__restrict__ kval_02,
    double *__restrict__ kval_10, double *__restrict__ kval_11, double *__restrict__ kval_12,
    double *__restrict__ kval_20, double *__restrict__ kval_21, double *__restrict__ kval_22,
    double *__restrict__ bcrs_mval,
    double *__restrict__ keval_00, double *__restrict__ keval_01, double *__restrict__ keval_02,
    double *__restrict__ keval_10, double *__restrict__ keval_11, double *__restrict__ keval_12,
    double *__restrict__ keval_20, double *__restrict__ keval_21, double *__restrict__ keval_22,
    int nnz_bcrs)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nnz_bcrs)
        return;

    bcrs_col_ind[k] = coo_col[k];
    const double *src = &kmat_coo_val[9 * k];
    kval_00[k] = src[0];
    kval_01[k] = src[1];
    kval_02[k] = src[2];
    kval_10[k] = src[3];
    kval_11[k] = src[4];
    kval_12[k] = src[5];
    kval_20[k] = src[6];
    kval_21[k] = src[7];
    kval_22[k] = src[8];
    bcrs_mval[k] = mmat_coo_val[k];
    const double *ksrc = &kemat_coo_val[9 * k];
    keval_00[k] = ksrc[0];
    keval_01[k] = ksrc[1];
    keval_02[k] = ksrc[2];
    keval_10[k] = ksrc[3];
    keval_11[k] = ksrc[4];
    keval_12[k] = ksrc[5];
    keval_20[k] = ksrc[6];
    keval_21[k] = ksrc[7];
    keval_22[k] = ksrc[8];
}

__global__ void count_rows_kernel(
    const int *__restrict__ coo_row,
    int *__restrict__ row_ptr,
    int nnz)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nnz)
        return;
    atomicAdd(&row_ptr[coo_row[k] + 1], 1);
}

__global__ void extract_bc_correction(
    int num_nodes,
    const int *__restrict__ row_ptr,
    const int *__restrict__ col_ind,
    const double *__restrict__ kval_00, const double *__restrict__ kval_01, const double *__restrict__ kval_02,
    const double *__restrict__ kval_10, const double *__restrict__ kval_11, const double *__restrict__ kval_12,
    const double *__restrict__ kval_20, const double *__restrict__ kval_21, const double *__restrict__ kval_22,
    const int *__restrict__ bc_flag,
    double *__restrict__ bc_corr_00, double *__restrict__ bc_corr_01, double *__restrict__ bc_corr_02,
    double *__restrict__ bc_corr_10, double *__restrict__ bc_corr_11, double *__restrict__ bc_corr_12,
    double *__restrict__ bc_corr_20, double *__restrict__ bc_corr_21, double *__restrict__ bc_corr_22)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes)
        return;

    if (bc_flag[i])
    {
        bc_corr_00[i] = bc_corr_01[i] = bc_corr_02[i] = 0.0;
        bc_corr_10[i] = bc_corr_11[i] = bc_corr_12[i] = 0.0;
        bc_corr_20[i] = bc_corr_21[i] = bc_corr_22[i] = 0.0;
        return;
    }

    double c00 = 0, c01 = 0, c02 = 0;
    double c10 = 0, c11 = 0, c12 = 0;
    double c20 = 0, c21 = 0, c22 = 0;

    for (int p = row_ptr[i]; p < row_ptr[i + 1]; p++)
    {
        if (!bc_flag[col_ind[p]])
            continue;
        c00 += kval_00[p];
        c01 += kval_01[p];
        c02 += kval_02[p];
        c10 += kval_10[p];
        c11 += kval_11[p];
        c12 += kval_12[p];
        c20 += kval_20[p];
        c21 += kval_21[p];
        c22 += kval_22[p];
    }

    bc_corr_00[i] = c00;
    bc_corr_01[i] = c01;
    bc_corr_02[i] = c02;
    bc_corr_10[i] = c10;
    bc_corr_11[i] = c11;
    bc_corr_12[i] = c12;
    bc_corr_20[i] = c20;
    bc_corr_21[i] = c21;
    bc_corr_22[i] = c22;
}

__global__ void apply_bc_to_lhs(
    int num_nodes,
    const int *__restrict__ row_ptr,
    const int *__restrict__ col_ind,
    double *__restrict__ kval_00, double *__restrict__ kval_01, double *__restrict__ kval_02,
    double *__restrict__ kval_10, double *__restrict__ kval_11, double *__restrict__ kval_12,
    double *__restrict__ kval_20, double *__restrict__ kval_21, double *__restrict__ kval_22,
    const int *__restrict__ bc_flag)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes)
        return;

    const int is_bc = bc_flag[i];

    for (int p = row_ptr[i]; p < row_ptr[i + 1]; p++)
    {
        if (is_bc || bc_flag[col_ind[p]])
        {
            kval_00[p] = kval_01[p] = kval_02[p] = 0.0;
            kval_10[p] = kval_11[p] = kval_12[p] = 0.0;
            kval_20[p] = kval_21[p] = kval_22[p] = 0.0;
        }
        if (is_bc && col_ind[p] == i)
        {
            kval_00[p] = kval_11[p] = kval_22[p] = 1.0;
        }
    }
}

__global__ void build_block_jacobi(
    int num_nodes,
    const int *__restrict__ row_ptr,
    const int *__restrict__ col_ind,
    const double *__restrict__ kval_00, const double *__restrict__ kval_01, const double *__restrict__ kval_02,
    const double *__restrict__ kval_10, const double *__restrict__ kval_11, const double *__restrict__ kval_12,
    const double *__restrict__ kval_20, const double *__restrict__ kval_21, const double *__restrict__ kval_22,
    double *__restrict__ inv_diag_00, double *__restrict__ inv_diag_01, double *__restrict__ inv_diag_02,
    double *__restrict__ inv_diag_10, double *__restrict__ inv_diag_11, double *__restrict__ inv_diag_12,
    double *__restrict__ inv_diag_20, double *__restrict__ inv_diag_21, double *__restrict__ inv_diag_22)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes)
        return;

    double d[9] = {};

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

    double detJ = d[0] * (d[4] * d[8] - d[5] * d[7]) +
                  d[1] * (d[5] * d[6] - d[3] * d[8]) +
                  d[2] * (d[3] * d[7] - d[4] * d[6]);

    inv_diag_00[i] = (d[4] * d[8] - d[5] * d[7]) / detJ;
    inv_diag_01[i] = (d[2] * d[7] - d[1] * d[8]) / detJ;
    inv_diag_02[i] = (d[1] * d[5] - d[2] * d[4]) / detJ;
    inv_diag_10[i] = (d[5] * d[6] - d[3] * d[8]) / detJ;
    inv_diag_11[i] = (d[0] * d[8] - d[2] * d[6]) / detJ;
    inv_diag_12[i] = (d[2] * d[3] - d[0] * d[5]) / detJ;
    inv_diag_20[i] = (d[3] * d[7] - d[4] * d[6]) / detJ;
    inv_diag_21[i] = (d[1] * d[6] - d[0] * d[7]) / detJ;
    inv_diag_22[i] = (d[0] * d[4] - d[1] * d[3]) / detJ;
}

// --- Newmark-β RHS の一時ベクトル構築 ---
__global__ void kernel_build_newmark_tmp(
    int num_nodes, double dt_inv2, double dt_inv,
    const double *v0, const double *v1, const double *v2,
    const double *a0, const double *a1, const double *a2,
    double *t0, double *t1, double *t2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes)
        return;
    t0[i] = v0[i] * 4.0 * dt_inv + a0[i];
    t1[i] = v1[i] * 4.0 * dt_inv + a1[i];
    t2[i] = v2[i] * 4.0 * dt_inv + a2[i];
}

// --- BCRS 質量行列 SpMV: y = M * x ---
__global__ void kernel_bcrs_spmv_m(
    int num_nodes,
    const int *row_ptr, const int *col_ind, const double *val,
    const double *x0, const double *x1, const double *x2,
    double *y0, double *y1, double *y2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes)
        return;
    double s0 = 0.0, s1 = 0.0, s2 = 0.0;
    for (int p = row_ptr[i]; p < row_ptr[i + 1]; p++)
    {
        int j = col_ind[p];
        double v = val[p];
        s0 += v * x0[j];
        s1 += v * x1[j];
        s2 += v * x2[j];
    }
    y0[i] = s0;
    y1[i] = s1;
    y2[i] = s2;
}

// --- 境界条件を右辺に適用（自由節点の補正） ---
__global__ void kernel_apply_bc_to_rhs(
    int num_nodes,
    const double *bc_val, const int *bc_flag,
    const double *bc_corr_00, const double *bc_corr_01, const double *bc_corr_02,
    const double *bc_corr_10, const double *bc_corr_11, const double *bc_corr_12,
    const double *bc_corr_20, const double *bc_corr_21, const double *bc_corr_22,
    double *rhs_0, double *rhs_1, double *rhs_2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes)
        return;
    double bv0 = bc_val[0], bv1 = bc_val[1], bv2 = bc_val[2];
    rhs_0[i] -= bc_corr_00[i] * bv0 + bc_corr_01[i] * bv1 + bc_corr_02[i] * bv2;
    rhs_1[i] -= bc_corr_10[i] * bv0 + bc_corr_11[i] * bv1 + bc_corr_12[i] * bv2;
    rhs_2[i] -= bc_corr_20[i] * bv0 + bc_corr_21[i] * bv1 + bc_corr_22[i] * bv2;
    if (bc_flag[i])
    {
        rhs_0[i] = bc_val[0];
        rhs_1[i] = bc_val[1];
        rhs_2[i] = bc_val[2];
    }
}

// --- ドット積のリダクション (3成分同時) ---
// result[0] にドット積を加算する
// ブロックサイズは最大1024（Warp数最大32）を想定
__global__ void kernel_dot3_reduce(
    int start, int end,
    const double *a0, const double *a1, const double *a2,
    const double *b0, const double *b1, const double *b2,
    double *__restrict__ result)
{
    // 各Warpの合計値を保存するための共有メモリ（最大32個で十分）
    __shared__ double warp_sums[32];

    int tid = threadIdx.x;
    int i = start + blockIdx.x * blockDim.x + tid;
    int lane_id = tid % 32;
    int warp_id = tid / 32;

    double val = 0.0;
    if (i < end)
    {
        val = a0[i] * b0[i] + a1[i] * b1[i] + a2[i] * b2[i];
    }

    // --- 1段階目: Warp内リダクション ---
    // 各Warp（32スレッド）の中で値を集約し、レーン0に合計を集める
    for (int offset = 16; offset > 0; offset /= 2)
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // --- 2段階目: 共有メモリへの書き込み ---
    // 各Warpの代表スレッド（レーン0）だけが、共有メモリに結果を書き込む
    if (lane_id == 0)
    {
        warp_sums[warp_id] = val;
    }

    // 全Warpが書き込み終わるのを1回だけ待つ
    __syncthreads();

    // --- 3段階目: ブロック全体の集計（最初のWarpだけが働く） ---
    if (warp_id == 0)
    {
        // ブロック内のWarp数を算出
        int num_warps = blockDim.x / 32;

        // 共有メモリから各Warpの合計値を読み出す（範囲外は0にする）
        val = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0;

        // 再度Warpリダクションを行い、ブロック全体の合計を算出
        for (int offset = 16; offset > 0; offset /= 2)
        {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }

        // ブロックの代表スレッド（スレッド0）が、グローバルメモリに1回だけ加算
        if (tid == 0)
        {
            atomicAdd(result, val);
        }
    }
}

// --- PCG初期化カーネル: r=b-Ax, z=C^{-1}r, p=z, rz/b_norm/r_norm計算 ---
__global__ void kernel_pcg_init(
    int num_owned,
    const int *row_ptr, const int *col_ind,
    const double *kval_00, const double *kval_01, const double *kval_02,
    const double *kval_10, const double *kval_11, const double *kval_12,
    const double *kval_20, const double *kval_21, const double *kval_22,
    const double *inv_d_00, const double *inv_d_01, const double *inv_d_02,
    const double *inv_d_10, const double *inv_d_11, const double *inv_d_12,
    const double *inv_d_20, const double *inv_d_21, const double *inv_d_22,
    const double *b0, const double *b1, const double *b2,
    const double *x0, const double *x1, const double *x2,
    double *Ap0, double *Ap1, double *Ap2,
    double *r0, double *r1, double *r2,
    double *z0, double *z1, double *z2,
    double *p0, double *p1, double *p2,
    double *d_rz, double *d_bnorm, double *d_rnorm)
{
    __shared__ double s_rz[BLOCK_SIZE];
    __shared__ double s_bn[BLOCK_SIZE];
    __shared__ double s_rn[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double local_rz = 0.0, local_bn = 0.0, local_rn = 0.0;

    if (i < num_owned)
    {
        // Ap = K * x
        double y0 = 0.0, y1 = 0.0, y2 = 0.0;
        for (int p = row_ptr[i]; p < row_ptr[i + 1]; p++)
        {
            int j = col_ind[p];
            double xj0 = x0[j], xj1 = x1[j], xj2 = x2[j];
            y0 += kval_00[p] * xj0 + kval_01[p] * xj1 + kval_02[p] * xj2;
            y1 += kval_10[p] * xj0 + kval_11[p] * xj1 + kval_12[p] * xj2;
            y2 += kval_20[p] * xj0 + kval_21[p] * xj1 + kval_22[p] * xj2;
        }
        Ap0[i] = y0;
        Ap1[i] = y1;
        Ap2[i] = y2;

        double bi0 = b0[i], bi1 = b1[i], bi2 = b2[i];
        double ri0 = bi0 - y0, ri1 = bi1 - y1, ri2 = bi2 - y2;
        r0[i] = ri0;
        r1[i] = ri1;
        r2[i] = ri2;

        // z = C^{-1} r
        double zi0 = inv_d_00[i] * ri0 + inv_d_01[i] * ri1 + inv_d_02[i] * ri2;
        double zi1 = inv_d_10[i] * ri0 + inv_d_11[i] * ri1 + inv_d_12[i] * ri2;
        double zi2 = inv_d_20[i] * ri0 + inv_d_21[i] * ri1 + inv_d_22[i] * ri2;
        z0[i] = zi0;
        z1[i] = zi1;
        z2[i] = zi2;
        p0[i] = zi0;
        p1[i] = zi1;
        p2[i] = zi2;

        local_rz = ri0 * zi0 + ri1 * zi1 + ri2 * zi2;
        local_bn = bi0 * bi0 + bi1 * bi1 + bi2 * bi2;
        local_rn = ri0 * ri0 + ri1 * ri1 + ri2 * ri2;
    }

    s_rz[tid] = local_rz;
    s_bn[tid] = local_bn;
    s_rn[tid] = local_rn;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            s_rz[tid] += s_rz[tid + s];
            s_bn[tid] += s_bn[tid + s];
            s_rn[tid] += s_rn[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        atomicAdd(d_rz, s_rz[0]);
        atomicAdd(d_bnorm, s_bn[0]);
        atomicAdd(d_rnorm, s_rn[0]);
    }
}

// --- x += alpha * p ---
__global__ void kernel_axpy(int n, double alpha,
                            const double *p0, const double *p1, const double *p2,
                            double *x0, double *x1, double *x2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    x0[i] += alpha * p0[i];
    x1[i] += alpha * p1[i];
    x2[i] += alpha * p2[i];
}

// --- r -= alpha * Ap, r_normを計算 ---
__global__ void kernel_update_r(
    int num_owned, double alpha,
    const double *Ap0, const double *Ap1, const double *Ap2,
    double *r0, double *r1, double *r2,
    double *d_rnorm)
{
    __shared__ double sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double local_rn = 0.0;
    if (i < num_owned)
    {
        double ri0 = r0[i] - alpha * Ap0[i];
        double ri1 = r1[i] - alpha * Ap1[i];
        double ri2 = r2[i] - alpha * Ap2[i];
        r0[i] = ri0;
        r1[i] = ri1;
        r2[i] = ri2;
        local_rn = ri0 * ri0 + ri1 * ri1 + ri2 * ri2;
    }
    sdata[tid] = local_rn;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(d_rnorm, sdata[0]);
}

// --- z = C^{-1}r, rz_new計算 ---
__global__ void kernel_precond_and_rz(
    int num_owned,
    const double *inv_d_00, const double *inv_d_01, const double *inv_d_02,
    const double *inv_d_10, const double *inv_d_11, const double *inv_d_12,
    const double *inv_d_20, const double *inv_d_21, const double *inv_d_22,
    const double *r0, const double *r1, const double *r2,
    double *z0, double *z1, double *z2,
    double *d_rz_new)
{
    __shared__ double sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double local_rz = 0.0;
    if (i < num_owned)
    {
        double ri0 = r0[i], ri1 = r1[i], ri2 = r2[i];
        double zi0 = inv_d_00[i] * ri0 + inv_d_01[i] * ri1 + inv_d_02[i] * ri2;
        double zi1 = inv_d_10[i] * ri0 + inv_d_11[i] * ri1 + inv_d_12[i] * ri2;
        double zi2 = inv_d_20[i] * ri0 + inv_d_21[i] * ri1 + inv_d_22[i] * ri2;
        z0[i] = zi0;
        z1[i] = zi1;
        z2[i] = zi2;
        local_rz = ri0 * zi0 + ri1 * zi1 + ri2 * zi2;
    }
    sdata[tid] = local_rz;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(d_rz_new, sdata[0]);
}

// --- p = z + beta * p ---
__global__ void kernel_update_p(
    int num_owned, double beta,
    const double *z0, const double *z1, const double *z2,
    double *p0, double *p1, double *p2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_owned)
        return;
    p0[i] = z0[i] + beta * p0[i];
    p1[i] = z1[i] + beta * p1[i];
    p2[i] = z2[i] + beta * p2[i];
}

// --- 送信バッファパック ---
__global__ void kernel_pack_send_buffer(
    int count, int offset,
    const int *send_nodes,
    const double *p0, const double *p1, const double *p2,
    double *sb0, double *sb1, double *sb2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    int node = send_nodes[offset + i];
    sb0[offset + i] = p0[node];
    sb1[offset + i] = p1[node];
    sb2[offset + i] = p2[node];
}

// --- SpMV (1つのWarpが1行を担当するので、行数×32個のthreadが必要) ---
__global__ void kernel_spmv(
    int start, int end,
    const int *__restrict__ row_ptr, const int *__restrict__ col_ind,
    const double *__restrict__ kval_00, const double *__restrict__ kval_01, const double *__restrict__ kval_02,
    const double *__restrict__ kval_10, const double *__restrict__ kval_11, const double *__restrict__ kval_12,
    const double *__restrict__ kval_20, const double *__restrict__ kval_21, const double *__restrict__ kval_22,
    const double *__restrict__ p0, const double *__restrict__ p1, const double *__restrict__ p2,
    double *__restrict__ Ap0, double *__restrict__ Ap1, double *__restrict__ Ap2)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // 1つのWarpが1つの行を担当する
    int i = start + warp_id;

    double s0 = 0.0, s1 = 0.0, s2 = 0.0;

    if (i < end)
    {
        int row_start = row_ptr[i];
        int row_end = row_ptr[i + 1];

        // col_ind,kvalへのコアレスアクセスのため、連続するThreadが連続するpを処理するようにループする
        for (int p = row_start + lane_id; p < row_end; p += 32)
        {
            int j = col_ind[p];
            double px = p0[j], py = p1[j], pz = p2[j];

            s0 += kval_00[p] * px + kval_01[p] * py + kval_02[p] * pz;
            s1 += kval_10[p] * px + kval_11[p] * py + kval_12[p] * pz;
            s2 += kval_20[p] * px + kval_21[p] * py + kval_22[p] * pz;
        }
        // 32スレッド分の s0, s1, s2 をレーン0に集約する
        for (int offset = 16; offset > 0; offset /= 2)
        {
            s0 += __shfl_down_sync(0xffffffff, s0, offset);
            s1 += __shfl_down_sync(0xffffffff, s1, offset);
            s2 += __shfl_down_sync(0xffffffff, s2, offset);
        }
        // 代表スレッド（レーン0）のみが最終結果をグローバルメモリに書き込む
        if (lane_id == 0)
        {
            Ap0[i] = s0;
            Ap1[i] = s1;
            Ap2[i] = s2;
        }
    }
}

__global__ void update_coords(
    const double *__restrict__ ref_node_coords,
    const double *__restrict__ u_tmp_0,
    const double *__restrict__ u_tmp_1,
    const double *__restrict__ u_tmp_2,
    double *__restrict__ node_coords,
    int num_nodes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes)
        return;

    node_coords[i * 3 + 0] = ref_node_coords[i * 3 + 0] + u_tmp_0[i];
    node_coords[i * 3 + 1] = ref_node_coords[i * 3 + 1] + u_tmp_1[i];
    node_coords[i * 3 + 2] = ref_node_coords[i * 3 + 2] + u_tmp_2[i];
}

// --- Newmark-β 更新カーネル ---
__global__ void kernel_newmark_update(
    int num_nodes, double dt, double dt_inv, double dt_inv2,
    const int *bc_flag, const double *bc_val_u,
    double *u_tmp_0, double *u_tmp_1, double *u_tmp_2,
    double *u_prv_0, double *u_prv_1, double *u_prv_2,
    double *v_tmp_0, double *v_tmp_1, double *v_tmp_2,
    double *a_tmp_0, double *a_tmp_1, double *a_tmp_2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes)
        return;

    double u_new_0 = u_tmp_0[i], u_new_1 = u_tmp_1[i], u_new_2 = u_tmp_2[i];
    double u_old_0 = u_prv_0[i], u_old_1 = u_prv_1[i], u_old_2 = u_prv_2[i];
    double v_old_0 = v_tmp_0[i], v_old_1 = v_tmp_1[i], v_old_2 = v_tmp_2[i];
    double a_old_0 = a_tmp_0[i], a_old_1 = a_tmp_1[i], a_old_2 = a_tmp_2[i];

    if (bc_flag[i])
    {
        u_new_0 = bc_val_u[0];
        u_new_1 = bc_val_u[1];
        u_new_2 = bc_val_u[2];
    }

    double a_new_0 = (u_new_0 - u_old_0) * 4.0 * dt_inv2 - v_old_0 * 4.0 * dt_inv - a_old_0;
    double a_new_1 = (u_new_1 - u_old_1) * 4.0 * dt_inv2 - v_old_1 * 4.0 * dt_inv - a_old_1;
    double a_new_2 = (u_new_2 - u_old_2) * 4.0 * dt_inv2 - v_old_2 * 4.0 * dt_inv - a_old_2;
    double v_new_0 = v_old_0 + (a_new_0 + a_old_0) * dt / 2.0;
    double v_new_1 = v_old_1 + (a_new_1 + a_old_1) * dt / 2.0;
    double v_new_2 = v_old_2 + (a_new_2 + a_old_2) * dt / 2.0;

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

// ============================================================
// CPU関数
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
    double a = (5.0 - std::sqrt(5.0)) / 20.0;
    double b = (5.0 + 3.0 * std::sqrt(5.0)) / 20.0;
    double gauss_points[4][3] = {
        {a, a, a}, {b, a, a}, {a, b, a}, {a, a, b}};
    calculate_dN(dN0, gauss_points[0][0], gauss_points[0][1], gauss_points[0][2]);
    calculate_dN(dN1, gauss_points[1][0], gauss_points[1][1], gauss_points[1][2]);
    calculate_dN(dN2, gauss_points[2][0], gauss_points[2][1], gauss_points[2][2]);
    calculate_dN(dN3, gauss_points[3][0], gauss_points[3][1], gauss_points[3][2]);
}

void calculate_N(double N[10], double r, double s, double t)
{
    double a = 1.0 - r - s - t;
    N[0] = (2.0 * a - 1.0) * a;
    N[1] = (2.0 * r - 1.0) * r;
    N[2] = (2.0 * s - 1.0) * s;
    N[3] = (2.0 * t - 1.0) * t;
    N[4] = 4.0 * r * a;
    N[5] = 4.0 * r * s;
    N[6] = 4.0 * s * a;
    N[7] = 4.0 * t * a;
    N[8] = 4.0 * r * t;
    N[9] = 4.0 * s * t;
}

void gauss_integrate_N(double N0[10], double N1[10], double N2[10], double N3[10])
{
    double a = (5.0 - std::sqrt(5.0)) / 20.0;
    double b = (5.0 + 3.0 * std::sqrt(5.0)) / 20.0;
    double gauss_points[4][3] = {
        {a, a, a}, {b, a, a}, {a, b, a}, {a, a, b}};
    calculate_N(N0, gauss_points[0][0], gauss_points[0][1], gauss_points[0][2]);
    calculate_N(N1, gauss_points[1][0], gauss_points[1][1], gauss_points[1][2]);
    calculate_N(N2, gauss_points[2][0], gauss_points[2][1], gauss_points[2][2]);
    calculate_N(N3, gauss_points[3][0], gauss_points[3][1], gauss_points[3][2]);
}

int sort_and_merge_bcoo(
    int nnz_coo, int num_nodes,
    int *coo_row, int *coo_col,
    double *kmat_coo_val, double *mmat_coo_val, double *kemat_coo_val)
{
    thrust::device_vector<int64_t> keys(nnz_coo);
    thrust::device_vector<BlockVal> vals(nnz_coo);

    int block = 256;
    int grid = (nnz_coo + block - 1) / block;
    pack_coo<<<grid, block>>>(
        coo_row, coo_col, kmat_coo_val, mmat_coo_val, kemat_coo_val,
        thrust::raw_pointer_cast(keys.data()),
        thrust::raw_pointer_cast(vals.data()),
        nnz_coo, num_nodes);
    CUDA_CHECK(cudaGetLastError());

    // ★変更点★ 軽い (int64, int) ペアだけでソートする
    thrust::device_vector<int> perm(nnz_coo);
    thrust::sequence(perm.begin(), perm.end());
    thrust::sort_by_key(keys.begin(), keys.end(), perm.begin());

    // perm に従って BlockVal を並び替え（ここは radix_sort を介さないので安全）
    thrust::device_vector<BlockVal> vals_sorted(nnz_coo);
    thrust::gather(perm.begin(), perm.end(),
                   vals.begin(), vals_sorted.begin());
    vals.swap(vals_sorted);

    // 以降は今までと同じ。reduce_by_key は radix_sort を使わないので BlockVal で OK
    thrust::device_vector<int64_t> keys_out(nnz_coo);
    thrust::device_vector<BlockVal> vals_out(nnz_coo);

    auto end = thrust::reduce_by_key(
        keys.begin(), keys.end(),
        vals.begin(),
        keys_out.begin(),
        vals_out.begin());

    int nnz_bcrs = end.first - keys_out.begin();

    grid = (nnz_bcrs + block - 1) / block;
    unpack_bcrs<<<grid, block>>>(
        thrust::raw_pointer_cast(keys_out.data()),
        thrust::raw_pointer_cast(vals_out.data()),
        coo_row, coo_col, kmat_coo_val, mmat_coo_val, kemat_coo_val,
        nnz_bcrs, num_nodes);

    return nnz_bcrs;
}

double inverse_3_3_mat(double mat[9], double inv_mat[9])
{
    double det_mat = mat[0] * (mat[4] * mat[8] - mat[5] * mat[7]) +
                     mat[1] * (mat[5] * mat[6] - mat[3] * mat[8]) +
                     mat[2] * (mat[3] * mat[7] - mat[4] * mat[6]);
    double inv_det = 1.0 / det_mat;
    inv_mat[0] = (mat[4] * mat[8] - mat[5] * mat[7]) * inv_det;
    inv_mat[1] = (mat[2] * mat[7] - mat[1] * mat[8]) * inv_det;
    inv_mat[2] = (mat[1] * mat[5] - mat[2] * mat[4]) * inv_det;
    inv_mat[3] = (mat[5] * mat[6] - mat[3] * mat[8]) * inv_det;
    inv_mat[4] = (mat[0] * mat[8] - mat[2] * mat[6]) * inv_det;
    inv_mat[5] = (mat[2] * mat[3] - mat[0] * mat[5]) * inv_det;
    inv_mat[6] = (mat[3] * mat[7] - mat[4] * mat[6]) * inv_det;
    inv_mat[7] = (mat[1] * mat[6] - mat[0] * mat[7]) * inv_det;
    inv_mat[8] = (mat[0] * mat[4] - mat[1] * mat[3]) * inv_det;
    return det_mat;
}

void build_bcrs(
    int *coo_row, int *coo_col,
    double *kmat_coo_val, double *mmat_coo_val, double *kemat_coo_val,
    int nnz_bcrs, int num_nodes,
    int *bcrs_row_ptr, int *bcrs_col_ind,
    double *kval_00, double *kval_01, double *kval_02,
    double *kval_10, double *kval_11, double *kval_12,
    double *kval_20, double *kval_21, double *kval_22,
    double *bcrs_mval,
    double *keval_00, double *keval_01, double *keval_02,
    double *keval_10, double *keval_11, double *keval_12,
    double *keval_20, double *keval_21, double *keval_22)
{
    int block = 256;
    int grid = (nnz_bcrs + block - 1) / block;

    unpack_bcrs_kernel<<<grid, block>>>(
        coo_col, kmat_coo_val, mmat_coo_val, kemat_coo_val,
        bcrs_col_ind,
        kval_00, kval_01, kval_02,
        kval_10, kval_11, kval_12,
        kval_20, kval_21, kval_22,
        bcrs_mval,
        keval_00, keval_01, keval_02,
        keval_10, keval_11, keval_12,
        keval_20, keval_21, keval_22,
        nnz_bcrs);

    cudaMemset(bcrs_row_ptr, 0, sizeof(int) * (num_nodes + 1));
    count_rows_kernel<<<grid, block>>>(coo_row, bcrs_row_ptr, nnz_bcrs);

    thrust::inclusive_scan(
        thrust::device_pointer_cast(bcrs_row_ptr + 1),
        thrust::device_pointer_cast(bcrs_row_ptr + 1 + num_nodes),
        thrust::device_pointer_cast(bcrs_row_ptr + 1));
}

void write_vtk_displacement(
    const char *filename, double *node_coords, int num_nodes,
    int *ele_nodes, int num_elements, double *displacement_0, double *displacement_1, double *displacement_2, double total_time)
{
    FILE *fp = fopen(filename, "w");
    fprintf(fp, "# vtk DataFile Version 2.0\n");
    fprintf(fp, "FEM displacement result\n");
    fprintf(fp, "ASCII\n");
    fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");
    fprintf(fp, "\nFIELD FieldData 1\n");
    fprintf(fp, "TOTALTIME 1 1 double\n");
    fprintf(fp, "%.10e\n", total_time);
    fprintf(fp, "POINTS %d double\n", num_nodes);
    for (int i = 0; i < num_nodes; i++)
        fprintf(fp, "%.15e %.15e %.15e\n", node_coords[i * 3 + 0], node_coords[i * 3 + 1], node_coords[i * 3 + 2]);
    int cells_size = num_elements * 11;
    fprintf(fp, "CELLS %d %d\n", num_elements, cells_size);
    for (int e = 0; e < num_elements; e++)
    {
        fprintf(fp, "10");
        for (int i = 0; i < 10; i++)
            fprintf(fp, " %d", ele_nodes[e * 10 + i]);
        fprintf(fp, "\n");
    }
    fprintf(fp, "CELL_TYPES %d\n", num_elements);
    for (int e = 0; e < num_elements; e++)
        fprintf(fp, "24\n");
    fprintf(fp, "POINT_DATA %d\n", num_nodes);
    fprintf(fp, "SCALARS NODE_ID int 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int i = 0; i < num_nodes; i++)
        fprintf(fp, "%d\n", i + 1);
    fprintf(fp, "VECTORS DISPLACEMENT double\n");
    for (int i = 0; i < num_nodes; i++)
        fprintf(fp, "%.15e %.15e %.15e\n", displacement_0[i], displacement_1[i], displacement_2[i]);
    fclose(fp);
}

void write_node_disp_csv(
    const char *filename, double *u, int num_steps, int sample_freq, double dt)
{
    FILE *csv_fp = fopen(filename, "w");
    fprintf(csv_fp, "time,disp_x,disp_y,disp_z,disp_mag\n");
    for (int step = 0; step <= num_steps; step += sample_freq)
    {
        double t = step * dt;
        int idx = step / sample_freq;
        double ux = u[idx * 3 + 0], uy = u[idx * 3 + 1], uz = u[idx * 3 + 2];
        double mag = sqrt(ux * ux + uy * uy + uz * uz);
        fprintf(csv_fp, "%.10e,%.10e,%.10e,%.10e,%.10e\n", t, ux, uy, uz, mag);
    }
    fclose(csv_fp);
}

int pcg_solve(
    DeviceData &dd,
    MPI_Request *request,
    int num_neighbors,
    int *neighbor_ranks,
    int *recv_starts,
    int *recv_counts,
    int *send_starts,
    int *send_counts,
    int num_inner,
    int num_owned,
    int num_nodes,
    double tol,
    int max_iter)
{
    int grid_owned = (num_owned + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid_nodes = (num_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // リダクション用のデバイスメモリ（3要素: rz, b_norm, r_norm）
    double h_vals[3] = {0.0, 0.0, 0.0};
    CUDA_CHECK(cudaMemset(dd.d_reduce, 0.0, 3 * sizeof(double)));

    // --- 初期化: r=b-Ax, z=C^{-1}r, p=z ---
    kernel_pcg_init<<<grid_owned, BLOCK_SIZE>>>(
        num_owned,
        dd.row_ptr, dd.col_ind,
        dd.keval_00, dd.keval_01, dd.keval_02,
        dd.keval_10, dd.keval_11, dd.keval_12,
        dd.keval_20, dd.keval_21, dd.keval_22,
        dd.inv_diag_00, dd.inv_diag_01, dd.inv_diag_02,
        dd.inv_diag_10, dd.inv_diag_11, dd.inv_diag_12,
        dd.inv_diag_20, dd.inv_diag_21, dd.inv_diag_22,
        dd.rhs_0, dd.rhs_1, dd.rhs_2,
        dd.delta_u_0, dd.delta_u_1, dd.delta_u_2,
        dd.Ap_0, dd.Ap_1, dd.Ap_2,
        dd.r_0, dd.r_1, dd.r_2,
        dd.z_0, dd.z_1, dd.z_2,
        dd.p_0, dd.p_1, dd.p_2,
        &dd.d_reduce[0], &dd.d_reduce[1], &dd.d_reduce[2]);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_vals, dd.d_reduce, 3 * sizeof(double), cudaMemcpyDeviceToHost));
    double rz = h_vals[0], b_norm = h_vals[1], r_norm = h_vals[2];

    MPI_Iallreduce(MPI_IN_PLACE, &rz, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request[0]);
    MPI_Iallreduce(MPI_IN_PLACE, &b_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request[1]);
    MPI_Iallreduce(MPI_IN_PLACE, &r_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request[2]);
    MPI_Waitall(3, request, MPI_STATUSES_IGNORE);
    if (b_norm < 1e-16)
        return 0;
    if (r_norm / b_norm < tol * tol)
        return 0;

    int iter;
    for (iter = 0; iter < max_iter; iter++)
    {
        // --- ゴースト節点の通信 ---
        for (int n = 0; n < num_neighbors; n++)
        {
            int ss = send_starts[n];
            int sc = send_counts[n];
            int grid_send = (sc + BLOCK_SIZE - 1) / BLOCK_SIZE;
            if (grid_send > 0)
            {
                kernel_pack_send_buffer<<<grid_send, BLOCK_SIZE>>>(
                    sc, ss, dd.send_nodes,
                    dd.p_0, dd.p_1, dd.p_2,
                    dd.send_buffer_0, dd.send_buffer_1, dd.send_buffer_2);
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // GPU-aware MPI: デバイスポインタを直接送受信
        for (int n = 0; n < num_neighbors; n++)
        {
            int ss = send_starts[n];
            int sc = send_counts[n];
            MPI_Isend(&dd.send_buffer_0[ss], sc, MPI_DOUBLE, neighbor_ranks[n], 0, MPI_COMM_WORLD, &request[n]);
            MPI_Isend(&dd.send_buffer_1[ss], sc, MPI_DOUBLE, neighbor_ranks[n], 1, MPI_COMM_WORLD, &request[num_neighbors + n]);
            MPI_Isend(&dd.send_buffer_2[ss], sc, MPI_DOUBLE, neighbor_ranks[n], 2, MPI_COMM_WORLD, &request[2 * num_neighbors + n]);
            MPI_Irecv(&dd.p_0[recv_starts[n]], recv_counts[n], MPI_DOUBLE, neighbor_ranks[n], 0, MPI_COMM_WORLD, &request[3 * num_neighbors + n]);
            MPI_Irecv(&dd.p_1[recv_starts[n]], recv_counts[n], MPI_DOUBLE, neighbor_ranks[n], 1, MPI_COMM_WORLD, &request[4 * num_neighbors + n]);
            MPI_Irecv(&dd.p_2[recv_starts[n]], recv_counts[n], MPI_DOUBLE, neighbor_ranks[n], 2, MPI_COMM_WORLD, &request[5 * num_neighbors + n]);
        }

        // --- 内側節点の SpMV + pAp ---
        double h_pAp = 0.0;
        CUDA_CHECK(cudaMemset(dd.d_reduce, 0.0, sizeof(double)));

        if (num_inner > 0)
        {
            int grid_inner = (num_inner + BLOCK_SIZE - 1) * 32 / BLOCK_SIZE;
            kernel_spmv<<<grid_inner, BLOCK_SIZE>>>(
                0, num_inner,
                dd.row_ptr, dd.col_ind,
                dd.keval_00, dd.keval_01, dd.keval_02,
                dd.keval_10, dd.keval_11, dd.keval_12,
                dd.keval_20, dd.keval_21, dd.keval_22,
                dd.p_0, dd.p_1, dd.p_2,
                dd.Ap_0, dd.Ap_1, dd.Ap_2);
            CUDA_CHECK(cudaDeviceSynchronize());
            kernel_dot3_reduce<<<grid_inner, BLOCK_SIZE>>>(
                0, num_inner,
                dd.p_0, dd.p_1, dd.p_2,
                dd.Ap_0, dd.Ap_1, dd.Ap_2,
                &dd.d_reduce[0]);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // ゴースト節点の通信完了を待つ
        MPI_Waitall(6 * num_neighbors, request, MPI_STATUSES_IGNORE);

        // --- 外側節点の SpMV + pAp ---
        int num_bdr = num_owned - num_inner;
        if (num_bdr > 0)
        {
            int grid_bdr = (num_bdr + BLOCK_SIZE - 1) * 32 / BLOCK_SIZE;
            kernel_spmv<<<grid_bdr, BLOCK_SIZE>>>(
                num_inner, num_owned,
                dd.row_ptr, dd.col_ind,
                dd.keval_00, dd.keval_01, dd.keval_02,
                dd.keval_10, dd.keval_11, dd.keval_12,
                dd.keval_20, dd.keval_21, dd.keval_22,
                dd.p_0, dd.p_1, dd.p_2,
                dd.Ap_0, dd.Ap_1, dd.Ap_2);
            CUDA_CHECK(cudaDeviceSynchronize());
            kernel_dot3_reduce<<<grid_bdr, BLOCK_SIZE>>>(
                num_inner, num_owned,
                dd.p_0, dd.p_1, dd.p_2,
                dd.Ap_0, dd.Ap_1, dd.Ap_2,
                &dd.d_reduce[0]);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        CUDA_CHECK(cudaMemcpy(&h_pAp, dd.d_reduce, sizeof(double), cudaMemcpyDeviceToHost));
        MPI_Allreduce(MPI_IN_PLACE, &h_pAp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double alpha = rz / h_pAp;

        // x += alpha * p
        kernel_axpy<<<grid_nodes, BLOCK_SIZE>>>(
            num_nodes, alpha,
            dd.p_0, dd.p_1, dd.p_2,
            dd.delta_u_0, dd.delta_u_1, dd.delta_u_2);

        // r -= alpha * Ap, r_norm計算
        CUDA_CHECK(cudaMemset(dd.d_reduce, 0.0, sizeof(double)));
        kernel_update_r<<<grid_owned, BLOCK_SIZE>>>(
            num_owned, alpha,
            dd.Ap_0, dd.Ap_1, dd.Ap_2,
            dd.r_0, dd.r_1, dd.r_2,
            &dd.d_reduce[0]);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&r_norm, dd.d_reduce, sizeof(double), cudaMemcpyDeviceToHost));
        MPI_Allreduce(MPI_IN_PLACE, &r_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if (r_norm / b_norm < tol * tol)
        {
            iter++;
            break;
        }

        // z = C^{-1}r, rz_new計算
        double rz_new = 0.0;
        CUDA_CHECK(cudaMemset(dd.d_reduce, 0.0, sizeof(double)));
        kernel_precond_and_rz<<<grid_owned, BLOCK_SIZE>>>(
            num_owned,
            dd.inv_diag_00, dd.inv_diag_01, dd.inv_diag_02,
            dd.inv_diag_10, dd.inv_diag_11, dd.inv_diag_12,
            dd.inv_diag_20, dd.inv_diag_21, dd.inv_diag_22,
            dd.r_0, dd.r_1, dd.r_2,
            dd.z_0, dd.z_1, dd.z_2,
            &dd.d_reduce[0]);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&rz_new, dd.d_reduce, sizeof(double), cudaMemcpyDeviceToHost));
        MPI_Allreduce(MPI_IN_PLACE, &rz_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double beta = rz_new / rz;

        kernel_update_p<<<grid_owned, BLOCK_SIZE>>>(
            num_owned, beta,
            dd.z_0, dd.z_1, dd.z_2,
            dd.p_0, dd.p_1, dd.p_2);

        rz = rz_new;
    }

    return iter;
}

// ============================================================
// ヘルパー: デバイスメモリ確保 + コピー
// ============================================================
template <typename T>
T *device_alloc(int n)
{
    T *ptr;
    CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
    return ptr;
}

template <typename T>
T *device_alloc_copy(const T *host, int n)
{
    T *ptr;
    CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(ptr, host, n * sizeof(T), cudaMemcpyHostToDevice));
    return ptr;
}

template <typename T>
T *device_alloc_zero(int n)
{
    T *ptr;
    CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
    CUDA_CHECK(cudaMemset(ptr, 0, n * sizeof(T)));
    return ptr;
}

// ============================================================
// main
// ============================================================
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

    double mesh_time = MPI_Wtime() - start_time;
    std::cout << "Mesh reading time: " << mesh_time << " seconds" << std::endl;

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
    double ratio = cfg.get_double("ratio");
    enum class MassVersion { CONSISTENT, LUMPED, SECOND };
    MassVersion mass_version = MassVersion::CONSISTENT;
    if (cfg.has("mass"))
    {
        std::string mv = cfg.get_string("mass");
        if (mv == "lumped")        mass_version = MassVersion::LUMPED;
        else if (mv == "second")   mass_version = MassVersion::SECOND;
    }

    printf("c1: %.2f m/s, c2: %.2f m/s\n, rho: %.2e kg/m^3\n", c1, c2, rho);
    if (rank == 0)
    {
        const char *mv_str = (mass_version == MassVersion::LUMPED)  ? "HRZ lumped (analytical)" :
                             (mass_version == MassVersion::SECOND)   ? "HRZ lumped (FiSTR 4-point Gauss)" :
                                                                       "consistent";
        printf("mass matrix: %s\n", mv_str);
    }

    double lambda = rho * (c1 * c1 - 2 * c2 * c2);
    double mu = rho * c2 * c2;
    double dt = duration / num_steps;
    double dt_inv = 1.0 / dt;
    double dt_inv2 = dt_inv * dt_inv;

    printf("使用可能な最大スレッド数：%d\n", omp_get_max_threads());

    int num_gpus;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    printf("使用可能な最大GPU数：%d\n", num_gpus);

    CUDA_CHECK(cudaSetDevice(rank % num_gpus));

    double *node_coords = mesh.coords_ptr();
    int num_nodes = mesh.num_total;
    int num_inner = mesh.num_inner;
    int num_bdr = mesh.num_bdr;
    int num_owned = mesh.num_owned;
    int num_ghost = mesh.num_ghost;
    int *ele_nodes = mesh.elem_ptr();
    int num_elements = mesh.num_total_elems;
    int grid_nodes = (num_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid_elements = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

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
        send_starts[i] = 0;
        send_counts[i] = mesh.neighbors[i].send_size();
        if (i > 0)
            send_starts[i] = send_starts[i - 1] + send_counts[i - 1];
        send_nodes_vec.insert(send_nodes_vec.end(), mesh.neighbors[i].send_nodes.begin(), mesh.neighbors[i].send_nodes.end());
    }
    send_starts[num_neighbors] = send_starts[num_neighbors - 1] + send_counts[num_neighbors - 1];
    int total_send = send_starts[num_neighbors];
    int *send_nodes_ptr = send_nodes_vec.data();

    if (target_node >= 0)
    {
        printf("Target node local ID: %d\n", target_node);
        printf("Target node global ID: %d\n", mesh.local_to_global[target_node]);
        printf("Target node coordinates: (%.3f, %.3f, %.3f)\n",
               node_coords[target_node * 3],
               node_coords[target_node * 3 + 1],
               node_coords[target_node * 3 + 2]);
    }

    double *u = new double[(num_steps / sample_freq + 1) * 3]();
    double *u_0 = new double[num_nodes]();
    double *u_1 = new double[num_nodes]();
    double *u_2 = new double[num_nodes]();
    int *bc_flag = new int[num_nodes]();
    double bc_val_u[3];

    // 境界条件
#pragma omp parallel for
    for (int i = 0; i < num_nodes; i++)
    {
        if (node_coords[i * 3 + 2] < 1e-6)
        {
            bc_flag[i] = 1;
        }
    }

    double dN0[30] = {0.0}, dN1[30] = {0.0}, dN2[30] = {0.0}, dN3[30] = {0.0};
    gauss_integrate(dN0, dN1, dN2, dN3);
    cudaMemcpyToSymbol(c_dN0, dN0, 30 * sizeof(double));
    cudaMemcpyToSymbol(c_dN1, dN1, 30 * sizeof(double));
    cudaMemcpyToSymbol(c_dN2, dN2, 30 * sizeof(double));
    cudaMemcpyToSymbol(c_dN3, dN3, 30 * sizeof(double));

    double N0[10] = {0.0}, N1[10] = {0.0}, N2[10] = {0.0}, N3[10] = {0.0};
    gauss_integrate_N(N0, N1, N2, N3);
    cudaMemcpyToSymbol(c_N0, N0, 10 * sizeof(double));
    cudaMemcpyToSymbol(c_N1, N1, 10 * sizeof(double));
    cudaMemcpyToSymbol(c_N2, N2, 10 * sizeof(double));
    cudaMemcpyToSymbol(c_N3, N3, 10 * sizeof(double));

    DeviceData dd;
    dd.node_coords = device_alloc_copy(node_coords, num_nodes * 3);
    dd.ele_nodes = device_alloc_copy(ele_nodes, num_elements * 10);
    dd.bc_flag = device_alloc_copy(bc_flag, num_nodes);

    // 剛性行列、質量行列計算
    dd.kmat_coo_val = device_alloc_zero<double>(100 * num_elements * 9);
    dd.mmat_coo_val = device_alloc_zero<double>(100 * num_elements);
    dd.kemat_coo_val = device_alloc_zero<double>(100 * num_elements * 9);
    dd.coo_row = device_alloc_zero<int>(100 * num_elements);
    dd.coo_col = device_alloc_zero<int>(100 * num_elements);

    // 基準となる座標をコピーして記録
    dd.ref_node_coords = device_alloc_copy(node_coords, num_nodes * 3);

    // 変位・速度・加速度 (ゼロ初期化)
    dd.u_tmp_0 = device_alloc_zero<double>(num_nodes);
    dd.u_tmp_1 = device_alloc_zero<double>(num_nodes);
    dd.u_tmp_2 = device_alloc_zero<double>(num_nodes);
    dd.v_tmp_0 = device_alloc_zero<double>(num_nodes);
    dd.v_tmp_1 = device_alloc_zero<double>(num_nodes);
    dd.v_tmp_2 = device_alloc_zero<double>(num_nodes);
    dd.a_tmp_0 = device_alloc_zero<double>(num_nodes);
    dd.a_tmp_1 = device_alloc_zero<double>(num_nodes);
    dd.a_tmp_2 = device_alloc_zero<double>(num_nodes);
    dd.u_prv_0 = device_alloc_zero<double>(num_nodes);
    dd.u_prv_1 = device_alloc_zero<double>(num_nodes);
    dd.u_prv_2 = device_alloc_zero<double>(num_nodes);
    dd.delta_u_0 = device_alloc_zero<double>(num_nodes);
    dd.delta_u_1 = device_alloc_zero<double>(num_nodes);
    dd.delta_u_2 = device_alloc_zero<double>(num_nodes);

    if (mass_version == MassVersion::SECOND)
        construct_mat_second<<<grid_elements, BLOCK_SIZE>>>(
            dd.node_coords, dd.ref_node_coords, dd.delta_u_0, dd.delta_u_1, dd.delta_u_2, ratio,
            dd.ele_nodes, num_elements,
            lambda, mu, rho, dt, dd.kmat_coo_val, dd.mmat_coo_val, dd.kemat_coo_val, dd.coo_row, dd.coo_col);
    else if (mass_version == MassVersion::LUMPED)
        construct_mat_lumped<<<grid_elements, BLOCK_SIZE>>>(
            dd.node_coords, dd.ref_node_coords, dd.delta_u_0, dd.delta_u_1, dd.delta_u_2, ratio,
            dd.ele_nodes, num_elements,
            lambda, mu, rho, dt, dd.kmat_coo_val, dd.mmat_coo_val, dd.kemat_coo_val, dd.coo_row, dd.coo_col);
    else
        construct_mat<<<grid_elements, BLOCK_SIZE>>>(
            dd.node_coords, dd.ref_node_coords, dd.delta_u_0, dd.delta_u_1, dd.delta_u_2, ratio,
            dd.ele_nodes, num_elements,
            lambda, mu, rho, dt, dd.kmat_coo_val, dd.mmat_coo_val, dd.kemat_coo_val, dd.coo_row, dd.coo_col);

    int nnz_bcrs = sort_and_merge_bcoo(100 * num_elements, num_nodes,
                                       dd.coo_row, dd.coo_col, dd.kmat_coo_val, dd.mmat_coo_val, dd.kemat_coo_val);

    dd.kval_00 = device_alloc_zero<double>(nnz_bcrs);
    dd.kval_01 = device_alloc_zero<double>(nnz_bcrs);
    dd.kval_02 = device_alloc_zero<double>(nnz_bcrs);
    dd.kval_10 = device_alloc_zero<double>(nnz_bcrs);
    dd.kval_11 = device_alloc_zero<double>(nnz_bcrs);
    dd.kval_12 = device_alloc_zero<double>(nnz_bcrs);
    dd.kval_20 = device_alloc_zero<double>(nnz_bcrs);
    dd.kval_21 = device_alloc_zero<double>(nnz_bcrs);
    dd.kval_22 = device_alloc_zero<double>(nnz_bcrs);
    dd.mval = device_alloc_zero<double>(nnz_bcrs);
    dd.keval_00 = device_alloc_zero<double>(nnz_bcrs);
    dd.keval_01 = device_alloc_zero<double>(nnz_bcrs);
    dd.keval_02 = device_alloc_zero<double>(nnz_bcrs);
    dd.keval_10 = device_alloc_zero<double>(nnz_bcrs);
    dd.keval_11 = device_alloc_zero<double>(nnz_bcrs);
    dd.keval_12 = device_alloc_zero<double>(nnz_bcrs);
    dd.keval_20 = device_alloc_zero<double>(nnz_bcrs);
    dd.keval_21 = device_alloc_zero<double>(nnz_bcrs);
    dd.keval_22 = device_alloc_zero<double>(nnz_bcrs);
    dd.row_ptr = device_alloc_zero<int>(num_nodes + 1);
    dd.col_ind = device_alloc_zero<int>(nnz_bcrs);
    CUDA_CHECK(cudaDeviceSynchronize());

    build_bcrs(dd.coo_row, dd.coo_col, dd.kmat_coo_val, dd.mmat_coo_val, dd.kemat_coo_val, nnz_bcrs, num_nodes,
               dd.row_ptr, dd.col_ind, dd.kval_00, dd.kval_01, dd.kval_02,
               dd.kval_10, dd.kval_11, dd.kval_12, dd.kval_20, dd.kval_21, dd.kval_22, dd.mval,
               dd.keval_00, dd.keval_01, dd.keval_02, dd.keval_10, dd.keval_11, dd.keval_12, dd.keval_20, dd.keval_21, dd.keval_22);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "Number of non-zero blocks in bcrs: " << nnz_bcrs << std::endl;

    dd.bc_corr_00 = device_alloc_zero<double>(num_nodes);
    dd.bc_corr_01 = device_alloc_zero<double>(num_nodes);
    dd.bc_corr_02 = device_alloc_zero<double>(num_nodes);
    dd.bc_corr_10 = device_alloc_zero<double>(num_nodes);
    dd.bc_corr_11 = device_alloc_zero<double>(num_nodes);
    dd.bc_corr_12 = device_alloc_zero<double>(num_nodes);
    dd.bc_corr_20 = device_alloc_zero<double>(num_nodes);
    dd.bc_corr_21 = device_alloc_zero<double>(num_nodes);
    dd.bc_corr_22 = device_alloc_zero<double>(num_nodes);
    dd.inv_diag_00 = device_alloc_zero<double>(num_nodes);
    dd.inv_diag_01 = device_alloc_zero<double>(num_nodes);
    dd.inv_diag_02 = device_alloc_zero<double>(num_nodes);
    dd.inv_diag_10 = device_alloc_zero<double>(num_nodes);
    dd.inv_diag_11 = device_alloc_zero<double>(num_nodes);
    dd.inv_diag_12 = device_alloc_zero<double>(num_nodes);
    dd.inv_diag_20 = device_alloc_zero<double>(num_nodes);
    dd.inv_diag_21 = device_alloc_zero<double>(num_nodes);
    dd.inv_diag_22 = device_alloc_zero<double>(num_nodes);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 境界条件の補正値計算と、ブロックヤコビの構築
    extract_bc_correction<<<grid_nodes, BLOCK_SIZE>>>(
        num_nodes, dd.row_ptr, dd.col_ind, dd.keval_00, dd.keval_01, dd.keval_02,
        dd.keval_10, dd.keval_11, dd.keval_12, dd.keval_20, dd.keval_21, dd.keval_22, dd.bc_flag,
        dd.bc_corr_00, dd.bc_corr_01, dd.bc_corr_02, dd.bc_corr_10, dd.bc_corr_11, dd.bc_corr_12,
        dd.bc_corr_20, dd.bc_corr_21, dd.bc_corr_22);
    CUDA_CHECK(cudaDeviceSynchronize());

    apply_bc_to_lhs<<<grid_nodes, BLOCK_SIZE>>>(
        num_nodes, dd.row_ptr, dd.col_ind, dd.keval_00, dd.keval_01, dd.keval_02,
        dd.keval_10, dd.keval_11, dd.keval_12, dd.keval_20, dd.keval_21, dd.keval_22,
        dd.bc_flag);
    CUDA_CHECK(cudaDeviceSynchronize());

    build_block_jacobi<<<grid_nodes, BLOCK_SIZE>>>(
        num_nodes, dd.row_ptr, dd.col_ind,
        dd.keval_00, dd.keval_01, dd.keval_02,
        dd.keval_10, dd.keval_11, dd.keval_12,
        dd.keval_20, dd.keval_21, dd.keval_22,
        dd.inv_diag_00, dd.inv_diag_01, dd.inv_diag_02,
        dd.inv_diag_10, dd.inv_diag_11, dd.inv_diag_12,
        dd.inv_diag_20, dd.inv_diag_21, dd.inv_diag_22);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 右辺・一時ベクトル
    dd.rhs_0 = device_alloc_zero<double>(num_nodes);
    dd.rhs_1 = device_alloc_zero<double>(num_nodes);
    dd.rhs_2 = device_alloc_zero<double>(num_nodes);
    dd.tmp_0 = device_alloc_zero<double>(num_nodes);
    dd.tmp_1 = device_alloc_zero<double>(num_nodes);
    dd.tmp_2 = device_alloc_zero<double>(num_nodes);
    dd.rhs_original_0 = device_alloc_zero<double>(num_nodes);
    dd.rhs_original_1 = device_alloc_zero<double>(num_nodes);
    dd.rhs_original_2 = device_alloc_zero<double>(num_nodes);
    dd.f_int_0 = device_alloc_zero<double>(num_nodes);
    dd.f_int_1 = device_alloc_zero<double>(num_nodes);
    dd.f_int_2 = device_alloc_zero<double>(num_nodes);

    // PCGベクトル
    dd.r_0 = device_alloc_zero<double>(num_nodes);
    dd.r_1 = device_alloc_zero<double>(num_nodes);
    dd.r_2 = device_alloc_zero<double>(num_nodes);
    dd.z_0 = device_alloc_zero<double>(num_nodes);
    dd.z_1 = device_alloc_zero<double>(num_nodes);
    dd.z_2 = device_alloc_zero<double>(num_nodes);
    dd.p_0 = device_alloc_zero<double>(num_nodes);
    dd.p_1 = device_alloc_zero<double>(num_nodes);
    dd.p_2 = device_alloc_zero<double>(num_nodes);
    dd.Ap_0 = device_alloc_zero<double>(num_nodes);
    dd.Ap_1 = device_alloc_zero<double>(num_nodes);
    dd.Ap_2 = device_alloc_zero<double>(num_nodes);

    // 境界条件の値
    dd.bc_val_u = device_alloc_zero<double>(3);

    // MPI通信用
    dd.send_nodes = device_alloc_copy(send_nodes_ptr, total_send);
    dd.send_buffer_0 = device_alloc_zero<double>(total_send);
    dd.send_buffer_1 = device_alloc_zero<double>(total_send);
    dd.send_buffer_2 = device_alloc_zero<double>(total_send);

    // リダクション用
    dd.d_reduce = device_alloc_zero<double>(3);

    CUDA_CHECK(cudaDeviceSynchronize());
    double preprocessing_time = MPI_Wtime() - start_time;
    if (rank == 0)
    {
        std::cout << "Preprocessing time: " << preprocessing_time << " seconds" << std::endl;
    }

    char output_dir[256];
    sprintf(output_dir, "results/disp");
    mkdir("results", 0755);
    mkdir(output_dir, 0755);
    char vtk_filename[512];
    sprintf(vtk_filename, "%s/disp_step_%04d.vtk", output_dir, 0);
    write_vtk_displacement(vtk_filename, node_coords, num_nodes, ele_nodes, num_elements,
                           u_0, u_1, u_2, 0.0);

    // ============================================================
    // タイムステップループ(増分が方程式の解)
    // ============================================================
    for (int step = 1; step <= num_steps; step++)
    {
        CUDA_CHECK(cudaDeviceSynchronize());
        double step_start_time = MPI_Wtime();

        // delta uのリセット
        CUDA_CHECK(cudaMemset(dd.delta_u_0, 0.0, num_nodes * sizeof(double)));
        CUDA_CHECK(cudaMemset(dd.delta_u_1, 0.0, num_nodes * sizeof(double)));
        CUDA_CHECK(cudaMemset(dd.delta_u_2, 0.0, num_nodes * sizeof(double)));

        double t = step * dt;
        bc_val_u[0] = 0.0;
        bc_val_u[1] = 0.05 * sin(t) - 0.05 * sin(t - dt);
        bc_val_u[2] = 0.0;
        CUDA_CHECK(cudaMemcpy(dd.bc_val_u, bc_val_u, 3 * sizeof(double), cudaMemcpyHostToDevice));

        // Newmark-β RHS構築
        kernel_build_newmark_tmp<<<grid_nodes, BLOCK_SIZE>>>(
            num_nodes, dt_inv2, dt_inv,
            dd.v_tmp_0, dd.v_tmp_1, dd.v_tmp_2,
            dd.a_tmp_0, dd.a_tmp_1, dd.a_tmp_2,
            dd.tmp_0, dd.tmp_1, dd.tmp_2);

        // 質量行列SpMV: rhs_original = M * tmp
        kernel_bcrs_spmv_m<<<grid_nodes, BLOCK_SIZE>>>(
            num_nodes, dd.row_ptr, dd.col_ind, dd.mval,
            dd.tmp_0, dd.tmp_1, dd.tmp_2,
            dd.rhs_original_0, dd.rhs_original_1, dd.rhs_original_2);

        while (true)
        {
            CUDA_CHECK(cudaDeviceSynchronize());
            double inner_loop_start_time = MPI_Wtime();

            // rhs = rhs_original - f_int
            CUDA_CHECK(cudaMemcpy(dd.rhs_0, dd.rhs_original_0, num_nodes * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(dd.rhs_1, dd.rhs_original_1, num_nodes * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(dd.rhs_2, dd.rhs_original_2, num_nodes * sizeof(double), cudaMemcpyDeviceToDevice));
            kernel_axpy<<<grid_nodes, BLOCK_SIZE>>>(
                num_nodes, -1.0,
                dd.f_int_0, dd.f_int_1, dd.f_int_2,
                dd.rhs_0, dd.rhs_1, dd.rhs_2);

            // 外力の寄与
            if (force_node >= 0)
            {
                double h_rhs_val;
                double *d_rhs_target;
                if (force_dof == 0)
                    d_rhs_target = &dd.rhs_0[force_node];
                else if (force_dof == 1)
                    d_rhs_target = &dd.rhs_1[force_node];
                else
                    d_rhs_target = &dd.rhs_2[force_node];

                CUDA_CHECK(cudaMemcpy(&h_rhs_val, d_rhs_target, sizeof(double), cudaMemcpyDeviceToHost));
                h_rhs_val += force_magnitude;
                CUDA_CHECK(cudaMemcpy(d_rhs_target, &h_rhs_val, sizeof(double), cudaMemcpyHostToDevice));
            }

            // 境界条件を右辺に適用
            kernel_apply_bc_to_rhs<<<grid_nodes, BLOCK_SIZE>>>(
                num_nodes, dd.bc_val_u, dd.bc_flag,
                dd.bc_corr_00, dd.bc_corr_01, dd.bc_corr_02,
                dd.bc_corr_10, dd.bc_corr_11, dd.bc_corr_12,
                dd.bc_corr_20, dd.bc_corr_21, dd.bc_corr_22,
                dd.rhs_0, dd.rhs_1, dd.rhs_2);

            CUDA_CHECK(cudaDeviceSynchronize());

            // PCGソルバー
            int iter = pcg_solve(dd, requests, num_neighbors, neighbor_ranks,
                                 recv_starts, recv_counts, send_starts, send_counts,
                                 num_inner, num_owned, num_nodes, 1e-12, num_nodes * 3);

            MPI_Allreduce(MPI_IN_PLACE, &iter, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

            if (rank == 0)
            {
                std::cout << "Step " << step << ", PCG iterations: " << iter << std::endl;
            }

            // delta_uに解が入っているので、これを増分として加算
            kernel_axpy<<<grid_nodes, BLOCK_SIZE>>>(
                num_nodes, 1.0,
                dd.delta_u_0, dd.delta_u_1, dd.delta_u_2,
                dd.u_tmp_0, dd.u_tmp_1, dd.u_tmp_2);

            double delta_u_norm;
            CUDA_CHECK(cudaMemset(dd.d_reduce, 0.0, sizeof(double)));
            kernel_dot3_reduce<<<grid_nodes, BLOCK_SIZE>>>(
                0, num_nodes, dd.delta_u_0, dd.delta_u_1, dd.delta_u_2,
                dd.delta_u_0, dd.delta_u_1, dd.delta_u_2, &dd.d_reduce[0]);
            CUDA_CHECK(cudaMemcpy(&delta_u_norm, &dd.d_reduce[0], sizeof(double), cudaMemcpyDeviceToHost));
            std::cout << "Step " << step << ", delta_u norm: " << delta_u_norm << std::endl;

            // 剛性行列、質量行列の再構築、右辺ベクトルの更新
            {
                // 2回目以降のループでは、境界条件の増分はゼロにする
                bc_val_u[0] = 0.0;
                bc_val_u[1] = 0.0;
                bc_val_u[2] = 0.0;
                CUDA_CHECK(cudaMemcpy(dd.bc_val_u, bc_val_u, 3 * sizeof(double), cudaMemcpyHostToDevice));

                if (mass_version == MassVersion::SECOND)
                    construct_mat_second<<<grid_elements, BLOCK_SIZE>>>(
                        dd.node_coords, dd.ref_node_coords, dd.delta_u_0, dd.delta_u_1, dd.delta_u_2, ratio,
                        dd.ele_nodes, num_elements,
                        lambda, mu, rho, dt, dd.kmat_coo_val, dd.mmat_coo_val, dd.kemat_coo_val, dd.coo_row, dd.coo_col);
                else if (mass_version == MassVersion::LUMPED)
                    construct_mat_lumped<<<grid_elements, BLOCK_SIZE>>>(
                        dd.node_coords, dd.ref_node_coords, dd.delta_u_0, dd.delta_u_1, dd.delta_u_2, ratio,
                        dd.ele_nodes, num_elements,
                        lambda, mu, rho, dt, dd.kmat_coo_val, dd.mmat_coo_val, dd.kemat_coo_val, dd.coo_row, dd.coo_col);
                else
                    construct_mat<<<grid_elements, BLOCK_SIZE>>>(
                        dd.node_coords, dd.ref_node_coords, dd.delta_u_0, dd.delta_u_1, dd.delta_u_2, ratio,
                        dd.ele_nodes, num_elements,
                        lambda, mu, rho, dt, dd.kmat_coo_val, dd.mmat_coo_val, dd.kemat_coo_val, dd.coo_row, dd.coo_col);

                int nnz_bcrs = sort_and_merge_bcoo(100 * num_elements, num_nodes,
                                                   dd.coo_row, dd.coo_col, dd.kmat_coo_val, dd.mmat_coo_val, dd.kemat_coo_val);

                build_bcrs(dd.coo_row, dd.coo_col, dd.kmat_coo_val, dd.mmat_coo_val, dd.kemat_coo_val, nnz_bcrs, num_nodes,
                           dd.row_ptr, dd.col_ind,
                           dd.kval_00, dd.kval_01, dd.kval_02,
                           dd.kval_10, dd.kval_11, dd.kval_12,
                           dd.kval_20, dd.kval_21, dd.kval_22,
                           dd.mval,
                           dd.keval_00, dd.keval_01, dd.keval_02,
                           dd.keval_10, dd.keval_11, dd.keval_12,
                           dd.keval_20, dd.keval_21, dd.keval_22);

                extract_bc_correction<<<grid_nodes, BLOCK_SIZE>>>(
                    num_nodes, dd.row_ptr, dd.col_ind,
                    dd.keval_00, dd.keval_01, dd.keval_02,
                    dd.keval_10, dd.keval_11, dd.keval_12,
                    dd.keval_20, dd.keval_21, dd.keval_22,
                    dd.bc_flag,
                    dd.bc_corr_00, dd.bc_corr_01, dd.bc_corr_02,
                    dd.bc_corr_10, dd.bc_corr_11, dd.bc_corr_12,
                    dd.bc_corr_20, dd.bc_corr_21, dd.bc_corr_22);

                apply_bc_to_lhs<<<grid_nodes, BLOCK_SIZE>>>(
                    num_nodes, dd.row_ptr, dd.col_ind,
                    dd.keval_00, dd.keval_01, dd.keval_02,
                    dd.keval_10, dd.keval_11, dd.keval_12,
                    dd.keval_20, dd.keval_21, dd.keval_22,
                    dd.bc_flag);

                build_block_jacobi<<<grid_nodes, BLOCK_SIZE>>>(
                    num_nodes, dd.row_ptr, dd.col_ind,
                    dd.keval_00, dd.keval_01, dd.keval_02,
                    dd.keval_10, dd.keval_11, dd.keval_12,
                    dd.keval_20, dd.keval_21, dd.keval_22,
                    dd.inv_diag_00, dd.inv_diag_01, dd.inv_diag_02,
                    dd.inv_diag_10, dd.inv_diag_11, dd.inv_diag_12,
                    dd.inv_diag_20, dd.inv_diag_21, dd.inv_diag_22);

                // 右辺ベクトルの更新: f_int += Kmat * delta_u
                int grid_spmv = (num_nodes * 32 + BLOCK_SIZE - 1) / BLOCK_SIZE; // スレッドあたり32要素処理
                kernel_spmv<<<grid_spmv, BLOCK_SIZE>>>(
                    0, num_nodes,
                    dd.row_ptr, dd.col_ind,
                    dd.kval_00, dd.kval_01, dd.kval_02,
                    dd.kval_10, dd.kval_11, dd.kval_12,
                    dd.kval_20, dd.kval_21, dd.kval_22,
                    dd.delta_u_0, dd.delta_u_1, dd.delta_u_2,
                    dd.tmp_0, dd.tmp_1, dd.tmp_2);
                kernel_axpy<<<grid_nodes, BLOCK_SIZE>>>(
                    num_nodes, 1.0,
                    dd.tmp_0, dd.tmp_1, dd.tmp_2,
                    dd.f_int_0, dd.f_int_1, dd.f_int_2);
                // rhs_original -= 4/dt^2 * M * delta_u
                kernel_bcrs_spmv_m<<<grid_nodes, BLOCK_SIZE>>>(
                    num_nodes, dd.row_ptr, dd.col_ind, dd.mval,
                    dd.delta_u_0, dd.delta_u_1, dd.delta_u_2,
                    dd.tmp_0, dd.tmp_1, dd.tmp_2);
                kernel_axpy<<<grid_nodes, BLOCK_SIZE>>>(
                    num_nodes, -4.0 * dt_inv2,
                    dd.tmp_0, dd.tmp_1, dd.tmp_2,
                    dd.rhs_original_0, dd.rhs_original_1, dd.rhs_original_2);

                // 変位で座標を更新
                update_coords<<<grid_nodes, BLOCK_SIZE>>>(
                    dd.ref_node_coords,
                    dd.u_tmp_0, dd.u_tmp_1, dd.u_tmp_2,
                    dd.node_coords, num_nodes);

                // delta uのリセット
                CUDA_CHECK(cudaMemset(dd.delta_u_0, 0.0, num_nodes * sizeof(double)));
                CUDA_CHECK(cudaMemset(dd.delta_u_1, 0.0, num_nodes * sizeof(double)));
                CUDA_CHECK(cudaMemset(dd.delta_u_2, 0.0, num_nodes * sizeof(double)));
            }

            CUDA_CHECK(cudaDeviceSynchronize());

            double inner_loop_time = MPI_Wtime() - inner_loop_start_time;
            if (rank == 0)
            {
                std::cout << "Step " << step << " inner loop completed in " << inner_loop_time << " seconds." << std::endl;
            }

            // delta uがゼロに近いなら収束とみなしてループを抜ける
            if (std::sqrt(delta_u_norm) < 1e-8)
            {
                std::cout << "Converged" << std::endl;
                break;
            }
        }

        // 改めて境界条件の値をセット（更新のため）
        bc_val_u[0] = 0.0;
        bc_val_u[1] = 0.05 * sin(t);
        bc_val_u[2] = 0.0;
        CUDA_CHECK(cudaMemcpy(dd.bc_val_u, bc_val_u, 3 * sizeof(double), cudaMemcpyHostToDevice));

        // Newmark-β更新
        kernel_newmark_update<<<grid_nodes, BLOCK_SIZE>>>(
            num_nodes, dt, dt_inv, dt_inv2,
            dd.bc_flag, dd.bc_val_u,
            dd.u_tmp_0, dd.u_tmp_1, dd.u_tmp_2,
            dd.u_prv_0, dd.u_prv_1, dd.u_prv_2,
            dd.v_tmp_0, dd.v_tmp_1, dd.v_tmp_2,
            dd.a_tmp_0, dd.a_tmp_1, dd.a_tmp_2);

        // サンプリング
        if (step % sample_freq == 0)
        {
            if (target_node >= 0)
            {
                double h_u[3];
                CUDA_CHECK(cudaMemcpy(&h_u[0], &dd.u_prv_0[target_node], sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(&h_u[1], &dd.u_prv_1[target_node], sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(&h_u[2], &dd.u_prv_2[target_node], sizeof(double), cudaMemcpyDeviceToHost));
                u[(step / sample_freq) * 3 + 0] = h_u[0];
                u[(step / sample_freq) * 3 + 1] = h_u[1];
                u[(step / sample_freq) * 3 + 2] = h_u[2];
                std::cout << "Step " << step << ", Target node displacement: ("
                          << h_u[0] << ", " << h_u[1] << ", " << h_u[2] << ")" << std::endl;
            }
            {
                CUDA_CHECK(cudaMemcpy(u_0, dd.u_prv_0, num_nodes * sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(u_1, dd.u_prv_1, num_nodes * sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(u_2, dd.u_prv_2, num_nodes * sizeof(double), cudaMemcpyDeviceToHost));
                sprintf(vtk_filename, "%s/disp_step_%04d.vtk", output_dir, step);
                write_vtk_displacement(vtk_filename, node_coords, num_nodes, ele_nodes, num_elements,
                                       u_0, u_1, u_2, t);
            }
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        double step_time = MPI_Wtime() - step_start_time;
        if (rank == 0)
        {
            std::cout << "Step " << step << " completed in " << step_time << " seconds." << std::endl;
        }
    }

    if (target_node >= 0)
    {
        time_t now = time(nullptr);
        struct tm *lt = localtime(&now);
        char timestamp[64];
        strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", lt);

        char output_dir[256];
        sprintf(output_dir, "results/%s", timestamp);
        mkdir("results", 0755);
        mkdir(output_dir, 0755);

        char csv_filename[512];
        sprintf(csv_filename, "%s/target_disp.csv", output_dir);
        write_node_disp_csv(csv_filename, u, num_steps, sample_freq, dt);

        printf("Output: %s/\n", output_dir);
        printf("CSV: %s\n", csv_filename);
    }

    // ============================================================
    // GPU メモリ解放
    // ============================================================
    cudaFree(dd.row_ptr);
    cudaFree(dd.col_ind);
    cudaFree(dd.kval_00);
    cudaFree(dd.kval_01);
    cudaFree(dd.kval_02);
    cudaFree(dd.kval_10);
    cudaFree(dd.kval_11);
    cudaFree(dd.kval_12);
    cudaFree(dd.kval_20);
    cudaFree(dd.kval_21);
    cudaFree(dd.kval_22);
    cudaFree(dd.mval);
    cudaFree(dd.keval_00);
    cudaFree(dd.keval_01);
    cudaFree(dd.keval_02);
    cudaFree(dd.keval_10);
    cudaFree(dd.keval_11);
    cudaFree(dd.keval_12);
    cudaFree(dd.keval_20);
    cudaFree(dd.keval_21);
    cudaFree(dd.keval_22);
    cudaFree(dd.bc_flag);
    cudaFree(dd.bc_corr_00);
    cudaFree(dd.bc_corr_01);
    cudaFree(dd.bc_corr_02);
    cudaFree(dd.bc_corr_10);
    cudaFree(dd.bc_corr_11);
    cudaFree(dd.bc_corr_12);
    cudaFree(dd.bc_corr_20);
    cudaFree(dd.bc_corr_21);
    cudaFree(dd.bc_corr_22);
    cudaFree(dd.inv_diag_00);
    cudaFree(dd.inv_diag_01);
    cudaFree(dd.inv_diag_02);
    cudaFree(dd.inv_diag_10);
    cudaFree(dd.inv_diag_11);
    cudaFree(dd.inv_diag_12);
    cudaFree(dd.inv_diag_20);
    cudaFree(dd.inv_diag_21);
    cudaFree(dd.inv_diag_22);
    cudaFree(dd.u_tmp_0);
    cudaFree(dd.u_tmp_1);
    cudaFree(dd.u_tmp_2);
    cudaFree(dd.v_tmp_0);
    cudaFree(dd.v_tmp_1);
    cudaFree(dd.v_tmp_2);
    cudaFree(dd.a_tmp_0);
    cudaFree(dd.a_tmp_1);
    cudaFree(dd.a_tmp_2);
    cudaFree(dd.u_prv_0);
    cudaFree(dd.u_prv_1);
    cudaFree(dd.u_prv_2);
    cudaFree(dd.delta_u_0);
    cudaFree(dd.delta_u_1);
    cudaFree(dd.delta_u_2);
    cudaFree(dd.rhs_0);
    cudaFree(dd.rhs_1);
    cudaFree(dd.rhs_2);
    cudaFree(dd.tmp_0);
    cudaFree(dd.tmp_1);
    cudaFree(dd.tmp_2);
    cudaFree(dd.rhs_original_0);
    cudaFree(dd.rhs_original_1);
    cudaFree(dd.rhs_original_2);
    cudaFree(dd.f_int_0);
    cudaFree(dd.f_int_1);
    cudaFree(dd.f_int_2);
    cudaFree(dd.r_0);
    cudaFree(dd.r_1);
    cudaFree(dd.r_2);
    cudaFree(dd.z_0);
    cudaFree(dd.z_1);
    cudaFree(dd.z_2);
    cudaFree(dd.p_0);
    cudaFree(dd.p_1);
    cudaFree(dd.p_2);
    cudaFree(dd.Ap_0);
    cudaFree(dd.Ap_1);
    cudaFree(dd.Ap_2);
    cudaFree(dd.bc_val_u);
    cudaFree(dd.send_nodes);
    cudaFree(dd.send_buffer_0);
    cudaFree(dd.send_buffer_1);
    cudaFree(dd.send_buffer_2);
    cudaFree(dd.d_reduce);

    // CPU メモリ解放
    delete[] bc_flag;
    delete[] u;
    delete[] u_0;
    delete[] u_1;
    delete[] u_2;
    delete[] neighbor_ranks;
    delete[] recv_starts;
    delete[] recv_counts;
    delete[] send_starts;
    delete[] send_counts;

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