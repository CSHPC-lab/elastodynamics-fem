#include "vtk_reader.hpp"
#include <iostream>
#include <string>

void construct_kmat(double *kmat_coo, const double *node_coords, const int *ele_nodes, int num_elements)
{
    // ここに要素剛性行列 kmat_coo を構築するコードを実装
    // node_coords: [num_nodes * 3] 連続配置: x0,y0,z0,x1,y1,z1,...
    // ele_nodes: [num_elements * 10] 連続配置: 要素0の10節点, 要素1の10節点,...
    // kmat_coo は [900 * num_elements] の配列で、各要素の行列を連続配置で格納
}

void construct_mmat(double *mmat_coo, const double *node_coords, const int *ele_nodes, int num_elements)
{
    // ここに要素質量行列 mmat_coo を構築するコードを実装
    // node_coords: [num_nodes * 3] 連続配置: x0,y0,z0,x1,y1,z1,...
    // ele_nodes: [num_elements * 10] 連続配置: 要素0の10節点, 要素1の10節点,...
    // mmat_coo は [900 * num_elements] の配列で、各要素の行列を連続配置で格納
    double local_mmat[100] = {
        6.0 / 2520.0, 1.0 / 2520.0, 1.0 / 2520.0, 1.0 / 2520.0, -4.0 / 2520.0, -6.0 / 2520.0, -4.0 / 2520.0, -6.0 / 2520.0, -6.0 / 2520.0, -4.0 / 2520.0,
        1.0 / 2520.0, 6.0 / 2520.0, 1.0 / 2520.0, 1.0 / 2520.0, -4.0 / 2520.0, -4.0 / 2520.0, -6.0 / 2520.0, -4.0 / 2520.0, -6.0 / 2520.0, -6.0 / 2520.0,
        1.0 / 2520.0, 1.0 / 2520.0, 6.0 / 2520.0, 1.0 / 2520.0, -6.0 / 2520.0, -4.0 / 2520.0, -4.0 / 2520.0, -6.0 / 2520.0, -4.0 / 2520.0, -6.0 / 2520.0,
        1.0 / 2520.0, 1.0 / 2520.0, 1.0 / 2520.0, 6.0 / 2520.0, -6.0 / 2520.0, -6.0 / 2520.0, -6.0 / 2520.0, -4.0 / 2520.0, -4.0 / 2520.0, -4.0 / 2520.0,
        -4.0 / 2520.0, -4.0 / 2520.0, -6.0 / 2520.0, -6.0 / 2520.0, 32.0 / 2520.0, 16.0 / 2520.0, 16.0 / 2520.0, 16.0 / 2520.0, 8.0 / 2520.0, 16.0 / 2520.0,
        -6.0 / 2520.0, -4.0 / 2520.0, -4.0 / 2520.0, -6.0 / 2520.0, 16.0 / 2520.0, 32.0 / 2520.0, 16.0 / 2520.0, 16.0 / 2520.0, 16.0 / 2520.0, 8.0 / 2520.0,
        -4.0 / 2520.0, -6.0 / 2520.0, -4.0 / 2520.0, -6.0 / 2520.0, 16.0 / 2520.0, 16.0 / 2520.0, 32.0 / 2520.0, 8.0 / 2520.0, 16.0 / 2520.0, 16.0 / 2520.0,
        -6.0 / 2520.0, -4.0 / 2520.0, -6.0 / 2520.0, -4.0 / 2520.0, 16.0 / 2520.0, 16.0 / 2520.0, 8.0 / 2520.0, 32.0 / 2520.0, 16.0 / 2520.0, 16.0 / 2520.0,
        -6.0 / 2520.0, -6.0 / 2520.0, -4.0 / 2520.0, -4.0 / 2520.0, 8.0 / 2520.0, 16.0 / 2520.0, 16.0 / 2520.0, 16.0 / 2520.0, 32.0 / 2520.0, 16.0 / 2520.0,
        -4.0 / 2520.0, -6.0 / 2520.0, -6.0 / 2520.0, -4.0 / 2520.0, 16.0 / 2520.0, 8.0 / 2520.0, 16.0 / 2520.0, 16.0 / 2520.0, 16.0 / 2520.0, 32.0 / 2520.0};
}

int main()
{
    std::string filepath = "column.vtk";

    FEMMesh mesh = read_vtk(filepath);
    print_mesh_info(mesh);

    double *node_coords = mesh.coords_ptr();
    int num_nodes = mesh.num_nodes;
    double u[num_nodes * 3] = {0}; // 変位ベクトルの配列（初期化）
    int *ele_nodes = mesh.tet_ptr();
    int num_elements = mesh.num_tets;
    double kmat_coo[900 * num_elements] = {0}; // 要素剛性行列の配列（初期化）
    double mmat_coo[900 * num_elements] = {0}; // 要素質量行列の配列（初期化）

    std::cout << "node[0] = (" << node_coords[0] << ", "
              << node_coords[1] << ", "
              << node_coords[2] << ")" << std::endl;

    std::cout << "tet[0] = (" << node_coords[3 * ele_nodes[0]] << ", "
              << node_coords[3 * ele_nodes[0] + 1] << ", "
              << node_coords[3 * ele_nodes[0] + 2] << "), ("
              << node_coords[3 * ele_nodes[1]] << ", "
              << node_coords[3 * ele_nodes[1] + 1] << ", "
              << node_coords[3 * ele_nodes[1] + 2] << "), ("
              << node_coords[3 * ele_nodes[2]] << ", "
              << node_coords[3 * ele_nodes[2] + 1] << ", "
              << node_coords[3 * ele_nodes[2] + 2] << "), ("
              << node_coords[3 * ele_nodes[3]] << ", "
              << node_coords[3 * ele_nodes[3] + 1] << ", "
              << node_coords[3 * ele_nodes[3] + 2] << "), ("
              << node_coords[3 * ele_nodes[4]] << ", "
              << node_coords[3 * ele_nodes[4] + 1] << ", "
              << node_coords[3 * ele_nodes[4] + 2] << "), ("
              << node_coords[3 * ele_nodes[5]] << ", "
              << node_coords[3 * ele_nodes[5] + 1] << ", "
              << node_coords[3 * ele_nodes[5] + 2] << "), ("
              << node_coords[3 * ele_nodes[6]] << ", "
              << node_coords[3 * ele_nodes[6] + 1] << ", "
              << node_coords[3 * ele_nodes[6] + 2] << "), ("
              << node_coords[3 * ele_nodes[7]] << ", "
              << node_coords[3 * ele_nodes[7] + 1] << ", "
              << node_coords[3 * ele_nodes[7] + 2] << "), ("
              << node_coords[3 * ele_nodes[8]] << ", "
              << node_coords[3 * ele_nodes[8] + 1] << ", "
              << node_coords[3 * ele_nodes[8] + 2] << "), ("
              << node_coords[3 * ele_nodes[9]] << ", "
              << node_coords[3 * ele_nodes[9] + 1] << ", "
              << node_coords[3 * ele_nodes[9] + 2] << ")" << std::endl;

    return 0;
}

/*実行コマンド
ssh lynx10
cd /data3/kusumoto/elastodynamics-fem/
g++ main.cpp vtk_reader.cpp
./a.out
*/