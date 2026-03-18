#include "vtk_reader.hpp"
#include <iostream>
#include <string>

int main()
{
    std::string filepath = "column.vtk";

    FEMMesh mesh = read_vtk(filepath);
    print_mesh_info(mesh);

    std::cout << "node[0] = (" << mesh.node_coords[0] << ", "
              << mesh.node_coords[1] << ", "
              << mesh.node_coords[2] << ")" << std::endl;

    std::cout << "tet[0] = (" << mesh.node_coords[3 * mesh.tet_nodes[0]] << ", "
              << mesh.node_coords[3 * mesh.tet_nodes[0] + 1] << ", "
              << mesh.node_coords[3 * mesh.tet_nodes[0] + 2] << "), ("
              << mesh.node_coords[3 * mesh.tet_nodes[1]] << ", "
              << mesh.node_coords[3 * mesh.tet_nodes[1] + 1] << ", "
              << mesh.node_coords[3 * mesh.tet_nodes[1] + 2] << "), ("
              << mesh.node_coords[3 * mesh.tet_nodes[2]] << ", "
              << mesh.node_coords[3 * mesh.tet_nodes[2] + 1] << ", "
              << mesh.node_coords[3 * mesh.tet_nodes[2] + 2] << "), ("
              << mesh.node_coords[3 * mesh.tet_nodes[3]] << ", "
              << mesh.node_coords[3 * mesh.tet_nodes[3] + 1] << ", "
              << mesh.node_coords[3 * mesh.tet_nodes[3] + 2] << "), ("
              << mesh.node_coords[3 * mesh.tet_nodes[4]] << ", "
              << mesh.node_coords[3 * mesh.tet_nodes[4] + 1] << ", "
              << mesh.node_coords[3 * mesh.tet_nodes[4] + 2] << "), ("
              << mesh.node_coords[3 * mesh.tet_nodes[5]] << ", "
              << mesh.node_coords[3 * mesh.tet_nodes[5] + 1] << ", "
              << mesh.node_coords[3 * mesh.tet_nodes[5] + 2] << "), ("
              << mesh.node_coords[3 * mesh.tet_nodes[6]] << ", "
              << mesh.node_coords[3 * mesh.tet_nodes[6] + 1] << ", "
              << mesh.node_coords[3 * mesh.tet_nodes[6] + 2] << "), ("
              << mesh.node_coords[3 * mesh.tet_nodes[7]] << ", "
              << mesh.node_coords[3 * mesh.tet_nodes[7] + 1] << ", "
              << mesh.node_coords[3 * mesh.tet_nodes[7] + 2] << "), ("
              << mesh.node_coords[3 * mesh.tet_nodes[8]] << ", "
              << mesh.node_coords[3 * mesh.tet_nodes[8] + 1] << ", "
              << mesh.node_coords[3 * mesh.tet_nodes[8] + 2] << "), ("
              << mesh.node_coords[3 * mesh.tet_nodes[9]] << ", "
              << mesh.node_coords[3 * mesh.tet_nodes[9] + 1] << ", "
              << mesh.node_coords[3 * mesh.tet_nodes[9] + 2] << ")" << std::endl;

    return 0;
}

/*実行コマンド
g++ main.cpp vtk_reader.cpp
./a.out
*/