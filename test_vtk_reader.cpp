#include "vtk_reader.hpp"
#include <iostream>
#include <cmath>

double tet_volume(const FEMMesh& mesh, int e)
{
    const int* nd = &mesh.tet_nodes[e * 10];
    const double* p0 = &mesh.node_coords[nd[0] * 3];
    const double* p1 = &mesh.node_coords[nd[1] * 3];
    const double* p2 = &mesh.node_coords[nd[2] * 3];
    const double* p3 = &mesh.node_coords[nd[3] * 3];

    double a[3] = {p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]};
    double b[3] = {p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]};
    double c[3] = {p3[0]-p0[0], p3[1]-p0[1], p3[2]-p0[2]};

    double det = a[0]*(b[1]*c[2] - b[2]*c[1])
               - a[1]*(b[0]*c[2] - b[2]*c[0])
               + a[2]*(b[0]*c[1] - b[1]*c[0]);

    return std::abs(det) / 6.0;
}

void check_midpoint_linearity(const FEMMesh& mesh)
{
    int edge_map[6][3] = {
        {4, 0, 1}, {5, 1, 2}, {6, 0, 2},
        {7, 0, 3}, {8, 1, 3}, {9, 2, 3}
    };

    double max_err = 0.0;
    for (int i = 0; i < mesh.num_tets; ++i) {
        const int* nd = &mesh.tet_nodes[i * 10];
        for (int e = 0; e < 6; ++e) {
            int mid = nd[edge_map[e][0]];
            int va  = nd[edge_map[e][1]];
            int vb  = nd[edge_map[e][2]];
            for (int d = 0; d < 3; ++d) {
                double expected = 0.5 * (mesh.node_coords[va*3+d]
                                       + mesh.node_coords[vb*3+d]);
                double actual   = mesh.node_coords[mid*3+d];
                double err = std::abs(actual - expected);
                if (err > max_err) max_err = err;
            }
        }
    }
    std::cout << "  Midpoint linearity check: max error = " << max_err << std::endl;
    if (max_err < 1e-10) {
        std::cout << "  -> OK: All mid-edge nodes are linear." << std::endl;
    } else {
        std::cout << "  -> WARNING: Mid-edge nodes deviate." << std::endl;
    }
}

int main(int argc, char** argv)
{
    std::string filepath = "column.vtk";
    if (argc > 1) filepath = argv[1];

    try {
        FEMMesh mesh = read_vtk(filepath);
        print_mesh_info(mesh);

        double total_vol = 0.0;
        for (int i = 0; i < mesh.num_tets; ++i) {
            total_vol += tet_volume(mesh, i);
        }
        std::cout << std::endl;
        std::cout << "  Total volume: " << total_vol << std::endl;
        std::cout << "  Expected:     10000" << std::endl;
        std::cout << std::endl;

        check_midpoint_linearity(mesh);

        // OpenACC での使い方:
        //
        // double* coords = mesh.coords_ptr();
        // int*    tets   = mesh.tet_ptr();
        // int     n_tets = mesh.num_tets;
        // int     n_nodes = mesh.num_nodes;
        //
        // #pragma acc data copyin(coords[0:n_nodes*3], tets[0:n_tets*10])
        // {
        //     #pragma acc parallel loop
        //     for (int e = 0; e < n_tets; ++e) {
        //         const int* nd = &tets[e * 10];
        //         // ...
        //     }
        // }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    // mesh はここでスコープを抜けて自動解放される
    return 0;
}
