#include "vtk_reader.hpp"

#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <cstring>

// =============================================================================
//  VTK Cell Type IDs
// =============================================================================
static constexpr int VTK_QUADRATIC_TRIANGLE = 22;
static constexpr int VTK_QUADRATIC_TETRA    = 24;

// =============================================================================
//  read_vtk
// =============================================================================
FEMMesh read_vtk(const std::string& filepath)
{
    std::ifstream ifs(filepath);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    FEMMesh mesh;
    std::string line;

    // ヘッダ (4行)
    for (int i = 0; i < 4; ++i) {
        if (!std::getline(ifs, line)) {
            throw std::runtime_error("Unexpected end of file in header");
        }
    }
    if (line.find("UNSTRUCTURED_GRID") == std::string::npos) {
        throw std::runtime_error("Not an UNSTRUCTURED_GRID VTK file");
    }

    // ----- POINTS -----
    while (std::getline(ifs, line)) {
        if (line.find("POINTS") == 0) break;
    }
    {
        std::istringstream iss(line);
        std::string keyword, dtype;
        iss >> keyword >> mesh.num_nodes >> dtype;
    }

    mesh.node_coords.resize(mesh.num_nodes * 3);
    for (int i = 0; i < mesh.num_nodes * 3; ++i) {
        ifs >> mesh.node_coords[i];
    }

    // ----- CELLS -----
    while (std::getline(ifs, line)) {
        if (line.find("CELLS") == 0) break;
    }
    int num_cells = 0, total_ints = 0;
    {
        std::istringstream iss(line);
        std::string keyword;
        iss >> keyword >> num_cells >> total_ints;
    }

    std::vector<int> cell_sizes(num_cells);
    std::vector<std::vector<int>> cell_data(num_cells);
    for (int i = 0; i < num_cells; ++i) {
        ifs >> cell_sizes[i];
        cell_data[i].resize(cell_sizes[i]);
        for (int j = 0; j < cell_sizes[i]; ++j) {
            ifs >> cell_data[i][j];
        }
    }

    // ----- CELL_TYPES -----
    while (std::getline(ifs, line)) {
        if (line.find("CELL_TYPES") == 0) break;
    }

    std::vector<int> cell_types(num_cells);
    for (int i = 0; i < num_cells; ++i) {
        ifs >> cell_types[i];
    }
    ifs.close();

    // ----- カウント -----
    mesh.num_tets = 0;
    mesh.num_tris = 0;
    for (int i = 0; i < num_cells; ++i) {
        if      (cell_types[i] == VTK_QUADRATIC_TETRA)    ++mesh.num_tets;
        else if (cell_types[i] == VTK_QUADRATIC_TRIANGLE) ++mesh.num_tris;
    }

    // ----- 連続配列に格納 -----
    mesh.tet_nodes.resize(mesh.num_tets * 10);
    mesh.tri_nodes.resize(mesh.num_tris * 6);

    int tet_idx = 0, tri_idx = 0;
    for (int i = 0; i < num_cells; ++i) {
        if (cell_types[i] == VTK_QUADRATIC_TETRA) {
            if (cell_sizes[i] != 10) {
                throw std::runtime_error(
                    "Quadratic tetra cell " + std::to_string(i) +
                    ": expected 10 nodes, got " + std::to_string(cell_sizes[i]));
            }
            std::memcpy(&mesh.tet_nodes[tet_idx * 10],
                        cell_data[i].data(), 10 * sizeof(int));
            ++tet_idx;

        } else if (cell_types[i] == VTK_QUADRATIC_TRIANGLE) {
            if (cell_sizes[i] != 6) {
                throw std::runtime_error(
                    "Quadratic tri cell " + std::to_string(i) +
                    ": expected 6 nodes, got " + std::to_string(cell_sizes[i]));
            }
            std::memcpy(&mesh.tri_nodes[tri_idx * 6],
                        cell_data[i].data(), 6 * sizeof(int));
            ++tri_idx;
        }
    }

    return mesh;
}

// =============================================================================
//  print_mesh_info
// =============================================================================
void print_mesh_info(const FEMMesh& mesh)
{
    std::cout << "=== FEM Mesh Summary ===" << std::endl;
    std::cout << "  Nodes:                " << mesh.num_nodes << std::endl;
    std::cout << "  Quadratic tetrahedra: " << mesh.num_tets  << std::endl;
    std::cout << "  Quadratic triangles:  " << mesh.num_tris  << std::endl;
    std::cout << "  DOFs (3D vector):     " << mesh.num_nodes * 3 << std::endl;
    std::cout << std::endl;

    if (mesh.num_nodes > 0) {
        double xmin = mesh.node_coords[0], xmax = xmin;
        double ymin = mesh.node_coords[1], ymax = ymin;
        double zmin = mesh.node_coords[2], zmax = zmin;
        for (int i = 0; i < mesh.num_nodes; ++i) {
            double x = mesh.node_coords[i * 3 + 0];
            double y = mesh.node_coords[i * 3 + 1];
            double z = mesh.node_coords[i * 3 + 2];
            xmin = std::min(xmin, x); xmax = std::max(xmax, x);
            ymin = std::min(ymin, y); ymax = std::max(ymax, y);
            zmin = std::min(zmin, z); zmax = std::max(zmax, z);
        }
        std::cout << "  Bounding box:" << std::endl;
        std::cout << "    X: [" << xmin << ", " << xmax << "]" << std::endl;
        std::cout << "    Y: [" << ymin << ", " << ymax << "]" << std::endl;
        std::cout << "    Z: [" << zmin << ", " << zmax << "]" << std::endl;
        std::cout << std::endl;
    }

    if (mesh.num_tets > 0) {
        std::cout << "  First tet10: [";
        for (int j = 0; j < 10; ++j) {
            std::cout << mesh.tet_nodes[j];
            if (j < 9) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    if (mesh.num_tris > 0) {
        std::cout << "  First tri6:  [";
        for (int j = 0; j < 6; ++j) {
            std::cout << mesh.tri_nodes[j];
            if (j < 5) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
}
