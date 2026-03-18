#ifndef VTK_READER_HPP
#define VTK_READER_HPP

#include <string>
#include <vector>

// =============================================================================
//  VTK Unstructured Grid Reader for Quadratic Tetrahedral FEM
// =============================================================================
//
//  std::vector で管理し、OpenACC には .data() で生ポインタを渡す。
//  スコープを抜ければ自動で解放されるため free 不要。
//
//  二次四面体の節点順序 (VTK convention):
//
//        3
//       /|\
//      / | \           頂点: 0, 1, 2, 3
//     9  |  8          辺中間節点:
//    /   7   \           4 = 辺(0,1), 5 = 辺(1,2), 6 = 辺(0,2)
//   /    |    \          7 = 辺(0,3), 8 = 辺(1,3), 9 = 辺(2,3)
//  0-----|-----2
//   \    |    /
//    4   |   5
//     \  |  /
//      \ | /
//        1
//
//  二次三角形の節点順序 (VTK convention):
//
//    2
//    |\            頂点: 0, 1, 2
//    | \           辺中間節点:
//    5  4            3 = 辺(0,1), 4 = 辺(1,2), 5 = 辺(0,2)
//    |   \
//    0--3--1
//
// =============================================================================

struct FEMMesh {
    // --- 節点データ ---
    int num_nodes = 0;
    std::vector<double> node_coords;  // [num_nodes * 3]  連続配置: x0,y0,z0,x1,y1,z1,...

    // --- 二次四面体要素 (体積要素) ---
    int num_tets = 0;
    std::vector<int> tet_nodes;       // [num_tets * 10]  連続配置: 要素0の10節点, 要素1の10節点,...

    // --- 二次三角形要素 (境界面要素) ---
    int num_tris = 0;
    std::vector<int> tri_nodes;       // [num_tris * 6]   連続配置: 要素0の6節点, 要素1の6節点,...

    // OpenACC 用の生ポインタ取得
    double* coords_ptr()       { return node_coords.data(); }
    int*    tet_ptr()          { return tet_nodes.data(); }
    int*    tri_ptr()          { return tri_nodes.data(); }
    const double* coords_ptr() const { return node_coords.data(); }
    const int*    tet_ptr()    const { return tet_nodes.data(); }
    const int*    tri_ptr()    const { return tri_nodes.data(); }
};

// VTKファイルを読み込み、FEMMesh を返す。
// 失敗時は std::runtime_error を投げる。
FEMMesh read_vtk(const std::string& filepath);

// FEMMesh の概要を標準出力に表示する（デバッグ用）。
void print_mesh_info(const FEMMesh& mesh);

#endif // VTK_READER_HPP
