/*実行コマンド
g++ -Iinclude column.cpp -Llib -lgmsh
export LD_LIBRARY_PATH=$(pwd)/lib:$LD_LIBRARY_PATH
./a.out
*/
#include <gmsh.h>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <stdexcept>

static std::size_t count_mesh_nodes()
{
    std::vector<std::size_t> nodeTags;
    std::vector<double> coord;
    std::vector<double> parametricCoord;
    gmsh::model::mesh::getNodes(nodeTags, coord, parametricCoord,
                                -1, -1, false, false);
    return nodeTags.size();
}

int main(int argc, char **argv)
{
    const double mesh_size = 2.0; // メッシュサイズの目安
    int parallel_parts = 1;       // パーティション数（MPI並列数に合わせる）
    if (argc >= 2)
        parallel_parts = std::atoi(argv[1]);
    if (parallel_parts < 1)
        throw std::runtime_error("parallel_parts must be >= 1");

    gmsh::initialize();
    gmsh::model::add("column");

    // 10x10x100m の直方体を作成（OpenCASCADE kernel を使用）
    // 戻り値としてボリュームのタグを受け取る
    int volTag = gmsh::model::occ::addBox(0, 0, 0, 10, 10, 100);

    // 固定したい節点を追加
    int ptTag = gmsh::model::occ::addPoint(5, 5, 50);
    int ptTag2 = gmsh::model::occ::addPoint(5, 5, 100);

    // ボリュームと点の交差を取り、正しいトポロジー関係を構築する
    std::vector<std::pair<int, int>> outDimTags;
    std::vector<std::vector<std::pair<int, int>>> outDimTagsMap;
    // gmsh::model::occ::fragment({{3, volTag}}, {{0, ptTag}}, outDimTags, outDimTagsMap);
    gmsh::model::occ::fragment({{3, volTag}}, {{0, ptTag}, {0, ptTag2}}, outDimTags, outDimTagsMap);

    // fragment 演算の後に synchronize を実行
    gmsh::model::occ::synchronize();

    // メッシュサイズの設定
    gmsh::option::setNumber("Mesh.CharacteristicLengthMax", mesh_size);

    // HXTの並列Delaunayは実行ごとに点挿入順が揺れ、節点数まで変わり得る。
    // 強スケーリング比較では同一メッシュが必須なので、デフォルトは1スレッドに固定する。
    int mesh_threads = 1;
    if (const char *env_threads = std::getenv("GMSH_MESH_THREADS"))
        mesh_threads = std::max(1, std::atoi(env_threads));
    gmsh::option::setNumber("General.NumThreads", mesh_threads);
    gmsh::option::setNumber("Mesh.MaxNumThreads1D", mesh_threads);
    gmsh::option::setNumber("Mesh.MaxNumThreads2D", mesh_threads);
    gmsh::option::setNumber("Mesh.MaxNumThreads3D", mesh_threads);
    gmsh::option::setNumber("Mesh.Algorithm3D", 10); // HXT

    // 3Dメッシュ生成
    gmsh::model::mesh::generate(3);

    // 中間節点を直線上に配置（曲げない）
    gmsh::option::setNumber("Mesh.SecondOrderLinear", 1);

    // 二次要素に変換（10節点四面体）
    gmsh::model::mesh::setOrder(2);

    // fragment や高次化で万一重複節点ができた場合は、partition前に必ず潰す。
    gmsh::model::mesh::removeDuplicateNodes();
    const std::size_t base_num_nodes = count_mesh_nodes();
    std::cout << "base mesh nodes: " << base_num_nodes << std::endl;

    // 保存
    std::cout << "start writing..." << std::endl;
    gmsh::write("column.vtk");

    // FrontISTR変換用に .msh (v2.2 ASCII) も出力
    gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);
    gmsh::write("column.msh");

    // ===== パーティション追加 =====
    // K-way: 3分割以上ではRecursiveより均等になりやすい
    gmsh::option::setNumber("Mesh.MetisAlgorithm", 2);

    // 通信量最小化 (edge-cut=1 より実際のMPI通信に近い)
    gmsh::option::setNumber("Mesh.MetisObjective", 2);

    // 負荷不均衡の許容度 (小さいほど均等, デフォルト30)
    // 1にすると完全均等を目指すが遅くなる場合がある
    gmsh::option::setNumber("Mesh.MetisMaxLoadImbalance", 2);

    // パーティション間の接続数を最小化
    gmsh::option::setNumber("Mesh.MetisMinConn", 1);

    // パーティション境界のトポロジーを作る ($PartitionedEntities用)
    gmsh::option::setNumber("Mesh.PartitionCreateTopology", 1);

    // ゴーストセルを作る ($GhostElements用)
    gmsh::option::setNumber("Mesh.PartitionCreateGhostCells", 1);

    if (parallel_parts > 1)
    {
        gmsh::model::mesh::partition(parallel_parts);
        // PartitionCreateTopology は境界上のpartitioned entityを作る。
        // Gmshの設定やバージョンによって重複節点が残ると、単体メッシュと
        // partitionメッシュが別問題になるので、出力前に再度潰して検査する。
        gmsh::model::mesh::removeDuplicateNodes();
    }

    const std::size_t partitioned_num_nodes = count_mesh_nodes();
    std::cout << "partitioned mesh nodes: " << partitioned_num_nodes << std::endl;
    if (partitioned_num_nodes != base_num_nodes)
    {
        std::cerr << "ERROR: partitioning changed the number of mesh nodes: "
                  << base_num_nodes << " -> " << partitioned_num_nodes << std::endl;
        gmsh::finalize();
        return 2;
    }

    // MSH 4.1 で出力
    std::cout << "start writing partitioned mesh..." << std::endl;
    gmsh::option::setNumber("Mesh.MshFileVersion", 4.1);
    gmsh::write("column_4.msh");

    gmsh::finalize();
    return 0;
}
