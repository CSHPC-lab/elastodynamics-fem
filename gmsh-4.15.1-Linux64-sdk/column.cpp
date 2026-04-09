/*実行コマンド
g++ -Iinclude column.cpp -Llib -lgmsh
export LD_LIBRARY_PATH=$(pwd)/lib:$LD_LIBRARY_PATH
./a.out
*/
#include <gmsh.h>
#include <vector>

int main(int argc, char **argv)
{
    gmsh::initialize();
    gmsh::model::add("column");

    // 10x10x100m の直方体を作成（OpenCASCADE kernel を使用）
    // 戻り値としてボリュームのタグを受け取る
    int volTag = gmsh::model::occ::addBox(0, 0, 0, 10, 10, 100);

    // 固定したい節点を追加
    int ptTag = gmsh::model::occ::addPoint(5, 5, 50);

    // ボリュームと点の交差を取り、正しいトポロジー関係を構築する
    std::vector<std::pair<int, int>> outDimTags;
    std::vector<std::vector<std::pair<int, int>>> outDimTagsMap;
    gmsh::model::occ::fragment({{3, volTag}}, {{0, ptTag}}, outDimTags, outDimTagsMap);

    // fragment 演算の後に synchronize を実行
    gmsh::model::occ::synchronize();

    // メッシュサイズの設定
    gmsh::option::setNumber("Mesh.CharacteristicLengthMax", 5.0);

    // より高品質な3Dメッシュを求める場合は、HXTアルゴリズム(10)やFrontal(4)の指定が有効です（任意）
    // gmsh::option::setNumber("Mesh.Algorithm3D", 10);

    // 3Dメッシュ生成
    gmsh::model::mesh::generate(3);

    // 二次要素に変換（10節点四面体）
    gmsh::model::mesh::setOrder(2);

    // 中間節点を直線上に配置（曲げない）
    gmsh::option::setNumber("Mesh.SecondOrderLinear", 1);

    // 保存
    gmsh::write("column.vtk");

    // FrontISTR変換用に .msh (v2.2 ASCII) も出力
    gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);
    gmsh::write("column.msh");

    // ===== パーティション追加 =====
    int nParts = 16; // MPI並列数に合わせる

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

    gmsh::model::mesh::partition(nParts);

    // MSH 4.1 で出力
    gmsh::option::setNumber("Mesh.MshFileVersion", 4.1);
    gmsh::write("column_4.msh");

    gmsh::finalize();
    return 0;
}