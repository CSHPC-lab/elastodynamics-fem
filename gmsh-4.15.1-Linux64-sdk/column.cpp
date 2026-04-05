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
    int volTag = gmsh::model::occ::addBox(0, 0, 0, 10, 1, 1);

    // 固定したい節点を追加
    int ptTag = gmsh::model::occ::addPoint(10, 0.5, 0.5);

    // ボリュームと点の交差を取り、正しいトポロジー関係を構築する
    std::vector<std::pair<int, int>> outDimTags;
    std::vector<std::vector<std::pair<int, int>>> outDimTagsMap;
    gmsh::model::occ::fragment({{3, volTag}}, {{0, ptTag}}, outDimTags, outDimTagsMap);

    // fragment 演算の後に synchronize を実行
    gmsh::model::occ::synchronize();

    // 削除: gmsh::model::mesh::embed(0, {pt}, 3, 1); は不要になります

    // メッシュサイズの設定
    gmsh::option::setNumber("Mesh.CharacteristicLengthMax", 0.4);

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

    gmsh::finalize();
    return 0;
}