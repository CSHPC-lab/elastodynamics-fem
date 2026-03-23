#include <gmsh.h>
#include <vector>

int main(int argc, char **argv)
{
    gmsh::initialize();
    gmsh::model::add("pierced_sphere");

    // --- 母体：半径 20 の球 ---
    int sphere = gmsh::model::occ::addSphere(0, 0, 0, 20);

    // --- 3軸方向に貫通する円筒（半径 8, 長さ 60） ---
    int cylX = gmsh::model::occ::addCylinder(-30, 0, 0, 60, 0, 0, 8);
    int cylY = gmsh::model::occ::addCylinder(0, -30, 0, 0, 60, 0, 8);
    int cylZ = gmsh::model::occ::addCylinder(0, 0, -30, 0, 0, 60, 8);

    // --- ブーリアン演算：球から3本の円筒をくり抜く ---
    std::vector<std::pair<int, int>> outDimTags;
    std::vector<std::vector<std::pair<int, int>>> outDimTagsMap;
    gmsh::model::occ::cut(
        {{3, sphere}},                     // 対象
        {{3, cylX}, {3, cylY}, {3, cylZ}}, // 工具
        outDimTags, outDimTagsMap);

    gmsh::model::occ::synchronize();

    // --- メッシュ設定 ---
    gmsh::option::setNumber("Mesh.CharacteristicLengthMax", 2.0);
    gmsh::option::setNumber("Mesh.CharacteristicLengthMin", 0.5);

    // 3Dメッシュ生成
    gmsh::model::mesh::generate(3);

    // 二次要素（10節点四面体）
    gmsh::model::mesh::setOrder(2);
    gmsh::option::setNumber("Mesh.SecondOrderLinear", 1);

    gmsh::write("pierced_sphere.vtk");

    gmsh::finalize();
    return 0;
}