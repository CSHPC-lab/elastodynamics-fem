#include <gmsh.h>

int main(int argc, char **argv)
{
    gmsh::initialize();
    gmsh::model::add("column");

    // 10x10x100m の直方体を作成（OpenCASCADE kernel を使用）
    // addBox(x, y, z, dx, dy, dz)
    gmsh::model::occ::addBox(0, 0, 0, 10, 10, 100);
    gmsh::model::occ::synchronize();

    // メッシュサイズの設定（値を小さくすると要素が細かくなる）
    gmsh::option::setNumber("Mesh.CharacteristicLengthMax", 5.0);

    // 3Dメッシュ生成（四面体がデフォルト）
    gmsh::model::mesh::generate(3);

    // 二次要素に変換（10節点四面体）
    gmsh::model::mesh::setOrder(2);

    // 念のため：中間節点を直線上に配置（曲げない）
    gmsh::option::setNumber("Mesh.SecondOrderLinear", 1);

    // 保存
    gmsh::write("column.vtk");

    gmsh::finalize();
    return 0;
}

/*実行コマンド
g++ -Iinclude column.cpp -Llib -lgmsh
export LD_LIBRARY_PATH=.../lib:$LD_LIBRARY_PATH
./a.out
*/