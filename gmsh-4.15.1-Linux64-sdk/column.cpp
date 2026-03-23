#include <gmsh.h>

int main(int argc, char **argv)
{
    gmsh::initialize();
    gmsh::model::add("column");

    // 10x10x100m の直方体を作成（OpenCASCADE kernel を使用）
    // addBox(x, y, z, dx, dy, dz)
    gmsh::model::occ::addBox(0, 0, 0, 10, 10, 100);

    // 例：(5, 5, 50) に節点を固定したい
    int pt = gmsh::model::occ::addPoint(5, 5, 50);

    gmsh::model::occ::synchronize();

    // この点をボリューム 1 に埋め込む
    gmsh::model::mesh::embed(0, {pt}, 3, 1);
    //                       ↑dim=0(点)  ↑dim=3, tag=1(ボリューム)

    // メッシュサイズの設定（値を小さくすると要素が細かくなる）
    gmsh::option::setNumber("Mesh.CharacteristicLengthMax", 1.0);

    // 3Dメッシュ生成（四面体がデフォルト）
    gmsh::model::mesh::generate(3);

    // 二次要素に変換（10節点四面体）
    gmsh::model::mesh::setOrder(2);

    // 念のため：中間節点を直線上に配置（曲げない）
    gmsh::option::setNumber("Mesh.SecondOrderLinear", 1);

    // 保存
    gmsh::write("column.vtk");

    // FrontISTR変換用に .msh (v2.2 ASCII) も出力
    gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);
    gmsh::write("column.msh");

    gmsh::finalize();
    return 0;
}

/*実行コマンド
g++ -Iinclude column.cpp -Llib -lgmsh
export LD_LIBRARY_PATH=.../lib:$LD_LIBRARY_PATH
または
export LD_LIBRARY_PATH=/home/apple/elastodynamics-fem/gmsh-4.15.1-Linux64-sdk/lib:$LD_LIBRARY_PATH
./a.out
python3 gmsh2fistr.py column.msh
*/