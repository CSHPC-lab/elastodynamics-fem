"""
VTK ファイル群から指定節点の変位を時系列CSVに出力

使い方:
  python3 vtk2csv.py frontistr/work/column_fistr.res.0.*.vtk --node 9 --csv frontistr/work/0.5.csv
  python3 vtk2csv.py results/0.5/result_*.vtk --node 9 --csv results/0.5/0.5.csv
"""

import sys
import os
import glob
import re


def read_vtk(vtk_path):
    """VTK Legacy ファイルから時刻・節点ID・変位を読む"""
    with open(vtk_path, "r") as f:
        lines = f.readlines()

    total_time = 0.0
    node_ids = []
    displacements = []
    n_nodes = 0
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # FIELD DATA から TOTALTIME を取得
        if line.startswith("FIELD"):
            i += 1
            while i < len(lines):
                field_line = lines[i].strip()
                if field_line.startswith("TOTALTIME"):
                    i += 1
                    total_time = float(lines[i].strip())
                    i += 1
                    break
                i += 1
            continue

        # POINT_DATA から節点数を取得
        if line.startswith("POINT_DATA"):
            n_nodes = int(line.split()[1])
            i += 1
            continue

        # NODE_ID スカラーを読む
        if line.startswith("SCALARS NODE_ID"):
            i += 1  # LOOKUP_TABLE default
            i += 1
            for _ in range(n_nodes):
                node_ids.append(int(lines[i].strip()))
                i += 1
            continue

        # DISPLACEMENT ベクトルを読む
        if line.startswith("VECTORS DISPLACEMENT"):
            i += 1
            for _ in range(n_nodes):
                vals = lines[i].split()
                displacements.append((float(vals[0]), float(vals[1]), float(vals[2])))
                i += 1
            continue

        i += 1

    # NODE_ID → 変位の辞書を作成
    disp = {}
    if node_ids and displacements:
        for nid, d in zip(node_ids, displacements):
            disp[nid] = d
    else:
        # NODE_ID フィールドが無い場合は 0-based インデックスをキーにする
        for idx, d in enumerate(displacements):
            disp[idx] = d

    return total_time, disp


def extract_step_number(filepath):
    """ファイル名から数値ステップを抽出 (ソート用)
    対応パターン:
      column_fistr.res.0.42.vtk → 42  (ドット区切り、.vtk除去後)
      result_42.vtk             → 42  (アンダースコア区切り)
      result42.vtk              → 42  (末尾数字)
    """
    basename = os.path.basename(filepath)

    # .vtk 拡張子を除去してから判定
    if basename.endswith(".vtk"):
        basename = basename[:-4]

    # パターン1: .数字 で終わる (column_fistr.res.0.42)
    m = re.search(r'\.(\d+)$', basename)
    if m:
        return int(m.group(1))

    # パターン2: _数字 で終わる (result_42)
    m = re.search(r'_(\d+)$', basename)
    if m:
        return int(m.group(1))

    # パターン3: 末尾の数字 (result42)
    m = re.search(r'(\d+)$', basename)
    if m:
        return int(m.group(1))

    return 0


# --- メイン ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方: python3 vtk2csv.py <vtk files...> [--node N] [--csv output.csv]")
        sys.exit(1)

    track_node = 9  # デフォルト: 節点9
    csv_path = None
    vtk_args = []

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--node":
            track_node = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--csv":
            csv_path = sys.argv[i + 1]
            i += 2
        else:
            vtk_args.append(sys.argv[i])
            i += 1

    # VTKファイルのglob展開
    vtk_files = []
    for arg in vtk_args:
        expanded = glob.glob(arg)
        if expanded:
            vtk_files.extend(expanded)
        else:
            vtk_files.append(arg)
    vtk_files = [f for f in vtk_files if f.endswith(".vtk")]
    vtk_files.sort(key=extract_step_number)

    if not vtk_files:
        print("エラー: VTKファイルが見つかりません")
        sys.exit(1)

    # デフォルトCSVファイル名
    if csv_path is None:
        csv_path = os.path.join(
            os.path.dirname(vtk_files[0]) or ".", f"node{track_node}_disp.csv"
        )

    print(f"追跡節点: {track_node}")
    print(f"CSV出力先: {csv_path}")
    print(f"VTKファイル数: {len(vtk_files)}")

    time_series = []

    for vtk_path in vtk_files:
        total_time, disp = read_vtk(vtk_path)
        dx, dy, dz = disp.get(track_node, (0, 0, 0))
        mag = (dx**2 + dy**2 + dz**2) ** 0.5
        time_series.append((total_time, dx, dy, dz, mag))

        step = extract_step_number(vtk_path)
        print(f"  step {step:4d}  t={total_time:8.4f}  "
              f"disp=({dx:+.6e}, {dy:+.6e}, {dz:+.6e})  |u|={mag:.6e}")

    # CSV出力
    with open(csv_path, "w") as f:
        f.write("time,disp_x,disp_y,disp_z,disp_mag\n")
        for t, dx, dy, dz, mag in time_series:
            f.write(f"{t:.10e},{dx:.10e},{dy:.10e},{dz:.10e},{mag:.10e}\n")

    print(f"\n完了: {len(vtk_files)} ファイル読み込み → {csv_path}")