"""
FrontISTR .msh + .res → VTK 変換スクリプト（変位のみ）

使い方:
  python3 msh_res2vtk.py work/column_fistr.msh work/column_fistr.res.0.*
"""

import sys
import os
import glob
import re


def read_msh(msh_path):
    """HEC-MW メッシュファイルを読む"""
    with open(msh_path, "r") as f:
        content = f.read().replace("\r\n", "\n")
    lines = content.split("\n")

    nodes = {}
    elements = []
    section = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("!NODE"):
            section = "NODE"
            i += 1
            continue
        elif line.startswith("!ELEMENT"):
            section = "ELEMENT"
            i += 1
            continue
        elif line.startswith("!"):
            section = None
            i += 1
            continue

        if section == "NODE" and line:
            parts = line.split(",")
            nid = int(parts[0])
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            nodes[nid] = (x, y, z)
        elif section == "ELEMENT" and line:
            parts = line.split(",")
            eid = int(parts[0])
            conn = [int(p) for p in parts[1:]]
            elements.append((eid, conn))

        i += 1

    return nodes, elements


def read_res_disp(res_path):
    """FrontISTR 結果ファイルから変位と時刻を読む"""
    with open(res_path, "r") as f:
        lines = f.readlines()

    # ヘッダからTOTALTIMEを取得
    total_time = 0.0
    i = 0
    while i < len(lines):
        if lines[i].strip() == "TOTALTIME":
            total_time = float(lines[i + 1].strip())
            i += 2
            continue
        if lines[i].strip() == "*data":
            i += 1
            break
        i += 1

    # n_nodes n_elems
    parts = lines[i].split()
    nn = int(parts[0])
    i += 1

    # n_ndata n_cdata
    i += 1

    # component counts
    parts = lines[i].split()
    n_items = int(parts[0])
    comp_counts = [int(p) for p in parts[1:]]
    total_comps = sum(comp_counts)
    i += 1

    # data names
    for _ in range(n_items):
        i += 1

    # 各節点のデータを読む
    disp = {}
    for _ in range(nn):
        nid = int(lines[i].strip())
        i += 1
        all_vals = []
        while len(all_vals) < total_comps:
            all_vals.extend(float(v) for v in lines[i].split())
            i += 1
        disp[nid] = (all_vals[0], all_vals[1], all_vals[2])

    return total_time, disp


def write_vtk(vtk_path, nodes, elements, disp, total_time):
    """VTK Legacy 形式で書き出す（FIELD DATA に時刻を埋め込み）"""
    sorted_nids = sorted(nodes.keys())
    nid_to_idx = {nid: idx for idx, nid in enumerate(sorted_nids)}
    n_nodes = len(sorted_nids)
    n_elems = len(elements)

    with open(vtk_path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("FrontISTR result\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")

        # --- 時刻を FIELD DATA として埋め込み ---
        f.write("\nFIELD FieldData 1\n")
        f.write(f"TOTALTIME 1 1 double\n")
        f.write(f"{total_time:.10e}\n")

        f.write(f"\nPOINTS {n_nodes} double\n")
        for nid in sorted_nids:
            x, y, z = nodes[nid]
            f.write(f"{x:.10e} {y:.10e} {z:.10e}\n")

        # 要素 (Tet10: FrontISTR→VTK で中間節点[8],[9]を入れ替え)
        total_ints = sum(len(conn) + 1 for _, conn in elements)
        f.write(f"\nCELLS {n_elems} {total_ints}\n")
        for eid, conn in elements:
            idx_conn = [nid_to_idx[n] for n in conn]
            if len(idx_conn) == 10:
                idx_conn[8], idx_conn[9] = idx_conn[9], idx_conn[8]
            f.write(f"{len(conn)} " + " ".join(str(c) for c in idx_conn) + "\n")

        f.write(f"\nCELL_TYPES {n_elems}\n")
        for eid, conn in elements:
            nn = len(conn)
            if nn == 10:
                f.write("24\n")
            elif nn == 4:
                f.write("10\n")
            elif nn == 8:
                f.write("12\n")
            elif nn == 20:
                f.write("25\n")
            else:
                f.write("10\n")

        # --- 節点ID → VTKインデックス対応表を埋め込み ---
        f.write(f"\nPOINT_DATA {n_nodes}\n")
        f.write("SCALARS NODE_ID int 1\n")
        f.write("LOOKUP_TABLE default\n")
        for nid in sorted_nids:
            f.write(f"{nid}\n")

        f.write("VECTORS DISPLACEMENT double\n")
        for nid in sorted_nids:
            dx, dy, dz = disp.get(nid, (0, 0, 0))
            f.write(f"{dx:.10e} {dy:.10e} {dz:.10e}\n")


def extract_step_number(filepath):
    """ファイル名から数値ステップを抽出 (ソート用)"""
    m = re.search(r'\.(\d+)$', filepath)
    return int(m.group(1)) if m else 0


# --- メイン ---
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("使い方: python3 msh_res2vtk.py <mesh.msh> <res files...>")
        sys.exit(1)

    msh_path = sys.argv[1]
    res_args = sys.argv[2:]

    # 結果ファイルのglob展開 + .vtk/.csv除外
    res_files = []
    for arg in res_args:
        expanded = glob.glob(arg)
        if expanded:
            res_files.extend(expanded)
        else:
            res_files.append(arg)
    res_files = [f for f in res_files if not f.endswith((".vtk", ".csv"))]
    res_files.sort(key=extract_step_number)

    print(f"メッシュ読み込み: {msh_path}")
    nodes, elements = read_msh(msh_path)
    print(f"  節点数: {len(nodes)}, 要素数: {len(elements)}")

    for res_path in res_files:
        total_time, disp = read_res_disp(res_path)

        vtk_path = res_path + ".vtk"
        write_vtk(vtk_path, nodes, elements, disp, total_time)

        step = extract_step_number(res_path)
        print(f"  step {step:4d}  t={total_time:8.4f}  → {vtk_path}")

    print(f"\n完了: {len(res_files)} ファイル変換")
