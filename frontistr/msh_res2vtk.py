"""
FrontISTR .msh + .res → VTK 変換スクリプト（完全汎用・各成分の最大/最小トラッキング版）

使い方:
  python3 msh_res2vtk.py work/column_fistr.msh work/column_fistr.res.0.*
"""

import sys
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
            nodes[nid] = (float(parts[1]), float(parts[2]), float(parts[3]))
        elif section == "ELEMENT" and line:
            parts = line.split(",")
            eid = int(parts[0])
            conn = [int(p) for p in parts[1:]]
            elements.append((eid, conn))

        i += 1

    return nodes, elements


def read_res_generic(res_path):
    """ヘッダ情報を自動解析して全ての節点・要素データを読み込む"""
    with open(res_path, "r") as f:

        def next_line():
            for line in f:
                s = line.strip()
                if s:
                    return s
            return None

        total_time = 0.0
        line = next_line()

        # ヘッダ検索
        while line is not None:
            if line == "TOTALTIME":
                total_time = float(next_line())
            elif line == "*data":
                break
            line = next_line()

        line = next_line()
        if not line:
            return total_time, [], []

        # 節点数、要素数
        nn, ne = map(int, line.split())

        # 節点データ数、要素データ数
        n_ndata, n_cdata = map(int, next_line().split())

        # 節点データ定義の読み取り
        nodal_meta = []
        if n_ndata > 0:
            comps = list(map(int, next_line().split()))
            for c in comps:
                nodal_meta.append({"name": next_line(), "comp": c, "data": {}})

        # 要素データ定義の読み取り
        elem_meta = []
        if n_cdata > 0:
            comps = list(map(int, next_line().split()))
            for c in comps:
                elem_meta.append({"name": next_line(), "comp": c, "data": {}})

        # 節点データの読み取り
        if n_ndata > 0:
            total_comps = sum(m["comp"] for m in nodal_meta)
            for _ in range(nn):
                nid = int(next_line())
                vals = []
                while len(vals) < total_comps:
                    vals.extend(map(float, next_line().split()))

                v_idx = 0
                for m in nodal_meta:
                    c = m["comp"]
                    m["data"][nid] = vals[v_idx : v_idx + c]
                    v_idx += c

        # 要素データの読み取り
        if n_cdata > 0:
            total_comps = sum(m["comp"] for m in elem_meta)
            for _ in range(ne):
                eid = int(next_line())
                vals = []
                while len(vals) < total_comps:
                    vals.extend(map(float, next_line().split()))

                v_idx = 0
                for m in elem_meta:
                    c = m["comp"]
                    m["data"][eid] = vals[v_idx : v_idx + c]
                    v_idx += c

    return total_time, nodal_meta, elem_meta


def write_vtk_field(f, meta, item_ids):
    """VTKのデータ型を判別して書き出し"""
    name = meta["name"].replace(" ", "_")
    comp = meta["comp"]
    data = meta["data"]

    if comp == 1:
        f.write(f"SCALARS {name} double 1\nLOOKUP_TABLE default\n")
        for iid in item_ids:
            f.write(f"{data.get(iid, [0.0])[0]:.10e}\n")
    elif comp == 3:
        f.write(f"VECTORS {name} double\n")
        for iid in item_ids:
            v = data.get(iid, [0.0] * 3)
            f.write(f"{v[0]:.10e} {v[1]:.10e} {v[2]:.10e}\n")
    elif comp == 6:
        f.write(f"TENSORS {name} double\n")
        for iid in item_ids:
            v = data.get(iid, [0.0] * 6)
            f.write(f"{v[0]:.10e} {v[3]:.10e} {v[5]:.10e}\n")
            f.write(f"{v[3]:.10e} {v[1]:.10e} {v[4]:.10e}\n")
            f.write(f"{v[5]:.10e} {v[4]:.10e} {v[2]:.10e}\n")
    else:
        f.write(f"SCALARS {name} double {comp}\nLOOKUP_TABLE default\n")
        for iid in item_ids:
            v = data.get(iid, [0.0] * comp)
            f.write(" ".join(f"{x:.10e}" for x in v) + "\n")


def write_vtk_generic(vtk_path, nodes, elements, nodal_meta, elem_meta, total_time):
    sorted_nids = sorted(nodes.keys())
    nid_to_idx = {nid: idx for idx, nid in enumerate(sorted_nids)}
    n_nodes = len(sorted_nids)
    n_elems = len(elements)

    with open(vtk_path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("FrontISTR result\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")

        f.write("\nFIELD FieldData 1\n")
        f.write("TOTALTIME 1 1 double\n")
        f.write(f"{total_time:.10e}\n")

        f.write(f"\nPOINTS {n_nodes} double\n")
        for nid in sorted_nids:
            x, y, z = nodes[nid]
            f.write(f"{x:.10e} {y:.10e} {z:.10e}\n")

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
            f.write(
                "24\n"
                if nn == 10
                else (
                    "10\n"
                    if nn == 4
                    else "12\n" if nn == 8 else "25\n" if nn == 20 else "10\n"
                )
            )

        if nodal_meta:
            f.write(f"\nPOINT_DATA {n_nodes}\n")
            f.write("SCALARS NODE_ID int 1\nLOOKUP_TABLE default\n")
            for nid in sorted_nids:
                f.write(f"{nid}\n")
            for m in nodal_meta:
                write_vtk_field(f, m, sorted_nids)

        if elem_meta:
            f.write(f"\nCELL_DATA {n_elems}\n")
            f.write("SCALARS ELEM_ID int 1\nLOOKUP_TABLE default\n")
            for eid, _ in elements:
                f.write(f"{eid}\n")
            for m in elem_meta:
                write_vtk_field(f, m, [e[0] for e in elements])


def extract_step_number(filepath):
    m = re.search(r"\.(\d+)$", filepath)
    return int(m.group(1)) if m else 0


# --- メイン ---
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("使い方: python3 msh_res2vtk.py <mesh.msh> <res files...>")
        sys.exit(1)

    msh_path = sys.argv[1]
    res_args = sys.argv[2:]

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

    # 6成分テンソル（応力・ひずみ等）の各成分ごとの最大・最小トラッカー
    comp_trackers = {}
    comp_names = ["XX", "YY", "ZZ", "XY", "YZ", "ZX"]

    for res_path in res_files:
        total_time, nodal_meta, elem_meta = read_res_generic(res_path)

        vtk_path = res_path + ".vtk"
        write_vtk_generic(vtk_path, nodes, elements, nodal_meta, elem_meta, total_time)

        step = extract_step_number(res_path)
        print(f"  step {step:4d}  t={total_time:8.4f}  → {vtk_path}")

        # 出力された全ての変数から、成分数が6のもの（応力・ひずみ）を探して各成分を追跡
        for m in nodal_meta + elem_meta:
            if m["comp"] == 6:
                name = m["name"]
                if name not in comp_trackers:
                    comp_trackers[name] = {
                        c: {
                            "max": -float("inf"),
                            "max_info": None,
                            "min": float("inf"),
                            "min_info": None,
                        }
                        for c in comp_names
                    }

                for iid, s in m["data"].items():
                    coords = nodes.get(iid, None)  # 節点なら座標を取得
                    for i, c_name in enumerate(comp_names):
                        val = s[i]
                        # 最大値の更新（引張側）
                        if val > comp_trackers[name][c_name]["max"]:
                            comp_trackers[name][c_name]["max"] = val
                            comp_trackers[name][c_name]["max_info"] = (
                                step,
                                total_time,
                                iid,
                                coords,
                            )
                        # 最小値の更新（圧縮側）
                        if val < comp_trackers[name][c_name]["min"]:
                            comp_trackers[name][c_name]["min"] = val
                            comp_trackers[name][c_name]["min_info"] = (
                                step,
                                total_time,
                                iid,
                                coords,
                            )

    print(f"\n完了: {len(res_files)} ファイル変換")

    # 最大・最小値情報の出力
    for name, tracker in comp_trackers.items():
        print("\n" + "=" * 70)
        print(f"【 {name} の各成分の最大・最小値 (全ステップ・全データ中) 】")
        print("=" * 70)
        for c_name in comp_names:
            max_val = tracker[c_name]["max"]
            max_info = tracker[c_name]["max_info"]
            min_val = tracker[c_name]["min"]
            min_info = tracker[c_name]["min_info"]

            print(f"■ 成分 {c_name}")
            if max_info:
                step, t, iid, coords = max_info
                coord_str = (
                    f"X={coords[0]:.4f}, Y={coords[1]:.4f}, Z={coords[2]:.4f}"
                    if coords
                    else "N/A"
                )
                print(
                    f"  [最大(引張)] {max_val: 13.6e} (Step: {step:3d}, ID: {iid}, 座標: {coord_str})"
                )
            if min_info:
                step, t, iid, coords = min_info
                coord_str = (
                    f"X={coords[0]:.4f}, Y={coords[1]:.4f}, Z={coords[2]:.4f}"
                    if coords
                    else "N/A"
                )
                print(
                    f"  [最小(圧縮)] {min_val: 13.6e} (Step: {step:3d}, ID: {iid}, 座標: {coord_str})"
                )
            print("-" * 70)
