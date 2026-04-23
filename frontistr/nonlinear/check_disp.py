#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FrontISTR 動的解析の z=0 節点における変位を検証するスクリプト。

想定:
    - メッシュファイル: <basename>.msh (HEC-MW 形式、!NODE セクションを含む)
        * スクリプトと同階層に配置
    - 結果ファイル    : <results_dir>/<basename>.res.<partition>.<step>
        * 1 番目の数字がパーティション、2 番目の数字が時間ステップ
    - 時刻は .res 内の TOTALTIME を使用 (ファイル名からは算出しない)
    - 期待値          : z=0 の節点で Dx=0, Dy = A*sin(omega*t), Dz=0

各ステップについて、そのステップに属するすべてのパーティションの .res を
読み込み、変位を結合したうえで z=0 節点の網羅チェックを行います。
どのパーティションにも現れない z=0 節点があれば警告します。

Usage:
    python3 check_disp.py
    python3 check_disp.py --amplitude 0.1 --omega 2.0 -v
    python3 check_disp.py --partition 0        # 特定パーティションのみ

標準ライブラリのみ使用 (仮想環境不要)。
"""

import argparse
import math
import os
import re
import sys


# ---------------------------------------------------------------------------
# ファイル読み込み
# ---------------------------------------------------------------------------
def parse_mesh(mesh_path):
    """HEC-MW .msh ファイルの !NODE セクションを読み、{id: (x, y, z)} を返す。"""
    nodes = {}
    in_node_section = False
    with open(mesh_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("!"):
                header = stripped.split(",")[0].strip().upper()
                in_node_section = header == "!NODE"
                continue
            if in_node_section:
                parts = [p.strip() for p in stripped.split(",")]
                if len(parts) >= 4:
                    try:
                        nid = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                        nodes[nid] = (x, y, z)
                    except ValueError:
                        pass
    return nodes


def parse_result(res_path):
    """
    FrontISTR の .res.<partition>.<step> を読む。

    Returns:
        {
            'total_time': float,               # TOTALTIME (必須)
            'displacement': {id: (dx, dy, dz)},
            'node_names': [str, ...],
        }
    """
    with open(res_path, "r") as f:
        lines = [ln.rstrip("\n") for ln in f]

    # ----- TOTALTIME を探す -----
    total_time = None
    for i, ln in enumerate(lines):
        if ln.strip().upper() == "TOTALTIME":
            for j in range(i + 1, len(lines)):
                if lines[j].strip():
                    try:
                        total_time = float(lines[j].split()[0])
                    except ValueError:
                        pass
                    break
            break
    if total_time is None:
        raise ValueError("TOTALTIME が見つかりません: {0}".format(res_path))

    # ----- *data の位置を探す -----
    data_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith("*data"):
            data_idx = i + 1
            break
    if data_idx is None:
        raise ValueError("'*data' セクションが見つかりません: {0}".format(res_path))

    def next_nonblank(idx):
        while idx < len(lines) and not lines[idx].strip():
            idx += 1
        return idx

    # 1 行目: n_node n_elem (このパーティションの節点数・要素数)
    data_idx = next_nonblank(data_idx)
    n_node = int(lines[data_idx].split()[0])
    data_idx += 1

    # 2 行目: n_node_results n_elem_results
    data_idx = next_nonblank(data_idx)
    parts = lines[data_idx].split()
    n_nres = int(parts[0])
    n_eres = int(parts[1])
    data_idx += 1

    # 成分数 (n_nres + n_eres 個の整数)
    comp_counts = []
    while len(comp_counts) < n_nres + n_eres:
        data_idx = next_nonblank(data_idx)
        comp_counts.extend(int(x) for x in lines[data_idx].split())
        data_idx += 1
    comp_counts = comp_counts[: n_nres + n_eres]
    node_comps = comp_counts[:n_nres]

    # 結果名 (node -> elem の順に 1 行ずつ)
    node_names = []
    for _ in range(n_nres):
        data_idx = next_nonblank(data_idx)
        node_names.append(lines[data_idx].strip())
        data_idx += 1
    for _ in range(n_eres):
        data_idx = next_nonblank(data_idx)
        data_idx += 1  # 要素結果名は読み飛ばす

    # ----- DISPLACEMENT の位置 -----
    if "DISPLACEMENT" not in node_names:
        raise ValueError(
            "'DISPLACEMENT' が節点結果に含まれていません: {0}".format(node_names)
        )
    disp_pos = node_names.index("DISPLACEMENT")
    offset = sum(node_comps[:disp_pos])
    disp_n = node_comps[disp_pos]
    total_per_node = sum(node_comps)

    # ----- 節点データをトークン列として読む -----
    tokens = []
    for ln in lines[data_idx:]:
        tokens.extend(ln.split())

    stride = 1 + total_per_node
    displacements = {}
    for i_node in range(n_node):
        base = i_node * stride
        if base + stride > len(tokens):
            break
        nid = int(tokens[base])
        vals = tokens[base + 1 : base + 1 + total_per_node]
        d = [float(vals[offset + k]) for k in range(disp_n)]
        while len(d) < 3:
            d.append(0.0)
        displacements[nid] = (d[0], d[1], d[2])

    return {
        "total_time": total_time,
        "displacement": displacements,
        "node_names": node_names,
    }


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="z=0 の全節点で変位が (0, A*sin(omega*t), 0) になっているかを検証します。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mesh",
        default="column_fistr.msh",
        help="メッシュファイル名 (スクリプトと同階層)",
    )
    parser.add_argument(
        "--results-dir", default="results", help="結果ファイルを置いたディレクトリ名"
    )
    parser.add_argument(
        "--basename",
        default="column_fistr",
        help="結果ファイルのベース名 (<basename>.res.<part>.<step>)",
    )
    parser.add_argument(
        "--partition",
        type=int,
        default=None,
        help="特定のパーティションのみ検査 (未指定なら全パーティションをマージ)",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=1.0,
        help="振幅 A (期待値: Dy = A * sin(omega*t))",
    )
    parser.add_argument(
        "--omega", type=float, default=1.0, help="角振動数 omega [rad/s]"
    )
    parser.add_argument("--z-tol", type=float, default=1e-6, help="z=0 判定の許容誤差")
    parser.add_argument(
        "--atol", type=float, default=1e-6, help="変位比較の絶対許容誤差"
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-4,
        help="変位比較の相対許容誤差 (|期待値|に対して)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="NG 節点の詳細を表示"
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    mesh_path = os.path.join(script_dir, args.mesh)
    results_dir = os.path.join(script_dir, args.results_dir)

    # ----- メッシュ読み込み -----
    print("[mesh] {0}".format(mesh_path))
    if not os.path.exists(mesh_path):
        print("  ERROR: メッシュファイルが見つかりません")
        sys.exit(1)
    nodes = parse_mesh(mesh_path)
    print("  節点数: {0}".format(len(nodes)))

    z0_nodes = sorted(nid for nid, xyz in nodes.items() if abs(xyz[2]) < args.z_tol)
    print("  z=0 節点数 (メッシュ全体): {0}".format(len(z0_nodes)))
    if not z0_nodes:
        print("  z=0 の節点が見つかりません。終了します。")
        sys.exit(1)
    z0_set = set(z0_nodes)

    # ----- 結果ファイルを収集・グルーピング -----
    print("[results] {0}".format(results_dir))
    if not os.path.isdir(results_dir):
        print("  ERROR: ディレクトリが見つかりません")
        sys.exit(1)

    pat = re.compile(r"^{0}\.res\.(\d+)\.(\d+)$".format(re.escape(args.basename)))
    # {step: [(partition, path), ...]}
    groups = {}
    partitions_seen = set()
    for name in os.listdir(results_dir):
        m = pat.match(name)
        if not m:
            continue
        p = int(m.group(1))
        s = int(m.group(2))
        if args.partition is not None and p != args.partition:
            continue
        groups.setdefault(s, []).append((p, os.path.join(results_dir, name)))
        partitions_seen.add(p)

    if not groups:
        if args.partition is not None:
            print(
                "  パーティション {0} のファイルが見つかりません".format(args.partition)
            )
        else:
            print("  パターン '{0}.res.<p>.<s>' が見つかりません".format(args.basename))
        sys.exit(1)

    steps = sorted(groups.keys())
    n_parts = len(partitions_seen)
    print("  検出ステップ数: {0}".format(len(steps)))
    print("  検出パーティション: {0} 個 ({1})".format(n_parts, sorted(partitions_seen)))

    # ----- 各ステップを検証 -----
    print()
    print(
        "期待値: Dx = 0,  Dy = {0} * sin({1} * t),  Dz = 0".format(
            args.amplitude, args.omega
        )
    )
    print("許容  : atol = {0},  rtol = {1}".format(args.atol, args.rtol))
    print()
    hdr = "  step       t [s]      checked     max|err|    worst status"
    print(hdr)
    print("-" * len(hdr))

    n_ok = 0
    n_ng = 0
    # 全ステップを通じて一度も検査されなかった z=0 節点を追跡
    never_checked = set(z0_nodes)

    for step in steps:
        # このステップの全パーティションを読み、変位をマージ
        merged_disp = {}
        total_time = None
        parse_failed = False
        for p, fp in sorted(groups[step]):
            try:
                res = parse_result(fp)
            except Exception as e:
                print("{0:>6d}  ERROR parsing part {1}: {2}".format(step, p, e))
                parse_failed = True
                break
            if total_time is None:
                total_time = res["total_time"]
            merged_disp.update(res["displacement"])
        if parse_failed:
            n_ng += 1
            continue

        t = total_time

        ex = 0.0
        ey = args.amplitude * math.sin(args.omega * t)
        ez = 0.0
        expected = (ex, ey, ez)
        ref = max(abs(ex), abs(ey), abs(ez))
        tol_eff = args.atol + args.rtol * ref

        max_err = 0.0
        worst_nid = None
        ng_here = []
        checked = 0
        missing = []
        for nid in z0_nodes:
            if nid not in merged_disp:
                missing.append(nid)
                continue
            checked += 1
            never_checked.discard(nid)
            d = merged_disp[nid]
            errs = (d[0] - expected[0], d[1] - expected[1], d[2] - expected[2])
            err = max(abs(e) for e in errs)
            if err > max_err:
                max_err = err
                worst_nid = nid
            if err > tol_eff:
                ng_here.append((nid, d, errs))

        # 網羅失敗も NG 扱い (カバレッジ不足)
        coverage_ng = len(missing) > 0
        value_ng = len(ng_here) > 0

        if coverage_ng or value_ng:
            n_ng += 1
            status = "NG"
        else:
            n_ok += 1
            status = "OK"

        worst_str = "-" if worst_nid is None else str(worst_nid)
        coverage_str = "{0}/{1}".format(checked, len(z0_nodes))
        if status == "NG" or step % 100 == 0:
            print(
                "{0:>6d} {1:>11.5f} {2:>11s} {3:>12.3e} {4:>8} {5:>6}".format(
                    step, t, coverage_str, max_err, worst_str, status
                )
            )

        if args.verbose:
            if missing:
                preview = ", ".join(str(n) for n in missing[:10])
                more = (
                    ""
                    if len(missing) <= 10
                    else " ... 他 {0} 個".format(len(missing) - 10)
                )
                print(
                    "        未検査 ({0} 個): {1}{2}".format(
                        len(missing), preview, more
                    )
                )
            for nid, d, errs in ng_here[:5]:
                print(
                    "        node {0}: d=({1:+.3e}, {2:+.3e}, {3:+.3e})  "
                    "err=({4:+.2e}, {5:+.2e}, {6:+.2e})".format(
                        nid, d[0], d[1], d[2], errs[0], errs[1], errs[2]
                    )
                )
            if len(ng_here) > 5:
                print("        ... 他 {0} 節点が値 NG".format(len(ng_here) - 5))

    # ----- 集計 -----
    print("-" * len(hdr))
    print("合計: {0} ステップ中  OK={1},  NG={2}".format(len(steps), n_ok, n_ng))

    if never_checked:
        print()
        print(
            "[警告] 全ステップを通じて一度も検査されなかった z=0 節点が "
            "{0} 個あります:".format(len(never_checked))
        )
        nc = sorted(never_checked)
        preview = ", ".join(str(n) for n in nc[:20])
        more = "" if len(nc) <= 20 else " ... 他 {0} 個".format(len(nc) - 20)
        print("  {0}{1}".format(preview, more))
        if args.partition is not None:
            print(
                "  --partition {0} に限定しているため、他パーティションに"
                "属する節点かもしれません。".format(args.partition)
            )
        else:
            print("  メッシュには存在するが、どの .res にも出現していません。")
    elif n_ng == 0:
        print("すべての z=0 節点について、全ステップで変位が期待値と一致しました。")


if __name__ == "__main__":
    main()
