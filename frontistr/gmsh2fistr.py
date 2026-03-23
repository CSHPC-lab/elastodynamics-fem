#!/usr/bin/env python3
"""
gmsh2fistr.py
Gmsh .msh (v2.2 ASCII) → FrontISTR HEC-MW 形式変換スクリプト

  - 10節点四面体 (Gmsh type 11 → HEC-MW TYPE=342)
  - z < eps の節点を BOTTOM グループに自動登録
  - z > L-eps の節点を TOP グループに登録（モニタリング用）
  - (5,5,50) 最近傍の節点IDを表示（!DYNAMIC のモニタリングノードに使用）
  - sin(t) の AMPLITUDE テーブルを生成

使い方:
  python3 gmsh2fistr.py column.msh

出力:
  column_fistr.msh   ... FrontISTR メッシュファイル
  hecmw_ctrl.dat     ... 全体制御ファイル（線形用・非線形用）
  column_linear.cnt  ... 線形動的解析の制御ファイル
  column_nonlinear.cnt ... 非線形動的解析（有限変形）の制御ファイル
"""

import sys
import math
import os


# ============================================================
# パラメータ設定
# ============================================================
# 材料定数
RHO   = 2000.0        # 密度 [kg/m^3]
C1    = 200.0          # P波速度 [m/s]
C2    = 100.0          # S波速度 [m/s]
MU    = RHO * C2**2    # μ = 2.0e7 Pa
LAM   = RHO * C1**2 - 2*MU  # λ = 4.0e7 Pa
E     = MU * (3*LAM + 2*MU) / (LAM + MU)  # ≈ 5.333e7 Pa
NU    = LAM / (2*(LAM + MU))               # = 1/3

# 境界条件判定
Z_BOTTOM_EPS = 1.0e-6
Z_TOP = 100.0
Z_TOP_EPS = 1.0e-6

# 時間積分パラメータ
DT       = 0.01        # 時間刻み [s]
T_END    = 20.0        # 終了時間 [s]
N_STEPS  = int(T_END / DT)

# モニタリング点の目標座標
MONITOR_TARGET = (5.0, 5.0, 50.0)

# sin(t) amplitude テーブルの時間間隔
AMP_DT = DT    # Δt と同一間隔でサンプル（補間不要）


# ============================================================
# Gmsh v2.2 パーサー
# ============================================================
def parse_gmsh_v22(filepath):
    """Gmsh v2.2 ASCII .msh ファイルを読み込む"""
    nodes = {}       # {node_id: (x, y, z)}
    elements = []    # [(elem_id, type, [tags], [node_ids])]

    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line == '$Nodes':
            i += 1
            n_nodes = int(lines[i].strip())
            i += 1
            for _ in range(n_nodes):
                parts = lines[i].strip().split()
                nid = int(parts[0])
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                nodes[nid] = (x, y, z)
                i += 1

        elif line == '$Elements':
            i += 1
            n_elems = int(lines[i].strip())
            i += 1
            for _ in range(n_elems):
                parts = lines[i].strip().split()
                eid = int(parts[0])
                etype = int(parts[1])
                ntags = int(parts[2])
                tags = [int(parts[3 + j]) for j in range(ntags)]
                enodes = [int(x) for x in parts[3 + ntags:]]
                elements.append((eid, etype, tags, enodes))
                i += 1
        else:
            i += 1

    return nodes, elements


# ============================================================
# 変換メイン処理
# ============================================================
def convert(gmsh_file):
    print(f"読み込み: {gmsh_file}")
    nodes, elements = parse_gmsh_v22(gmsh_file)
    print(f"  節点数: {len(nodes)}")
    print(f"  要素数（全タイプ）: {len(elements)}")

    # Tet10 のみ抽出 (Gmsh type 11)
    tet10_elems = [(eid, enodes) for eid, etype, tags, enodes in elements
                   if etype == 11]
    print(f"  Tet10 要素数: {len(tet10_elems)}")

    if len(tet10_elems) == 0:
        print("ERROR: Tet10 (type 11) 要素が見つかりません")
        sys.exit(1)

    # Tet10 に含まれる節点のみ抽出（下位要素を除外）
    used_nodes = set()
    for _, enodes in tet10_elems:
        used_nodes.update(enodes)

    # 節点IDのリマッピング（1始まり連番）
    sorted_nids = sorted(used_nodes)
    old2new = {old: new + 1 for new, old in enumerate(sorted_nids)}

    # --- 境界節点グループ（座標フィルタ） ---
    bottom_nodes = []
    top_nodes = []
    for nid in sorted_nids:
        x, y, z = nodes[nid]
        if z < Z_BOTTOM_EPS:
            bottom_nodes.append(old2new[nid])
        if z > Z_TOP - Z_TOP_EPS:
            top_nodes.append(old2new[nid])

    print(f"  BOTTOM (z=0) 節点数: {len(bottom_nodes)}")
    print(f"  TOP (z=100) 節点数: {len(top_nodes)}")

    # --- モニタリングノードの検索 ---
    min_dist = float('inf')
    monitor_nid = 1
    tx, ty, tz = MONITOR_TARGET
    for nid in sorted_nids:
        x, y, z = nodes[nid]
        d = (x - tx)**2 + (y - ty)**2 + (z - tz)**2
        if d < min_dist:
            min_dist = d
            monitor_nid = old2new[nid]
    print(f"  モニタリングノード: {monitor_nid} "
          f"(目標 {MONITOR_TARGET} からの距離 {math.sqrt(min_dist):.4f})")

    # --- sin(t) AMPLITUDE テーブル生成 ---
    amp_points = []
    t = 0.0
    while t <= T_END + AMP_DT * 0.5:
        amp_points.append((math.sin(t), t))
        t += AMP_DT

    # ============================================================
    # FrontISTR メッシュファイル出力
    # ============================================================
    basename = os.path.splitext(os.path.basename(gmsh_file))[0]
    msh_out = f"{basename}_fistr.msh"

    with open(msh_out, 'w') as f:
        # ヘッダー
        f.write("!HEADER\n")
        f.write(f" {basename} - Converted from Gmsh by gmsh2fistr.py\n")

        # 節点
        f.write("!NODE\n")
        for nid in sorted_nids:
            new_id = old2new[nid]
            x, y, z = nodes[nid]
            f.write(f"  {new_id}, {x:.10E}, {y:.10E}, {z:.10E}\n")

        # 要素 (TYPE=342 = 10節点四面体)
        # Gmsh Tet10 の節点順序は HEC-MW TYPE=342 と同一
        f.write("!ELEMENT, TYPE=342, EGRP=ALL\n")
        for new_eid, (old_eid, enodes) in enumerate(tet10_elems, start=1):
            new_nodes = [old2new[n] for n in enodes]
            node_str = ", ".join(str(n) for n in new_nodes)
            f.write(f"  {new_eid}, {node_str}\n")

        # セクション定義
        f.write("!SECTION, TYPE=SOLID, EGRP=ALL, MATERIAL=MAT1\n")
        f.write("\n")

        # 材料定義
        f.write("!MATERIAL, NAME=MAT1, ITEM=2\n")
        f.write("!ITEM=1, SUBITEM=2\n")
        f.write(f"  {E:.6E}, {NU:.6f}\n")
        f.write("!ITEM=2\n")
        f.write(f"  {RHO:.6E}\n")

        # 節点グループ
        f.write("!NGROUP, NGRP=BOTTOM\n")
        _write_id_list(f, bottom_nodes)

        f.write("!NGROUP, NGRP=TOP\n")
        _write_id_list(f, top_nodes)

        f.write("!NGROUP, NGRP=MONITOR\n")
        f.write(f"  {monitor_nid}\n")

        # AMPLITUDE 定義 (sin(t))
        f.write("!AMPLITUDE, NAME=SIN_AMP, VALUE=ABSOLUTE\n")
        _write_amplitude(f, amp_points)

        f.write("!END\n")

    print(f"\n出力: {msh_out}")

    # ============================================================
    # hecmw_ctrl.dat（線形用）
    # ============================================================
    ctrl_linear = "hecmw_ctrl_linear.dat"
    with open(ctrl_linear, 'w') as f:
        f.write("## FrontISTR 全体制御ファイル（線形動的解析用）\n")
        f.write("## hecmw_part1 は part_in / part_out を参照\n")
        f.write("## fistr1 は fstrMSH を参照\n")
        f.write("!MESH, NAME=part_in, TYPE=HECMW-ENTIRE\n")
        f.write(f"  {msh_out}\n")
        f.write("!MESH, NAME=part_out, TYPE=HECMW-DIST\n")
        f.write("  column_linear.p\n")
        f.write("!MESH, NAME=fstrMSH, TYPE=HECMW-DIST\n")
        f.write("  column_linear.p\n")
        f.write("!CONTROL, NAME=fstrCNT\n")
        f.write("  column_linear.cnt\n")
        f.write("!RESULT, NAME=fstrRES, IO=OUT\n")
        f.write("  column_linear.res\n")
        f.write("!RESULT, NAME=vis_out, IO=OUT\n")
        f.write("  column_linear_vis\n")
        f.write("!SUBDIR, ON\n")
    print(f"出力: {ctrl_linear}")

    # ============================================================
    # hecmw_ctrl.dat（非線形用）
    # ============================================================
    ctrl_nonlinear = "hecmw_ctrl_nonlinear.dat"
    with open(ctrl_nonlinear, 'w') as f:
        f.write("## FrontISTR 全体制御ファイル（非線形動的解析用）\n")
        f.write("## hecmw_part1 は part_in / part_out を参照\n")
        f.write("## fistr1 は fstrMSH を参照\n")
        f.write("!MESH, NAME=part_in, TYPE=HECMW-ENTIRE\n")
        f.write(f"  {msh_out}\n")
        f.write("!MESH, NAME=part_out, TYPE=HECMW-DIST\n")
        f.write("  column_nonlinear.p\n")
        f.write("!MESH, NAME=fstrMSH, TYPE=HECMW-DIST\n")
        f.write("  column_nonlinear.p\n")
        f.write("!CONTROL, NAME=fstrCNT\n")
        f.write("  column_nonlinear.cnt\n")
        f.write("!RESULT, NAME=fstrRES, IO=OUT\n")
        f.write("  column_nonlinear.res\n")
        f.write("!RESULT, NAME=vis_out, IO=OUT\n")
        f.write("  column_nonlinear_vis\n")
        f.write("!SUBDIR, ON\n")
    print(f"出力: {ctrl_nonlinear}")

    # ============================================================
    # column_linear.cnt（線形動的解析）
    # ============================================================
    cnt_linear = "column_linear.cnt"
    with open(cnt_linear, 'w') as f:
        f.write("## FrontISTR 解析制御ファイル - 線形動的解析 (微小変形)\n")
        f.write("## 10x10x100m 弾性体カラム, z=0面: (0, sin(t), 0)\n")
        f.write("!VERSION\n")
        f.write("  3\n")
        f.write("!WRITE,RESULT,FREQUENCY=100\n")
        f.write("!WRITE,VISUAL,FREQUENCY=100\n")
        f.write("\n")
        f.write("!SOLUTION, TYPE=DYNAMIC\n")
        f.write("!DYNAMIC, TYPE=LINEAR\n")
        f.write("!! idx_eqa(11=implicit Newmark), idx_resp(1=time history)\n")
        f.write(f"  11, 1\n")
        f.write(f"!! t_start, t_end, n_step, dt\n")
        f.write(f"  0.0, {T_END}, {N_STEPS}, {DT:.6E}\n")
        f.write(f"!! gamma, beta (Newmark-beta: 0.5, 0.25 = average acceleration)\n")
        f.write(f"  0.5, 0.25\n")
        f.write(f"!! idx_mas(1=lumped), idx_dmp(1=Rayleigh), ray_m, ray_k\n")
        f.write(f"  1, 1, 0.0, 0.0\n")
        f.write(f"!! nout, node_monit, nout_monit\n")
        f.write(f"  100, {monitor_nid}, 1\n")
        f.write(f"!! iout_list (disp, vel, acc, react, stress, strain)\n")
        f.write(f"  1, 1, 1, 1, 1, 1\n")
        f.write("\n")
        f.write("## 境界条件\n")
        f.write("!! z=0面: x方向固定, z方向固定\n")
        f.write("!BOUNDARY\n")
        f.write("  BOTTOM, 1, 1, 0.0\n")
        f.write("  BOTTOM, 3, 3, 0.0\n")
        f.write("!! z=0面: y方向 = sin(t)\n")
        f.write("!BOUNDARY, AMP=SIN_AMP\n")
        f.write("  BOTTOM, 2, 2, 1.0\n")
        f.write("\n")
        f.write("## 材料 (メッシュファイルで定義済み)\n")
        f.write("\n")
        f.write("## ソルバー設定\n")
        f.write("!SOLVER,METHOD=CG,PRECOND=1,ITERLOG=NO,TIMELOG=YES\n")
        f.write("  10000, 1\n")
        f.write("  1.0e-08, 1.0, 0.0\n")
        f.write("!END\n")
    print(f"出力: {cnt_linear}")

    # ============================================================
    # column_nonlinear.cnt（非線形動的解析：有限変形）
    # ============================================================
    cnt_nonlinear = "column_nonlinear.cnt"
    with open(cnt_nonlinear, 'w') as f:
        f.write("## FrontISTR 解析制御ファイル - 非線形動的解析 (有限変形)\n")
        f.write("## 10x10x100m 弾性体カラム, z=0面: (0, sin(t), 0)\n")
        f.write("## 幾何学的非線形性を考慮 (!DYNAMIC, TYPE=NONLINEAR)\n")
        f.write("!VERSION\n")
        f.write("  3\n")
        f.write("!WRITE,RESULT,FREQUENCY=100\n")
        f.write("!WRITE,VISUAL,FREQUENCY=100\n")
        f.write("\n")
        f.write("!SOLUTION, TYPE=DYNAMIC\n")
        f.write("!DYNAMIC, TYPE=NONLINEAR\n")
        f.write("!! idx_eqa(11=implicit Newmark), idx_resp(1=time history)\n")
        f.write(f"  11, 1\n")
        f.write(f"!! t_start, t_end, n_step, dt\n")
        f.write(f"  0.0, {T_END}, {N_STEPS}, {DT:.6E}\n")
        f.write(f"!! gamma, beta\n")
        f.write(f"  0.5, 0.25\n")
        f.write(f"!! idx_mas, idx_dmp, ray_m, ray_k\n")
        f.write(f"  1, 1, 0.0, 0.0\n")
        f.write(f"!! nout, node_monit, nout_monit\n")
        f.write(f"  100, {monitor_nid}, 1\n")
        f.write(f"!! iout_list\n")
        f.write(f"  1, 1, 1, 1, 1, 1\n")
        f.write("\n")
        f.write("## 境界条件\n")
        f.write("!BOUNDARY, GRPID=1\n")
        f.write("  BOTTOM, 1, 1, 0.0\n")
        f.write("  BOTTOM, 3, 3, 0.0\n")
        f.write("!BOUNDARY, GRPID=1, AMP=SIN_AMP\n")
        f.write("  BOTTOM, 2, 2, 1.0\n")
        f.write("\n")
        f.write("## ステップ制御（非線形のみ必要）\n")
        f.write("!STEP, CONVERG=1.0e-6\n")
        f.write("  BOUNDARY, 1\n")
        f.write("\n")
        f.write("## 材料定義（非線形解析では .cnt で指定）\n")
        f.write("!ELASTIC\n")
        f.write(f"  {E:.6E}, {NU:.6f}\n")
        f.write("!DENSITY\n")
        f.write(f"  {RHO:.6E}\n")
        f.write("\n")
        f.write("## ソルバー設定\n")
        f.write("!SOLVER,METHOD=CG,PRECOND=1,ITERLOG=NO,TIMELOG=YES\n")
        f.write("  10000, 1\n")
        f.write("  1.0e-08, 1.0, 0.0\n")
        f.write("!END\n")
    print(f"出力: {cnt_nonlinear}")

    # ============================================================
    # 実行手順の表示
    # ============================================================
    print("\n" + "=" * 60)
    print("FrontISTR 実行手順 (Wisteria)")
    print("=" * 60)
    print()
    print("【Wisteria にファイル転送】")
    print(f"  scp {msh_out} {ctrl_linear} {ctrl_nonlinear}")
    print(f"      {cnt_linear} {cnt_nonlinear}")
    print(f"      job_linear.sh job_nonlinear.sh wisteria:workdir/")
    print()
    print("【線形動的解析（微小変形）】")
    print(f"  pjsub job_linear.sh")
    print()
    print("【非線形動的解析（有限変形）】")
    print(f"  pjsub job_nonlinear.sh")
    print()
    print(f"結果ファイル:")
    print(f"  *_vis_psf.* → ParaView で可視化")
    print(f"  dyna_disp_{monitor_nid}.txt → モニタリングノード変位履歴")
    print()
    print(f"モニタリングノード ID: {monitor_nid}")
    print(f"材料: E={E:.4E} Pa, ν={NU:.6f}, ρ={RHO:.1f} kg/m³")
    print(f"時間積分: Newmark-β (γ=0.5, β=0.25), Δt={DT}s, T={T_END}s")


# ============================================================
# ユーティリティ関数
# ============================================================
def _write_id_list(f, ids, per_line=10):
    """節点/要素IDリストを書き出す（1行あたり per_line 個）"""
    for i in range(0, len(ids), per_line):
        chunk = ids[i:i + per_line]
        f.write("  " + ", ".join(str(n) for n in chunk) + "\n")


def _write_amplitude(f, points):
    """AMPLITUDE テーブルを書き出す（1行に value,time ペア4組まで）"""
    pairs_per_line = 4
    for i in range(0, len(points), pairs_per_line):
        chunk = points[i:i + pairs_per_line]
        parts = []
        for val, t in chunk:
            parts.append(f"{val:.8f}")
            parts.append(f"{t:.4f}")
        f.write("  " + ", ".join(parts) + "\n")


# ============================================================
# エントリーポイント
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"使い方: python3 {sys.argv[0]} <input.msh>")
        print(f"  例: python3 {sys.argv[0]} column.msh")
        sys.exit(1)

    convert(sys.argv[1])