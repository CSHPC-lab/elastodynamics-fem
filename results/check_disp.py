#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VTK (legacy ASCII, UNSTRUCTURED_GRID) ファイルから z=0 節点の変位を検証する。

想定:
    - 結果ファイル: <results_dir>/disp_step_NNNN.vtk
        * NNNN は 0 埋め 4 桁のステップ番号 (0000, 0001, ...)
        * パーティション分割は無し (1 ステップ 1 ファイル)
    - 各ファイルに以下が含まれる:
        FIELD FieldData ... TOTALTIME
        POINTS <n> double
        POINT_DATA <n>
        VECTORS DISPLACEMENT double
    - z=0 の判定は POINTS に書かれた (原形状の) z 座標で行う
    - 期待値: z=0 の点で Dx=0, Dy = A*sin(omega*t), Dz=0

Usage:
    python3 check_disp.py
    python3 check_disp.py --amplitude 0.05 --omega 1.0 -v --results-dir disp

標準ライブラリのみ使用 (仮想環境不要)。
"""

import argparse
import math
import os
import re
import sys


# VTK のセクション名として値トークンの走査を打ち切るキーワード
VTK_SECTION_KEYWORDS = {
    "SCALARS",
    "VECTORS",
    "TENSORS",
    "NORMALS",
    "TEXTURE_COORDINATES",
    "POINT_DATA",
    "CELL_DATA",
    "CELLS",
    "CELL_TYPES",
    "LINES",
    "POLYGONS",
    "VERTICES",
    "TRIANGLE_STRIPS",
    "LOOKUP_TABLE",
    "FIELD",
    "METADATA",
}


def parse_vtk(path):
    """
    Returns:
        {
            'total_time'   : float,
            'coords'       : [(x, y, z), ...],        # POINTS
            'displacement' : [(dx, dy, dz), ...],     # VECTORS DISPLACEMENT
        }
    """
    with open(path, "r") as f:
        lines = f.read().split("\n")

    total_time = None
    n_points = None
    coords = None
    displacement = None

    def collect_floats(start_idx, needed):
        """start_idx から float を needed 個集めて、消費後の行番号と値のリストを返す。
        セクションキーワードに出会ったら打ち切る。"""
        vals = []
        idx = start_idx
        while len(vals) < needed and idx < len(lines):
            s = lines[idx].strip()
            if s:
                first = s.split()[0].upper()
                if first in VTK_SECTION_KEYWORDS:
                    break
                try:
                    vals.extend(float(x) for x in s.split())
                except ValueError:
                    break
            idx += 1
        return idx, vals

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        upper = line.upper()
        tokens = line.split()

        # ----- FIELD FieldData N -----
        if upper.startswith("FIELD "):
            n_arrays = int(tokens[2])
            i += 1
            for _ in range(n_arrays):
                while i < len(lines) and not lines[i].strip():
                    i += 1
                if i >= len(lines):
                    break
                hdr = lines[i].split()
                name = hdr[0]
                ncomp = int(hdr[1])
                ntuples = int(hdr[2])
                i += 1
                i, vals = collect_floats(i, ncomp * ntuples)
                if name.upper() == "TOTALTIME" and vals:
                    total_time = vals[0]
            continue

        # ----- POINTS n dtype -----
        if upper.startswith("POINTS "):
            n_points = int(tokens[1])
            i += 1
            i, vals = collect_floats(i, n_points * 3)
            coords = [
                (vals[3 * k], vals[3 * k + 1], vals[3 * k + 2]) for k in range(n_points)
            ]
            continue

        # ----- VECTORS name dtype -----
        if upper.startswith("VECTORS "):
            name = tokens[1]
            i += 1
            if name.upper() == "DISPLACEMENT" and n_points is not None:
                i, vals = collect_floats(i, n_points * 3)
                if len(vals) >= n_points * 3:
                    displacement = [
                        (vals[3 * k], vals[3 * k + 1], vals[3 * k + 2])
                        for k in range(n_points)
                    ]
            continue

        i += 1

    if total_time is None:
        raise ValueError("TOTALTIME が見つかりません: {0}".format(path))
    if coords is None:
        raise ValueError("POINTS が見つかりません: {0}".format(path))
    if displacement is None:
        raise ValueError("VECTORS DISPLACEMENT が見つかりません: {0}".format(path))

    return {
        "total_time": total_time,
        "coords": coords,
        "displacement": displacement,
    }


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="VTK ファイルの z=0 節点で Dy = A*sin(omega*t) になっているかを検証します。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir", default="results", help="VTK ファイルを置いたディレクトリ"
    )
    parser.add_argument(
        "--prefix",
        default="disp_step_",
        help="ファイル名のプレフィックス (<prefix>NNNN.vtk)",
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
    results_dir = os.path.join(script_dir, args.results_dir)

    # ----- 結果ファイルを収集 -----
    print("[results] {0}".format(results_dir))
    if not os.path.isdir(results_dir):
        print("  ERROR: ディレクトリが見つかりません")
        sys.exit(1)

    pat = re.compile(r"^{0}(\d+)\.vtk$".format(re.escape(args.prefix)))
    files = []
    for name in os.listdir(results_dir):
        m = pat.match(name)
        if m:
            files.append((int(m.group(1)), os.path.join(results_dir, name)))
    files.sort(key=lambda x: x[0])
    print("  検出ファイル数: {0}".format(len(files)))
    if not files:
        print("  パターン '{0}NNNN.vtk' が見つかりません".format(args.prefix))
        sys.exit(1)

    # ----- 最初のファイルから z=0 点のインデックスを決める -----
    # (全ファイルで POINTS は不変という前提)
    step0, path0 = files[0]
    res0 = parse_vtk(path0)
    coords = res0["coords"]
    n_total = len(coords)

    z0_indices = [k for k, (x, y, z) in enumerate(coords) if abs(z) < args.z_tol]
    print("  POINTS 総数        : {0}".format(n_total))
    print("  z=0 の点数          : {0}".format(len(z0_indices)))
    if not z0_indices:
        print("  z=0 の点が見つかりません。終了します。")
        sys.exit(1)

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

    for step, fp in files:
        try:
            res = parse_vtk(fp)
        except Exception as e:
            print("{0:>6d}  ERROR: {1}".format(step, e))
            n_ng += 1
            continue

        # 念のため POINTS 数の一致を確認 (形状が途中で変わる VTK は通常無い)
        if len(res["coords"]) != n_total:
            print(
                "{0:>6d}  WARN: POINTS 数が初期ファイルと異なる "
                "({1} -> {2})".format(step, n_total, len(res["coords"]))
            )

        t = res["total_time"]
        ex = 0.0
        ey = args.amplitude * math.sin(args.omega * t)
        ez = 0.0
        expected = (ex, ey, ez)
        ref = max(abs(ex), abs(ey), abs(ez))
        tol_eff = args.atol + args.rtol * ref

        disp = res["displacement"]
        max_err = 0.0
        worst_idx = None
        ng_here = []
        checked = 0
        for k in z0_indices:
            if k >= len(disp):
                continue
            checked += 1
            d = disp[k]
            errs = (d[0] - expected[0], d[1] - expected[1], d[2] - expected[2])
            err = max(abs(e) for e in errs)
            if err > max_err:
                max_err = err
                worst_idx = k
            if err > tol_eff:
                ng_here.append((k, d, errs))

        coverage_ng = checked < len(z0_indices)
        value_ng = len(ng_here) > 0
        status = "NG" if (coverage_ng or value_ng) else "OK"
        if status == "OK":
            n_ok += 1
        else:
            n_ng += 1

        # 表示用の 1-indexed 番号 (VTK 本体は 0-indexed だが人間向けに +1)
        worst_str = "-" if worst_idx is None else str(worst_idx + 1)
        cov_str = "{0}/{1}".format(checked, len(z0_indices))
        if status == "NG" or step % 100 == 0:
            print(
                "{0:>6d} {1:>11.5f} {2:>11s} {3:>12.3e} {4:>8} {5:>6}".format(
                    step, t, cov_str, max_err, worst_str, status
                )
            )

        if args.verbose and ng_here:
            for k, d, errs in ng_here[:5]:
                # 座標も出すと、どの位置の点が NG か分かって便利
                c = res["coords"][k]
                print(
                    "        point #{0} @ ({1:g},{2:g},{3:g}): "
                    "d=({4:+.3e}, {5:+.3e}, {6:+.3e})  "
                    "err=({7:+.2e}, {8:+.2e}, {9:+.2e})".format(
                        k + 1,
                        c[0],
                        c[1],
                        c[2],
                        d[0],
                        d[1],
                        d[2],
                        errs[0],
                        errs[1],
                        errs[2],
                    )
                )
            if len(ng_here) > 5:
                print("        ... 他 {0} 点が値 NG".format(len(ng_here) - 5))

    print("-" * len(hdr))
    print("合計: {0} ステップ中  OK={1},  NG={2}".format(len(files), n_ok, n_ng))
    if n_ng == 0:
        print("すべての z=0 点について、全ステップで変位が期待値と一致しました。")


if __name__ == "__main__":
    main()
