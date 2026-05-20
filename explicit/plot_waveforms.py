#!/usr/bin/env python3
"""
観測点波形をプロット・比較する。

Usage:
conda activate visualize
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# パラメタ
# ============================================================

# プロットしたいファイル: (path, label) のリスト
# 複数並べれば重ね描き比較
FILES = [
    ("1.dat", "1"),
]

# 時間刻み [s]  (main.f の dt と合わせる)
DT = 0.012

# 観測点数 (main.f の nobs と合わせる)
NOBS = 3

# 出力 PNG
OUT = "waveforms.png"

# ============================================================


def load(path, nobs):
    """nobs 行ごとに観測点がインターリーブ。各観測点を (nt, 3) に分離して返す。"""
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[:, None]
    nt = data.shape[0] // nobs
    if nt * nobs != data.shape[0]:
        print(
            f"  warning: {path} has {data.shape[0]} rows, not a multiple of nobs={nobs}; "
            f"truncating to {nt * nobs}"
        )
    data = data[: nt * nobs]
    return [data[i::nobs] for i in range(nobs)]


def main():
    datasets = []
    for path, label in FILES:
        obs = load(path, NOBS)
        print(f"{path}: nt={obs[0].shape[0]}, nobs={len(obs)}, ncomp={obs[0].shape[1]}")
        datasets.append((label, obs))

    ncomp = datasets[0][1][0].shape[1]
    comp_names = (
        ["u_x", "u_y", "u_z"][:ncomp] if ncomp <= 3 else [f"c{j}" for j in range(ncomp)]
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(
        NOBS,
        ncomp,
        figsize=(4 * ncomp + 1, 1.8 * NOBS + 1),
        sharex=True,
        squeeze=False,
    )

    for i in range(NOBS):
        for j in range(ncomp):
            ax = axes[i, j]
            for k, (label, obs) in enumerate(datasets):
                y = obs[i][:, j]
                x = np.arange(len(y)) * DT
                ax.plot(x, y, color=colors[k % len(colors)], lw=0.7, label=label)

            ax.grid(True, alpha=0.3)
            if j == 0:
                ax.set_ylabel(f"obs{i+1}\n{comp_names[j]} [m]", fontsize=8)
            else:
                ax.set_ylabel(f"{comp_names[j]} [m]", fontsize=8)
            if i == 0:
                ax.set_title(comp_names[j], fontsize=9)
            if i == NOBS - 1:
                ax.set_xlabel("Time [s]")

    if len(datasets) > 1:
        axes[0, 0].legend(fontsize=8, loc="upper right")

    fig.suptitle("Displacement waveforms (main.f output)", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
