# FrontISTR 動的解析セットアップ

## 概要

10×10×100m 均質弾性体カラムの動的解析

- **境界条件**: z=0 面にディリクレ条件 (0, sin(t), 0)
- **材料**: c1=200, c2=100, ρ=2000 → E≈5.333×10⁷ Pa, ν=1/3
- **解析タイプ**: 微小変形（線形）と有限変形（非線形）の2ケース

## ファイル構成

```
column_mesh.cpp              ... Gmsh メッシュ生成 (C++, 元のスクリプト + .msh出力)
gmsh2fistr.py                ... Gmsh .msh → FrontISTR 変換スクリプト
job_linear.sh                ... Wisteria ジョブスクリプト（線形）
job_nonlinear.sh             ... Wisteria ジョブスクリプト（非線形）
```

変換スクリプト実行後に以下が生成される:

```
column_fistr.msh             ... FrontISTR メッシュファイル (HEC-MW形式)
hecmw_ctrl_linear.dat        ... 全体制御ファイル（線形用, part_in/part_out/fstrMSH 定義込み）
hecmw_ctrl_nonlinear.dat     ... 全体制御ファイル（非線形用）
column_linear.cnt            ... 解析制御ファイル（微小変形）
column_nonlinear.cnt         ... 解析制御ファイル（有限変形）
```

## 手順

### 1. メッシュ生成（ローカル or ログインノード）

```bash
g++ -o gen_mesh column_mesh.cpp -lgmsh
./gen_mesh
# → column.msh (v2.2 ASCII) と column.vtk が生成される
```

### 2. FrontISTR 形式に変換（ローカル or ログインノード）

```bash
python3 gmsh2fistr.py column.msh
```

### 3. Wisteria にファイル転送

```bash
# 必要なファイルを作業ディレクトリに配置
scp column_fistr.msh hecmw_ctrl_linear.dat hecmw_ctrl_nonlinear.dat \
    column_linear.cnt column_nonlinear.cnt \
    job_linear.sh job_nonlinear.sh \
    wisteria:/path/to/workdir/
```

### 4. Wisteria でジョブ投入

```bash
ssh wisteria
cd /path/to/workdir

# --- 線形動的解析（微小変形） ---
pjsub job_linear.sh

# --- 非線形動的解析（有限変形） ---
pjsub job_nonlinear.sh
```

ジョブ内部では以下が順に実行される:
1. `cp hecmw_ctrl_*.dat hecmw_ctrl.dat` で制御ファイルを設定
2. `PJM_MPI_PROC` から `hecmw_part_ctrl.dat` を自動生成
3. `hecmw_part1` がメッシュを N プロセス用に分割
4. `fistr1` が MPI N 並列で解析実行

### 5. 結果の可視化

```bash
# ParaView で VTK 出力を開く
paraview column_linear_vis_psf.*.pvtu

# モニタリングノードの変位時刻歴:
# dyna_disp_<node_id>.txt
```

## 時間積分パラメータ

| パラメータ | 値 |
|---|---|
| 手法 | Newmark-β (implicit) |
| γ | 0.5 |
| β | 0.25 (average acceleration) |
| Δt | 0.01 s |
| 総時間 | 20.0 s |
| ステップ数 | 2000 |

## 注意事項

- `!AMPLITUDE` の sin(t) テーブルは Δt と同一の 0.01s 間隔でサンプリング（補間なし）
- 非線形ケースの `!DYNAMIC, TYPE=NONLINEAR` は幾何学的非線形性（有限変形定式化）を有効にする。材料は線形弾性のまま
- 非線形ケースで `!ELASTIC` + `!DENSITY` を .cnt で再定義しているのは FrontISTR の仕様に従ったもの
- `hecmw_part_ctrl.dat` はジョブスクリプト内で `PJM_MPI_PROC` から自動生成されるため、`#PJM --mpi proc=N` を変更するだけで分割数が連動する
- MPI プロセス数を変更する場合はジョブスクリプトの `#PJM --mpi proc=` の値だけ書き換えればよい
- `!SUBDIR, ON` により結果ファイルがサブディレクトリに出力される