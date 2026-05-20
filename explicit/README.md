# explicit/ — 陽解法 3D 弾性波動シミュレーション

## 概要

点震源（モーメントテンソル）を与え、3D直方体ドメインの弾性波動を陽解法で計算するコード。
空間離散化に **Hermite 型 8 節点六面体要素（1 節点 12 自由度）**、
時間積分に **中心差分法（2 次精度）** を使用する。

GPU 版（`main_oacc.f90`）は OpenACC による NVIDIA GPU オフロードに対応する。

---

## 物理モデル

### 支配方程式

3 次元線形等方弾性体の運動方程式：

```
ρ ü = ∇·σ + f
σ = λ(∇·u)I + μ(∇u + ∇uᵀ)
```

- ρ: 密度、λ・μ: ラメ定数、u: 変位ベクトル、f: 体積力（震源）
- P 波速度 c₁ = √((λ+2μ)/ρ)、S 波速度 c₂ = √(μ/ρ)

### 境界条件

| 面 | 条件 |
|---|---|
| z = 0（底面）| 固定境界（変位ゼロ） |
| x = 0, x = xmax | 固定境界 |
| y = 0, y = ymax | 固定境界 |
| z = zmax（上面）| **自由表面**（境界条件なし）|

### 震源モデル

点モーメントテンソル震源。走向（strike）・傾斜（dip）・すべり角（rake）から
モーメントテンソル行列 M を構成し、等価節点力として入力する。

震源時間関数：滑らかなランプ関数（risetime = 2 s）
```
source(t) = moment × S(t)
S(t) = 2t²/rt²          (0 ≤ t ≤ rt/2)
S(t) = 1 - 2(t-rt)²/rt² (rt/2 < t ≤ rt)
S(t) = 1                 (t > rt)
```

---

## 要素定式化（Hermite 型 Hex8）

### 1 節点 12 自由度の意味

各節点は変位とその 1 次空間微分を自由度として持つ：

```
各節点の DOF（変位成分 α = x,y,z ごと）:
  モード 1: uα
  モード 2: ∂uα/∂x
  モード 3: ∂uα/∂y
  モード 4: ∂uα/∂z
→ 4 モード × 3 方向 = 12 DOF/節点
```

これはテイラー展開の 1 次近似係数を節点自由度とした Hermite 型補間に相当する。
標準の Hex8（1 節点 3 DOF）と比べて高次補間が可能で、弾性波の数値分散を低減できる。

### 要素あたりの規模

- 8 節点 × 12 DOF = **96 DOF/要素**
- 要素剛性行列: 96×96（kek・keg 各 1 枚、全要素共通）
- 節点質量行列: 4×4（3 空間方向に共通のモード間結合）

---

## ファイル構成

| ファイル | 内容 |
|---|---|
| `main.f` | メインプログラム（CPU 版、固定形式 Fortran） |
| `main_oacc.f90` | GPU 版（OpenACC、自由形式 Fortran）`coemoment`・`svd_mgtn` を末尾に含む |
| `keme.f` | `def_ke(ds, kek, keg)` — 要素剛性行列の解析値を設定<br>`cmp_me(ds, rho, rmtmp)` — 要素質量行列の計算 |
| `pointsource.f` | `cmp_eff_fault(coem, ds, ft)` — モーメントテンソル→等価節点力<br>`cmp_eqBg(ds, x1, x2, x3, eqBg)` — B 行列の解析計算 |
| `job.sh` | SLURM ジョブスクリプト（CPU 版） |
| `job_oacc.sh` | SLURM ジョブスクリプト（GPU 版） |
| `measurement.sh` | Nsight Compute プロファイル取得用ジョブ |

---

## コンパイルと実行

### CPU 版

```bash
cd /data3/kusumoto/elastodynamics-fem/explicit

gfortran -O2 -ffixed-line-length-none -mcmodel=medium \
  main.f keme.f pointsource.f \
  /usr/lib/x86_64-linux-gnu/lapack/liblapack.so.3 -o a.out

./a.out
```

### GPU 版（OpenACC）

```bash
. /etc/profile.d/modules.sh && module load nvhpc/25.1

nvfortran -O2 -acc -gpu=cc80,cuda12.6 \
  main_oacc.f90 keme.f pointsource.f \
  /usr/lib/x86_64-linux-gnu/lapack/liblapack.so.3 -o a_oacc.out

./a_oacc.out
```

### SLURM 投入

```bash
export PATH=$PATH:/opt/slurm/22.05.2/bin
sbatch job_oacc.sh          # GPU 版
sbatch job.sh               # CPU 版
squeue -u kusumoto          # 状態確認
tail -f ../cpp_log/slurm.<JOB_ID>.out   # ログ監視
```

---

## パラメータ

| パラメータ | 値 | 物理的意味 |
|---|---|---|
| `nex`, `ney`, `nez` | 101, 101, 50 | x・y・z 方向の要素数 |
| `ds` | 270 m | 格子間隔（正方格子） |
| `dt` | 0.012 s | 時間刻み |
| `nt` | 100,000 | 総時間ステップ数（総時間 1200 s） |
| `nobs` | 3 | 観測点数 |
| `kd` | 2 | 材料種別数（地殻・マントル） |
| `c1(1), c1(2)` | 3900, 7800 m/s | P 波速度（地殻・マントル） |
| `c2(1), c2(2)` | 2250, 4500 m/s | S 波速度（地殻・マントル） |
| `rho(1), rho(2)` | 2500, 3000 kg/m³ | 密度（地殻・マントル） |
| `strike` | 30° | 断層の走向 |
| `dip` | 40° | 断層の傾斜角 |
| `rake` | 50° | すべり角 |
| `moment` | 1×10¹⁵ N·m | 地震モーメント |
| `rt` | 2.0 s | 震源時間関数の立ち上がり時間 |

### ドメインサイズ

```
x: nex × ds = 101 × 270 =  27,270 m
y: ney × ds = 101 × 270 =  27,270 m
z: nez × ds =  50 × 270 =  13,500 m
```

震源位置（`fault(1..3)`）: ドメイン中央水平位置、深さ 13,500 - 2,025 = 11,475 m

### 安定条件（CFL）

中心差分の安定条件: dt ≤ ds / c_max

```
c_max = 7800 m/s（マントル P 波速度）
ds / c_max = 270 / 7800 ≈ 0.0346 s
dt = 0.012 s → 安全率 ≈ 0.35  （Hermite 要素は標準より厳しい制約を持つ）
```

### 材料境界

深さ方向で 2 層に分類：
```
要素中心 z > nez*ds - 2700 m  →  num(ie) = 1（地殻、上層 2700 m）
それ以外                       →  num(ie) = 2（マントル）
```

---

## データ構造

| 配列 | サイズ | 内容 |
|---|---|---|
| `up(12*n)` / `un(12*n)` / `um(12*n)` | 各 12×530,604 | 現ステップ・1 步前・2 步前の変位 |
| `rm(4,4,n)` | 4×4×530,604 | 節点質量行列（前処理後は逆行列を上書き格納） |
| `kek(96,96)` | 96×96 | 体積弾性剛性行列（λ 成分、全要素共通） |
| `keg(96,96)` | 96×96 | せん断剛性行列（μ 成分、全要素共通） |
| `cny(8,ne)` | 8×510,050 | 要素-節点接続テーブル |
| `num(ne)` | 510,050 | 材料番号（1=地殻, 2=マントル） |
| `flag(nex+1,ney+1,nez+1)` | 102×102×51 | 3D 格子インデックス → 節点番号変換 |
| `source(nt)` | 100,000 | 震源時間関数 × モーメント |

変位配列のレイアウト（節点 id の 12 DOF）：
```
up(12*(id-1)+1)  ... up(12*(id-1)+12)
= [u_x, u_y, u_z, ∂u_x/∂x, ..., ∂u_z/∂z]
  └── モード1 ──┘ └────── モード2〜4 × 3方向 ──────┘
```

---

## アルゴリズム

### 前処理（時間ループ前、CPU）

```
1. 震源・観測点の座標設定
2. 格子インデックステーブル flag(i,j,k) の構築
3. 要素-節点テーブル cny(8,ie) の構築、材料番号 num(ie) の設定
4. 質量行列アセンブル: rm(:,:,id) += cmp_me() の全要素スキャッタ加算
5. 節点ごとに SVD で逆行列化（rm に上書き）
6. 要素剛性行列 def_ke() → kek・keg（全要素共通）
7. 等価節点力定数 ft_const = -cmp_eff_fault() を事前計算（GPU 版のみ）
```

### 時間ループ（中心差分、GPU でオフロード）

```
do it = 1, nt

  [フェーズ 1] up = 0 で初期化

  [フェーズ 2] 内力計算（全要素ループ）  ← 計算量の大半
    do ie = 1, ne
      gather:  ut(96) = un(12節点分の変位)
      compute: ft(96) = Kl*kek*ut + Gl*keg*ut
      if ie == ie_src: ft += ft_const * source(it)   [震源力]
      scatter: up(8節点) += ft   [!$acc atomic で競合回避]
    enddo

  [フェーズ 3] 境界条件（5 面を強制ゼロ）

  [フェーズ 4] 質量行列の適用 + 時間更新
    do id = 1, n
      up = -dt² * M⁻¹(id) * up     [M⁻¹ は前処理済み]
      up = up + 2*un - um           [中心差分時間更新]
      um = un; un = up              [時刻シフト]
    enddo

  [フェーズ 5] 出力: nobs 観測点の up を書き出し

enddo
```

### 中心差分の時間積分式

$$u^{n+1} = -\Delta t^2 M^{-1} f^{int}(u^n) + 2u^n - u^{n-1}$$

---

## 出力ファイル（output.dat）

各時間ステップで nobs=3 行を書き出す。

```
フォーマット: (float64) ux  uy  uz
行数: nt × nobs = 100,000 × 3 = 300,000 行
```

観測点座標（ドメイン内、z = zmax 自由表面）：
```
obs1: fault + (4725, 4725, 0) m
obs2: fault + (7425, 7425, 0) m  （2700 m 間隔で断層から離れる）
obs3: fault + (10125, 10125, 0) m
```

Python での読み込み例：
```python
import numpy as np
data = np.loadtxt('output.dat')   # shape: (300000, 3)
# 観測点 1 の z 方向変位時系列
uz_obs1 = data[0::3, 2]           # 3行ごとに obs1
```

---

## OpenACC 実装の注意点（main_oacc.f90）

### matmul を使えない理由

`kek`（96×96）× `ut`（96）の `matmul` 組み込み関数は、nvfortran が内部一時配列（`kek$r` 等）を生成するため `!$acc parallel loop` 内で **S-0155 コンパイルエラー**が発生する。このため内積をスカラー `val` で展開している：

```fortran
! CPU 版 main.f（matmul）
ft = Kl(in)*matmul(kek,ut) + Gl(in)*matmul(keg,ut)

! GPU 版 main_oacc.f90（展開、val のみ private）
do i1=1,8; do ii=1,12
  val = 0.d0
  do jj=1,8; do kk=1,12
    val = val + (Kl*kek(...) + Gl*keg(...))*un(...)
  enddo; enddo
  up(...) += val   ! !$acc atomic update
enddo; enddo
```

### GPU で並列化されているループ

| ループ | ディレクティブ | 並列度 |
|---|---|---|
| `up=0.` 初期化 | `!$acc parallel loop` | ne×12 ≈ 630 万 |
| 内力 `do ie=1,ne` | `!$acc parallel loop private(val,...)` | 51 万 |
| 境界条件（3 種） | `!$acc parallel loop collapse(2)` | 各 1〜2 万 |
| 質量適用 `do id=1,n` | `!$acc parallel loop private(upm,fpm,...)` | 53 万 |
| 観測点抽出 | `!$acc parallel loop collapse(2)` | 9 |

### データ転送の最小化

- `!$acc data copyin(...)`: ループ開始前に 1 回だけ転送
- 観測点 3 点分（`obs_buf`、54 バイト）のみ毎ステップ `!$acc update host`
- `up`（51 MB）はデバイス上に常駐、毎ステップの転送なし
