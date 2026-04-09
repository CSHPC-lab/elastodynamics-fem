# MPI 主要関数リファレンス（C++, 非同期優先）

## 1. 初期化・終了・基本情報

```cpp
MPI_Init(&argc, &argv);
//   argc, argv: main() の引数をそのまま渡す。MPI環境を初期化。

int rank, size;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//   comm:  コミュニケータ（通常 MPI_COMM_WORLD = 全プロセス）
//   &rank: [OUT] 自プロセスの番号 (0 ~ size-1)

MPI_Comm_size(MPI_COMM_WORLD, &size);
//   comm:  コミュニケータ
//   &size: [OUT] コミュニケータ内の総プロセス数

MPI_Finalize();
//   引数なし。MPI環境を終了。以降MPI関数は呼べない。
```

---

## 2. Point-to-Point 通信

### 2.1 非同期送受信（基本はこちら）

```cpp
MPI_Request req;

MPI_Isend(
    buf,            // [IN]  送信データの先頭ポインタ
    count,          // [IN]  要素数
    MPI_DOUBLE,     // [IN]  データ型 (MPI_INT, MPI_FLOAT, MPI_CHAR, ...)
    dest,           // [IN]  送信先 rank
    tag,            // [IN]  メッセージ識別用タグ (任意の int)
    MPI_COMM_WORLD, // [IN]  コミュニケータ
    &req            // [OUT] 非同期操作ハンドル
);

MPI_Irecv(
    buf,            // [OUT] 受信バッファの先頭ポインタ
    count,          // [IN]  受信する最大要素数
    MPI_DOUBLE,     // [IN]  データ型
    source,         // [IN]  送信元 rank（MPI_ANY_SOURCE で任意）
    tag,            // [IN]  タグ（MPI_ANY_TAG で任意）
    MPI_COMM_WORLD, // [IN]  コミュニケータ
    &req            // [OUT] 非同期操作ハンドル
);
```

### 2.2 完了待ち

```cpp
MPI_Wait(
    &req,              // [IN/OUT] 待つハンドル。完了後 MPI_REQUEST_NULL になる
    MPI_STATUS_IGNORE  // [OUT]    状態情報（不要なら MPI_STATUS_IGNORE）
);

// 複数まとめて待つ（ゴースト交換で多用）
MPI_Request reqs[n];
MPI_Waitall(
    n,                  // [IN]  ハンドル数
    reqs,               // [IN/OUT] ハンドル配列
    MPI_STATUSES_IGNORE // [OUT] 状態配列（不要なら MPI_STATUSES_IGNORE）
);
```

### 2.3 同期版（参考）

```cpp
MPI_Send(buf, count, datatype, dest, tag, comm);
//   引数は MPI_Isend と同じ（req が無い）。送信完了までブロック。

MPI_Recv(buf, count, datatype, source, tag, comm, &status);
//   引数は MPI_Irecv と同じ（req の代わりに status）。受信完了までブロック。
```

---

## 3. 集団通信（Collective）

### 3.1 Iallreduce — 全員で演算 → 全員に結果

```cpp
MPI_Request req;
MPI_Iallreduce(
    sendbuf,        // [IN]  入力データ（MPI_IN_PLACE なら recvbuf を入出力に兼用）
    recvbuf,        // [OUT] 結果の格納先
    count,          // [IN]  要素数
    MPI_DOUBLE,     // [IN]  データ型
    MPI_SUM,        // [IN]  演算 (MPI_SUM / MPI_MAX / MPI_MIN / MPI_PROD / ...)
    MPI_COMM_WORLD, // [IN]  コミュニケータ
    &req            // [OUT] 非同期ハンドル
);
MPI_Wait(&req, MPI_STATUS_IGNORE);
```

**MPI_IN_PLACE の使い方（バッファ節約）:**
```cpp
double val = local_value;
MPI_Iallreduce(MPI_IN_PLACE, &val, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &req);
// 完了後 val に全プロセスの合計が入る
```

### 3.2 Ireduce — 全員で演算 → root だけに結果

```cpp
MPI_Ireduce(
    sendbuf,        // [IN]  入力データ
    recvbuf,        // [OUT] 結果（root のみ有効）
    count,          // [IN]  要素数
    MPI_DOUBLE,     // [IN]  データ型
    MPI_SUM,        // [IN]  演算
    root,           // [IN]  結果を受け取る rank
    MPI_COMM_WORLD, // [IN]  コミュニケータ
    &req            // [OUT] 非同期ハンドル
);
```

### 3.3 Ibcast — root の値を全員にコピー

```cpp
MPI_Ibcast(
    buf,            // [IN/OUT] root:送信データ / 他:受信バッファ
    count,          // [IN]  要素数
    MPI_DOUBLE,     // [IN]  データ型
    root,           // [IN]  送信元 rank
    MPI_COMM_WORLD, // [IN]  コミュニケータ
    &req            // [OUT] 非同期ハンドル
);
```

### 3.4 Iallgather — 各プロセスの値を全員が集める

```cpp
MPI_Iallgather(
    sendbuf,        // [IN]  自分が提供するデータ
    sendcount,      // [IN]  自分が送る要素数
    MPI_DOUBLE,     // [IN]  送信データ型
    recvbuf,        // [OUT] 全員分の結果 (サイズ = sendcount * size)
    recvcount,      // [IN]  各プロセスから受け取る要素数 (= sendcount)
    MPI_DOUBLE,     // [IN]  受信データ型
    MPI_COMM_WORLD, // [IN]  コミュニケータ
    &req            // [OUT] 非同期ハンドル
);
```

### 3.5 Iscatter — root のデータを均等に分配

```cpp
MPI_Iscatter(
    sendbuf,        // [IN]  root: 全データ (サイズ = sendcount * size) / 他: 無視
    sendcount,      // [IN]  各プロセスに送る要素数
    MPI_DOUBLE,     // [IN]  送信データ型
    recvbuf,        // [OUT] 受け取るバッファ
    recvcount,      // [IN]  受け取る要素数 (= sendcount)
    MPI_DOUBLE,     // [IN]  受信データ型
    root,           // [IN]  送信元 rank
    MPI_COMM_WORLD, // [IN]  コミュニケータ
    &req            // [OUT] 非同期ハンドル
);
```

### 3.6 Igather — 全員のデータを root に集める

```cpp
MPI_Igather(
    sendbuf,        // [IN]  自分のデータ
    sendcount,      // [IN]  自分が送る要素数
    MPI_DOUBLE,     // [IN]  送信データ型
    recvbuf,        // [OUT] root: 全データ格納先 (サイズ = recvcount * size) / 他: 無視
    recvcount,      // [IN]  各プロセスから受け取る要素数
    MPI_DOUBLE,     // [IN]  受信データ型
    root,           // [IN]  収集先 rank
    MPI_COMM_WORLD, // [IN]  コミュニケータ
    &req            // [OUT] 非同期ハンドル
);
```

---

## 4. 同期バリア

```cpp
MPI_Barrier(MPI_COMM_WORLD);
//   comm: コミュニケータ。全プロセスがここに到達するまでブロック。
//   デバッグ用。性能クリティカルなところでは避ける。

// 非同期版
MPI_Ibarrier(MPI_COMM_WORLD, &req);
```

---

## 5. 主なデータ型定数

| MPI定数       | C++型          |
|---------------|----------------|
| MPI_INT       | int            |
| MPI_LONG      | long           |
| MPI_FLOAT     | float          |
| MPI_DOUBLE    | double         |
| MPI_CHAR      | char           |
| MPI_UNSIGNED  | unsigned int   |
| MPI_LONG_LONG | long long      |

---

## 6. 主な演算定数 (Reduce系で使用)

| MPI定数    | 意味       |
|------------|------------|
| MPI_SUM    | 合計       |
| MPI_PROD   | 積         |
| MPI_MAX    | 最大値     |
| MPI_MIN    | 最小値     |
| MPI_LAND   | 論理AND    |
| MPI_LOR    | 論理OR     |
| MPI_MAXLOC | 最大値と位置 |
| MPI_MINLOC | 最小値と位置 |

---

## 7. FEMソルバでの典型パターン

```cpp
// ゴースト節点の値交換（非同期 Send/Recv）
std::vector<MPI_Request> reqs(2 * num_neighbors);
int idx = 0;
for (auto& [nbr_rank, send_nodes, recv_nodes] : comm_table) {
    // boundary → 隣接の ghost へ
    MPI_Isend(send_buf.data(), send_nodes.size(), MPI_DOUBLE,
              nbr_rank, 0, MPI_COMM_WORLD, &reqs[idx++]);
    // 隣接の boundary → 自分の ghost へ
    MPI_Irecv(recv_buf.data(), recv_nodes.size(), MPI_DOUBLE,
              nbr_rank, 0, MPI_COMM_WORLD, &reqs[idx++]);
}
// 通信中に inner 節点の計算を進める
compute_inner_nodes();
// 通信完了を待ってから boundary 節点の計算
MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
compute_boundary_nodes();

// CG法の内積（Allreduce）
double local_dot = 0.0;
for (int i = 0; i < n_local; i++) local_dot += r[i] * r[i];
double global_dot;
MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
```
