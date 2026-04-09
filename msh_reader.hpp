#ifndef MSH_READER_HPP
#define MSH_READER_HPP

#include <string>
#include <vector>
#include <map>
#include <set>

// ── 隣接パーティションとの通信情報 ──────────────────────

struct CommNeighbor
{
    int partition_id;

    // 自パーティション所有かつ相手にも必要な節点（ローカルID）
    // SpMV前に相手へ送信する
    // グローバルID昇順で格納 → 相手のrecv順序と自動的に一致
    std::vector<int> send_nodes;

    // 相手パーティション所有のゴースト節点（連続ブロック）
    // SpMV前に相手から受信する
    int recv_start; // ローカルID配列中の開始位置
    int recv_count; // 節点数

    int *send_ptr() { return send_nodes.data(); }
    int send_size() const { return static_cast<int>(send_nodes.size()); }
    const int *send_ptr() const { return send_nodes.data(); }
};

// ── パーティションごとのFEMメッシュ ─────────────────────

struct FEMMesh
{
    int my_partition = -1; // 1-indexed (MSH 4.1に合わせる)

    // 節点配置: [inner | bdr | ghost_nbr0 | ghost_nbr1 | ...]
    //  inner : 自パーティションのみが参照する節点
    //  bdr   : 自パーティション所有だが、他も参照する境界節点
    //  ghost : 他パーティション所有だが自分の要素が参照する節点
    int num_inner = 0;
    int num_bdr = 0;
    int num_owned = 0; // num_inner + num_bdr
    int num_ghost = 0;
    int num_total = 0; // num_owned + num_ghost

    std::vector<double> node_coords;    // xyz interleaved, 3 * num_total
    std::vector<int> local_to_global;   // local → global nodeTag
    std::map<int, int> global_to_local; // global nodeTag → local

    // 要素配置: [owned | ghost]
    //  owned : 自パーティションの体積エンティティに属する要素
    //  ghost : $GhostElements で自パーティションがゴースト先の要素
    int num_owned_elems = 0;
    int num_ghost_elems = 0;
    int num_total_elems = 0;
    std::vector<int> elem_nodes; // Tet10: 10 * num_total_elems

    // 隣接パーティション情報
    std::vector<CommNeighbor> neighbors;

    // ポインタアクセス
    double *coords_ptr() { return node_coords.data(); }
    int *elem_ptr() { return elem_nodes.data(); }
    const double *coords_ptr() const { return node_coords.data(); }
    const int *elem_ptr() const { return elem_nodes.data(); }
    int num_neighbors() const { return static_cast<int>(neighbors.size()); }
};

FEMMesh read_msh(const std::string &filepath, int partition);
void print_mesh_info(const FEMMesh &mesh);
int get_local_id(const FEMMesh &mesh, int global_id);

#endif // MSH_READER_HPP
