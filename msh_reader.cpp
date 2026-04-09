#include "msh_reader.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <array>

// ── 内部データ構造 ──────────────────────────────────────

struct NodeInfo
{
    std::array<double, 3> coords;
    std::pair<int, int> entity; // (entityDim, entityTag)
};

struct ElemInfo
{
    int tag;
    int entity_tag;         // 体積エンティティのタグ
    std::vector<int> nodes; // Tet10: 10節点のグローバルID
};

struct GhostInfo
{
    int owner;            // 所有パーティション (1-indexed)
    std::set<int> ghosts; // ゴースト先パーティション群
};

// ════════════════════════════════════════════════════════
//  MSH 4.1 パーサ
// ════════════════════════════════════════════════════════

FEMMesh read_msh(const std::string &filepath, int partition)
{
    std::ifstream ifs(filepath);
    if (!ifs)
        throw std::runtime_error("Cannot open: " + filepath);

    FEMMesh mesh;
    mesh.my_partition = partition;

    // --- 全体データ ---
    // エンティティ (dim, tag) → 所属パーティション集合
    std::map<std::pair<int, int>, std::set<int>> entity_partitions;
    std::map<int, NodeInfo> all_nodes;   // nodeTag → 情報
    std::vector<ElemInfo> all_elems;     // Tet10要素のみ
    std::map<int, GhostInfo> ghost_info; // elemTag → ゴースト情報

    std::string line;
    while (std::getline(ifs, line))
    {
        // ─────────────────────────────────────────────
        // $PartitionedEntities: dim 0-2 のパーティション情報
        // ─────────────────────────────────────────────
        if (line == "$PartitionedEntities")
        {
            // ヘッダ: numPartitions
            std::getline(ifs, line);

            // ゴーストエンティティ
            std::getline(ifs, line);
            int num_ghost_ent = std::stoi(line);
            for (int i = 0; i < num_ghost_ent; ++i)
                std::getline(ifs, line);

            // エンティティ数: points curves surfaces volumes
            std::getline(ifs, line);
            std::istringstream cnt(line);
            int n_pts, n_curves, n_surfs, n_vols;
            cnt >> n_pts >> n_curves >> n_surfs >> n_vols;

            // Points: tag parentDim parentTag nParts parts... X Y Z nPhys phys...
            for (int i = 0; i < n_pts; ++i)
            {
                std::getline(ifs, line);
                std::istringstream iss(line);
                int tag, pDim, pTag, nParts;
                iss >> tag >> pDim >> pTag >> nParts;
                std::set<int> parts;
                for (int j = 0; j < nParts; ++j)
                {
                    int p;
                    iss >> p;
                    parts.insert(p);
                }
                entity_partitions[{0, tag}] = parts;
            }

            // Curves: tag parentDim parentTag nParts parts... bbox(6) nPhys phys... nBnd bnd...
            for (int i = 0; i < n_curves; ++i)
            {
                std::getline(ifs, line);
                std::istringstream iss(line);
                int tag, pDim, pTag, nParts;
                iss >> tag >> pDim >> pTag >> nParts;
                std::set<int> parts;
                for (int j = 0; j < nParts; ++j)
                {
                    int p;
                    iss >> p;
                    parts.insert(p);
                }
                entity_partitions[{1, tag}] = parts;
            }

            // Surfaces: 同様
            for (int i = 0; i < n_surfs; ++i)
            {
                std::getline(ifs, line);
                std::istringstream iss(line);
                int tag, pDim, pTag, nParts;
                iss >> tag >> pDim >> pTag >> nParts;
                std::set<int> parts;
                for (int j = 0; j < nParts; ++j)
                {
                    int p;
                    iss >> p;
                    parts.insert(p);
                }
                entity_partitions[{2, tag}] = parts;
            }

            // Volumes: 同様
            for (int i = 0; i < n_vols; ++i)
            {
                std::getline(ifs, line);
                std::istringstream iss(line);
                int tag, pDim, pTag, nParts;
                iss >> tag >> pDim >> pTag >> nParts;
                std::set<int> parts;
                for (int j = 0; j < nParts; ++j)
                {
                    int p;
                    iss >> p;
                    parts.insert(p);
                }
                entity_partitions[{3, tag}] = parts;
            }
        }

        // ─────────────────────────────────────────────
        // $Nodes
        // ─────────────────────────────────────────────
        else if (line == "$Nodes")
        {
            std::getline(ifs, line);
            std::istringstream hdr(line);
            int n_blocks, n_nodes, min_tag, max_tag;
            hdr >> n_blocks >> n_nodes >> min_tag >> max_tag;

            for (int b = 0; b < n_blocks; ++b)
            {
                std::getline(ifs, line);
                std::istringstream bh(line);
                int e_dim, e_tag, parametric, n_in_block;
                bh >> e_dim >> e_tag >> parametric >> n_in_block;

                // タグ行を先に全部読む
                std::vector<int> tags(n_in_block);
                for (int i = 0; i < n_in_block; ++i)
                {
                    std::getline(ifs, line);
                    tags[i] = std::stoi(line);
                }
                // 座標行
                for (int i = 0; i < n_in_block; ++i)
                {
                    std::getline(ifs, line);
                    std::istringstream cs(line);
                    double x, y, z;
                    cs >> x >> y >> z;
                    all_nodes[tags[i]] = {{x, y, z}, {e_dim, e_tag}};
                }
            }
        }

        // ─────────────────────────────────────────────
        // $Elements (Tet10 = type 11 のみ収集)
        // ─────────────────────────────────────────────
        else if (line == "$Elements")
        {
            std::getline(ifs, line);
            std::istringstream hdr(line);
            int n_blocks, n_elems, min_tag, max_tag;
            hdr >> n_blocks >> n_elems >> min_tag >> max_tag;

            for (int b = 0; b < n_blocks; ++b)
            {
                std::getline(ifs, line);
                std::istringstream bh(line);
                int e_dim, e_tag, elem_type, n_in_block;
                bh >> e_dim >> e_tag >> elem_type >> n_in_block;

                for (int i = 0; i < n_in_block; ++i)
                {
                    std::getline(ifs, line);
                    if (elem_type != 11) // Tet10以外はスキップ
                        continue;

                    std::istringstream es(line);
                    ElemInfo ei;
                    es >> ei.tag;
                    ei.entity_tag = e_tag;
                    ei.nodes.resize(10);
                    for (int j = 0; j < 10; ++j)
                        es >> ei.nodes[j];
                    // 節点の順序が最後だけ入れ替わっているので戻す
                    int tmp_id = ei.nodes[8];
                    ei.nodes[8] = ei.nodes[9];
                    ei.nodes[9] = tmp_id;
                    all_elems.push_back(std::move(ei));
                }
            }
        }

        // ─────────────────────────────────────────────
        // $GhostElements
        // ─────────────────────────────────────────────
        else if (line == "$GhostElements")
        {
            std::getline(ifs, line);
            int n_ghost = std::stoi(line);
            for (int i = 0; i < n_ghost; ++i)
            {
                std::getline(ifs, line);
                std::istringstream gs(line);
                int e_tag, owner, n_ghost_parts;
                gs >> e_tag >> owner >> n_ghost_parts;
                GhostInfo gi;
                gi.owner = owner;
                for (int j = 0; j < n_ghost_parts; ++j)
                {
                    int gp;
                    gs >> gp;
                    gi.ghosts.insert(gp);
                }
                ghost_info[e_tag] = gi;
            }
        }
    }

    // ════════════════════════════════════════════════════
    //  1. 要素の所有者から「節点の真の所有者」を確定
    // ════════════════════════════════════════════════════
    std::map<int, std::set<int>> node_element_owners;
    for (const auto &e : all_elems)
    {
        auto vit = entity_partitions.find({3, e.entity_tag});
        if (vit != entity_partitions.end() && !vit->second.empty())
        {
            int e_owner = *vit->second.begin();
            for (int n : e.nodes)
                node_element_owners[n].insert(e_owner);
        }
    }

    std::map<int, int> node_owner;
    for (const auto &[n, owners] : node_element_owners)
    {
        node_owner[n] = *owners.begin(); // 若いIDが勝者
    }

    // ════════════════════════════════════════════════════
    //  2. 究極の断捨離 ＆ 通信ニーズの逆算
    // ════════════════════════════════════════════════════
    // Gmshの指示を無視し、「誰がその要素を手元に残すか(keepers)」を論理的に判定する

    std::map<int, std::set<int>> node_sharers; // 節点 → それを必要とする全パーティション
    std::vector<const ElemInfo *> owned_elems, ghost_elems;

    for (const auto &e : all_elems)
    {
        int e_owner = -1;
        auto vit = entity_partitions.find({3, e.entity_tag});
        if (vit != entity_partitions.end() && !vit->second.empty())
            e_owner = *vit->second.begin();

        if (e_owner == -1)
            continue;

        // 【判定】この要素は、誰の手元に残るか？ (自分の担当節点を含むパーティション全員)
        std::set<int> keepers;
        for (int n : e.nodes)
        {
            keepers.insert(node_owner[n]);
        }

        // 要素を残すパーティションは、当然その要素の全節点を必要とする
        for (int n : e.nodes)
        {
            for (int p : keepers)
                node_sharers[n].insert(p);
        }

        // 現在の自分のパーティションにとって、この要素は必要か？
        if (keepers.count(partition))
        {
            if (e_owner == partition)
                owned_elems.push_back(&e);
            else
                ghost_elems.push_back(&e); // Gmshの指示を無視し、自力でゴースト認定！
        }
    }

    // ════════════════════════════════════════════════════
    //  3. 節点を inner / bdr / ghost に分類しローカルIDを付与
    // ════════════════════════════════════════════════════
    std::set<int> local_node_set;
    for (auto *e : owned_elems)
        for (int n : e->nodes)
            local_node_set.insert(n);
    for (auto *e : ghost_elems)
        for (int n : e->nodes)
            local_node_set.insert(n);

    std::set<int> pure_inner_set;
    std::set<int> bdr_set;
    std::map<int, std::set<int>> ghost_by_owner;

    for (int n : local_node_set)
    {
        int true_owner = node_owner[n];
        if (true_owner == partition)
        {
            // 必要とするのが自分だけなら inner、他人も必要なら bdr(送信)
            if (node_sharers[n].size() == 1)
                pure_inner_set.insert(n);
            else
                bdr_set.insert(n);
        }
        else
        {
            ghost_by_owner[true_owner].insert(n);
        }
    }

    int lid = 0;
    auto add_node = [&](int gid)
    {
        mesh.global_to_local[gid] = lid;
        mesh.local_to_global.push_back(gid);
        auto &c = all_nodes.at(gid).coords;
        mesh.node_coords.push_back(c[0]);
        mesh.node_coords.push_back(c[1]);
        mesh.node_coords.push_back(c[2]);
        lid++;
    };

    // 順序を確定
    for (int n : pure_inner_set)
        add_node(n);
    mesh.num_inner = lid;

    for (int n : bdr_set)
        add_node(n);
    mesh.num_bdr = lid - mesh.num_inner;
    mesh.num_owned = lid;

    for (auto &[owner, nodes] : ghost_by_owner)
        for (int n : nodes)
            add_node(n);
    mesh.num_ghost = lid - mesh.num_owned;
    mesh.num_total = lid;

    // ════════════════════════════════════════════════════
    //  4. 要素の節点リストをローカルIDに変換
    // ════════════════════════════════════════════════════
    mesh.num_owned_elems = static_cast<int>(owned_elems.size());
    mesh.num_ghost_elems = static_cast<int>(ghost_elems.size());
    mesh.num_total_elems = mesh.num_owned_elems + mesh.num_ghost_elems;
    mesh.elem_nodes.reserve(mesh.num_total_elems * 10);

    for (auto *e : owned_elems)
        for (int n : e->nodes)
            mesh.elem_nodes.push_back(mesh.global_to_local.at(n));
    for (auto *e : ghost_elems)
        for (int n : e->nodes)
            mesh.elem_nodes.push_back(mesh.global_to_local.at(n));

    // ════════════════════════════════════════════════════
    //  5. 通信テーブルの構築
    // ════════════════════════════════════════════════════
    std::set<int> all_nbrs;
    for (auto &[owner, nodes] : ghost_by_owner)
        all_nbrs.insert(owner);

    std::map<int, std::set<int>> send_map;
    for (int n : bdr_set)
    {
        // ここを node_partitions ではなく node_sharers を参照するように修正
        for (int p : node_sharers[n])
        {
            if (p != partition)
            {
                send_map[p].insert(n);
                all_nbrs.insert(p);
            }
        }
    }

    for (int nbr : all_nbrs)
    {
        CommNeighbor cn;
        cn.partition_id = nbr;

        if (send_map.count(nbr))
            for (int g : send_map[nbr])
                cn.send_nodes.push_back(mesh.global_to_local.at(g));

        if (ghost_by_owner.count(nbr) && !ghost_by_owner[nbr].empty())
        {
            cn.recv_count = static_cast<int>(ghost_by_owner[nbr].size());
            cn.recv_start = mesh.global_to_local.at(*ghost_by_owner[nbr].begin());
        }
        else
        {
            cn.recv_count = 0;
            cn.recv_start = -1;
        }

        mesh.neighbors.push_back(cn);
    }

    return mesh;
}

// ── デバッグ出力 ─────────────────────────────────────────

void print_mesh_info(const FEMMesh &mesh)
{
    std::cout << "=== Partition " << mesh.my_partition << " ===" << std::endl;
    std::cout << "  Nodes:" << std::endl;
    std::cout << "    inner: " << mesh.num_inner << std::endl;
    std::cout << "    bdr  : " << mesh.num_bdr << std::endl;
    std::cout << "    owned: " << mesh.num_owned << " (inner + bdr)" << std::endl;
    std::cout << "    ghost: " << mesh.num_ghost << std::endl;
    std::cout << "    total: " << mesh.num_total << std::endl;
    std::cout << "  Elements:" << std::endl;
    std::cout << "    owned: " << mesh.num_owned_elems << std::endl;
    std::cout << "    ghost: " << mesh.num_ghost_elems << std::endl;
    std::cout << "    total: " << mesh.num_total_elems << std::endl;
    std::cout << "  Neighbors: " << mesh.neighbors.size() << std::endl;
    for (const auto &cn : mesh.neighbors)
    {
        std::cout << "    -> partition " << cn.partition_id
                  << "  send " << cn.send_nodes.size()
                  << " nodes, recv " << cn.recv_count
                  << " nodes (start: " << cn.recv_start << ")" << std::endl;
    }
    std::cout << std::endl;
}

// グローバル節点番号を入力としてローカル節点番号を返す関数(存在しない場合は -1)
int get_local_id(const FEMMesh &mesh, int global_id)
{
    auto it = mesh.global_to_local.find(global_id);
    return (it != mesh.global_to_local.end()) ? it->second : -1;
}