#!/usr/bin/env python3
"""Convert column.msh (Gmsh 2.2 format) to column_fistr.msh (FrontISTR format).

Gmsh element types used:
  15 : point (1 node)
   8 : line3 (3-node 2nd-order line)
   9 : tri6  (6-node 2nd-order triangle) -- surface faces
  11 : tet10 (10-node 2nd-order tetrahedron) -- volume

Geometric tags for surface faces (addBox(0,0,0, 10,10,100)):
  1 : x=0   face
  2 : x=10  face
  3 : y=0   face
  4 : y=10  face
  5 : z=0   face  -> FIX group (fixed end)
  6 : z=100 face  -> FORCE_NODE (top center, node at centroid)

Node ordering for tet10:
  Gmsh type 11 and FrontISTR type 342 share the same 10-node ordering,
  except for midpoints n4/n5/n6 and n8/n9 which differ.

python3 convert_gmsh_to_fistr.py
"""

import sys
import os

# ---------------------------------------------------------------------------
# Material / load parameters — config.txt と一致させる
# ---------------------------------------------------------------------------
YOUNG = 160000000.0 / 3.0  # E [Pa]  (c1=200, c2=100, rho=2000 より)
POISSON = 1.0 / 3.0  # ν
DENSITY = 2000.0  # rho [kg/m^3]
DURATION = 1000.0  # シミュレーション総時間 [s]  (config.txt: duration)
FORCE_START = -10800000.0  # 点荷重の開始値 [N]          (config.txt: force_start)
FORCE_END = -11200000.0  # 点荷重の終端値 [N]          (config.txt: force_end)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_gmsh(filename):
    """Return (nodes, tet10_elems, tri6_by_geomtag).

    nodes          : dict  id -> (x, y, z)
    tet10_elems    : list  of (eid, [n0..n9])  -- preserves file order
    tri6_by_geomtag: dict  geom_tag -> list of (eid, [n0..n5])
    """
    nodes = {}
    tet10_elems = []
    tri6_by_geomtag = {}

    section = None
    with open(filename) as f:
        for raw in f:
            line = raw.strip()
            if line == "$Nodes":
                section = "nodes"
                continue
            if line == "$EndNodes":
                section = None
                continue
            if line == "$Elements":
                section = "elements"
                continue
            if line == "$EndElements":
                section = None
                continue
            if line.startswith("$"):
                section = None
                continue

            if not line:
                continue
            parts = line.split()

            if section == "nodes":
                if len(parts) == 4:
                    try:
                        nid = int(parts[0])
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        nodes[nid] = (x, y, z)
                    except ValueError:
                        pass

            elif section == "elements":
                if len(parts) < 4:
                    continue
                try:
                    eid = int(parts[0])
                    etype = int(parts[1])
                    ntags = int(parts[2])
                    # tags[0] = physical tag, tags[1] = geometric tag
                    tags = [int(parts[3 + i]) for i in range(ntags)]
                    geom_tag = tags[1] if ntags >= 2 else 0
                    node_ids = [
                        int(parts[3 + ntags + i]) for i in range(len(parts) - 3 - ntags)
                    ]

                    if etype == 11:  # tet10
                        if len(node_ids) != 10:
                            raise ValueError(
                                f"tet10 element {eid} has {len(node_ids)} nodes"
                            )
                        tmp = node_ids[4]
                        node_ids[4] = node_ids[5]
                        node_ids[5] = node_ids[6]
                        node_ids[6] = tmp
                        tmp = node_ids[8]
                        node_ids[8] = node_ids[9]
                        node_ids[9] = tmp
                        tet10_elems.append((eid, node_ids))

                    elif etype == 9:  # tri6
                        if len(node_ids) != 6:
                            raise ValueError(
                                f"tri6 element {eid} has {len(node_ids)} nodes"
                            )
                        tri6_by_geomtag.setdefault(geom_tag, []).append((eid, node_ids))

                except ValueError as e:
                    print(f"  Warning: {e}", file=sys.stderr)

    return nodes, tet10_elems, tri6_by_geomtag


def face_nodes(tri6_list):
    """Unique node IDs from a list of tri6 elements, sorted ascending."""
    return sorted({n for _, nids in tri6_list for n in nids})


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def fmt_node_coord(v):
    """Format a node coordinate in the same style as REVOCAP output."""
    return f"{v:.8e}"


def write_fistr_msh(output_file, nodes, tet10_elems, fix_nodes, force_node):
    with open(output_file, "w") as f:

        # --- header ---
        f.write("!HEADER\n")
        f.write(" HECMW_MSH File generated by convert_gmsh_to_fistr.py\n")

        # --- nodes ---
        f.write("!NODE\n")
        for nid in sorted(nodes):
            x, y, z = nodes[nid]
            f.write(
                f"        {nid:6d}, {fmt_node_coord(x)}, "
                f"{fmt_node_coord(y)}, {fmt_node_coord(z)}\n"
            )

        # --- volume elements ---
        f.write("!ELEMENT, TYPE=342, EGRP=column_0\n")
        for eid, nids in tet10_elems:
            f.write(f"{eid}," + ",".join(str(n) for n in nids) + "\n")

        # --- material ---
        f.write("!MATERIAL, NAME=M1, ITEM=2\n")
        f.write("!ITEM=1, SUBITEM=2\n")
        f.write(f" {YOUNG}, {POISSON}\n")
        f.write("!ITEM=2, SUBITEM=1\n")
        f.write(f" {DENSITY:.1E}\n")

        # --- section ---
        f.write("!SECTION, TYPE=SOLID, EGRP=column_0, MATERIAL=M1\n")

        # --- FIX node group (z=0 face: geom_tag=5) ---
        f.write("!NGROUP, NGRP=FIX\n")
        for nid in fix_nodes:
            f.write(f"{nid},\n")

        # --- FORCE_NODE group (z=100 top face centroid: geom_tag=6) ---
        f.write("!NGROUP, NGRP=FORCE_NODE\n")
        f.write(f"{force_node},\n")

        # --- amplitude: FORCE_START -> FORCE_END over 0 -> DURATION [s] ---
        f.write("!AMPLITUDE, NAME=AMP1\n")
        f.write(f"{FORCE_START:.5E}\t,\t0.00000E+00\n")
        f.write(f"{FORCE_END:.5E}\t,\t{DURATION:.5E}\n")

        f.write("!END\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    input_file = os.path.join(os.path.dirname(__file__), "column.msh")
    output_file = os.path.join(os.path.dirname(__file__), "column_fistr.msh")

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    print(f"Reading  : {input_file}")
    nodes, tet10_elems, tri6_by_geomtag = parse_gmsh(input_file)

    print(f"  nodes          : {len(nodes)}")
    print(f"  tet10 elements : {len(tet10_elems)}")
    print(
        f"  tri6 by geom   : { {k: len(v) for k, v in sorted(tri6_by_geomtag.items())} }"
    )

    # --- FIX group: all nodes on z=0 face (geom_tag=5) ---
    if 5 not in tri6_by_geomtag:
        sys.exit("ERROR: no tri6 elements with geom_tag=5 found (expected z=0 face)")
    fix_nodes = face_nodes(tri6_by_geomtag[5])
    print(f"  FIX group      : {len(fix_nodes)} nodes (geom_tag=5, z=0)")

    # --- FORCE_NODE: config.txt の force_node に合わせて固定 ---
    force_node = 2
    fx, fy, fz = nodes[force_node]
    print(f"  FORCE_NODE     : {force_node}  ({fx}, {fy}, {fz})")

    print(f"Writing  : {output_file}")
    write_fistr_msh(output_file, nodes, tet10_elems, fix_nodes, force_node)
    print("Done.")


if __name__ == "__main__":
    main()
