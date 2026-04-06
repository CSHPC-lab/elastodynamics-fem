#!/usr/bin/env python3
"""Convert column.msh (Gmsh 2.2 format) to column_fistr.msh (FrontISTR format).

Gmsh element types used:
  15 : point (1 node)
   8 : line3 (3-node 2nd-order line)
   9 : tri6  (6-node 2nd-order triangle) -- surface faces
  11 : tet10 (10-node 2nd-order tetrahedron) -- volume

Geometric tags for surface faces:
  1 : x=0  face  -> FIX group (fixed end)
  2 : x=10 face  -> free end  (CL1 = node at centroid)
  3 : y=0  face
  4 : y=1  face
  5 : z=0  face
  6 : z=1  face

Node ordering for tet10:
  Gmsh type 11 and FrontISTR type 342 share the same 10-node ordering:
    corners  : n0, n1, n2, n3
    midpoints: n4=mid(n0,n1), n5=mid(n1,n2), n6=mid(n0,n2),
               n7=mid(n0,n3), n8=mid(n1,n3), n9=mid(n2,n3)
  Therefore no reordering is needed when copying element connectivity.

python3 convert_gmsh_to_fistr.py
"""

import sys
import os


# ---------------------------------------------------------------------------
# Material / amplitude parameters
# ---------------------------------------------------------------------------
YOUNG      = 4000.0
POISSON    = 0.3
DENSITY    = 1.0e-9
AMP_DT     = 2.5e-6   # amplitude table time step [s]
AMP_END    = 5.0e-3   # amplitude table end time [s]


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
            if line == '$Nodes':
                section = 'nodes'
                continue
            if line == '$EndNodes':
                section = None
                continue
            if line == '$Elements':
                section = 'elements'
                continue
            if line == '$EndElements':
                section = None
                continue
            if line.startswith('$'):
                section = None
                continue

            if not line:
                continue
            parts = line.split()

            if section == 'nodes':
                if len(parts) == 4:
                    try:
                        nid = int(parts[0])
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        nodes[nid] = (x, y, z)
                    except ValueError:
                        pass

            elif section == 'elements':
                if len(parts) < 4:
                    continue
                try:
                    eid   = int(parts[0])
                    etype = int(parts[1])
                    ntags = int(parts[2])
                    # tags[0] = physical tag, tags[1] = geometric tag
                    tags     = [int(parts[3 + i]) for i in range(ntags)]
                    geom_tag = tags[1] if ntags >= 2 else 0
                    node_ids = [int(parts[3 + ntags + i])
                                for i in range(len(parts) - 3 - ntags)]

                    if etype == 11:   # tet10
                        if len(node_ids) != 10:
                            raise ValueError(f'tet10 element {eid} has {len(node_ids)} nodes')
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
                            raise ValueError(f'tri6 element {eid} has {len(node_ids)} nodes')
                        tri6_by_geomtag.setdefault(geom_tag, []).append((eid, node_ids))

                except ValueError as e:
                    print(f'  Warning: {e}', file=sys.stderr)

    return nodes, tet10_elems, tri6_by_geomtag


def face_nodes(tri6_list):
    """Unique node IDs from a list of tri6 elements, sorted ascending."""
    return sorted({n for _, nids in tri6_list for n in nids})


def find_centroid_node(nodes, tri6_list):
    """Return the node ID closest to the centroid of the face.

    The centroid of the free-end face (x=10, y in [0,1], z in [0,1])
    is (10, 0.5, 0.5).  Node 1 in column.msh is placed exactly there,
    but we find it dynamically so the script stays general.
    """
    face_nids = face_nodes(tri6_list)
    xs = [nodes[n][0] for n in face_nids]
    ys = [nodes[n][1] for n in face_nids]
    zs = [nodes[n][2] for n in face_nids]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    cz = sum(zs) / len(zs)

    best_nid  = None
    best_dist = float('inf')
    for nid in face_nids:
        x, y, z = nodes[nid]
        d = (x - cx)**2 + (y - cy)**2 + (z - cz)**2
        if d < best_dist:
            best_dist = d
            best_nid  = nid
    return best_nid


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def fmt_node_coord(v):
    """Format a node coordinate in the same style as REVOCAP output."""
    return f'{v:.8e}'


def fmt_amp_time(t):
    """Format amplitude time as 5-digit scientific notation (e.g. 1.00000E-05)."""
    return f'{t:.5E}'


def write_fistr_msh(output_file, nodes, tet10_elems, fix_nodes, cl1_node):
    with open(output_file, 'w') as f:

        # --- header ---
        f.write('!HEADER\n')
        f.write(' HECMW_MSH File generated by convert_gmsh_to_fistr.py\n')

        # --- nodes ---
        f.write('!NODE\n')
        for nid in sorted(nodes):
            x, y, z = nodes[nid]
            f.write(f'        {nid:6d}, {fmt_node_coord(x)}, '
                    f'{fmt_node_coord(y)}, {fmt_node_coord(z)}\n')

        # --- volume elements ---
        # Gmsh tet10 (type 11) and FrontISTR tet10 (type 342) share the same
        # node ordering, so the connectivity is copied verbatim.
        f.write('!ELEMENT, TYPE=342, EGRP=column_0\n')
        for eid, nids in tet10_elems:
            f.write(f'{eid},' + ','.join(str(n) for n in nids) + '\n')

        # --- material ---
        f.write('!MATERIAL, NAME=M1, ITEM=2\n')
        f.write('!ITEM=1, SUBITEM=2\n')
        f.write(f' {YOUNG}, {POISSON}\n')
        f.write('!ITEM=2, SUBITEM=1\n')
        f.write(f' {DENSITY:.1E}\n')

        # --- section ---
        f.write('!SECTION, TYPE=SOLID, EGRP=column_0, MATERIAL=M1\n')

        # --- FIX node group (x=0 face: all nodes from geom_tag=1 triangles) ---
        f.write('!NGROUP, NGRP=FIX\n')
        for nid in fix_nodes:
            f.write(f'{nid},\n')

        # --- CL1 node group (centroid of free-end face) ---
        f.write('!NGROUP, NGRP=CL1\n')
        f.write(f'{cl1_node},\n')

        # # --- amplitude (constant 1.0 step load) ---
        # f.write('!AMPLITUDE, NAME=AMP1\n')
        # n_steps = round(AMP_END / AMP_DT)
        # for i in range(n_steps + 1):
        #     t = i * AMP_DT
        #     f.write(f'1.00000E+00\t,\t{fmt_amp_time(t)}\n')

        f.write('!END\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    input_file  = os.path.join(os.path.dirname(__file__), 'column.msh')
    output_file = os.path.join(os.path.dirname(__file__), 'column_fistr.msh')

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    print(f'Reading  : {input_file}')
    nodes, tet10_elems, tri6_by_geomtag = parse_gmsh(input_file)

    print(f'  nodes          : {len(nodes)}')
    print(f'  tet10 elements : {len(tet10_elems)}')
    print(f'  tri6 by geom   : { {k: len(v) for k, v in sorted(tri6_by_geomtag.items())} }')

    # --- FIX group: all nodes on x=0 face (geom_tag = 1) ---
    if 1 not in tri6_by_geomtag:
        sys.exit('ERROR: no tri6 elements with geom_tag=1 found (expected x=0 face)')
    fix_nodes = face_nodes(tri6_by_geomtag[1])
    print(f'  FIX group      : {len(fix_nodes)} nodes (geom_tag=1)')

    # --- CL1: centroid node of x=10 face (geom_tag = 2) ---
    if 2 not in tri6_by_geomtag:
        sys.exit('ERROR: no tri6 elements with geom_tag=2 found (expected x=10 face)')
    cl1_node = find_centroid_node(nodes, tri6_by_geomtag[2])
    cx, cy, cz = nodes[cl1_node]
    print(f'  CL1 node       : {cl1_node}  ({cx}, {cy}, {cz})')

    print(f'Writing  : {output_file}')
    write_fistr_msh(output_file, nodes, tet10_elems, fix_nodes, cl1_node)
    print('Done.')


if __name__ == '__main__':
    main()
