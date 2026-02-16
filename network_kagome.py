""" Build network and manage all its properties.

This module contains classes and functions connected with building Delaunay
network, setting boundary condition on it and evolving.

Notable classes
-------
Graph(nx.graph.Graph)
    container for network and its properties

Notable functions
-------
build_delaunay_net(SimInputData) -> Graph
    build Delaunay network with parameters from config

TO DO:
fix build_delaunay_net (better choose input/output nodes, fix intersections -
sometimes edges cross, like 1-2 in a given network), comment it
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from collections import defaultdict
from scipy.stats import truncnorm
import networkx as nx
import numpy as np
import scipy.sparse as spr
import scipy.spatial as spt
from itertools import combinations

from config import SimInputData
if TYPE_CHECKING:
    from incidence import Incidence


import numpy as np
import networkx as nx
import scipy.spatial as spt


def hex_packing_points(n_rows: int, n_cols: int, a: float = 1.0,
                       remove_midline: bool = False, cutoff_x_frac: float = 0.2):
    """
    Hexagonal packing (triangular lattice) of cylinder centers.

    Parameters
    ----------
    n_rows, n_cols : int
        Number of rows (y direction) and columns (x direction).
    a : float
        Nearest-neighbor spacing of centers.
    remove_midline : bool
        Optional: remove a single midline row segment near the inlet (like your diamond_lattice_graph did).
        This can be used to create a splitter / notch.
    cutoff_x_frac : float
        Midline points are removed for x < cutoff_x_frac * domain_width.

    Returns
    -------
    pts : (N,2) ndarray
        Center coordinates.
    """
    dy = np.sqrt(3) / 2 * a
    pts = []

    # "midline" in physical y
    mid_row = int(round((n_rows - 1) / 2))
    mid_y = mid_row * dy
    cutoff_x = cutoff_x_frac * (n_cols * a)

    for i in range(n_rows):
        y = i * dy
        x_shift = 0.5 * a if (i % 2 == 1) else 0.0
        for j in range(n_cols):
            x = j * a + x_shift

            if remove_midline and np.isclose(y, mid_y) and (x < cutoff_x):
                continue

            pts.append((x, y))

    return np.asarray(pts, dtype=float)


def kagome_from_delaunay(points: np.ndarray,
                         max_edge_ratio: float = 1.8,
                         store_maps: bool = True):
    """
    Build throat-centered Kagome network from Delaunay triangulation of cylinder centers.

    Nodes  : Delaunay edges (unordered pair of center indices).
    Edges  : between throat-nodes that belong to the same Delaunay triangle.

    Parameters
    ----------
    points : (N,2) ndarray
        Cylinder center coordinates.
    max_edge_ratio : float
        Skip Delaunay triangles that contain any side longer than max_edge_ratio * a0,
        where a0 is estimated from the lower-percentile neighbor distance.
        Useful if you remove centers (prevents "bridging" triangles across gaps).
    store_maps : bool
        If True, store helpful metadata on nodes/edges.

    Returns
    -------
    Gk : nx.Graph
        Kagome graph with node attribute 'pos' = (x,y).
    """
    tri = spt.Delaunay(points)
    simplices = tri.simplices  # (nT, 3) indices into points

    # estimate a0 as a "typical" nearest-neighbor distance using Delaunay edges
    # (robust enough for regular arrays; still ok for mild disorder)
    delaunay_edges = set()
    for (i, j, k) in simplices:
        delaunay_edges.add(tuple(sorted((i, j))))
        delaunay_edges.add(tuple(sorted((j, k))))
        delaunay_edges.add(tuple(sorted((k, i))))

    edge_lens = []
    for (i, j) in delaunay_edges:
        edge_lens.append(np.linalg.norm(points[i] - points[j]))
    edge_lens = np.asarray(edge_lens)
    a0 = np.percentile(edge_lens, 10) if edge_lens.size else 1.0  # "typical" short edge

    # map center-pair -> kagome node id
    pair_to_node = {}
    node_pos = {}
    next_id = 0

    def get_node_for_pair(i, j):
        nonlocal next_id
        key = (i, j) if i < j else (j, i)
        if key in pair_to_node:
            return pair_to_node[key]
        nid = next_id
        next_id += 1
        pair_to_node[key] = nid
        p = 0.5 * (points[key[0]] + points[key[1]])
        node_pos[nid] = (float(p[0]), float(p[1]))
        return nid

    Gk = nx.Graph()

    # create nodes + edges triangle-by-triangle
    for (i, j, k) in simplices:
        # triangle side lengths
        lij = np.linalg.norm(points[i] - points[j])
        ljk = np.linalg.norm(points[j] - points[k])
        lki = np.linalg.norm(points[k] - points[i])

        # prune "weird" big triangles (typically only appear if you removed points)
        if max(lij, ljk, lki) > max_edge_ratio * a0:
            continue

        n_ij = get_node_for_pair(i, j)
        n_jk = get_node_for_pair(j, k)
        n_ki = get_node_for_pair(k, i)

        # add nodes (idempotent)
        if n_ij not in Gk:
            Gk.add_node(n_ij, pos=node_pos[n_ij])
            if store_maps:
                Gk.nodes[n_ij]["center_pair"] = (min(i, j), max(i, j))
        if n_jk not in Gk:
            Gk.add_node(n_jk, pos=node_pos[n_jk])
            if store_maps:
                Gk.nodes[n_jk]["center_pair"] = (min(j, k), max(j, k))
        if n_ki not in Gk:
            Gk.add_node(n_ki, pos=node_pos[n_ki])
            if store_maps:
                Gk.nodes[n_ki]["center_pair"] = (min(k, i), max(k, i))

        # connect the three throat-nodes around this pore triangle
        Gk.add_edge(n_ij, n_jk)
        Gk.add_edge(n_jk, n_ki)
        Gk.add_edge(n_ki, n_ij)

        if store_maps:
            pore = tuple(sorted((int(i), int(j), int(k))))
            # annotate edges with which pore they came from (optional, last-wins if shared)
            Gk.edges[n_ij, n_jk]["pore_triangle"] = pore
            Gk.edges[n_jk, n_ki]["pore_triangle"] = pore
            Gk.edges[n_ki, n_ij]["pore_triangle"] = pore

    return Gk

class Graph(nx.graph.Graph):
    """ Contains network and all its properties.

    This class is derived from networkx Graph and contains all information
    abount the network and its properties.

    Attributes
    -------
    in_nodes : list
        list of inlet nodes
    out_nodes : list
        list of outlet nodes
    boundary_edges : list
        list of edges assuring PBC
    triangles : list
        list of positions of triangle centers
    """
    in_nodes: np.ndarray
    out_nodes: np.ndarray
    in_vec: np.ndarray
    out_vec: np.ndarray
    in_vec_a: np.ndarray
    in_vec_b: np.ndarray
    zero_nodes = []
    boundary_edges = []
    boundary_nodes = []
    merged_triangles = []
    vol_nodes: np.ndarray

    def __init__(self):
        nx.graph.Graph.__init__(self)
        self.in_nodes = []
        self.out_nodes = []
        

    def update_network(self, inc:Incidence, edges: Edges) -> None:
        """ Update diameters and flow in the graph.

        Parameters
        -------
        edges : Edges class object
            all edges in network and their parameters
            edge_list - array of tuples (n1, n2) with n1, n2 being nodes
            connected by edge with a given index
            diams - diameters of edges
            flow - flow in edges
        """
        nx.set_edge_attributes(self, dict(zip(edges.edge_list_draw, edges.diams)), \
            'd')
        nx.set_edge_attributes(self, dict(zip(edges.edge_list_draw, edges.flow)), \
            'q')

def find_node(graph: Graph, pos: tuple[float, float]) -> int:
    """ Find node in the graph closest to the given position.

    Parameters
    -------
    graph : Graph class object
        network and all its properties

    pos : tuple
        approximate position of the wanted node

    Returns
    -------
    n_min : int
        index of the node closest to the given position
    """
    def r_squared(node):
        x, y = graph.nodes[node]['pos']
        r_sqr = (x - pos[0]) ** 2 + (y - pos[1]) ** 2
        return r_sqr
    r_min = len(graph.nodes())
    n_min = 0
    for node in graph.nodes():
        r = r_squared(node)
        if r < r_min:
            r_min = r
            n_min = node
    return n_min

def set_geometry(sid: SimInputData, graph: Graph) -> None:
    """ Set input and output nodes based on network geometry.

    Parameters
    -------
    sid : SimInputData class object
        all config parameters of the simulation
        n - network size
        nsq - number of nodes
        geo - network geometry
        in_nodes_own - position of inlet nodes in custom geometry
        out_nodes_own - position of outlet nodes in custom geometry

    graph : Graph class object
        network and all its properties
        in_nodes - list of inlet nodes
        out_nodes - list of outlet nodes
    """
    # rectangular geometry - nodes on the left side are the inlet and nodes on
    # the right side are the outlet
    if sid.geo == 'rect':
        graph.in_nodes = np.arange(0, sid.n, 1)
        graph.out_nodes = np.arange(sid.n * (sid.n - 1), sid.nsq, 1)
        graph.in_vec = np.concatenate((np.ones(sid.n), np.zeros(sid.n * (sid.n - 1))))
        graph.out_vec = np.concatenate((np.zeros(sid.n * (sid.n - 1)), np.ones(sid.n)))
    # own geometry - inlet and outlet nodes are found based on the positions
    # given in config
    elif sid.geo == 'own':
        in_nodes_pos = sid.in_nodes_own
        out_nodes_pos = sid.out_nodes_own
        in_nodes = []
        out_nodes = []
        for pos in in_nodes_pos:
            in_nodes.append(find_node(pos))
        for pos in out_nodes_pos:
            out_nodes.append(find_node(pos))
        graph.in_nodes = np.array(in_nodes)
        graph.out_nodes = np.array(out_nodes)
    else:
        raise ValueError(f"Unknown geometry type: {sid.geo}")
    sid.Q_in = sid.qin * 2 * len(graph.in_nodes)

import networkx as nx

def subdivide_edges_into_three(G, pos_attr="pos", prefix="sub", copy_edge_attrs=True):
    """
    Subdivide every edge (u,v) into u--a--b--v by inserting two new nodes.

    Parameters
    ----------
    G : nx.Graph
        Undirected graph with node positions stored in node attribute `pos_attr`.
    pos_attr : str
        Name of node attribute holding (x,y) position.
    prefix : str
        Prefix for new nodes (used only to avoid collisions).
    copy_edge_attrs : bool
        If True, copy original edge attributes onto each new edge segment.

    Returns
    -------
    H : nx.Graph
        Subdivided graph.
    edge_map : dict
        edge_map[ekey] = [(u,a), (a,b), (b,v)] where ekey is canonical undirected edge key.
        Useful for transferring per-edge data or rebuilding matrices later.
    new_nodes : list
        List of newly created node labels.
    """
    if pos_attr is None:
        raise ValueError("pos_attr must be provided (e.g. 'pos').")

    # Helper: canonical undirected edge key (no ordering assumptions on node types)
    def ekey(u, v):
        # frozenset is safe even if node labels are mixed types
        return frozenset((u, v))

    # Start with a copy so node attributes (pos) are preserved
    H = G.copy()

    # We will remove original edges and add new nodes/edges
    original_edges = list(G.edges(data=True))

    # Precompute a collision-safe node generator
    # (uses integers appended until unique)
    k = 0
    def new_node():
        nonlocal k
        while True:
            candidate = (prefix, k)
            k += 1
            if candidate not in H:
                return candidate

    edge_map = {}
    new_nodes = []

    for u, v, attr in original_edges:
        # positions
        pu = H.nodes[u].get(pos_attr, None)
        pv = H.nodes[v].get(pos_attr, None)
        if pu is None or pv is None:
            raise KeyError(f"Missing '{pos_attr}' for node {u} or {v}")

        x_u, y_u = pu
        x_v, y_v = pv

        # create subdivision nodes
        a = new_node()
        b = new_node()

        ax = x_u + (x_v - x_u) * (1.0 / 3.0)
        ay = y_u + (y_v - y_u) * (1.0 / 3.0)
        bx = x_u + (x_v - x_u) * (2.0 / 3.0)
        by = y_u + (y_v - y_u) * (2.0 / 3.0)

        H.add_node(a, **{pos_attr: (ax, ay), "is_sub": True})
        H.add_node(b, **{pos_attr: (bx, by), "is_sub": True})
        new_nodes.extend([a, b])

        # remove the original edge
        if H.has_edge(u, v):
            H.remove_edge(u, v)

        # add the 3 edges, optionally copying attributes
        seg1 = (u, a)
        seg2 = (a, b)
        seg3 = (b, v)

        if copy_edge_attrs:
            # Copy original edge attributes to all segments
            H.add_edge(*seg1, **attr)
            H.add_edge(*seg2, **attr)
            H.add_edge(*seg3, **attr)
        else:
            H.add_edge(*seg1)
            H.add_edge(*seg2)
            H.add_edge(*seg3)

        # store mapping from original edge to new segments (in order u->v)
        edge_map[ekey(u, v)] = [seg1, seg2, seg3]

    return H, edge_map, new_nodes


class Edges():
    """ Contains all data connected with network edges.

    This class is a container for all information about network edges and their
    type in the network graph.

    Attributes
    -------
    diams : numpy ndarray
        diameters of edges

    lens : numpy ndarray
        lengths of edges

    flow : numpy ndarray
        flow in edges

    inlet : numpy ndarray
        edges connected to inlet (vector with ones for inlet edge indices and
        zero otherwise)

    outlet : numpy ndarray
        edges connected to outlet (vector with ones for outlet edge indices and
        zero otherwise)

    edge_list : numpy ndarray
        array of tuples (n1, n2) with n1, n2 being nodes connected by edge with
        a given index

    boundary_list : numpy ndarray
        edges connecting the boundaries (assuring PBC; vector with ones for
        boundary edge indices and zero otherwise); we need them to disinclude
        them for drawing, to make the draw legible

    diams_initial : numpy ndarray
        initial diameters of edges; used for checking how much precipitation
        happened in each part of graph
    """
    diams: np.ndarray
    "diameters of edges"
    lens: np.ndarray
    "lengths of edges"
    flow: np.ndarray
    "flow in edges"
    inlet: np.ndarray
    ("edges connected to inlet (vector with ones for inlet edge indices and \
     zero otherwise)")
    outlet: np.ndarray
    ("edges connected to outlet (vector with ones for outlet edge indices and \
     zero otherwise)")
    edge_list: np.ndarray
    ("array of tuples (n1, n2) with n1, n2 being nodes connected by edge with \
     a given index")
    boundary_list: np.ndarray
    ("edges connecting the boundaries (assuring PBC; vector with ones for \
     boundary edge indices and zero otherwise); we need them to disinclude \
     them for drawing, to make the draw legible")
    merged: np.ndarray
    "edges which were merged and should now be omitted"
    transversed: np.ndarray
    "edges which were merged as transverse"
    hierarchy1: np.ndarray
    def __init__(self, diams, lens, flow, edge_list, edge_list_draw, boundary_list, hierarchy1, hierarchy2):
        self.diams = diams
        self.lens = lens
        self.flow = flow
        self.edge_list = edge_list
        self.edge_list_draw = edge_list_draw
        self.boundary_list = boundary_list
        self.diams_initial = diams
        self.diams_min = diams
        self.merged = np.zeros_like(diams)
        self.transversed = np.zeros_like(diams)
        self.hierarchy0 = hierarchy1
        self.hierarchy1 = hierarchy2


import numpy as np
import networkx as nx
from itertools import combinations


import numpy as np
import networkx as nx
from itertools import combinations

def build_edge_pair_hierarchy_two_ends(G):
    """
    Parameters
    ----------
    G : networkx.Graph
        Undirected pore network graph with node attribute 'pos' (x,y).
        G.edges() defines the global edge ordering.

    Returns
    -------
    node_pairs : dict
        node -> list of (e_idx1, e_idx2, angle) sorted by ascending angle.

    hier0, hier1 : np.ndarray
        Each of shape (n_edges, max_rank_endX).

        For edge e:
          - hier0[e, :] encodes the pair-rank hierarchy at endpoint 0
            (G.edges()[e][0]).
          - hier1[e, :] encodes the hierarchy at endpoint 1
            (G.edges()[e][1]).

        Convention:
          - value 0   : no partner at this rank
          - value > 0 : partner edge has global index (value - 1).
    """

    edges = list(G.edges())
    n_edges = len(edges)

    def canon_edge(u, v):
        return (u, v) if u <= v else (v, u)

    # canonical edge -> global index
    edge_to_idx = {
        canon_edge(u, v): idx
        for idx, (u, v) in enumerate(edges)
    }

    # positions
    pos = nx.get_node_attributes(G, "pos")
    pos_arr = {n: np.asarray(p, dtype=float) for n, p in pos.items()}

    node_pairs = {}

    # per-edge-end temporary storage of rows (variable length)
    rows_end0 = [None] * n_edges
    rows_end1 = [None] * n_edges

    # --- loop over nodes ---
    for node in G.nodes():
        inc_edges = list(G.edges(node))
        deg = len(inc_edges)
        if deg < 2:
            continue

        p_node = pos_arr[node]

        # local info: for each incident edge: global idx, which end (0/1), direction
        e_indices = []
        end_ids = []   # 0 or 1 (which endpoint of the global edge is this node)
        vecs = []

        for (u, v) in inc_edges:
            e_idx = edge_to_idx[canon_edge(u, v)]
            if edges[e_idx][0] == node:
                end_id = 0
                other = edges[e_idx][1]
            else:
                end_id = 1
                other = edges[e_idx][0]

            e_indices.append(e_idx)
            end_ids.append(end_id)
            vecs.append(pos_arr[other] - p_node)

        e_indices = np.array(e_indices, dtype=int)
        end_ids   = np.array(end_ids, dtype=int)
        vecs      = np.stack(vecs, axis=0)  # (deg, 2)

        norms = np.linalg.norm(vecs, axis=1)
        norms[norms == 0.0] = 1.0

        pairs = []  # (local_i, local_j, angle, global_ei, global_ej)

        for i_loc, j_loc in combinations(range(deg), 2):
            v_i = vecs[i_loc]
            v_j = vecs[j_loc]
            denom = norms[i_loc] * norms[j_loc]
            cosang = np.dot(v_i, v_j) / denom
            cosang = np.clip(cosang, -1.0, 1.0)
            angle = np.arccos(cosang)

            e_i = e_indices[i_loc]
            e_j = e_indices[j_loc]
            pairs.append((i_loc, j_loc, angle, e_i, e_j))

        # sort by angle
        pairs.sort(key=lambda x: x[2])

        # store global node_pairs if you want to inspect / debug
        node_pairs[node] = [(e_i, e_j, ang) for (_, _, ang, e_i, e_j) in pairs]

        # build local (deg x n_pairs) hierarchy for this node
        n_pairs = len(pairs)
        hier_local = -np.ones((deg, n_pairs), dtype=int)

        for rank, (i_loc, j_loc, angle, e_i, e_j) in enumerate(pairs):
            hier_local[i_loc, rank] = e_j
            hier_local[j_loc, rank] = e_i

        # store row for each edge-end
        for loc in range(deg):
            e_idx = int(e_indices[loc])
            end_id = int(end_ids[loc])
            row = hier_local[loc, :]  # 1D, length = n_pairs, entries = partner or -1

            if end_id == 0:
                rows_end0[e_idx] = row
            else:
                rows_end1[e_idx] = row

    # --- build rectangular arrays hier0, hier1 (0 = no partner, >0 = idx+1) ---
    max_rank0 = max((len(r) for r in rows_end0 if r is not None), default=0)
    max_rank1 = max((len(r) for r in rows_end1 if r is not None), default=0)

    hier0 = np.zeros((n_edges, max_rank0), dtype=int)
    hier1 = np.zeros((n_edges, max_rank1), dtype=int)

    for e_idx, row in enumerate(rows_end0):
        if row is None:
            continue
        L = min(len(row), max_rank0)
        valid = row[:L] >= 0
        # convert partner index -> partner index +1, keep 0 as "no partner"
        hier0[e_idx, :L][valid] = row[:L][valid] + 1

    for e_idx, row in enumerate(rows_end1):
        if row is None:
            continue
        L = min(len(row), max_rank1)
        valid = row[:L] >= 0
        hier1[e_idx, :L][valid] = row[:L][valid] + 1
    # print(hier0)
    # print(hier1)
    return node_pairs, hier0, hier1

import numpy as np
import networkx as nx
from itertools import combinations

def build_edge_pair_hierarchy_pillar_hugging(G: nx.Graph):
    """
    Streamline pairing for throat-centered Kagome from pillar array.

    At a Kagome node N representing throat (a,b), each incident edge N--M
    corresponds to "hugging" the pillar shared by N and M (either a or b).
    Streamlined continuation = pair edges that hug the same pillar at N.

    Returns
    -------
    node_pairs : dict
        node -> list of (e_i, e_j, keytuple) sorted by preference
        (keytuple contains (same_pillar, angle) for debugging)
    hier0, hier1 : np.ndarray
        Like your original: per edge-end an ordered list of partner edges.
        Stored as (partner_edge_idx + 1), 0 means "none".
    """
    edges = list(G.edges())
    n_edges = len(edges)

    def canon_edge(u, v):
        return (u, v) if u <= v else (v, u)

    edge_to_idx = {canon_edge(u, v): idx for idx, (u, v) in enumerate(edges)}

    pos = nx.get_node_attributes(G, "pos")
    pos_arr = {n: np.asarray(p, dtype=float) for n, p in pos.items()}

    # must exist from your kagome_from_delaunay(store_maps=True)
    cp = nx.get_node_attributes(G, "center_pair")

    node_pairs = {}
    rows_end0 = [None] * n_edges
    rows_end1 = [None] * n_edges

    for node in G.nodes():
        inc_edges = list(G.edges(node))
        deg = len(inc_edges)
        if deg < 2:
            continue

        if node not in cp:
            raise KeyError("Missing node attribute 'center_pair'. "
                           "Call kagome_from_delaunay(..., store_maps=True).")

        node_pair = cp[node]
        node_set = (node_pair[0], node_pair[1])

        p_node = pos_arr[node]

        e_indices = []
        end_ids = []
        vecs = []
        shared_pillar = []  # for each incident edge: which pillar (of the node's pair) this edge hugs

        for (u, v) in inc_edges:
            e_idx = edge_to_idx[canon_edge(u, v)]
            if edges[e_idx][0] == node:
                end_id = 0
                other = edges[e_idx][1]
            else:
                end_id = 1
                other = edges[e_idx][0]

            e_indices.append(e_idx)
            end_ids.append(end_id)
            vecs.append(pos_arr[other] - p_node)

            # Determine which pillar is shared between node and neighbor
            # (neighbor throat should share exactly one pillar with this throat)
            other_pair = cp.get(other, None)
            if other_pair is None:
                shared_pillar.append(-1)
            else:
                # intersection of two 2-tuples
                s = -1
                if other_pair[0] == node_set[0] or other_pair[1] == node_set[0]:
                    s = node_set[0]
                elif other_pair[0] == node_set[1] or other_pair[1] == node_set[1]:
                    s = node_set[1]
                shared_pillar.append(int(s))

        e_indices = np.array(e_indices, dtype=int)
        end_ids   = np.array(end_ids, dtype=int)
        vecs      = np.stack(vecs, axis=0)
        shared_pillar = np.array(shared_pillar, dtype=int)

        norms = np.linalg.norm(vecs, axis=1)
        norms[norms == 0.0] = 1.0

        pairs = []  # (i_loc, j_loc, angle, e_i, e_j, same_pillar)

        for i_loc, j_loc in combinations(range(deg), 2):
            v_i = vecs[i_loc]
            v_j = vecs[j_loc]
            denom = norms[i_loc] * norms[j_loc]
            cosang = np.dot(v_i, v_j) / denom
            cosang = np.clip(cosang, -1.0, 1.0)
            angle = np.arccos(cosang)

            e_i = int(e_indices[i_loc])
            e_j = int(e_indices[j_loc])

            # Prefer pairing edges that "hug" the same pillar at this throat
            same = (shared_pillar[i_loc] >= 0) and (shared_pillar[i_loc] == shared_pillar[j_loc])

            pairs.append((i_loc, j_loc, float(angle), e_i, e_j, same))

        # Preference order:
        #  1) same-pillar pairs first (streamlined, stays near same cylinder)
        #  2) otherwise, avoid "straight-through" as a fallback (we rank small angles earlier)
        #     so angle ASC puts ~60°/120° ahead of ~180°
        pairs.sort(key=lambda x: (0 if x[5] else 1, x[2]))

        node_pairs[node] = [(e_i, e_j, (same, ang)) for (_, _, ang, e_i, e_j, same) in pairs]

        # Build local hierarchy: for each incident edge, list partners in this preference order
        n_pairs = len(pairs)
        hier_local = -np.ones((deg, n_pairs), dtype=int)
        for rank, (i_loc, j_loc, angle, e_i, e_j, same) in enumerate(pairs):
            hier_local[i_loc, rank] = e_j
            hier_local[j_loc, rank] = e_i

        # store row for each edge-end
        for loc in range(deg):
            e_idx = int(e_indices[loc])
            end_id = int(end_ids[loc])
            row = hier_local[loc, :]
            if end_id == 0:
                rows_end0[e_idx] = row
            else:
                rows_end1[e_idx] = row

    max_rank0 = max((len(r) for r in rows_end0 if r is not None), default=0)
    max_rank1 = max((len(r) for r in rows_end1 if r is not None), default=0)

    hier0 = np.zeros((n_edges, max_rank0), dtype=int)
    hier1 = np.zeros((n_edges, max_rank1), dtype=int)

    for e_idx, row in enumerate(rows_end0):
        if row is None:
            continue
        L = min(len(row), max_rank0)
        valid = row[:L] >= 0
        hier0[e_idx, :L][valid] = row[:L][valid] + 1

    for e_idx, row in enumerate(rows_end1):
        if row is None:
            continue
        L = min(len(row), max_rank1)
        valid = row[:L] >= 0
        hier1[e_idx, :L][valid] = row[:L][valid] + 1

    return node_pairs, hier0, hier1


import numpy as np

def rotate_translate_points(P: np.ndarray, angle_deg: float, pad: float = 0.0):
    """
    Rotate points by angle_deg around their centroid, then translate so min x,y are at pad.
    """
    th = np.deg2rad(angle_deg)
    R = np.array([[np.cos(th), -np.sin(th)],
                  [np.sin(th),  np.cos(th)]], dtype=float)

    c = P.mean(axis=0)
    Pr = (P - c) @ R.T + c

    # translate to positive coords (handy for inlet/outlet detection)
    mins = Pr.min(axis=0)
    Pr = Pr - mins + pad
    return Pr

def remove_edges_in_splitter_strip(G, *,
                                  x_frac: float = 0.2,
                                  pillar_diam: float = 1.0,
                                  pos_attr: str = "pos"):
    """
    Remove edges intersecting a splitter strip:
      x in [x_min, x_min + x_frac*(x_max-x_min)]
      y in [y_mid - pillar_diam/2, y_mid + pillar_diam/2]
    """
    pos = nx.get_node_attributes(G, pos_attr)
    xs = np.array([p[0] for p in pos.values()])
    ys = np.array([p[1] for p in pos.values()])

    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    y_mid = 0.5 * (y_min + y_max)

    x0 = x_min
    x1 = x_min + x_frac * (x_max - x_min)
    y0 = y_mid - 0.5 * pillar_diam
    y1 = y_mid + 0.5 * pillar_diam

    # Liang–Barsky segment-rectangle intersection
    def seg_intersects_rect(p1, p2):
        xA, yA = p1
        xB, yB = p2
        dx = xB - xA
        dy = yB - yA
        t0, t1 = 0.0, 1.0
        for p, q in [(-dx, xA - x0), (dx, x1 - xA),
                     (-dy, yA - y0), (dy, y1 - yA)]:
            if p == 0.0:
                if q < 0.0:
                    return False
            else:
                t = q / p
                if p < 0.0:
                    if t > t1: return False
                    if t > t0: t0 = t
                else:
                    if t < t0: return False
                    if t < t1: t1 = t
        return True

    to_remove = []
    for u, v in list(G.edges()):
        if seg_intersects_rect(pos[u], pos[v]):
            to_remove.append((u, v))

    G.remove_edges_from(to_remove)
    return {"removed_edges": to_remove,
            "splitter_box": (x0, x1, y0, y1),
            "y_mid": y_mid}

def keep_largest_component(G):
    comps = list(nx.connected_components(G))
    if not comps:
        return G
    largest = max(comps, key=len)
    return G.subgraph(largest).copy()

import numpy as np

import numpy as np

def tri_centers_rect_symmetric(m_height: int, n_length: int, a: float):
    """
    Triangular (equilateral) packing in an axis-aligned rectangle with:
      - left/right boundaries aligned with x
      - no horizontal nearest-neighbor direction
      - symmetry about the horizontal midline

    m_height: number of pillars in height (your case: 20)
    n_length: number of pillar columns along x
    a       : center-to-center spacing (also the vertical NN spacing here)

    Returns
    -------
    pts : (N,2) array of pillar centers
    Lx, Ly : rectangle size that these centers fill (use for barrier placement)
    """
    sx = (np.sqrt(3) / 2.0) * a
    Ly = (m_height - 1) * a               # for m=20 -> Ly=19a, midline at 9.5a
    Lx = (n_length  - 1) * sx

    pts = []
    for i in range(n_length):
        x = i * sx
        y_off = 0.5 * a if (i % 2 == 1) else 0.0

        # to keep everything inside [0, Ly] AND preserve symmetry about Ly/2:
        # even columns: y = 0, a, ..., (m-1)a  (m points)
        # odd  columns: y = a/2, 3a/2, ..., (m-3/2)a (m-1 points)
        j_max = m_height - 1 if (i % 2 == 0) else (m_height - 2)
        for j in range(j_max + 1):
            y = j * a + y_off
            pts.append((x, y))

    return np.asarray(pts, float), Lx, Ly

import networkx as nx

def remove_barrier_edges(G, *, Lx, Ly, pillar_diam, x_frac=0.2, pos_attr="pos"):
    x0, x1 = 0.0, x_frac * Lx
    ymid = 0.5 * Ly
    y0, y1 = ymid - 0.5 * pillar_diam, ymid + 0.5 * pillar_diam

    pos = nx.get_node_attributes(G, pos_attr)

    def seg_intersects_rect(p1, p2):
        xA, yA = p1; xB, yB = p2
        dx = xB - xA; dy = yB - yA
        t0, t1 = 0.0, 1.0
        for p, q in [(-dx, xA - x0), (dx, x1 - xA),
                     (-dy, yA - y0), (dy, y1 - yA)]:
            if p == 0.0:
                if q < 0.0: return False
            else:
                t = q / p
                if p < 0.0:
                    if t > t1: return False
                    if t > t0: t0 = t
                else:
                    if t < t0: return False
                    if t < t1: t1 = t
        return True

    kill = [(u, v) for (u, v) in list(G.edges())
            if seg_intersects_rect(pos[u], pos[v])]
    G.remove_edges_from(kill)
    return kill



import networkx as nx
import numpy as np

def build_delaunay_net(sid: SimInputData, inc: Incidence) -> tuple(Graph, Edges):
    # --- Kagome build parameters ---
    a = 1#getattr(sid, "a", 1.0)  # allow optional lattice spacing in config
    Lx = (sid.n - 1) * a
    Ly = (sid.m - 1) * (np.sqrt(3)/2 * a)

    centers, Lx, Ly = tri_centers_rect_symmetric(m_height=sid.m, n_length=sid.n, a=a)

    graph_init = kagome_from_delaunay(centers, max_edge_ratio=1.8)

    remove_barrier_edges(graph_init, Lx=Lx, Ly=Ly, pillar_diam=0.5, x_frac=0.16)

    remove_edges_in_splitter_strip(
        graph_init,
        x_frac=0.16,
        pillar_diam=0.5#sid.pillar_diam  # or sid.a if that’s your diameter unit
    )

    graph_init = keep_largest_component(graph_init)

    # 3) Wrap into your Graph subclass
    graph = Graph()
    graph.add_nodes_from(graph_init.nodes(data=True))
    graph.add_edges_from(graph_init.edges(data=True))

    # Node positions
    pos = nx.get_node_attributes(graph, "pos")

    # sizes
    sid.nsq = graph.number_of_nodes()
    sid.ne  = graph.number_of_edges()

    # --- edge arrays ---
    diams = np.ones(sid.ne)
    lens  = np.ones(sid.ne)
    flow  = np.zeros(sid.ne)
    boundary_edges = np.zeros(sid.ne)

    # IMPORTANT: freeze edge ordering as a list (EdgeView is not stable)
    edge_list_draw = list(graph.edges())

    # nodes are already 0..N-1 from our builder, but keep mapping robust
    nodes = list(graph.nodes())
    node_to_index = {node: idx for idx, node in enumerate(nodes)}

    edge_list = []
    for (u, v) in edge_list_draw:
        edge_list.append((node_to_index[u], node_to_index[v]))

    # hierarchy (works fine on Kagome; degree ~4)
    #node_pairs, hier0, hier1 = build_edge_pair_hierarchy_two_ends(graph)
    node_pairs, hier0, hier1 = build_edge_pair_hierarchy_pillar_hugging(graph)
    edges = Edges(diams, lens, flow, edge_list, edge_list_draw, boundary_edges, hier0, hier1)

    # --- inlet/outlet selection (use a spacing-based epsilon) ---
    xs = np.array([p[0] for p in pos.values()])
    ys = np.array([p[1] for p in pos.values()])
    pos_in  = float(xs.min())
    pos_out = float(xs.max())

    # estimate a "typical" Kagome edge length to set boundary thickness robustly
    if sid.ne > 0:
        elens = []
        for (u, v) in edge_list_draw:
            pu = np.asarray(pos[u], float)
            pv = np.asarray(pos[v], float)
            elens.append(np.linalg.norm(pu - pv))
        elens = np.asarray(elens)
        eps = 0.6 * np.percentile(elens, 10)  # ~ one small edge
    else:
        eps = 0.1

    graph.in_nodes = []
    graph.out_nodes = []

    graph.in_vec   = np.zeros(sid.nsq)
    graph.in_vec_a = np.zeros(sid.nsq)
    graph.in_vec_b = np.zeros(sid.nsq)

    graph.out_vec   = np.zeros(sid.nsq)
    graph.out_vec_a = np.zeros(sid.nsq)
    graph.out_vec_b = np.zeros(sid.nsq)


    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    bound_x = x_min + 0.2 * (x_max - x_min)     # “first 20%”
    bound_y = 0.5 * (y_min + y_max)             # mid-height
    sid.bound_x = bound_x
    sid.bound_y = bound_y

    for node in graph.nodes():
        x, y = pos[node]
        idx = node_to_index[node]

        if x <= pos_in + eps:
            graph.in_nodes.append(idx)
            if y < bound_y:
                graph.in_vec_a[idx] = 1
            else:
                graph.in_vec_b[idx] = 1

        if x >= pos_out - eps:
            graph.out_nodes.append(idx)
            graph.out_vec[idx] = 1
            if y < bound_y:
                graph.out_vec_a[idx] = 1
            else:
                graph.out_vec_b[idx] = 1

    graph.in_vec = graph.in_vec_a + graph.in_vec_b
    sid.Q_in = sid.qin * 2 * len(graph.in_nodes)

    # set edge attrs for plotting/IO
    nx.set_edge_attributes(graph, dict(zip(edge_list_draw, diams)), "d")
    nx.set_edge_attributes(graph, dict(zip(edge_list_draw, flow)),  "q")
    nx.set_edge_attributes(graph, dict(zip(edge_list_draw, lens)),  "l")

    graph.vol_nodes = sid.v0 * np.ones_like(graph.in_vec)

    return graph, edges
