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


import networkx as nx
import numpy as np

def diamond_lattice_graph(nx_nodes, ny_nodes):
    """
    Rectangular diamond lattice:
    - physical x in [0, ny_nodes-1]
    - physical y in [0, nx_nodes-1]
    - edges at ±45° to the axes, realized via a 2x finer index grid.

    Nodes exist only at (i,j) with i,j both even or both odd.
    Their physical positions are pos = (j/2, i/2).

    Nodes in the first 20% of the width (x < 0.2 * ny_nodes)
    that are exactly at half of the height (y == (nx_nodes - 1)/2)
    are removed.
    """

    G = nx.Graph()

    # define the "removed" band in physical coordinates
    cutoff_x = 0.2 * ny_nodes           # first 20% of width
    mid_y = (nx_nodes - 1) / 2.0        # half height in physical y

    # add nodes with positions, skipping the removed ones
    for i in range(2 * nx_nodes):
        for j in range(2 * ny_nodes):
            # only use even-even and odd-odd to make a clean diamond pattern
            if (i % 2) != (j % 2):
                continue
            if j == 2 * ny_nodes - 1:
                continue
            if i == 2 * nx_nodes - 1:
                continue

            x = j / 2.0
            y = i / 2.0

            # skip nodes in the first 20% of width at half height
            if x < cutoff_x and np.isclose(y, mid_y):
                continue

            G.add_node((i, j), pos=(x, y))

    # add diagonal edges, only if both endpoints exist
    for i in range(2 * nx_nodes):
        for j in range(2 * ny_nodes):
            if (i, j) not in G:
                continue

            # down-right neighbor (i+1, j+1)
            if i + 1 < 2 * nx_nodes and j + 1 < 2 * ny_nodes:
                if (i + 1, j + 1) in G:
                    G.add_edge((i, j), (i + 1, j + 1))


            # down-left neighbor (i+1, j-1)
            if i + 1 < 2 * nx_nodes and j - 1 >= 0:
                if (i + 1, j - 1) in G:
                    G.add_edge((i, j), (i + 1, j - 1))

    return G


def build_delaunay_net(sid: SimInputData, inc: Incidence) \
    -> tuple(Graph, Edges):
    """ Build Delaunay network with parameters from config.

    This function creates Delaunay network with size and boundary condition
    taken from config file. It saves it to Graph class instance.

    Parameters
    -------
    sid : SimInputData class object
        all config parameters of the simulation

    Returns
    -------
    graph : Graph class object
        network and all its properties
    """
    # graph_init_init = nx.hexagonal_lattice_graph(10 * sid.m, 10 * sid.n, periodic = False, with_positions = True)
    # #graph_init = subdivide_edges_into_three(graph_init_init)
    # graph_init = graph_init_init
    # graph = Graph()
    # graph.add_nodes_from(graph_init.nodes())
    # graph.add_edges_from(graph_init.edges())
    # pos = nx.get_node_attributes(graph_init, 'pos')
    # nx.set_node_attributes(graph, pos, 'pos')
    # #graph = nx.hexagonal_lattice_graph(2 * m, 2 * n, periodic = False, with_positions = True)
    # pos = nx.get_node_attributes(graph, 'pos')

    # diamond lattice: start from a square grid
    graph_init_init = diamond_lattice_graph(sid.m, sid.n)
    edges_list = list(graph_init_init.edges())   # [(u0,v0), (u1,v1), ...]
    # canonicalize order for undirected graph
    edge_to_idx = {tuple(sorted(e)): i for i, e in enumerate(edges_list)}
    e1 = ((19, 39), (20, 40))
    e2 = ((21, 39), (20, 40))

    special_edge_indices = [
        edge_to_idx[tuple(sorted(e1))],
        edge_to_idx[tuple(sorted(e2))]
]
    
    G1, edge_map, new_nodes = subdivide_edges_into_three(graph_init_init)
    mapping = {old: i for i, old in enumerate(G1.nodes())}
    graph_init = nx.relabel_nodes(G1, mapping, copy=True)

    # (Optional) Update edge_map if you use it later
    # edge_map_i = {
    #     key: [(mapping[u], mapping[v]) for (u, v) in segs]
    #     for key, segs in edge_map.items()
    # }

    graph = Graph()
    graph.add_nodes_from(graph_init.nodes())
    graph.add_edges_from(graph_init.edges())

    pos = nx.get_node_attributes(graph_init, 'pos')
    nx.set_node_attributes(graph, pos, 'pos')
    pos = nx.get_node_attributes(graph, 'pos')




    # nx.draw_networkx_edges(graph, pos)
    # theta  = np.radians(45)
    # R      = np.array([[ np.cos(theta), -np.sin(theta)],
    #                 [ np.sin(theta),  np.cos(theta)]])
    # for node in graph.nodes():
    #     graph.nodes[node]['pos'] = R @ graph.nodes[node]['pos'] - (sid.m, sid.n)

    x_max = sid.n
    x_min = 0
    y_max = (2 * sid.m) * np.sqrt(3) / 2
    y_min = 0

    # graph2 = graph.copy()
    # for node in graph2.nodes():
    #     x, y = graph.nodes[node]['pos']
    #     if x < x_min or x > x_max or y < y_min or y > y_max:
    #         graph.remove_node(node)
    # pos = nx.get_node_attributes(graph, 'pos')
    # print(pos)
    
    # for edge in graph.copy().edges():
    #     if (pos[edge[0]][0] < sid.bound_x and pos[edge[1]][0] < sid.bound_x) and ((pos[edge[0]][1] < sid.bound_y and pos[edge[1]][1] > sid.bound_y) or (pos[edge[0]][1] > sid.bound_y and pos[edge[1]][1] < sid.bound_y)):
    #         graph.remove_edge(edge[0], edge[1])
    # for node in graph.copy().nodes():
    #     if pos[node][0] < sid.bound_x and pos[node][1] < sid.bound_y + 0.5 and pos[node][1] > sid.bound_y - 0.5:
    #         graph.remove_node(node)
    # for node in graph.copy().nodes():
    #     if (pos[node][1] == np.sqrt(3) / 2 and pos[node][0] == 0) or (pos[node][1] == (sid.m * 2 - 1) * np.sqrt(3) / 2 and pos[node][0] == 0) or pos[node][1] == sid.y_min or pos[node][1] >= sid.y_max:
    #         graph.remove_node(node)

    sid.ne = len(graph.edges())
    sid.nsq = len(graph.nodes())

    #normal = np.random.randn(sid.ne)
    #diams = np.exp(sid.d0 + sid.sigma_d0 * normal)
    #diams = np.clip(diams, 0, 50)
    diams = np.ones(sid.ne)
    lens = np.ones(sid.ne)
    flow = np.zeros(sid.ne)
    boundary_edges = np.zeros(sid.ne)
    edge_list_draw = graph.edges()
    edge_list = []
    
    nodes = list(graph.nodes())
    node_to_index = {node: idx for idx, node in enumerate(nodes)}

    for edge in graph.edges():
        edge_list.append((node_to_index[edge[0]], node_to_index[edge[1]]))

    node_pairs, hier0, hier1 = build_edge_pair_hierarchy_two_ends(graph)
    edges = Edges(diams, lens, flow, edge_list, edge_list_draw, boundary_edges, hier0, hier1)

    pos_in = np.min(np.array(list(pos.values()))[:, 0])
    pos_out = np.max(np.array(list(pos.values()))[:, 0])
    print(pos_in, pos_out)

    graph.in_vec = np.zeros(sid.nsq)
    graph.in_vec_a = np.zeros(sid.nsq)
    graph.in_vec_b = np.zeros(sid.nsq)
    graph.out_vec = np.zeros(sid.nsq)
    graph.out_vec_a = np.zeros(sid.nsq)
    graph.out_vec_b = np.zeros(sid.nsq)
    pos_x_max = pos_out
    pos_y_max = 0
    for node in graph.nodes():
        #if pos[node][0] <= pos_in + 0.8:
        if pos[node][0] <= pos_in + 0.1:
            graph.in_nodes.append(node_to_index[node])
            if pos[node][1] < sid.bound_y:
                graph.in_vec_a[node_to_index[node]] = 1
            else:
                graph.in_vec_b[node_to_index[node]] = 1
        #if pos[node][0] >= pos_out - 0.8:
        if pos[node][0] >= pos_out - 0.1:
            if pos[node][1] > pos_y_max:
                pos_y_max = pos[node][1]
            graph.out_nodes.append(node_to_index[node])
            graph.out_vec[node_to_index[node]] = 1
            if pos[node][1] < sid.bound_y:
                graph.out_vec_a[node_to_index[node]] = 1
            else:
                graph.out_vec_b[node_to_index[node]] = 1
    print(pos_x_max, pos_y_max)
    graph.in_vec = graph.in_vec_a + graph.in_vec_b
    sid.Q_in = sid.qin * 2 * len(graph.in_nodes)
    # WARNING
    #
    # Networkx changes order of edges, make sure you use edge_list every time you plot!!!
    # 
    #
    nx.set_edge_attributes(graph, dict(zip(edge_list_draw, diams)), 'd')
    nx.set_edge_attributes(graph, dict(zip(edge_list_draw, flow)), 'q')
    nx.set_edge_attributes(graph, dict(zip(edge_list_draw, lens)), 'l')

    edges.special = special_edge_indices

    return graph, edges
