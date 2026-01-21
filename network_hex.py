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
from itertools import combinations
import networkx as nx
import numpy as np
import scipy.sparse as spr
import scipy.spatial as spt

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
    hierarchy: np.ndarray
    def __init__(self, diams, lens, flow, edge_list, edge_list_draw, boundary_list, hierarchy):
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
        self.hierarchy = hierarchy




def build_edge_pair_hierarchy(G):
    """
    Parameters
    ----------
    G : networkx.Graph
        Undirected pore network graph.
    pos : dict
        Mapping node -> (x, y) coordinates (or array-like of length 2).
    edges : list of (u, v), optional
        Global list of edges, defining their indices. If None, uses list(G.edges()).
        The index of an edge is its position in this list.

    Returns
    -------
    node_pairs : dict
        node -> list of ((e_idx1, e_idx2), angle) sorted by ascending angle.
        e_idx* are global edge indices (0-based).
    hierarchy : np.ndarray, shape (n_edges, max_len)
        For each edge i (row), an ordered list of partner edges (columns).
        hierarchy[i, k] == 0  : no partner at this slot.
        hierarchy[i, k] == j>0: partner edge has index (j-1) in the global edge list.
        (So we store edge indices + 1, using 0 as "no edge".)
    """

    edges = list(G.edges())
    n_edges = len(edges)

    # Canonical representation of edges for undirected graph
    def canon_edge(u, v):
        return (u, v) if u <= v else (v, u)

    # Map canonical edge -> global index (0-based)
    edge_to_idx = {canon_edge(u, v): idx for idx, (u, v) in enumerate(edges)}

    pos = nx.get_node_attributes(G, 'pos')
    # Pre-ensure pos arrays
    pos_arr = {n: np.asarray(p, dtype=float) for n, p in pos.items()}

    # Collect per-node pairs and global partner lists
    node_pairs = {}  # node -> list of (e_idx1, e_idx2, angle)
    partner_angles = {i: [] for i in range(n_edges)}  # edge_idx -> list of (partner_idx, angle)

    for node in G.nodes():
        inc_edges = list(G.edges(node))  # each is (node, nbr) for an undirected Graph

        deg = len(inc_edges)
        if deg < 2:
            continue  # no pairs at this node

        # For each incident edge: direction vector and global edge index
        vecs = []
        e_indices = []
        p_node = pos_arr[node]

        for (u, v) in inc_edges:
            # (u, v) is oriented so that one end is 'node'
            other = v if u == node else u
            vec = pos_arr[other] - p_node
            vecs.append(vec)
            e_indices.append(edge_to_idx[canon_edge(u, v)])

        vecs = np.stack(vecs, axis=0)  # shape (deg, 2)
        norms = np.linalg.norm(vecs, axis=1)
        # guard against zero-length edges
        norms[norms == 0.0] = 1.0

        pairs = []

        # Compute angle between each pair of incident edges
        for i_local, j_local in combinations(range(deg), 2):
            e_i = e_indices[i_local]
            e_j = e_indices[j_local]

            v_i = vecs[i_local]
            v_j = vecs[j_local]

            denom = norms[i_local] * norms[j_local]
            cosang = np.dot(v_i, v_j) / denom
            cosang = np.clip(cosang, -1.0, 1.0)
            angle = np.arccos(cosang)  # in radians

            pairs.append((e_i, e_j, angle))

            # Update global partner lists (symmetrically)
            partner_angles[e_i].append((e_j, angle))
            partner_angles[e_j].append((e_i, angle))

        # sort node's pairs by angle
        pairs.sort(key=lambda x: x[2])  # smallest angle first
        node_pairs[node] = pairs

    # Build global hierarchy array per edge
    max_len = max((len(lst) for lst in partner_angles.values()), default=0)
    hierarchy = np.zeros((n_edges, max_len), dtype=int)

    for e_idx in range(n_edges):
        lst = partner_angles[e_idx]
        if not lst:
            continue
        # Sort partners by increasing angle
        lst_sorted = sorted(lst, key=lambda x: x[1])
        # Fill row with partner indices (+1 so 0 can mean "no partner")
        for k, (partner_idx, angle) in enumerate(lst_sorted):
            hierarchy[e_idx, k] = partner_idx + 1

    return node_pairs, hierarchy

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
    graph_init = nx.hexagonal_lattice_graph(sid.m, sid.n, periodic = False, with_positions = True)
    graph = Graph()
    graph.add_nodes_from(graph_init.nodes())
    graph.add_edges_from(graph_init.edges())
    pos = nx.get_node_attributes(graph_init, 'pos')
    nx.set_node_attributes(graph, pos, 'pos')
    # for edge in graph.copy().edges():
    #     if (pos[edge[0]][0] < sid.bound_x and pos[edge[1]][0] < sid.bound_x) and ((pos[edge[0]][1] < sid.bound_y and pos[edge[1]][1] > sid.bound_y) or (pos[edge[0]][1] > sid.bound_y and pos[edge[1]][1] < sid.bound_y)):
    #         graph.remove_edge(edge[0], edge[1])
    for node in graph.copy().nodes():
        if pos[node][0] < sid.bound_x and pos[node][1] < sid.bound_y + 0.1 and pos[node][1] > sid.bound_y - 0.1:
            graph.remove_node(node)
    for node in graph.copy().nodes():
        if (pos[node][1] == np.sqrt(3) / 2 and pos[node][0] == 0) or (pos[node][1] == (sid.m * 2 - 1) * np.sqrt(3) / 2 and pos[node][0] == 0) or pos[node][1] == sid.y_min or pos[node][1] >= sid.y_max:
            graph.remove_node(node)

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

    node_pairs, hierarchy = build_edge_pair_hierarchy(graph)
    edges = Edges(diams, lens, flow, edge_list, edge_list_draw, boundary_edges, hierarchy)

    pos_in = np.min(np.array(list(pos.values()))[:, 0])
    pos_out = np.max(np.array(list(pos.values()))[:, 0])
    print(pos_in, pos_out)

    graph.in_vec = np.zeros(sid.nsq)
    graph.in_vec_a = np.zeros(sid.nsq)
    graph.in_vec_b = np.zeros(sid.nsq)
    graph.out_vec = np.zeros(sid.nsq)
    pos_x_max = pos_out
    pos_y_max = 0
    for node in graph.nodes():
        if pos[node][0] == pos_in:
            graph.in_nodes.append(node_to_index[node])
            if pos[node][1] < sid.bound_y:
                graph.in_vec_a[node_to_index[node]] = 1
            else:
                graph.in_vec_b[node_to_index[node]] = 1
        if pos[node][0] == pos_out:
            if pos[node][1] > pos_y_max:
                pos_y_max = pos[node][1]
            graph.out_nodes.append(node_to_index[node])
            graph.out_vec[node_to_index[node]] = 1
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

    return graph, edges
