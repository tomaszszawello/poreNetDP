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
from scipy.stats import truncnorm
import networkx as nx
import numpy as np
import scipy.sparse as spr
import scipy.spatial as spt

from config import SimInputData
#if TYPE_CHECKING:
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
    zero_nodes = []
    boundary_edges = []
    boundary_nodes = []
    merged_triangles = []

    def __init__(self):
        nx.graph.Graph.__init__(self)

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
        merged_number = inc.plot.sum(axis = 0)
        np.savetxt('merged_number.txt', merged_number)
        diams = inc.plot @ edges.diams / merged_number
        flow = inc.plot @ edges.flow / merged_number
        nx.set_edge_attributes(self, dict(zip(edges.edge_list, diams)), \
            'd')
        nx.set_edge_attributes(self, dict(zip(edges.edge_list, flow)), \
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
        graph.out_nodes = np.arange(sid.n * (sid.m - 1), sid.nsq, 1)
        graph.in_vec = np.concatenate((np.ones(sid.n), \
            np.zeros(sid.n * (sid.m - 1))))
        graph.out_vec = np.concatenate((np.zeros(sid.n * (sid.m - 1)), \
            np.ones(sid.n)))
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

class Triangles():
    """ Container for information on space between edges.
    
    """
    tlist = []
    "list of triangles in the network"
    boundary = []
    "list of triangles on the boundary (to exclude for drawing)"
    incidence = []
    "incidence matrix for triangles and edges"
    volume = []
    "vector of geometrical volume of each triangle"
    centers = []
    "positions of centers of triangles"
    def __init__(self):
        self.tlist = []
        self.boundary = []
        self.incidence = []
        self.volume = []
        self.centers = []


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
    triangles: np.ndarray
    "number of triangles neighbouring each edge (usually 2)"
    alpha_b : np.ndarray
    ("vector scaling the effective reaction parameter for reaction B \
    (defaultly equal 1, but could be < 1 when there is not enough volume to \
    dissolve, for the reaction to proceed as usual)")
    def __init__(self, diams, lens, flow, edge_list, boundary, center, grain):
        self.diams = diams
        self.lens = lens
        self.flow = flow
        self.edge_list = edge_list
        self.boundary = boundary
        self.center = center
        self.diams_initial = diams
        self.active = np.zeros_like(diams)
        self.inlet = np.zeros_like(diams)
        self.outlet = np.zeros_like(diams)
        self.grain = grain


import networkx as nx
from networkx.algorithms import planarity

import networkx as nx


def relabel_hex_graph_to_indices(G, grains, boundary_edges_by_grain, center_edges_by_grain):
    """
    Relabel nodes of G to consecutive integers 0..N-1 and update
    grains / edge dictionaries accordingly.

    Parameters
    ----------
    G : nx.Graph
        Graph with arbitrary node labels (tuples, etc.).
    grains : list[tuple]
        grains[gid] is a tuple of corner *node labels* (as in G).
    boundary_edges_by_grain : dict[int, list[tuple]]
        boundary_edges_by_grain[gid] is list of (u, v) *node labels*.
    center_edges_by_grain : dict[int, list[tuple]]
        center_edges_by_grain[gid] is list of (center, corner) *node labels*.

    Returns
    -------
    G_int : nx.Graph
        Same graph but with nodes relabeled to integers 0..N-1.
    grains_int : list[tuple]
        grains_int[gid] is a tuple of integer node IDs.
    boundary_edges_by_grain_int : dict[int, list[tuple]]
        Same structure but edges now use integer IDs.
    center_edges_by_grain_int : dict[int, list[tuple]]
        Same structure but edges now use integer IDs.
    mapping : dict
        mapping[old_label] = new_int_id
    """
    # Assign integer IDs in the order nodes appear in G
    mapping = {old: idx for idx, old in enumerate(G.nodes())}

    # Relabel graph
    G_int = nx.relabel_nodes(G, mapping, copy=True)

    # Convert grains
    grains_int = [tuple(mapping[n] for n in grain) for grain in grains]

    # Convert boundary edges
    boundary_edges_by_grain_int = {}
    for gid, edges in boundary_edges_by_grain.items():
        boundary_edges_by_grain_int[gid] = [
            (mapping[u], mapping[v]) for (u, v) in edges
        ]

    # Convert center edges
    center_edges_by_grain_int = {}
    for gid, edges in center_edges_by_grain.items():
        center_edges_by_grain_int[gid] = [
            (mapping[u], mapping[v]) for (u, v) in edges
        ]

    return G_int, grains_int, boundary_edges_by_grain_int, center_edges_by_grain_int


def build_single_hex_graph(m: int, n: int):
    """
    Build a single NetworkX graph containing both subnetworks, with positions.

    Parameters
    ----------
    m, n : int
        Size parameters passed to nx.hexagonal_lattice_graph(m, n).

    Returns
    -------
    G : nx.Graph
        A graph containing:
          - hexagonal lattice edges with edge['grain_edge'] = 1, edge['center_edge'] = 0
          - center–corner edges with edge['grain_edge'] = 0, edge['center_edge'] = 1

        Each edge also has an attribute `grains`:
          - for lattice edges: tuple of grain IDs (length 1 on boundary, 2 in interior)
          - for center–corner edges: (gid,) of that center's grain

        Each node has a node attribute 'pos' = (x, y) giving its position.

    grains : list[tuple]
        grains[gid] is a 6-tuple of corner nodes forming the gid-th hexagonal grain.

    boundary_edges_by_grain : dict[int, list[tuple]]
        boundary_edges_by_grain[gid] is a list of (u, v) lattice edges
        on the boundary of grain gid.

    center_edges_by_grain : dict[int, list[tuple]]
        center_edges_by_grain[gid] is a list of (center_node, corner_node)
        edges from the center of grain gid to its 6 corners.
    """
    # --- Base hexagonal lattice of corner nodes (with positions) ---
    lattice = nx.hexagonal_lattice_graph(m, n)  # with_positions=True by default

    # Extract positions from the lattice
    lattice_pos = nx.get_node_attributes(lattice, "pos")

    # --- Planar embedding to enumerate faces (hexagonal grains) ---
    is_planar, emb = planarity.check_planarity(lattice)
    if not is_planar:
        raise ValueError("Lattice graph should be planar but isn't.")

    half_edges_seen = set()
    faces = []
    for u in emb:
        for v in emb[u]:
            if (u, v) not in half_edges_seen:
                face = emb.traverse_face(u, v, mark_half_edges=half_edges_seen)
                faces.append(tuple(face))

    # Keep only hexagonal faces as grains
    grains = [face for face in faces if len(face) == 6]

    # --- Map each undirected edge to grain IDs that use it ---
    edge_to_grains = {}
    for gid, cyc in enumerate(grains):
        L = len(cyc)
        for i in range(L):
            u, v = cyc[i], cyc[(i + 1) % L]
            e = tuple(sorted((u, v)))
            edge_to_grains.setdefault(e, []).append(gid)

    # --- Single graph with both subnetworks ---
    G = nx.Graph()

    # Add corner nodes with their positions and any other attributes
    G.add_nodes_from(
        (node, {"pos": lattice_pos[node], **lattice.nodes[node]})
        for node in lattice.nodes
    )

    boundary_edges_by_grain = {gid: [] for gid in range(len(grains))}
    center_edges_by_grain = {gid: [] for gid in range(len(grains))}

    # 1) Add hexagonal lattice ("grain") edges
    for u, v in lattice.edges:
        e = tuple(sorted((u, v)))
        grains_for_edge = tuple(edge_to_grains.get(e, ()))
        G.add_edge(
            u, v,
            grains=grains_for_edge,
            grain_edge=1,
            center_edge=0,
        )
        # Record per-grain boundary edges
        for gid in grains_for_edge:
            boundary_edges_by_grain[gid].append((u, v))

    # 2) Add center nodes and their edges to corners (with positions)
    for gid, cyc in enumerate(grains):
        # Compute center position as mean of corner positions
        xs = [lattice_pos[c][0] for c in cyc]
        ys = [lattice_pos[c][1] for c in cyc]
        center_pos = (sum(xs) / len(xs), sum(ys) / len(ys))

        center_node = ("center", gid)
        G.add_node(center_node, grain_id=gid, is_center=True, pos=center_pos)

        for corner in cyc:
            G.add_edge(
                center_node, corner,
                grains=(gid,),
                grain_edge=0,
                center_edge=1,
            )
            center_edges_by_grain[gid].append((center_node, corner))

    return relabel_hex_graph_to_indices(G, grains, boundary_edges_by_grain, center_edges_by_grain)

import numpy as np
from scipy.sparse import coo_matrix


import numpy as np
from scipy.sparse import coo_matrix


def build_hex_matrices(G, grains, boundary_edges_by_grain, center_edges_by_grain):
    """
    Build sparse matrices for a hex graph with grain centers.

    Parameters
    ----------
    G : nx.Graph
        Graph returned by build_single_hex_graph (contains both lattice
        and center–corner edges).
    grains : list[tuple]
        grains[gid] is a tuple of corner nodes.
    boundary_edges_by_grain : dict[int, list[tuple]]
        boundary_edges_by_grain[gid] is a list of (u, v) lattice edges
        on the boundary of grain gid.
    center_edges_by_grain : dict[int, list[tuple]]
        center_edges_by_grain[gid] is a list of (center_node, corner_node)
        edges for grain gid.

    Returns
    -------
    A_center : scipy.sparse.csr_matrix, shape (num_grains, num_edges)
        A_center[gid, e] = 1 if edge e is a center–corner edge of grain gid.

    A_boundary : scipy.sparse.csr_matrix, shape (num_grains, num_edges)
        A_boundary[gid, e] = 1 if edge e is a boundary edge of grain gid.

    B_incidence : scipy.sparse.csr_matrix, shape (num_edges, num_nodes)
        Oriented incidence matrix. For edge e = (u, v):
          - one +1 at the chosen "tail" node
          - one -1 at the chosen "head" node
        Orientation is arbitrary but fixed by node indexing.

    node_list : list
        node_list[j] is the node at column j of B_incidence.

    edge_list : list[tuple]
        edge_list[i] is the (u, v) edge at row i of B_incidence and
        column i of A_center / A_boundary.
    """
    num_grains = len(grains)

    # Fix an ordering of nodes and edges
    node_list = list(G.nodes())
    edge_list = list(G.edges())

    num_nodes = len(node_list)
    num_edges = len(edge_list)

    # Maps for indexing
    node_index = {node: j for j, node in enumerate(node_list)}

    # Use frozenset({u, v}) as canonical edge key (type-agnostic)
    edge_index = {frozenset((u, v)): i for i, (u, v) in enumerate(edge_list)}

    # ------------------------------------------------------------------
    # 1) Grain–edge incidence: center edges
    # ------------------------------------------------------------------
    rows_c, cols_c, data_c = [], [], []

    for gid in range(num_grains):
        for (u, v) in center_edges_by_grain[gid]:
            ekey = frozenset((u, v))
            eidx = edge_index[ekey]
            rows_c.append(gid)
            cols_c.append(eidx)
            data_c.append(1.0)

    A_center = coo_matrix(
        (data_c, (rows_c, cols_c)), shape=(num_grains, num_edges)
    ).tocsr()

    # ------------------------------------------------------------------
    # 2) Grain–edge incidence: boundary edges
    # ------------------------------------------------------------------
    rows_b, cols_b, data_b = [], [], []

    for gid in range(num_grains):
        for (u, v) in boundary_edges_by_grain[gid]:
            ekey = frozenset((u, v))
            eidx = edge_index[ekey]
            rows_b.append(gid)
            cols_b.append(eidx)
            data_b.append(1.0)

    A_boundary = coo_matrix(
        (data_b, (rows_b, cols_b)), shape=(num_grains, num_edges)
    ).tocsr()

    # ------------------------------------------------------------------
    # 3) Standard oriented incidence matrix (edge × node)
    # ------------------------------------------------------------------
    rows_i, cols_i, data_i = [], [], []

    for eidx, (u, v) in enumerate(edge_list):
        iu = node_index[u]
        iv = node_index[v]

        # Choose an orientation: smaller node index -> tail (+1)
        if iu < iv:
            tail, head = iu, iv
        else:
            tail, head = iv, iu

        rows_i.extend([eidx, eidx])
        cols_i.extend([tail, head])
        data_i.extend([1.0, -1.0])

    B_incidence = coo_matrix(
        (data_i, (rows_i, cols_i)), shape=(num_edges, num_nodes)
    ).tocsr()

    return A_center, A_boundary, B_incidence, node_list, edge_list



def build_delaunay_net(sid: SimInputData) \
    -> tuple(Graph, Edges, Incidence):
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
    G, grains, boundary_edges, center_edges = build_single_hex_graph(sid.m, sid.n)
    
    grain_center, grain_boundary, incidence, nodes, edges = build_hex_matrices(
        G, grains, boundary_edges, center_edges
    )
    graph = Graph()
    graph.add_nodes_from(G.nodes())
    graph.add_edges_from(G.edges())

    pos = nx.get_node_attributes(G, 'pos')

    nx.set_node_attributes(graph, pos, 'pos')

    sid.ne = len(graph.edges())
    sid.nsq = len(graph.nodes())
    sid.ntr = len(grains)

    #normal = np.random.randn(sid.ne)
    #diams = np.exp(sid.d0 + sid.sigma_d0 * normal)
    #diams = np.clip(diams, 0, 50)
    diams = 1 * (grain_boundary.T @ np.ones(sid.ntr) > 0)
    lens = np.ones(sid.ne)
    flow = np.zeros(sid.ne)
    center = 1 * (grain_center.T @ np.ones(sid.ntr) > 0)
    boundary = 1 * (grain_boundary.T @ np.ones(sid.ntr) > 0)
    grain = sid.grain_vol * np.ones(len(grains))
    
    edge_list = []
    
    nodes = list(graph.nodes())
    node_to_index = {node: idx for idx, node in enumerate(nodes)}

    for edge in graph.edges():
        edge_list.append((node_to_index[edge[0]], node_to_index[edge[1]]))

    edges = Edges(diams, lens, flow, edge_list, boundary, center, grain)

    pos_in = np.min(np.array(list(pos.values()))[:, 0])
    pos_out = np.max(np.array(list(pos.values()))[:, 0])
    print(pos_in, pos_out)

    graph.in_nodes = []
    graph.out_nodes = []
    graph.in_vec = np.zeros(sid.nsq)
    graph.out_vec = np.zeros(sid.nsq)
    for node in graph.nodes():
        #if pos[node][0] <= pos_in + 0.8:
        if pos[node][0] == pos_in:
            graph.in_nodes.append(node_to_index[node])
            graph.in_vec[node_to_index[node]] = 1

    # WARNING
    #
    # Networkx changes order of edges, make sure you use edge_list every time you plot!!!
    # 
    #
    nx.set_edge_attributes(graph, dict(zip(edge_list, diams)), 'd')
    nx.set_edge_attributes(graph, dict(zip(edge_list, flow)), 'q')
    nx.set_edge_attributes(graph, dict(zip(edge_list, lens)), 'l')

    inc = Incidence()
    inc.incidence = incidence
    inc.center = grain_center
    inc.boundary = grain_boundary

    edges.active = 1. * (np.abs(inc.incidence) @ graph.in_vec > 0)
    #print(graph.nodes())
    #print(graph.edges())
    return graph, edges, inc
