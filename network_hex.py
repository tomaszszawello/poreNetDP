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
    def __init__(self, diams, lens, flow, edge_list, edge_list_draw, boundary_list):
        self.diams = diams
        self.lens = lens
        self.flow = flow
        self.edge_list = edge_list
        self.edge_list_draw = edge_list_draw
        self.boundary_list = boundary_list
        self.diams_initial = diams
        self.merged = np.zeros_like(diams)
        self.transversed = np.zeros_like(diams)

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

    normal = np.random.randn(sid.ne)
    diams = np.exp(sid.d0 + sid.sigma_d0 * normal)
    diams = np.clip(diams, 0, 50)
    #diams = np.ones(sid.ne)
    lens = np.ones(sid.ne)
    flow = np.zeros(sid.ne)
    boundary_edges = np.zeros(sid.ne)
    edge_list_draw = graph.edges()
    edge_list = []
    
    nodes = list(graph.nodes())
    node_to_index = {node: idx for idx, node in enumerate(nodes)}

    for edge in graph.edges():
        edge_list.append((node_to_index[edge[0]], node_to_index[edge[1]]))

    edges = Edges(diams, lens, flow, edge_list, edge_list_draw, boundary_edges)

    pos_in = np.min(np.array(list(pos.values()))[:, 0])
    pos_out = np.max(np.array(list(pos.values()))[:, 0])
    print(pos_in, pos_out)

    graph.in_vec = np.zeros(sid.nsq)
    graph.in_vec_a = np.zeros(sid.nsq)
    graph.in_vec_b = np.zeros(sid.nsq)
    graph.out_vec = np.zeros(sid.nsq)

    for node in graph.nodes():
        if pos[node][0] == pos_in:
            graph.in_nodes.append(node_to_index[node])
            print(pos[node])
            if pos[node][1] < sid.bound_y:
                graph.in_vec_a[node_to_index[node]] = 1
            else:
                graph.in_vec_b[node_to_index[node]] = 1
        if pos[node][0] == pos_out:
            graph.out_nodes.append(node_to_index[node])
            graph.out_vec[node_to_index[node]] = 1

    graph.in_vec = graph.in_vec_a + graph.in_vec_b
    print(np.sum(graph.in_vec_a), np.sum(graph.in_vec_b), np.sum(graph.out_vec))
    # WARNING
    #
    # Networkx changes order of edges, make sure you use edge_list every time you plot!!!
    # 
    #
    nx.set_edge_attributes(graph, dict(zip(edge_list_draw, diams)), 'd')
    nx.set_edge_attributes(graph, dict(zip(edge_list_draw, flow)), 'q')
    nx.set_edge_attributes(graph, dict(zip(edge_list_draw, lens)), 'l')

    return graph, edges
