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
        graph.out_nodes = np.arange(sid.n * (sid.n - 1), sid.nsq, 1)
        graph.in_vec = np.concatenate((np.ones(sid.n), \
            np.zeros(sid.n * (sid.n - 1))))
        graph.out_vec = np.concatenate((np.zeros(sid.n * (sid.n - 1)), \
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
    def __init__(self, diams, lens, flow, edge_list, boundary_list):
        self.diams = diams
        self.lens = lens
        self.flow = flow
        self.edge_list = edge_list
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
    points_left = np.linspace([0, 0], [0, sid.n - 1], sid.n) + \
        np.array([0, 0.5])
    points_right = np.linspace([0, 0], [0, sid.n - 1], sid.n) + \
        np.array([sid.n, 0.5])
    points_top = np.random.uniform(0.5, sid.n - 0.5, (sid.n, 2)) * \
        np.array([1, 0]) + np.random.uniform(0, 1, (sid.n, 2)) * \
        np.array([0, 1])
    points_bottom = np.random.uniform(0.5, sid.n - 0.5, (sid.n, 2)) * \
        np.array([1, 0]) + np.array([0, sid.n]) - \
        np.random.uniform(0, 1, (sid.n, 2)) * np.array([0, 1])
    points_middle = np.random.uniform(0.5, sid.n - 0.5, \
        (sid.n * (sid.n - 4), 2)) * np.array([1, 0]) + np.random.uniform(1, \
        sid.n - 1, (sid.n * (sid.n - 4), 2)) * np.array([0, 1])
    points = np.concatenate((points_middle, points_left, points_right, \
        points_top, points_bottom))
    points = np.array(sorted(points, key = lambda elem: (elem[0], elem[1])))

    points_above_pbc = points.copy() + np.array([0, sid.n])
    points_below_pbc = points.copy() + np.array([0, -sid.n])
    points_right_pbc =  points.copy() + np.array([sid.n, 0])
    points_left_pbc = points.copy() + np.array([-sid.n, 0])

    if sid.periodic == 'none':
        pos = points
    elif sid.periodic == 'top': 
        pos = np.concatenate([points, points_above_pbc, points_below_pbc])
    elif sid.periodic == 'side':
        pos = np.concatenate([points, points_right_pbc, points_left_pbc])
    elif sid.periodic == 'all':
        pos = np.concatenate([points, points_above_pbc, points_below_pbc, \
            points_right_pbc, points_left_pbc])
    else:
        raise ValueError("Unknown boundary condition type.")

    del_tri = spt.Delaunay(pos)
    # create a set for edges that are indexes of the points
    edge_list = dict()
    boundary_edges = []
    boundary_nodes = []
    lens = []
    pipe_diams = []
    edge_index = 0

    merge_matrix_row = []
    merge_matrix_col = []
    merge_matrix_data = []

    for tri in range(del_tri.nsimplex):
        # for each edge of the triangle
        # sort the vertices
        # (sorting avoids duplicated edges being added to the set)
        # and add to the edges set
        n1, n2, n3 = sorted(del_tri.simplices[tri])

        m_n3 = 0
        bound = False
        if n3 < sid.nsq:
            pass
        elif n2 < sid.nsq:
            m_n3 = (n3 // sid.nsq) * sid.nsq
            bound = True
        else:
            continue
        n1_new, n2_new, n3_new = n1, n2, n3 - m_n3
        if n1_new == n2_new or n2_new == n3_new or n1_new == n3_new:
            continue
        if bound:
            boundary_nodes.extend((n1_new, n2_new, n3_new))
        lens_tr = (np.linalg.norm(np.array(pos[n1]) - np.array(pos[n2])), \
            np.linalg.norm(np.array(pos[n1]) - np.array(pos[n3])), \
            np.linalg.norm(np.array(pos[n2]) - np.array(pos[n3])))

        edge_index_list = []
        
        for i, edge in enumerate((sorted((n1_new, n2_new)), \
            sorted((n1_new, n3_new)), sorted((n2_new, n3_new)))):
            node1, node2 = edge
            if (node1, node2) not in edge_list:
                edge_list[(node1, node2)] = edge_index
                cur_edge_index = edge_index

                if sid.initial_pipe:
                    if pos[node1][1] < sid.n / 2 + sid.pipe_width and pos[node1][1] > sid.n / 2 - sid.pipe_width and pos[node2][1] < sid.n / 2 + sid.pipe_width and pos[node2][1] > sid.n / 2 - sid.pipe_width:
                        pipe_diams.append(sid.pipe_diam)
                    else:
                        pipe_diams.append(0)
                lens.append(lens_tr[i])
                edge_index += 1
                if bound and i > 0:
                    boundary_edges.append(1)
                else:
                    boundary_edges.append(0)
            else:
                cur_edge_index = edge_list[(node1, node2)]
                if bound and i > 0:
                    boundary_edges[cur_edge_index] = 1
            edge_index_list.append(cur_edge_index)

        merge_matrix_row.extend(2 * edge_index_list)
        merge_matrix_col.extend(np.roll(edge_index_list, 1))
        merge_matrix_col.extend(np.roll(edge_index_list, 2))
        for index in list(np.roll(edge_index_list, 2)) \
            + list(np.roll(edge_index_list, 1)):
            merge_matrix_data.append(lens[index] / 2)


    boundary_edges = np.array(boundary_edges)

    edge_list = list(edge_list)
    sid.ne = len(edge_list)

    if sid.noise == 'gaussian':
        diams = np.array(truncnorm.rvs(sid.dmin, sid.dmax, loc = sid.d0, \
            scale = sid.sigma_d0, size = len(edge_list)))
    elif sid.noise == 'lognormal':
        normal = np.random.randn(len(edge_list))
        diams = np.exp(sid.d0 + sid.sigma_d0 * normal)
        diams = np.clip(diams, 0, 50)
    elif sid.noise == 'klognormal':
        normal = np.random.randn(len(edge_list))
        lognormal = np.exp(sid.d0 + sid.sigma_d0 * normal)
        diams4 = lognormal * (lens / np.average(lens))
        diams = diams4 ** 0.25
    elif sid.noise == 'file_lognormal_d':
        diams_array = np.loadtxt(sid.noise_filename).T
        diams = []
        for n1, n2 in edge_list:
            diams.append((diams_array[n1 // sid.n, n1 % sid.n] + \
                diams_array[n2 // sid.n, n2 % sid.n]) / 2)
        diams = np.array(diams)
    elif sid.noise == 'file_lognormal_k':
        k_array = np.loadtxt(sid.noise_filename).T
        k_array = np.roll(k_array, sid.n ** 2 // 3)
        k = []
        for n1, n2 in edge_list:
            k.append((k_array[n1 // sid.n, n1 % sid.n] + \
                k_array[n2 // sid.n, n2 % sid.n]) / 2)
        diams = (np.array(k) * (np.array(lens) / np.average(lens))) ** 0.25
    else:
        raise ValueError(f'Unknown noise type: {sid.noise}')
    lens = np.array(lens)
    
    diams /= np.average(diams)
    if sid.initial_pipe:
        diams += np.array(pipe_diams)
    
    merge_matrix_data = np.array(merge_matrix_data) / np.average(lens)
    inc.merge = spr.csr_matrix((merge_matrix_data, (merge_matrix_row, \
        merge_matrix_col)), shape=(sid.ne, sid.ne)) * sid.merge_length
    lens = lens / np.average(lens)
    flow = np.zeros(len(edge_list))

    edges = Edges(diams, lens, flow, edge_list, boundary_edges)

    graph = Graph()
    graph.add_nodes_from(list(range(sid.nsq)))
    graph.add_edges_from(edge_list)
    graph.boundary_nodes = boundary_nodes
    # WARNING
    #
    # Networkx changes order of edges, make sure you use edge_list 
    # every time you plot!!!
    # 
    #
    nx.set_edge_attributes(graph, dict(zip(edge_list, diams)), 'd')
    nx.set_edge_attributes(graph, dict(zip(edge_list, flow)), 'q')
    nx.set_edge_attributes(graph, dict(zip(edge_list, lens)), 'l')

    nx.set_node_attributes(graph, dict(zip(list(range(sid.nsq)), pos)), 'pos')

    return graph, edges
