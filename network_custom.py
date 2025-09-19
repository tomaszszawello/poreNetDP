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

import re

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
    def __init__(self, diams, lens, flow, edge_list, boundary_list, triangles = np.array([])):
        self.diams = diams
        self.lens = lens
        self.flow = flow
        self.edge_list = edge_list
        self.boundary_list = boundary_list
        self.diams_initial = diams
        self.merged = np.zeros_like(diams)
        self.transversed = np.zeros_like(diams)
        self.A = np.zeros_like(diams)
        self.B = np.zeros_like(diams)
        self.diams_draw = diams.copy()
        self.triangles = triangles
        self.alpha_b = np.zeros_like(diams)

def load_net_to_networkx(path, multigraph=False):
    """
    Load your 'net.txt' into a NetworkX graph.

    Row format (after header lines):
      <id> <x> <y> <z> <type> <b>   ( nb_id, pore_id ) ( nb_id, pore_id ) ...

    - Nodes: attrs x,y,z (float), type (int), b (int)
    - Edges: attr 'pore' (int). With multigraph=False, multiple pores on same
      undirected edge are merged into a list.
    """
    G = nx.MultiGraph() if multigraph else nx.Graph()
    pair_re = re.compile(r'\(\s*(\d+)\s*,\s*([+-]?\d+)\s*\)')

    def _parse_head(line: str):
        # Only look at the part before the first neighbor tuple
        head_part = line.split('(', 1)[0]
        nums = re.findall(r'[+-]?\d+(?:\.\d+)?', head_part)
        if len(nums) < 6:
            return None  # not a node line

        node_id = int(float(nums[0]))
        x = float(nums[1]); y = float(nums[2]); z = float(nums[3])
        ntype = int(float(nums[4]))

        if len(nums) >= 7:
            bval = int(float(nums[5]))
            n_neigh = int(float(nums[6]))
        else:
            bval = None  # 'b' missing on this line
            n_neigh = int(float(nums[5]))

        return node_id, x, y, z, ntype, bval, n_neigh

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        edge_ind = 0
        rem_list = []
        for raw in f:
            s = raw.strip()
            if not s or s.startswith('#'):
                continue

            head = _parse_head(s)
            if not head:
                continue

            node_id, x, y, z, ntype, bval, n_neigh = head
            # if x > 12.5:
            #     rem_list.append(node_id)
            if y < 1.25:
                ntype = 1
                y = 1.25
            if y > 17.25:
                ntype = -1
                y = 17.25
            G.add_node(node_id, pos = (x, y), z=z, type=ntype, b=bval)

            pairs = pair_re.findall(s)
            # Be defensive if n_neigh disagrees with how many pairs are present
            for nb_str, pore_str in pairs[: max(0, n_neigh) ]:
                nb, pore = int(nb_str), edge_ind
                edge_ind += 1
                if multigraph:
                    G.add_edge(node_id, nb, pore=pore)
                else:
                    if G.has_edge(node_id, nb):
                        data = G.get_edge_data(node_id, nb)
                        prev = data.get('pore')
                        if prev is None:
                            data['pore'] = pore
                        elif isinstance(prev, list):
                            if pore not in prev:
                                prev.append(pore)
                        else:
                            if pore != prev:
                                data['pore'] = [prev, pore]
                    else:
                        G.add_edge(node_id, nb, pore=pore)
    #G.remove_nodes_from(rem_list)
    G = nx.convert_node_labels_to_integers(
       G, first_label=0, ordering='sorted', label_attribute='orig_id')
    G.add_edge(0, 19)
    G.add_edge(0, 1)

    return G

def enumerate_triangles(G):
    # For MultiGraph, ignore parallel edges for triangle structure
    H = nx.Graph(G) if G.is_multigraph() else G
    if H.is_directed():
        H = H.to_undirected()

    tris = []
    for u in H:
        Nu = set(H[u])
        for v in Nu:
            if v <= u:
                continue
            common = Nu & set(H[v])
            for w in common:
                if w <= v:
                    continue
                tris.append((u, v, w))   # triangle as a 3-tuple of node IDs
    return tris

def build_delaunay_net(sid: SimInputData, inc: Incidence) \
    -> tuple(Graph, Edges, Triangles):
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
    G = load_net_to_networkx("net.txt", multigraph=True)
    pos = nx.get_node_attributes(G, "pos")
    x_vals = [x for (x, _) in pos.values()]
    xmin, xmax = min(x_vals), max(x_vals)
    xmid = (xmin + xmax) / 2.0
    pos_flipped = {n: (2*xmid - x, y) for n, (x, y) in pos.items()}
    nx.set_node_attributes(G, pos_flipped, "pos")
    sid.nsq = len(G.nodes())
    # pos = np.zeros((sid.nsq, 2))
    # for i, nodei in enumerate(G):
    #     pos[i] = G.nodes[nodei]["pos"]
    # print(pos)
    #pos = np.array(list(nx.get_node_attributes(G, 'pos').values()))
    pos = nx.get_node_attributes(G, "pos")
    #print(pos)
    tris = enumerate_triangles(G)
    
    sid.ne = len(G.edges())
    sid.n = 19
    sid.m = 16
    # create a set for edges that are indexes of the points
    edge_list = dict()
    boundary_edges = []
    boundary_nodes = []
    lens = []
    pipe_diams = []
    edge_index = 0

    triangles = Triangles()
    triangles_inc_row = []
    triangles_inc_col = []
    triangles_inc_data = []

    merge_matrix_row = []
    merge_matrix_col = []
    merge_matrix_data = []

    for n1, n2, n3 in tris:
        # for each edge of the triangle
        # sort the vertices
        # (sorting avoids duplicated edges being added to the set)
        # and add to the edges set
        lens_tr = (np.linalg.norm(np.array(pos[n1]) - np.array(pos[n2])), \
            np.linalg.norm(np.array(pos[n1]) - np.array(pos[n3])), \
            np.linalg.norm(np.array(pos[n2]) - np.array(pos[n3])))

        edge_index_list = []
        
        triangles.tlist.append((n1, n2, n3))
        triangles.boundary.append(0)
        triangles.centers.append((np.array(pos[n1]) + np.array(pos[n2]) + np.array(pos[n3])) / 3)
        triangles.volume.append(np.abs(pos[n1][0] * (pos[n2][1] - pos[n3][1]) \
            + pos[n2][0] * (pos[n3][1] - pos[n1][1]) + pos[n3][0] \
            * (pos[n1][1] - pos[n2][1])) / 2)
        
        for i, edge in enumerate((sorted((n1, n2)), \
            sorted((n1, n3)), sorted((n2, n3)))):
            node1, node2 = edge
            if (node1, node2) not in edge_list:
                edge_list[(node1, node2)] = edge_index
                cur_edge_index = edge_index
                lens.append(lens_tr[i])
                edge_index += 1
                boundary_edges.append(0)
            else:
                cur_edge_index = edge_list[(node1, node2)]
            edge_index_list.append(cur_edge_index)

            triangles_inc_row.append(cur_edge_index)
            triangles_inc_col.append(len(triangles.tlist) - 1)
            triangles_inc_data.append(1)

        merge_matrix_row.extend(2 * edge_index_list)
        merge_matrix_col.extend(np.roll(edge_index_list, 1))
        merge_matrix_col.extend(np.roll(edge_index_list, 2))
        for index in list(np.roll(edge_index_list, 2)) \
            + list(np.roll(edge_index_list, 1)):
            merge_matrix_data.append(lens[index] / 2)


    boundary_edges = np.array(boundary_edges)

    edge_list = list(edge_list)
    sid.ne = len(edge_list)

    sid.ntr = len(triangles.tlist)
    triangles.volume = np.array(triangles.volume)
    # triangles.volume = triangles.volume / np.average(triangles.volume) \
    #     * sid.V_tot / sid.ntr
    triangles.boundary = np.array(triangles.boundary)
    triangles.incidence = spr.csr_matrix((triangles_inc_data, (triangles_inc_row, triangles_inc_col)), shape=(sid.ne, sid.ntr))

    diams = np.ones(sid.ne)
    lens = np.array(lens)

    merge_matrix_data = np.array(merge_matrix_data) / np.average(lens)
    inc.merge = spr.csr_matrix((merge_matrix_data, (merge_matrix_row, \
        merge_matrix_col)), shape=(sid.ne, sid.ne)) * sid.merge_length
    lens = lens / np.average(lens)
    flow = np.zeros(len(edge_list))

    edge_triangles = np.array(np.sum(triangles.incidence, axis = 1))[:, 0]

    edges = Edges(diams, lens, flow, edge_list, boundary_edges, edge_triangles)

    triangles.volume = triangles.volume / np.average(triangles.volume) * sid.V_tot

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

    nx.set_node_attributes(graph, nx.get_node_attributes(G, "pos"), 'pos')
    graph.in_nodes = []
    graph.out_nodes = []
    graph.in_vec = np.zeros(sid.nsq)
    graph.out_vec = np.zeros(sid.nsq)
    for node in G.nodes():
        if G.nodes[node]['type'] == 1:
            graph.in_nodes.append(node)
            graph.in_vec[node] = 1
        if G.nodes[node]['type'] == -1:
            graph.out_nodes.append(node)
            graph.out_vec[node] = 1
    sid.Q_in = sid.qin * 2 * len(graph.in_nodes)
    # print("strange edege: ", graph[1][20]['d'])
    # raise ValueError
    return graph, edges, triangles
