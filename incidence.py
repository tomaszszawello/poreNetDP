""" Build classes for incidence and edge data and initialize them.

This module contains classes and functions converting network data to sparse
incidence matrices and vectors used for faster calculation.

Notable classes
-------
Incidence
    container for incidence matrices

Edges
    container for information on network edges

Notable functions
-------
create_matrices(SimInputData, Graph, Incidence) -> Edges
    initialize matrices and edge data in Incidence and Edges classes
"""

from scipy.stats import truncnorm
import numpy as np
import scipy.sparse as spr

from config import SimInputData
from network import Graph


class Incidence():
    """ Contains all necessary incidence matrices.

    This class is a container for all incidence matrices i.e. sparse matrices
    with non-zero indices for connections between edges and certain nodes.
    """
    incidence: spr.csr_matrix = spr.csr_matrix(0)
    "connections of all edges with all nodes (ne x nsq)"
    middle: spr.csr_matrix = spr.csr_matrix(0)
    "connections between all nodes but inlet and outlet (nsq x nsq)"
    boundary: spr.csr_matrix = spr.csr_matrix(0)
    "identity matrix for inlet and outlet nodes, zero elsewhere (nsq x nsq)"
    inlet: spr.csr_matrix = spr.csr_matrix(0)
    "connections of edges with inlet nodes (ne x nsq)"

class Edges():
    """ Contains all data connected with network edges.

    This class is a container for all information about network edges and their
    type in the network graph.
    """
    apertures: np.ndarray
    "apertures of edges"
    fracture_lens: np.ndarray
    "fracture lengths of edges"
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

    def __init__(self, apertures, fracture_lens, lens, flow, in_edges, \
        out_edges, edge_list):
        self.apertures = apertures
        self.fracture_lens = fracture_lens
        self.lens = lens
        self.flow = flow
        self.inlet = in_edges
        self.outlet = out_edges
        self.edge_list = edge_list

def create_matrices(sid: SimInputData, graph: Graph, inc: Incidence) -> Edges:
    """ Create incidence matrices and edges class for graph parameters.

    This function takes the network and based on its properties creates
    matrices of connections for different types of nodes and edges.
    It later updates the matrices in Incidence class and returns Edges class
    for easy access to the parameters of edges in the network.

    Parameters
    -------
    sid : SimInputData class object
        all config parameters of the simulation

    inc : Incidence class object
        matrices of incidence

    graph : Graph class object
        network and all its properties

    Returns
    -------
    edges : Edges class object
        all edges in network and their parameters
    """
    sid.n_edges = len(graph.edges())
    # data for standard incidence matrix (ne x nsq)
    data, row, col = [], [], []
    # vectors of edges parameters (ne)
    apertures, fracture_lens, lens, flow, edge_list = [], [], [], [], []
    # data for matrix keeping connections of only middle nodes (nsq x nsq)
    data_mid, row_mid, col_mid = [], [], []
    # data for diagonal matrix for input and output (nsq x nsq)
    data_bound, row_bound, col_bound = [], [], []
    # data for matrix keeping connections of only input nodes (ne x nsq)
    data_in, row_in, col_in = [], [], []
    reg_nodes = [] # list of regular nodes (not inlet or outlet)
    in_edges = np.zeros(sid.n_edges)
    out_edges = np.zeros(sid.n_edges)
    for i, e in enumerate(graph.edges()):
        n1, n2 = e
        b = graph[n1][n2]['b']
        l = graph[n1][n2]['length']
        if l == 0 or b == 0:
            print ('!!!!!!!!!!!!!!!!!')
        q = graph[n1][n2]['q']
        # if (n1 in graph.in_nodes and n2 not in graph.in_nodes) or (n1 in graph.out_nodes and n2 not in graph.out_nodes):
        #     data.append(1)
        #     row.append(i)
        #     col.append(n1)
        #     data.append(-1)
        #     row.append(i)
        #     col.append(n2)
        # else:
        data.append(-1)
        row.append(i)
        col.append(n1)
        data.append(1)
        row.append(i)
        col.append(n2)
        apertures.append(b)
        lens.append(l)
        flow.append(q)
        fracture_lens.append(graph[n1][n2]['area'] / b)
        edge_list.append((n1, n2))
        # middle matrix has 1 in coordinates of all connected regular nodes
        # so it can be later multiplied elementwise by any other matrix for
        # which we want to set specific boundary condition for inlet and outlet
        if (n1 not in graph.in_nodes and n1 not in graph.out_nodes) \
            and (n2 not in graph.in_nodes and n2 not in graph.out_nodes):
            data_mid.extend((1, 1))
            row_mid.extend((n1, n2))
            col_mid.extend((n2, n1))
            reg_nodes.extend((n1, n2))
        # in middle matrix we include also connection to the inlet node, but
        # only from "one side" (rows for inlet nodes must be all equal 0)
        # in inlet matrix, we include full incidence for inlet nodes and edges
        elif n1 not in graph.in_nodes and n2 in graph.in_nodes:
            data_mid.append(1)
            row_mid.append(n1)
            col_mid.append(n2)
            data_in.append(1)
            row_in.append(i)
            col_in.append(n1)
            data_in.append(-1)
            row_in.append(i)
            col_in.append(n2)
            in_edges[i] = 1
        elif n1 in graph.in_nodes and n2 not in graph.in_nodes:
            data_mid.append(1)
            row_mid.append(n2)
            col_mid.append(n1)
            data_in.append(1)
            row_in.append(i)
            col_in.append(n2)
            data_in.append(-1)
            row_in.append(i)
            col_in.append(n1)
            in_edges[i] = 1
        elif (n1 not in graph.out_nodes and n2 in graph.out_nodes) \
            or (n1 in graph.out_nodes and n2 not in graph.out_nodes):
            out_edges[i] = 1
    # in boundary matrix, we set identity to rows corresponding to inlet and
    # outlet nodes
    for node in graph.in_nodes + graph.out_nodes:
        data_bound.append(1)
        row_bound.append(node)
        col_bound.append(node)
    reg_nodes = list(set(reg_nodes))
    # in middle matrix, we also include 1 on diagonal for regular nodes, so the
    # diagonal of a given matrix is not zeroed when multiplied elementwise
    for node in reg_nodes:
        data_mid.append(1)
        row_mid.append(node)
        col_mid.append(node)
    # make edge parameters dimensionless
    # apertures = np.array(truncnorm.rvs(-1, 1, loc = 1, \
    #     scale = 0.01, size = len(list(apertures))))
    apertures = np.array(apertures) / sid.b0
    lens = np.array(lens) / sid.l0
    sid.w0 = np.average(fracture_lens)
    print(sid.b0, sid.w0, sid.l0)
    #fracture_lens = np.array(fracture_lens) / sid.l0
    fracture_lens = np.array(fracture_lens) / sid.w0
    inc.incidence = spr.csr_matrix((data, (row, col)), shape=(sid.n_edges, \
        sid.n_nodes))
    inc.middle = spr.csr_matrix((data_mid, (row_mid, col_mid)), \
        shape = (sid.n_nodes, sid.n_nodes))
    inc.boundary = spr.csr_matrix((data_bound, (row_bound, col_bound)), \
        shape = (sid.n_nodes, sid.n_nodes))
    inc.inlet = spr.csr_matrix((data_in, (row_in, col_in)), \
        shape = (sid.n_edges, sid.n_nodes))
    apertures, fracture_lens, lens, flow = np.array(apertures), \
        np.array(fracture_lens), np.array(lens), np.array(flow)
    edges = Edges(apertures, fracture_lens, lens, flow, in_edges, out_edges, \
        edge_list)
    return edges
    