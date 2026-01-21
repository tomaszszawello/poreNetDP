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

import numpy as np
import scipy.sparse as spr

from config import SimInputData
from network import Edges, Graph


class Incidence():
    """ Contains all necessary incidence matrices.

    This class is a container for all incidence matrices i.e. sparse matrices
    with non-zero indices for connections between edges and certain nodes.

    Attributes
    -------
    incidence : scipy sparse csr matrix (ne x nsq)
        connections of all edges with all nodes

    middle : scipy sparse csr matrix (nsq x nsq)
        connections between all nodes but inlet and outlet

    boundary : scipy sparse csr matrix (nsq x nsq)
        identity matrix for inlet and outlet nodes, zero elsewhere

    inlet : scipy sparse csr matrix (ne x nsq)
        connections of edges with inlet nodes
    """
    incidence: spr.csr_matrix = spr.csr_matrix(0)
    "connections of all edges with all nodes (ne x nsq)"
    middle: spr.csr_matrix = spr.csr_matrix(0)
    "connections between all nodes but inlet and outlet (nsq x nsq)"
    boundary: spr.csr_matrix = spr.csr_matrix(0)
    "identity matrix for inlet and outlet nodes, zero elsewhere (nsq x nsq)"
    inlet: spr.csr_matrix = spr.csr_matrix(0)
    "connections of edges with inlet nodes (ne x nsq)"
    merge: spr.csr_matrix = spr.csr_matrix(0)
    "threshold values for edge merging (ne x ne)"
    plot: spr.csr_matrix = spr.csr_matrix(0)
    "matrix for plotting diameters with merging (ne x ne)"


def create_matrices(sid: SimInputData, graph: Graph, inc: Incidence, \
    edges: Edges) -> None:
    """ Create incidence matrices and edges class for graph parameters.

    This function takes the network and based on its properties creates
    matrices of connections for different types of nodes, edges and cells
    (triangles). It later updates the matrices in Incidence class and returns
    Edges class for easy access to the parameters of edges in the network.

    Parameters
    -------
    sid : SimInputData class object
        all config parameters of the simulation
        ne - number of edges
        nsq - number of nodes in the network squared

    inc : Incidence class object
        matrices of incidence
        incidence - connections of all edges with all nodes
        middle - connections between all nodes but inlet and outlet
        boundary - identity matrix for inlet and outlet nodes, zero elsewhere
        inlet - connections of edges with inlet nodes
        triangles - assignment of edges to neighbouring triangles

    graph : Graph class object
        network and all its properties
        in_nodes - inlet nodes
        out_nodes - outlet nodes
        boundary_edges - edges assuring PBC

    Returns
    -------
    edges : Edges class object
        all edges in network and their parameters
    """
    # data for standard incidence matrix (ne x nsq)
    data, row, col = [], [], []
    # vectors of edges parameters (ne)
    data_mid, row_mid, col_mid = [], [], []
    # data for diagonal matrix for input and output (nsq x nsq)
    data_bound, row_bound, col_bound = [], [], []
    # data for matrix keeping connections of only input nodes (ne x nsq)
    data_in, row_in, col_in = [], [], []
    reg_nodes = [] # list of regular nodes (not inlet or outlet)
    in_edges = np.zeros(sid.ne)
    out_edges = np.zeros(sid.ne)
    nodes = list(graph.nodes())
    node_to_index = {node: idx for idx, node in enumerate(nodes)}
    for i, e in enumerate(edges.edge_list_draw):
        n1 = node_to_index[e[0]]
        n2 = node_to_index[e[1]]
        data.append(-1)
        row.append(i)
        col.append(n1)
        data.append(1)
        row.append(i)
        col.append(n2)
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
        # if (n1 in graph.in_nodes + graph.out_nodes and n2 \
        #     in graph.in_nodes + graph.out_nodes):
        #     edges.boundary_list[i] = 1
        #     if n1 in graph.in_nodes and n2 in graph.in_nodes:
        #         in_edges[i] = 1
        #     if n1 in graph.out_nodes and n2 in graph.out_nodes:
        #         out_edges[i] = 1
    # in boundary matrix, we set identity to rows corresponding to inlet and
    # outlet nodes
    for node in np.concatenate((graph.in_nodes, graph.out_nodes)):
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
    inc.incidence = spr.csr_matrix((data, (row, col)), shape=(sid.ne, sid.nsq))
    inc.middle = spr.csr_matrix((data_mid, (row_mid, col_mid)), \
        shape = (sid.nsq, sid.nsq))
    inc.boundary = spr.csr_matrix((data_bound, (row_bound, col_bound)), \
        shape = (sid.nsq, sid.nsq))
    inc.inlet = spr.csr_matrix((data_in, (row_in, col_in)), \
        shape = (sid.ne, sid.nsq))
    inc.plot = spr.csr_matrix(spr.diags(np.ones(sid.ne)))
    # we calculate how many triangles each edge has as neighbors (1 or 2)
    edges.inlet = in_edges
    edges.outlet = out_edges

import numpy as np
import scipy.sparse as spr
import networkx as nx

import numpy as np
import scipy.sparse as spr
import networkx as nx

def build_edge_incidence_right(inc: Incidence, graph: Graph) -> spr.csr_matrix:
    """
    Build an edge–edge incidence matrix E such that:

      For each edge e with endpoints (nL, nR) ordered by x-coordinate
      (x(nL) <= x(nR)), row e has 1s in columns f where:

        - edge f is incident to nR (the *right* end of e), and
        - if m is the other endpoint of f (m != nR),
          then x(m) > x(nR)  (edge f goes further to the right).

      All other entries are 0.

    Shape: (ne, ne); rows & cols ordered like rows of inc.incidence.

    Assumptions:
    - inc.incidence is (ne x nnodes) with ±1 for edge–node incidence.
    - The columns of inc.incidence correspond to `nodes = list(graph.nodes())`
      in that exact order.
    - Graph nodes are your (i, j) tuples from diamond_lattice_graph.
    - Node positions are stored as attribute 'pos': node -> (x, y).
    """

    B = inc.incidence.tocsr()        # (ne, nnodes)
    ne, nnodes = B.shape

    # Node ordering used for columns of B
    nodes = list(graph.nodes())
    if len(nodes) != nnodes:
        raise ValueError(
            "Number of graph nodes and incidence columns differ; "
            "ensure incidence was built with the same node order."
        )

    # absolute incidence: 1 if edge incident to node
    B_abs = (B != 0).astype(int)
    B_csr = B_abs.tocsr()
    B_csc = B_abs.tocsc()

    # x-coordinates per column index j
    pos = nx.get_node_attributes(graph, "pos")  # node -> (x, y)
    x_coords = np.zeros(nnodes, dtype=float)
    for j, n in enumerate(nodes):
        x_coords[j] = pos[n][0]

    # We'll fill a LIL matrix then convert to CSR
    E = spr.lil_matrix((ne, ne), dtype=int)

    # Loop over edges e
    for e in range(ne):
        # endpoints of edge e: two node indices (columns) where B[e,:] != 0
        endpoints_e = B_csr[e].indices
        if endpoints_e.size != 2:
            # should not happen in this lattice, but guard anyway
            continue

        j1, j2 = endpoints_e
        # pick right endpoint: larger x
        if x_coords[j1] <= x_coords[j2]:
            idx_right = j2
        else:
            idx_right = j1

        xR = x_coords[idx_right]

        # all edges incident to the right endpoint
        edges_at_right = B_csc[:, idx_right].indices
        if edges_at_right.size == 0:
            continue

        # among them, take only edges that go further right from that node
        for f in edges_at_right:
            if f == e:
                continue  # skip self

            endpoints_f = B_csr[f].indices
            if endpoints_f.size != 2:
                continue

            k1, k2 = endpoints_f
            # other node of f, relative to right endpoint
            if k1 == idx_right:
                other = k2
            elif k2 == idx_right:
                other = k1
            else:
                # f should contain idx_right if edges_at_right is correct,
                # but guard anyway
                continue

            # only keep f if it goes further right than the right endpoint
            if x_coords[other] > xR:
                E[e, f] = 1

    return E.tocsr()
