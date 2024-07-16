""" Calculate substance B concentration (dissolution).

This module contains functions for solving the advection-reaction equation for
substance B concentration. It constructs a result vector for the matrix
equation (constant throughout the simulation) and the matrix with coefficients
corresponding to aforementioned equation. Function solve_equation from module
utils is used to solve the equation for B concentration.

Notable functions
-------
solve_dissolution(SimInputData, Incidence, Graph, Edges, spr.csc_matrix) \
    -> np.ndarray
    calculate substance B concentration
"""

import numpy as np
import scipy.sparse as spr

from config import SimInputData
from network import Edges, Graph
from incidence import Incidence
from utils import solve_equation


def create_vector(sid: SimInputData, graph: Graph) -> spr.csc_matrix:
    """ Create vector result for B concentration calculation.

    For inlet nodes elements of the vector correspond explicitly
    to the concentration in nodes, for other it corresponds to
    mixing condition.

    Parameters
    -------
    sid : simInputData class object
        all config parameters of the simulation
        nsq - number of nodes in the network
        cb_in - substance B concentration in inlet nodes

    graph : Graph class object
        network and all its properties
        in_nodes - inlet nodes

    Returns
    ------
    scipy sparse vector
        result vector for B concentration calculation
    """
    # data, row, col = [], [], []
    # for node in graph.in_nodes:
    #     data.append(sid.cb_in)
    #     row.append(node)
    #     col.append(0)
    # return spr.csc_matrix((data, (row, col)), shape=(sid.nsq, 1))
    return sid.cb_in * graph.in_vec

def solve_dissolution(sid: SimInputData, inc: Incidence, graph: Graph, \
    edges: Edges, cb_b: spr.csc_matrix) -> np.ndarray:
    """ Calculate B concentration.

    This function solves the advection-reaction equation for substance B
    concentration. We assume substance A is always available.

    Parameters
    -------
    sid : simInputData class object
        all config parameters of the simulation
        Da : float
        G : float

    inc : Incidence class object
        matrices of incidence
        incidence : scipy sparse csr matrix (ne x nsq)

    graph : Graph class object
        network and all its properties
        in_nodes : list
        out_nodes : list

    edges : Edges class object
        all edges in network and their parameters
        diams : numpy ndarray (ne)
        lens : numpy ndarray (ne)
        flow : numpy ndarray (ne)

    cb_b : scipy sparse csc matrix (nsq x 1)
        result vector for substance B concentration calculation

    Returns
    -------
    cb : numpy array (nsq)
        vector of substance B concentration in nodes
    """
    # find incidence for cb (only upstream flow matters)
    cb_inc = 1 * (inc.incidence.T @ (spr.diags(edges.flow) \
        @ inc.incidence > 0) != 0)
    # cb_inc = np.abs(inc.incidence.T @ (spr.diags(edges.flow) \
    #    @ inc.incidence > 0))
    alpha = np.clip(1 / np.abs(sid.Da / (1 + sid.G * edges.diams) \
        * edges.diams * edges.lens / edges.flow), 0, 1)
    # find vector with non-diagonal coefficients
    qc = edges.flow * (1 - np.abs(alpha * sid.Da / (1 + sid.G * edges.diams) \
        * edges.diams * edges.lens / edges.flow))
    qc = np.array(np.ma.fix_invalid(qc, fill_value = 0))
    qc_matrix = np.abs(inc.incidence.T @ spr.diags(qc) @ inc.incidence)
    cb_matrix = cb_inc.multiply(qc_matrix)
    # find diagonal coefficients (inlet flow for each node)
    #diag = np.array(-np.abs(inc.incidence.T @ spr.diags(edges.flow) @ inc.incidence).sum(axis = 1).reshape((1, sid.nsq)) / 2)[0] #
    
    #diag = graph.in_vec + 1 * (diag == 0) + diag * (1 + graph.out_vec - graph.in_vec)
    
    diag = -np.abs(inc.incidence.T) @ np.abs(edges.flow) / 2
    # # set diagonal for input nodes to 1
    # for node in graph.in_nodes:
    #     diag[node] = 1
    # # multiply diagonal for output nodes (they have no outlet, so inlet flow
    # # is equal to whole flow); also fix for nodes which are connected only to
    # # other out_nodes - without it we get a singular matrix (whole row of
    # # zeros)
    # for node in graph.out_nodes:
    #     if diag[node] != 0:
    #         diag[node] *= 2
    #     else:
    #         diag[node] = 1
    # # fix for nodes with no connections
    # for i, node in enumerate(diag):
    #     if node == 0:
    #         diag[i] = 1

    diag = diag * (1 - graph.in_vec + graph.out_vec) + graph.in_vec
    diag += 1 * (diag == 0)
    # replace diagonal
    cb_matrix.setdiag(diag)
    cb = solve_equation(cb_matrix, cb_b)
    return cb
