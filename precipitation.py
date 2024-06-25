""" Calculate substance C concentration (precipitation).

This module contains functions for solving the advection-reaction equation for
substance C concentration. It constructs a result vector for the matrix
equation (dependent on B concentration, so recalculated each iteration)
and the matrix with coefficients corresponding to aforementioned equation.
Function solve_equation from module utils is used to solve the equation for
C concentration. If precipitation is off, then C concentration is assumed
to be zero.

Notable functions
-------
solve_precipitation(SimInputData, Incidence, Graph, Edges, np.ndarray) \
    -> np.ndarray
    calculate substance C concentration
"""

import numpy as np
import scipy.sparse as spr

from config import SimInputData
from network import Edges, Graph
from incidence import Incidence
from utils import solve_equation


def create_vector(sid: SimInputData, inc: Incidence, graph: Graph, \
    edges: Edges, cb: np.ndarray) -> spr.csc_matrix:
    """ Creates vector result for C concentration calculation.

    This function creates the result vector used to solve the equation for
    substance C concentration. For inlet nodes elements of the vector
    correspond explicitly to the concentration in nodes, for other they
    include part from the mixing condition and part from dissolution.

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

    cb : numpy ndarray (nsq)
        vector of substance B concentration

    Returns
    -------
    cc_b : scipy sparse csc matrix (nsq x 1)
        vector result for substance C concentration calculation
    """
    # find incidence for cb (only upstream flow matters)
    cb_inc = np.abs(inc.incidence.T @ (spr.diags(edges.flow) \
        @ inc.incidence > 0))
    # find vector with non-diagonal coefficients
    qc = edges.flow / (sid.K - 1) * (np.exp(-sid.Da / (1 + sid.G * \
        edges.diams) * edges.diams * edges.lens / np.abs(edges.flow)) - \
        np.exp(-sid.Da * sid.K / (1 + sid.G * sid.K * edges.diams) \
        * edges.diams * edges.lens / np.abs(edges.flow)))
    qc_matrix = np.abs(inc.incidence.T @ spr.diags(qc) @ inc.incidence)
    cb_matrix = cb_inc.multiply(qc_matrix)
    cb_matrix.setdiag(np.zeros(sid.nsq)) # set diagonal to zero
    cc_b = -cb_matrix @ cb
    cc_b = cc_b * (1 - graph.in_vec) + graph.in_vec * sid.cc_in
    # for node in graph.in_nodes:
    #     cc_b[node] = sid.cc_in # set result for input nodes to cc_in
    return cc_b

def solve_precipitation(sid: SimInputData, inc: Incidence, graph: Graph, \
    edges: Edges, cb) -> np.ndarray:
    """ Calculate C concentration.

    This function solves the advection-reaction equation for substance C
    concentration. We assume that we can always precipitate more. If
    precipitation is disabled in simulation, we only return vector of zeros.

    Parameters
    -------
    sid : simInputData class object
        all config parameters of the simulation

    inc : Incidence class object
        matrices of incidence

    graph : Graph class object
        network and all its properties

    edges : Edges class object
        all edges in network and their parameters

    cb : numpy array (nsq)
        vector of substance B concentration

    Returns
    -------
    cc : numpy array (nsq)
        vector of substance C concentration
    """
    # include precipitation
    if sid.include_cc:
        return solve(sid, inc, graph, edges, cb)
    # don't include precipitation
    else:
        return np.zeros(sid.nsq)


def solve(sid: SimInputData, inc: Incidence, graph: Graph, edges: Edges, \
    cb: np.ndarray) -> np.ndarray:
    """ Calculate C concentration.

    This function solves the advection-reaction equation for substance C
    concentration. We assume precipitation is always possible.

    Parameters
    -------
    sid : simInputData class object
        all config parameters of the simulation

    inc : Incidence class object
        matrices of incidence

    graph : Graph class object
        network and all its properties

    edges : Edges class object
        all edges in network and their parameters

    cb : numpy array (nsq)
        vector of substance B concentration

    Returns
    -------
    cc : numpy array (nsq)
        vector of substance C concentration
    """
    # find incidence for cc (only upstream flow matters)
    cc_inc = np.abs(inc.incidence.T @ (spr.diags(edges.flow) \
        @ inc.incidence > 0))
    # find vector with non-diagonal coefficients
    qc = edges.flow * np.exp(-sid.Da * sid.K / (1 + sid.G * sid.K * \
        edges.diams) * edges.diams * edges.lens / np.abs(edges.flow))
    qc_matrix = np.abs(inc.incidence.T @ spr.diags(qc) @ inc.incidence)
    cc_matrix = cc_inc.multiply(qc_matrix)
    # find diagonal coefficients (inlet flow for each node)
    diag = -np.abs(inc.incidence.T) @ np.abs(edges.flow) / 2
    diag = diag * (1 - graph.in_vec + graph.out_vec) + graph.in_vec
    diag += 1 * (diag == 0)
    cc_matrix.setdiag(diag) # replace diagonal
    cc_b = create_vector(sid, inc, graph, edges, cb)
    cc = solve_equation(cc_matrix, cc_b)
    return cc
