""" Calculate solvent concentration.

This module contains functions for solving the advection-reaction equation for
solvent concentration. It constructs a result vector for the matrix
equation (constant throughout the simulation) and the matrix with coefficients
corresponding to aforementioned equation. Function solve_equation from module
utils is used to solve the equation for solvent concentration.

Notable functions
-------
solve_dissolution(SimInputData, Incidence, Graph, Edges, spr.csc_matrix) \
    -> numpy ndarray
    calculate solvent concentration
"""

import numpy as np
import scipy.sparse as spr

from config import SimInputData
from incidence import Edges, Incidence
from network import Graph
from utils import solve_equation


def create_result_vector(sid:SimInputData, graph: Graph) -> spr.csc_matrix:
    """ Creates vector result for solvent concentration calculation.
    
    This function builds a result vector for solving advection-reaction
    equation for solvent concentration. For inlet nodes elements of the
    vector correspond explicitly to the concentration in nodes, for other it
    corresponds to mixing condition.

    Parameters
    -------
    sid : SimInputData class object
        all config parameters of the simulation

    graph : Graph class object
        network and all its properties
    
    Returns
    ------
    scipy sparse csc matrix
        result vector for solvent concentration calculation
    """
    data, row, col = [], [], []
    for node in graph.in_nodes:
        data.append(sid.concentration_in)
        row.append(node)
        col.append(0)
    return spr.csc_matrix((data, (row, col)), shape=(sid.n_nodes, 1))

def create_vector(sid:SimInputData) -> np.ndarray:
    """ Creates variable vector for solvent concentration calculation.

    Parameters
    -------
    sid : SimInputData class object
        all config parameters of the simulation

    Returns
    -------
    numpy ndarray
        vector of solvent concentration in nodes
    """
    return np.zeros(sid.n_nodes, dtype = float)

def solve_dissolution(sid: SimInputData, inc: Incidence, graph: Graph, \
    edges: Edges, concentration_b: spr.csc_matrix) -> np.ndarray:
    """ Calculates solvent concentration.

    This function solves the advection-reaction equation for solvent
    concentration. We assume dissolving substance is always available.

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

    concentration_b : scipy sparse csc matrix
        result vector for solvent concentration calculation

    Returns
    -------
    concentration : numpy array
        vector of solvent concentration in nodes
    """
    # find incidence for concentration calculation (only upstream flow matters)
    concentration_inc =  1 * (inc.incidence.T @ (spr.diags(edges.flow) \
        @ inc.incidence > 0) != 0)
    # find vector with non-diagonal coefficients (exponential decrease of
    # along concentration along edges)
    qc = edges.fracture_lens * edges.flow * np.exp(-np.abs(sid.Da \
        / (1 + sid.G * edges.apertures) * edges.lens / edges.flow))
    qc_matrix = np.abs(inc.incidence.T @ spr.diags(qc) @ inc.incidence)
    concentration_matrix = concentration_inc.multiply(qc_matrix)
    diag_old = concentration_matrix.diagonal()
    # find diagonal coefficients (inlet flow for each node)
    diag = -np.abs(inc.incidence.T) @ np.abs(edges.flow * edges.fracture_lens) \
        / 2
    # set diagonal for inlet nodes to 1 - there concentration equals the inlet
    # concentration sid.concentration_in
    # for node in graph.in_nodes:
    #     diag[node] = 1
    # # multiply diagonal for output nodes (they have no outlet, so inlet flow
    # # is equal to whole flow)
    # for node in graph.out_nodes:
    #     diag[node] *= 2
    # fix for nodes with 0 flow - without it we get a singular matrix
    #diag = diag * (diag != 0) + 1 * (diag == 0)
    # replace diagonal
    #concentration_matrix.setdiag(diag)
    diag = diag * (1 - graph.in_vec + graph.out_vec) + graph.in_vec
    diag += 1 * (diag == 0)
    # replace diagonal
    #cb_matrix.setdiag(diag)
    concentration_matrix = concentration_matrix + spr.diags(diag - diag_old)
    # calculate concentration of solvent in whole system
    concentration = solve_equation(concentration_matrix, concentration_b)
    return concentration
