""" Calculate pressure and flow in the system.

This module contains functions for solving the Hagen-Poiseuille and continuity
equations for pressure and flow. It assumes constant inflow boundary condition.
It constructs a result vector for the matrix equation (constant throughout the
simulation) and the matrix with coefficients corresponding to aforementioned
equation. Function solve_equation from module utils is used to solve the
equations for flow.

Notable functions
-------
solve_flow(SimInputData, Incidence, Graph, Edges, spr.csc_matrix) \
    -> numpy ndarray
    calculate pressure and update flow in network edges
"""

import numpy as np
import scipy.sparse as spr

from config import SimInputData
from incidence import Edges, Incidence
from network import Graph
from utils import solve_equation


def create_result_vector(sid:SimInputData, graph: Graph) -> spr.csc_matrix:
    """ Creates vector result for pressure calculation.
    
    This function builds a result vector for solving equation for pressure.
    For inlet and outlet nodes elements of the vector correspond explicitly
    to the pressure in nodes, for regular nodes elements of the vector equal
    0 correspond to flow continuity.

    Parameters
    -------
    sid : SimInputData class object
        all config parameters of the simulation

    graph : Graph class object
        network and all its properties
    
    Returns
    -------
    scipy sparse vector
        result vector for pressure calculation
    """
    # data, row, col = [], [], []
    # for node in graph.in_nodes:
    #     data.append(1)
    #     row.append(node)
    #     col.append(0)
    # return spr.csc_matrix((data, (row, col)), shape=(sid.n_nodes, 1))
    return graph.in_vec

def create_vector(sid:SimInputData) -> np.ndarray:
    """ Creates variable vector for pressure calculation.

    Parameters
    -------
    sid : SimInputData class object
        all config parameters of the simulation

    Returns
    -------
    numpy ndarray
        vector of pressure in nodes
    """
    return np.zeros(sid.n_nodes, dtype = float)

def solve_flow(sid: SimInputData, inc: Incidence, graph: Graph, edges: Edges, \
    pressure_b: spr.csc_matrix) -> np.ndarray:
    """ Calculates pressure and flow.

    Parameters
    -------
    sid : SimInputData class object
        all config parameters of the simulation

    inc : Incidence class object
        matrices of incidence

    edges : Edges class object
        all edges in network and their parameters

    pressure_b : scipy sparse vector
        result vector for pressure equation

    Returns
    -------
    pressure : numpy ndarray
        vector of pressure in nodes
    """
    # create matrix (nsq x nsq) for solving equations for pressure and flow
    # to find pressure in each node
    p_matrix = inc.incidence.T @ spr.diags(edges.fracture_lens \
        * edges.apertures ** 3 / edges.lens) @ inc.incidence
    # for all inlet nodes we set the same pressure, for outlet nodes we set
    # zero pressure; so for boundary nodes we zero the elements of p_matrix
    # and add identity for those rows
    #p_matrix = p_matrix.multiply(inc.middle) + inc.boundary
    p_matrix = p_matrix.multiply((1 - graph.in_vec - graph.out_vec)[:, np.newaxis]) + spr.diags(graph.in_vec + graph.out_vec)
    # solve matrix @ pressure = pressure_b
    pressure = solve_equation(p_matrix, pressure_b)
    # normalize pressure in inlet nodes to match condition for constant inlet
    # flow
    q_in = np.abs(np.sum(edges.fracture_lens * edges.apertures ** 3 \
        / edges.lens * (inc.inlet @ pressure)))
    #pressure *= sid.q_in * np.sum(edges.inlet) / q_in
    pressure *= sid.q_in / q_in
    # update flow in edges
    edges.flow = edges.apertures ** 3 / edges.lens * (inc.incidence @ pressure)
    return pressure
