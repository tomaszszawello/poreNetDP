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
    -> np.ndarray
    calculate pressure and update flow in network edges
"""

import numpy as np
import scipy.sparse as spr

from config import SimInputData
from network import Edges, Graph
from incidence import Incidence
from utils import solve_equation


def create_vector(sid: SimInputData, graph: Graph) -> spr.csc_matrix:
    """ Creates vector result for pressure calculation.

    For inlet and outlet nodes elements of the vector correspond explicitly
    to the pressure in nodes, for regular nodes elements of the vector equal
    0 correspond to flow continuity.

    Parameters
    -------
    sid : SimInputData class object
        all config parameters of the simulation
        nsq - number of nodes in the network squared

    graph : Graph class object
        network and all its properties
        in_nodes - inlet nodes

    Returns
    -------
    scipy sparse vector
        result vector for pressure calculation
    """
    return graph.in_vec

def solve_flow(sid: SimInputData, inc: Incidence, graph: Graph, edges: Edges, \
    pressure_b: spr.csc_matrix) -> np.ndarray:
    """ Calculates pressure and flow.

    Parameters
    -------
    sid : SimInputData class object
        all config parameters of the simulation
        qin - characteristic flow for inlet edge

    inc : Incidence class object
        matrices of incidence; here all of shape (ne x nsq)
        incidence - incidence of all nodes and edges
        middle - incidence of nodes and edges for all but inlet and outlet
        boundary - incidence of nodes and edges for inlet and outlet
        inlet - incidence of nodes and edges for inlet

    graph : Graph class object
        network and all its properties
        in_nodes - inlet nodes

    edges : Edges class object
        all edges in network and their parameters
        diams - diameters
        lens - lengths

    pressure_b : scipy sparse vector
        result vector for pressure equation

    Returns
    -------
    pressure : numpy ndarray
        vector of pressure in nodes
    """
    # create matrix (nsq x nsq) for solving equations for pressure and flow
    # to find pressure in each node
    p_matrix = inc.incidence.T @ spr.diags(edges.diams ** 4 / edges.lens) \
        @ inc.incidence
    # for all inlet nodes we set the same pressure, for outlet nodes we set
    # zero pressure; so for boundary nodes we zero the elements of p_matrix
    # and add identity for those rows
    p_matrix = p_matrix.multiply((1 - graph.in_vec - graph.out_vec)[:, np.newaxis]) + spr.diags(graph.in_vec + graph.out_vec)
    diag = p_matrix.diagonal()
    diag_old = diag.copy()
    # fix for nodes with no connections
    diag += 1 * (diag == 0)
    p_matrix += spr.diags(diag - diag_old)
    # solve matrix @ pressure = pressure_b
    pressure = solve_equation(p_matrix, pressure_b)
    # normalize pressure in inlet nodes to match condition for constant inlet
    # flow
    q_in = np.abs(np.sum(edges.diams ** 4 / edges.lens * (inc.inlet \
        @ pressure)))
    pressure *= sid.Q_in / q_in
    # update flow
    edges.flow = edges.diams ** 4 / edges.lens * (inc.incidence @ pressure)
    #p_continuity = p_matrix @ pressure * (1 - graph.in_vec - graph.out_vec)
    
    return pressure
