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
from data import Data
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
    # data, row, col = [], [], []
    # for node in graph.in_nodes:
    #     data.append(1)
    #     row.append(node)
    #     col.append(0)
    # return spr.csc_matrix((data, (row, col)), shape=(sid.nsq, 1))
    
    # pressure_b = np.concatenate([2 * np.ones(sid.n // 2), np.ones(sid.n // 2), np.zeros(sid.nsq - sid.n)])
    # sid.Q_in = np.sum(pressure_b)
    # return pressure_b
    
    return graph.in_vec

def solve_flow(sid: SimInputData, inc: Incidence, graph: Graph, edges: Edges, data: Data, \
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
    pressure_b_cb = graph.in_vec_a
    pressure_b_cc = graph.in_vec_b
    # create matrix (nsq x nsq) for solving equations for pressure and flow
    # to find pressure in each node
    
    # cond_e = edges.diams ** 4 / edges.lens
    # cond_down = inc.right @ cond_e   # (ne,)
    # cond_zero = 1 * (inc.right @ (1 * (cond_e == 0)) > 0)

    # A = inc.right.tocsr()
    # ne = A.shape[0]

    # cond_down_min = np.full(ne, np.inf)

    # for e in range(ne):
    #     start = A.indptr[e]
    #     end   = A.indptr[e+1]
    #     cols  = A.indices[start:end]   # neighbor edges f of e
    #     if cols.size > 0:
    #         cond_down_min[e] = cond_e[cols].min()

    # cond = cond_e.copy()
    # #mask = (inc.right @ np.ones(sid.ne) > 0) * (cond_e > 0) * (cond_down_min > 0)
    # mask = (cond_e + cond_down_min > 0)
    # cond[mask] = cond_e[mask] * cond_down_min[mask] / (cond_e[mask] + cond_down_min[mask])
    # #cond[mask] = cond_e[mask] * cond_down[mask] / (cond_e[mask] + cond_down[mask])
    # cond = cond * (1 - cond_zero) + edges.outlet * cond_e
    # edges.cond = cond

   
   
    # # local Poiseuille conductance for each edge
    # cond_e = edges.diams**4 / edges.lens     # shape (ne,)

    # A = inc.right.tocsr()                    # edge->right-neighbour adjacency
    # ne = A.shape[0]

    # # For each edge e: minimal cond_e of its right neighbours
    # cond_down_min = np.full(ne, np.nan)      # NaN = "no neighbour"
    # has_neighbor  = np.zeros(ne, dtype=bool)
    # zero_neighbor = np.zeros(ne, dtype=bool) # True if any right neighbour has cond=0

    # for e in range(ne):
    #     start = A.indptr[e]
    #     end   = A.indptr[e+1]
    #     cols  = A.indices[start:end]         # neighbour edges f of e

    #     if cols.size == 0:
    #         continue  # no right neighbours → leave NaN, has_neighbor[e]=False

    #     has_neighbor[e] = True
    #     neigh_conds = cond_e[cols]

    #     cond_down_min[e] = neigh_conds.min()
    #     zero_neighbor[e] = np.any(neigh_conds == 0.0)

    # # start from local conductance
    # cond = cond_e.copy()

    # outlet_mask = edges.outlet.astype(bool)

    # # 1) Edges that *do* have right neighbours and are *not* outlets
    # active = has_neighbor & (~outlet_mask)

    # # 2) Among them, those with at least one zero-conductance neighbour ⇒ cond = 0
    # mask_zero = active & zero_neighbor
    # cond[mask_zero] = 0.0

    # # 3) Remaining active edges with strictly positive self & downstream cond:
    # mask_harm = active & (~zero_neighbor) & (cond_e > 0) & (cond_down_min > 0)

    # g0   = cond_e[mask_harm]
    # gmin = cond_down_min[mask_harm]

    # # harmonic mean of g0 and gmin
    # #cond[mask_harm] = g0 * gmin / (g0 + gmin)
    # cond[mask_harm] = 1 / (1 / g0 + sid.cond_weight / gmin)

    # # 4) Outlets: enforce original conductance (even if they had neighbours)
    # cond[outlet_mask] = cond_e[outlet_mask]

    # # Result
    # edges.cond = cond
    cond = edges.diams**4 / edges.lens
    edges.cond = cond

    p_matrix = inc.incidence.T @ spr.diags(cond) \
        @ inc.incidence
    # for all inlet nodes we set the same pressure, for outlet nodes we set
    # zero pressure; so for boundary nodes we zero the elements of p_matrix
    # and add identity for those rows
    p_matrix_cb = p_matrix.multiply(1 - pressure_b_cb[:, np.newaxis] - graph.out_vec[:, np.newaxis]) + spr.diags(pressure_b_cb + graph.out_vec)
    p_matrix_cc = p_matrix.multiply(1 - pressure_b_cc[:, np.newaxis] - graph.out_vec[:, np.newaxis]) + spr.diags(pressure_b_cc + graph.out_vec)
    #p_matrix = p_matrix.multiply(inc.middle) + inc.boundary
    diag = p_matrix_cb.diagonal()
    # fix for nodes with no connections
    diag_old = diag.copy()
    diag += 1 * (diag == 0)
    # replace diagonal
    p_matrix_cb += spr.diags(diag - diag_old)
    diag = p_matrix_cc.diagonal()

    # fix for nodes with no connections
    diag_old = diag.copy()
    diag += 1 * (diag == 0)
    # replace diagonal
    p_matrix_cc += spr.diags(diag - diag_old)
    # replace diagonal

    # solve matrix @ pressure = pressure_b
    print("Solving b pressure")
    pressure_cb = solve_equation(p_matrix_cb, pressure_b_cb)
    print("Solving c pressure")
    pressure_cc = solve_equation(p_matrix_cc, pressure_b_cc)
    print("Pressure solved")
    flow_cb = np.abs(cond * (inc.incidence @ pressure_cb))
    flow_cc = np.abs(cond * (inc.incidence @ pressure_cc))
    data.cond_ratio_cb.append(np.sum(flow_cb * (np.abs(inc.incidence) @ graph.out_vec_b > 0)) / np.sum(flow_cb * (np.abs(inc.incidence) @ graph.in_vec_a > 0)))
    data.cond_ratio_cc.append(np.sum(flow_cc * (np.abs(inc.incidence) @ graph.out_vec_a > 0)) / np.sum(flow_cc * (np.abs(inc.incidence) @ graph.in_vec_b > 0)))
    # normalize pressure in inlet nodes to match condition for constant inlet
    # flow
    pressure = pressure_cb * (1 + sid.q_rate) + pressure_cc * (1 - sid.q_rate)


    
    q_in = np.abs(np.sum(cond * (inc.inlet \
        @ pressure)))
    pressure *= sid.Q_in / q_in
    # update flow
    edges.flow = cond * (inc.incidence @ pressure)
    p_continuity = p_matrix @ pressure * (1 - graph.in_vec - graph.out_vec)
    print(np.sum(np.abs(p_continuity)))
    Q_in = np.sum(edges.inlet * np.abs(edges.flow))
    Q_out = np.sum(edges.outlet * np.abs(edges.flow))
    q_in = np.abs(np.sum(cond * (inc.inlet \
        @ pressure)))
    print('Q_in =', Q_in, 'Q_out =', Q_out)
    print(np.sum(edges.inlet), np.sum(edges.outlet), q_in)

    return pressure

def solve_flow_nodes(sid: SimInputData, inc: Incidence, graph: Graph, edges: Edges, data: Data, \
    pressure_b: spr.csc_matrix, node_clogging: np.ndarray) -> np.ndarray:
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
    pressure_b_cb = graph.in_vec_a
    pressure_b_cc = graph.in_vec_b

    # edges.cond = cond
    cond = 1 / (1 / (edges.diams**4 / edges.lens) + 3 * np.pi / 16 / sid.chi0 * np.abs(inc.incidence) @ (1 / node_clogging ** 3))
    cond = np.array(np.ma.fix_invalid(cond, fill_value = 0))
    edges.cond = cond

    p_matrix = inc.incidence.T @ spr.diags(cond) \
        @ inc.incidence
    # for all inlet nodes we set the same pressure, for outlet nodes we set
    # zero pressure; so for boundary nodes we zero the elements of p_matrix
    # and add identity for those rows
    p_matrix_cb = p_matrix.multiply(1 - pressure_b_cb[:, np.newaxis] - graph.out_vec[:, np.newaxis]) + spr.diags(pressure_b_cb + graph.out_vec)
    p_matrix_cc = p_matrix.multiply(1 - pressure_b_cc[:, np.newaxis] - graph.out_vec[:, np.newaxis]) + spr.diags(pressure_b_cc + graph.out_vec)
    #p_matrix = p_matrix.multiply(inc.middle) + inc.boundary
    diag = p_matrix_cb.diagonal()
    # fix for nodes with no connections
    diag_old = diag.copy()
    diag += 1 * (diag == 0)
    # replace diagonal
    p_matrix_cb += spr.diags(diag - diag_old)
    diag = p_matrix_cc.diagonal()

    # fix for nodes with no connections
    diag_old = diag.copy()
    diag += 1 * (diag == 0)
    # replace diagonal
    p_matrix_cc += spr.diags(diag - diag_old)
    # replace diagonal

    # solve matrix @ pressure = pressure_b
    print("Solving b pressure")
    pressure_cb = solve_equation(p_matrix_cb, pressure_b_cb)
    print("Solving c pressure")
    pressure_cc = solve_equation(p_matrix_cc, pressure_b_cc)
    print("Pressure solved")
    flow_cb = np.abs(cond * (inc.incidence @ pressure_cb))
    flow_cc = np.abs(cond * (inc.incidence @ pressure_cc))
    data.cond_ratio_cb.append(np.sum(flow_cb * (np.abs(inc.incidence) @ graph.out_vec_b > 0)) / np.sum(flow_cb * (np.abs(inc.incidence) @ graph.in_vec_a > 0)))
    data.cond_ratio_cc.append(np.sum(flow_cc * (np.abs(inc.incidence) @ graph.out_vec_a > 0)) / np.sum(flow_cc * (np.abs(inc.incidence) @ graph.in_vec_b > 0)))
    # normalize pressure in inlet nodes to match condition for constant inlet
    # flow
    pressure = pressure_cb * (1 + sid.q_rate) + pressure_cc * (1 - sid.q_rate)


    
    q_in = np.abs(np.sum(cond * (inc.inlet \
        @ pressure)))
    pressure *= sid.Q_in / q_in
    # update flow
    edges.flow = cond * (inc.incidence @ pressure)
    p_continuity = p_matrix @ pressure * (1 - graph.in_vec - graph.out_vec)
    print(np.sum(np.abs(p_continuity)))
    Q_in = np.sum(edges.inlet * np.abs(edges.flow))
    Q_out = np.sum(edges.outlet * np.abs(edges.flow))
    q_in = np.abs(np.sum(cond * (inc.inlet \
        @ pressure)))
    print('Q_in =', Q_in, 'Q_out =', Q_out)
    print(np.sum(edges.inlet), np.sum(edges.outlet), q_in)

    return pressure
