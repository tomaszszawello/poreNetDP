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
    # return sid.cb_in * np.concatenate([np.ones(sid.n // 2), np.zeros(sid.nsq - sid.n // 2)]), \
    #     sid.cc_in * np.concatenate([np.zeros(sid.n // 2), np.ones(sid. n - sid.n // 2), np.zeros(sid. nsq - sid.n)])
    return sid.cb_in * graph.in_vec

def create_vector_prec(sid: SimInputData, graph: Graph, inc, edges, cb) -> spr.csc_matrix:
    # find incidence for cb (only upstream flow matters)
    cb_inc = np.abs(inc.incidence.T @ (spr.diags(edges.flow) \
        @ inc.incidence > 0))
    # find vector with non-diagonal coefficients
    qc = edges.diams * edges.lens * sid.Da / (1 + sid.G * edges.diams)
    qc_matrix = np.abs(inc.incidence.T @ spr.diags(qc) @ inc.incidence)
    cb_matrix = cb_inc.multiply(qc_matrix)
    cb_matrix.setdiag(np.zeros(sid.nsq)) # set diagonal to zero
    cc_b = -cb_matrix @ cb
    cc_b = cc_b * (1 - graph.in_vec) + graph.in_vec * sid.cc_in
    # for node in graph.in_nodes:
    #     cc_b[node] = sid.cc_in # set result for input nodes to cc_in
    return cc_b, sid.cd_in * graph.in_vec


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
    c_inc = 1 * (inc.incidence.T @ (spr.diags(edges.flow) \
        @ inc.incidence > 0) != 0)
    # cb_inc = np.abs(inc.incidence.T @ (spr.diags(edges.flow) \
    #    @ inc.incidence > 0))
    # find vector with non-diagonal coefficients
    qc = edges.flow * np.exp(-np.abs(sid.Da / (1 + sid.G * edges.diams) \
        * edges.diams * edges.lens / edges.flow))
    qc = np.array(np.ma.fix_invalid(qc, fill_value = 0))
    qc_matrix = np.abs(inc.incidence.T @ spr.diags(qc) @ inc.incidence)
    cb_matrix = c_inc.multiply(qc_matrix)
    # find diagonal coefficients (inlet flow for each node)
    diag = -np.abs(inc.incidence.T) @ np.abs(edges.flow) / 2
    diag = diag * (1 - graph.in_vec + graph.out_vec) + graph.in_vec
    diag += 1 * (diag == 0)
    # replace diagonal
    cb_matrix.setdiag(diag)
    cb = solve_equation(cb_matrix, cb_b)
    return cb



def solve_precipitation_nr(sid: SimInputData, inc: Incidence, graph: Graph, \
    edges: Edges, cc_b: spr.csc_matrix, cd_b: spr.csc_matrix) -> np.ndarray:
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
    c_inc = 1 * (inc.incidence.T @ (spr.diags(edges.flow) \
        @ inc.incidence > 0) != 0)
    # find vector with non-diagonal coefficients
    q_cc = edges.flow * np.exp(-np.abs(sid.Da * sid.K / (1 + sid.G * sid.K * edges.diams) \
        * edges.diams * edges.lens / edges.flow))
    q_cc = np.array(np.ma.fix_invalid(q_cc, fill_value = 0))
    q_cc_matrix = np.abs(inc.incidence.T @ spr.diags(q_cc) @ inc.incidence)
    cc_matrix = c_inc.multiply(q_cc_matrix)
    # find diagonal coefficients (inlet flow for each node)
    diag = -np.abs(inc.incidence.T) @ np.abs(edges.flow) / 2
    diag = diag * (1 - graph.in_vec + graph.out_vec) + graph.in_vec
    diag += 1 * (diag == 0)
    # replace diagonal
    cc_matrix.setdiag(diag)
    cc = solve_equation(cc_matrix, cc_b)

    # find vector with non-diagonal coefficients
    q_cd = edges.flow * np.exp(-np.abs(sid.Da * sid.K / (1 + sid.G * sid.K * edges.diams) \
        * edges.diams * edges.lens / edges.flow))
    q_cd = np.array(np.ma.fix_invalid(q_cd, fill_value = 0))
    q_cd_matrix = np.abs(inc.incidence.T @ spr.diags(q_cd) @ inc.incidence)
    cd_matrix = c_inc.multiply(q_cd_matrix)
    cd_matrix.setdiag(diag)
    cd = solve_equation(cd_matrix, cd_b)

    cc_prev = np.zeros(sid.nsq)
    cd_prev = np.zeros(sid.nsq)



    while np.linalg.norm(cc - cc_prev) > sid.c_th or np.linalg.norm(cd - cd_prev) > sid.c_th:
        cc_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ cc
        q_cc = edges.flow * np.exp(-np.abs(sid.Da * sid.K / (1 + sid.G * sid.K * edges.diams) \
            * cc_in / sid.Kp * edges.diams * edges.lens / edges.flow))
        q_cc = np.array(np.ma.fix_invalid(q_cc, fill_value = 0))
        q_cc_matrix = np.abs(inc.incidence.T @ spr.diags(q_cc) @ inc.incidence)
        cc_matrix = c_inc.multiply(q_cc_matrix)
        cc_matrix.setdiag(diag)        
        f_cc = (1 - graph.in_vec) * (cc_matrix @ cc - cc_b)
        cd_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ cd
        q_cd = edges.flow * np.exp(-np.abs(sid.Da * sid.K / (1 + sid.G * sid.K * edges.diams) \
            * cc_in / sid.Kp * edges.diams * edges.lens / edges.flow))
        q_cd = np.array(np.ma.fix_invalid(q_cd, fill_value = 0))
        q_cd_matrix = np.abs(inc.incidence.T @ spr.diags(q_cd) @ inc.incidence)
        cd_matrix = c_inc.multiply(q_cd_matrix)
        cd_matrix.setdiag(diag)
        f_cd = (1 - graph.in_vec) * (cd_matrix @ cd)

        dq_cd = -cc_in * sid.Da * sid.K / (1 + sid.G * sid.K * edges.diams) / sid.Kp \
            * edges.diams * edges.lens * np.exp(-np.abs(sid.Da * sid.K / (1 + sid.G * sid.K * edges.diams) \
            * cd_in / sid.Kp * edges.diams * edges.lens / edges.flow)) # cd?
        dq_cd = np.array(np.ma.fix_invalid(dq_cd, fill_value = 0))
        dq_cd_matrix = np.abs(inc.incidence.T @ spr.diags(dq_cd) @ inc.incidence)
        dcd_matrix = c_inc.multiply(dq_cd_matrix)
        dcd_matrix.setdiag(np.zeros(sid.nsq))

        dq_cc = -cd_in * sid.Da * sid.K / (1 + sid.G * sid.K * edges.diams) / sid.Kp \
            * edges.diams * edges.lens * np.exp(-np.abs(sid.Da * sid.K / (1 + sid.G * sid.K * edges.diams) \
            * cc_in / sid.Kp * edges.diams * edges.lens / edges.flow))
        dq_cc = np.array(np.ma.fix_invalid(dq_cc, fill_value = 0))
        dq_cc_matrix = np.abs(inc.incidence.T @ spr.diags(dq_cc) @ inc.incidence)
        dcc_matrix = c_inc.multiply(dq_cc_matrix)
        dcc_matrix.setdiag(np.zeros(sid.nsq))

        dc_matrix = spr.vstack([spr.hstack([cc_matrix, dcd_matrix]), spr.hstack([dcc_matrix, cd_matrix])])
        f = np.concatenate((f_cc, f_cd))
        in_vec = np.concatenate((graph.in_vec, graph.in_vec))
        dc_matrix = spr.diags(1 - in_vec) @ dc_matrix + spr.diags(in_vec)
        delta_c = solve_equation(dc_matrix, -f)
        cc_prev = cc.copy()
        cd_prev = cd.copy()
        cc += delta_c[:sid.nsq]
        cd += delta_c[sid.nsq:]
        print(np.linalg.norm(cc - cc_prev), np.linalg.norm(cd - cd_prev))
    return cc, cd


def solve_precipitation(sid: SimInputData, inc: Incidence, graph: Graph, \
    edges: Edges, cc_b: spr.csc_matrix, cd_b: spr.csc_matrix, cc, cd) -> np.ndarray:
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
    cc_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ cc
    cd_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ cd
    c_inc = 1 * (inc.incidence.T @ (spr.diags(edges.flow) \
        @ inc.incidence > 0) != 0)
    # cb_inc = np.abs(inc.incidence.T @ (spr.diags(edges.flow) \
    #    @ inc.incidence > 0))
    # find vector with non-diagonal coefficients
    qc = edges.flow * np.exp(-np.abs(cd_in / sid.Kp * sid.Da * sid.K / (1 + sid.G * sid.K * edges.diams) \
        * edges.diams * edges.lens / edges.flow))
    qc = np.array(np.ma.fix_invalid(qc, fill_value = 0))
    qc_matrix = np.abs(inc.incidence.T @ spr.diags(qc) @ inc.incidence)
    cc_matrix = c_inc.multiply(qc_matrix)
    qc = edges.flow * np.exp(-np.abs(cc_in / sid.Kp * sid.Da * sid.K / (1 + sid.G * sid.K * edges.diams) \
    * edges.diams * edges.lens / edges.flow))
    qc = np.array(np.ma.fix_invalid(qc, fill_value = 0))
    qc_matrix = np.abs(inc.incidence.T @ spr.diags(qc) @ inc.incidence)
    cd_matrix = c_inc.multiply(qc_matrix)
    # find diagonal coefficients (inlet flow for each node)
    diag = -np.abs(inc.incidence.T) @ np.abs(edges.flow) / 2
    diag = diag * (1 - graph.in_vec + graph.out_vec) + graph.in_vec
    diag += 1 * (diag == 0)
    # replace diagonal
    cc_matrix.setdiag(diag)
    cd_matrix.setdiag(diag)
    cc = solve_equation(cc_matrix, cc_b)
    cd = solve_equation(cd_matrix, cd_b)
    return cc, cd
