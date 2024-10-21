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
    return sid.cb_in * graph.in_vec_a, sid.cc_in * graph.in_vec_b

def solve_dissolution_nr(sid: SimInputData, inc: Incidence, graph: Graph, \
    edges: Edges, cb_b: spr.csc_matrix, cc_b: spr.csc_matrix) -> np.ndarray:
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
    q_cb = edges.flow * np.exp(-np.abs(sid.Da / (1 + sid.G * edges.diams) \
        * edges.diams * edges.lens / edges.flow))
    q_cb = np.array(np.ma.fix_invalid(q_cb, fill_value = 0))
    q_cb_matrix = np.abs(inc.incidence.T @ spr.diags(q_cb) @ inc.incidence)
    cb_matrix = c_inc.multiply(q_cb_matrix)
    # find diagonal coefficients (inlet flow for each node)
    diag = -np.abs(inc.incidence.T) @ np.abs(edges.flow) / 2
    diag = diag * (1 - graph.in_vec + graph.out_vec) + graph.in_vec
    diag += 1 * (diag == 0)
    # replace diagonal
    cb_matrix.setdiag(diag)
    cb = solve_equation(cb_matrix, cb_b)

    # find vector with non-diagonal coefficients
    q_cc = edges.flow * np.exp(-np.abs(sid.Da / (1 + sid.G * edges.diams) \
        * edges.diams * edges.lens / edges.flow))
    q_cc = np.array(np.ma.fix_invalid(q_cc, fill_value = 0))
    q_cc_matrix = np.abs(inc.incidence.T @ spr.diags(q_cc) @ inc.incidence)
    cc_matrix = c_inc.multiply(q_cc_matrix)
    cc_matrix.setdiag(diag)
    cc = solve_equation(cc_matrix, cc_b)

    cb_prev = np.zeros(sid.nsq)
    cc_prev = np.zeros(sid.nsq)



    while np.linalg.norm(cb - cb_prev) > sid.c_th or np.linalg.norm(cc - cc_prev) > sid.c_th:
        cc_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ cc
        q_cb = edges.flow * np.exp(-np.abs(sid.Da / (1 + sid.G * edges.diams) \
            * cc_in / sid.c_eq * edges.diams * edges.lens / edges.flow))
        q_cb = np.array(np.ma.fix_invalid(q_cb, fill_value = 0))
        q_cb_matrix = np.abs(inc.incidence.T @ spr.diags(q_cb) @ inc.incidence)
        cb_matrix = c_inc.multiply(q_cb_matrix)
        cb_matrix.setdiag(diag)        
        f_cb = (1 - graph.in_vec) * (cb_matrix @ cb)
        cb_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ cb
        q_cc = edges.flow * np.exp(-np.abs(sid.Da / (1 + sid.G * edges.diams) \
            * cb_in / sid.c_eq * edges.diams * edges.lens / edges.flow))
        q_cc = np.array(np.ma.fix_invalid(q_cc, fill_value = 0))
        q_cc_matrix = np.abs(inc.incidence.T @ spr.diags(q_cc) @ inc.incidence)
        cc_matrix = c_inc.multiply(q_cc_matrix)
        cc_matrix.setdiag(diag)
        f_cc = (1 - graph.in_vec) * (cc_matrix @ cc)

        dq_cc = -cb_in * sid.Da / (1 + sid.G * edges.diams) / sid.c_eq \
            * edges.diams * edges.lens * np.exp(-np.abs(sid.Da / (1 + sid.G * edges.diams) \
            * cc_in / sid.c_eq * edges.diams * edges.lens / edges.flow))
        dq_cc = np.array(np.ma.fix_invalid(dq_cc, fill_value = 0))
        dq_cc_matrix = np.abs(inc.incidence.T @ spr.diags(dq_cc) @ inc.incidence)
        dcc_matrix = c_inc.multiply(dq_cc_matrix)
        dcc_matrix.setdiag(np.zeros(sid.nsq))

        dq_cb = -cc_in * sid.Da / (1 + sid.G * edges.diams) / sid.c_eq \
            * edges.diams * edges.lens * np.exp(-np.abs(sid.Da / (1 + sid.G * edges.diams) \
            * cb_in / sid.c_eq * edges.diams * edges.lens / edges.flow))
        dq_cb = np.array(np.ma.fix_invalid(dq_cb, fill_value = 0))
        dq_cb_matrix = np.abs(inc.incidence.T @ spr.diags(dq_cb) @ inc.incidence)
        dcb_matrix = c_inc.multiply(dq_cb_matrix)
        dcb_matrix.setdiag(np.zeros(sid.nsq))

        dc_matrix = spr.vstack([spr.hstack([cb_matrix, dcc_matrix]), spr.hstack([dcb_matrix, cc_matrix])])
        f = np.concatenate((f_cb, f_cc))
        in_vec = np.concatenate((graph.in_vec, graph.in_vec))
        dc_matrix = spr.diags(1 - in_vec) @ dc_matrix + spr.diags(in_vec)
        delta_c = solve_equation(dc_matrix, -f)
        cb_prev = cb.copy()
        cc_prev = cc.copy()
        cb += delta_c[:sid.nsq]
        cc += delta_c[sid.nsq:]
        print(np.linalg.norm(cb - cb_prev), np.linalg.norm(cc - cc_prev))
    return cb, cc

def solve_dissolution_nr2(sid: SimInputData, inc: Incidence, graph: Graph, \
    edges: Edges, cb_b: spr.csc_matrix, cc_b: spr.csc_matrix, cb, cc) -> np.ndarray:
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
    # find diagonal coefficients (inlet flow for each node)
    diag = -np.abs(inc.incidence.T) @ np.abs(edges.flow) / 2
    diag = diag * (1 - graph.in_vec + graph.out_vec) + graph.in_vec
    diag += 1 * (diag == 0)
    # replace diagonal


    cb_prev = np.zeros(sid.nsq)
    cc_prev = np.zeros(sid.nsq)



    while np.linalg.norm(cb - cb_prev) > sid.c_th or np.linalg.norm(cc - cc_prev) > sid.c_th:
        cc_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ cc
        q_cb = edges.flow * np.exp(-np.abs(sid.Da / (1 + sid.G * edges.diams) \
            * cc_in / sid.c_eq * edges.diams * edges.lens / edges.flow))
        q_cb = np.array(np.ma.fix_invalid(q_cb, fill_value = 0))
        q_cb_matrix = np.abs(inc.incidence.T @ spr.diags(q_cb) @ inc.incidence)
        cb_matrix = c_inc.multiply(q_cb_matrix)
        cb_matrix.setdiag(diag)        
        f_cb = (1 - graph.in_vec) * (cb_matrix @ cb)
        cb_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ cb
        q_cc = edges.flow * np.exp(-np.abs(sid.Da / (1 + sid.G * edges.diams) \
            * cb_in / sid.c_eq * edges.diams * edges.lens / edges.flow))
        q_cc = np.array(np.ma.fix_invalid(q_cc, fill_value = 0))
        q_cc_matrix = np.abs(inc.incidence.T @ spr.diags(q_cc) @ inc.incidence)
        cc_matrix = c_inc.multiply(q_cc_matrix)
        cc_matrix.setdiag(diag)
        f_cc = (1 - graph.in_vec) * (cc_matrix @ cc)

        dq_cc = -cb_in * sid.Da / (1 + sid.G * edges.diams) / sid.c_eq \
            * edges.diams * edges.lens * np.exp(-np.abs(sid.Da / (1 + sid.G * edges.diams) \
            * cc_in / sid.c_eq * edges.diams * edges.lens / edges.flow))
        dq_cc = np.array(np.ma.fix_invalid(dq_cc, fill_value = 0))
        dq_cc_matrix = np.abs(inc.incidence.T @ spr.diags(dq_cc) @ inc.incidence)
        dcc_matrix = c_inc.multiply(dq_cc_matrix)
        dcc_matrix.setdiag(np.zeros(sid.nsq))

        dq_cb = -cc_in * sid.Da / (1 + sid.G * edges.diams) / sid.c_eq \
            * edges.diams * edges.lens * np.exp(-np.abs(sid.Da / (1 + sid.G * edges.diams) \
            * cb_in / sid.c_eq * edges.diams * edges.lens / edges.flow))
        dq_cb = np.array(np.ma.fix_invalid(dq_cb, fill_value = 0))
        dq_cb_matrix = np.abs(inc.incidence.T @ spr.diags(dq_cb) @ inc.incidence)
        dcb_matrix = c_inc.multiply(dq_cb_matrix)
        dcb_matrix.setdiag(np.zeros(sid.nsq))

        dc_matrix = spr.vstack([spr.hstack([cb_matrix, dcc_matrix]), spr.hstack([dcb_matrix, cc_matrix])])
        f = np.concatenate((f_cb, f_cc))
        in_vec = np.concatenate((graph.in_vec, graph.in_vec))
        dc_matrix = spr.diags(1 - in_vec) @ dc_matrix + spr.diags(in_vec)
        delta_c = solve_equation(dc_matrix, -f)
        cb_prev = cb.copy()
        cc_prev = cc.copy()
        cb += delta_c[:sid.nsq]
        cc += delta_c[sid.nsq:]
        print(np.linalg.norm(cb - cb_prev), np.linalg.norm(cc - cc_prev))
    return cb, cc

def solve_dissolution(sid: SimInputData, inc: Incidence, graph: Graph, \
    edges: Edges, cb_b: spr.csc_matrix, cc_b: spr.csc_matrix, cb, cc) -> np.ndarray:
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
    cb_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ cb
    cc_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ cc
    c_inc = 1 * (inc.incidence.T @ (spr.diags(edges.flow) \
        @ inc.incidence > 0) != 0)
    # cb_inc = np.abs(inc.incidence.T @ (spr.diags(edges.flow) \
    #    @ inc.incidence > 0))
    # find vector with non-diagonal coefficients
    qc = edges.flow * np.exp(-np.abs(cc_in / sid.c_eq * sid.Da / (1 + sid.G * edges.diams) \
        * edges.diams * edges.lens / edges.flow))
    qc = np.array(np.ma.fix_invalid(qc, fill_value = 0))
    qc_matrix = np.abs(inc.incidence.T @ spr.diags(qc) @ inc.incidence)
    cb_matrix = c_inc.multiply(qc_matrix)
    qc = edges.flow * np.exp(-np.abs(cb_in / sid.c_eq * sid.Da / (1 + sid.G * edges.diams) \
    * edges.diams * edges.lens / edges.flow))
    qc = np.array(np.ma.fix_invalid(qc, fill_value = 0))
    qc_matrix = np.abs(inc.incidence.T @ spr.diags(qc) @ inc.incidence)
    cc_matrix = c_inc.multiply(qc_matrix)
    # find diagonal coefficients (inlet flow for each node)
    diag = -np.abs(inc.incidence.T) @ np.abs(edges.flow) / 2
    diag = diag * (1 - graph.in_vec + graph.out_vec) + graph.in_vec
    diag += 1 * (diag == 0)
    # replace diagonal
    cb_matrix.setdiag(diag)
    cc_matrix.setdiag(diag)
    cb = solve_equation(cb_matrix, cb_b)
    cc = solve_equation(cc_matrix, cc_b)
    return cb, cc

def solve_dissolution_an(sid: SimInputData, inc: Incidence, graph: Graph, \
    edges: Edges, cb_b: spr.csc_matrix, cc_b: spr.csc_matrix) -> np.ndarray:
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
    #np.savetxt('cinc.txt', c_inc.toarray())
    c_down = 1 * (inc.incidence.T @ (spr.diags(edges.flow) \
        @ inc.incidence < 0) != 0)
    # find vector with non-diagonal coefficients
    q_cb = edges.flow # * np.exp(-np.abs(sid.Da / (1 + sid.G * edges.diams) \
    # * edges.diams * edges.lens / edges.flow))
    q_cb = np.array(np.ma.fix_invalid(q_cb, fill_value = 0))
    q_cb_matrix = np.abs(inc.incidence.T @ spr.diags(q_cb) @ inc.incidence)
    cb_matrix = c_inc.multiply(q_cb_matrix)
    cb_matrix = cb_matrix.multiply(1 - graph.in_vec[:, np.newaxis])
    # find diagonal coefficients (inlet flow for each node)
    diag = -np.abs(inc.incidence.T) @ np.abs(edges.flow) / 2
    diag = diag * (1 - graph.in_vec + graph.out_vec) + graph.in_vec
    diag += 1 * (diag == 0)
    # replace diagonal
    cb_matrix.setdiag(diag)
    cb = solve_equation(cb_matrix, cb_b)

    diag_out = -np.abs(inc.incidence.T) @ np.abs(edges.flow) / 2
    diag_out = diag_out * (1 - graph.in_vec + graph.out_vec) + graph.in_vec
    # find vector with non-diagonal coefficients
    q_cc = edges.flow # * np.exp(-np.abs(sid.Da / (1 + sid.G * edges.diams) \
    #     * edges.diams * edges.lens / edges.flow))
    q_cc = np.array(np.ma.fix_invalid(q_cc, fill_value = 0))
    q_cc_matrix = np.abs(inc.incidence.T @ spr.diags(q_cc) @ inc.incidence)
    cc_matrix = c_inc.multiply(q_cc_matrix)
    cc_matrix = cc_matrix.multiply(1 - graph.in_vec[:, np.newaxis])
    cc_matrix.setdiag(diag)
    cc = solve_equation(cc_matrix, cc_b)

    cb_prev = np.zeros(sid.nsq)
    cc_prev = np.zeros(sid.nsq)
    #cb = cb_b
    #cc = cc_b
    #upstream = 1 * (spr.diags(edges.flow) @ inc.incidence > 0).T
    upstream = 1 * (inc.incidence.T @ spr.diags(edges.flow) < 0)
    
    #downstream = 1 * spr.diags(edges.flow) @ inc.incidence < 0
    #np.savetxt('upstream.txt', upstream.toarray())
    while np.linalg.norm(cb - cb_prev) > sid.c_th or np.linalg.norm(cc - cc_prev) > sid.c_th:
        cb_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ cb
        cc_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ cc
        alpha = np.abs(sid.Da / (1 + sid.G * edges.diams) \
            / sid.c_eq * edges.diams * edges.lens * (edges.diams > sid.dmin) / edges.flow)
        exp_a = np.exp(-alpha * (cb_in - cc_in))
        alpha = np.array(np.ma.fix_invalid(alpha, fill_value = 0))
        exp_a = np.array(np.ma.fix_invalid(exp_a, fill_value = 0)) * (cb_in >= cc_in) + np.array(np.ma.fix_invalid(exp_a, fill_value = 1e10)) * (cc_in > cb_in)
        cb_out = cb_in * (cb_in - cc_in) / (cb_in - cc_in * exp_a)
        cc_out = cb_in * (cb_in - cc_in) / (cb_in - cc_in * exp_a) + cc_in - cb_in
        cb_out_fix = (cb_in == cc_in) * cb_in / (1 + alpha * cb_in)
        cb_out = np.array(np.ma.fix_invalid(cb_out, fill_value = 0))
        cb_out = cb_out + cb_out_fix
        cc_out = np.array(np.ma.fix_invalid(cc_out, fill_value = 0))
        cc_out = cc_out + cb_out_fix
        f_cb = (1 - graph.in_vec) * (upstream @ (np.abs(edges.flow) * cb_out) + diag_out * cb)
        f_cc = (1 - graph.in_vec) * (upstream @ (np.abs(edges.flow) * cc_out) + diag_out * cc)


        dcb_cb = (cb_in ** 2 + cc_in * exp_a * (cb_in * (-2 - alpha * (cb_in - cc_in)) + cc_in)) / (cb_in - cc_in * exp_a) ** 2
        dcb_cb = np.array(np.ma.fix_invalid(dcb_cb, fill_value = 0))
        dcb_cb_fix = (cb_in == cc_in) / (1 + alpha * cb_in) ** 2
        dcb_cb = dcb_cb + dcb_cb_fix
        dcc_cb = (dcb_cb - 1) * (cb_in != cc_in)
        dcb_cc = (cb_in * (-cb_in + exp_a * (cb_in + alpha * cc_in * (cb_in - cc_in)))) / (cb_in - cc_in * exp_a) ** 2
        dcb_cc = np.array(np.ma.fix_invalid(dcb_cc, fill_value = 0))
        dcb_cc = dcb_cc
        dcc_cc = (dcb_cc + 1) * (cb_in != cc_in) + (cb_in == cc_in) / (1 + alpha * cc_in) ** 2


        dcb_cb_matrix = upstream @ spr.diags(np.abs(edges.flow) * dcb_cb) @ np.abs(inc.incidence)
        dcb_cb_matrix.setdiag(diag)
        dcb_cc_matrix = upstream @ spr.diags(np.abs(edges.flow) * dcb_cc) @ np.abs(inc.incidence)
        dcb_cc_matrix.setdiag(0)
        dcc_cb_matrix = upstream @ spr.diags(np.abs(edges.flow) * dcc_cb) @ np.abs(inc.incidence)
        dcb_cc_matrix.setdiag(0)
        dcc_cc_matrix = upstream @ spr.diags(np.abs(edges.flow) * dcc_cc) @ np.abs(inc.incidence)
        dcc_cc_matrix.setdiag(diag)

        dc_matrix = spr.vstack([spr.hstack([dcb_cb_matrix, dcb_cc_matrix]), spr.hstack([dcc_cb_matrix, dcc_cc_matrix])])
        f = np.concatenate((f_cb, f_cc))
        in_vec = np.concatenate((graph.in_vec, graph.in_vec))
        dc_matrix = dc_matrix.multiply(1 - in_vec[:, np.newaxis]) + spr.diags(in_vec)
        delta_c = solve_equation(dc_matrix, -f)
        cb_prev = cb.copy()
        cc_prev = cc.copy()
        cb += delta_c[:sid.nsq]
        cc += delta_c[sid.nsq:]
        cb *= (cb > 0)
        cc *= (cc > 0)
        print(np.linalg.norm(cb - cb_prev), np.linalg.norm(cc - cc_prev))
    
    return cb, cc

def solve_dissolution_an2(sid: SimInputData, inc: Incidence, graph: Graph, \
    edges: Edges, cb_b: spr.csc_matrix, cc_b: spr.csc_matrix, cb, cc) -> np.ndarray:
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
    c_down = 1 * (inc.incidence.T @ (spr.diags(edges.flow) \
        @ inc.incidence < 0) != 0)

    # find diagonal coefficients (inlet flow for each node)
    diag = -np.abs(inc.incidence.T) @ np.abs(edges.flow) / 2
    diag = diag * (1 - graph.in_vec + graph.out_vec) + graph.in_vec
    diag += 1 * (diag == 0)

    cb_prev = np.zeros(sid.nsq)
    cc_prev = np.zeros(sid.nsq)
    #cb = cb_b
    #cc = cc_b
    upstream = 1 * (inc.incidence.T @ spr.diags(edges.flow) < 0)
    downstream = spr.diags(edges.flow) @ inc.incidence < 0

    diag_out = -np.abs(inc.incidence.T) @ np.abs(edges.flow) / 2
    diag_out = diag_out * (1 - graph.in_vec + graph.out_vec) + graph.in_vec
    while np.linalg.norm(cb - cb_prev) > sid.c_th or np.linalg.norm(cc - cc_prev) > sid.c_th:
        cb_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ cb
        cc_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ cc
        alpha = np.abs(sid.Da / (1 + sid.G * edges.diams) \
            / sid.c_eq * edges.diams * edges.lens * (edges.diams > sid.dmin) / edges.flow)
        exp_a = np.exp(-alpha * (cb_in - cc_in))
        alpha = np.array(np.ma.fix_invalid(alpha, fill_value = 0))
        exp_a = np.array(np.ma.fix_invalid(exp_a, fill_value = 0))
        cb_out = cb_in * (cb_in - cc_in) / (cb_in - cc_in * exp_a)
        cc_out = cb_in * (cb_in - cc_in) / (cb_in - cc_in * exp_a) + cc_in - cb_in
        cb_out = np.array(np.ma.fix_invalid(cb_out, fill_value = -1))
        cb_out = cb_out * (cb_out != -1) + cb_in * (cb_out == -1)
        cc_out = np.array(np.ma.fix_invalid(cc_out, fill_value = -1))
        cc_out = cc_out * (cc_out != -1) + cc_in * (cc_out == -1)
        f_cb = (1 - graph.in_vec) * (upstream @ (np.abs(edges.flow) * cb_out) + diag_out * cb)
        f_cc = (1 - graph.in_vec) * (upstream @ (np.abs(edges.flow) * cc_out) + diag_out * cc)


        dcb_cb = (cb_in ** 2 + cc_in * exp_a * (cb_in * (-2 - alpha * (cb_in - cc_in)) + cc_in)) / (cb_in - cc_in * exp_a) ** 2
        dcb_cb = np.array(np.ma.fix_invalid(dcb_cb, fill_value = 1000))
        dcb_cb = dcb_cb * (dcb_cb != 1000) + 1 / (1 + alpha * cb_in) ** 2 * (dcb_cb == 1000)
        dcc_cb = dcb_cb - 1
        dcb_cc = (cb_in * (-cb_in + exp_a * (cb_in + alpha * cc_in * (cb_in - cc_in)))) / (cb_in - cc_in * exp_a) ** 2
        dcb_cc = np.array(np.ma.fix_invalid(dcb_cc, fill_value = 1000))
        dcb_cc = dcb_cc * (dcb_cc != 1000) + 1 / (1 + alpha * cc_in) ** 2 * (dcb_cc == 1000)
        dcc_cc = dcb_cc + 1

        dcb_cb_matrix = upstream @ spr.diags(np.abs(edges.flow) * dcb_cb) @ np.abs(inc.incidence)
        dcb_cb_matrix.setdiag(diag)
        dcb_cc_matrix = upstream @ spr.diags(np.abs(edges.flow) * dcb_cc) @ np.abs(inc.incidence)
        dcb_cc_matrix.setdiag(0)
        dcc_cb_matrix = upstream @ spr.diags(np.abs(edges.flow) * dcc_cb) @ np.abs(inc.incidence)
        dcb_cc_matrix.setdiag(0)
        dcc_cc_matrix = upstream @ spr.diags(np.abs(edges.flow) * dcc_cc) @ np.abs(inc.incidence)
        dcc_cc_matrix.setdiag(diag)

        dc_matrix = spr.vstack([spr.hstack([dcb_cb_matrix, dcb_cc_matrix]), spr.hstack([dcc_cb_matrix, dcc_cc_matrix])])
        f = np.concatenate((f_cb, f_cc))
        in_vec = np.concatenate((graph.in_vec, graph.in_vec))
        #np.savetxt('dc_matrix.txt', dc_matrix.toarray())
        dc_matrix = dc_matrix.multiply(1 - in_vec[:, np.newaxis]) + spr.diags(in_vec)
        # np.savetxt('dc_matrix2.txt', dc_matrix.toarray())
        # np.savetxt('upstream.txt', (upstream @ upstream.T).toarray())
        
        diag_c = dc_matrix.diagonal()
        #print(np.sum(diag_c))
        f = f * (diag_c != 0)
        diag_c = diag_c * (diag_c != 0) + 1 * (diag_c == 0)
        #print(np.sum(f))
        dc_matrix = dc_matrix.multiply(1 - (diag_c == 0)[:, np.newaxis])
        dc_matrix.setdiag(diag_c)
        delta_c = solve_equation(dc_matrix, -f)
        cb_prev = cb.copy()
        cc_prev = cc.copy()
        cb += delta_c[:sid.nsq]
        cc += delta_c[sid.nsq:]
        cb *= (cb > 0)
        cc *= (cc > 0)
        print(np.linalg.norm(cb - cb_prev), np.linalg.norm(cc - cc_prev))
    if np.sum(cb < -1e-3) or np.sum(cc < -1e-3):
        nminus = np.where(cb < -1e-3)[0][0]
        print('node ', nminus, cb[nminus], cc[nminus])
        print(diag_c[nminus], f[nminus])
        #np.savetxt('dc_matrix.txt', dc_matrix.toarray())
    #np.savetxt('cb_in.txt', (spr.diags(edges.flow) @ inc.incidence > 0).toarray())
    return cb, cc

def solve_dissolution_v2(sid: SimInputData, inc: Incidence, graph: Graph, \
    edges: Edges, cb_b: spr.csc_matrix, cc_b: spr.csc_matrix, cb, cc) -> np.ndarray:
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
    c_down = 1 * (inc.incidence.T @ (spr.diags(edges.flow) \
        @ inc.incidence < 0) != 0)

    # find diagonal coefficients (inlet flow for each node)
    diag = -np.abs(inc.incidence.T) @ np.abs(edges.flow) / 2
    diag = diag * (1 - graph.in_vec + graph.out_vec) + graph.in_vec
    diag += 1 * (diag == 0)

    cb_prev = np.zeros(sid.nsq)
    cc_prev = np.zeros(sid.nsq)
    #cb = cb_b
    #cc = cc_b
    upstream = 1 * (inc.incidence.T @ spr.diags(edges.flow) < 0)
    downstream = spr.diags(edges.flow) @ inc.incidence < 0

    diag_out = -np.abs(inc.incidence.T) @ np.abs(edges.flow) / 2
    diag_out = diag_out * (1 - graph.in_vec + graph.out_vec) + graph.in_vec
    while np.linalg.norm(cb - cb_prev) > sid.c_th or np.linalg.norm(cc - cc_prev) > sid.c_th:
        cb_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ cb
        cc_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ cc
        alpha = np.abs(sid.Da / (1 + sid.G * edges.diams) \
            / sid.c_eq * edges.diams * edges.lens * (edges.diams > sid.dmin) / edges.flow)
        exp_a = np.exp(-alpha * (cb_in - cc_in))
        alpha = np.array(np.ma.fix_invalid(alpha, fill_value = 0))
        exp_a = np.array(np.ma.fix_invalid(exp_a, fill_value = 0)) * (cb_in >= cc_in) + np.array(np.ma.fix_invalid(exp_a, fill_value = 1e10)) * (cc_in > cb_in)
        cb_out = cb_in * (cb_in - cc_in) / (cb_in - cc_in * exp_a)
        cc_out = cb_in * (cb_in - cc_in) / (cb_in - cc_in * exp_a) + cc_in - cb_in
        cb_out_fix = (cb_in == cc_in) * cb_in / (1 + alpha * cb_in)
        cb_out = np.array(np.ma.fix_invalid(cb_out, fill_value = 0))
        cb_out = cb_out + cb_out_fix
        cc_out = np.array(np.ma.fix_invalid(cc_out, fill_value = 0))
        cc_out = cc_out + cb_out_fix
        f_cb = (1 - graph.in_vec) * (upstream @ (np.abs(edges.flow) * cb_out) + diag_out * cb)
        f_cc = (1 - graph.in_vec) * (upstream @ (np.abs(edges.flow) * cc_out) + diag_out * cc)


        dcb_cb = (cb_in ** 2 + cc_in * exp_a * (cb_in * (-2 - alpha * (cb_in - cc_in)) + cc_in)) / (cb_in - cc_in * exp_a) ** 2
        dcb_cb = np.array(np.ma.fix_invalid(dcb_cb, fill_value = 0))
        dcb_cb_fix = (cb_in == cc_in) / (1 + alpha * cb_in) ** 2
        dcb_cb = dcb_cb + dcb_cb_fix
        dcc_cb = (dcb_cb - 1) * (cb_in != cc_in)
        dcb_cc = (cb_in * (-cb_in + exp_a * (cb_in + alpha * cc_in * (cb_in - cc_in)))) / (cb_in - cc_in * exp_a) ** 2
        dcb_cc = np.array(np.ma.fix_invalid(dcb_cc, fill_value = 0))
        dcb_cc = dcb_cc
        dcc_cc = (dcb_cc + 1) * (cb_in != cc_in) + (cb_in == cc_in) / (1 + alpha * cc_in) ** 2

        dcb_cb_matrix = upstream @ spr.diags(np.abs(edges.flow) * dcb_cb) @ np.abs(inc.incidence)
        dcb_cb_matrix.setdiag(diag)
        dcb_cc_matrix = upstream @ spr.diags(np.abs(edges.flow) * dcb_cc) @ np.abs(inc.incidence)
        dcb_cc_matrix.setdiag(0)
        dcc_cb_matrix = upstream @ spr.diags(np.abs(edges.flow) * dcc_cb) @ np.abs(inc.incidence)
        dcb_cc_matrix.setdiag(0)
        dcc_cc_matrix = upstream @ spr.diags(np.abs(edges.flow) * dcc_cc) @ np.abs(inc.incidence)
        dcc_cc_matrix.setdiag(diag)

        dc_matrix = spr.vstack([spr.hstack([dcb_cb_matrix, dcb_cc_matrix]), spr.hstack([dcc_cb_matrix, dcc_cc_matrix])])
        f = np.concatenate((f_cb, f_cc))
        in_vec = np.concatenate((graph.in_vec, graph.in_vec))
        #np.savetxt('dc_matrix.txt', dc_matrix.toarray())
        dc_matrix = dc_matrix.multiply(1 - in_vec[:, np.newaxis]) + spr.diags(in_vec)
        # np.savetxt('dc_matrix2.txt', dc_matrix.toarray())
        # np.savetxt('upstream.txt', (upstream @ upstream.T).toarray())
        
        diag_c = dc_matrix.diagonal()
        #print(np.sum(diag_c))
        f = f * (diag_c != 0)
        diag_c = diag_c * (diag_c != 0) + 1 * (diag_c == 0)
        #print(np.sum(f))
        dc_matrix = dc_matrix.multiply(1 - (diag_c == 0)[:, np.newaxis])
        dc_matrix.setdiag(diag_c)
        delta_c = solve_equation(dc_matrix, -f)
        cb_prev = cb.copy()
        cc_prev = cc.copy()
        cb += delta_c[:sid.nsq]
        cc += delta_c[sid.nsq:]
        cb *= (cb > 0)
        cc *= (cc > 0)
        print(np.linalg.norm(cb - cb_prev), np.linalg.norm(cc - cc_prev))
    if np.sum(cb < -1e-3) or np.sum(cc < -1e-3):
        nminus = np.where(cb < -1e-3)[0][0]
        print('node ', nminus, cb[nminus], cc[nminus])
        print(diag_c[nminus], f[nminus])
        #np.savetxt('dc_matrix.txt', dc_matrix.toarray())
    #np.savetxt('cb_in.txt', (spr.diags(edges.flow) @ inc.incidence > 0).toarray())
    return cb, cc
