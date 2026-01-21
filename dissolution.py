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
    
    #downstream = 1 * (spr.diags(edges.flow) @ inc.incidence < 0)
    #np.savetxt('upstream.txt', upstream.toarray())
    while np.linalg.norm(cb - cb_prev) > sid.c_th or np.linalg.norm(cc - cc_prev) > sid.c_th:
        cb_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ cb
        cc_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ cc
        # alpha = np.abs(sid.Da / (1 + sid.G * edges.diams) \
        #     / sid.c_eq * edges.diams * edges.lens / edges.flow)
        alpha = np.abs(sid.Da / (1 + sid.G * edges.diams) \
           / sid.c_eq * (edges.diams > 0) * edges.lens / edges.flow)
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
            / sid.c_eq * edges.diams * edges.lens / edges.flow)
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
        diag_old = dcb_cb_matrix.diagonal()
        #dcb_cb_matrix.setdiag(diag)
        dcb_cb_matrix += spr.diags(diag - diag_old)
        dcb_cc_matrix = upstream @ spr.diags(np.abs(edges.flow) * dcb_cc) @ np.abs(inc.incidence)
        #dcb_cc_matrix.setdiag(0)
        diag_old = dcb_cc_matrix.diagonal()
        dcb_cc_matrix -= spr.diags(diag_old)
        dcc_cb_matrix = upstream @ spr.diags(np.abs(edges.flow) * dcc_cb) @ np.abs(inc.incidence)
        #dcb_cc_matrix.setdiag(0)
        diag_old = dcc_cb_matrix.diagonal()
        dcc_cb_matrix -= spr.diags(diag_old)
        dcc_cc_matrix = upstream @ spr.diags(np.abs(edges.flow) * dcc_cc) @ np.abs(inc.incidence)
        #dcc_cc_matrix.setdiag(diag)
        diag_old = dcc_cc_matrix.diagonal()
        dcc_cc_matrix += spr.diags(diag - diag_old)

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
        diag_old = dc_matrix.diagonal()
        dc_matrix += spr.diags(diag_c - diag_old)
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
        #alpha = np.abs(sid.Da / (1 + sid.G * edges.diams) \
        #    / sid.c_eq * edges.diams * edges.lens / edges.flow)
        alpha = np.abs(sid.Da / (1 + sid.G * edges.diams) \
           / sid.c_eq * (edges.diams > 0) * edges.lens / edges.flow)
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
        diag_old = dcb_cb_matrix.diagonal()
        #dcb_cb_matrix.setdiag(diag)
        dcb_cb_matrix += spr.diags(diag - diag_old)
        dcb_cc_matrix = upstream @ spr.diags(np.abs(edges.flow) * dcb_cc) @ np.abs(inc.incidence)
        #dcb_cc_matrix.setdiag(0)
        diag_old = dcb_cc_matrix.diagonal()
        dcb_cc_matrix -= spr.diags(diag_old)
        dcc_cb_matrix = upstream @ spr.diags(np.abs(edges.flow) * dcc_cb) @ np.abs(inc.incidence)
        #dcb_cc_matrix.setdiag(0)
        diag_old = dcc_cb_matrix.diagonal()
        dcc_cb_matrix -= spr.diags(diag_old)
        dcc_cc_matrix = upstream @ spr.diags(np.abs(edges.flow) * dcc_cc) @ np.abs(inc.incidence)
        #dcc_cc_matrix.setdiag(diag)
        diag_old = dcc_cc_matrix.diagonal()
        dcc_cc_matrix += spr.diags(diag - diag_old)

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
        #dc_matrix.setdiag(diag_c)
        diag_old = dc_matrix.diagonal()
        dc_matrix += spr.diags(diag_c - diag_old)
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

import numpy as np
import scipy.sparse as spr

def solve_dissolution_mixing(
    sid: SimInputData,
    edges: Edges,
    inc: Incidence,
    graph: Graph,
    alpha_eff: spr.csr_matrix,
    max_iter: int = 50
):
    """
    Solve dissolution with mixing matrix alpha_eff in edge space.

    Unknowns:
        cB_in[e], cC_in[e]  (inlet concentrations on each edge e)

    Equations (for non-inlet edges):
        |q_k| cB_in[k] - sum_j alphaEff[k,j] |q_j| cB_out[j] = 0
        |q_k| cC_in[k] - sum_j alphaEff[k,j] |q_j| cC_out[j] = 0

    Boundary (inlet edges):
        cB_in[e] = cb_inlet_value
        cC_in[e] = cc_inlet_value
    """

    ne = sid.ne
    q_abs = np.abs(edges.flow).astype(float)
    Q = spr.diags(q_abs)              # (ne x ne)

    # Mixing flux matrix M[k,j] = alpha_eff[k,j] * |q_j|
    M = alpha_eff.T @ Q                 # (ne x ne)

    # Inlet edges mask (assume edges.inlet is (ne,) 0/1 or bool)
    inlet = np.asarray(edges.inlet, dtype=bool)
    internal = ~inlet
    inlet_a = 1 * (np.abs(inc.incidence) @ graph.in_vec_a > 0)
    inlet_b = 1 * (np.abs(inc.incidence) @ graph.in_vec_b > 0)

    # Initial guesses (you can choose something smarter)
    cB_in = np.ones(ne, dtype=float)
    cC_in = np.ones(ne, dtype=float)

    # Enforce initial BC at inlets
    cB_in[inlet_a] = sid.cb_in
    cC_in[inlet_b] = sid.cc_in

    for it in range(max_iter):
        cB_in_old = cB_in.copy()
        cC_in_old = cC_in.copy()

        # --- reaction along edges: compute cB_out, cC_out and derivatives ---

        alpha_reac = np.abs(
            sid.Da / (1.0 + sid.G * edges.diams) / sid.c_eq *
            (edges.diams > 0.0) * edges.lens / edges.flow
        )

        cb_in = cB_in
        cc_in = cC_in

        exp_a = np.exp(-alpha_reac * (cb_in - cc_in))
        alpha_reac = np.ma.fix_invalid(alpha_reac, fill_value=0.0).filled(0.0)

        exp_a_fix1 = np.ma.fix_invalid(exp_a, fill_value=0.0).filled(0.0)
        exp_a_fix2 = np.ma.fix_invalid(exp_a, fill_value=1e10).filled(1e10)
        exp_a = exp_a_fix1 * (cb_in >= cc_in) + exp_a_fix2 * (cc_in > cb_in)

        # main formulas
        num = cb_in * (cb_in - cc_in)
        den = cb_in - cc_in * exp_a

        cb_out = num / den
        cc_out = num / den + cc_in - cb_in

        cb_out = np.ma.fix_invalid(cb_out, fill_value=0.0).filled(0.0)
        cc_out = np.ma.fix_invalid(cc_out, fill_value=0.0).filled(0.0)

        # special case cb_in == cc_in
        same = (cb_in == cc_in)
        cb_out_fix = same * cb_in / (1.0 + alpha_reac * cb_in)
        cb_out += cb_out_fix
        cc_out += cb_out_fix

        # --- derivatives w.r.t cb_in and cc_in (your formulas) ---

        dcb_cb = (cb_in**2 + cc_in * exp_a *
                  (cb_in * (-2.0 - alpha_reac * (cb_in - cc_in)) + cc_in)) / (den**2)
        dcb_cb = np.ma.fix_invalid(dcb_cb, fill_value=0.0).filled(0.0)
        dcb_cb_fix = same / (1.0 + alpha_reac * cb_in)**2
        dcb_cb += dcb_cb_fix

        dcc_cb = (dcb_cb - 1.0) * (~same)

        dcb_cc = (cb_in * (-cb_in + exp_a *
                  (cb_in + alpha_reac * cc_in * (cb_in - cc_in)))) / (den**2)
        dcb_cc = np.ma.fix_invalid(dcb_cc, fill_value=0.0).filled(0.0)

        dcc_cc = (dcb_cc + 1.0) * (~same) + same / (1.0 + alpha_reac * cc_in)**2

        # --- residuals R_B, R_C (size ne each) ---

        # flux from upstream edges
        fluxB_in = M @ cb_out   # (ne,)
        fluxC_in = M @ cc_out   # (ne,)

        R_B = q_abs * cB_in - fluxB_in
        R_C = q_abs * cC_in - fluxC_in

        # Dirichlet boundary on inlet edges
        R_B[inlet_a] = cB_in[inlet_a] - sid.cb_in
        R_C[inlet_b] = cC_in[inlet_b] - sid.cc_in

        # --- Jacobian assembly (2ne x 2ne) ---

        # diag of derivatives
        D_dcb_cb = spr.diags(dcb_cb)
        D_dcb_cc = spr.diags(dcb_cc)
        D_dcc_cb = spr.diags(dcc_cb)
        D_dcc_cc = spr.diags(dcc_cc)

        # R_B = Q cB_in - M cb_out
        J_BB = Q - M @ D_dcb_cb   # ∂R_B/∂cB_in
        J_BC = - M @ D_dcb_cc     # ∂R_B/∂cC_in

        # R_C = Q cC_in - M cc_out
        J_CB = - M @ D_dcc_cb     # ∂R_C/∂cB_in
        J_CC = Q - M @ D_dcc_cc   # ∂R_C/∂cC_in

        # apply Dirichlet BC rows for inlet edges: R = c - c_bc
        if np.any(inlet):
            # zero out rows and put 1 on diagonal for inlet edges
            mask_in = spr.diags(inlet.astype(float))
            mask_int = spr.diags((~inlet).astype(float))

            # for B
            J_BB = mask_int @ J_BB + mask_in
            J_BC = mask_int @ J_BC   # zero rows at inlets

            # for C
            J_CB = mask_int @ J_CB
            J_CC = mask_int @ J_CC + mask_in

        # build big Jacobian block matrix
        J_top = spr.hstack([J_BB, J_BC])
        J_bot = spr.hstack([J_CB, J_CC])
        J = spr.vstack([J_top, J_bot]).tocsr()

        # big residual vector
        R = np.concatenate([R_B, R_C])

        # Solve J * delta = -R
        delta = solve_equation(J, -R)

        delta_B = delta[:ne]
        delta_C = delta[ne:]

        cB_in += delta_B
        cC_in += delta_C

        # clamp to non-negative
        cB_in = np.maximum(cB_in, 0.0)
        cC_in = np.maximum(cC_in, 0.0)

        # convergence check
        diff_B = np.linalg.norm(cB_in - cB_in_old)
        diff_C = np.linalg.norm(cC_in - cC_in_old)
        print("iter", it, "ΔB =", diff_B, "ΔC =", diff_C)

        if diff_B < sid.c_th and diff_C < sid.c_th:
            break

    return cB_in, cC_in #, cb_out, cc_out


import numpy as np
import scipy.sparse as spr

def reaction_along_edges(sid, edges, cb_in, cc_in):
    cb_in = np.asarray(cb_in, float)
    cc_in = np.asarray(cc_in, float)

    alpha = np.abs(
        sid.Da / (1.0 + sid.G * edges.diams) / sid.c_eq *
        (edges.diams > 0.0) * edges.lens / edges.flow
    )
    alpha = np.ma.fix_invalid(alpha, fill_value=0.0).filled(0.0)

    delta = cb_in - cc_in
    arg = -alpha * delta
    arg = np.clip(arg, -50.0, 50.0)
    exp_a = np.exp(arg)

    num = cb_in * (cb_in - cc_in)
    den = cb_in - cc_in * exp_a
    eps = 1e-12
    den_safe = np.where(np.abs(den) < eps, np.sign(den) * eps, den)

    cb_out_main = num / den_safe
    cc_out_main = num / den_safe + cc_in - cb_in

    cb_out_main = np.ma.fix_invalid(cb_out_main, fill_value=0.0).filled(0.0)
    cc_out_main = np.ma.fix_invalid(cc_out_main, fill_value=0.0).filled(0.0)

    same = np.abs(cb_in - cc_in) < 1e-10
    cb_out_same = cb_in / (1.0 + alpha * cb_in)
    cc_out_same = cb_out_same

    cb_out = np.where(same, cb_out_same, cb_out_main)
    cc_out = np.where(same, cc_out_same, cc_out_main)

    cb_out = np.maximum(cb_out, 0.0)
    cc_out = np.maximum(cc_out, 0.0)

    return cb_out, cc_out



def reaction_derivatives(sid, edges, cB_in, cC_in, cb_out, cc_out):
    """
    Derivatives of cB_out, cC_out wrt cB_in, cC_in.
    Vectorized version of your dcb_cb, dcb_cc, dcc_cb, dcc_cc.
    """

    cb_in = cB_in
    cc_in = cC_in

    alpha = np.abs(
        sid.Da / (1.0 + sid.G * edges.diams) / sid.c_eq *
        (edges.diams > 0.0) * edges.lens / edges.flow
    )
    alpha = np.ma.fix_invalid(alpha, fill_value=0.0).filled(0.0)

    delta = cb_in - cc_in
    arg = -alpha * delta
    arg = np.clip(arg, -50.0, 50.0)
    exp_a = np.exp(arg)

    num = cb_in * (cb_in - cc_in)
    den = cb_in - cc_in * exp_a
    eps = 1e-12
    den_safe = np.where(np.abs(den) < eps, np.sign(den) * eps, den)

    same = np.abs(cb_in - cc_in) < 1e-10

    # dcb_out / dcb_in
    dcb_cb = (cb_in**2 + cc_in * exp_a *
              (cb_in * (-2.0 - alpha * (cb_in - cc_in)) + cc_in)) / (den_safe**2)
    dcb_cb = np.ma.fix_invalid(dcb_cb, fill_value=0.0).filled(0.0)
    dcb_cb_fix = same / (1.0 + alpha * cb_in)**2
    dcb_cb += dcb_cb_fix

    # dcc_out / dcb_in
    dcc_cb = (dcb_cb - 1.0) * (~same)

    # dcb_out / dcc_in
    dcb_cc = (cb_in * (-cb_in + exp_a *
              (cb_in + alpha * cc_in * (cb_in - cc_in)))) / (den_safe**2)
    dcb_cc = np.ma.fix_invalid(dcb_cc, fill_value=0.0).filled(0.0)

    # dcc_out / dcc_in
    dcc_cc = (dcb_cc + 1.0) * (~same) + same / (1.0 + alpha * cc_in)**2

    return dcb_cb, dcb_cc, dcc_cb, dcc_cc


def solve_dissolution_edges_with_alpha(
    sid: SimInputData,
    edges: Edges,
    alpha_eff: spr.csr_matrix,
    cb_bc: np.ndarray,      # (ne,) inlet B concentration per edge
    cc_bc: np.ndarray,      # (ne,) inlet C concentration per edge
    inlet_mask: np.ndarray, # (ne,) bool, True for inlet edges
    max_iter: int = 50
):
    """
    Edge-based Newton solver including mixing (alpha_eff) and reaction.

    Unknowns: cB_in[e], cC_in[e]  (inlet conc on each edge)
    Equations for non-inlet edges k:

        sum_j alpha_eff[j,k] |q_j| cB_out[j] - |q_k| cB_in[k] = 0
        sum_j alpha_eff[j,k] |q_j| cC_out[j] - |q_k| cC_in[k] = 0

    For inlet edges (inlet_mask == True):
        cB_in[k] = cb_bc[k]
        cC_in[k] = cc_bc[k]
    """

    ne = sid.ne
    q_abs = np.abs(edges.flow).astype(float)
    Q = spr.diags(q_abs)

    # Flux-mixing matrix: M[k,j] = alpha_eff[j,k] * |q_j|
    # note the transpose: this matches your find_concentration operator
    #M = alpha_eff.T @ Q
    M = Q @ alpha_eff.T

    inlet = inlet_mask.astype(bool)
    internal = ~inlet

    # initial guess: BC on inlets, 0 elsewhere (you can choose something else)
    cB_in = cb_bc
    cC_in = cc_bc
    #cB_in[inlet] = cb_bc[inlet]
    #cC_in[inlet] = cc_bc[inlet]

    for it in range(max_iter):
        cB_old = cB_in.copy()
        cC_old = cC_in.copy()

        # 1) reaction along edges
        cb_out, cc_out = reaction_along_edges(sid, edges, cB_in, cC_in)

        # 2) flux into each edge via mixing
        fluxB_in = M @ cb_out   # (ne,)
        fluxC_in = M @ cc_out

        # 3) residuals
        R_B = fluxB_in - q_abs * cB_in
        R_C = fluxC_in - q_abs * cC_in

        # Dirichlet BC on inlets: c_in = c_bc
        R_B[inlet] = cB_in[inlet] - cb_bc[inlet]
        R_C[inlet] = cC_in[inlet] - cc_bc[inlet]

        # 4) Jacobian
        dcb_cb, dcb_cc, dcc_cb, dcc_cc = reaction_derivatives(
            sid, edges, cB_in, cC_in, cb_out, cc_out
        )

        D_dcb_cb = spr.diags(dcb_cb)
        D_dcb_cc = spr.diags(dcb_cc)
        D_dcc_cb = spr.diags(dcc_cb)
        D_dcc_cc = spr.diags(dcc_cc)

        # R_B = M cb_out(cB_in,cC_in) - Q cB_in
        J_BB = M @ D_dcb_cb - Q   # ∂R_B / ∂cB_in
        J_BC = M @ D_dcb_cc       # ∂R_B / ∂cC_in

        # R_C = M cc_out(cB_in,cC_in) - Q cC_in
        J_CB = M @ D_dcc_cb       # ∂R_C / ∂cB_in
        J_CC = M @ D_dcc_cc - Q   # ∂R_C / ∂cC_in

        # apply Dirichlet rows on inlets
        if np.any(inlet):
            mask_in  = spr.diags(inlet.astype(float))
            mask_int = spr.diags((~inlet).astype(float))

            # for B
            J_BB = mask_int @ J_BB + mask_in
            J_BC = mask_int @ J_BC

            # for C
            J_CB = mask_int @ J_CB
            J_CC = mask_int @ J_CC + mask_in

        # big block Jacobian
        J_top = spr.hstack([J_BB, J_BC])
        J_bot = spr.hstack([J_CB, J_CC])
        J = spr.vstack([J_top, J_bot]).tocsr()
        diag_J = J.diagonal()
        J += spr.diags(1 * (diag_J == 0))

        R = np.concatenate([R_B, R_C])

        # solve
        delta = solve_equation(J, -R)
        dB = delta[:ne]
        dC = delta[ne:]

        cB_in += dB
        cC_in += dC

        cB_in = np.maximum(cB_in, 0.0)
        cC_in = np.maximum(cC_in, 0.0)

        diff_B = np.linalg.norm(cB_in - cB_old)
        diff_C = np.linalg.norm(cC_in - cC_old)
        print(f"iter {it}: ΔB={diff_B:.3e}, ΔC={diff_C:.3e}")

        if diff_B < sid.c_th and diff_C < sid.c_th:
            break

    # final outlet concentrations
    cb_out, cc_out = reaction_along_edges(sid, edges, cB_in, cC_in)
    return cB_in, cC_in
