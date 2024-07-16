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
    edges: Edges, cc_b: spr.csc_matrix, cd_b: spr.csc_matrix, cb) -> np.ndarray:
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
        * edges.diams * edges.lens / edges.flow * sid.cd_in / sid.Kp))
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
        * edges.diams * edges.lens / edges.flow * sid.cd_in / sid.Kp))
    q_cd = np.array(np.ma.fix_invalid(q_cd, fill_value = 0))
    q_cd_matrix = np.abs(inc.incidence.T @ spr.diags(q_cd) @ inc.incidence)
    cd_matrix = c_inc.multiply(q_cd_matrix)
    cd_matrix.setdiag(diag)
    cd = solve_equation(cd_matrix, cd_b)

    cc_prev = np.zeros(sid.nsq)
    cd_prev = np.zeros(sid.nsq)

    #cb = sid.cb_in * np.ones(sid.nsq)
    cd = sid.cd_in * np.ones(sid.nsq)
    cc = sid.cb_in - cb + sid.cc_in

    qc_inc = 1 * ((spr.diags(edges.flow) @ inc.incidence > 0) != 0).T
    while np.linalg.norm(cc - cc_prev) > sid.c_th or np.linalg.norm(cd - cd_prev) > sid.c_th:
        cb_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ np.abs(cb)
        cc_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ np.abs(cc)
        cd_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ np.abs(cd)
        ksi = np.clip(cd_in * sid.K / (1 + sid.K * sid.G * edges.diams) - sid.Kp / (1 + sid.G * edges.diams), None, 0)
        exp_p = np.exp(-sid.Da * sid.K / (1 + sid.G * sid.K * edges.diams) \
            * cd_in / sid.Kp * edges.diams * edges.lens / np.abs(edges.flow))
        exp_p = np.array(np.ma.fix_invalid(exp_p, fill_value = 0))
        exp_d = np.exp(-sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow))
        exp_d = np.array(np.ma.fix_invalid(exp_d, fill_value = 0))

        qc2 = -np.abs(edges.flow) / (1 + sid.G * edges.diams) * sid.Kp * (1 - exp_d) / ksi
        qc2 = np.array(np.ma.fix_invalid(qc2, fill_value = 0))
        qc_matrix = np.abs(inc.incidence.T) @ spr.diags(qc2) @ np.abs(inc.incidence)
        cc_matrix = c_inc.multiply(qc_matrix)
        cc_matrix.setdiag(np.zeros(sid.nsq))
        f_cc2 = cc_matrix @ cb
        #np.savetxt('fcc2.txt', f_cc2)

        qc1 = np.abs(edges.flow) * exp_p
        qc_matrix = np.abs(inc.incidence.T) @ spr.diags(qc1) @ np.abs(inc.incidence)
        cc_matrix = c_inc.multiply(qc_matrix)
        cc_matrix.setdiag(diag)
        f_cc1 = cc_matrix @ cc



        # qc = edges.flow * sid.Kp / (sid.Kp - cd_in) * (np.exp(-sid.Da / (1 + sid.G * \
        #     edges.diams) * edges.diams * edges.lens / np.abs(edges.flow)) - \
        #     np.exp(-sid.Da * sid.K / (1 + sid.G * sid.K * edges.diams) \
        #     * edges.diams * edges.lens / np.abs(edges.flow) * cd_in / sid.Kp))
        # qc = np.array(np.ma.fix_invalid(qc, fill_value = 0))
        # qc_matrix = np.abs(inc.incidence.T @ spr.diags(qc) @ inc.incidence)
        # cb_matrix = c_inc.multiply(qc_matrix)
        # cb_matrix.setdiag(np.zeros(sid.nsq)) # set diagonal to zero
        # cc_b = cb_matrix @ cb

        f_cc = (1 - graph.in_vec) * (f_cc1 + f_cc2)

        # q_cc = edges.flow * exp_p
        # q_cc_matrix = np.abs(inc.incidence.T @ spr.diags(q_cc) @ inc.incidence)
        # cc_matrix = c_inc.multiply(q_cc_matrix)
        # cc_matrix.setdiag(diag)
        
        # dq_cc_cd = - sid.K / (sid.Kp * ksi ** 2) \
        #     * (exp_p * cc_in * ksi ** 2 * sid.Da * edges.diams * edges.lens - cb_in * sid.Kp \
        #     * (sid.Da * edges.diams * edges.lens * ksi + sid.Kp * np.abs(edges.flow) * (1 - exp_d)))
        
        dq_cc_cd = -np.abs(edges.flow) * cb_in * (1 - exp_d) / (1 + sid.G * edges.diams) / (1 + sid.G * edges.diams * sid.K) * sid.Kp * sid.K / ksi ** 2 \
            - cc_in * exp_p * sid.K * edges.diams * edges.lens * sid.Da / sid.Kp / (1 + sid.G * edges.diams * sid.K)
        dq_cc_cd = np.array(np.ma.fix_invalid(dq_cc_cd, fill_value = 0))
        dcc_cd_matrix = np.abs(inc.incidence.T) @ spr.diags(dq_cc_cd) @ np.abs(inc.incidence)
        dcc_cd_matrix.setdiag(np.zeros(sid.nsq))

        dq_cd_cc = np.abs(edges.flow) * (exp_p - 1)
        dq_cd_cc_matrix = np.abs(inc.incidence.T) @ spr.diags(dq_cd_cc) @ np.abs(inc.incidence)
        dq_cd_cc_matrix = c_inc.multiply(dq_cd_cc_matrix)
        dq_cd_cc_matrix.setdiag(np.zeros(sid.nsq))

        qc = np.abs(edges.flow) * (1 - np.exp(-np.abs(sid.Da / (1 + sid.G * edges.diams) \
            * edges.diams * edges.lens / edges.flow)))
        qc = np.array(np.ma.fix_invalid(qc, fill_value = 0))
        qc_matrix = np.abs(inc.incidence.T @ spr.diags(qc) @ inc.incidence)
        cb_matrix = c_inc.multiply(qc_matrix)
        cb_matrix.setdiag(np.zeros(sid.nsq))

        cd_mix = cd
        cd_matrix = c_inc.multiply(np.abs(inc.incidence.T @ spr.diags(np.abs(edges.flow)) @ inc.incidence))
        cd_matrix.setdiag(diag)
        f_cd = (1 - graph.in_vec) * (cd_matrix @ cd_mix + dq_cd_cc_matrix @ cc + f_cc2 - cb_matrix @ cb)
        #np.savetxt('f_cd.txt', f_cd)
        #np.savetxt('f_cc.txt', f_cc)
        dq_cd_cd = cd_matrix + dcc_cd_matrix



        dc_matrix = spr.vstack([spr.hstack([cc_matrix, dcc_cd_matrix]), spr.hstack([dq_cd_cc_matrix, dq_cd_cd])])
        f = np.concatenate((f_cc, f_cd))
        in_vec = np.concatenate((graph.in_vec, graph.in_vec))
        dc_matrix = spr.diags(1 - in_vec) @ dc_matrix + spr.diags(in_vec)
        delta_c = solve_equation(dc_matrix, -f)
        cc_prev = cc.copy()
        cd_prev = cd.copy()
        cc += delta_c[:sid.nsq]
        cd += delta_c[sid.nsq:]
        print(np.linalg.norm(cc - cc_prev), np.linalg.norm(cd - cd_prev))
        # np.savetxt('cb.txt', cb)
        # np.savetxt('cc.txt', cc)
        # np.savetxt('cd.txt', cd)
        # np.savetxt('cc_p.txt', cc_prev)
        # np.savetxt('cd_p.txt', cd_prev)
        # np.savetxt('dc.txt', dc_matrix.toarray())
        # np.savetxt('inc.txt', (inc.incidence.T @ inc.incidence).toarray())
        if np.sum(cd < -1):
            print(cc)
            print(cd)
            raise ValueError
    #print(cc)
    #print(cd)
    return cc, cd


def solve_precipitation_nr2(sid: SimInputData, inc: Incidence, graph: Graph, \
    edges: Edges, cc_b: spr.csc_matrix, cd_b: spr.csc_matrix, cb, cc, cd) -> np.ndarray:
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
    diag = -np.abs(inc.incidence.T) @ np.abs(edges.flow) / 2
    diag = diag * (1 - graph.in_vec + graph.out_vec) + graph.in_vec
    diag += 1 * (diag == 0)


    cc_prev = np.zeros(sid.nsq)
    cd_prev = np.zeros(sid.nsq)

    while np.linalg.norm(cc - cc_prev) > sid.c_th or np.linalg.norm(cd - cd_prev) > sid.c_th:
        cb_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ np.abs(cb)
        cc_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ np.abs(cc)
        cd_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ np.abs(cd)
        ksi = np.clip(cd_in * sid.K / (1 + sid.K * sid.G * edges.diams) - sid.Kp / (1 + sid.G * edges.diams), None, 0)
        exp_p = np.exp(-sid.Da * sid.K / (1 + sid.G * sid.K * edges.diams) \
            * cd_in / sid.Kp * edges.diams * edges.lens / np.abs(edges.flow))
        exp_p = np.array(np.ma.fix_invalid(exp_p, fill_value = 0))
        exp_d = np.exp(-sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow))
        exp_d = np.array(np.ma.fix_invalid(exp_d, fill_value = 0))

        qc2 = -np.abs(edges.flow) / (1 + sid.G * edges.diams) * sid.Kp * (1 - exp_d) / ksi
        qc2 = np.array(np.ma.fix_invalid(qc2, fill_value = 0))
        qc_matrix = np.abs(inc.incidence.T) @ spr.diags(qc2) @ np.abs(inc.incidence)
        cc_matrix = c_inc.multiply(qc_matrix)
        cc_matrix.setdiag(np.zeros(sid.nsq))
        f_cc2 = cc_matrix @ cb
        #np.savetxt('fcc2.txt', f_cc2)

        qc1 = np.abs(edges.flow) * exp_p
        qc_matrix = np.abs(inc.incidence.T) @ spr.diags(qc1) @ np.abs(inc.incidence)
        cc_matrix = c_inc.multiply(qc_matrix)
        cc_matrix.setdiag(diag)
        f_cc1 = cc_matrix @ cc



        # qc = edges.flow * sid.Kp / (sid.Kp - cd_in) * (np.exp(-sid.Da / (1 + sid.G * \
        #     edges.diams) * edges.diams * edges.lens / np.abs(edges.flow)) - \
        #     np.exp(-sid.Da * sid.K / (1 + sid.G * sid.K * edges.diams) \
        #     * edges.diams * edges.lens / np.abs(edges.flow) * cd_in / sid.Kp))
        # qc = np.array(np.ma.fix_invalid(qc, fill_value = 0))
        # qc_matrix = np.abs(inc.incidence.T @ spr.diags(qc) @ inc.incidence)
        # cb_matrix = c_inc.multiply(qc_matrix)
        # cb_matrix.setdiag(np.zeros(sid.nsq)) # set diagonal to zero
        # cc_b = cb_matrix @ cb

        f_cc = (1 - graph.in_vec) * (f_cc1 + f_cc2)

        # q_cc = edges.flow * exp_p
        # q_cc_matrix = np.abs(inc.incidence.T @ spr.diags(q_cc) @ inc.incidence)
        # cc_matrix = c_inc.multiply(q_cc_matrix)
        # cc_matrix.setdiag(diag)
        
        # dq_cc_cd = - sid.K / (sid.Kp * ksi ** 2) \
        #     * (exp_p * cc_in * ksi ** 2 * sid.Da * edges.diams * edges.lens - cb_in * sid.Kp \
        #     * (sid.Da * edges.diams * edges.lens * ksi + sid.Kp * np.abs(edges.flow) * (1 - exp_d)))
        
        dq_cc_cd = -np.abs(edges.flow) * cb_in * (1 - exp_d) / (1 + sid.G * edges.diams) / (1 + sid.G * edges.diams * sid.K) * sid.Kp * sid.K / ksi ** 2 \
            - cc_in * exp_p * sid.K * edges.diams * edges.lens * sid.Da / sid.Kp / (1 + sid.G * edges.diams * sid.K)
        dq_cc_cd = np.array(np.ma.fix_invalid(dq_cc_cd, fill_value = 0))
        dcc_cd_matrix = np.abs(inc.incidence.T) @ spr.diags(dq_cc_cd) @ np.abs(inc.incidence)
        dcc_cd_matrix.setdiag(np.zeros(sid.nsq))

        dq_cd_cc = np.abs(edges.flow) * (exp_p - 1)
        dq_cd_cc_matrix = np.abs(inc.incidence.T) @ spr.diags(dq_cd_cc) @ np.abs(inc.incidence)
        dq_cd_cc_matrix = c_inc.multiply(dq_cd_cc_matrix)
        dq_cd_cc_matrix.setdiag(np.zeros(sid.nsq))

        qc = np.abs(edges.flow) * (1 - np.exp(-np.abs(sid.Da / (1 + sid.G * edges.diams) \
            * edges.diams * edges.lens / edges.flow)))
        qc = np.array(np.ma.fix_invalid(qc, fill_value = 0))
        qc_matrix = np.abs(inc.incidence.T @ spr.diags(qc) @ inc.incidence)
        cb_matrix = c_inc.multiply(qc_matrix)
        cb_matrix.setdiag(np.zeros(sid.nsq))

        cd_mix = cd
        cd_matrix = c_inc.multiply(np.abs(inc.incidence.T @ spr.diags(np.abs(edges.flow)) @ inc.incidence))
        cd_matrix.setdiag(diag)
        f_cd = (1 - graph.in_vec) * (cd_matrix @ cd_mix + dq_cd_cc_matrix @ cc + f_cc2 - cb_matrix @ cb)
        #np.savetxt('f_cd.txt', f_cd)
        #np.savetxt('f_cc.txt', f_cc)
        dq_cd_cd = cd_matrix + dcc_cd_matrix



        dc_matrix = spr.vstack([spr.hstack([cc_matrix, dcc_cd_matrix]), spr.hstack([dq_cd_cc_matrix, dq_cd_cd])])
        f = np.concatenate((f_cc, f_cd))
        in_vec = np.concatenate((graph.in_vec, graph.in_vec))
        dc_matrix = spr.diags(1 - in_vec) @ dc_matrix + spr.diags(in_vec)
        delta_c = solve_equation(dc_matrix, -f)
        cc_prev = cc.copy()
        cd_prev = cd.copy()
        cc += delta_c[:sid.nsq]
        cd += delta_c[sid.nsq:]
        print(np.linalg.norm(cc - cc_prev), np.linalg.norm(cd - cd_prev))
        # np.savetxt('cb.txt', cb)
        # np.savetxt('cc.txt', cc)
        # np.savetxt('cd.txt', cd)
        # np.savetxt('cc_p.txt', cc_prev)
        # np.savetxt('cd_p.txt', cd_prev)
        # np.savetxt('dc.txt', dc_matrix.toarray())
        # np.savetxt('inc.txt', (inc.incidence.T @ inc.incidence).toarray())
        if np.sum(cd < -1):
            print(cc)
            print(cd)
            raise ValueError
    #print(cc)
    #print(cd)
    return cc, cd



def solve_precipitation_nr_vx(sid: SimInputData, inc: Incidence, graph: Graph, \
    edges: Edges, cc_b: spr.csc_matrix, cd_b: spr.csc_matrix, cb) -> np.ndarray:
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
        * edges.diams * edges.lens / edges.flow * sid.cd_in / sid.Kp))
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
        * edges.diams * edges.lens / edges.flow * sid.cd_in / sid.Kp))
    q_cd = np.array(np.ma.fix_invalid(q_cd, fill_value = 0))
    q_cd_matrix = np.abs(inc.incidence.T @ spr.diags(q_cd) @ inc.incidence)
    cd_matrix = c_inc.multiply(q_cd_matrix)
    cd_matrix.setdiag(diag)
    cd = solve_equation(cd_matrix, cd_b)

    cc_prev = np.zeros(sid.nsq)
    cd_prev = np.zeros(sid.nsq)

    #cd = sid.cd_in * np.ones(sid.nsq)
    #cc = sid.cb_in - cb + sid.cc_in

    while np.linalg.norm(cc - cc_prev) > 0.5 or np.linalg.norm(cd - cd_prev) > 0.5: #sid.c_th:
        cb_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ np.abs(cb)
        cc_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ np.abs(cc)
        cd_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ np.abs(cd)
        ksi = cd_in * sid.K / (1 + sid.K * sid.G * edges.diams) - sid.Kp / (1 + sid.G * edges.diams)
        if np.sum(ksi >= 0):
            print('ksi!')
            np.savetxt('ksi.txt', ksi)
        exp_p = np.exp(-sid.Da * sid.K / (1 + sid.G * sid.K * edges.diams) \
            * cd_in / sid.Kp * edges.diams * edges.lens / np.abs(edges.flow))
        exp_p = np.array(np.ma.fix_invalid(exp_p, fill_value = 0))
        exp_d = np.exp(-sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow))
        exp_d = np.array(np.ma.fix_invalid(exp_d, fill_value = 0))

        qc2 = -np.abs(edges.flow) / (1 + sid.G * edges.diams) * sid.Kp * (exp_p - exp_d) / ksi
        qc2 = np.array(np.ma.fix_invalid(qc2, fill_value = 0))
        qc_matrix = np.abs(inc.incidence.T) @ spr.diags(qc2) @ np.abs(inc.incidence)
        cc_matrix = c_inc.multiply(qc_matrix)
        cc_matrix.setdiag(np.zeros(sid.nsq))
        f_cc2 = cc_matrix @ cb
        #np.savetxt('fcc2.txt', f_cc2)

        qc1 = np.abs(edges.flow) * exp_p
        qc_matrix = np.abs(inc.incidence.T) @ spr.diags(qc1) @ np.abs(inc.incidence)
        cc_matrix = c_inc.multiply(qc_matrix)
        cc_matrix.setdiag(diag)
        f_cc1 = cc_matrix @ cc

        f_cc = (1 - graph.in_vec) * (f_cc1 + f_cc2)

        dq_cc_cd = cb_in / (1 + sid.G * edges.diams) / (1 + sid.G * edges.diams * sid.K) * sid.K / ksi ** 2 \
            * (sid.Da * edges.diams * edges.lens * ksi * exp_p + np.abs(edges.flow) * sid.Kp * (exp_p - exp_d)) \
            - cc_in * exp_p * sid.K * edges.diams * edges.lens * sid.Da / sid.Kp / (1 + sid.G * edges.diams * sid.K)
        dq_cc_cd = np.array(np.ma.fix_invalid(dq_cc_cd, fill_value = - cc_in * exp_p * sid.K * edges.diams * edges.lens * sid.Da / sid.Kp / (1 + sid.G * edges.diams * sid.K)))
        dcc_cd_matrix = np.abs(inc.incidence.T) @ spr.diags(dq_cc_cd) @ np.abs(inc.incidence)
        dcc_cd_matrix.setdiag(np.zeros(sid.nsq))

        dq_cd_cc = np.abs(edges.flow) * (exp_p - 1)
        dq_cd_cc_matrix = np.abs(inc.incidence.T) @ spr.diags(dq_cd_cc) @ np.abs(inc.incidence)
        dq_cd_cc_matrix = c_inc.multiply(dq_cd_cc_matrix)
        dq_cd_cc_matrix.setdiag(np.zeros(sid.nsq))

        qc = np.abs(edges.flow) * (1 - np.exp(-np.abs(sid.Da / (1 + sid.G * edges.diams) \
            * edges.diams * edges.lens / edges.flow)))
        qc = np.array(np.ma.fix_invalid(qc, fill_value = 0))
        qc_matrix = np.abs(inc.incidence.T @ spr.diags(qc) @ inc.incidence)
        cb_matrix = c_inc.multiply(qc_matrix)
        cb_matrix.setdiag(np.zeros(sid.nsq))

        cd_mix = cd
        cd_matrix = c_inc.multiply(np.abs(inc.incidence.T @ spr.diags(np.abs(edges.flow)) @ inc.incidence))
        cd_matrix.setdiag(diag)
        f_cd = (1 - graph.in_vec) * (cd_matrix @ cd_mix + dq_cd_cc_matrix @ cc + f_cc2 - cb_matrix @ cb)
        #np.savetxt('f_cd.txt', f_cd)
        #np.savetxt('f_cc.txt', f_cc)
        dq_cd_cd = cd_matrix + dcc_cd_matrix



        dc_matrix = spr.vstack([spr.hstack([cc_matrix, dcc_cd_matrix]), spr.hstack([dq_cd_cc_matrix, dq_cd_cd])])
        f = np.concatenate((f_cc, f_cd))
        in_vec = np.concatenate((graph.in_vec, graph.in_vec))
        dc_matrix = spr.diags(1 - in_vec) @ dc_matrix + spr.diags(in_vec)
        delta_c = solve_equation(dc_matrix, -f)
        cc_prev = cc.copy()
        cd_prev = cd.copy()
        cc += delta_c[:sid.nsq]
        cd += delta_c[sid.nsq:]
        print(np.linalg.norm(cc - cc_prev), np.linalg.norm(cd - cd_prev))

        if np.sum(cd < -1):
            print(cc)
            print(cd)
            raise ValueError

    return cc, cd

def solve_precipitation_nr2_vx(sid: SimInputData, inc: Incidence, graph: Graph, \
    edges: Edges, cc_b: spr.csc_matrix, cd_b: spr.csc_matrix, cb, cc, cd) -> np.ndarray:
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
    diag = -np.abs(inc.incidence.T) @ np.abs(edges.flow) / 2
    diag = diag * (1 - graph.in_vec + graph.out_vec) + graph.in_vec
    diag += 1 * (diag == 0)
    
    if np.sum(cd) < 1.1 * np.sum(sid.cd_in * edges.inlet):
        cd = sid.cd_in * np.ones(sid.nsq)
        cc = sid.cb_in - cb + sid.cc_in
        print('correcting initial cc, cd')

    cc_prev = np.zeros(sid.nsq)
    cd_prev = np.zeros(sid.nsq)
    it = 0
    while np.linalg.norm(cc - cc_prev) > sid.c_th or np.linalg.norm(cd - cd_prev) > sid.c_th:
        it += 1
        cb_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ np.abs(cb)
        cc_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ np.abs(cc)
        cd_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ np.abs(cd)
        ksi = cd_in * sid.K / (1 + sid.K * sid.G * edges.diams) - sid.Kp / (1 + sid.G * edges.diams)
        if np.sum(ksi >= -0.001):
            print('ksi!')
            np.savetxt('ksi.txt', ksi)
        exp_p = np.exp(-sid.Da * sid.K / (1 + sid.G * sid.K * edges.diams) \
            * cd_in / sid.Kp * edges.diams * edges.lens / np.abs(edges.flow))
        exp_p = np.array(np.ma.fix_invalid(exp_p, fill_value = 0))
        exp_d = np.exp(-sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow))
        exp_d = np.array(np.ma.fix_invalid(exp_d, fill_value = 0))

        qc2 = -np.abs(edges.flow) / (1 + sid.G * edges.diams) * sid.Kp * (exp_p - exp_d) / ksi
        qc2 = np.array(np.ma.fix_invalid(qc2, fill_value = edges.diams * edges.lens * sid.Da / (1 + sid.G * edges.diams)))
        qc_matrix = np.abs(inc.incidence.T) @ spr.diags(qc2) @ np.abs(inc.incidence)
        cc_matrix = c_inc.multiply(qc_matrix)
        cc_matrix.setdiag(np.zeros(sid.nsq))
        f_cc2 = cc_matrix @ cb
        #np.savetxt('fcc2.txt', f_cc2)

        qc1 = np.abs(edges.flow) * exp_p
        qc_matrix = np.abs(inc.incidence.T) @ spr.diags(qc1) @ np.abs(inc.incidence)
        cc_matrix = c_inc.multiply(qc_matrix)
        cc_matrix.setdiag(diag)
        f_cc1 = cc_matrix @ cc

        f_cc = (1 - graph.in_vec) * (f_cc1 + f_cc2)

        dq_cc_cd = cb_in / (1 + sid.G * edges.diams) / (1 + sid.G * edges.diams * sid.K) * sid.K / ksi ** 2 \
            * (sid.Da * edges.diams * edges.lens * ksi * exp_p + np.abs(edges.flow) * sid.Kp * (exp_p - exp_d)) \
            - cc_in * exp_p * sid.K * edges.diams * edges.lens * sid.Da / sid.Kp / (1 + sid.G * edges.diams * sid.K)
        dq_cc_cd = np.array(np.ma.fix_invalid(dq_cc_cd, fill_value = 0))
        dcc_cd_matrix = np.abs(inc.incidence.T) @ spr.diags(dq_cc_cd) @ np.abs(inc.incidence)
        dcc_cd_matrix.setdiag(np.zeros(sid.nsq))

        dq_cd_cc = np.abs(edges.flow) * (exp_p - 1)
        dq_cd_cc_matrix = np.abs(inc.incidence.T) @ spr.diags(dq_cd_cc) @ np.abs(inc.incidence)
        dq_cd_cc_matrix = c_inc.multiply(dq_cd_cc_matrix)
        dq_cd_cc_matrix.setdiag(np.zeros(sid.nsq))

        qc = np.abs(edges.flow) * (1 - exp_d)
        qc_matrix = np.abs(inc.incidence.T @ spr.diags(qc) @ inc.incidence)
        cb_matrix = c_inc.multiply(qc_matrix)
        cb_matrix.setdiag(np.zeros(sid.nsq))

        cd_mix = cd
        cd_matrix = c_inc.multiply(np.abs(inc.incidence.T @ spr.diags(np.abs(edges.flow)) @ inc.incidence))
        cd_matrix.setdiag(diag)
        f_cd = (1 - graph.in_vec) * (cd_matrix @ cd_mix + dq_cd_cc_matrix @ cc + f_cc2 - cb_matrix @ cb)
        #np.savetxt('f_cd.txt', f_cd)
        #np.savetxt('f_cc.txt', f_cc)
        dq_cd_cd = cd_matrix + dcc_cd_matrix



        dc_matrix = spr.vstack([spr.hstack([cc_matrix, dcc_cd_matrix]), spr.hstack([dq_cd_cc_matrix, dq_cd_cd])])
        f = np.concatenate((f_cc, f_cd))
        c_zero = 1 * (np.concatenate((cc, cd)) <= 0)
        f = f * (1 - c_zero)
        dc_matrix = spr.diags(1 - 1 * (f == 0)) @ dc_matrix + spr.diags(1 * (f == 0))
        #in_vec = np.concatenate((graph.in_vec, graph.in_vec))
        #dc_matrix = spr.diags(1 - in_vec) @ dc_matrix + spr.diags(in_vec)
        delta_c = solve_equation(dc_matrix, -f)
        cc_prev = cc.copy()
        cd_prev = cd.copy()
        cc += delta_c[:sid.nsq]
        cd += delta_c[sid.nsq:]
        cc = np.clip(cc, 0, None)
        cd = np.clip(cd, 0, None)
        print(np.linalg.norm(cc - cc_prev), np.linalg.norm(cd - cd_prev))
        if it > 100:
            cd = sid.cd_in * np.ones(sid.nsq)
            cc = sid.cb_in - cb + sid.cc_in
            # np.savetxt('dc.txt', dc_matrix.toarray())

            # np.savetxt('cc.txt', cc)
            # np.savetxt('cd.txt', cd)
            print('restarting iterations')
        elif it > 300:
            raise ValueError('NR didnt converge')
    return cc, cd


def solve_precipitation_nr2_vx2(sid: SimInputData, inc: Incidence, graph: Graph, \
    edges: Edges, cc_b: spr.csc_matrix, cd_b: spr.csc_matrix, cb, cc, cd) -> np.ndarray:
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
    diag = -np.abs(inc.incidence.T) @ np.abs(edges.flow) / 2
    diag = diag * (1 - graph.in_vec + graph.out_vec) + graph.in_vec
    diag += 1 * (diag == 0)


    cc_prev = np.zeros(sid.nsq)
    cd_prev = np.zeros(sid.nsq)
    it = 0
    while np.linalg.norm(cc - cc_prev) > sid.c_th or np.linalg.norm(cd - cd_prev) > sid.c_th:
        it += 1
        cb_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ cb
        cc_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ cc
        cd_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ cd
        ksi = 1 - cd_in * sid.K / sid.Kp * (1 + sid.G * edges.diams) / (1 + sid.K * sid.G * edges.diams)
        np.savetxt('ksi.txt', ksi)
        if np.sum(1 * (ksi == 0)):
            print('ksi!')
            np.savetxt('ksi.txt', ksi)
        exp_p = np.exp(-sid.Da * sid.K / (1 + sid.G * sid.K * edges.diams) \
            * cd_in / sid.Kp * edges.diams * edges.lens / np.abs(edges.flow))
        exp_p = np.array(np.ma.fix_invalid(exp_p, fill_value = 0))
        exp_d = np.exp(-sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow))
        exp_d = np.array(np.ma.fix_invalid(exp_d, fill_value = 0))

        qc2 = np.abs(edges.flow) * (exp_d - exp_p) / ksi
        qc2 = np.array(np.ma.fix_invalid(qc2, fill_value = edges.diams * edges.lens * sid.Da / (1 + sid.G * edges.diams)))
        qc_matrix = np.abs(inc.incidence.T) @ spr.diags(qc2) @ np.abs(inc.incidence)
        cc_matrix = c_inc.multiply(qc_matrix)
        cc_matrix.setdiag(np.zeros(sid.nsq))
        f_cc2 = cc_matrix @ cb
        #np.savetxt('fcc2.txt', f_cc2)

        qc1 = np.abs(edges.flow) * exp_p
        qc_matrix = np.abs(inc.incidence.T) @ spr.diags(qc1) @ np.abs(inc.incidence)
        cc_matrix = c_inc.multiply(qc_matrix)
        cc_matrix.setdiag(diag)
        f_cc1 = cc_matrix @ cc

        f_cc = (1 - graph.in_vec) * (f_cc1 + f_cc2)

        dq_cc_cd = cb_in / (1 + sid.G * edges.diams * sid.K) * sid.K / sid.Kp / ksi ** 2 \
            * (sid.Da * edges.diams * edges.lens * ksi * exp_p + np.abs(edges.flow) * (1 + sid.G * edges.diams) * (exp_d - exp_p)) \
            - cc_in * exp_p * sid.K * edges.diams * edges.lens * sid.Da / sid.Kp / (1 + sid.G * edges.diams * sid.K)
        dq_cc_cd = np.array(np.ma.fix_invalid(dq_cc_cd, fill_value = 0))
        dcc_cd_matrix = np.abs(inc.incidence.T) @ spr.diags(dq_cc_cd) @ np.abs(inc.incidence)
        dcc_cd_matrix.setdiag(np.zeros(sid.nsq))

        dq_cd_cc = np.abs(edges.flow) * (exp_p - 1)
        dq_cd_cc_matrix = np.abs(inc.incidence.T) @ spr.diags(dq_cd_cc) @ np.abs(inc.incidence)
        dq_cd_cc_matrix = c_inc.multiply(dq_cd_cc_matrix)
        dq_cd_cc_matrix.setdiag(np.zeros(sid.nsq))

        qc = np.abs(edges.flow) * (1 - exp_d)
        qc_matrix = np.abs(inc.incidence.T @ spr.diags(qc) @ inc.incidence)
        cb_matrix = c_inc.multiply(qc_matrix)
        cb_matrix.setdiag(np.zeros(sid.nsq))

        cd_mix = cd
        cd_matrix = c_inc.multiply(np.abs(inc.incidence.T @ spr.diags(np.abs(edges.flow)) @ inc.incidence))
        cd_matrix.setdiag(diag)
        f_cd = (1 - graph.in_vec) * (cd_matrix @ cd_mix + dq_cd_cc_matrix @ cc + f_cc2 - cb_matrix @ cb)

        dq_cd_cd = cd_matrix + dcc_cd_matrix



        dc_matrix = spr.vstack([spr.hstack([cc_matrix, dcc_cd_matrix]), spr.hstack([dq_cd_cc_matrix, dq_cd_cd])])
        f = np.concatenate((f_cc, f_cd))
        in_vec = np.concatenate((graph.in_vec, graph.in_vec))
        c_zero = 1 * (np.concatenate((cc, cd)) == 0)
        f = f * (1 - c_zero)
        dc_matrix = spr.diags(1 - 1 * (f == 0)) @ dc_matrix + spr.diags(1 * (f == 0))
        delta_c = solve_equation(dc_matrix, -f)
        cc_prev = cc.copy()
        cd_prev = cd.copy()
        cc += delta_c[:sid.nsq]
        cd += delta_c[sid.nsq:]
        cc = np.clip(cc, 0, None)
        cd = np.clip(cd, 0, None)
        print(np.linalg.norm(cc - cc_prev), np.linalg.norm(cd - cd_prev))
        if np.linalg.norm(cc - cc_prev) > 1e5:
            np.savetxt('dc.txt', dc_matrix.toarray())
            np.savetxt('cc.txt', cc)
            np.savetxt('cd.txt', cd)
            np.savetxt('f_cd.txt', f_cd)
            np.savetxt('f_cc.txt', f_cc)
            raise ValueError('norm')
        if it > 100:
            #np.savetxt('dc.txt', dc_matrix.toarray())
            np.savetxt('f_cd.txt', f_cd)
            np.savetxt('f_cc.txt', f_cc)
            np.savetxt('cc.txt', cc)
            np.savetxt('cd.txt', cd)
            raise ValueError
        if np.sum(cd < 0):
            raise ValueError('cd negative') 
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
