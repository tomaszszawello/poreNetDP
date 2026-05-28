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
from volumes import Volumes


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
    return sid.cb_0 * graph.in_vec

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
    # find vector with non-diagonal coefficients
    qc = edges.flow * np.exp(-np.abs(sid.Da / (1 + sid.G * edges.diams) \
            * edges.diams * edges.lens / edges.flow))
    qc = np.array(np.ma.fix_invalid(qc, fill_value = 0))
    qc_matrix = np.abs(inc.incidence.T @ spr.diags(qc) @ inc.incidence)
    cb_matrix = cb_inc.multiply(qc_matrix)
    # find diagonal coefficients (inlet flow for each node)
    diag = -np.abs(inc.incidence.T) @ np.abs(edges.flow) / 2
    # set diagonal for input nodes to 1
    diag = diag * (1 - graph.in_vec + graph.out_vec) + graph.in_vec
    diag += 1 * (diag == 0)
    # replace diagonal
    diag_old = cb_matrix.diagonal()
    cb_matrix += spr.diags(diag - diag_old)
    cb = solve_equation(cb_matrix, cb_b)
    return cb

def solve_dissolution_nucleation(sid: SimInputData, inc: Incidence, graph: Graph, \
    edges: Edges, cb_b: spr.csc_matrix) -> np.ndarray:
    """ Calculate B concentration with passivated fraction
    TODO: Unfortunate code duplication here - want to preserve main-loop logic
    """
    # find incidence for cb (only upstream flow matters)
    cb_inc = 1 * (inc.incidence.T @ (spr.diags(edges.flow) \
        @ inc.incidence > 0) != 0)
    # find vector with non-diagonal coefficients
    qc = edges.flow * np.exp(-np.abs((1 - edges.ftrans) * sid.Da / (1 + sid.G * edges.diams) \
            * edges.diams * edges.lens / edges.flow))
    qc = np.array(np.ma.fix_invalid(qc, fill_value = 0))
    qc_matrix = np.abs(inc.incidence.T @ spr.diags(qc) @ inc.incidence)
    cb_matrix = cb_inc.multiply(qc_matrix)
    # find diagonal coefficients (inlet flow for each node)
    diag = -np.abs(inc.incidence.T) @ np.abs(edges.flow) / 2
    # set diagonal for input nodes to 1
    diag = diag * (1 - graph.in_vec + graph.out_vec) + graph.in_vec
    diag += 1 * (diag == 0)
    # replace diagonal
    diag_old = cb_matrix.diagonal()
    cb_matrix += spr.diags(diag - diag_old)
    cb = solve_equation(cb_matrix, cb_b)
    return cb


def solve_dissolution_nr(sid: SimInputData, inc: Incidence, graph: Graph, \
    edges: Edges, vols: Volumes, cb_b) -> np.ndarray:
    """ Calculate B concentration with tracking of A volume.
    """
    # find incidence for cb (only upstream flow matters)
    cb_inc = 1 * (inc.incidence.T @ (spr.diags(edges.flow) \
        @ inc.incidence > 0) != 0)
    # find diagonal coefficients (inlet flow for each node)
    diag = -np.abs(inc.incidence.T) @ np.abs(edges.flow) / 2
    # multiply diagonal for output nodes (they have no outlet, so inlet flow
    # is equal to whole flow); also fix for nodes which are connected only to
    # other out_nodes - without it we get a singular matrix
    diag = diag * (1 - graph.in_vec + graph.out_vec) + graph.in_vec
    diag += 1 * (diag == 0)

    max_proj_iters = getattr(sid, "proj_iters", 3)
    eps = 1e-12

    alpha_b_tr = 1. * (vols.vol_a > 0)
    alpha_b = np.array(np.ma.fix_invalid((vols.triangles @ alpha_b_tr) \
        / edges.triangles, fill_value = 0.))

    for it_alpha in range(max_proj_iters):
        exp_b = np.exp(-np.abs(sid.Da * alpha_b / (1 + sid.G * edges.diams) * \
            edges.diams * edges.lens / edges.flow))
        exp_b = np.array(np.ma.fix_invalid(exp_b, fill_value = 0.)) # fix for 0 / 0
        # if there is available volume, we include reduction of cb in pore
        # (exp_b != 0), if not, then q_in cb_in = q_out cb_out (exp_b = 0)
        qc = edges.flow * exp_b
        qc_matrix = np.abs(inc.incidence.T @ spr.diags(qc) @ inc.incidence)
        cb_matrix = cb_inc.multiply(qc_matrix)
        diag_old = cb_matrix.diagonal()
        cb_matrix += spr.diags(diag - diag_old)
        cb = solve_equation(cb_matrix, cb_b) # calculate concentration of B

        growth_matrix = np.abs((spr.diags(edges.flow) @ inc.incidence > 0))
        cb_growth = growth_matrix @ cb # choose upstream concentration of B for the
        # calculation of growth
        growth = cb_growth * np.abs(edges.flow) / sid.Da * (1 - exp_b) * sid.dt
        growth = np.array(np.ma.fix_invalid(growth, fill_value = 0.)) # fix for
        # zero surface

        vol_a_dissolved = vols.triangles.T @ (growth / edges.triangles)
        vol_a_dissolved = np.array(np.ma.fix_invalid(vol_a_dissolved, fill_value = 0.))

        alpha_scale = np.ones(sid.ntr)
        alpha_scale = np.minimum(alpha_scale, \
            np.divide(vols.vol_a, vol_a_dissolved + eps))

        if np.all(alpha_scale >= 1. - eps):
            break

        alpha_b_tr = np.clip(alpha_b_tr * alpha_scale, 0., 1.)
        alpha_b = np.array(np.ma.fix_invalid((vols.triangles @ alpha_b_tr) \
            / edges.triangles, fill_value = 0.))

    # solve once more with the final projected alpha_b, so cb and growth match
    # the accepted reaction rates
    exp_b = np.exp(-np.abs(sid.Da * alpha_b / (1 + sid.G * edges.diams) * \
        edges.diams * edges.lens / edges.flow))
    exp_b = np.array(np.ma.fix_invalid(exp_b, fill_value = 0.)) # fix for 0 / 0
    qc = edges.flow * exp_b
    qc_matrix = np.abs(inc.incidence.T @ spr.diags(qc) @ inc.incidence)
    cb_matrix = cb_inc.multiply(qc_matrix)
    diag_old = cb_matrix.diagonal()
    cb_matrix += spr.diags(diag - diag_old)
    cb = solve_equation(cb_matrix, cb_b) # calculate concentration of B

    growth_matrix = np.abs((spr.diags(edges.flow) @ inc.incidence > 0))
    cb_growth = growth_matrix @ cb # choose upstream concentration o

    edges.alpha_b = alpha_b
    q_a = np.minimum(vols.vol_a, vol_a_dissolved)
    vols.vol_a = np.maximum(vols.vol_a - q_a, 0.)
    return cb


