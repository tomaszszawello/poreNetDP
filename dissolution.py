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
    cb_matrix.setdiag(diag)
    cb = solve_equation(cb_matrix, cb_b)
    return cb


def solve_dissolution_nr(sid: SimInputData, inc: Incidence, graph: Graph, \
    edges: Edges, vols: Volumes, cb_b) -> np.ndarray:
    """ Calculate B concentration with tracking of A volume.

    This function solves the advection-reaction equation for substance B
    concentration. We track the volume of substance A and when we need to
    dissolve more than is available, we rescale the local effective reaction
    rate constant. We recalculate substance B concentration with new reaction
    rates and check the dissolution again and rescale the reaction rates again.
    We iterate using the Newton - Raphson method until the reaction rate
    scaling vector (alpha) converges.

    Parameters
    -------
    sid : simInputData class object
        all config parameters of the simulation
        Da : float
        G : float
        it_alpha_th : float
        it_limit : int

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
        alpha_b : numpy ndarray (ne)

    vols : Volumes class object
        volumes of substances in network triangles and their properties
        vol_a : numpy ndarray (ntr)

    cb_b : scipy sparse csc matrix (nsq x 1)
        result vector for substance B concentration calculation

    Returns
    -------
    cb : numpy array (nsq)
        vector of substance B concentration in nodes

    Raises
    -------
    IterationError
        if iterating of reaction rate does not converge
        (more iterations than sid.it_limit)
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
    #alpha_b = np.ones(sid.ne) # vector scaling the reaction constants in edges
    alpha_b = 1 * ((vols.triangles @ vols.vol_a) > 0)
    # according to A availibility
    alpha_b_tr = np.ones(sid.ntr) # vector scaling the reaction constants in
    # triangles according to A availibility
    alpha_b_tr_prev = np.zeros(sid.ntr)
    it_alpha = 0
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
    f_alpha = vols.vol_a - vol_a_dissolved # check if overdissolved
    f_alpha_check = f_alpha < 0
    # iterate using N-R until alpha_b_tr is the same (up to certain threshold)
    # in consecutive iterations; alpha_b for each edge is a function of
    # alpha_b_tr for the triangles neighbouring the edge, so we use matrix N-R
    # df(alpha i-1) @ delta_alpha = f(alpha i-1)
    # alpha i = alpha i-1 + delta_alpha
    while np.linalg.norm(alpha_b_tr - alpha_b_tr_prev) > sid.it_alpha_th:
        alpha_b_tr_prev = alpha_b_tr.copy()
        df_alpha = -cb_growth * exp_b * sid.dt / (1 + sid.G * edges.diams) / edges.triangles
        df_alpha = np.array(np.ma.fix_invalid(df_alpha, fill_value = 0.)) # fix
        # for zero surface
        # we calculate delta_alpha only where overdissolved, that's why we use
        # f_alpha_check
        df_alpha_matrix = spr.diags(1 * f_alpha_check) @ vols.triangles.T @ \
            spr.diags(df_alpha) @ vols.triangles
        # we set rows without overdissolution to identity
        df_alpha_matrix += spr.diags(1 * (df_alpha_matrix.diagonal() == 0))
        delta_alpha = solve_equation(df_alpha_matrix, -f_alpha * f_alpha_check)
        # we clip the reaction rate to [0,1], as N-R sometimes overshoots and
        # we only want to slow down the reaction, not fasten
        alpha_b_tr = np.clip(alpha_b_tr + delta_alpha, 0, 1)
        alpha_b = np.array(np.ma.fix_invalid((vols.triangles @ (alpha_b_tr)) / edges.triangles, fill_value = 0.))
        # if alpha_b != identity, we recalculate B concentrations (which
        # change when we change alpha_b) and dissolved volumes and iterate
        # until alpha_b converges
        if np.sum(alpha_b) != sid.ne:
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
            f_alpha = vols.vol_a - vol_a_dissolved # check if overdissolved
            f_alpha_check += (f_alpha < 0)
            it_alpha += 1
        if it_alpha > sid.it_limit:
            raise ValueError("Iterating for dissolution did not converge")
    edges.alpha_b = alpha_b
    return cb
