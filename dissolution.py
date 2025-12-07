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
    alpha_b = (vols.triangles @ (1 * (vols.vol_a > 0))) / edges.triangles
    alpha_b = np.ma.fix_invalid(alpha_b, fill_value = 0)
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
    edges.alpha_b = alpha_b
    print(np.min(cb), np.max(cb))
    return cb
    # growth_matrix = np.abs((spr.diags(edges.flow) @ inc.incidence > 0))
    # cb_growth = growth_matrix @ cb # choose upstream concentration of B for the
    # # calculation of growth
    # growth = cb_growth * np.abs(edges.flow) / sid.Da * (1 - exp_b) * sid.dt
    # growth = np.array(np.ma.fix_invalid(growth, fill_value = 0.)) # fix for
    # # zero surface
    # growth0 = growth.copy()
    # vol_a_dissolved = vols.triangles.T @ (growth / edges.triangles)
    # f_alpha = vols.vol_a - vol_a_dissolved # check if overdissolved
    # f_alpha_check = f_alpha < 0
    # f_alpha0 = f_alpha.copy()
    # print(vols.vol_a)
    # print(vol_a_dissolved)
    # # iterate using N-R until alpha_b_tr is the same (up to certain threshold)
    # # in consecutive iterations; alpha_b for each edge is a function of
    # # alpha_b_tr for the triangles neighbouring the edge, so we use matrix N-R
    # # df(alpha i-1) @ delta_alpha = f(alpha i-1)
    # # alpha i = alpha i-1 + delta_alpha
    # while np.linalg.norm(alpha_b_tr - alpha_b_tr_prev) > sid.it_alpha_th:
    #     alpha_b_tr_prev = alpha_b_tr.copy()
    #     df_alpha = -cb_growth * exp_b * sid.dt * edges.diams * edges.lens / (1 + sid.G * edges.diams) / edges.triangles * 100
    #     df_alpha = np.array(np.ma.fix_invalid(df_alpha, fill_value = 0.)) # fix
    #     # for zero surface
    #     # we calculate delta_alpha only where overdissolved, that's why we use
    #     # f_alpha_check
    #     df_alpha_matrix = spr.diags(1 * f_alpha_check) @ vols.triangles.T @ \
    #         spr.diags(df_alpha) @ vols.triangles
    #     # we set rows without overdissolution to identity
    #     df_alpha_matrix += spr.diags(1 * (df_alpha_matrix.diagonal() == 0))
    #     delta_alpha = solve_equation(df_alpha_matrix, -f_alpha * f_alpha_check)
    #     # we clip the reaction rate to [0,1], as N-R sometimes overshoots and
    #     # we only want to slow down the reaction, not fasten
    #     alpha_b_tr = np.clip(alpha_b_tr + delta_alpha, 0, 1)
    #     alpha_b = np.array(np.ma.fix_invalid((vols.triangles @ (alpha_b_tr)) / edges.triangles, fill_value = 0.))
    #     print(delta_alpha)
    #     # print(alpha_b_tr)
    #     print(alpha_b)
    #     #print(df_alpha_matrix[-6])
    #     #print(df_alpha)
    #     # if alpha_b != identity, we recalculate B concentrations (which
    #     # change when we change alpha_b) and dissolved volumes and iterate
    #     # until alpha_b converges
    #     if np.sum(alpha_b) != sid.ne:
    #         exp_b = np.exp(-np.abs(sid.Da * alpha_b / (1 + sid.G * edges.diams) * \
    #             edges.diams * edges.lens / edges.flow))
    #         exp_b = np.array(np.ma.fix_invalid(exp_b, fill_value = 0.)) # fix for 0 / 0
    #         # if there is available volume, we include reduction of cb in pore
    #         # (exp_b != 0), if not, then q_in cb_in = q_out cb_out (exp_b = 0)
    #         qc = edges.flow * exp_b
    #         qc_matrix = np.abs(inc.incidence.T @ spr.diags(qc) @ inc.incidence)
    #         cb_matrix = cb_inc.multiply(qc_matrix)
    #         diag_old = cb_matrix.diagonal()
    #         cb_matrix += spr.diags(diag - diag_old)
    #         cb = solve_equation(cb_matrix, cb_b) # calculate concentration of B
    #         growth_matrix = np.abs((spr.diags(edges.flow) @ inc.incidence > 0))
    #         cb_growth = growth_matrix @ cb # choose upstream concentration of B for the
    #         # calculation of growth
    #         growth = cb_growth * np.abs(edges.flow) / sid.Da * (1 - exp_b) * sid.dt
    #         growth = np.array(np.ma.fix_invalid(growth, fill_value = 0.)) # fix for
    #         # zero surface
    #         vol_a_dissolved = vols.triangles.T @ (growth / edges.triangles)
    #         #f_alpha = vols.vol_a - vol_a_dissolved # check if overdissolved
    #         vol_change = vols.triangles.T @ (growth - growth0)
    #         f_alpha = f_alpha0 - vol_change
    #         f_alpha_check |= (f_alpha < 0)
    #         print(vols.vol_a)
    #         print(f_alpha)
    #         it_alpha += 1
    #     if it_alpha > sid.it_limit:
    #         raise ValueError("Iterating for dissolution did not converge")
    # edges.alpha_b = alpha_b
    # return cb


def solve_dissolution_nr2(sid: SimInputData, inc: Incidence, graph: Graph, \
    edges: Edges, vols: Volumes, cb_b) -> np.ndarray:
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
    alpha_b = (vols.triangles @ (1 * (vols.vol_a > 0))) / edges.triangles
    # according to A availibility
    alpha_b_tr = np.ones(sid.ntr) # vector scaling the reaction constants in
    # triangles according to A availibility
    alpha_b_prev = np.zeros(sid.ne)
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
    vol_a_dissolved = np.array(np.ma.fix_invalid(vol_a_dissolved, fill_value = 0))
    vol_a_dissolved_real = np.min([vol_a_dissolved, vols.vol_a], axis = 0)
    change = growth / edges.triangles * (vols.triangles @ (vol_a_dissolved_real / vol_a_dissolved))
    change = np.array(np.ma.fix_invalid(change, fill_value = 0.)) # fix for

    f_alpha = change - growth # check if overdissolved
    f_alpha_check = f_alpha < 0

    # iterate using N-R until alpha_b_tr is the same (up to certain threshold)
    # in consecutive iterations; alpha_b for each edge is a function of
    # alpha_b_tr for the triangles neighbouring the edge, so we use matrix N-R
    # df(alpha i-1) @ delta_alpha = f(alpha i-1)
    # alpha i = alpha i-1 + delta_alpha
    while np.linalg.norm(alpha_b - alpha_b_prev) > sid.it_alpha_th:
        #alpha_b_tr_prev = alpha_b_tr.copy()
        print(f'Iteration difference: {np.linalg.norm(alpha_b - alpha_b_prev)}')
        # print(vols.vol_a)
        # print(f_alpha)
        #df_alpha = (exp_b - 1) * edges.diams * edges.lens / (1 + sid.G * edges.diams) / edges.triangles / np.abs(edges.flow) / (vols.triangles @ ((vols.triangles.T @ (exp_b -1)) ** 2 / vols.vol_a))
        
        df_alpha = -growth * cb_growth * exp_b * sid.dt * edges.diams * edges.lens * (vols.triangles @ (vol_a_dissolved_real / vol_a_dissolved ** 2)) / (1 + sid.G * edges.diams) / edges.triangles \
            + cb_growth * exp_b * sid.dt * edges.diams * edges.lens * (vols.triangles @ (vol_a_dissolved_real / vol_a_dissolved)) / (1 + sid.G * edges.diams) / edges.triangles
        
        df_alpha = np.array(np.ma.fix_invalid(df_alpha, fill_value = 0.)) # fix
        # for zero surface
        # we calculate delta_alpha only where overdissolved, that's why we use
        # f_alpha_check
        df_alpha_matrix = spr.diags(1 * f_alpha_check * df_alpha)
        # we set rows without overdissolution to identity
        df_alpha_matrix += spr.diags(1 * (df_alpha_matrix.diagonal() == 0))
        delta_alpha = solve_equation(df_alpha_matrix, -f_alpha * f_alpha_check)
        # we clip the reaction rate to [0,1], as N-R sometimes overshoots and
        # we only want to slow down the reaction, not fasten
        alpha_b_prev = alpha_b.copy()
        alpha_b = np.clip(alpha_b + delta_alpha, 0, 1)
        #alpha_b = np.array(np.ma.fix_invalid((vols.triangles @ (alpha_b_tr)) / edges.triangles, fill_value = 0.))
        # print(delta_alpha)
        # print(alpha_b)
        #print(df_alpha_matrix[-6])
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
            vol_a_dissolved = np.array(np.ma.fix_invalid(vol_a_dissolved, fill_value = 0))
            vol_a_dissolved_real = np.min([vol_a_dissolved, vols.vol_a], axis = 0)
            change = growth / edges.triangles * (vols.triangles @ (vol_a_dissolved_real / vol_a_dissolved))
            change = np.array(np.ma.fix_invalid(change, fill_value = 0.)) # fix for
            f_alpha = change - growth # check if overdissolved
            # print(vol_a_dissolved)
            # print(vol_a_dissolved_real)
            f_alpha_check |= (f_alpha < 0)
            it_alpha += 1
        if it_alpha > sid.it_limit:
            raise ValueError("Iterating for dissolution did not converge")
    edges.alpha_b = alpha_b
    return cb

def solve_dissolution_safe(
    sid: SimInputData,
    inc: Incidence,
    graph: Graph,
    edges: Edges,
    vols: Volumes,
    cb_b,
    max_alpha_iter: int = 5,
    tol_alpha: float = 1e-3,
):
    """
    Safer iterative dissolution scheme:

    - No Newton on alpha_b.
    - At each iteration:
        * compute cb and requested growth with current alpha_b
        * compute requested triangle dissolution
        * compute triangle safety factors f_t = min(1, vol_a / D0_t)
        * assemble edge safety s_e = min_{tri neighbors} f_t (via vols.triangles)
        * update alpha_b <- alpha_b * s_e
    - Iterates until alpha_b stabilizes or max_alpha_iter is reached.

    Returns
    -------
    cb : np.ndarray
        Node concentrations of B (final iteration).
    alpha_b : np.ndarray
        Per-edge reaction factor in [0, 1].
    growth_scaled : np.ndarray
        Per-edge dissolved volume actually used in this step (consistent with alpha_b).
    """

    # ---------- 0. Precompute some fixed stuff ----------

    E, N = inc.incidence.shape
    q = edges.flow
    abs_q = np.abs(q)

    # Upstream-incidence for B (same as your cb_inc)
    cb_inc = 1 * (inc.incidence.T @ (spr.diags(edges.flow) @ inc.incidence > 0) != 0)

    # Node "incoming flow" diagonal (same as your diag)

    # diag = -np.abs(inc.incidence.T) @ abs_q / 2.0
    # diag = diag * (1 - graph.in_vec + graph.out_vec) + graph.in_vec
    
    diag = -(1 * (spr.diags(edges.flow) @ inc.incidence < 0)).T @ abs_q
    diag = diag * (1 - graph.in_vec) + graph.in_vec
    diag += 1 * (diag == 0)  # avoid zeros on diagonal

    # Edge–triangle incidence, assumed shape (ne, ntr)
    T = vols.triangles  # could be sparse or dense
    ne = sid.ne
    ntr = sid.ntr

    # Number (or sum of weights) of triangles per edge. Avoid division by zero.
    edges_tri = np.maximum(edges.triangles, 1.0)

    # For mapping triangles → edges via min-reduction, we need (row, col) indices
    # Works for both sparse and dense arrays.
    tri_rows, tri_cols = T.nonzero()

    eps_q = 1e-12
    q_safe = np.where(abs_q > eps_q, q, 1.0)

    # ---------- 1. Initial guess for alpha_b ----------

    # 1 where there is some A volume, 0 otherwise, averaged per edge
    alpha_b = (T @ (1 * (vols.vol_a > 0))) / edges_tri
    alpha_b = np.array(np.ma.fix_invalid(alpha_b, fill_value=0.0))
    alpha_b = np.clip(alpha_b, 0.0, 1.0)

    # ---------- 2. Iteration on alpha_b ----------

    growth_scaled = np.zeros_like(alpha_b)

    for it_alpha in range(max_alpha_iter):
        alpha_prev = alpha_b.copy()

        # ---- 2.1 compute exp_b with current alpha_b ----
        exp_b = np.exp(
            -np.abs(
                sid.Da * alpha_b
                / (1.0 + sid.G * edges.diams)
                * edges.diams * edges.lens / q_safe
            )
        )
        exp_b = np.array(np.ma.fix_invalid(exp_b, fill_value=0.0))

        # ---- 2.2 solve transport for cb given exp_b ----
        qc = q * exp_b
        qc_matrix = np.abs(inc.incidence.T @ spr.diags(qc) @ inc.incidence)

        cb_matrix = cb_inc.multiply(qc_matrix)
        diag_old = cb_matrix.diagonal()
        cb_matrix += spr.diags(diag - diag_old)

        cb = solve_equation(cb_matrix, cb_b)

        # upstream B for growth (same as your cb_growth)
        growth_matrix = np.abs((spr.diags(edges.flow) @ inc.incidence > 0))
        cb_growth = growth_matrix @ cb

        # ---- 2.3 requested growth on each edge ----
        growth_req = cb_growth * abs_q / sid.Da * (1.0 - exp_b) * sid.dt
        growth_req = np.array(np.ma.fix_invalid(growth_req, fill_value=0.0))

        # ---- 2.4 requested triangle dissolution D0_t ----
        # split growth_req equally (or by weight) across triangles per edge
        growth_per_edge = growth_req / edges_tri
        # D0_t = sum over edges: T[e,t] * growth_per_edge[e]
        D0_t = T.T @ growth_per_edge

        # ---- 2.5 triangle safety factors f_t ----
        v_eps = 1e-16
        f_t = np.ones(ntr)
        mask_over = D0_t > v_eps
        # where D0_t > 0: f_t = min(1, vol_a / D0_t)
        f_t[mask_over] = np.minimum(
            1.0,
            vols.vol_a[mask_over] / (D0_t[mask_over] + v_eps)
        )

        # ---- 2.6 edge safety s_e via min over neighbor triangles ----
        # start with all ones; edges without triangles stay 1
        s_e = np.ones(ne)
        # for each nonzero (edge=row, triangle=col), enforce s_e[e] = min(s_e[e], f_t[t])
        np.minimum.at(s_e, tri_rows, f_t[tri_cols])
        s_e = np.clip(s_e, 0.0, 1.0)

        # ---- 2.7 update alpha_b and store scaled growth ----
        alpha_b = np.clip(alpha_b * s_e, 0.0, 1.0)
        growth_scaled = growth_req * s_e  # this is the actually-used dissolution per edge

        # ---- 2.8 check convergence in alpha_b ----
        diff_alpha = np.linalg.norm(alpha_b - alpha_prev, ord=np.inf)
        print(f"dissolution alpha_b iter {it_alpha}: diff={diff_alpha:.3e}")
        if diff_alpha < tol_alpha:
            break

    # Optional: you can recompute final D_t if you want to check
    # how close we are to saturating vol_a:
    growth_per_edge_final = growth_scaled / edges_tri
    vol_a_dissolved_real = T.T @ growth_per_edge_final
    vol_a_dissolved_real = np.minimum(vol_a_dissolved_real, vols.vol_a)

    # store alpha_b back into edges
    edges.alpha_b = alpha_b

    return cb

