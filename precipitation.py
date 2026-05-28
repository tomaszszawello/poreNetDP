""" Calculate substance C concentration (precipitation).

This module contains functions for solving the advection-reaction equation for
substance C concentration. It constructs a result vector for the matrix
equation (dependent on B concentration, so recalculated each iteration)
and the matrix with coefficients corresponding to aforementioned equation.
Function solve_equation from module utils is used to solve the equation for
C concentration. If precipitation is off, then C concentration is assumed
to be zero.

Notable functions
-------
solve_precipitation(SimInputData, Incidence, Graph, Edges, np.ndarray) \
    -> np.ndarray
    calculate substance C concentration
"""

import numpy as np
import scipy.sparse as spr

from config import SimInputData
from network import Edges, Graph
from incidence import Incidence
from utils import solve_equation


def create_vector(sid: SimInputData, inc: Incidence, graph: Graph, \
    edges: Edges, cb: np.ndarray) -> spr.csc_matrix:
    """ Creates vector result for C concentration calculation.

    This function creates the result vector used to solve the equation for
    substance C concentration. For inlet nodes elements of the vector
    correspond explicitly to the concentration in nodes, for other they
    include part from the mixing condition and part from dissolution.

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

    cb : numpy ndarray (nsq)
        vector of substance B concentration

    Returns
    -------
    cc_b : scipy sparse csc matrix (nsq x 1)
        vector result for substance C concentration calculation
    """
    # find incidence for cb (only upstream flow matters)
    cb_inc = np.abs(inc.incidence.T @ (spr.diags(edges.flow) \
        @ inc.incidence > 0))
    # find vector with non-diagonal coefficients
    K_pref = (1 + sid.G * edges.diams) / (1 + sid.G * sid.K * edges.diams)
    qc = edges.flow / (K_pref * sid.K - 1) * (np.exp(-sid.Da / (1 + sid.G * \
        edges.diams) * edges.diams * edges.lens / np.abs(edges.flow)) - \
        np.exp(-sid.Da * sid.K / (1 + sid.G * sid.K * edges.diams) \
        * edges.diams * edges.lens / np.abs(edges.flow)))
    qc_matrix = np.abs(inc.incidence.T @ spr.diags(qc) @ inc.incidence)
    cb_matrix = cb_inc.multiply(qc_matrix)
    cb_matrix.setdiag(np.zeros(sid.nsq)) # set diagonal to zero
    cc_b = -cb_matrix @ cb
    cc_b = cc_b * (1 - graph.in_vec) + graph.in_vec * sid.cc_in
    return cc_b

def create_vector_nucleation(sid: SimInputData, inc: Incidence, graph: Graph, \
    edges: Edges, cb: np.ndarray) -> spr.csc_matrix:
    """ Creates vector result for C concentration calculation with nucleation
    """
    # find incidence for cb (only upstream flow matters)
    cb_inc = np.abs(inc.incidence.T @ (spr.diags(edges.flow) \
        @ inc.incidence > 0))
    # find vector with non-diagonal coefficients
    f_ratio = edges.ftrans / (1 - edges.ftrans)
    K_pref = f_ratio * (1 + sid.G * edges.diams) / (1 + sid.G * sid.K * edges.diams)
    qc = edges.flow / (K_pref * sid.K - 1) * (np.exp(-(1 - edges.ftrans) * sid.Da / (1 + sid.G * \
        edges.diams) * edges.diams * edges.lens / np.abs(edges.flow)) - \
        np.exp(-edges.ftrans * sid.Da * sid.K / (1 + sid.G * sid.K * edges.diams) \
        * edges.diams * edges.lens / np.abs(edges.flow)))

    qc_matrix = np.abs(inc.incidence.T @ spr.diags(qc) @ inc.incidence)
    cb_matrix = cb_inc.multiply(qc_matrix)
    cb_matrix.setdiag(np.zeros(sid.nsq)) # set diagonal to zero
    cc_b = -cb_matrix @ cb
    cc_b = cc_b * (1 - graph.in_vec) + graph.in_vec * sid.cc_in
    return cc_b

    
from volumes import Volumes

def create_vector_nr(sid: SimInputData, graph: Graph, inc, edges, cb) -> spr.csc_matrix:
    c_inc = 1 * (inc.incidence.T @ (spr.diags(edges.flow) \
        @ inc.incidence > 0) != 0)
    # find vector with non-diagonal coefficients
    qc = edges.flow * (1 + sid.G * sid.K * edges.diams) / (sid.K - 1) * (np.exp(-edges.alpha_b * sid.Da / (1 + sid.G * \
        edges.diams) * edges.diams * edges.lens / np.abs(edges.flow)) - \
        np.exp(-edges.alpha_b * sid.Da * sid.K / (1 + sid.G * sid.K * edges.diams) \
        * edges.diams * edges.lens / np.abs(edges.flow)))
    qc = np.array(np.ma.fix_invalid(qc, fill_value = 0))
    qc_matrix = np.abs(inc.incidence.T @ spr.diags(qc) @ inc.incidence)
    cb_matrix = c_inc.multiply(qc_matrix)
    #cb_matrix.setdiag(np.zeros(sid.nsq)) # set diagonal to zero
    diag_old = cb_matrix.diagonal()
    cb_matrix -= spr.diags(diag_old)
    cc_b = -cb_matrix @ cb
    cc_b = cc_b * (1 - graph.in_vec) + graph.in_vec * sid.cc_in
    cd_b = sid.cd_in * graph.in_vec
    
    q_cc = edges.flow * np.exp(-np.abs(edges.alpha_b * sid.Da * sid.K / (1 + sid.G * sid.K * edges.diams) \
        * edges.diams * edges.lens / edges.flow * sid.cd_in / sid.Kp))
    q_cc = np.array(np.ma.fix_invalid(q_cc, fill_value = 0))
    q_cc_matrix = np.abs(inc.incidence.T @ spr.diags(q_cc) @ inc.incidence)
    cc_matrix = c_inc.multiply(q_cc_matrix)
    # find diagonal coefficients (inlet flow for each node)
    diag = -np.abs(inc.incidence.T) @ np.abs(edges.flow) / 2
    diag = diag * (1 - graph.in_vec + graph.out_vec) + graph.in_vec
    diag += 1 * (diag == 0)
    # replace diagonal
    #cc_matrix.setdiag(diag)
    diag_old = cc_matrix.diagonal()
    cc_matrix += spr.diags(diag - diag_old)
    cc = solve_equation(cc_matrix, cc_b)

    # find vector with non-diagonal coefficients
    # q_cd = edges.flow * np.exp(-np.abs(edges.alpha_b * sid.Da * sid.K / (1 + sid.G * sid.K * edges.diams) \
    #     * edges.diams * edges.lens / edges.flow * sid.cd_in / sid.Kp))
    cb_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ np.abs(cb)
    q_cd = edges.flow * np.exp(-np.abs(edges.alpha_b * sid.Da * sid.K / (1 + sid.G * sid.K * edges.diams) \
        * edges.diams * edges.lens / edges.flow * sid.cd_in * cb_in / sid.Kp))
    q_cd = np.array(np.ma.fix_invalid(q_cd, fill_value = 0))
    q_cd_matrix = np.abs(inc.incidence.T @ spr.diags(q_cd) @ inc.incidence)
    cd_matrix = c_inc.multiply(q_cd_matrix)
    #cd_matrix.setdiag(diag)
    diag_old = cd_matrix.diagonal()
    cd_matrix += spr.diags(diag - diag_old)
    cd = solve_equation(cd_matrix, cd_b)
    #cd = sid.cd_in * (sid.cb_in - cb)
    return cc, cd

def solve_precipitation(sid: SimInputData, inc: Incidence, graph: Graph, \
    edges: Edges, cb) -> np.ndarray:
    """ Calculate C concentration.

    This function solves the advection-reaction equation for substance C
    concentration. We assume that we can always precipitate more. If
    precipitation is disabled in simulation, we only return vector of zeros.

    Parameters
    -------
    sid : simInputData class object
        all config parameters of the simulation

    inc : Incidence class object
        matrices of incidence

    graph : Graph class object
        network and all its properties

    edges : Edges class object
        all edges in network and their parameters

    cb : numpy array (nsq)
        vector of substance B concentration

    Returns
    -------
    cc : numpy array (nsq)
        vector of substance C concentration
    """
    if sid.include_precipitation:
        if sid.include_nucleation:
            return solve_nucleation(sid, inc, graph, edges, cb)
        else:
            return solve(sid, inc, graph, edges, cb)
    else:
        return np.zeros(sid.nsq)


def solve(sid: SimInputData, inc: Incidence, graph: Graph, edges: Edges, \
    cb: np.ndarray) -> np.ndarray:
    """ Calculate C concentration.

    This function solves the advection-reaction equation for substance C
    concentration. We assume precipitation is always possible.

    Parameters
    -------
    sid : simInputData class object
        all config parameters of the simulation

    inc : Incidence class object
        matrices of incidence

    graph : Graph class object
        network and all its properties

    edges : Edges class object
        all edges in network and their parameters

    cb : numpy array (nsq)
        vector of substance B concentration

    Returns
    -------
    cc : numpy array (nsq)
        vector of substance C concentration
    """
    # find incidence for cc (only upstream flow matters)
    cc_inc = np.abs(inc.incidence.T @ (spr.diags(edges.flow) \
        @ inc.incidence > 0))
    # find vector with non-diagonal coefficients
    qc = edges.flow * np.exp(-sid.Da * sid.K / (1 + sid.G * sid.K * \
        edges.diams) * edges.diams * edges.lens / np.abs(edges.flow))
    qc_matrix = np.abs(inc.incidence.T @ spr.diags(qc) @ inc.incidence)
    cc_matrix = cc_inc.multiply(qc_matrix)
    # find diagonal coefficients (inlet flow for each node)
    diag = -np.abs(inc.incidence.T) @ np.abs(edges.flow) / 2
    diag = diag * (1 - graph.in_vec + graph.out_vec) + graph.in_vec
    diag += 1 * (diag == 0)
    diag_old = cc_matrix.diagonal()
    cc_matrix += spr.diags(diag - diag_old)
    cc_b = create_vector(sid, inc, graph, edges, cb)
    cc = solve_equation(cc_matrix, cc_b)
    return cc

def solve_nucleation(sid: SimInputData, inc: Incidence, graph: Graph, edges: Edges, \
    cb: np.ndarray) -> np.ndarray:
    """ Calculate C concentration with nucleation / passivation
    This function solves the advection-reaction equation for substance C
    concentration. We assume precipitation is always possible.
    """
    # find incidence for cc (only upstream flow matters)
    cc_inc = np.abs(inc.incidence.T @ (spr.diags(edges.flow) \
        @ inc.incidence > 0))
    # find vector with non-diagonal coefficients
    qc = edges.flow * np.exp(-edges.ftrans * sid.Da * sid.K / (1 + sid.G * sid.K * \
        edges.diams) * edges.diams * edges.lens / np.abs(edges.flow))
    qc_matrix = np.abs(inc.incidence.T @ spr.diags(qc) @ inc.incidence)
    cc_matrix = cc_inc.multiply(qc_matrix)
    # find diagonal coefficients (inlet flow for each node)
    diag = -np.abs(inc.incidence.T) @ np.abs(edges.flow) / 2
    diag = diag * (1 - graph.in_vec + graph.out_vec) + graph.in_vec
    diag += 1 * (diag == 0)
    diag_old = cc_matrix.diagonal()
    cc_matrix += spr.diags(diag - diag_old)
    cc_b = create_vector_nucleation(sid, inc, graph, edges, cb)
    cc = solve_equation(cc_matrix, cc_b)
    return cc


def solve_precipitation_nr9_vxx(sid, inc, graph, edges, vols, cb, cc, cd,
                        tol: float = 1e-2,
                        max_iter: int = 100,
                        red: float = 0.5,
                        lam_min: float = 1e-10):
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    if np.sum(edges.alpha_b == 0) > 0:
        print(np.sum(edges.alpha_b == 0))
        print(np.sum(edges.alpha_b < 0))
    # ------------------------------------------------------------------
    # 0.  Handy aliases & basic sparse helpers
    E, N = inc.incidence.shape
    Inc  = inc.incidence              # ensure CSR
    abs_Q = np.abs(edges.flow)                # |Q_e|

    mask_flow  = abs_Q > 0           # flowing edges
    mask_zero  = ~mask_flow          # q == 0

    # Upstream‑selector matrix  U  (|E|×|N|, CSR, entries 0/1)
    U = 1 * (spr.diags(edges.flow) @ Inc > 0)

    # Downstream‑selector matrix  D
    D = 1 * (spr.diags(edges.flow) @ Inc < 0)

    #Q_in = np.abs(inc.incidence.T) @ np.abs(edges.flow) / 2 * (1 - graph.in_vec + graph.out_vec) + graph.in_vec
    Q_in = D.T @ abs_Q
    Q_in = Q_in * (1 - graph.in_vec) + graph.in_vec


    #in_vec = np.concatenate((graph.in_vec, graph.in_vec))
    in_vec = np.copy(graph.in_vec)
    c_inc = 1 * (inc.incidence.T @ (spr.diags(edges.flow) \
            @ inc.incidence > 0) != 0)


    d  = edges.diams
    L  = edges.lens
    q  = np.abs(edges.flow)
    eps_q = 1e-12
    q_safe = np.where(q > eps_q, q, 1.0)   # any nonzero dummy
    mask_zero = q <= eps_q                              # magnitude only

    B_pref = edges.alpha_b * sid.Da * d / (1.0 + sid.G * d)
    k = sid.Da * sid.K * d / (sid.Kp * (1.0 + sid.G * sid.K * d))

    tau0 = 1e-12
    diff_cc = 0
    z = np.log(cd)

    # Edges considered "active" for advection
    eps_q = 1e-12
    active = q > eps_q

    # For each node: does it have any incident active edge?
    inc_abs = np.abs(inc.incidence)
    incident_active = (inc_abs.T @ active) > 0   # shape (N,)

    # "Internal floating" nodes = not inlet, not outlet, no active edges
    floating = (~incident_active)

    if np.any(floating):
        #print("Floating nodes:", np.where(floating)[0])
        # Treat them as pseudo-Dirichlet: fix cc, cd = something
        cc[floating] = 0.0       # or initial guess, or neighbor's value
        cd[floating] = 0.0
        in_vec[floating] = 1

    in_vec_conc = np.concatenate([in_vec, in_vec])

    def safe_exp(x):
        if np.max(x) > 700:
            print('large exp!')
        return np.exp(np.clip(x, -700.0, 700.0))

    for it in range(1, max_iter + 1):

        cb_in  = U @ cb
        cb_out = D @ cb
        cc_in  = U @ cc
        cd_in  = U @ cd


        B   = B_pref / q_safe
        eB  = safe_exp(-np.abs(B * L))

        C   = k * cd_in
        lam = C / q_safe
        g   = safe_exp(-np.abs(edges.alpha_c * lam * L))

        den = k * cd_in - B_pref

        A_e  = edges.alpha_b * sid.Da * d * cb_in / (1.0 + sid.G * d)

        tau  = abs(den) / (abs(k*cd_in) + abs(B_pref) + 1.0)
        w    = tau / (tau + tau0)           # 0 ≤ w ≤ 1 ,  tau0 ≈ 1e-6
        # generic part
        alpha = A_e / den
        g_gen = safe_exp(-np.abs(edges.alpha_c * lam * L))

        # --- new: mask for alpha_b = 0 edges ---
        mask_ab0 = (edges.alpha_b == 0)
        # for general edges (alpha_b != 0):
        alpha[~mask_ab0]  = A_e[~mask_ab0] / den[~mask_ab0]

        # for alpha_b == 0, force alpha = 0
        alpha[mask_ab0] = 0.0

        cc_gen = alpha * eB + (cc_in - alpha) * g_gen
        # but for alpha_b == 0, override with cc_in * g_gen to avoid any lingering numeric issues:
        cc_gen[mask_ab0] = cc_in[mask_ab0] * g_gen[mask_ab0]

        # resonant part
        g_res  = safe_exp(-np.abs(B * L))
        cc_res = (A_e/q_safe * L + cc_in) * g_res
        cc_res = np.ma.fix_invalid(cc_res, fill_value = 0)
        # blended value
        cc_out = w * cc_gen + (1-w) * cc_res
        cc_out[mask_zero]      = cc_in[mask_zero]
        
        dccout_dccu = w * g_gen + (1-w) * g_res
        alpha_prime = -A_e * k / den**2
        g_prime     = -edges.alpha_c * (k / q_safe) * L * g
        g_prime = np.ma.fix_invalid(g_prime, fill_value = 0)
        dccout_dcd  = w * (alpha_prime * eB - alpha_prime * g + (cc_in - alpha) * g_prime)
        dccout_dcd[mask_ab0] = cc_in[mask_ab0] * g_prime[mask_ab0]

        cd_out = cd_in - ((cb_in - cb_out) + (cc_in - cc_out))
        cd_out[mask_zero]      = cd_in[mask_zero]
        dcdout_dccu = -1.0 + dccout_dccu
        dcdout_dcd  =  1.0 + dccout_dcd      # *** fixed sign ***


        dccout_dccu[mask_zero] = 1.0
        dccout_dcd[mask_zero]  = 0.0
        dcdout_dccu[mask_zero] = 0.0  # since cd_out = cd_in - (cc_in - cc_out), but cc_out = cc_in
        dcdout_dcd[mask_zero]  = 1.0

        delta_cb = cb_in - cb_out
        over = cd_out < 0.0
        if np.any(over):
            cd_out[over] = 0.0

            delta_cc_new = cd_in[over] - delta_cb[over]
            cc_out[over] = cc_in[over] - delta_cc_new  # = cc_in - cd_in + delta_cb

            # Derivative model in capped regime (capacity-limited)
            # cc_out = cc_in - cd_in + delta_cb
            dccout_dccu[over] = 1.0   # ∂cc_out/∂cc_in = 1
            dccout_dcd[over]  = -1.0  # ∂cc_out/∂cd_in = -1

            # cd_out is clamped to 0; no dependence on upstream concentrations
            dcdout_dccu[over] = 0.0
            dcdout_dcd[over]  = 0.0


        # ---- 4. residual vector --------------------------------------
        F_cc = cc * Q_in - (D.T @ (abs_Q * cc_out))
        F_cd = cd * Q_in - (D.T @ (abs_Q * cd_out))
        F_cc *= (1 - in_vec)
        F_cd *= (1 - in_vec)
        F = np.concatenate((F_cc, F_cd))

        # ---- 5. Jacobian blocks --------------------------------------
        Dg         = spr.diags(abs_Q * dccout_dccu)
        Ddcc_dcd   = spr.diags(abs_Q * dccout_dcd)
        Ddcd_dcc   = spr.diags(abs_Q * dcdout_dccu)
        Ddcd_dcd   = spr.diags(abs_Q * dcdout_dcd)


        J_cc_cc_0 = D.T @ Dg       @ U
        J_cc_cd_0 = D.T @ Ddcc_dcd @ U
        J_cd_cc_0 = D.T @ Ddcd_dcc @ U
        J_cd_cd_0 = D.T @ Ddcd_dcd @ U
 
        J_cc_cc = spr.diags(Q_in) - J_cc_cc_0
        J_cc_cd =           - J_cc_cd_0
        J_cd_cc =           - J_cd_cc_0
        J_cd_cd = spr.diags(Q_in) - J_cd_cd_0


        J = spr.vstack((spr.hstack((J_cc_cc, J_cc_cd)),
                        spr.hstack((J_cd_cc, J_cd_cd))))

        J = spr.diags(1 - in_vec_conc) @ J + spr.diags(in_vec_conc)
        J_diag = J.diagonal()
        J += spr.diags(1 * (J_diag == 0))
        F *= 1 * (J_diag != 0)
        #J = spr.diags(1 - 1 * (F == 0)) @ J + spr.diags(1 * (F == 0))

        # ---- 6. Newton step ------------------------------------------
        delta = solve_equation(J, -F)
        g     = J.T @ F

        if np.dot(g, delta) >= 0:
            # Levenberg
            mu = 1e-4 * spla.norm(J, np.inf)
            delta = solve_equation(J.T @ J + mu*spr.diags(np.ones(2 * sid.nsq)), -g)
            if np.dot(g, delta) >= 0:
                # Gradient
                delta = -g

        phi0 = 0.5 * np.dot(F, F)
        dphi0 = np.dot(g, delta)

        delta_cc = delta[:N]
        delta_cd = delta[N:]

        # tolerance for considering "at the bound"
        bound_tol = 1e-10

        # active at lower bound for cc: cc == 0 and step wants to go more negative
        active_cc_low  = (cc <= 0.0 + bound_tol)      & (delta_cc < 0.0)
        # active at upper bound for cc
        active_cc_high = (cc >= sid.cb_in - bound_tol) & (delta_cc > 0.0)

        # same for cd
        active_cd_low  = (cd <= 0.0 + bound_tol)      & (delta_cd < 0.0)
        active_cd_high = (cd >= sid.cd_in - bound_tol) & (delta_cd > 0.0)

        # freeze those components
        delta_cc[active_cc_low | active_cc_high] = 0.0
        delta_cd[active_cd_low | active_cd_high] = 0.0

        delta[:N]  = delta_cc
        delta[N:]  = delta_cd

        # Armijo back-tracking
        lams = 1.0
        rho = 1e-4
        #phi0 = 0.5*np.dot(F, F)
        #dphi0 = np.dot(g, delta)

        while lams >= lam_min:
            cc_trial = cc + lams * delta[:N]
            cd_trial = cd + lams * delta[N:]

            # cc_trial = np.clip(cc_trial, 0.0, sid.cb_in)
            # cd_trial = np.clip(cd_trial, 0.0, sid.cd_in)
            # residual at trial point (quick re‑eval) ------------------
            cc_in_t = U @ cc_trial
            cd_in_t = U @ cd_trial
            den_t   = k * cd_in_t - B_pref
            tau     = abs(den_t) / (abs(k*cd_in_t) + abs(B_pref) + 1.0)
            w       = tau / (tau + tau0)
            lam_t   = (k * cd_in_t) / q_safe

            g_t     = safe_exp(-np.abs(edges.alpha_c * lam_t * L))
            alpha_t = np.zeros_like(A_e)
            alpha_t[~mask_ab0]  = A_e[~mask_ab0] / den_t[~mask_ab0]
            alpha_t[mask_ab0]   = 0.0

            g_gen_t  = safe_exp(-np.abs(edges.alpha_c * lam_t * L))
            cc_gen_t = alpha_t * eB + (cc_in_t - alpha_t) * g_gen_t
            cc_gen_t[mask_ab0] = cc_in_t[mask_ab0] * g_gen_t[mask_ab0]

            g_res_t  = safe_exp(-np.abs(B * L))
            cc_res_t = (A_e/q_safe * L + cc_in_t) * g_res_t
            cc_res_t = np.ma.fix_invalid(cc_res_t, fill_value = 0)

            cc_out_t = w * cc_gen_t + (1-w) * cc_res_t
            cc_out_t[mask_zero]      = cc_in_t[mask_zero]

            cd_out_t = cd_in_t - ((cb_in - cb_out) + (cc_in_t - cc_out_t))
            cd_out_t[mask_zero] = cd_in_t[mask_zero]

            # >>> STOICH CAP ALSO HERE <<<
            delta_cb_t = cb_in - cb_out              # cb is fixed, so same as base
            over_t     = cd_out_t < 0.0
            if np.any(over_t):
                cd_out_t[over_t] = 0.0
                delta_cc_new_t   = cd_in_t[over_t] - delta_cb_t[over_t]
                cc_out_t[over_t] = cc_in_t[over_t] - delta_cc_new_t
            # <<< END STOICH CAP >>>

            F_cc_t = cc_trial * Q_in - (D.T @ (abs_Q * cc_out_t))
            F_cd_t = cd_trial * Q_in - (D.T @ (abs_Q * cd_out_t))
            F_cc_t *= (1 - in_vec)
            F_cd_t *= (1 - in_vec)
            F_t = np.concatenate((F_cc_t, F_cd_t))
            phi_t = 0.5 * np.dot(F_t, F_t)
            
            #print(np.linalg.norm(F_t))
            #print(np.linalg.norm(F))
            if phi_t < phi0:           # MONOTONE-only condition
                break
            lams *= red

        # if lams < lam_min:
        #     print("Line search stagnation, taking small gradient step")
        #     grad = J.T @ F
        #     step = -grad
        #     step_norm = np.linalg.norm(step)
        #     if step_norm > 0:
        #         step *= (1e-3 / step_norm)
        #     cc += step[:N]
        #     cd += step[N:]
        #     cc = np.clip(cc, 0.0, sid.cb_in)
        #     cd = np.clip(cd, 0.0, sid.cd_in)
        #     # go to next Newton iteration
        #     continue

        # if lams < lam_min:
        #     raise RuntimeError("Line search failed even in steepest descent")
        print(f"λ={lams:.5f}   phi_t={phi_t:.3e}   sumF={F_t.sum():.3e}")
        rel_cc = F_cc_t / np.maximum(Q_in, 1e-12)
        rel_cd = F_cd_t / np.maximum(Q_in, 1e-12)
        print("max rel_cc", np.max(np.abs(rel_cc)))
        print("max rel_cd", np.max(np.abs(rel_cd)))

        # delta_cd   = delta[N:]                       # from linear solve
        # cd_target  = cd + lams * delta_cd
        # cd_low     = 0.25 * cd
        # cd_high    = 4.0  * cd
        # cd_next    = np.minimum(np.maximum(cd_target, cd_low), cd_high)
        # delta[N:]  = (cd_next - cd) / lams
        # delta_cc   = delta[:N]                       # from linear solve
        # cc_target  = cc + delta_cc
        # cc_low     = 0.25 * cc
        # cc_high    = 4.0  * cc
        # cc_next    = np.minimum(np.maximum(cc_target, cc_low), cc_high)
        # delta[:N]  = cc_next - cc
        
        cc_prev = cc.copy()
        cd_prev = cd.copy()
        diff_cc = np.linalg.norm(delta[:N]) #/ max(1.0, np.linalg.norm(cc))
        diff_cd = np.linalg.norm(delta[N:]) #/ max(1.0, np.linalg.norm(cd))
        # if diff_cc_prev - diff_cc < 1e-2:
        #     lams /= 2
        cc += lams * delta[:N]
        cd += lams * delta[N:]
        # cc = np.clip(cc, 0.0, sid.cb_in)
        # cd = np.clip(cd, 0.0, sid.cd_in)
        if phi_t < sid.it_alpha_th:
            print(f"Newton converged in {it} iterations (Δcc={diff_cc:.1e}, Δcd={diff_cd:.1e})")
            break
        # if np.linalg.norm(cc - cc_prev) < sid.it_alpha_th and np.linalg.norm(cd - cd_prev) < sid.it_alpha_th:
        #     print(f"Newton converged in {it} iterations (Δcc={diff_cc:.1e}, Δcd={diff_cd:.1e})")
        #     break
        if it == 50 or np.isnan(np.linalg.norm(F_t)):
            print("Newton: restarting iterations")
            cc, cd = create_vector_nr(sid, graph, inc, edges, cb)


        # if np.sum(cc < -0.05) > 0:
        #     print('cc < 0')
        #     raise ValueError
        #cc = np.clip(cc, 0, None)
        #cd = np.clip(cd, 0, None)
    else:
        raise RuntimeError("Newton did not converge within the iteration limit")

    # clip outputs

    cc = np.clip(cc, 0.0, sid.cb_in)
    cd = np.clip(cd, 0.0, sid.cd_in)
    
        
    return cc, cd

def solve_precipitation_nr(sid: SimInputData, inc: Incidence, graph: Graph, \
    edges: Edges, cb: np.ndarray, cc: np.ndarray, cd: np.ndarray, \
    tol: float | None = None, max_iter: int = 100, red: float = 0.5, \
    lam_min: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """ Calculate C and D concentrations after precipitation.
    """
    Inc = inc.incidence
    ne, nsq = Inc.shape
    tol = sid.it_alpha_th if tol is None else tol

    q_eps = 1e-12
    tau0 = 1e-12
    bound_tol = 1e-10

    flow = edges.flow
    abs_flow = np.abs(flow)
    flow_active = abs_flow > q_eps
    flow_zero = ~flow_active
    q_safe = np.where(flow_active, abs_flow, 1.)

    upstream = 1 * (spr.diags(flow) @ Inc > 0)
    downstream = 1 * (spr.diags(flow) @ Inc < 0)

    in_vec = graph.in_vec.copy()
    floating = (np.abs(Inc).T @ flow_active) == 0
    if np.any(floating):
        cc[floating] = 0.
        cd[floating] = 0.
        in_vec[floating] = 1

    in_vec_conc = np.concatenate([in_vec, in_vec])
    free_vec = 1 - in_vec
    free_vec_conc = 1 - in_vec_conc

    Q_in = downstream.T @ abs_flow
    Q_in = Q_in * free_vec + in_vec

    diam = edges.diams
    lens = edges.lens
    alpha_b0 = edges.alpha_b == 0

    B_pref = edges.alpha_b * sid.Da * diam / (1. + sid.G * diam)
    k = sid.Da * sid.K * diam / (sid.Kp * (1. + sid.G * sid.K * diam))
    B = B_pref / q_safe
    exp_b = np.exp(np.clip(-np.abs(B * lens), -700., 700.))

    cb_in = upstream @ cb
    cb_out = downstream @ cb
    dcb = cb_in - cb_out
    Ae = B_pref * cb_in

    def safe_exp(x):
        return np.exp(np.clip(x, -700., 700.))

    def edge_values(cc_vec, cd_vec, derivative = False):
        cc_in = upstream @ cc_vec
        cd_in = upstream @ cd_vec

        den = k * cd_in - B_pref
        lam = k * cd_in / q_safe
        exp_c = safe_exp(-np.abs(edges.alpha_c * lam * lens))

        tau = np.abs(den) / (np.abs(k * cd_in) + np.abs(B_pref) + 1.)
        w = tau / (tau + tau0)

        alpha = np.zeros(ne)
        alpha[~alpha_b0] = np.divide(Ae[~alpha_b0], den[~alpha_b0], \
            out = np.zeros(np.sum(~alpha_b0)), where = den[~alpha_b0] != 0)

        cc_gen = alpha * exp_b + (cc_in - alpha) * exp_c
        cc_gen[alpha_b0] = cc_in[alpha_b0] * exp_c[alpha_b0]

        cc_res = (Ae / q_safe * lens + cc_in) * exp_b
        cc_res = np.array(np.ma.fix_invalid(cc_res, fill_value = 0.))

        cc_out = w * cc_gen + (1 - w) * cc_res
        cc_out[flow_zero] = cc_in[flow_zero]

        cd_out = cd_in - (dcb + cc_in - cc_out)
        cd_out[flow_zero] = cd_in[flow_zero]

        if derivative:
            dcc_dcc = w * exp_c + (1 - w) * exp_b

            alpha_prime = np.zeros(ne)
            alpha_prime[~alpha_b0] = np.divide(-Ae[~alpha_b0] * k[~alpha_b0], \
                den[~alpha_b0] ** 2, out = np.zeros(np.sum(~alpha_b0)), \
                where = den[~alpha_b0] != 0)

            exp_c_prime = -edges.alpha_c * k / q_safe * lens * exp_c
            exp_c_prime = np.array(np.ma.fix_invalid(exp_c_prime, fill_value = 0.))

            dcc_dcd = w * (alpha_prime * exp_b - alpha_prime * exp_c \
                + (cc_in - alpha) * exp_c_prime)
            dcc_dcd[alpha_b0] = cc_in[alpha_b0] * exp_c_prime[alpha_b0]

            dcd_dcc = -1. + dcc_dcc
            dcd_dcd = 1. + dcc_dcd

            dcc_dcc[flow_zero] = 1.
            dcc_dcd[flow_zero] = 0.
            dcd_dcc[flow_zero] = 0.
            dcd_dcd[flow_zero] = 1.

        over = cd_out < 0.
        if np.any(over):
            cd_out[over] = 0.
            cc_out[over] = cc_in[over] - cd_in[over] + dcb[over]

            if derivative:
                dcc_dcc[over] = 1.
                dcc_dcd[over] = -1.
                dcd_dcc[over] = 0.
                dcd_dcd[over] = 0.

        if derivative:
            return cc_out, cd_out, dcc_dcc, dcc_dcd, dcd_dcc, dcd_dcd
        return cc_out, cd_out

    def residual(cc_vec, cd_vec):
        cc_out, cd_out = edge_values(cc_vec, cd_vec)

        F_cc = cc_vec * Q_in - downstream.T @ (abs_flow * cc_out)
        F_cd = cd_vec * Q_in - downstream.T @ (abs_flow * cd_out)
        F_cc *= free_vec
        F_cd *= free_vec

        return np.concatenate([F_cc, F_cd]), F_cc, F_cd

    def linear_system(cc_vec, cd_vec):
        cc_out, cd_out, dcc_dcc, dcc_dcd, dcd_dcc, dcd_dcd = \
            edge_values(cc_vec, cd_vec, derivative = True)

        F_cc = cc_vec * Q_in - downstream.T @ (abs_flow * cc_out)
        F_cd = cd_vec * Q_in - downstream.T @ (abs_flow * cd_out)
        F_cc *= free_vec
        F_cd *= free_vec
        F = np.concatenate([F_cc, F_cd])

        Dcc_dcc = spr.diags(abs_flow * dcc_dcc)
        Dcc_dcd = spr.diags(abs_flow * dcc_dcd)
        Dcd_dcc = spr.diags(abs_flow * dcd_dcc)
        Dcd_dcd = spr.diags(abs_flow * dcd_dcd)

        J_cc_cc = spr.diags(Q_in) - downstream.T @ Dcc_dcc @ upstream
        J_cc_cd =                 - downstream.T @ Dcc_dcd @ upstream
        J_cd_cc =                 - downstream.T @ Dcd_dcc @ upstream
        J_cd_cd = spr.diags(Q_in) - downstream.T @ Dcd_dcd @ upstream

        J = spr.vstack([spr.hstack([J_cc_cc, J_cc_cd]), \
                        spr.hstack([J_cd_cc, J_cd_cd])])

        J = spr.diags(free_vec_conc) @ J + spr.diags(in_vec_conc)
        J_zero = J.diagonal() == 0
        if np.any(J_zero):
            J += spr.diags(1 * J_zero)
            F[J_zero] = 0.

        return F, F_cc, F_cd, J

    for it in range(1, max_iter + 1):
        F, F_cc, F_cd, J = linear_system(cc, cd)
        grad = J.T @ F
        delta = solve_equation(J, -F)

        if np.dot(grad, delta) >= 0:
            mu = 1e-4 * spla.norm(J, np.inf)
            delta = solve_equation(J.T @ J + mu * spr.eye(2 * nsq), -grad)
            if np.dot(grad, delta) >= 0:
                delta = -grad

        delta_cc = delta[:nsq]
        delta_cd = delta[nsq:]

        delta_cc[((cc <= bound_tol) & (delta_cc < 0.)) \
            | ((cc >= sid.cb_in - bound_tol) & (delta_cc > 0.))] = 0.
        delta_cd[((cd <= bound_tol) & (delta_cd < 0.)) \
            | ((cd >= sid.cd_in - bound_tol) & (delta_cd > 0.))] = 0.

        delta[:nsq] = delta_cc
        delta[nsq:] = delta_cd

        phi0 = 0.5 * np.dot(F, F)
        line_search = False
        lam = 1.

        while lam >= lam_min:
            cc_trial = cc + lam * delta[:nsq]
            cd_trial = cd + lam * delta[nsq:]
            F_trial, F_cc_trial, F_cd_trial = residual(cc_trial, cd_trial)
            phi_trial = 0.5 * np.dot(F_trial, F_trial)

            if phi_trial < phi0:
                line_search = True
                break
            lam *= red

        if not line_search or np.isnan(phi_trial):
            print('Newton: restarting iterations')
            cc, cd = create_vector_nr(sid, graph, inc, edges, cb)
            continue

        cc = cc_trial
        cd = cd_trial

        diff_cc = np.linalg.norm(lam * delta[:nsq])
        diff_cd = np.linalg.norm(lam * delta[nsq:])

        if phi_trial < tol:
            print(f'Newton converged in {it} iterations (Δcc={diff_cc:.1e}, Δcd={diff_cd:.1e})')
            break

        if it == 50:
            print('Newton: restarting iterations')
            cc, cd = create_vector_nr(sid, graph, inc, edges, cb)
    else:
        raise RuntimeError('Newton did not converge within the iteration limit')

    cc = np.clip(cc, 0., sid.cb_in)
    cd = np.clip(cd, 0., sid.cd_in)

    return cc, cd
