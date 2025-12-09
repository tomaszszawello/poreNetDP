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
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import bicgstab

from config import SimInputData
from network import Edges, Graph
from incidence import Incidence
from utils import solve_equation
from volumes import Volumes

def create_vector_nr(sid: SimInputData, graph: Graph, inc, edges, cb) -> spr.csc_matrix:
    c_inc = 1 * (inc.incidence.T @ (spr.diags(edges.flow) \
        @ inc.incidence > 0) != 0)
    # find vector with non-diagonal coefficients
    qc = edges.flow / (sid.K - 1) * (np.exp(-edges.alpha_b * sid.Da / (1 + sid.G * \
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
        print("Floating nodes:", np.where(floating)[0])
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

def solve_precipitation_nr10_vxx(sid, inc, graph, edges, vols, cb, cc, cd,
                        tol: float = 1e-2,
                        max_iter: int = 100,
                        red: float = 0.5,
                        lam_min: float = 1e-10):

    if np.sum(edges.alpha_b == 0) > 0:
        print(np.sum(edges.alpha_b == 0))
        print(np.sum(edges.alpha_b < 0))

    # ------------------------------------------------------------------
    # 0.  Handy aliases & basic sparse helpers
    # ------------------------------------------------------------------
    E, N = inc.incidence.shape
    Inc  = inc.incidence              # ensure CSR
    abs_Q = np.abs(edges.flow)        # |Q_e|

    # Upstream-selector matrix  U  (|E|×|N|, CSR, entries 0/1)
    U = 1 * (spr.diags(edges.flow) @ Inc > 0)

    # Downstream-selector matrix  D
    D = 1 * (spr.diags(edges.flow) @ Inc < 0)

    # Node inflow
    Q_in = D.T @ abs_Q
    Q_in = Q_in * (1 - graph.in_vec) + graph.in_vec

    in_vec = np.copy(graph.in_vec)

    d  = edges.diams
    L  = edges.lens
    q  = np.abs(edges.flow)
    eps_q   = 1e-12
    q_safe  = np.where(q > eps_q, q, 1.0)
    mask_zero = q <= eps_q

    B_pref = edges.alpha_b * sid.Da * d / (1.0 + sid.G * d)
    k      = sid.Da * sid.K * d / (sid.Kp * (1.0 + sid.G * sid.K * d))

    tau0 = 1e-12

    # Edges considered "active" for advection
    active = q > eps_q

    # For each node: does it have any incident active edge?
    inc_abs = np.abs(inc.incidence)
    incident_active = (inc_abs.T @ active) > 0   # shape (N,)

    # Treat *any* node with no active edges as pseudo-Dirichlet
    floating = (~incident_active)
    if np.any(floating):
        print("Floating nodes:", np.where(floating)[0])
        cc[floating] = 0.0
        cd[floating] = 0.0
        in_vec[floating] = 1

    in_vec_conc = np.concatenate([in_vec, in_vec])

    # --- tolerances for "simple" (no C–D reaction) edges ---
    cd_edge_tol = 1e-8
    kcd_rel_tol = 1e-6

    def safe_exp(x):
        if np.max(x) > 700:
            print('large exp!')
        return np.exp(np.clip(x, -700.0, 700.0))

    # ------------------------------------------------------------------
    # Newton iterations
    # ------------------------------------------------------------------
    for it in range(1, max_iter + 1):

        cb_in  = U @ cb
        cb_out = D @ cb
        cc_in  = U @ cc
        cd_in  = U @ cd

        # ---- 1. Edge-wise analytic solution pieces -------------------
        B   = B_pref / q_safe
        eB  = safe_exp(-np.abs(B * L))

        C   = k * cd_in
        lam = C / q_safe

        den = k * cd_in - B_pref
        A_e = edges.alpha_b * sid.Da * d * cb_in / (1.0 + sid.G * d)

        tau = np.abs(den) / (np.abs(k * cd_in) + np.abs(B_pref) + 1.0)
        w   = tau / (tau + tau0)

        # mask for alpha_b = 0 edges
        mask_ab0 = (edges.alpha_b == 0)

        # --- detect edges where C–D reaction is negligible ("pure dissolution") ---
        small_cd   = np.abs(cd_in) < cd_edge_tol
        small_kcd  = np.abs(k * cd_in) < kcd_rel_tol * (np.abs(B_pref) + 1e-16)
        mask_simple = small_cd | small_kcd
        mask_full   = ~mask_simple

        # generic part (only meaningful on mask_full)
        g_gen = safe_exp(-np.abs(edges.alpha_c * lam * L))
        alpha = np.zeros_like(A_e)
        alpha[mask_full & (~mask_ab0)] = A_e[mask_full & (~mask_ab0)] / den[mask_full & (~mask_ab0)]

        cc_gen = alpha * eB + (cc_in - alpha) * g_gen
        cc_gen[mask_ab0] = cc_in[mask_ab0] * g_gen[mask_ab0]

        # resonant part
        g_res  = safe_exp(-np.abs(B * L))
        cc_res = (A_e/q_safe * L + cc_in) * g_res
        cc_res = np.ma.fix_invalid(cc_res, fill_value=0)

        # blended value (initially for all edges)
        cc_out = w * cc_gen + (1.0 - w) * cc_res

        # ---- pure-dissolution override on "simple" edges -------------
        # Physical mass balance with no C–D reaction:
        #   C_out = C_in + (B_in - B_out)
        #   D_out = D_in
        cc_out_simple = cc_in + (cb_in - cb_out)
        cc_out[mask_simple] = cc_out_simple[mask_simple]

        # q = 0 edges: just copy upstream C
        cc_out[mask_zero] = cc_in[mask_zero]

        # derivatives wrt cc_in, cd_in
        dccout_dccu = np.zeros_like(cc_in)
        dccout_dcd  = np.zeros_like(cc_in)

        # full C–D + dissolution edges
        dccout_dccu[mask_full] = w[mask_full] * g_gen[mask_full] + (1.0 - w[mask_full]) * g_res[mask_full]

        g    = safe_exp(-np.abs(edges.alpha_c * lam * L))
        alpha_prime = np.zeros_like(A_e)
        alpha_prime[mask_full & (~mask_ab0)] = -A_e[mask_full & (~mask_ab0)] * k[mask_full & (~mask_ab0)] / den[mask_full & (~mask_ab0)]**2

        g_prime = -edges.alpha_c * (k / q_safe) * L * g
        g_prime = np.ma.fix_invalid(g_prime, fill_value=0)

        dccout_dcd[mask_full] = w[mask_full] * (alpha_prime[mask_full] * eB[mask_full]
                                                - alpha_prime[mask_full] * g[mask_full]
                                                + (cc_in[mask_full] - alpha[mask_full]) * g_prime[mask_full])
        dccout_dcd[mask_full & mask_ab0] = cc_in[mask_full & mask_ab0] * g_prime[mask_full & mask_ab0]

        # pure dissolution: cc_out = cc_in + (cb_in - cb_out)
        dccout_dccu[mask_simple] = 1.0
        dccout_dcd[mask_simple]  = 0.0

        # q=0 edges again
        dccout_dccu[mask_zero] = 1.0
        dccout_dcd[mask_zero]  = 0.0

        # D outflow
        cd_out = cd_in - ((cb_in - cb_out) + (cc_in - cc_out))
        cd_out[mask_simple] = cd_in[mask_simple]       # pure dissolution: D unchanged
        cd_out[mask_zero]   = cd_in[mask_zero]

        dcdout_dccu = np.zeros_like(cc_in)
        dcdout_dcd  = np.zeros_like(cc_in)

        # full edges:
        dcdout_dccu[mask_full] = -1.0 + dccout_dccu[mask_full]
        dcdout_dcd[mask_full]  =  1.0 + dccout_dcd[mask_full]

        # pure dissolution edges:
        #   cd_out = cd_in → dcdout_dccu = 0, dcdout_dcd = 1
        dcdout_dccu[mask_simple] = 0.0
        dcdout_dcd[mask_simple]  = 1.0

        # q=0 edges
        dcdout_dccu[mask_zero] = 0.0
        dcdout_dcd[mask_zero]  = 1.0

        delta_cb = cb_in - cb_out
        over = cd_out < 0.0
        if np.any(over):
            # enforce cd_out >= 0 and consistent cc_out
            cd_out[over] = 0.0

            delta_cc_new = cd_in[over] - delta_cb[over]
            cc_out[over] = cc_in[over] - delta_cc_new

            # Derivative model in capped regime
            dccout_dccu[over] = 1.0
            dccout_dcd[over]  = 0.0
            dcdout_dccu[over] = 0.0
            dcdout_dcd[over]  = 0.0

        # ---- 2. Residual vector --------------------------------------
        F_cc = cc * Q_in - (D.T @ (abs_Q * cc_out))
        F_cd = cd * Q_in - (D.T @ (abs_Q * cd_out))
        F_cc *= (1 - in_vec)
        F_cd *= (1 - in_vec)
        F = np.concatenate((F_cc, F_cd))

        # ---- 3. Jacobian blocks --------------------------------------
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

        # protect against accidental zero rows
        J_diag = J.diagonal()
        J += spr.diags(1 * (J_diag == 0))
        F *= 1 * (J_diag != 0)

        # ---- 5a. Treat "dead" D nodes as Dirichlet cd = 0 ------------
        cd_tol = 1e-4
        dead_cd_nodes = (cd <= cd_tol)    # shape (N,)

        if np.any(dead_cd_nodes):
            idx_cd_global = np.arange(N, 2 * N)[dead_cd_nodes]

            F[idx_cd_global] = cd[dead_cd_nodes]

            row_scale = np.ones(2 * N)
            row_scale[idx_cd_global] = 0.0
            J = spr.diags(row_scale) @ J

            diag_fix = np.zeros(2 * N)
            diag_fix[idx_cd_global] = 1.0
            J += spr.diags(diag_fix)

        # ---- 4. Newton step ------------------------------------------
        delta = solve_equation(J, -F)
        g     = J.T @ F

        # Levenberg / gradient fallback
        if np.dot(g, delta) >= 0:
            mu = 1e-4 * spla.norm(J, np.inf)
            delta = solve_equation(J.T @ J + mu*spr.diags(np.ones(2 * sid.nsq)), -g)
            if np.dot(g, delta) >= 0:
                delta = -g

        phi0   = 0.5 * np.dot(F, F)
        delta_cc = delta[:N]
        delta_cd = delta[N:]

        # ---- 4a. Step limiting ---------------------------------
        max_rel_step     = 0.5       # at most 50% of current value
        max_abs_step_cc  = 0.25 * sid.cb_in
        max_abs_step_cd  = 0.25 * sid.cd_in

        den_cc_step = np.maximum(np.abs(cc), 1e-8)
        den_cd_step = np.maximum(np.abs(cd), 1e-8)

        delta_cc = np.clip(delta_cc,
                           -max_rel_step * den_cc_step,
                           +max_rel_step * den_cc_step)
        delta_cd = np.clip(delta_cd,
                           -max_rel_step * den_cd_step,
                           +max_rel_step * den_cd_step)

        delta_cc = np.clip(delta_cc, -max_abs_step_cc, +max_abs_step_cc)
        delta_cd = np.clip(delta_cd, -max_abs_step_cd, +max_abs_step_cd)

        # ---- 4b. Freeze components at hard bounds --------------------
        bound_tol = 1e-10
        active_cc_low  = (cc <= 0.0 + bound_tol)        & (delta_cc < 0.0)
        active_cc_high = (cc >= sid.cb_in - bound_tol)  & (delta_cc > 0.0)
        active_cd_low  = (cd <= 0.0 + bound_tol)        & (delta_cd < 0.0)
        active_cd_high = (cd >= sid.cd_in - bound_tol)  & (delta_cd > 0.0)

        delta_cc[active_cc_low | active_cc_high] = 0.0
        delta_cd[active_cd_low | active_cd_high] = 0.0

        delta[:N] = delta_cc
        delta[N:] = delta_cd

        # ---- 5. Armijo back-tracking --------------------------------
        lams = 1.0
        rho  = 1e-4

        while lams >= lam_min:
            cc_trial = cc + lams * delta_cc
            cd_trial = cd + lams * delta_cd

            # residual at trial point (quick re-eval)
            cc_in_t = U @ cc_trial
            cd_in_t = U @ cd_trial

            den_t = k * cd_in_t - B_pref
            tau_t = np.abs(den_t) / (np.abs(k * cd_in_t) + np.abs(B_pref) + 1.0)
            w_t   = tau_t / (tau_t + tau0)
            lam_t = (k * cd_in_t) / q_safe

            g_t      = safe_exp(-np.abs(edges.alpha_c * lam_t * L))
            alpha_t  = np.zeros_like(A_e)
            alpha_t[~mask_ab0] = A_e[~mask_ab0] / den_t[~mask_ab0]
            alpha_t[mask_ab0]  = 0.0

            g_gen_t  = safe_exp(-np.abs(edges.alpha_c * lam_t * L))
            cc_gen_t = alpha_t * eB + (cc_in_t - alpha_t) * g_gen_t
            cc_gen_t[mask_ab0] = cc_in_t[mask_ab0] * g_gen_t[mask_ab0]

            g_res_t  = safe_exp(-np.abs(B * L))
            cc_res_t = (A_e/q_safe * L + cc_in_t) * g_res_t
            cc_res_t = np.ma.fix_invalid(cc_res_t, fill_value=0)

            cc_out_t = w_t * cc_gen_t + (1.0 - w_t) * cc_res_t

            # --- pure dissolution mask at trial ---
            small_cd_t  = np.abs(cd_in_t) < cd_edge_tol
            small_kcd_t = np.abs(k * cd_in_t) < kcd_rel_tol * (np.abs(B_pref) + 1e-16)
            mask_simple_t = small_cd_t | small_kcd_t

            cc_out_simple_t = cc_in_t + (cb_in - cb_out)
            cc_out_t[mask_simple_t] = cc_out_simple_t[mask_simple_t]

            # q = 0 edges
            cc_out_t[mask_zero] = cc_in_t[mask_zero]

            cd_out_t = cd_in_t - ((cb_in - cb_out) + (cc_in_t - cc_out_t))
            cd_out_t[mask_simple_t] = cd_in_t[mask_simple_t]
            cd_out_t[mask_zero]     = cd_in_t[mask_zero]

            F_cc_t = cc_trial * Q_in - (D.T @ (abs_Q * cc_out_t))
            F_cd_t = cd_trial * Q_in - (D.T @ (abs_Q * cd_out_t))
            F_cc_t *= (1 - in_vec)
            F_cd_t *= (1 - in_vec)
            F_t     = np.concatenate((F_cc_t, F_cd_t))

            # apply same "dead D" residual logic at trial point
            dead_cd_nodes_t = (cd_trial <= cd_tol)
            if np.any(dead_cd_nodes_t):
                idx_cd_global_t = np.arange(N, 2 * N)[dead_cd_nodes_t]
                F_t[idx_cd_global_t] = cd_trial[dead_cd_nodes_t]

            phi_t   = 0.5 * np.dot(F_t, F_t)

            if phi_t < phi0:   # monotone condition
                break
            lams *= red

        print(f"λ={lams:.5f}   phi_t={phi_t:.3e}   sumF={F_t.sum():.3e}")
        # after you've defined dead_cd_nodes and Q_tol:

        Q_tol = 1e-3  # as you tried

        mask_rel_cc = (Q_in > Q_tol)         # active C-equations
        mask_rel_cd = (Q_in > Q_tol) & (~dead_cd_nodes)  # active D-equations

        rel_cc = np.zeros_like(F_cc_t)
        rel_cd = np.zeros_like(F_cd_t)

        rel_cc[mask_rel_cc] = F_cc_t[mask_rel_cc] / Q_in[mask_rel_cc]
        rel_cd[mask_rel_cd] = F_cd_t[mask_rel_cd] / Q_in[mask_rel_cd]

        max_rel_cc = np.max(np.abs(rel_cc)) if np.any(mask_rel_cc) else 0.0
        max_rel_cd = np.max(np.abs(rel_cd)) if np.any(mask_rel_cd) else 0.0

        print("max rel_cc", max_rel_cc)
        print("max rel_cd", max_rel_cd)


        # ---- 6. Accept step & check convergence ----------------------
        cc_prev = cc.copy()
        cd_prev = cd.copy()

        # use the accepted trial
        cc = cc_trial
        cd = cd_trial

        # hard physical bounds
        cc = np.clip(cc, 0.0, sid.cb_in)
        cd = np.clip(cd, 0.0, sid.cd_in)

        diff_cc = np.linalg.norm(cc - cc_prev)
        diff_cd = np.linalg.norm(cd - cd_prev)

        # convergence based on residuals + step size
        res_tol  = 1e-2
        step_tol = 1e-4
        # ----- RELATIVE RESIDUALS, IGNORING WEIRD NODES -----
 


        if (max_rel_cc < res_tol and max_rel_cd < res_tol
            and diff_cc < step_tol and diff_cd < step_tol):
            print(f"Newton converged in {it} iterations (Δcc={diff_cc:.1e}, Δcd={diff_cd:.1e})")
            break

        # global residual norm
        res_norm = np.sqrt(2.0 * phi_t)

        res_tol_global = 1e-10  # you can tune this
        step_tol       = 1e-4   # as you had

        if (res_norm < res_tol_global and diff_cc < step_tol and diff_cd < step_tol):
            print(f"Newton converged (global) in {it} iterations "
                f"(‖F‖={res_norm:.1e}, Δcc={diff_cc:.1e}, Δcd={diff_cd:.1e})")
            break


        if it == max_iter:
            print("Warning: Newton hit max_iter; using last iterate "
                  f"(max_rel_cc={max_rel_cc:.3e}, max_rel_cd={max_rel_cd:.3e})")
            break

    else:
        raise RuntimeError("Newton did not converge within the iteration limit")

    return cc, cd
