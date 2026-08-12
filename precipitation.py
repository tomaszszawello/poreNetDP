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
solve_precipitation_nr9_vxx_d0_hindering(...)
    concentration-aware precipitation hindering using upstream D0
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


def solve_precipitation_nr9_vxx(sid, inc, graph, edges, vols, cb, cc, cd,
                        tol: float = 1e-2,
                        max_iter: int = 100,
                        red: float = 0.5,
                        lam_min: float = 1e-10):
    # Newton iterations mutate cc/cd in place (e.g. `cc += lams * delta[:N]`).
    # Copy on entry so a failed solve never corrupts the caller's arrays —
    # callers rely on their cc/cd being untouched when this raises.
    cc = cc.copy()
    cd = cd.copy()
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
    flow_diag = spr.diags(edges.flow)
    U = 1 * (flow_diag @ Inc > 0)

    # Downstream‑selector matrix  D
    D = 1 * (flow_diag @ Inc < 0)
    D_T = D.T.tocsr()   # CSR transpose: reused in residual & Jacobian

    #Q_in = np.abs(inc.incidence.T) @ np.abs(edges.flow) / 2 * (1 - graph.in_vec + graph.out_vec) + graph.in_vec
    Q_in = D_T @ abs_Q
    Q_in = Q_in * (1 - graph.in_vec) + graph.in_vec


    #in_vec = np.concatenate((graph.in_vec, graph.in_vec))
    in_vec = np.copy(graph.in_vec)


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

    # Precompute fixed diagonal matrices as CSR (avoids dia→CSR conversion each
    # Newton iteration when used in arithmetic with other CSR matrices)
    Q_in_diag    = spr.diags(Q_in).tocsr()
    in_vec_d_not = spr.diags(1.0 - in_vec_conc).tocsr()
    in_vec_d     = spr.diags(in_vec_conc).tocsr()
    U_csr        = U.tocsr()  # ensure CSR for efficient .multiply()
    _n_restarts  = 0          # limit restarts to avoid infinite loops
    _best_phi    = np.inf
    _best_cc     = cc.copy()
    _best_cd     = cd.copy()

    def safe_exp(x):
        if np.max(x) > 700:
            print('large exp!')
        return np.exp(np.clip(x, -700.0, 700.0))

    for it in range(1, max_iter + 1):

        cb_in  = U_csr @ cb
        cb_out = D @ cb
        cc_in  = U_csr @ cc
        cd_in  = U_csr @ cd


        B   = B_pref / q_safe
        eB  = safe_exp(-np.abs(B * L))

        C   = k * cd_in
        lam = C / q_safe
        g   = safe_exp(-np.abs(edges.alpha_c * lam * L))

        den = edges.alpha_c * k * cd_in - B_pref

        A_e  = edges.alpha_b * sid.Da * d * cb_in / (1.0 + sid.G * d)

        tau  = abs(den) / (abs(edges.alpha_c * k*cd_in) + abs(B_pref) + 1.0)
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
        alpha_prime = -A_e * edges.alpha_c * k / den**2
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
        F_cc = cc * Q_in - D_T @ (abs_Q * cc_out)
        F_cd = cd * Q_in - D_T @ (abs_Q * cd_out)
        F_cc *= (1 - in_vec)
        F_cd *= (1 - in_vec)
        F = np.concatenate((F_cc, F_cd))

        # Early exit: skip J build + gssv when input is already machine-converged
        # (e.g. reconciliation NR called with cc/cd that were just converged).
        # Use a tight threshold (1e-20) to avoid accepting a loose warm-start.
        phi_pre = 0.5 * np.dot(F, F)
        if phi_pre < 1e-20:
            print(f"Newton converged in {it-1} iterations (phi={phi_pre:.3e})")
            break

        # ---- 5. Jacobian blocks (D_T @ U_csr.multiply avoids 4 diag creations)
        J_cc_cc_0 = D_T @ U_csr.multiply((abs_Q * dccout_dccu).reshape(-1, 1))
        J_cc_cd_0 = D_T @ U_csr.multiply((abs_Q * dccout_dcd).reshape(-1, 1))
        J_cd_cc_0 = D_T @ U_csr.multiply((abs_Q * dcdout_dccu).reshape(-1, 1))
        J_cd_cd_0 = D_T @ U_csr.multiply((abs_Q * dcdout_dcd).reshape(-1, 1))

        J_cc_cc = Q_in_diag - J_cc_cc_0
        J_cc_cd =           - J_cc_cd_0
        J_cd_cc =           - J_cd_cc_0
        J_cd_cd = Q_in_diag - J_cd_cd_0


        J = spr.vstack((spr.hstack((J_cc_cc, J_cc_cd)),
                        spr.hstack((J_cd_cc, J_cd_cd))))

        J = in_vec_d_not @ J + in_vec_d
        J_diag = J.diagonal()
        zero_diag = J_diag == 0
        if np.any(zero_diag):
            J += spr.diags(zero_diag.astype(float)).tocsr()
        F *= ~zero_diag
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
            cc_in_t = U_csr @ cc_trial
            cd_in_t = U_csr @ cd_trial
            den_t   = edges.alpha_c * k * cd_in_t - B_pref
            tau     = abs(den_t) / (abs(edges.alpha_c * k*cd_in_t) + abs(B_pref) + 1.0)
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

            F_cc_t = cc_trial * Q_in - D_T @ (abs_Q * cc_out_t)
            F_cd_t = cd_trial * Q_in - D_T @ (abs_Q * cd_out_t)
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

        # If the line search completely failed (λ→0), don't waste 48 more
        # no-op iterations — restart immediately with a fresh initial guess.
        if lams < lam_min and _n_restarts == 0:
            _n_restarts += 1
            print(f"Newton: line search stagnated at it={it}, restarting with fresh initial guess")
            cc, cd = create_vector_nr(sid, graph, inc, edges, cb)
            continue  # go straight to next iteration with fresh cc/cd
        elif lams < lam_min:
            phi_max = 100 * sid.it_alpha_th
            if _best_phi < phi_max:
                print(f"Newton: stagnated again at it={it}, returning best (phi={_best_phi:.3e})")
                cc = np.clip(_best_cc, 0.0, sid.cb_in)
                cd = np.clip(_best_cd, 0.0, sid.cd_in)
                return cc, cd
            raise RuntimeError(
                f"Newton: second stagnation, best phi={_best_phi:.3e} >= phi_max={phi_max:.3e}"
            )

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
        if phi_t < _best_phi:
            _best_phi = phi_t
            _best_cc  = cc.copy()
            _best_cd  = cd.copy()
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
        phi_max = 100 * sid.it_alpha_th   # 100× convergence threshold (e.g. 1.0 when thr=0.01)
        if _best_phi < phi_max:
            print(f"Newton max_iter reached; returning best solution (phi={_best_phi:.3e})")
            cc = np.clip(_best_cc, 0.0, sid.cb_in)
            cd = np.clip(_best_cd, 0.0, sid.cd_in)
            return cc, cd
        raise RuntimeError(
            f"Newton failed: best phi={_best_phi:.3e} exceeds phi_max={phi_max:.3e}"
        )

    # clip outputs

    cc = np.clip(cc, 0.0, sid.cb_in)
    cd = np.clip(cd, 0.0, sid.cd_in)
    
        
    return cc, cd


def solve_precipitation_nr9_vxx_d0_hindering(sid, inc, graph, edges, vols, cb, cc, cd,
                        tol: float = 1e-2,
                        max_iter: int = 100,
                        red: float = 0.5,
                        lam_min: float = 1e-10):
    """Newton solver with D0-dependent transverse hindering of precipitation.

    This is a concentration-aware variant of :func:`solve_precipitation_nr9_vxx`.
    The legacy pseudo-first-order precipitation coefficient is

        P_old(D0) = Da * K * d * (D0 / Kp) / (1 + G * K * d),

    whereas this solver uses

        P(D0) = Da * K * d * (D0 / Kp)
                / (1 + G * K * d * (D0 / Kp)).

    Here ``D0`` is the upstream edge concentration of species D (``cd_in``).
    Because D0 is still assumed constant along an individual edge, the analytic
    single-edge solution used by ``nr9_vxx`` is retained.  The Newton Jacobian
    is updated with the exact derivative

        dP/dD0 = (Da * K * d / Kp)
                  / (1 + G * K * d * D0 / Kp)**2.

    Negative intermediate D0 values can occur during a line-search trial.  They
    are treated as zero only when evaluating the precipitation coefficient; the
    nodal mass-balance equations and the existing stoichiometric cap are left
    unchanged.
    """
    # Newton iterations mutate cc/cd in place (e.g. `cc += lams * delta[:N]`).
    # Copy on entry so a failed solve never corrupts the caller's arrays —
    # callers rely on their cc/cd being untouched when this raises.
    cc = cc.copy()
    cd = cd.copy()
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
    flow_diag = spr.diags(edges.flow)
    U = 1 * (flow_diag @ Inc > 0)

    # Downstream‑selector matrix  D
    D = 1 * (flow_diag @ Inc < 0)
    D_T = D.T.tocsr()   # CSR transpose: reused in residual & Jacobian

    #Q_in = np.abs(inc.incidence.T) @ np.abs(edges.flow) / 2 * (1 - graph.in_vec + graph.out_vec) + graph.in_vec
    Q_in = D_T @ abs_Q
    Q_in = Q_in * (1 - graph.in_vec) + graph.in_vec


    #in_vec = np.concatenate((graph.in_vec, graph.in_vec))
    in_vec = np.copy(graph.in_vec)


    d  = edges.diams
    L  = edges.lens
    q  = np.abs(edges.flow)
    eps_q = 1e-12
    q_safe = np.where(q > eps_q, q, 1.0)   # any nonzero dummy
    mask_zero = q <= eps_q                              # magnitude only

    B_pref = edges.alpha_b * sid.Da * d / (1.0 + sid.G * d)

    if sid.Kp <= 0:
        raise ValueError("sid.Kp must be positive")

    # Concentration-aware pseudo-first-order precipitation coefficient:
    #   P(D0) = k0 * D0 / (1 + h * D0)
    # where h*D0 = G*K*d*(D0/Kp).
    precip_k0 = sid.Da * sid.K * d / sid.Kp
    precip_h  = sid.G * sid.K * d / sid.Kp

    def precipitation_coefficient(cd_up):
        """Return P(D0) and dP/dD0 for every edge."""
        cd_rate = np.maximum(np.asarray(cd_up, dtype=float), 0.0)
        hindering = 1.0 + precip_h * cd_rate
        coeff = precip_k0 * cd_rate / hindering
        derivative = precip_k0 / hindering**2
        derivative = np.where(np.asarray(cd_up) >= 0.0, derivative, 0.0)
        return coeff, derivative

    diff_cc = 0

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

    # Precompute fixed diagonal matrices as CSR (avoids dia→CSR conversion each
    # Newton iteration when used in arithmetic with other CSR matrices)
    Q_in_diag    = spr.diags(Q_in).tocsr()
    in_vec_d_not = spr.diags(1.0 - in_vec_conc).tocsr()
    in_vec_d     = spr.diags(in_vec_conc).tocsr()
    U_csr        = U.tocsr()  # ensure CSR for efficient .multiply()
    _n_restarts  = 0          # limit restarts to avoid infinite loops
    _best_phi    = np.inf
    _best_cc     = cc.copy()
    _best_cd     = cd.copy()

    def safe_exp(x):
        if np.max(x) > 700:
            print('large exp!')
        return np.exp(np.clip(x, -700.0, 700.0))

    for it in range(1, max_iter + 1):

        cb_in  = U_csr @ cb
        cb_out = D @ cb
        cc_in  = U_csr @ cc
        cd_in  = U_csr @ cd


        B   = B_pref / q_safe
        eB  = safe_exp(-np.abs(B * L))

        P, dP_dcd = precipitation_coefficient(cd_in)

        # Stable analytic edge solution.  With
        #   r = B_pref,  s = alpha_c * P,  x = L / |q|,
        # the C outlet concentration is
        #   C1 = C0 exp(-s x) + A_e [exp(-r x)-exp(-s x)]/(s-r).
        # The quotient is evaluated from its Taylor limit near s=r.  This is
        # important here because the D0-dependent coefficient can hit the
        # resonant condition s=r at common inlet concentrations.
        x_edge = L / q_safe
        s_rate = edges.alpha_c * P
        ds_dcd = edges.alpha_c * dP_dcd
        g      = safe_exp(-np.abs(s_rate * x_edge))
        den    = s_rate - B_pref

        A_e = edges.alpha_b * sid.Da * d * cb_in / (1.0 + sid.G * d)

        y = den * x_edge
        near_resonance = np.abs(y) < 1e-6
        phi = np.empty_like(den)
        phi_prime = np.empty_like(den)

        regular = ~near_resonance
        phi[regular] = (eB[regular] - g[regular]) / den[regular]
        phi_prime[regular] = (
            x_edge[regular] * g[regular] * den[regular]
            - (eB[regular] - g[regular])
        ) / den[regular]**2

        yr = y[near_resonance]
        xr = x_edge[near_resonance]
        er = eB[near_resonance]
        # phi = eB*x*(1-y/2+y^2/6-y^3/24+y^4/120+...)
        phi[near_resonance] = er * xr * (
            1.0 - 0.5*yr + yr**2/6.0 - yr**3/24.0 + yr**4/120.0
        )
        # d(phi)/d(s-r) = eB*x^2*(-1/2+y/3-y^2/8+y^3/30-y^4/144+...)
        phi_prime[near_resonance] = er * xr**2 * (
            -0.5 + yr/3.0 - yr**2/8.0 + yr**3/30.0 - yr**4/144.0
        )

        cc_out = cc_in * g + A_e * phi
        dccout_dccu = g
        dccout_dcd = (
            -cc_in * x_edge * g + A_e * phi_prime
        ) * ds_dcd

        cc_out = np.asarray(np.ma.fix_invalid(cc_out, fill_value=0.0))
        dccout_dcd = np.asarray(np.ma.fix_invalid(dccout_dcd, fill_value=0.0))
        cc_out[mask_zero] = cc_in[mask_zero]

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
        F_cc = cc * Q_in - D_T @ (abs_Q * cc_out)
        F_cd = cd * Q_in - D_T @ (abs_Q * cd_out)
        F_cc *= (1 - in_vec)
        F_cd *= (1 - in_vec)
        F = np.concatenate((F_cc, F_cd))

        # Early exit: skip J build + gssv when input is already machine-converged
        # (e.g. reconciliation NR called with cc/cd that were just converged).
        # Use a tight threshold (1e-20) to avoid accepting a loose warm-start.
        phi_pre = 0.5 * np.dot(F, F)
        if phi_pre < 1e-20:
            print(f"Newton converged in {it-1} iterations (phi={phi_pre:.3e})")
            break

        # ---- 5. Jacobian blocks (D_T @ U_csr.multiply avoids 4 diag creations)
        J_cc_cc_0 = D_T @ U_csr.multiply((abs_Q * dccout_dccu).reshape(-1, 1))
        J_cc_cd_0 = D_T @ U_csr.multiply((abs_Q * dccout_dcd).reshape(-1, 1))
        J_cd_cc_0 = D_T @ U_csr.multiply((abs_Q * dcdout_dccu).reshape(-1, 1))
        J_cd_cd_0 = D_T @ U_csr.multiply((abs_Q * dcdout_dcd).reshape(-1, 1))

        J_cc_cc = Q_in_diag - J_cc_cc_0
        J_cc_cd =           - J_cc_cd_0
        J_cd_cc =           - J_cd_cc_0
        J_cd_cd = Q_in_diag - J_cd_cd_0


        J = spr.vstack((spr.hstack((J_cc_cc, J_cc_cd)),
                        spr.hstack((J_cd_cc, J_cd_cd))))

        J = in_vec_d_not @ J + in_vec_d
        J_diag = J.diagonal()
        zero_diag = J_diag == 0
        if np.any(zero_diag):
            J += spr.diags(zero_diag.astype(float)).tocsr()
        F *= ~zero_diag
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
            cc_in_t = U_csr @ cc_trial
            cd_in_t = U_csr @ cd_trial
            P_t, _ = precipitation_coefficient(cd_in_t)
            s_rate_t = edges.alpha_c * P_t
            den_t = s_rate_t - B_pref
            g_t = safe_exp(-np.abs(s_rate_t * x_edge))

            y_t = den_t * x_edge
            near_t = np.abs(y_t) < 1e-6
            phi_t_edge = np.empty_like(den_t)
            regular_t = ~near_t
            phi_t_edge[regular_t] = (
                eB[regular_t] - g_t[regular_t]
            ) / den_t[regular_t]

            yt = y_t[near_t]
            xt = x_edge[near_t]
            et = eB[near_t]
            phi_t_edge[near_t] = et * xt * (
                1.0 - 0.5*yt + yt**2/6.0 - yt**3/24.0 + yt**4/120.0
            )

            cc_out_t = cc_in_t * g_t + A_e * phi_t_edge
            cc_out_t = np.asarray(np.ma.fix_invalid(cc_out_t, fill_value=0.0))
            cc_out_t[mask_zero] = cc_in_t[mask_zero]

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

            F_cc_t = cc_trial * Q_in - D_T @ (abs_Q * cc_out_t)
            F_cd_t = cd_trial * Q_in - D_T @ (abs_Q * cd_out_t)
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

        # If the line search completely failed (λ→0), don't waste 48 more
        # no-op iterations — restart immediately with a fresh initial guess.
        if lams < lam_min and _n_restarts == 0:
            _n_restarts += 1
            print(f"Newton: line search stagnated at it={it}, restarting with fresh initial guess")
            cc, cd = create_vector_nr(sid, graph, inc, edges, cb)
            continue  # go straight to next iteration with fresh cc/cd
        elif lams < lam_min:
            phi_max = 100 * sid.it_alpha_th
            if _best_phi < phi_max:
                print(f"Newton: stagnated again at it={it}, returning best (phi={_best_phi:.3e})")
                cc = np.clip(_best_cc, 0.0, sid.cb_in)
                cd = np.clip(_best_cd, 0.0, sid.cd_in)
                return cc, cd
            raise RuntimeError(
                f"Newton: second stagnation, best phi={_best_phi:.3e} >= phi_max={phi_max:.3e}"
            )

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
        if phi_t < _best_phi:
            _best_phi = phi_t
            _best_cc  = cc.copy()
            _best_cd  = cd.copy()
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
        phi_max = 100 * sid.it_alpha_th   # 100× convergence threshold (e.g. 1.0 when thr=0.01)
        if _best_phi < phi_max:
            print(f"Newton max_iter reached; returning best solution (phi={_best_phi:.3e})")
            cc = np.clip(_best_cc, 0.0, sid.cb_in)
            cd = np.clip(_best_cd, 0.0, sid.cd_in)
            return cc, cd
        raise RuntimeError(
            f"Newton failed: best phi={_best_phi:.3e} exceeds phi_max={phi_max:.3e}"
        )

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



def solve_precipitation_kp(sid, inc, graph, edges, vols, cb, cc, cd,
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

        epsS = 1e-3  # tune; smaller = sharper switch
        zS   = (cc_in * cd_in / sid.Kp) - 1.0
        H    = 0.5 * (1.0 + zS / np.sqrt(zS*zS + epsS*epsS))  # in (0,1)
        
        cd_eff = cd_in * H
        C   = k * cd_eff
        lam = C / q_safe
        den = k * cd_eff - B_pref
        lam = C / q_safe
        g   = safe_exp(-np.abs(edges.alpha_c * lam * L))

        den = edges.alpha_c * k * cd_in - B_pref

        A_e  = edges.alpha_b * sid.Da * d * cb_in / (1.0 + sid.G * d)

        tau  = abs(den) / (abs(edges.alpha_c * k*cd_in) + abs(B_pref) + 1.0)
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
        alpha_prime = -A_e * (k*H) / den**2
        g_prime     = -edges.alpha_c * ((k*H) / q_safe) * L * g
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
            den_t   = edges.alpha_c * k * cd_in_t - B_pref
            tau     = abs(den_t) / (abs(edges.alpha_c * k*cd_in_t) + abs(B_pref) + 1.0)
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

def solve_precipitation_safe(
    sid, inc, graph, edges, vols, cb, cc, cd,
    max_alpha_iter: int = 1,
    tol_alpha: float = 1e-3,
):
    """Iterative precipitation solver with alpha_c updated from available space.

    Mirrors solve_dissolution_safe in dissolution.py:
    - Initialise alpha_c from triangle availability (1 if space remains, 0 if full).
    - Each outer iteration: set edges.alpha_c, run NR, compute requested
      precipitation, clip alpha_c so no triangle is overfilled.
    - Iterate until alpha_c converges (or max_alpha_iter reached).
    - Stores final alpha_c in edges.alpha_c so solve_dp_vol uses the same value.

    Robustness: if the NR fails on the first outer iteration with the binary
    alpha_c (which can change abruptly between timesteps), one warm-start retry
    is attempted using min(edges.alpha_c_prev, alpha_c_binary) so the NR sees a
    smaller perturbation from the previous timestep.  If that also fails, the
    previous edges.alpha_c and cc/cd are kept unchanged for this step.
    If a later outer iteration fails, the last successfully converged result is used.
    """
    T         = vols.triangles
    edges_tri = np.maximum(edges.triangles, 1.0)
    tri_rows, tri_cols = T.nonzero()

    abs_q  = np.abs(edges.flow)
    q_safe = np.where(abs_q > 1e-12, abs_q, 1.0)

    # Available pore space (fixed for this timestep — computed before growth)
    available = vols.vol_max - vols.vol_a - vols.vol_e

    # Binary mask: 1 where triangle still has space, 0 where triangle is full.
    # This is the physically-correct starting point and allows alpha_c to recover
    # when pore space opens up after dissolution.
    alpha_c_binary = (T @ (1.0 * (available > 0))) / edges_tri
    alpha_c_binary = np.clip(
        np.array(np.ma.fix_invalid(alpha_c_binary, fill_value=0.0)), 0.0, 1.0)

    # Warm-start alternative: clip previous alpha_c to 0 for newly-blocked
    # triangles but otherwise keep the previous value.  Used as fallback when
    # the NR cannot converge from the binary initialization.
    alpha_c_warm = np.minimum(edges.alpha_c, alpha_c_binary)

    # Track the best (last converged) result so any later failure can revert.
    cc_best       = cc.copy()
    cd_best       = cd.copy()
    alpha_c_best  = edges.alpha_c.copy()

    alpha_c = alpha_c_binary.copy()

    for it_alpha in range(max_alpha_iter):
        alpha_prev = alpha_c.copy()
        edges.alpha_c = alpha_c

        # Initial guess for the NR.
        # it_alpha == 0: pass in the previous-step cc/cd (they come from outside).
        # it_alpha  > 0: alpha_c changed; re-init from create_vector_nr for a fresh start.
        if it_alpha == 0:
            cc_init, cd_init = cc, cd
        else:
            cc_init, cd_init = create_vector_nr(sid, graph, inc, edges, cb)

        try:
            cc_new, cd_new = solve_precipitation_nr9_vxx(
                sid, inc, graph, edges, vols, cb, cc_init, cd_init)

        except RuntimeError:
            if it_alpha == 0:
                # Binary alpha_c caused an NR failure (abrupt change between
                # timesteps).  Try the warm-start alpha_c as a one-off fallback.
                print(f"precipitation alpha_c iter 0: NR failed with binary alpha_c "
                      f"(n_blocked={int(np.sum(alpha_c == 0))}); "
                      f"retrying with warm-start alpha_c")
                alpha_c = alpha_c_warm.copy()
                edges.alpha_c = alpha_c
                cc_init2, cd_init2 = create_vector_nr(sid, graph, inc, edges, cb)
                try:
                    cc_new, cd_new = solve_precipitation_nr9_vxx(
                        sid, inc, graph, edges, vols, cb, cc_init2, cd_init2)
                except RuntimeError:
                    print(f"precipitation alpha_c iter 0: NR failed with warm-start "
                          f"alpha_c too; keeping previous result "
                          f"(n_blocked_prev={int(np.sum(alpha_c_best == 0))})")
                    edges.alpha_c = alpha_c_best
                    return cc_best, cd_best
            else:
                print(f"precipitation alpha_c iter {it_alpha}: NR failed; "
                      f"reverting to best previous result "
                      f"(n_blocked_best={int(np.sum(alpha_c_best == 0))})")
                edges.alpha_c = alpha_c_best
                return cc_best, cd_best

        # NR converged — update best-known result.
        cc, cd       = cc_new, cd_new
        cc_best      = cc.copy()
        cd_best      = cd.copy()
        alpha_c_best = alpha_c.copy()

        # Requested precipitation per edge (same formula as solve_dp_vol)
        growth_matrix = np.abs((spr.diags(edges.flow) @ inc.incidence > 0))
        cb_in = growth_matrix @ cb
        cc_in = growth_matrix @ cc
        cd_in = growth_matrix @ cd

        X_rate = cd_in * sid.K / (1.0 + sid.K * sid.G * edges.diams)
        Y_rate = sid.Kp / (1.0 + sid.G * edges.diams)
        ksi    = X_rate - edges.alpha_b * Y_rate

        exp_p2 = np.array(np.ma.fix_invalid(
            np.exp(-alpha_c * sid.Da * sid.K / (1.0 + sid.G * sid.K * edges.diams)
                   * cd_in / sid.Kp * edges.diams * edges.lens / q_safe),
            fill_value=0.0))
        exp_d2 = np.array(np.ma.fix_invalid(
            np.exp(-edges.alpha_b * sid.Da / (1.0 + sid.G * edges.diams)
                   * edges.diams * edges.lens / q_safe),
            fill_value=0.0))

        shrink_cc2 = cc_in * abs_q * sid.Gamma / sid.Da * (1.0 - exp_p2)
        shrink_cb2 = np.array(np.ma.fix_invalid(
            cb_in * abs_q * sid.Gamma / sid.Da
            * (X_rate * (1.0 - exp_d2) - edges.alpha_b * Y_rate * (1.0 - exp_p2)) / ksi,
            fill_value=0.0))

        precipitate_req = (shrink_cc2 + shrink_cb2) * sid.dt

        # Requested volume per triangle
        P0_t = T.T @ (precipitate_req / edges_tri)

        # Triangle safety factors: how much of the requested precipitation fits
        v_eps = 1e-16
        f_t = np.ones(sid.ntr)
        mask_over = P0_t > v_eps
        f_t[mask_over] = np.minimum(
            1.0, available[mask_over] / (P0_t[mask_over] + v_eps))

        # Edge safety = minimum over all neighbouring triangles
        s_e = np.ones(sid.ne)
        np.minimum.at(s_e, tri_rows, f_t[tri_cols])
        s_e = np.clip(s_e, 0.0, 1.0)

        alpha_c = np.clip(alpha_c * s_e, 0.0, 1.0)
        alpha_c_best = alpha_c.copy()  # post-safety-factor: the correct revert target

        diff_alpha = np.linalg.norm(alpha_c - alpha_prev, ord=np.inf)
        print(f"precipitation alpha_c iter {it_alpha}: "
              f"diff={diff_alpha:.3e}  n_blocked={int(np.sum(alpha_c == 0))}")
        if diff_alpha < tol_alpha:
            break

    # ── Reconciliation NR ────────────────────────────────────────────────────
    # The outer loop's last NR used alpha_c before the safety-factor update.
    # When alpha_c changed significantly (diff_alpha >= tol_alpha), run one
    # more NR with the definitive alpha_c to close the one-step-lag bias.
    # When diff_alpha < tol_alpha (alpha_c unchanged or only ε-changed),
    # the existing cc/cd are already consistent — skip the extra NR call.
    edges.alpha_c = alpha_c
    if diff_alpha < tol_alpha:
        return cc, cd

    try:
        cc_f, cd_f = solve_precipitation_nr9_vxx(
            sid, inc, graph, edges, vols, cb, cc, cd)
        cc, cd = cc_f, cd_f
    except RuntimeError:
        try:
            cc_init_f, cd_init_f = create_vector_nr(sid, graph, inc, edges, cb)
            cc_f, cd_f = solve_precipitation_nr9_vxx(
                sid, inc, graph, edges, vols, cb, cc_init_f, cd_init_f)
            cc, cd = cc_f, cd_f
        except RuntimeError:
            print("precipitation reconciliation NR: failed, using outer-loop cc/cd")
    return cc, cd


def _requested_precipitation_d0_hindering(
    sid, inc, edges, cb, cc, cd, alpha_c,
):
    """Return requested secondary-solid volume per edge for the D0 model.

    The calculation uses the same concentration-aware pseudo-first-order
    precipitation coefficient and the same stable single-edge analytical
    solution as :func:`solve_precipitation_nr9_vxx_d0_hindering`.

    Parameters
    ----------
    sid, inc, edges
        Standard simulation/network objects.
    cb, cc, cd : ndarray
        Converged nodal concentrations.
    alpha_c : ndarray
        Edge precipitation-availability factors used in the transport solve.

    Returns
    -------
    ndarray
        Requested precipitated solid volume on every edge over ``sid.dt``.

    Notes
    -----
    For an edge, stoichiometry gives the amount of D consumed as

        Delta D = (B_0 - B_1) + (C_0 - C_1).

    This avoids duplicating the older closed-form volume expression, which was
    derived for the legacy, concentration-independent hindering denominator.
    The demand is capped by the available upstream D concentration, exactly as
    in the nonlinear transport solver.
    """
    if sid.Kp <= 0:
        raise ValueError("sid.Kp must be positive")

    d = np.asarray(edges.diams, dtype=float)
    length = np.asarray(edges.lens, dtype=float)
    abs_q = np.abs(np.asarray(edges.flow, dtype=float))
    active = abs_q > 1e-12
    q_safe = np.where(active, abs_q, 1.0)

    # Upstream-selector matrix: one upstream node per flowing edge.
    upstream = (spr.diags(edges.flow) @ inc.incidence > 0).astype(float).tocsr()
    cb_in = np.asarray(upstream @ cb, dtype=float)
    cc_in = np.asarray(upstream @ cc, dtype=float)
    cd_in = np.asarray(upstream @ cd, dtype=float)
    cd_rate = np.maximum(cd_in, 0.0)

    # Dissolution coefficient and B outlet.
    b_rate = edges.alpha_b * sid.Da * d / (1.0 + sid.G * d)
    residence = length / q_safe

    def safe_exp(exponent):
        return np.exp(np.clip(exponent, -700.0, 700.0))

    exp_b = safe_exp(-np.abs(b_rate * residence))
    cb_out = cb_in * exp_b

    # Concentration-aware precipitation coefficient:
    # P(D0) = Da*K*d*(D0/Kp) / [1 + G*K*d*(D0/Kp)].
    d0_over_kp = cd_rate / sid.Kp
    hindering = 1.0 + sid.G * sid.K * d * d0_over_kp
    p_rate = sid.Da * sid.K * d * d0_over_kp / hindering
    sink_rate = np.asarray(alpha_c, dtype=float) * p_rate
    exp_p = safe_exp(-np.abs(sink_rate * residence))

    # Stable evaluation of
    #   C1 = C0 exp(-s x) + A [exp(-r x)-exp(-s x)]/(s-r),
    # including its finite limit at s == r.
    denominator = sink_rate - b_rate
    y = denominator * residence
    near_resonance = np.abs(y) < 1e-6

    quotient = np.empty_like(denominator)
    regular = ~near_resonance
    quotient[regular] = (
        exp_b[regular] - exp_p[regular]
    ) / denominator[regular]

    yr = y[near_resonance]
    xr = residence[near_resonance]
    er = exp_b[near_resonance]
    quotient[near_resonance] = er * xr * (
        1.0
        - 0.5 * yr
        + yr**2 / 6.0
        - yr**3 / 24.0
        + yr**4 / 120.0
    )

    source_amplitude = b_rate * cb_in
    cc_out = cc_in * exp_p + source_amplitude * quotient

    cb_out = np.where(np.isfinite(cb_out), cb_out, cb_in)
    cc_out = np.where(np.isfinite(cc_out), cc_out, cc_in)

    # The transport solver applies the same finite-D cap when its unconstrained
    # edge solution would require cd_out < 0.
    d_consumed = (cb_in - cb_out) + (cc_in - cc_out)
    d_consumed = np.minimum(np.maximum(d_consumed, 0.0), cd_rate)
    d_consumed[~active] = 0.0

    if abs(sid.Da) <= 1e-30:
        return np.zeros_like(abs_q)

    precipitate_req = (
        abs_q * d_consumed * sid.Gamma / sid.Da * sid.dt
    )
    return np.asarray(
        np.ma.fix_invalid(precipitate_req, fill_value=0.0), dtype=float
    )


def solve_precipitation_safe_d0(
    sid, inc, graph, edges, vols, cb, cc, cd,
    max_alpha_iter: int = 1,
    tol_alpha: float = 1e-3,
):
    """Safe precipitation wrapper for the D0-dependent hindering solver.

    This is the concentration-aware counterpart of
    :func:`solve_precipitation_safe`.  It updates ``edges.alpha_c`` from the
    pore volume available in adjacent triangles, solves transport with
    :func:`solve_precipitation_nr9_vxx_d0_hindering`, and limits the requested
    precipitation so that no triangle is overfilled.

    The legacy ``solve_precipitation_safe`` function is left unchanged.

    Parameters
    ----------
    max_alpha_iter : int, optional
        Maximum number of outer availability-factor iterations.  Must be at
        least one.
    tol_alpha : float, optional
        Infinity-norm tolerance for convergence of ``alpha_c``.

    Returns
    -------
    cc, cd : ndarray
        Converged nodal concentrations of C and D.
    """
    if max_alpha_iter < 1:
        raise ValueError("max_alpha_iter must be at least 1")

    triangles_incidence = vols.triangles
    edges_tri = np.maximum(edges.triangles, 1.0)
    tri_rows, tri_cols = triangles_incidence.nonzero()

    # Available pore volume at the beginning of this geometry-update step.
    available = vols.vol_max - vols.vol_a - vols.vol_e

    # Start from the physically admissible binary state.  This also lets an
    # edge recover when dissolution has opened pore space since the last step.
    alpha_c_binary = (
        triangles_incidence @ (1.0 * (available > 0))
    ) / edges_tri
    alpha_c_binary = np.clip(
        np.asarray(
            np.ma.fix_invalid(alpha_c_binary, fill_value=0.0), dtype=float
        ),
        0.0,
        1.0,
    )

    # If the abrupt binary update makes the nonlinear solve fail, retry once
    # from the previous edge availability, clipped by newly blocked triangles.
    alpha_c_warm = np.minimum(edges.alpha_c, alpha_c_binary)

    cc_best = cc.copy()
    cd_best = cd.copy()
    alpha_c_best = edges.alpha_c.copy()

    alpha_c = alpha_c_binary.copy()
    diff_alpha = np.inf

    for it_alpha in range(max_alpha_iter):
        alpha_prev = alpha_c.copy()
        edges.alpha_c = alpha_c

        if it_alpha == 0:
            cc_init, cd_init = cc, cd
        else:
            # ``create_vector_nr`` is used only as an initial guess.  The final
            # equations are always evaluated with the D0-dependent solver.
            cc_init, cd_init = create_vector_nr(sid, graph, inc, edges, cb)

        try:
            cc_new, cd_new = solve_precipitation_nr9_vxx_d0_hindering(
                sid, inc, graph, edges, vols, cb, cc_init, cd_init
            )

        except RuntimeError:
            if it_alpha == 0:
                print(
                    "precipitation D0 alpha_c iter 0: NR failed with binary "
                    f"alpha_c (n_blocked={int(np.sum(alpha_c == 0))}); "
                    "retrying with warm-start alpha_c"
                )
                alpha_c = alpha_c_warm.copy()
                edges.alpha_c = alpha_c
                cc_init2, cd_init2 = create_vector_nr(
                    sid, graph, inc, edges, cb
                )
                try:
                    cc_new, cd_new = (
                        solve_precipitation_nr9_vxx_d0_hindering(
                            sid,
                            inc,
                            graph,
                            edges,
                            vols,
                            cb,
                            cc_init2,
                            cd_init2,
                        )
                    )
                except RuntimeError:
                    print(
                        "precipitation D0 alpha_c iter 0: NR failed with "
                        "warm-start alpha_c too; keeping previous result "
                        f"(n_blocked_prev={int(np.sum(alpha_c_best == 0))})"
                    )
                    edges.alpha_c = alpha_c_best
                    return cc_best, cd_best
            else:
                print(
                    f"precipitation D0 alpha_c iter {it_alpha}: NR failed; "
                    "reverting to best previous result "
                    f"(n_blocked_best={int(np.sum(alpha_c_best == 0))})"
                )
                edges.alpha_c = alpha_c_best
                return cc_best, cd_best

        # The transport solve converged for the current alpha_c.
        cc, cd = cc_new, cd_new
        cc_best = cc.copy()
        cd_best = cd.copy()
        alpha_c_best = alpha_c.copy()

        # Requested volume is calculated from the same stable edge solution and
        # D0-dependent coefficient as the nonlinear transport solver.
        precipitate_req = _requested_precipitation_d0_hindering(
            sid, inc, edges, cb, cc, cd, alpha_c
        )

        # Distribute each edge request equally among its neighbouring triangles.
        requested_per_triangle = triangles_incidence.T @ (
            precipitate_req / edges_tri
        )

        volume_eps = 1e-16
        triangle_factor = np.ones(sid.ntr)
        requested_mask = requested_per_triangle > volume_eps
        triangle_factor[requested_mask] = np.minimum(
            1.0,
            available[requested_mask]
            / (requested_per_triangle[requested_mask] + volume_eps),
        )

        # An edge must respect the most restrictive adjacent triangle.
        edge_factor = np.ones(sid.ne)
        np.minimum.at(edge_factor, tri_rows, triangle_factor[tri_cols])
        edge_factor = np.clip(edge_factor, 0.0, 1.0)

        alpha_c = np.clip(alpha_c * edge_factor, 0.0, 1.0)
        alpha_c_best = alpha_c.copy()

        diff_alpha = np.linalg.norm(alpha_c - alpha_prev, ord=np.inf)
        print(
            f"precipitation D0 alpha_c iter {it_alpha}: "
            f"diff={diff_alpha:.3e}  "
            f"n_blocked={int(np.sum(alpha_c == 0))}"
        )
        if diff_alpha < tol_alpha:
            break

    # The final safety-factor update may have changed alpha_c after the last
    # transport solve.  Reconcile once with the definitive value.
    edges.alpha_c = alpha_c
    if diff_alpha < tol_alpha:
        return cc, cd

    try:
        cc_final, cd_final = solve_precipitation_nr9_vxx_d0_hindering(
            sid, inc, graph, edges, vols, cb, cc, cd
        )
        return cc_final, cd_final
    except RuntimeError:
        try:
            cc_init_final, cd_init_final = create_vector_nr(
                sid, graph, inc, edges, cb
            )
            cc_final, cd_final = solve_precipitation_nr9_vxx_d0_hindering(
                sid,
                inc,
                graph,
                edges,
                vols,
                cb,
                cc_init_final,
                cd_init_final,
            )
            return cc_final, cd_final
        except RuntimeError:
            print(
                "precipitation D0 reconciliation NR: failed, using "
                "outer-loop cc/cd"
            )
            return cc, cd
