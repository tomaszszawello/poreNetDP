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

def solve_precipitation_nr2_vxx(sid: SimInputData, inc: Incidence, graph: Graph, \
    edges: Edges, vols: Volumes, cb, cc, cd) -> np.ndarray:
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
    edges.alpha_c = 1 #* ((vols.triangles @ (1 * (vols.vol_e + vols.vol_a < vols.vol_max))) > 0)
    c_inc = 1 * (inc.incidence.T @ (spr.diags(edges.flow) \
        @ inc.incidence > 0) != 0)
    diag = -np.abs(inc.incidence.T) @ np.abs(edges.flow) / 2
    diag = diag * (1 - graph.in_vec + graph.out_vec) + graph.in_vec
    diag += 1 * (diag == 0)
    
    if np.sum(cd) < 1.1 * np.sum(sid.cd_in * edges.inlet) or (np.sum(1 - ((cd == 1) + (cd == 0))) == 0):
        cd = sid.cd_in * np.ones(sid.nsq)
        cc = sid.cb_in - cb + sid.cc_in
        print('correcting initial cc, cd')

    cc_prev = np.zeros(sid.nsq)
    cd_prev = np.zeros(sid.nsq)
    it = 0
    diff_c = np.linalg.norm(cc - cc_prev)
    diff_d = np.linalg.norm(cd - cd_prev)
    while diff_c > sid.c_th or diff_d > sid.c_th:
        it += 1
        cb_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ np.abs(cb)
        cc_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ np.abs(cc)
        cd_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ np.abs(cd)
        ksi = cd_in * sid.K / (1 + sid.K * sid.G * edges.diams) - sid.Kp / (1 + sid.G * edges.diams)
        # if np.sum(ksi >= -0.001):
        #     print('ksi!')
            #np.savetxt('ksi.txt', ksi)
        exp_p = np.exp(-edges.alpha_c * sid.Da * sid.K / (1 + sid.G * sid.K * edges.diams) \
            * cd_in / sid.Kp * edges.diams * edges.lens / np.abs(edges.flow))
        exp_p = np.array(np.ma.fix_invalid(exp_p, fill_value = 0))
        exp_d = np.exp(-edges.alpha_b * sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow))
        exp_d = np.array(np.ma.fix_invalid(exp_d, fill_value = 0))

        qc2 = -edges.alpha_b * np.abs(edges.flow) / (1 + sid.G * edges.diams) * sid.Kp * (exp_p - exp_d) / ksi
        #qc2 = np.array(np.ma.fix_invalid(qc2, fill_value = edges.diams * edges.lens * sid.Da / (1 + sid.G * edges.diams)))
        qc2 = np.array(np.ma.fix_invalid(qc2, fill_value = 0))
        qc_matrix = np.abs(inc.incidence.T) @ spr.diags(qc2) @ np.abs(inc.incidence)
        cc_matrix = c_inc.multiply(qc_matrix)
        #cc_matrix.setdiag(np.zeros(sid.nsq))
        diag_old = cc_matrix.diagonal()
        cc_matrix -= spr.diags(diag_old)
        f_cc2 = cc_matrix @ cb
        #np.savetxt('fcc2.txt', f_cc2)

        qc1 = np.abs(edges.flow) * exp_p
        qc_matrix = np.abs(inc.incidence.T) @ spr.diags(qc1) @ np.abs(inc.incidence)
        cc_matrix = c_inc.multiply(qc_matrix)
        #cc_matrix.setdiag(diag)
        diag_old = cc_matrix.diagonal()
        cc_matrix += spr.diags(diag - diag_old)
        f_cc1 = cc_matrix @ cc
        f_cc = (1 - graph.in_vec) * (f_cc1 + f_cc2)

        dq_cc_cd = edges.alpha_b * cb_in / (1 + sid.G * edges.diams) / (1 + sid.G * edges.diams * sid.K) * sid.K / ksi ** 2 \
            * (sid.Da * edges.diams * edges.lens * ksi * exp_p + np.abs(edges.flow) * sid.Kp * (exp_p - exp_d))
        dq_cc_cd = np.array(np.ma.fix_invalid(dq_cc_cd, fill_value = 0))
        dq_cc_cd -= cc_in * exp_p * sid.K * edges.diams * edges.lens * sid.Da / sid.Kp / (1 + sid.G * edges.diams * sid.K)
        dq_cc_cd = np.array(np.ma.fix_invalid(dq_cc_cd, fill_value = 0))
        dcc_cd_matrix = np.abs(inc.incidence.T) @ spr.diags(dq_cc_cd) @ np.abs(inc.incidence)
        #dcc_cd_matrix.setdiag(np.zeros(sid.nsq))
        diag_old = dcc_cd_matrix.diagonal()
        dcc_cd_matrix -= spr.diags(diag_old)

        dq_cd_cc = np.abs(edges.flow) * (exp_p - 1)
        dq_cd_cc_matrix = np.abs(inc.incidence.T) @ spr.diags(dq_cd_cc) @ np.abs(inc.incidence)
        dq_cd_cc_matrix = c_inc.multiply(dq_cd_cc_matrix)
        #dq_cd_cc_matrix.setdiag(np.zeros(sid.nsq))
        diag_old = dq_cd_cc_matrix.diagonal()
        dq_cd_cc_matrix -= spr.diags(diag_old)

        qc = np.abs(edges.flow) * (1 - exp_d)
        qc_matrix = np.abs(inc.incidence.T @ spr.diags(qc) @ inc.incidence)
        cb_matrix = c_inc.multiply(qc_matrix)
        #cb_matrix.setdiag(np.zeros(sid.nsq))
        diag_old = cb_matrix.diagonal()
        cb_matrix -= spr.diags(diag_old)

        cd_mix = cd
        cd_matrix = c_inc.multiply(np.abs(inc.incidence.T @ spr.diags(np.abs(edges.flow)) @ inc.incidence))
        #cd_matrix.setdiag(diag)
        diag_old = cd_matrix.diagonal()
        cd_matrix += spr.diags(diag - diag_old)
        f_cd = (1 - graph.in_vec) * (cd_matrix @ cd_mix + dq_cd_cc_matrix @ cc + f_cc2 - cb_matrix @ cb)
        #np.savetxt('f_cd.txt', f_cd)
        #np.savetxt('f_cc.txt', f_cc)
        dq_cd_cd = cd_matrix + dcc_cd_matrix


        if it == 1:
           print(f_cc)
           print(f_cd)
        dc_matrix = spr.vstack([spr.hstack([cc_matrix, dcc_cd_matrix]), spr.hstack([dq_cd_cc_matrix, dq_cd_cd])])
        f = np.concatenate((f_cc, f_cd))
        if it > 5:
            c_zero = 1 * (np.concatenate((cc, cd)) <= 0)
            #c_zero = 1 * (np.concatenate((cc, cd)) < 0)
            f = f * (1 - c_zero)
        dc_matrix = spr.diags(1 - 1 * (f == 0)) @ dc_matrix + spr.diags(1 * (f == 0))
        in_vec = np.concatenate((graph.in_vec, graph.in_vec))
        dc_matrix = spr.diags(1 - in_vec) @ dc_matrix + spr.diags(in_vec)
        delta_c = solve_equation(dc_matrix, -f)
        lam      = 1.0
        lam_min  = 1e-1            # allow very small steps before giving up
        red      = 0.5

        # while True:
        #     f_trial = f + lam * (-f)            # J·delta = -f
        #     if np.linalg.norm(f_trial) < np.linalg.norm(f):
        #         break                           # success
        #     lam *= red
        #     if lam < lam_min:
        #         lam = 1
        cc_prev = cc.copy()
        cd_prev = cd.copy()
        cc += lam * delta_c[:sid.nsq]
        cd += lam * delta_c[sid.nsq:]
        # cc = np.clip(cc, 0, sid.cb_in)
        # cd = np.clip(cd, 0, sid.cd_in)
        diff_c_new = np.linalg.norm(cc - cc_prev)
        diff_d_new = np.linalg.norm(cd - cd_prev)
        if np.abs(diff_c_new - diff_c) < 1e-3:
            cc -= 0.5 * lam * delta_c[:sid.nsq]
            cd -= 0.5 * lam * delta_c[sid.nsq:]
            # cc = np.clip(cc, 0, sid.cb_in)
            # cd = np.clip(cd, 0, sid.cd_in)
        diff_c = np.linalg.norm(cc - cc_prev)
        diff_d = np.linalg.norm(cd - cd_prev)
        print(np.linalg.norm(cc - cc_prev), np.linalg.norm(cd - cd_prev))
        if it == 30:
            cc, cd = create_vector_nr(sid, graph, inc, edges, cb)
            print('restarting iterations')
        elif it > 300:
            cc = -1 * np.ones_like(cc)
            cd = -1 * np.ones_like(cc)
            print('NR didnt converge')
            return cc, cd
    cc = np.clip(cc, 0, sid.cb_in)
    cd = np.clip(cd, 0, sid.cd_in)
    return cc, cd

def solve_precipitation_nr4_vxx(sid, inc, graph, edges, vols, cb, cc, cd,
                        tol: float = 1e-2,
                        max_iter: int = 300,
                        red: float = 0.5,
                        lam_min: float = 1e-3):
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 0.  Handy aliases & basic sparse helpers
    E, N = inc.incidence.shape
    Inc  = inc.incidence              # ensure CSR
    abs_Q = np.abs(edges.flow)                # |Q_e|

    # Upstream‑selector matrix  U  (|E|×|N|, CSR, entries 0/1)
    U = 1 * (spr.diags(edges.flow) @ Inc < 0)

    # Downstream‑selector matrix  D
    D = 1 * (spr.diags(edges.flow) @ Inc > 0)

    # Absolute‑value incidence |Inc|
    AInc = np.abs(Inc)

    # Per‑node inflow Σ|Q|
    Q_in = np.asarray(abs_Q @ AInc).ravel()
    Q_in[Q_in == 0.0] = 1.0                   # protect zeros

    in_vec = np.concatenate((graph.in_vec, graph.in_vec))

    # ------------------------------------------------------------------
    # 1.  Initial guess
    # cc = np.full(N, sid.cc_in, dtype=float)
    # cd = np.full(N, sid.cd_in, dtype=float)
    #inlet_mask = graph.in_vec.astype(bool)

    # ------------------------------------------------------------------
    # 2.  Edge‑wise geometry (constant)
    d  = edges.diams
    L  = edges.lens
    q  = abs_Q                                 # magnitude only

    B_pref = sid.Da * d / (1.0 + sid.G * d)
    k_pref = sid.Da * sid.K * d / (sid.Kp * (1.0 + sid.G * sid.K * d))

    eps = np.finfo(float).eps                  # tiny for divides


    # ------------------------------------------------------------------
    for it in range(1, max_iter + 1):
        # ---- 2. inlet values ----------------------------------------
        cb_in  = U @ cb
        cb_out = D @ cb
        cc_in  = U @ cc
        cd_in  = U @ cd

        # ---- 3. edge coefficients -----------------------------------
        # Safe exponent helper (avoid overflow/underflow)
        def safe_exp(x):
            return np.exp(np.clip(x, -700.0, 700.0))

        # Enforce a small positive lower bound on cd_in to keep λ > -∞
        cd_in_safe = np.maximum(cd_in, 1e-12)

        B   = B_pref / q
        eB  = safe_exp(-edges.alpha_b * B * L)

        k   = k_pref
        C   = k * cd_in_safe
        lam = C / q
        g   = safe_exp(-edges.alpha_c * lam * L)

        den = k * cd_in_safe - B_pref
        den[np.abs(den) < eps] = eps           # avoid singularity

        A_e  = sid.Da * d * cb_in / (1.0 + sid.G * d)
        alpha = A_e / den

        # derivatives wrt cd_in (use cd_in_safe) -------------
        alpha_prime = -A_e * k / den**2
        g_prime     = -edges.alpha_c * (k / q) * L * g

        # outlet cc -------------------------
        cc_out = alpha * eB + (cc_in - alpha) * g
        dccout_dccu = g
        dccout_dcd  = alpha_prime * eB - alpha_prime * g + (cc_in - alpha) * g_prime

        # outlet cd  (cb term kept!) --------
        cd_out = cd_in - ((cb_in - cb_out) + (cc_in - cc_out))
        dcdout_dccu = -1.0 + dccout_dccu
        dcdout_dcd  =  1.0 + dccout_dcd      # *** fixed sign ***

        # ---- 4. residual vector --------------------------------------
        F_cc = cc * Q_in - (D.T @ (abs_Q * cc_out))
        F_cd = cd * Q_in - (D.T @ (abs_Q * cd_out))
        F_cc *= (1 - graph.in_vec)
        F_cd *= (1 - graph.in_vec)
        F = np.concatenate((F_cc, F_cd))

        # ---- 5. Jacobian blocks --------------------------------------
        Dg         = spr.diags(abs_Q * dccout_dccu)
        Ddcc_dcd   = spr.diags(abs_Q * dccout_dcd)
        Ddcd_dcc   = spr.diags(abs_Q * dcdout_dccu)
        Ddcd_dcd   = spr.diags(abs_Q * dcdout_dcd)

        J_cc_cc = spr.diags(Q_in) - D.T @ Dg @ U
        J_cc_cd =           - D.T @ Ddcc_dcd @ U
        J_cd_cc =           - D.T @ Ddcd_dcc @ U
        J_cd_cd = spr.diags(Q_in) - D.T @ Ddcd_dcd @ U

        J = spr.vstack((spr.hstack((J_cc_cc, J_cc_cd)),
                        spr.hstack((J_cd_cc, J_cd_cd))))

        J = spr.diags(1 - in_vec) @ J + spr.diags(in_vec)

        # ---- 6. Newton step ------------------------------------------
        delta = spla.spsolve(J, -F)
        lam   = 1.0
        while lam >= lam_min:
            cc_trial = cc + lam * delta[:N]
            cd_trial = cd + lam * delta[N:]

            # residual at trial point (quick re‑eval) ------------------
            cc_in_t = U @ cc_trial
            cd_in_t = U @ cd_trial
            den_t   = k * cd_in_t - B_pref
            den_t[np.abs(den_t) < eps] = eps
            lam_t   = (k * cd_in_t) / q
            g_t     = np.exp(-edges.alpha_c * lam_t * L)
            alpha_t = sid.Da * d * cb_in / (1.0 + sid.G * d) / den_t
            cc_out_t = alpha_t * eB + (cc_in_t - alpha_t) * g_t
            cd_out_t = cd_in_t - ((cb_in - cb_out) + (cc_in_t - cc_out_t))

            F_cc_t = cc_trial * Q_in - (D.T @ (abs_Q * cc_out_t))
            F_cd_t = cd_trial * Q_in - (D.T @ (abs_Q * cd_out_t))
            F_t = np.concatenate((F_cc_t, F_cd_t))

            if np.linalg.norm(F_t) < np.linalg.norm(F):
                break
            lam *= red
        else:
            lam = 1.0

        cc += lam * delta[:N]
        cd += lam * delta[N:]

        diff_cc = np.linalg.norm(lam * delta[:N]) / max(1.0, np.linalg.norm(cc))
        diff_cd = np.linalg.norm(lam * delta[N:]) / max(1.0, np.linalg.norm(cd))
        print(diff_cc, diff_cd)
        if diff_cc < tol and diff_cd < tol:
            print(f"Newton converged in {it} iterations (Δcc={diff_cc:.1e}, Δcd={diff_cd:.1e})")
            break
        if it == 50:
            cc, cd = create_vector_nr(sid, graph, inc, edges, cb)
    else:
        raise RuntimeError("Newton did not converge within the iteration limit")

    # clip outputs
    cc = np.clip(cc, 0.0, sid.cb_in)
    cd = np.clip(cd, 0.0, sid.cd_in)
    return cc, cd

def solve_precipitation_nr5_vxx(sid, inc, graph, edges, vols, cb, cc, cd,
                        tol: float = 1e-2,
                        max_iter: int = 300,
                        red: float = 0.5,
                        lam_min: float = 1e-8):
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

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

    Q_in = np.abs(inc.incidence.T) @ np.abs(edges.flow) / 2 * (1 - graph.in_vec + graph.out_vec) + graph.in_vec

    in_vec = np.concatenate((graph.in_vec, graph.in_vec))

    c_inc = 1 * (inc.incidence.T @ (spr.diags(edges.flow) \
            @ inc.incidence > 0) != 0)


    d  = edges.diams
    L  = edges.lens
    q  = abs_Q                                 # magnitude only
    print(np.sum(q == 0))

    B_pref = edges.alpha_b * sid.Da * d / (1.0 + sid.G * d)
    k = sid.Da * sid.K * d / (sid.Kp * (1.0 + sid.G * sid.K * d))

    tau0 = 1e-2
    diff_cc = 0
   
    # ------------------------------------------------------------------
    for it in range(1, max_iter + 1):

        cb_in  = U @ cb
        cb_out = D @ cb
        cc_in  = U @ cc
        cd_in  = U @ cd

        def safe_exp(x):
            return np.exp(np.clip(x, -700.0, 700.0))

        B        = np.zeros_like(abs_Q)
        lam      = np.zeros_like(abs_Q)
        g        = np.zeros_like(abs_Q)
        cc_out        = np.empty_like(cd_in)
        dccout_dccu   = np.empty_like(cd_in)
        dccout_dcd    = np.empty_like(cd_in)

        B[mask_flow]   = B_pref[mask_flow] / abs_Q[mask_flow]
        lam[mask_flow] = (k[mask_flow] * cd_in[mask_flow]) / abs_Q[mask_flow]
        g[mask_flow]  = safe_exp(-edges.alpha_c * lam[mask_flow] * L[mask_flow])

        eB  = safe_exp(-B * L)

        den = k * cd_in - B_pref
        tau  = np.abs(den) / (np.abs(k*cd_in) + np.abs(B_pref) + 1.0)
        w    = tau / (tau + tau0)
        mask_gen = tau > eps_rel
        print(f'den zero: {np.min(np.abs(den))}')

        A_e  = edges.alpha_b * sid.Da * d * cb_in / (1.0 + sid.G * d)

        
        alpha = A_e[mask_gen] / den[mask_gen]
        alpha_prime = -A_e * k / den**2
        g_prime     = -edges.alpha_c * (k / q) * L * g
        g_prime[mask_zero] = 0
        g_gen          = safe_exp(-edges.alpha_c * lam[mask_gen] * L[mask_gen])
        cc_out[mask_gen]      = alpha * eB[mask_gen] + (cc_in[mask_gen] - alpha) * g_gen
        dccout_dccu[mask_gen] = g_gen
        dccout_dcd[mask_gen]  = (alpha_prime[mask_gen] * eB[mask_gen]
                         - alpha_prime[mask_gen] * g_gen
                         + (cc_in[mask_gen] - alpha) * g_prime[mask_gen])
        
        mask_res = ~mask_gen
        g_res          = safe_exp(-B[mask_res] * L[mask_res])
        cc_out[mask_res]      = (A_e[mask_res]/q[mask_res] * L[mask_res] + cc_in[mask_res]) * g_res
        dccout_dccu[mask_res] = g_res
        dccout_dcd[mask_res]  = 0.0
        #cc_out = alpha * eB + (cc_in - alpha) * g
        # resonant part
        #dccout_dccu = g

        #dccout_dcd  = alpha_prime * eB - alpha_prime * g + (cc_in - alpha) * g_prime
        
        cd_out = cd_in - ((cb_in - cb_out) + (cc_in - cc_out))
        cc_out[mask_zero] = cc_in[mask_zero]
        cd_out[mask_zero] = cd_in[mask_zero]
        dcdout_dccu = -1.0 + dccout_dccu
        dcdout_dcd  =  1.0 + dccout_dcd      # *** fixed sign ***

        dccout_dccu[mask_zero] = 1.0
        dccout_dcd[mask_zero] = 0.0
        dcdout_dccu[mask_zero] = 0.0
        dcdout_dcd[mask_zero] = 1.0

        # ---- 4. residual vector --------------------------------------
        F_cc = cc * Q_in - (D.T @ (abs_Q * cc_out))
        F_cd = cd * Q_in - (D.T @ (abs_Q * cd_out))
        F_cc *= (1 - graph.in_vec)
        F_cd *= (1 - graph.in_vec)
        F = np.concatenate((F_cc, F_cd))

        # ---- 5. Jacobian blocks --------------------------------------
        Dg         = spr.diags(abs_Q * dccout_dccu)
        Ddcc_dcd   = spr.diags(abs_Q * dccout_dcd)
        Ddcd_dcc   = spr.diags(abs_Q * dcdout_dccu)
        Ddcd_dcd   = spr.diags(abs_Q * dcdout_dcd)

        # J_cc_cc_0 = np.abs(inc.incidence.T) @ Dg @ np.abs(inc.incidence)
        # J_cc_cc_0 = c_inc.multiply(J_cc_cc_0)
        # diag_old = J_cc_cc_0.diagonal()
        # J_cc_cc_0 -= spr.diags(diag_old)

        # J_cc_cd_0 = np.abs(inc.incidence.T) @ Ddcc_dcd @ np.abs(inc.incidence)
        # J_cc_cd_0 = c_inc.multiply(J_cc_cd_0)
        # diag_old = J_cc_cd_0.diagonal()
        # J_cc_cd_0 -= spr.diags(diag_old)  

        # J_cd_cc_0 = np.abs(inc.incidence.T) @ Ddcd_dcc @ np.abs(inc.incidence)
        # J_cd_cc_0 = c_inc.multiply(J_cd_cc_0)
        # diag_old = J_cd_cc_0.diagonal()
        # J_cd_cc_0 -= spr.diags(diag_old)

        # J_cd_cd_0 = np.abs(inc.incidence.T) @ Ddcd_dcd @ np.abs(inc.incidence)
        # J_cd_cd_0 = c_inc.multiply(J_cd_cd_0)
        # diag_old = J_cd_cd_0.diagonal()
        # J_cd_cd_0 -= spr.diags(diag_old)  

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

        J = spr.diags(1 - in_vec) @ J + spr.diags(in_vec)
        J = spr.diags(1 - 1 * (F == 0)) @ J + spr.diags(1 * (F == 0))

        # ---- 6. Newton step ------------------------------------------
        delta = solve_equation(J, -F)
        g     = J.T @ F
        if np.dot(g, delta) >= 0:        # not descent → Levenberg
            print("Levenberg")
            mu = 1e-4 * spla.norm(J, np.inf)
            delta = solve_equation(J.T @ J + mu*spr.diags(np.ones(2 * sid.nsq)), -g)

            if np.dot(g, delta) >= 0:    # still not descent → gradient
                print('Gradient')    
                delta = -g

        # Armijo back-tracking
        lam = 1.0
        rho = 1e-2
        phi0 = 0.5*np.dot(F, F)
        dphi0 = np.dot(g, delta)
        lams   = 1.0
        while lams >= lam_min:
            cc_trial = cc + lams * delta[:N]
            cd_trial = cd + lams * delta[N:]

            # residual at trial point (quick re‑eval) ------------------
            cc_in_t = U @ cc_trial
            cd_in_t = U @ cd_trial
            den_t   = k * cd_in_t - B_pref
            tau  = np.abs(den_t) / (np.abs(k*cd_in_t) + np.abs(B_pref) + 1.0)
            mask_gen = tau > eps_rel
            lam_t   = (k * cd_in_t) / q
            g_t     = safe_exp(-edges.alpha_c * lam_t * L)
            alpha_t = A_e / den_t
            g_gen_t          = safe_exp(-edges.alpha_c * lam_t[mask_gen] * L[mask_gen])
            cc_out_t = np.empty_like(cd_in_t)
            cc_out_t[mask_gen]      = alpha_t[mask_gen] * eB[mask_gen] + (cc_in_t[mask_gen] - alpha_t[mask_gen]) * g_gen_t
            #cc_out_t = alpha_t * eB + (cc_in_t - alpha_t) * g_t
            mask_res = ~mask_gen
            g_res_t          = safe_exp(-B[mask_res] * L[mask_res])
            cc_out[mask_res]      = (A_e[mask_res]/q[mask_res] * L[mask_res] + cc_in_t[mask_res]) * g_res_t
            cd_out_t = cd_in_t - ((cb_in - cb_out) + (cc_in_t - cc_out_t))
            cc_out_t[mask_zero] = cc_in_t[mask_zero]
            cd_out_t[mask_zero] = cd_in_t[mask_zero]
            F_cc_t = cc_trial * Q_in - (D.T @ (abs_Q * cc_out_t))
            F_cd_t = cd_trial * Q_in - (D.T @ (abs_Q * cd_out_t))
            F_cc_t *= (1 - graph.in_vec)
            F_cd_t *= (1 - graph.in_vec)
            F_t = np.concatenate((F_cc_t, F_cd_t))
            phi_t  = 0.5 * np.dot(F_t, F_t)
            #
            #print(np.linalg.norm(F_t))
            #print(np.linalg.norm(F))
            if phi_t <= phi0 + rho * lams * dphi0:
                break
            lams *= red

            if lams < 1e-12:
                raise RuntimeError("Line search failed even in steepest descent")
        defect = np.linalg.norm(J @ delta + F) / np.linalg.norm(F)
        
        eps   = 1e-6
        cc_trial = cc + eps * delta[:N]
        cd_trial = cd + eps * delta[N:]

        # residual at trial point (quick re‑eval) ------------------
        cc_in_t = U @ cc_trial
        cd_in_t = U @ cd_trial
        den_t   = k * cd_in_t - B_pref
        lam_t   = (k * cd_in_t) / q
        g_t     = safe_exp(-edges.alpha_c * lam_t * L)
        alpha_t = A_e / den_t
        cc_out_t = alpha_t * eB + (cc_in_t - alpha_t) * g_t
        cd_out_t = cd_in_t - ((cb_in - cb_out) + (cc_in_t - cc_out_t))
        cc_out_t[mask_zero] = cc_in_t[mask_zero]
        cd_out_t[mask_zero] = cd_in_t[mask_zero]
        F_cc_t = cc_trial * Q_in - (D.T @ (abs_Q * cc_out_t))
        F_cd_t = cd_trial * Q_in - (D.T @ (abs_Q * cd_out_t))
        F_cc_t *= (1 - graph.in_vec)
        F_cd_t *= (1 - graph.in_vec)
        F_t = np.concatenate((F_cc_t, F_cd_t))

        
        sec   = (F_t - F)/eps
        mis   = np.linalg.norm(sec - J@delta) / np.linalg.norm(sec)
        print(f"mis: {mis}")
        print(f"λ={lams:.5f}   phi_t={phi_t:.3e}   F={np.linalg.norm(F_t):.3e}")
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
        
        #cc_prev = cc.copy()
        diff_cc = np.linalg.norm(delta[:N]) #/ max(1.0, np.linalg.norm(cc))
        diff_cd = np.linalg.norm(delta[N:]) #/ max(1.0, np.linalg.norm(cd))
        # if diff_cc_prev - diff_cc < 1e-2:
        #     lams /= 2
        cc += lams * delta[:N]
        cd += lams * delta[N:]
        print(diff_cc, diff_cd)
        print(f'cc: {np.min(cc)}, {np.max(cc)}, cd: {np.min(cd)}, {np.max(cd)}')
        #print(np.linalg.norm(F))
        if diff_cc < sid.it_alpha_th and diff_cd < sid.it_alpha_th:
            print(f"Newton converged in {it} iterations (Δcc={diff_cc:.1e}, Δcd={diff_cd:.1e})")
            break
        if phi_t < sid.it_alpha_th:
            print(f"Newton converged in {it} iterations (Δcc={diff_cc:.1e}, Δcd={diff_cd:.1e})")
            break
        if it == 50 or np.isnan(np.linalg.norm(F_t)):
            print("Newton: restarting iterations")
            cc, cd = create_vector_nr(sid, graph, inc, edges, cb)
        # if np.sum(edges.alpha_b == 0) > 0:
        #     print('alpha_b == 0')
        # if np.sum(cc < -0.05) > 0:
        #     print('cc < 0')
        #     raise ValueError
        cc = np.clip(cc, 0, None)
        cd = np.clip(cd, 0, None)
    else:
        raise RuntimeError("Newton did not converge within the iteration limit")

    # clip outputs

    cc = np.clip(cc, 0.0, sid.cb_in)
    cd = np.clip(cd, 0.0, sid.cd_in)
    
        
    return cc, cd

def solve_precipitation_nr7_vxx(sid: SimInputData, inc: Incidence, graph: Graph, \
    edges: Edges, vols: Volumes, cb, cc, cd) -> np.ndarray:
    """ Calculate B concentration.
    """
    # find incidence for cb (only upstream flow matters)
    edges.alpha_c = 1 #* ((vols.triangles @ (1 * (vols.vol_e + vols.vol_a < vols.vol_max))) > 0)
    
    
    # Upstream‑selector matrix  U  (|E|×|N|, CSR, entries 0/1)
    U = 1 * (spr.diags(edges.flow) @ inc.incidence > 0)

    # Downstream‑selector matrix  D
    D = 1 * (spr.diags(edges.flow) @ inc.incidence < 0)
    
    c_inc = 1 * (inc.incidence.T @ (spr.diags(edges.flow) \
        @ inc.incidence > 0) != 0)
    diag = -np.abs(inc.incidence.T) @ np.abs(edges.flow) / 2
    diag = diag * (1 - graph.in_vec + graph.out_vec) + graph.in_vec
    diag += 1 * (diag == 0)
    
    if np.sum(cd) < 1.1 * np.sum(sid.cd_in * edges.inlet) or (np.sum(1 - ((cd == 1) + (cd == 0))) == 0):
        cd = sid.cd_in * np.ones(sid.nsq)
        cc = sid.cb_in - cb + sid.cc_in
        print('correcting initial cc, cd')

    cc_prev = np.zeros(sid.nsq)
    cd_prev = np.zeros(sid.nsq)
    it = 0
    diff_c = np.linalg.norm(cc - cc_prev)
    diff_d = np.linalg.norm(cd - cd_prev)
    while diff_c > sid.c_th or diff_d > sid.c_th:
        it += 1
        cb_in = U @ np.abs(cb)
        cc_in = U @ np.abs(cc)
        cd_in = U @ np.abs(cd)
        ksi = cd_in * sid.K / (1 + sid.K * sid.G * edges.diams) - sid.Kp / (1 + sid.G * edges.diams)

        exp_p = np.exp(-edges.alpha_c * sid.Da * sid.K / (1 + sid.G * sid.K * edges.diams) \
            * cd_in / sid.Kp * edges.diams * edges.lens / np.abs(edges.flow))
        exp_p = np.array(np.ma.fix_invalid(exp_p, fill_value = 0))
        exp_d = np.exp(-edges.alpha_b * sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow))
        exp_d = np.array(np.ma.fix_invalid(exp_d, fill_value = 0))

        qc2 = -edges.alpha_b * np.abs(edges.flow) / (1 + sid.G * edges.diams) * sid.Kp * (exp_p - exp_d) / ksi
        qc2 = np.array(np.ma.fix_invalid(qc2, fill_value = 0))
        cc_matrix = D.T @ spr.diags(qc2) @ U
        diag_old = cc_matrix.diagonal()
        cc_matrix += spr.diags(diag - diag_old)
        f_cc2 = cc_matrix @ cb
        

        qc1 = np.abs(edges.flow) * exp_p
        cc_matrix = D.T @ spr.diags(qc1) @ U
        diag_old = cc_matrix.diagonal()
        cc_matrix += spr.diags(diag - diag_old)
        f_cc1 = cc_matrix @ cc
        f_cc = (1 - graph.in_vec) * (f_cc1 + f_cc2)

        dq_cc_cd = edges.alpha_b * cb_in / (1 + sid.G * edges.diams) / (1 + sid.G * edges.diams * sid.K) * sid.K / ksi ** 2 \
            * (sid.Da * edges.diams * edges.lens * ksi * exp_p + np.abs(edges.flow) * sid.Kp * (exp_p - exp_d))
        dq_cc_cd = np.array(np.ma.fix_invalid(dq_cc_cd, fill_value = 0))
        dq_cc_cd -= cc_in * exp_p * sid.K * edges.diams * edges.lens * sid.Da / sid.Kp / (1 + sid.G * edges.diams * sid.K)
        dq_cc_cd = np.array(np.ma.fix_invalid(dq_cc_cd, fill_value = 0))
        dcc_cd_matrix = D.T @ spr.diags(dq_cc_cd) @ U

        diag_old = dcc_cd_matrix.diagonal()
        dcc_cd_matrix -= spr.diags(diag_old)
        
        dq_cd_cc = np.abs(edges.flow) * (exp_p - 1)
        dq_cd_cc_matrix = D.T @ spr.diags(dq_cd_cc) @ U
        diag_old = dq_cd_cc_matrix.diagonal()
        dq_cd_cc_matrix -= spr.diags(diag_old)

        qc = np.abs(edges.flow) * (1 - exp_d)
        cb_matrix = D.T @ spr.diags(qc) @ U
        diag_old = cb_matrix.diagonal()
        cb_matrix -= spr.diags(diag_old)

        cd_mix = cd
        cd_matrix = D.T @ spr.diags(np.abs(edges.flow)) @ U
        diag_old = cd_matrix.diagonal()
        cd_matrix += spr.diags(diag - diag_old)
        f_cd = (1 - graph.in_vec) * (cd_matrix @ cd_mix + dq_cd_cc_matrix @ cc + f_cc2 - cb_matrix @ cb)
        #np.savetxt('f_cd.txt', f_cd)
        #np.savetxt('f_cc.txt', f_cc)
        dq_cd_cd = cd_matrix + dcc_cd_matrix

        dc_matrix = spr.vstack([spr.hstack([cc_matrix, dcc_cd_matrix]), spr.hstack([dq_cd_cc_matrix, dq_cd_cd])])
        f = np.concatenate((f_cc, f_cd))
        # if it > 5:
        #     c_zero = 1 * (np.concatenate((cc, cd)) <= 0)
        #     #c_zero = 1 * (np.concatenate((cc, cd)) < 0)
        #     f = f * (1 - c_zero)
        #dc_matrix = spr.diags(1 - 1 * (f == 0)) @ dc_matrix + spr.diags(1 * (f == 0))
        in_vec = np.concatenate((graph.in_vec, graph.in_vec))
        dc_matrix = spr.diags(1 - in_vec) @ dc_matrix + spr.diags(in_vec)
        delta_c = solve_equation(dc_matrix, -f)
        lam      = 1.0
        lam_min  = 1e-1            # allow very small steps before giving up
        red      = 0.5

        # while True:
        #     f_trial = f + lam * (-f)            # J·delta = -f
        #     if np.linalg.norm(f_trial) < np.linalg.norm(f):
        #         break                           # success
        #     lam *= red
        #     if lam < lam_min:
        #         lam = 1
        cc_prev = cc.copy()
        cd_prev = cd.copy()
        cc += lam * delta_c[:sid.nsq]
        cd += lam * delta_c[sid.nsq:]
        # cc = np.clip(cc, 0, sid.cb_in)
        # cd = np.clip(cd, 0, sid.cd_in)
        diff_c_new = np.linalg.norm(cc - cc_prev)
        diff_d_new = np.linalg.norm(cd - cd_prev)
        if np.abs(diff_c_new - diff_c) < 1e-3:
            cc -= 0.5 * lam * delta_c[:sid.nsq]
            cd -= 0.5 * lam * delta_c[sid.nsq:]
            # cc = np.clip(cc, 0, sid.cb_in)
            # cd = np.clip(cd, 0, sid.cd_in)
        diff_c = np.linalg.norm(cc - cc_prev)
        diff_d = np.linalg.norm(cd - cd_prev)
        print(np.linalg.norm(cc - cc_prev), np.linalg.norm(cd - cd_prev))
        if it == 30:
            cc, cd = create_vector_nr(sid, graph, inc, edges, cb)
            print('restarting iterations')
        elif it > 300:
            cc = -1 * np.ones_like(cc)
            cd = -1 * np.ones_like(cc)
            print('NR didnt converge')
            return cc, cd
    cc = np.clip(cc, 0, sid.cb_in)
    cd = np.clip(cd, 0, sid.cd_in)
    return cc, cd

def solve_precipitation_nr8_vxx(sid, inc, graph, edges, vols, cb, cc, cd,
                        tol: float = 1e-2,
                        max_iter: int = 300,
                        red: float = 0.5,
                        lam_min: float = 1e-8):
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

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

    Q_in = np.abs(inc.incidence.T) @ np.abs(edges.flow) / 2 * (1 - graph.in_vec + graph.out_vec) + graph.in_vec

    in_vec = np.concatenate((graph.in_vec, graph.in_vec))

    c_inc = 1 * (inc.incidence.T @ (spr.diags(edges.flow) \
            @ inc.incidence > 0) != 0)


    d  = edges.diams
    L  = edges.lens
    q  = abs_Q                                 # magnitude only
    print(np.sum(q == 0))

    B_pref = edges.alpha_b * sid.Da * d / (1.0 + sid.G * d)
    k = sid.Da * sid.K * d / (sid.Kp * (1.0 + sid.G * sid.K * d))

    tau0 = 1e-12
    diff_cc = 0
    y = np.log(np.maximum(cc, 1e-10))
    z = np.log(np.maximum(cd, 1e-10))
    # ------------------------------------------------------------------
    for it in range(1, max_iter + 1):
                  # new unknown
        cc = np.exp(y)
        cd = np.exp(z)          # everywhere in the edge model
        cb_in  = U @ cb
        cb_out = D @ cb
        cc_in  = U @ cc
        cd_in  = U @ cd

        def safe_exp(x):
            return np.exp(np.clip(x, -700.0, 700.0))

        B        = np.zeros_like(abs_Q)
        lam      = np.zeros_like(abs_Q)
        g        = np.zeros_like(abs_Q)

        B[mask_flow]   = B_pref[mask_flow] / abs_Q[mask_flow]
        lam[mask_flow] = (k[mask_flow] * cd_in[mask_flow]) / abs_Q[mask_flow]
        g[mask_flow]  = safe_exp(-edges.alpha_c * lam[mask_flow] * L[mask_flow])

        eB  = safe_exp(-B * L)

        den = k * cd_in - B_pref
        print(f'den zero: {np.min(np.abs(den))}')

        A_e  = edges.alpha_b * sid.Da * d * cb_in / (1.0 + sid.G * d)

        
        alpha = A_e / den
        
        
        cc_out = alpha * eB + (cc_in - alpha) * g
        # resonant part
        dccout_dccu = g
        alpha_prime = -A_e * k / den**2
        g_prime     = -edges.alpha_c * (k / q) * L * g
        g_prime[mask_zero] = 0
        dccout_dcd  = alpha_prime * eB - alpha_prime * g + (cc_in - alpha) * g_prime
        
        cd_out = cd_in - ((cb_in - cb_out) + (cc_in - cc_out))
        cc_out[mask_zero] = cc_in[mask_zero]
        cd_out[mask_zero] = cd_in[mask_zero]
        dcdout_dccu = -1.0 + dccout_dccu
        dcdout_dcd  =  1.0 + dccout_dcd      # *** fixed sign ***

        dccout_dccu[mask_zero] = 1.0
        dccout_dcd[mask_zero] = 0.0
        dcdout_dccu[mask_zero] = 0.0
        dcdout_dcd[mask_zero] = 1.0
        dccout_dy = dccout_dccu * cc_in
        dcdout_dy = dcdout_dccu * cc_in
        dccout_dz = dccout_dcd * cd_in
        dcdout_dz = dcdout_dcd * cd_in

        # ---- 4. residual vector --------------------------------------
        F_cc = cc * Q_in - (D.T @ (abs_Q * cc_out))
        F_cd = cd * Q_in - (D.T @ (abs_Q * cd_out))
        F_cc *= (1 - graph.in_vec)
        F_cd *= (1 - graph.in_vec)
        F = np.concatenate((F_cc, F_cd))

        # ---- 5. Jacobian blocks --------------------------------------


        Ddcc_dy = spr.diags(abs_Q * dccout_dy)
        Ddcd_dy = spr.diags(abs_Q * dcdout_dy)
        Ddcc_dz = spr.diags(abs_Q * dccout_dz)
        Ddcd_dz = spr.diags(abs_Q * dcdout_dz)
        J_cc_y = spr.diags(Q_in * cc) - D.T @ Ddcc_dy @ U
        J_cd_y  = - D.T @ Ddcd_dy @ U
        J_cc_z  = - D.T @ Ddcc_dz @ U
        J_z_z   = spr.diags(Q_in * cd) - D.T @ Ddcd_dz @ U

        J = spr.vstack((spr.hstack((J_cc_y, J_cc_z)),
                        spr.hstack((J_cd_y, J_z_z))))

        J = spr.diags(1 - in_vec) @ J + spr.diags(in_vec)
        #J = spr.diags(1 - 1 * (F == 0)) @ J + spr.diags(1 * (F == 0))

        # ---- 6. Newton step ------------------------------------------
        delta = solve_equation(J, -F)
        delta = np.clip(delta, -1.0, 1.0)

        g     = J.T @ F
        if np.dot(g, delta) >= 0:        # not descent → Levenberg
            print("Levenberg")
            mu = 1e-4 * spla.norm(J, np.inf)
            delta = solve_equation(J.T @ J + mu*spr.diags(np.ones(2 * sid.nsq)), -g)

            if np.dot(g, delta) >= 0:    # still not descent → gradient
                print('Gradient')    
                delta = -g

        #Armijo back-tracking
        lam = 1.0
        rho = 1e-2
        phi0 = 0.5*np.dot(F, F)
        dphi0 = np.dot(g, delta)
        lams   = 1.0
        while lams >= lam_min:
            y_trial = y + lams * delta[:N]
            #cc_trial = cc + lams * delta[:N]
            z_trial = z + lams * delta[N:]
            cc_trial = np.exp(y_trial)
            cd_trial = np.exp(z_trial)
            #cd_trial = np.clip(cd_trial, 1e-15, None)

            # residual at trial point (quick re‑eval) ------------------
            cc_in_t = U @ cc_trial
            cd_in_t = U @ cd_trial
            den_t   = k * cd_in_t - B_pref
            lam_t   = (k * cd_in_t) / q
            g_t     = safe_exp(-edges.alpha_c * lam_t * L)
            alpha_t = A_e / den_t
            cc_out_t = alpha_t * eB + (cc_in_t - alpha_t) * g_t
            cd_out_t = cd_in_t - ((cb_in - cb_out) + (cc_in_t - cc_out_t))
            cc_out_t[mask_zero] = cc_in_t[mask_zero]
            cd_out_t[mask_zero] = cd_in_t[mask_zero]
            F_cc_t = cc_trial * Q_in - (D.T @ (abs_Q * cc_out_t))
            F_cd_t = cd_trial * Q_in - (D.T @ (abs_Q * cd_out_t))
            F_cc_t *= (1 - graph.in_vec)
            F_cd_t *= (1 - graph.in_vec)
            F_t = np.concatenate((F_cc_t, F_cd_t))
            phi_t  = 0.5 * np.dot(F_t, F_t)
            #
            #print(np.linalg.norm(F_t))
            #print(np.linalg.norm(F))
            if phi_t <= phi0 + rho * lams * dphi0:
                break
            lams *= red

            if lams < 1e-12:
                raise RuntimeError("Line search failed even in steepest descent")
 
        diff_cc = np.linalg.norm(delta[:N]) #/ max(1.0, np.linalg.norm(cc))
        diff_cd = np.linalg.norm(delta[N:]) #/ max(1.0, np.linalg.norm(cd))

        y += lams * delta[:N]
        z += lams * delta[N:]
        cc = np.exp(y)
        cd = np.exp(z)
        print(diff_cc, diff_cd)
        print(f'cc: {np.min(cc)}, {np.max(cc)}, cd: {np.min(cd)}, {np.max(cd)}')
        print(np.linalg.norm(F_t))
        if diff_cc < sid.it_alpha_th and diff_cd < sid.it_alpha_th:
            print(f"Newton converged in {it} iterations (Δcc={diff_cc:.1e}, Δcd={diff_cd:.1e})")
            break
        if np.linalg.norm(phi_t) < sid.it_alpha_th:
            print(f"Newton converged in {it} iterations (Δcc={diff_cc:.1e}, Δcd={diff_cd:.1e})")
            break
        if it == 50 or np.isnan(np.linalg.norm(F_t)):
            print("Newton: restarting iterations")
            cc, cd = create_vector_nr(sid, graph, inc, edges, cb)

    else:
        raise RuntimeError("Newton did not converge within the iteration limit")

    # clip outputs

    cc = np.clip(cc, 0.0, sid.cb_in)
    cd = np.clip(cd, 0.0, sid.cd_in)
    
        
    return cc, cd

def solve_precipitation_nr9_vxx(sid, inc, graph, edges, vols, cb, cc, cd,
                        tol: float = 1e-2,
                        max_iter: int = 500,
                        red: float = 0.5,
                        lam_min: float = 1e-10):
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

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

    Q_in = np.abs(inc.incidence.T) @ np.abs(edges.flow) / 2 * (1 - graph.in_vec + graph.out_vec) + graph.in_vec

    in_vec = np.concatenate((graph.in_vec, graph.in_vec))

    c_inc = 1 * (inc.incidence.T @ (spr.diags(edges.flow) \
            @ inc.incidence > 0) != 0)


    d  = edges.diams
    L  = edges.lens
    q  = abs_Q                                 # magnitude only
    print(np.sum(q == 0))

    B_pref = edges.alpha_b * sid.Da * d / (1.0 + sid.G * d)
    k = sid.Da * sid.K * d / (sid.Kp * (1.0 + sid.G * sid.K * d))

    tau0 = 1e-12
    diff_cc = 0
    z = np.log(cd)
    for it in range(1, max_iter + 1):

        cb_in  = U @ cb
        cb_out = D @ cb
        cc_in  = U @ cc
        cd_in  = U @ cd

        def safe_exp(x):
            return np.exp(np.clip(x, -700.0, 700.0))

        B   = B_pref / q
        eB  = safe_exp(-B * L)

        
        C   = k * cd_in
        lam = C / q
        g   = safe_exp(-edges.alpha_c * lam * L)

        den = k * cd_in - B_pref
        #tau  = np.abs(den) / (np.abs(k*cd_in) + np.abs(B_pref) + 1.0)


        A_e  = edges.alpha_b * sid.Da * d * cb_in / (1.0 + sid.G * d)

        tau  = abs(den) / (abs(k*cd_in) + abs(B_pref) + 1.0)
        w    = tau / (tau + tau0)           # 0 ≤ w ≤ 1 ,  tau0 ≈ 1e-6
        # generic part
        alpha = A_e / den
        g_gen = safe_exp(-edges.alpha_c * lam * L)
        cc_gen = alpha * eB + (cc_in - alpha) * g_gen
        # resonant part
        g_res  = safe_exp(-B * L)
        cc_res = (A_e/q * L + cc_in) * g_res
        cc_res = np.ma.fix_invalid(cc_res, fill_value = 0)
        # blended value
        cc_out = w * cc_gen + (1-w) * cc_res
        dccout_dccu = w * g_gen + (1-w) * g_res
        alpha_prime = -A_e * k / den**2
        g_prime     = -edges.alpha_c * (k / q) * L * g
        g_prime = np.ma.fix_invalid(g_prime, fill_value = 0)
        dccout_dcd  = w * (alpha_prime * eB - alpha_prime * g + (cc_in - alpha) * g_prime)
        
        cd_out = cd_in - ((cb_in - cb_out) + (cc_in - cc_out))
        dcdout_dccu = -1.0 + dccout_dccu
        dcdout_dcd  =  1.0 + dccout_dcd      # *** fixed sign ***

        # ---- 4. residual vector --------------------------------------
        F_cc = cc * Q_in - (D.T @ (abs_Q * cc_out))
        F_cd = cd * Q_in - (D.T @ (abs_Q * cd_out))
        F_cc *= (1 - graph.in_vec)
        F_cd *= (1 - graph.in_vec)
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

        J = spr.diags(1 - in_vec) @ J + spr.diags(in_vec)
        J = spr.diags(1 - 1 * (F == 0)) @ J + spr.diags(1 * (F == 0))

        # ---- 6. Newton step ------------------------------------------
        delta = solve_equation(J, -F)
        g     = J.T @ F
        if np.dot(g, delta) >= 0:        # not descent → Levenberg
            print("Levenberg")
            mu = 1e-4 * spla.norm(J, np.inf)
            delta = solve_equation(J.T @ J + mu*spr.diags(np.ones(2 * sid.nsq)), -g)

            if np.dot(g, delta) >= 0:    # still not descent → gradient
                print('Gradient')    
                delta = -g

        # Armijo back-tracking
        lam = 1.0
        rho = 1e-4
        phi0 = 0.5*np.dot(F, F)
        dphi0 = np.dot(g, delta)
        lams   = 1.0
        while lams >= lam_min:
            cc_trial = cc + lams * delta[:N]
            cd_trial = cd + lams * delta[N:]

            cc_trial = np.clip(cc_trial, 0.0, sid.cb_in)
            cd_trial = np.clip(cd_trial, 0.0, sid.cd_in)
            # residual at trial point (quick re‑eval) ------------------
            cc_in_t = U @ cc_trial
            cd_in_t = U @ cd_trial
            den_t   = k * cd_in_t - B_pref
            tau  = abs(den_t) / (abs(k*cd_in_t) + abs(B_pref) + 1.0)
            w    = tau / (tau + tau0)
            lam_t   = (k * cd_in_t) / q
            g_t     = safe_exp(-edges.alpha_c * lam_t * L)
            alpha_t = A_e / den_t
            g_gen_t = safe_exp(-edges.alpha_c * lam_t * L)
            cc_gen_t = alpha_t * eB + (cc_in_t - alpha_t) * g_gen_t
            g_res_t  = safe_exp(-B * L)
            cc_res_t = (A_e/q * L + cc_in_t) * g_res_t
            cc_res_t = np.ma.fix_invalid(cc_res_t, fill_value = 0)
            cc_out_t = w * cc_gen_t + (1-w) * cc_res_t
            #cc_out_t = alpha_t * eB + (cc_in_t - alpha_t) * g_t
            cd_out_t = cd_in_t - ((cb_in - cb_out) + (cc_in_t - cc_out_t))
            F_cc_t = cc_trial * Q_in - (D.T @ (abs_Q * cc_out_t))
            F_cd_t = cd_trial * Q_in - (D.T @ (abs_Q * cd_out_t))
            F_cc_t *= (1 - graph.in_vec)
            F_cd_t *= (1 - graph.in_vec)
            F_t = np.concatenate((F_cc_t, F_cd_t))
            phi_t  = 0.5 * np.dot(F_t, F_t)
            
            #print(np.linalg.norm(F_t))
            #print(np.linalg.norm(F))
            if phi_t <= phi0 + rho * lams * dphi0:
                break
            lams *= red
        else:
            lams = 0.01#-0.01
        # if lams < lam_min:
        #     raise RuntimeError("Line search failed even in steepest descent")
        print(f"λ={lams:.5f}   phi_t={phi_t:.3e}   sumF={F_t.sum():.3e}")
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
        cc = np.clip(cc, 0.0, sid.cb_in)
        cd = np.clip(cd, 0.0, sid.cd_in)
        if phi_t < sid.it_alpha_th:
            print(f"Newton converged in {it} iterations (Δcc={diff_cc:.1e}, Δcd={diff_cd:.1e})")
            break
        if np.linalg.norm(cc - cc_prev) < sid.it_alpha_th and np.linalg.norm(cd - cd_prev) < sid.it_alpha_th:
            print(f"Newton converged in {it} iterations (Δcc={diff_cc:.1e}, Δcd={diff_cd:.1e})")
            break
        if it == 50 or np.isnan(np.linalg.norm(F_t)):
            print("Newton: restarting iterations")
            cc, cd = create_vector_nr(sid, graph, inc, edges, cb)
        # if np.sum(edges.alpha_b == 0) > 0:
        #     print('alpha_b == 0')
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