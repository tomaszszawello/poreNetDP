
from config import SimInputData
from incidence import Incidence
from network import Graph, Edges
from volumes import Volumes
from utils import solve_equation

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as spr

from scipy.optimize import root, newton_krylov
from scipy.sparse.linalg import LinearOperator

def create_vector(sid: SimInputData, graph: Graph):
    """ Create vector result for inc.incidence concentration calculation.
    """
    return np.concatenate([sid.cb_0 * graph.in_vec, np.zeros(2 * sid.ne)])

def solve_diffusion(sid: SimInputData, inc: Incidence, graph: Graph, edges: Edges, cb_vector):
    exp_plus = spr.diags(edges.lens)
    exp_minus = spr.diags(np.ones(sid.ne))
 
    upstream = 1 * (inc.incidence > 0) #+ 1 * (spr.diags(1 * (edges.flow == 0)) @ inc.incidence > 0)
    downstream = 1 * (inc.incidence < 0)
    
    flux_a = spr.diags(np.abs(edges.diams ** 2)) @ downstream - spr.diags(np.abs(edges.diams ** 2)) @ upstream
    
    flux_a_in = flux_a.T.multiply((1 - graph.in_vec - graph.out_vec)[:, np.newaxis]) #+ ((exp_plus * spr.diags(lam_plus_val)) @ downstream).T.multiply((graph.out_vec * (1 - cb_pe_fix))[:, np.newaxis])
    
    flux_b_in = 0 * flux_a_in 

    #flow_fix_pe = -downstream.T @ np.abs(edges.flow)
    flow_fix_pe = graph.in_vec + graph.out_vec #flow_fix_pe * (1 - graph.in_vec - graph.out_vec) + graph.in_vec + graph.out_vec
    
    cb_matrix = spr.vstack([spr.hstack([spr.diags(flow_fix_pe), flux_a_in, flux_b_in]), \
                    spr.hstack([-downstream, exp_plus, exp_minus]), \
                    spr.hstack([-upstream, spr.diags(np.zeros(sid.ne)), spr.diags(np.ones(sid.ne))]) \
                    ])
    cb_matrix = spr.diags(1 - inc.merge_vec) @ cb_matrix @ spr.diags(1 - inc.merge_vec) + spr.diags(inc.merge_vec)
    
    diag = cb_matrix.diagonal()
    diag_old = diag.copy()
    for node in np.where(np.abs(cb_matrix).sum(axis = 1) == 0)[0]:
        diag[node] = 1
        print(node)
    cb_matrix += spr.diags(diag - diag_old)
    res = solve_equation(cb_matrix, cb_vector)
    cb = res[:sid.nsq]
    edges.A = res[sid.nsq:sid.nsq+sid.ne]
    edges.inc.incidence = res[sid.nsq+sid.ne:]
    #np.savetxt('cb.txt', cb)
    #np.savetxt('cbm.txt', cb_matrix.toarray())
    #np.savetxt('lam.txt', lam_plus_zero)
    print(np.max(cb), np.min(cb))
    print(cb)
    return cb

def find_flow(sid, inc, edges, cosm):
    upstream = 1 * (inc.incidence < 0)
    edges.flow = edges.diams ** 2 / edges.lens * (inc.incidence @ cosm) / (upstream @ cosm)
    #q_in = np.sum(edges.inlet * np.abs(edges.flow))
    #edges.flow *= sid.Q_in / q_in

def solve_diffusion_nr(sid: SimInputData, inc: Incidence, edges: Edges, graph: Graph, cosm, pressure):
    # flow and mass continuity
    qp_matrix = inc.incidence.T @ spr.diags(edges.diams ** 4 / edges.lens) \
        @ inc.incidence
    jc_matrix = inc.incidence.T @ spr.diags(edges.diams ** 2 / edges.lens) \
        @ inc.incidence
    
    cosm_prev = np.zeros_like(cosm)
    pressure_prev = np.zeros_like(pressure)
    print(np.linalg.norm(cosm - cosm_prev), np.linalg.norm(pressure - pressure_prev))
    print(np.sum(edges.lens == 0))
    
    while np.linalg.norm(cosm - cosm_prev) > sid.c_th or np.linalg.norm(pressure - pressure_prev) > sid.c_th:
        qp = edges.diams ** 4 / edges.lens * (inc.incidence @ pressure)
        c_edge = np.abs(inc.incidence) @ cosm / 2
        print(np.sum(c_edge == 0))
        qc_matrix = inc.incidence.T @ spr.diags(edges.diams ** 2 / edges.lens / c_edge) \
            @ inc.incidence
        jp_matrix = inc.incidence.T @ spr.diags(edges.diams ** 4 / edges.lens * c_edge) \
            @ inc.incidence
        # fluid mass balance at nodes, with positive contribution from the node and negative from neighbors
        fq = (1 - graph.in_vec - graph.out_vec) * (sid.M * qc_matrix @ cosm + qp_matrix @ pressure)
        # solute mass balance, same as above
        fj = (1 - graph.in_vec - graph.out_vec) * (sid.M * jc_matrix @ cosm + jp_matrix @ pressure)
        #
        dfq_dc = -np.abs(inc.incidence.T) @ spr.diags(edges.diams ** 2 / edges.lens / c_edge ** 2) @ np.abs(inc.incidence)
        diag_old = dfq_dc.diagonal()
        dfq_dc -= spr.diags(diag_old)
        diag_new = dfq_dc @ cosm
        dfq_dc.multiply(cosm[:, np.newaxis])
        dfq_dc -= spr.diags(diag_new)    

        dfq_dp = -np.abs(inc.incidence.T) @ spr.diags(edges.diams ** 4 / edges.lens) @ np.abs(inc.incidence)
        diag_old = dfq_dp.diagonal()
        dfq_dp -= 2 * spr.diags(diag_old)
        
        # djc_dc_p = -np.abs(inc.incidence.T) @ spr.diags(edges.diams ** 4 / edges.lens) @ np.abs(inc.incidence)
        # diag_old = djc_dc_p.diagonal()
        # djc_dc_p -= spr.diags(diag_old)
        # diag_new = djc_dc_p @ pressure
        # djc_dc_p.multiply(pressure[:, np.newaxis])
        # djc_dc_p -= spr.diags(diag_new)
        # upstream = 1 * (inc.incidence.T @ (spr.diags(qp) @ inc.incidence > 0) != 0)
        # downstream = 1 * (inc.incidence.T @ (spr.diags(qp) @ inc.incidence < 0) != 0)
        # qstream = -np.abs(inc.incidence.T) @ spr.diags(edges.diams ** 4 / edges.lens * np.abs(qp) / 2) @ np.abs(inc.incidence)
        # djc_dc_p = upstream.multiply(qstream) + downstream.multiply(qstream)
        # diag_old = djc_dc_p.diagonal()
        # djc_dc_p -= spr.diags(diag_old)
        # diag_new = djc_dc_p @ np.ones(sid.nsq)
        # djc_dc_p += spr.diags(diag_new)
        djc_dc_p = 0.5 * inc.incidence.T @ spr.diags(qp) @ inc.incidence

        djc_dp = jp_matrix
        djc_dc = sid.M * jc_matrix + djc_dc_p

        d_matrix = spr.vstack([spr.hstack([dfq_dc, dfq_dp]), spr.hstack([djc_dc, djc_dp])])
        f = np.concatenate((fq, fj))
        in_vec = np.concatenate((graph.in_vec + graph.out_vec, graph.in_vec + graph.out_vec))
        
        d_matrix = d_matrix.multiply(1 - in_vec[:, np.newaxis]) + spr.diags(in_vec)
        
        # np.savetxt('upstream.txt', (upstream @ upstream.T).toarray())
        
        diag_c = d_matrix.diagonal()
        #print(np.sum(diag_c))
        f = f * (diag_c != 0)
        diag_c = diag_c * (diag_c != 0) + 1 * (diag_c == 0)
        #print(np.sum(f))
        d_matrix = d_matrix.multiply(1 - (diag_c == 0)[:, np.newaxis])
        #dc_matrix.setdiag(diag_c)
        diag_old = d_matrix.diagonal()
        d_matrix += spr.diags(diag_c - diag_old)
        np.savetxt('d_matrix.txt', d_matrix.toarray())
        np.savetxt('f.txt', f)
        delta_c = solve_equation(d_matrix, -f)
        cosm_prev = cosm.copy()
        pressure_prev = pressure.copy()
        cosm += delta_c[:sid.nsq]
        pressure += delta_c[sid.nsq:]
        #break
        print(np.linalg.norm(cosm - cosm_prev), np.linalg.norm(pressure - pressure_prev))
        print(np.sum(np.abs(f)))
    c_edge = np.abs(inc.incidence) @ cosm / 2
    edges.flow = edges.diams ** 4 / edges.lens * (inc.incidence @ pressure) + sid.M * edges.diams ** 2 / edges.lens / c_edge * (inc.incidence @ cosm)
    return cosm, pressure



def solve_diffusion_nr_merge(sid: SimInputData, inc: Incidence, edges: Edges, graph: Graph, cosm, pressure):
    # --- basics & masks (once) ---
    # make sure merged zeros are coalesced
    inc.incidence.sum_duplicates()
    inc.incidence.eliminate_zeros()

    N = inc.incidence.shape[1]
    E = inc.incidence.shape[0]

    # active edges/nodes (non-empty rows/cols in incidence)
    edge_active = (inc.incidence.getnnz(axis=1) > 0).astype(float)  # (E,)
    node_active = (inc.incidence.getnnz(axis=0) > 0).astype(float)  # (N,)
    Me = spr.diags(edge_active)                                     # E×E
    Mn = spr.diags(node_active)                                     # N×N

    # row-mask for internal concentration equations (keep your style)
    # (you can still separate c/p masks later if desired)
    base_mask = (1 - graph.in_vec - graph.out_vec) * node_active    # (N,)
    Mrow = spr.diags(base_mask)                                     # N×N

    Babs = abs(inc.incidence)   # for edge-averages
    eps = 1e-15

    cosm_prev = np.zeros_like(cosm)
    pressure_prev = np.zeros_like(pressure)

    while (np.linalg.norm(cosm - cosm_prev) > sid.c_th or
           np.linalg.norm(pressure - pressure_prev) > sid.c_th):

        # --- node values ---
        c = np.maximum(cosm, eps)
        p = pressure

        # --- edge differences / averages (no tail/head) ---
        dc   = inc.incidence @ c                    # Δc  (E,)
        dp   = inc.incidence @ p                    # Δp  (E,)
        cbar = np.maximum(0.5 * (Babs @ c), eps)    # c̄  (E,)

        # --- coefficients per edge ---
        K = sid.M / sid.Pe  * edges.diams**2 / edges.lens
        C =          edges.diams**4 / edges.lens
        G = 1 / sid.Pe * edges.diams**2 / edges.lens   # your naming

        # --- edge fluxes (masked) ---
        Q = -K * (dc / cbar) - C * dp
        J = -(K + G) * dc    - C * cbar * dp

        # mask out inactive edges in residuals
        Rq = inc.incidence.T @ (Me @ Q)     # solvent
        Rj = inc.incidence.T @ (Me @ J)     # solute

        # keep only internal-node rows (your style)
        fq = (Mrow @ Rq)
        fj = (Mrow @ Rj)
        f  = np.concatenate([fq, fj])

        # --- Jacobian (edge→node), masked by active edges ---
        inv_cbar         = 1.0 / cbar
        dc_over_cbar2    = dc / (cbar**2)

        # ∂Q/∂c  = -diag(K/c̄) B + diag(0.5 K Δc/c̄²) |B|
        Qdc_edge = -spr.diags(K * inv_cbar) @ inc.incidence \
                   + spr.diags(0.5 * K * dc_over_cbar2) @ Babs

        # ∂Q/∂p  = -diag(C) B
        Qdp_edge = -spr.diags(C) @ inc.incidence

        # ∂J/∂c  = -diag(K+G) B - diag(0.5 C Δp) |B|
        Jdc_edge = -spr.diags(K + G) @ inc.incidence \
                   - spr.diags(0.5 * C * dp) @ Babs

        # ∂J/∂p  = -diag(C c̄) B
        Jdp_edge = -spr.diags(C * cbar) @ inc.incidence

        # node blocks with edge mask + row mask
        dfq_dc = Mrow @ (inc.incidence.T @ (Me @ Qdc_edge))
        dfq_dp = Mrow @ (inc.incidence.T @ (Me @ Qdp_edge))
        djc_dc = Mrow @ (inc.incidence.T @ (Me @ Jdc_edge))
        djc_dp = Mrow @ (inc.incidence.T @ (Me @ Jdp_edge))

        d_matrix = spr.vstack([
            spr.hstack([dfq_dc, dfq_dp]),
            spr.hstack([djc_dc, djc_dp]),
        ]).tocsr()

        # --- your existing Dirichlet-row identity trick (unchanged) ---
        in_vec = np.concatenate((graph.in_vec + graph.out_vec,
                                 graph.in_vec + graph.out_vec))
        d_matrix = d_matrix.multiply(1 - in_vec[:, None]) + spr.diags(in_vec)

        # (optional) generic diagonal rescue; you can drop this once BC rows are set properly
        diag_c = d_matrix.diagonal()
        f = f * (diag_c != 0)
        diag_c = diag_c * (diag_c != 0) + 1 * (diag_c == 0)
        d_matrix = d_matrix.multiply(1 - (diag_c == 0)[:, None])
        d_matrix += spr.diags(diag_c - d_matrix.diagonal())

        # --- solve & update ---
        delta = solve_equation(d_matrix, -f)
        cosm_prev     = cosm.copy()
        pressure_prev = pressure.copy()
        cosm     += delta[:N]
        pressure += delta[N:]

        print(np.linalg.norm(cosm - cosm_prev), np.linalg.norm(pressure - pressure_prev))
        print(np.sum(np.abs(f)))

    # --- final flows on active edges only (recompute safely) ---
    c = np.maximum(cosm, eps)
    dc   = inc.incidence @ c
    dp   = inc.incidence @ pressure
    cbar = np.maximum(0.5 * (Babs @ c), eps)
    K = sid.M / sid.Pe  * edges.diams**2 / edges.lens
    C =          edges.diams**4 / edges.lens
    Q = -K * (dc / cbar) - C * dp
    print(np.linalg.norm(cosm - cosm_prev), np.linalg.norm(pressure - pressure_prev))
    print(np.sum(np.abs(f)))
    edges.flow = (Me @ -Q)  # zero on inactive edges
    return cosm, pressure

def find_residual(sid: SimInputData, inc: Incidence, edges: Edges, graph: Graph, cosm, pressure):
    inc.incidence.sum_duplicates()
    inc.incidence.eliminate_zeros()


    # active edges/nodes (non-empty rows/cols in incidence)
    edge_active = (inc.incidence.getnnz(axis=1) > 0).astype(float)  # (E,)
    node_active = (inc.incidence.getnnz(axis=0) > 0).astype(float)  # (N,)
    Me = spr.diags(edge_active)                                     # E×E
    Mn = spr.diags(node_active)                                     # N×N

    # row-mask for internal concentration equations (keep your style)
    # (you can still separate c/p masks later if desired)
    base_mask = (1 - graph.in_vec - graph.out_vec) * node_active    # (N,)
    Mrow = spr.diags(base_mask)                                     # N×N

    Babs = abs(inc.incidence)   # for edge-averages
    eps = 1e-15

    # --- node values ---
    c = np.maximum(cosm, eps)
    p = pressure

    # --- edge differences / averages (no tail/head) ---
    dc   = inc.incidence @ c                    # Δc  (E,)
    dp   = inc.incidence @ p                    # Δp  (E,)
    cbar = np.maximum(0.5 * (Babs @ c), eps)    # c̄  (E,)

    # --- coefficients per edge ---
    K = sid.M / sid.Pe  * edges.diams**2 / edges.lens
    C =          edges.diams**4 / edges.lens
    G = 1 / sid.Pe * edges.diams**2 / edges.lens   # your naming

    # --- edge fluxes (masked) ---
    Q = -K * (dc / cbar) - C * dp
    J = -(K + G) * dc    - C * cbar * dp

    # mask out inactive edges in residuals
    Rq = inc.incidence.T @ (Me @ Q)     # solvent
    Rj = inc.incidence.T @ (Me @ J)     # solute

    # keep only internal-node rows (your style)
    fq = (Mrow @ Rq)
    fj = (Mrow @ Rj)
    f  = np.concatenate([fq, fj])
    return f

def solve_diffusion_scipy(sid: SimInputData, inc: Incidence, edges: Edges, graph: Graph, cosm, pressure):
    # --- basics & masks (once) ---
    # make sure merged zeros are coalesced
    Babs = abs(inc.incidence)   # for edge-averages
    edge_active = (inc.incidence.getnnz(axis=1) > 0).astype(float)  # (E,)
    Me = spr.diags(edge_active)
    eps = 1e-15
    N = inc.incidence.shape[1]
    E = inc.incidence.shape[0]
    z0 = np.concatenate([cosm, pressure])
    
    def F(z):
        
        inc.incidence.sum_duplicates()
        inc.incidence.eliminate_zeros()


        # active edges/nodes (non-empty rows/cols in incidence)
        edge_active = (inc.incidence.getnnz(axis=1) > 0).astype(float)  # (E,)
        node_active = (inc.incidence.getnnz(axis=0) > 0).astype(float)  # (N,)
        Me = spr.diags(edge_active)                                     # E×E
        Mn = spr.diags(node_active)                                     # N×N

        # row-mask for internal concentration equations (keep your style)
        # (you can still separate c/p masks later if desired)
        base_mask = (1 - graph.in_vec - graph.out_vec) * node_active    # (N,)
        Mrow = spr.diags(base_mask)                                     # N×N

        Babs = abs(inc.incidence)   # for edge-averages
        eps = 1e-15

        # --- node values ---
        c = np.maximum(z[:N], eps)
        p = z[N:]

        # --- edge differences / averages (no tail/head) ---
        dc   = inc.incidence @ c                    # Δc  (E,)
        dp   = inc.incidence @ p                    # Δp  (E,)
        cbar = np.maximum(0.5 * (Babs @ c), eps)    # c̄  (E,)

        # --- coefficients per edge ---
        K = sid.M / sid.Pe  * edges.diams**2 / edges.lens
        C =          edges.diams**4 / edges.lens
        G = 1 / sid.Pe * edges.diams**2 / edges.lens   # your naming

        # --- edge fluxes (masked) ---
        Q = -K * (dc / cbar) - C * dp
        J = -(K + G) * dc    - C * cbar * dp

        # mask out inactive edges in residuals
        Rq = inc.incidence.T @ (Me @ Q)     # solvent
        Rj = inc.incidence.T @ (Me @ J)     # solute

        # keep only internal-node rows (your style)
        fq = (Mrow @ Rq)
        fj = (Mrow @ Rj)
        return np.concatenate([fq, fj])
    
    sol = newton_krylov(F, z0, f_tol=1e-8, maxiter=50, line_search='armijo')
    cosm = sol[:N]
    pressure = sol[N:]
    # --- final flows on active edges only (recompute safely) ---
    c = np.maximum(cosm, eps)
    dc   = inc.incidence @ c
    dp   = inc.incidence @ pressure
    cbar = np.maximum(0.5 * (Babs @ c), eps)
    K = sid.M / sid.Pe  * edges.diams**2 / edges.lens
    C =          edges.diams**4 / edges.lens
    Q = -K * (dc / cbar) - C * dp
    edges.flow = (Me @ -Q)  # zero on inactive edges
    return cosm, pressure

def solve_diffusion_scipy2(sid, inc, edges, graph, cosm, pressure, *, use_precond=True):
    B = inc.incidence.tocsr(copy=True)
    B.sum_duplicates(); B.eliminate_zeros()
    Babs = abs(B)

    N = B.shape[1]
    E = B.shape[0]
    eps = 1e-15

    # Active edges mask (skip merged edges)
    edge_active = (B.getnnz(axis=1) > 0).astype(float)
    Me = spr.diags(edge_active)

    # Separate row masks
    mask_c = (1 - (graph.in_vec + graph.out_vec)).astype(float)
    mask_p = (1 - getattr(graph, "p_dirichlet", np.zeros_like(mask_c))).astype(float)
    Mrow_c = spr.diags(mask_c)
    Mrow_p = spr.diags(mask_p)

    # Dirichlet values (provide arrays length N)
    c_bc_vals = getattr(graph, "c_dirichlet_values", cosm.copy())
    c_bc_vals = np.maximum(c_bc_vals, eps)
    w_bc = np.log(c_bc_vals)
    bc_c = (graph.in_vec + graph.out_vec).astype(bool)

    if hasattr(graph, "p_dirichlet"):
        bc_p = graph.p_dirichlet.astype(bool)
        p_bc_vals = getattr(graph, "p_dirichlet_values", pressure.copy())
    else:
        bc_p = np.zeros(N, dtype=bool)
        # pin one gauge (first active node)
        cols_nz = np.flatnonzero(B.getnnz(axis=0) > 0)
        if cols_nz.size:
            bc_p[cols_nz[0]] = True
        p_bc_vals = pressure.copy()

    # Coefficients
    d = edges.diams
    L = edges.lens
    K = sid.M * (d**2) / L                # can be ±
    C =          (d**4) / L
    G = getattr(sid, "D", 0.0) * (d**2) / L
    if not np.any(G > 0):
        G = G + 1e-12 * (np.median(np.abs(K[K != 0])) if np.any(K != 0) else 1.0)

    # Build residual in z = [w; p]
    def F(z):
        w = z[:N]
        p = z[N:]

        c = np.exp(w)
        dlogc = B @ w            # exact grad log c
        dp    = B @ p
        cbar  = 0.5 * (Babs @ c)
        dc    = B @ c

        Q = -K * dlogc - C * dp              # edge volumetric flow
        J = Q * cbar - G * dc                # edge solute flow

        Rq = B.T @ (Me @ Q)                  # node balances
        Rj = B.T @ (Me @ J)

        fq = (Mrow_p @ Rq)
        fj = (Mrow_c @ Rj)

        # Strong Dirichlet enforcement in residual
        fq[bc_p] = (p[bc_p] - p_bc_vals[bc_p])
        fj[bc_c] = (w[bc_c] - w_bc[bc_c])

        return np.concatenate([fq, fj])

    # Optional crude diagonal preconditioner (block-diag)
    inner_M = None
    if use_precond:
        # scale rows roughly by node degree and median coeffs
        deg = np.asarray(Babs.T @ (Me @ np.ones(E)))
        d_q = np.maximum(1e-12, deg * (np.median(np.abs(C)) + np.median(np.abs(K))))
        d_j = np.maximum(1e-12, deg * (np.median(np.abs(G)) + np.median(np.abs(K))))
        Dinv = 1.0 / np.concatenate([d_q, d_j])
        inner_M = LinearOperator((2*N, 2*N), matvec=lambda v: Dinv * v)

    # Initial guess in w-space
    w0 = np.log(np.maximum(cosm, eps))
    z0 = np.concatenate([w0, pressure])

    z = newton_krylov(F, z0, f_tol=1e-9, maxiter=60,
                      inner_M=inner_M, line_search="armijo")

    w = z[:N]
    p = z[N:]
    c = np.exp(w)

    # Final flows
    dlogc = B @ w
    dp    = B @ p
    Q     = -K * dlogc - C * dp
    edges.flow = Me @ Q

    return c, p


def solve_diffusion_nr2(sid: SimInputData, inc: Incidence, edges: Edges, graph: Graph, cosm, pressure):
    # flow and mass continuity
    # qp_matrix = inc.incidence.T @ spr.diags(edges.diams ** 4 / edges.lens) \
    #     @ inc.incidence
    # jc_matrix = inc.incidence.T @ spr.diags(edges.diams ** 2 / edges.lens) \
    #     @ inc.incidence
    
    cosm_prev = np.zeros_like(cosm)
    pressure_prev = np.zeros_like(pressure)
    # print(np.linalg.norm(cosm - cosm_prev), np.linalg.norm(pressure - pressure_prev))
    # print(np.sum(edges.lens == 0))
    while np.linalg.norm(cosm - cosm_prev) > sid.c_th or np.linalg.norm(pressure - pressure_prev) > sid.c_th:
        N    = inc.incidence.shape[1]
        E    = inc.incidence.shape[0]
        mask = (1 - graph.in_vec - graph.out_vec)

        eps = 1e-15

        # node values
        c = np.maximum(cosm, eps)
        p = pressure

        # edge indexing (precompute once, outside loop)
        #tail, head = extract_tail_head_fast(inc.incidence)

        ci, cj = c[inc.tail], c[inc.head]
        pi, pj = p[inc.tail], p[inc.head]

        cbar = 0.5*(ci + cj)
        cbar = np.maximum(cbar, eps)
        dc   = cj - ci
        dp   = pj - pi

        K = sid.M * edges.diams**2 / edges.lens
        C = edges.diams**4 / edges.lens
        G = sid.Pe * edges.diams**2 / edges.lens

        # fluxes
        Q = -K * (dc / cbar) - C * dp
        J = -(K + G) * dc - C * cbar * dp

        # residuals
        Rq = (inc.incidence.T @ Q) * mask
        Rj = (inc.incidence.T @ J) * mask

        # edge Jacobians
        row_e = np.arange(E)

        dQdc_tail =  K * cj / (cbar**2)
        dQdc_head = -K * ci / (cbar**2)
        dQdp_tail =  C
        dQdp_head = -C

        dJdc_tail =  (K + G) - 0.5*C*dp
        dJdc_head = -(K + G) - 0.5*C*dp
        dJdp_tail =  C * cbar
        dJdp_head = -C * cbar

        Qdc = spr.coo_matrix((dQdc_tail, (row_e, inc.tail)), shape=(E, N)) \
            + spr.coo_matrix((dQdc_head, (row_e, inc.head)), shape=(E, N))
        Qdp = spr.coo_matrix((dQdp_tail, (row_e, inc.tail)), shape=(E, N)) \
            + spr.coo_matrix((dQdp_head, (row_e, inc.head)), shape=(E, N))
        Jdc = spr.coo_matrix((dJdc_tail, (row_e, inc.tail)), shape=(E, N)) \
            + spr.coo_matrix((dJdc_head, (row_e, inc.head)), shape=(E, N))
        Jdp = spr.coo_matrix((dJdp_tail, (row_e, inc.tail)), shape=(E, N)) \
            + spr.coo_matrix((dJdp_head, (row_e, inc.head)), shape=(E, N))

        dfq_dc = spr.diags(mask) @ (inc.incidence.T @ Qdc)
        dfq_dp = spr.diags(mask) @ (inc.incidence.T @ Qdp)
        djc_dc = spr.diags(mask) @ (inc.incidence.T @ Jdc)
        djc_dp = spr.diags(mask) @ (inc.incidence.T @ Jdp)

        d_matrix = spr.vstack([spr.hstack([dfq_dc, dfq_dp]),
                            spr.hstack([djc_dc, djc_dp])]).tocsr()
        f = np.concatenate([Rq, Rj])

        in_vec = np.concatenate((graph.in_vec + graph.out_vec, graph.in_vec + graph.out_vec))
        
        d_matrix = d_matrix.multiply(1 - in_vec[:, np.newaxis]) + spr.diags(in_vec)
        
        # np.savetxt('upstream.txt', (upstream @ upstream.T).toarray())
        
        diag_c = d_matrix.diagonal()
        #print(np.sum(diag_c))
        f = f * (diag_c != 0)
        diag_c = diag_c * (diag_c != 0) + 1 * (diag_c == 0)
        #print(np.sum(f))
        d_matrix = d_matrix.multiply(1 - (diag_c == 0)[:, np.newaxis])
        #dc_matrix.setdiag(diag_c)
        diag_old = d_matrix.diagonal()
        d_matrix += spr.diags(diag_c - diag_old)
        # np.savetxt('d_matrix.txt', d_matrix.toarray())
        # np.savetxt('f.txt', f)
        delta_c = solve_equation(d_matrix, -f)
        cosm_prev = cosm.copy()
        pressure_prev = pressure.copy()
        cosm += delta_c[:sid.nsq]
        pressure += delta_c[sid.nsq:]
        #break
        print(np.linalg.norm(cosm - cosm_prev), np.linalg.norm(pressure - pressure_prev))
        print(np.sum(np.abs(f)))
    c_edge = np.abs(inc.incidence) @ cosm / 2
    edges.flow = edges.diams ** 4 / edges.lens * (inc.incidence @ pressure) + sid.M * edges.diams ** 2 / edges.lens / c_edge * (inc.incidence @ cosm)
    return cosm, pressure

def solve_diffusion_nr_scipy(sid: SimInputData, inc: Incidence, edges: Edges, graph: Graph, cosm, pressure):
    # --- basics & masks (once) ---
    # make sure merged zeros are coalesced
    inc.incidence.sum_duplicates()
    inc.incidence.eliminate_zeros()

    N = inc.incidence.shape[1]
    E = inc.incidence.shape[0]

    # active edges/nodes (non-empty rows/cols in incidence)
    edge_active = (inc.incidence.getnnz(axis=1) > 0).astype(float)  # (E,)
    node_active = (inc.incidence.getnnz(axis=0) > 0).astype(float)  # (N,)
    Me = spr.diags(edge_active)                                     # E×E
    Mn = spr.diags(node_active)                                     # N×N

    # row-mask for internal concentration equations (keep your style)
    # (you can still separate c/p masks later if desired)
    base_mask = (1 - graph.in_vec - graph.out_vec) * node_active    # (N,)
    Mrow = spr.diags(base_mask)                                     # N×N

    Babs = abs(inc.incidence)   # for edge-averages
    eps = 1e-15

    cosm_prev = np.zeros_like(cosm)
    pressure_prev = np.zeros_like(pressure)

    z0 = np.concatenate([cosm, pressure])

    def F(z):
        c = np.maximum(z[:sid.nsq], eps)
        p = z[sid.nsq:]

        # --- edge differences / averages (no tail/head) ---
        dc   = inc.incidence @ c                    # Δc  (E,)
        dp   = inc.incidence @ p                    # Δp  (E,)
        cbar = np.maximum(0.5 * (Babs @ c), eps)    # c̄  (E,)

        # --- coefficients per edge ---
        K = sid.M / sid.Pe  * edges.diams**2 / edges.lens
        C =          edges.diams**4 / edges.lens
        G = 1 / sid.Pe * edges.diams**2 / edges.lens   # your naming

        # --- edge fluxes (masked) ---
        Q = -K * (dc / cbar) - C * dp
        J = -(K + G) * dc    - C * cbar * dp

        # mask out inactive edges in residuals
        Rq = inc.incidence.T @ (Me @ Q)     # solvent
        Rj = inc.incidence.T @ (Me @ J)     # solute

        # keep only internal-node rows (your style)
        fq = (Mrow @ Rq)
        fj = (Mrow @ Rj)
        f  = np.concatenate([fq, fj])
        return -f

    def jac(z):
        c = np.maximum(z[:sid.nsq], eps)
        p = z[sid.nsq:]

        # --- edge differences / averages (no tail/head) ---
        dc   = inc.incidence @ c                    # Δc  (E,)
        dp   = inc.incidence @ p                    # Δp  (E,)
        cbar = np.maximum(0.5 * (Babs @ c), eps)    # c̄  (E,)

        # --- coefficients per edge ---
        K = sid.M / sid.Pe  * edges.diams**2 / edges.lens
        C =          edges.diams**4 / edges.lens
        G = 1 / sid.Pe * edges.diams**2 / edges.lens   # your naming

        # --- edge fluxes (masked) ---
        Q = -K * (dc / cbar) - C * dp
        J = -(K + G) * dc    - C * cbar * dp

        inv_cbar         = 1.0 / cbar
        dc_over_cbar2    = dc / (cbar**2)

        # ∂Q/∂c  = -diag(K/c̄) B + diag(0.5 K Δc/c̄²) |B|
        Qdc_edge = -spr.diags(K * inv_cbar) @ inc.incidence \
                   + spr.diags(0.5 * K * dc_over_cbar2) @ Babs

        # ∂Q/∂p  = -diag(C) B
        Qdp_edge = -spr.diags(C) @ inc.incidence

        # ∂J/∂c  = -diag(K+G) B - diag(0.5 C Δp) |B|
        Jdc_edge = -spr.diags(K + G) @ inc.incidence \
                   - spr.diags(0.5 * C * dp) @ Babs

        # ∂J/∂p  = -diag(C c̄) B
        Jdp_edge = -spr.diags(C * cbar) @ inc.incidence

        # node blocks with edge mask + row mask
        dfq_dc = Mrow @ (inc.incidence.T @ (Me @ Qdc_edge))
        dfq_dp = Mrow @ (inc.incidence.T @ (Me @ Qdp_edge))
        djc_dc = Mrow @ (inc.incidence.T @ (Me @ Jdc_edge))
        djc_dp = Mrow @ (inc.incidence.T @ (Me @ Jdp_edge))

        d_matrix = spr.vstack([
            spr.hstack([dfq_dc, dfq_dp]),
            spr.hstack([djc_dc, djc_dp]),
        ]).tocsr()

        # --- your existing Dirichlet-row identity trick (unchanged) ---
        in_vec = np.concatenate((graph.in_vec + graph.out_vec,
                                 graph.in_vec + graph.out_vec))
        d_matrix = d_matrix.multiply(1 - in_vec[:, None]) + spr.diags(in_vec)
        # (optional) generic diagonal rescue; you can drop this once BC rows are set properly
        diag_c = d_matrix.diagonal()
        diag_c = diag_c * (diag_c != 0) + 1 * (diag_c == 0)
        d_matrix = d_matrix.multiply(1 - (diag_c == 0)[:, None])
        d_matrix += spr.diags(diag_c - d_matrix.diagonal())
        return d_matrix.toarray()

    z = root(F, z0, method='hybr', jac=jac, tol=1e-8).x
    #z = root(F, z0, method='krylov', options={'fatol': 1e-9, 'maxiter': 60}).x
    cosm = np.maximum(z[:sid.nsq], eps)
    pressure = z[sid.nsq:]
    print(np.sum(F(z)))
    print(cosm)
    print(pressure)
    # --- final flows on active edges only (recompute safely) ---
    c = np.maximum(cosm, eps)
    dc   = inc.incidence @ c
    dp   = inc.incidence @ pressure
    cbar = np.maximum(0.5 * (Babs @ c), eps)
    K = sid.M / sid.Pe  * edges.diams**2 / edges.lens
    C =          edges.diams**4 / edges.lens
    Q = -K * (dc / cbar) - C * dp
    #print(np.linalg.norm(cosm - cosm_prev), np.linalg.norm(pressure - pressure_prev))
    #print(np.sum(np.abs(f)))
    edges.flow = (Me @ -Q)  # zero on inactive edges
    return cosm, pressure

def solve_diffusion_nr_scipy(sid, inc, edges, graph, cosm, pressure):
    # --- geometry (fixed) ---
    B = inc.incidence.tocsr(copy=True)
    B.sum_duplicates(); B.eliminate_zeros()
    Babs = abs(B)

    N = B.shape[1]
    E = B.shape[0]
    eps = 1e-15

    edge_active = (B.getnnz(axis=1) > 0).astype(float)
    Me = spr.diags(edge_active)

    # ---- BC flags and values ----
    # pressure BC at inlet/outlet -> 0
    bc_p = (graph.in_vec + graph.out_vec).astype(bool)
    p_bc_vals = np.zeros(N)

    # concentration BC: inlet 1.05, outlet 0.95 (put your own arrays if they vary)
    bc_c = (graph.in_vec + graph.out_vec).astype(bool)
    c_bc_vals = cosm.copy()
    c_bc_vals[graph.in_vec.astype(bool)] = 1.05
    c_bc_vals[graph.out_vec.astype(bool)] = 0.95
    c_bc_vals = np.maximum(c_bc_vals, eps)

    # row masks for internal balance equations
    mask_p = (~bc_p).astype(float)
    mask_c = (~bc_c).astype(float)
    Mrow_p = spr.diags(mask_p)
    Mrow_c = spr.diags(mask_c)

    # --- coefficients per edge (use your scaling) ---
    K = (sid.M / sid.Pe) * edges.diams**2 / edges.lens
    C =                    edges.diams**4 / edges.lens
    G = (1.0 / sid.Pe)   * edges.diams**2 / edges.lens

    z0 = np.concatenate([cosm, pressure])

    def F(z):
        c = np.maximum(z[:N], eps)
        p = z[N:]

        dc   = B @ c
        dp   = B @ p
        cbar = np.maximum(0.5 * (Babs @ c), eps)

        Q = -K * (dc / cbar) - C * dp
        J = -(K + G) * dc    - C * cbar * dp

        Rq = B.T @ (Me @ Q)     # solvent node balance
        Rj = B.T @ (Me @ J)     # solute node balance

        fq = (Mrow_p @ Rq)
        fj = (Mrow_c @ Rj)

        # --- strong Dirichlet equations (this is the key) ---
        fq[bc_p] = p[bc_p] - p_bc_vals[bc_p]
        fj[bc_c] = c[bc_c] - c_bc_vals[bc_c]

        return np.concatenate([fq, fj])

    def jac(z):
        c = np.maximum(z[:N], eps)
        p = z[N:]

        dc   = B @ c
        dp   = B @ p
        cbar = np.maximum(0.5 * (Babs @ c), eps)

        inv_cbar      = 1.0 / cbar
        dc_over_cbar2 = dc / (cbar**2)

        Qdc_edge = -spr.diags(K * inv_cbar) @ B \
                   + spr.diags(0.5 * K * dc_over_cbar2) @ Babs
        Qdp_edge = -spr.diags(C) @ B
        Jdc_edge = -spr.diags(K + G) @ B \
                   - spr.diags(0.5 * C * dp) @ Babs
        Jdp_edge = -spr.diags(C * cbar) @ B

        dfq_dc = Mrow_p @ (B.T @ (Me @ Qdc_edge))
        dfq_dp = Mrow_p @ (B.T @ (Me @ Qdp_edge))
        djc_dc = Mrow_c @ (B.T @ (Me @ Jdc_edge))
        djc_dp = Mrow_c @ (B.T @ (Me @ Jdp_edge))

        Jmat = spr.vstack([
            spr.hstack([dfq_dc, dfq_dp]),
            spr.hstack([djc_dc, djc_dp]),
        ]).tocsr()

        # --- overwrite Dirichlet rows with identity ---
        rows_p = np.where(bc_p)[0]            # pressure BC rows (top block)
        rows_c = np.where(bc_c)[0] + N        # concentration BC rows (bottom block)
        if rows_p.size:
            Jmat[rows_p, :] = 0.0
            Jmat[rows_p, rows_p] = 1.0
        if rows_c.size:
            Jmat[rows_c, :] = 0.0
            Jmat[rows_c, rows_c] = 1.0

        return Jmat.toarray()   # required by method='hybr'

    #sol = root(F, z0, method='hybr', jac=jac, tol=1e-8)
    sol = root(F, z0, method='krylov', tol=1e-8)
    z = sol.x
    cosm = z[:N]
    pressure = z[N:]
    print(cosm)
    print(pressure)
    c = np.maximum(cosm, eps)
    dc   = inc.incidence @ c
    dp   = inc.incidence @ pressure
    cbar = np.maximum(0.5 * (Babs @ c), eps)
    K = sid.M / sid.Pe  * edges.diams**2 / edges.lens
    C =          edges.diams**4 / edges.lens
    Q = -K * (dc / cbar) - C * dp
    #print(np.linalg.norm(cosm - cosm_prev), np.linalg.norm(pressure - pressure_prev))
    #print(np.sum(np.abs(f)))
    edges.flow = (Me @ -Q)  # zero on inactive edges
    return cosm, pressure

import numpy as np
import scipy.sparse as spr
from scipy.optimize import newton_krylov
from scipy.sparse.linalg import splu, LinearOperator

def solve_diffusion_jfnk(sid, inc, edges, graph, cosm, pressure):
    # --- geometry: freeze once ---
    B = inc.incidence.tocsr(copy=True)
    B.sum_duplicates(); B.eliminate_zeros()
    Babs = abs(B)
    N, E = B.shape[1], B.shape[0]
    eps = 1e-15

    # active edges
    Me = spr.diags((B.getnnz(axis=1) > 0).astype(float))

    # --- BC flags/values ---
    # pressure BCs (set p=0 at inlet/outlet or your preferred values)
    bc_p = graph.in_vec + graph.out_vec
    p_bc_vals = np.zeros(N)

    # concentration BCs (example: inlet 1.05, outlet 0.95; adapt to your arrays)
    bc_c = graph.in_vec + graph.out_vec
    c_bc_vals = cosm.copy()
    c_bc_vals[graph.in_vec.astype(bool)] = 1.05
    c_bc_vals[graph.out_vec.astype(bool)] = 0.95
    c_bc_vals = np.maximum(c_bc_vals, eps)

    # internal-row masks (separate!)
    Mrow_p = spr.diags(1-bc_p)
    Mrow_c = spr.diags(1 - bc_c)

    # --- coefficients (consistent scaling) ---
    K = (sid.M / sid.Pe) * edges.diams**2 / edges.lens
    C =                    edges.diams**4 / edges.lens
    G = (1.0 / sid.Pe)   * edges.diams**2 / edges.lens
    # tiny floor on G helps conditioning if it vanishes
    if not np.any(G > 0):
        G = G + 1e-12 * (np.median(np.abs(K[K!=0])) if np.any(K!=0) else 1.0)

    # mild block scaling so the two residual blocks have similar magnitude
    s_q = max(1e-12, np.median(np.abs(C)) + np.median(np.abs(K)))
    s_j = max(1e-12, np.median(np.abs(G)) + np.median(np.abs(K)))

    # --- residual: includes BC equations ---
    def F(z):
        c = np.maximum(z[:N], eps)
        p = z[N:]

        dc   = B @ c
        dp   = B @ p
        cbar = np.maximum(0.5 * (Babs @ c), eps)

        Q = -K * (dc / cbar) - C * dp
        J = -(K + G) * dc    - C * cbar * dp

        Rq = B.T @ (Me @ Q)            # solvent balance at nodes
        Rj = B.T @ (Me @ J)            # solute balance at nodes

        fq = Mrow_p @ Rq
        fj = Mrow_c @ Rj

        # Strong Dirichlet equations: pin BCs exactly
        fq[bc_p.astype(bool)] = (p[bc_p.astype(bool)] - p_bc_vals[bc_p.astype(bool)])
        fj[bc_c.astype(bool)] = (c[bc_c.astype(bool)] - c_bc_vals[bc_c.astype(bool)])

        # scale blocks for better Krylov conditioning
        return np.concatenate([fq / s_q, fj / s_j])

    # --- sparse block preconditioner M^{-1} ≈ diag((B^T C B)^{-1}, (B^T G B)^{-1}) ---
    Ap = (B.T @ (Me @ spr.diags(C)) @ B).tocsr()   # pressure-like block
    Aw = (B.T @ (Me @ spr.diags(G)) @ B).tocsr()   # concentration-like block

    # make BC rows identity in the preconditioner (so pinned rows are easy)
    # if np.any(bc_p):
    #     rows = np.where(bc_p)[0]
    #     Ap[rows, :] = 0.0; Ap[rows, rows] = 1.0
    # if np.any(bc_c):
    #     rows = np.where(bc_c)[0]
    #     Aw[rows, :] = 0.0; Aw[rows, rows] = 1.0
    Ap = Ap.multiply(1 - bc_p[:, None]) + spr.diags(bc_p)
    Aw = Aw.multiply(1 - bc_c[:, None]) + spr.diags(bc_c)
    # small diagonal bump for numerical stability
    Ap = (Ap + 1e-10 * spr.eye(N, format='csr')).tocsc()
    Aw = (Aw + 1e-10 * spr.eye(N, format='csr')).tocsc()

    lu_p = splu(Ap)
    lu_w = splu(Aw)

    def apply_M(v):
        # inverse of the block-diagonal preconditioner, including block scaling
        vt = v[:N] * (1.0 / s_q)
        vb = v[N:] * (1.0 / s_j)
        yt = lu_p.solve(vt)
        yb = lu_w.solve(vb)
        return np.concatenate([yt, yb])

    Mlin = LinearOperator((2*N, 2*N), matvec=apply_M)

    # --- initial guess (keep BCs consistent) ---
    z0 = np.concatenate([np.maximum(cosm, eps), pressure])
    z0[:N][bc_c.astype(bool)] = c_bc_vals[bc_c.astype(bool)]
    z0[N:][bc_p.astype(bool)] = p_bc_vals[bc_p.astype(bool)]

    # --- Newton–Krylov (Jacobian-free) ---
    z = newton_krylov(F, z0, f_tol=1e-9, maxiter=80,
                      inner_M=Mlin, line_search='armijo')

    c = np.maximum(z[:N], eps)
    p = z[N:]
    dc   = inc.incidence @ c
    dp   = inc.incidence @ p
    cbar = np.maximum(0.5 * (Babs @ c), eps)
    K = sid.M / sid.Pe  * edges.diams**2 / edges.lens
    C =          edges.diams**4 / edges.lens
    Q = -K * (dc / cbar) - C * dp
    edges.flow = (Me @ -Q)
    return c, p
