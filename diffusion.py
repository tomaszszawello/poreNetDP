
from config import SimInputData
from incidence import Incidence
from network import Graph, Edges
from volumes import Volumes
from utils import solve_equation

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as spr

def create_vector(sid: SimInputData, graph: Graph) -> spr.csc_matrix:
    """ Create vector result for B concentration calculation.
    """
    return np.concatenate([sid.cb_0 * graph.in_vec, np.zeros(2 * sid.ne)])

def create_vector_danckwerts(sid: SimInputData, inc: Incidence, graph: Graph, edges: Edges) -> spr.csc_matrix:
    """ Create vector result for B concentration calculation.
    """
    F = spr.diags(edges.flow)
    Z = spr.diags(((edges.flow == 0) & (edges.diams > 0)).astype(float))
    A_pos = ((F @ inc.incidence) > 0)
    Z_pos = ((Z @ inc.incidence) > 0)
    upstream = A_pos.maximum(Z_pos).astype(float)  # edges x nodes (1 where node is upstream)
    # Positive source: sum over upstream edges attached to inlet nodes
    qc_in = sid.Pe * (upstream.T @ np.abs(edges.flow)) * graph.in_vec  # shape: nsq
    return np.concatenate([sid.cb_0 * qc_in, np.zeros(2 * sid.ne)])

def solve_diffusion(sid: SimInputData, inc: Incidence, graph: Graph, edges: Edges, cb_vector):
    lam_plus_val = sid.Pe / 2 * edges.lens / edges.diams ** 2 * \
        (np.sqrt(np.abs(edges.flow) ** 2 + 4 * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) + np.abs(edges.flow))
    lam_plus_val = np.array(np.ma.fix_invalid(lam_plus_val, fill_value = 0))
    lam_minus_val = sid.Pe / 2 * edges.lens / edges.diams ** 2 * \
        (np.sqrt(np.abs(edges.flow) ** 2 + 4 * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) - np.abs(edges.flow)) * (edges.diams <= sid.dmax)
    lam_minus_val = np.array(np.ma.fix_invalid(lam_minus_val, fill_value = 0))
    lam_plus_zero = 1 * (1 * (lam_plus_val > sid.diffusion_exp_limit) + 1 * (edges.diams > sid.dmax) != 0)
    lam_plus_val = lam_plus_val * (1 - lam_plus_zero)

    exp_plus = spr.diags(np.exp(lam_plus_val) * (1 - lam_plus_zero) + lam_plus_zero)
    exp_plus2 = spr.diags(np.exp(lam_plus_val) * (1 - lam_plus_zero))
    exp_minus = spr.diags(np.exp(-lam_minus_val))
    exp_minus2 = spr.diags(np.exp(-lam_minus_val) * (1 - lam_plus_zero))
    
    lam_minus_val = lam_minus_val * (1 - lam_plus_zero)

    upstream = 1 * (spr.diags(edges.flow) @ inc.incidence > 0) + 1 * (spr.diags(1 * (edges.flow == 0)) @ inc.incidence > 0)
    downstream = 1 * (spr.diags(edges.flow) @ inc.incidence < 0)
    downstream2 = 1 * ((spr.diags(edges.flow) @ inc.incidence < 0).multiply((1 - lam_plus_zero)[:, np.newaxis]))
    
    flux_a = 1 * (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_plus2 @ downstream + (spr.diags(lam_plus_val * edges.diams ** 2)) @ upstream - (exp_plus * spr.diags(lam_plus_val * edges.diams ** 2)) @ downstream).multiply((1 - lam_plus_zero)[:, np.newaxis])
    flux_b = 1 * (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_minus @ downstream + (spr.diags(-lam_minus_val * edges.diams ** 2)) @ upstream + (exp_minus * spr.diags(lam_minus_val * edges.diams ** 2)) @ downstream)
    flux_a_in = flux_a.T.multiply((1 - graph.in_vec)[:, np.newaxis]) #+ ((exp_plus * spr.diags(lam_plus_val)) @ downstream).T.multiply((graph.out_vec * (1 - cb_pe_fix))[:, np.newaxis])
    

    flux_b_in = flux_b.T.multiply((1 - graph.in_vec)[:, np.newaxis]) #- ((exp_minus * spr.diags(lam_minus_val)) @ downstream).T.multiply((graph.out_vec * (1 - cb_pe_fix))[:, np.newaxis]) #- (spr.diags(np.abs(edges.flow)) @ exp_minus @ downstream).T.multiply((graph.out_vec * cb_pe_fix)[:, np.newaxis])

    flow_fix_pe = -sid.Pe * downstream.T @ np.abs(edges.flow)
    flow_fix_pe = flow_fix_pe * (1 - graph.in_vec) + graph.in_vec
    
    cb_matrix = spr.vstack([spr.hstack([spr.diags(flow_fix_pe), flux_a_in, flux_b_in]), \
                    spr.hstack([-downstream2, exp_plus, exp_minus2]), \
                    spr.hstack([-upstream, spr.diags(np.ones(sid.ne) - lam_plus_zero), spr.diags(np.ones(sid.ne))]) \
                    ])
    cb_matrix = spr.diags(1 - inc.merge_vec) @ cb_matrix @ spr.diags(1 - inc.merge_vec) + spr.diags(inc.merge_vec)
    
    diag = cb_matrix.diagonal()
    diag_old = diag.copy()
    for node in np.where(np.abs(cb_matrix).sum(axis = 1) == 0)[0]:
        diag[node] = 1
    cb_matrix += spr.diags(diag - diag_old)
    res = solve_equation(cb_matrix, cb_vector)
    cb = res[:sid.nsq]
    edges.A = res[sid.nsq:sid.nsq+sid.ne]
    edges.B = res[sid.nsq+sid.ne:]
    return cb

def solve_diffuson_vol(sid: SimInputData, inc: Incidence, graph: Graph,
                      edges: Edges, vols: Volumes, cb_vector, data) -> np.ndarray:
    """Calculate B concentration with tracking of A volume using a capacity projection to prevent overdissolution."""

    import numpy as np
    import scipy.sparse as spr

    eps = 1e-30

    # --- Build edge→grain weights W (rows sum to 1). Here weights ∝ grain volume. ---
    # A: edges x grains (adjacency)
    A = vols.triangles.tocsr().astype(float)
    tri_w = np.asarray(vols.vol_a, dtype=float)              # n_grains
    triangles_w = A @ spr.diags(tri_w)                       # edges x grains (weighted)
    edge_vol = np.asarray(triangles_w.sum(axis=1)).ravel()   # row sums
    inv_edge_vol = np.divide(1.0, edge_vol, out=np.zeros_like(edge_vol), where=edge_vol > 0)
    W = spr.diags(inv_edge_vol) @ triangles_w                # edges x grains, row-normalized

    # control variables: per-grain alphas
    alpha_tr = (vols.vol_a > 0).astype(float)                # grains
    # map to per-edge alpha via W
    alpha = np.asarray((W @ alpha_tr)).ravel()               # edges

    # ---- helper: one ADR solve + change computation given edge alpha ----

    def solve_transport_and_change_danckwerts(alpha_edge):
        # safe inverse |flow|
        inv_abs_flow = np.divide(1.0, np.abs(edges.flow),
                                 out=np.zeros_like(edges.flow, dtype=float),
                                 where=np.abs(edges.flow) > 0)

        lam_root = np.sqrt(np.abs(edges.flow) ** 2 +
                           4.0 * alpha_edge * sid.Da / (1.0 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3)
        lam_plus_val  = sid.Pe / (2.0 * edges.diams ** 2) * (lam_root + np.abs(edges.flow))
        lam_plus_val = np.array(np.ma.fix_invalid(lam_plus_val, fill_value = 0))
        lam_minus_val = sid.Pe / (2.0 * edges.diams ** 2) * (lam_root - np.abs(edges.flow))
        lam_minus_val = np.array(np.ma.fix_invalid(lam_minus_val, fill_value = 0))

        lam_plus_zero = (lam_plus_val * edges.lens > sid.diffusion_exp_limit).astype(float)
        lam_plus_val  = lam_plus_val  * (1.0 - lam_plus_zero)
        lam_minus_val = lam_minus_val * (1.0 - lam_plus_zero)

        exp_plus_diag   = np.exp(lam_plus_val * edges.lens) * (1.0 - lam_plus_zero) + lam_plus_zero
        exp_plus2_diag  = np.exp(lam_plus_val * edges.lens) * (1.0 - lam_plus_zero)
        exp_minus2_diag = np.exp(-lam_minus_val * edges.lens) * (1.0 - lam_plus_zero)

        exp_plus   = spr.diags(exp_plus_diag)
        exp_plus2  = spr.diags(exp_plus2_diag)
        exp_minus2 = spr.diags(exp_minus2_diag)

        # upstream / downstream
        F = spr.diags(edges.flow)
        zero_carrier = spr.diags(((edges.flow == 0) & (edges.diams > 0)).astype(float))
        A_pos = ((F @ inc.incidence) > 0)
        Z_pos = ((zero_carrier @ inc.incidence) > 0)
        A_neg = ((F @ inc.incidence) < 0)
        Z_neg = ((zero_carrier @ inc.incidence) < 0)
        upstream   = A_pos.maximum(Z_pos).astype(float)
        downstream = A_neg.maximum(Z_neg).astype(float)
        downstream2 = downstream.multiply((1.0 - lam_plus_zero)[:, np.newaxis])

        # flux blocks
        flux_a = (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_plus2 @ downstream
                  + spr.diags(lam_plus_val * edges.diams ** 2) @ upstream
                  - exp_plus2 @ spr.diags(lam_plus_val * edges.diams ** 2) @ downstream).multiply(
                      (1.0 - lam_plus_zero)[:, np.newaxis])
        flux_b = (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_minus2 @ downstream
                  - spr.diags(lam_minus_val * edges.diams ** 2) @ upstream
                  + exp_minus2 @ spr.diags(lam_minus_val * edges.diams ** 2) @ downstream).multiply(
                      (1.0 - lam_plus_zero)[:, np.newaxis])

        # Pe fix
        exp_pe_fix = np.exp(-alpha_edge * sid.Da / (1.0 + sid.G * edges.diams) *
                            edges.diams * edges.lens * inv_abs_flow)
        flux_b += sid.Pe * spr.diags(np.abs(edges.flow)) @ spr.diags(lam_plus_zero * exp_pe_fix) @ downstream

        flux_a_in = -flux_a.T#.multiply((1.0 - graph.in_vec)[:, np.newaxis])
        flux_b_in = -flux_b.T#.multiply((1.0 - graph.in_vec)[:, np.newaxis])

        flow_fix_pe = sid.Pe * downstream.T @ np.abs(edges.flow)
        flow_fix_pe = flow_fix_pe * (1.0 - graph.in_vec)
        flow_fix_pe += sid.Pe * upstream.T @ np.abs(edges.flow) * graph.in_vec

        zero_flow_fix = ((edges.flow == 0) & (alpha_edge == 0)).astype(float)
        exp_plus_eff_diag = exp_plus_diag.copy()
        exp_plus_eff_diag[zero_flow_fix.astype(bool)] = 1.0
        exp_plus_eff = spr.diags(exp_plus_eff_diag)

        cb_matrix = spr.vstack([
            spr.hstack([spr.diags(flow_fix_pe),           flux_a_in,                                 flux_b_in]),
            spr.hstack([-downstream2,                     exp_plus_eff,                              exp_minus2]),
            spr.hstack([-upstream,                        spr.diags(np.ones(sid.ne) - zero_flow_fix), spr.diags(np.ones(sid.ne))])
        ])
        merge_diag = spr.diags((1.0 - inc.merge_vec))
        cb_matrix = merge_diag @ cb_matrix @ merge_diag + spr.diags(inc.merge_vec.astype(float))


        rows_empty = (cb_matrix.getnnz(axis=1) == 0)
        if np.any(rows_empty):
            cb_matrix = cb_matrix + spr.diags(rows_empty.astype(float))

        res = solve_equation(cb_matrix, cb_vector)
        cb = res[:sid.nsq]
        edges.A = res[sid.nsq:sid.nsq + sid.ne]
        edges.B = res[sid.nsq + sid.ne:]

        # normalize
        J_in = np.sum(
            edges.inlet *
            (np.abs(edges.flow) * (edges.A * (1.0 - lam_plus_zero) + edges.B)
             - (1.0 - lam_plus_zero) / sid.Pe * edges.diams ** 2 * (lam_plus_val * edges.A - lam_minus_val * edges.B))
        )
        if (not np.isfinite(J_in)) or (J_in <= 0):
            raise ValueError("Non-positive or invalid inlet flux during normalization")
        #scale = sid.cb_0 * sid.Q_in / J_in
        #cb *= scale; edges.A *= scale; edges.B *= scale

        # edge loss rate ("change")
        change_pe_fix = lam_plus_zero * 2.0 * edges.B * np.abs(edges.flow) / sid.Da * (
            1.0 - np.exp(-alpha_edge * sid.Da / (1.0 + sid.G * edges.diams) * edges.diams * edges.lens * inv_abs_flow)
        )
        change_pe_fix = np.array(np.ma.fix_invalid(change_pe_fix, fill_value=0.0))
        change = ((1.0 - lam_plus_zero) * 2.0 * edges.diams ** 2 / (sid.Pe * sid.Da) *
                  (edges.A * (np.exp(lam_plus_val * edges.lens) - 1.0) * lam_minus_val
                   + edges.B * (1.0 - np.exp(-lam_minus_val * edges.lens)) * lam_plus_val)
                  + change_pe_fix)
        change = np.array(np.ma.fix_invalid(change, fill_value=0.0))

        return cb, lam_plus_val, lam_minus_val, lam_plus_zero, change


    # ---- iterate a couple times: solve → project → re-solve ----
    max_proj_iters = getattr(sid, "proj_iters", 3)

    for k in range(max_proj_iters):
        # 1) solve ADR with current alphas
        cb, lam_plus_val, lam_minus_val, lam_plus_zero, change = solve_transport_and_change_danckwerts(alpha)
        #cb, lam_plus_val, lam_minus_val, lam_plus_zero, change = solve_transport_and_change(alpha)

        # 2) predict per-grain loss this step: qg_hat = (W^T * change) * dt
        qg_hat = np.asarray((W.T @ change)).ravel() * sid.dt   # grains

        # 3) capacity projection: s = min(1, vol_a / (qg_hat + eps))
        s = np.minimum(1.0, np.divide(vols.vol_a, qg_hat + eps))

        # if nothing is overdissolving, we’re done
        if np.all(s >= 1.0 - 1e-12):
            break

        # 4) scale grain alphas and map back to edges, then loop
        alpha_tr = np.clip(alpha_tr * s, 0.0, 1.0)
        alpha = np.asarray((W @ alpha_tr)).ravel()

    # After projection iterations, accept the last solve results.
    # (If loop exited because of scaling, the final iteration already re-solved ADR.)

    # Update solid volumes by the accepted dissolution this step:
    # Use the last qg_hat we computed (if loop broke early, recompute once).
    if 'qg_hat' not in locals() or qg_hat.shape[0] != vols.vol_a.shape[0]:
        # recompute qg_hat for the final 'change'
        qg_hat = np.asarray((W.T @ change)).ravel() * sid.dt

    # Actual dissolved per grain (clamped by capacity)
    qg = np.minimum(vols.vol_a, qg_hat)
    vols.vol_a = np.maximum(vols.vol_a - qg, 0.0)

    # Outputs
    edges.alpha_b = alpha
    data.J_in = np.sum(
        edges.inlet *
        (np.abs(edges.flow) * (edges.A * (1.0 - lam_plus_zero) + edges.B)
         - (1.0 - lam_plus_zero) / sid.Pe * edges.diams ** 2 * (lam_plus_val * edges.A - lam_minus_val * edges.B))
    )
    out_sel = ((inc.incidence.T @ spr.diags(edges.flow)) < 0).astype(float)
    data.J_out = np.abs(out_sel @ (np.abs(edges.flow) * edges.outlet)) @ cb

    # simple sanity
    if np.any(cb < -1e-2):
        print(np.where(cb < 0)[0], cb[np.where(cb < 0)[0]])
        F = spr.diags(edges.flow)
        Z = spr.diags(((edges.flow == 0) & (edges.diams > 0)).astype(float))
        A_pos = ((F @ inc.incidence) > 0)
        Z_pos = ((Z @ inc.incidence) > 0)
        upstream = A_pos.maximum(Z_pos).astype(float)  # edges x nodes (1 where node is upstream)
        # Positive source: sum over upstream edges attached to inlet nodes
        qc_in = sid.Pe * (upstream.T @ np.abs(edges.flow))
        print(qc_in[np.where(cb < 0)[0]])
        raise ValueError("Negative concentration detected")
    return cb
