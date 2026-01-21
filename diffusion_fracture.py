
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

def solve_diffusion(sid: SimInputData, inc: Incidence, graph: Graph, edges: Edges, cb_vector):
    edges.alpha = 1 * (inc.boundary.T @ (edges.grain > 0)) / 2 * edges.active
    lam_plus_val = sid.Pe / 2 / edges.diams ** 2 * \
        (np.sqrt(np.abs(edges.flow) ** 2 + 4 * edges.alpha * sid.ksi * edges.diams ** 2 / sid.Pe ** 2) + np.abs(edges.flow))
    lam_plus_val = np.array(np.ma.fix_invalid(lam_plus_val, fill_value = 0))
    lam_minus_val = sid.Pe / 2 / edges.diams ** 2 * \
        (np.sqrt(np.abs(edges.flow) ** 2 + 4 * edges.alpha * sid.ksi * edges.diams ** 2 / sid.Pe ** 2) - np.abs(edges.flow))
    lam_minus_val = np.array(np.ma.fix_invalid(lam_minus_val, fill_value = 0))
    lam_plus_zero = ((lam_plus_val > sid.diffusion_exp_limit) |
                (edges.diams == 0) |
                (edges.flow == 0))
    lam_plus_zero = lam_plus_zero.astype(int)
    lam_plus_val = lam_plus_val * (1 - lam_plus_zero)

    exp_plus = spr.diags(np.exp(lam_plus_val * edges.lens) * (1 - lam_plus_zero) + lam_plus_zero)
    exp_plus2 = spr.diags(np.exp(lam_plus_val * edges.lens) * (1 - lam_plus_zero))
    exp_minus = spr.diags(np.exp(-lam_minus_val * edges.lens)) 
    exp_minus2 = spr.diags(np.exp(-lam_minus_val * edges.lens) * (1 - lam_plus_zero))
    
    lam_minus_val = lam_minus_val * (1 - lam_plus_zero)
    

    upstream = 1 * (spr.diags(edges.flow) @ inc.incidence > 0) + 1 * (spr.diags(1 * (edges.flow == 0)) @ inc.incidence < 0)
    downstream = 1 * (spr.diags(edges.flow) @ inc.incidence < 0) + 1 * (spr.diags(1 * (edges.flow == 0)) @ inc.incidence > 0)
    downstream2 = 1 * ((spr.diags(edges.flow) @ inc.incidence < 0).multiply((1 - lam_plus_zero)[:, np.newaxis])) + 1 * (spr.diags(1 * (edges.flow == 0)) @ inc.incidence > 0)
    
    flux_a = 1 * (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_plus2 @ downstream + (spr.diags(lam_plus_val * edges.diams ** 2)) @ upstream - (exp_plus * spr.diags(lam_plus_val * edges.diams ** 2)) @ downstream).multiply((1 - lam_plus_zero)[:, np.newaxis])
    flux_b = 1 * (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_minus @ downstream + (spr.diags(-lam_minus_val * edges.diams ** 2)) @ upstream + (exp_minus * spr.diags(lam_minus_val * edges.diams ** 2)) @ downstream)
    exp_pe_fix = np.exp(-edges.alpha * sid.ksi / sid.Pe * edges.lens / np.abs(edges.flow))
    exp_pe_fix = np.array(np.ma.fix_invalid(exp_pe_fix, fill_value = 0))
    flux_b += 1 * (sid.Pe * spr.diags(np.abs(edges.flow)) @ spr.diags((lam_plus_val > sid.diffusion_exp_limit) * exp_pe_fix) @ downstream)

    
    flux_a_in = flux_a.T.multiply((1 - graph.in_vec)[:, np.newaxis]) 
    flux_b_in = flux_b.T.multiply((1 - graph.in_vec)[:, np.newaxis]) #- ((exp_minus * spr.diags(lam_minus_val)) @ downstream).T.multiply((graph.out_vec * (1 - cb_pe_fix))[:, np.newaxis]) #- (spr.diags(np.abs(edges.flow)) @ exp_minus @ downstream).T.multiply((graph.out_vec * cb_pe_fix)[:, np.newaxis])
    flow_fix_pe = -sid.Pe * downstream.T @ np.abs(edges.flow)
    flow_fix_pe = flow_fix_pe * (1 - graph.in_vec) + graph.in_vec
    
    cb_matrix = spr.vstack([spr.hstack([spr.diags(flow_fix_pe), flux_a_in, flux_b_in]), \
                    spr.hstack([-downstream2, exp_plus, exp_minus2]), \
                    spr.hstack([-upstream, spr.diags(np.ones(sid.ne) - lam_plus_zero), spr.diags(np.ones(sid.ne))]) \
                    ])
    
    diag = cb_matrix.diagonal()
    diag_old = diag.copy()
    for node in np.where(np.abs(cb_matrix).sum(axis = 1) == 0)[0]:
        diag[node] = 1
    cb_matrix += spr.diags(diag - diag_old)
    res = solve_equation(cb_matrix, cb_vector)
    cb = res[:sid.nsq]
    edges.A = res[sid.nsq:sid.nsq+sid.ne]
    edges.B = res[sid.nsq+sid.ne:]
    print(np.max(cb), np.min(cb))
    return cb

def solve_diffusion_da(sid: SimInputData, inc: Incidence, graph: Graph, edges: Edges, cb_vector):
    edges.alpha = 1 * (inc.boundary.T @ (edges.grain > 0)) / 2 * edges.active
    lam_plus_val = sid.Pe / 2 / edges.diams ** 2 * \
        (np.sqrt(np.abs(edges.flow) ** 2 + 4 * edges.alpha * sid.Da * edges.diams ** 3 / sid.Pe) + np.abs(edges.flow))
    lam_plus_val = np.array(np.ma.fix_invalid(lam_plus_val, fill_value = 0))
    lam_minus_val = sid.Pe / 2 / edges.diams ** 2 * \
        (np.sqrt(np.abs(edges.flow) ** 2 + 4 * edges.alpha * sid.Da * edges.diams ** 3 / sid.Pe) - np.abs(edges.flow))
    lam_minus_val = np.array(np.ma.fix_invalid(lam_minus_val, fill_value = 0))
    lam_plus_zero = ((lam_plus_val > sid.diffusion_exp_limit) |
                (edges.diams == 0) |
                (edges.flow == 0))
    lam_plus_zero = lam_plus_zero.astype(int)
    lam_plus_val = lam_plus_val * (1 - lam_plus_zero)

    exp_plus = spr.diags(np.exp(lam_plus_val * edges.lens) * (1 - lam_plus_zero) + lam_plus_zero)
    exp_plus2 = spr.diags(np.exp(lam_plus_val * edges.lens) * (1 - lam_plus_zero))
    exp_minus = spr.diags(np.exp(-lam_minus_val * edges.lens)) 
    exp_minus2 = spr.diags(np.exp(-lam_minus_val * edges.lens) * (1 - lam_plus_zero))
    
    lam_minus_val = lam_minus_val * (1 - lam_plus_zero)
    

    upstream = 1 * (spr.diags(edges.flow) @ inc.incidence > 0) + 1 * (spr.diags(1 * (edges.flow == 0)) @ inc.incidence < 0)
    downstream = 1 * (spr.diags(edges.flow) @ inc.incidence < 0) + 1 * (spr.diags(1 * (edges.flow == 0)) @ inc.incidence > 0)
    downstream2 = 1 * ((spr.diags(edges.flow) @ inc.incidence < 0).multiply((1 - lam_plus_zero)[:, np.newaxis])) + 1 * (spr.diags(1 * (edges.flow == 0)) @ inc.incidence > 0)
    
    flux_a = 1 * (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_plus2 @ downstream + (spr.diags(lam_plus_val * edges.diams ** 2)) @ upstream - (exp_plus * spr.diags(lam_plus_val * edges.diams ** 2)) @ downstream).multiply((1 - lam_plus_zero)[:, np.newaxis])
    flux_b = 1 * (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_minus @ downstream + (spr.diags(-lam_minus_val * edges.diams ** 2)) @ upstream + (exp_minus * spr.diags(lam_minus_val * edges.diams ** 2)) @ downstream)
    exp_pe_fix = np.exp(-edges.alpha * sid.Da * edges.diams * edges.lens / np.abs(edges.flow))
    exp_pe_fix = np.array(np.ma.fix_invalid(exp_pe_fix, fill_value = 0))
    flux_b += 1 * (sid.Pe * spr.diags(np.abs(edges.flow)) @ spr.diags((lam_plus_val > sid.diffusion_exp_limit) * exp_pe_fix) @ downstream)

    
    flux_a_in = flux_a.T.multiply((1 - graph.in_vec)[:, np.newaxis]) 
    flux_b_in = flux_b.T.multiply((1 - graph.in_vec)[:, np.newaxis]) #- ((exp_minus * spr.diags(lam_minus_val)) @ downstream).T.multiply((graph.out_vec * (1 - cb_pe_fix))[:, np.newaxis]) #- (spr.diags(np.abs(edges.flow)) @ exp_minus @ downstream).T.multiply((graph.out_vec * cb_pe_fix)[:, np.newaxis])
    flow_fix_pe = -sid.Pe * downstream.T @ np.abs(edges.flow)
    flow_fix_pe = flow_fix_pe * (1 - graph.in_vec) + graph.in_vec
    
    cb_matrix = spr.vstack([spr.hstack([spr.diags(flow_fix_pe), flux_a_in, flux_b_in]), \
                    spr.hstack([-downstream2, exp_plus, exp_minus2]), \
                    spr.hstack([-upstream, spr.diags(np.ones(sid.ne) - lam_plus_zero), spr.diags(np.ones(sid.ne))]) \
                    ])
    
    diag = cb_matrix.diagonal()
    diag_old = diag.copy()
    for node in np.where(np.abs(cb_matrix).sum(axis = 1) == 0)[0]:
        diag[node] = 1
    cb_matrix += spr.diags(diag - diag_old)
    res = solve_equation(cb_matrix, cb_vector)
    cb = res[:sid.nsq]
    edges.A = res[sid.nsq:sid.nsq+sid.ne]
    edges.B = res[sid.nsq+sid.ne:]
    print(np.max(cb), np.min(cb))
    return cb

import numpy as np
import scipy.sparse as spr

def solve_diffusion_da_chat(sid: SimInputData, inc: Incidence, graph: Graph, edges: Edges, cb_vector):

    # --- alpha as in your code ---
    edges.alpha = 1 * (inc.boundary.T @ (edges.grain > 0)) / 2 * edges.active
    alpha = np.array(edges.alpha).astype(float).ravel()

    # ============================================================
    # NEW: effective flow used in ODEs
    #  - alpha==0 : keep advection-diffusion
    #  - alpha>0  : diffusion-reaction only => advection OFF
    # ============================================================
    flow_eff = np.array(edges.flow, dtype=float).copy()
    flow_eff[alpha > 0] = 0.0
    abs_flow_eff = np.abs(flow_eff)

    # ============================================================
    # Your lambda formulas, but with abs_flow_eff instead of abs(edges.flow)
    # ============================================================
    lam_plus_val = sid.Pe / 2 / edges.diams ** 2 * \
        (np.sqrt(abs_flow_eff ** 2 + 4 * alpha * sid.Da * edges.diams ** 3 / sid.Pe) + abs_flow_eff)
    lam_plus_val = np.array(np.ma.fix_invalid(lam_plus_val, fill_value=0))

    lam_minus_val = sid.Pe / 2 / edges.diams ** 2 * \
        (np.sqrt(abs_flow_eff ** 2 + 4 * alpha * sid.Da * edges.diams ** 3 / sid.Pe) - abs_flow_eff)
    lam_minus_val = np.array(np.ma.fix_invalid(lam_minus_val, fill_value=0))

    # ============================================================
    # NEW: linear edges = alpha==0 and flow==0 => pure diffusion => c=A+B*x
    # ============================================================
    lin_edges = (alpha == 0) & (abs_flow_eff == 0) & (edges.diams > 0) & (edges.active > 0)

    # ============================================================
    # FIX 1: lam_plus_zero should NOT include (flow == 0)
    # Keep it only as your overflow limiter + diam==0 protection
    # ============================================================
    lam_plus_zero = ((lam_plus_val > sid.diffusion_exp_limit) |
                     (edges.diams == 0))
    lam_plus_zero = lam_plus_zero.astype(int)

    # Do NOT drop A-mode for linear edges (they are handled separately)
    lam_plus_zero[lin_edges] = 0

    lam_plus_val = lam_plus_val * (1 - lam_plus_zero)
    lam_minus_val = lam_minus_val * (1 - lam_plus_zero)

    # --- safer exponentials (avoid inf) ---
    argp = np.clip(lam_plus_val * edges.lens, -700, 700)
    argm = np.clip(lam_minus_val * edges.lens, -700, 700)

    exp_plus_vec = np.exp(argp) * (1 - lam_plus_zero) + lam_plus_zero
    exp_plus2_vec = np.exp(argp) * (1 - lam_plus_zero)

    exp_minus_vec = np.exp(-argm)
    exp_minus2_vec = np.exp(-argm) * (1 - lam_plus_zero)

    # ============================================================
    # FIX 2: for linear edges we want c(L) = A + B*L
    # so exp_plus = 1, exp_minus2 = L
    # ============================================================
    exp_plus_vec[lin_edges] = 1.0
    exp_plus2_vec[lin_edges] = 0.0
    exp_minus_vec[lin_edges] = 1.0
    exp_minus2_vec[lin_edges] = edges.lens[lin_edges]

    exp_plus = spr.diags(exp_plus_vec)
    exp_plus2 = spr.diags(exp_plus2_vec)
    exp_minus = spr.diags(exp_minus_vec)
    exp_minus2 = spr.diags(exp_minus2_vec)

    # ============================================================
    # Upstream/downstream based on flow_eff (not edges.flow)
    # ============================================================
    upstream = 1 * (spr.diags(flow_eff) @ inc.incidence > 0) + \
               1 * (spr.diags(1 * (flow_eff == 0)) @ inc.incidence < 0)

    downstream = 1 * (spr.diags(flow_eff) @ inc.incidence < 0) + \
                 1 * (spr.diags(1 * (flow_eff == 0)) @ inc.incidence > 0)

    downstream2 = 1 * ((spr.diags(flow_eff) @ inc.incidence < 0).multiply((1 - lam_plus_zero)[:, np.newaxis])) + \
                  1 * (spr.diags(1 * (flow_eff == 0)) @ inc.incidence > 0)

    # ============================================================
    # flux_a / flux_b as in your code BUT using abs_flow_eff
    # ============================================================
    flux_a = 1 * (
        sid.Pe * spr.diags(abs_flow_eff) @ exp_plus2 @ downstream
        + (spr.diags(lam_plus_val * edges.diams ** 2)) @ upstream
        - (exp_plus * spr.diags(lam_plus_val * edges.diams ** 2)) @ downstream
    ).multiply((1 - lam_plus_zero)[:, np.newaxis])

    flux_b = 1 * (
        sid.Pe * spr.diags(abs_flow_eff) @ exp_minus @ downstream
        + (spr.diags(-lam_minus_val * edges.diams ** 2)) @ upstream
        + (exp_minus * spr.diags(lam_minus_val * edges.diams ** 2)) @ downstream
    )

    # ============================================================
    # FIX 3: add diffusion flux for linear edges (alpha==0, flow==0)
    # c(x) = A + Bx => dc/dx = B
    # outgoing flux at upstream:  -d^2 * B
    # outgoing flux at downstream:+d^2 * B
    # ============================================================
    lin_vec = lin_edges.astype(float)
    flux_b += spr.diags(-edges.diams ** 2 * lin_vec) @ upstream \
              + spr.diags(edges.diams ** 2 * lin_vec) @ downstream

    # ============================================================
    # Your "high-Pe fix" should ONLY apply to alpha==0 advection edges
    # ============================================================
    adv_edges = (alpha == 0) & (abs_flow_eff > 0)
    exp_pe_fix = np.zeros_like(abs_flow_eff, dtype=float)
    exp_pe_fix[adv_edges] = np.exp(-alpha[adv_edges] * sid.Da * edges.diams[adv_edges] *
                                  edges.lens[adv_edges] / abs_flow_eff[adv_edges])
    exp_pe_fix = np.array(np.ma.fix_invalid(exp_pe_fix, fill_value=0))

    flux_b += 1 * (
        sid.Pe * spr.diags(abs_flow_eff)
        @ spr.diags((lam_plus_val > sid.diffusion_exp_limit) * exp_pe_fix)
        @ downstream
    )

    # ============================================================
    # inlet masking (same as your code)
    # ============================================================
    flux_a_in = flux_a.T.multiply((1 - graph.in_vec)[:, np.newaxis])
    flux_b_in = flux_b.T.multiply((1 - graph.in_vec)[:, np.newaxis])

    # use flow_eff here too
    flow_fix_pe = -sid.Pe * downstream.T @ abs_flow_eff
    flow_fix_pe = flow_fix_pe * (1 - graph.in_vec) + graph.in_vec

    # ============================================================
    # FIX 4: upstream continuity row:
    # original: cb_up = (1-lam_plus_zero)*A + 1*B
    # linear:   cb_up = 1*A + 0*B
    # ============================================================
    A_up_coeff = (1 - lam_plus_zero).astype(float)
    B_up_coeff = np.ones(sid.ne, dtype=float)

    A_up_coeff[lin_edges] = 1.0
    B_up_coeff[lin_edges] = 0.0

    cb_matrix = spr.vstack([
        spr.hstack([spr.diags(flow_fix_pe), flux_a_in, flux_b_in]),
        spr.hstack([-downstream2, exp_plus, exp_minus2]),
        spr.hstack([-upstream, spr.diags(A_up_coeff), spr.diags(B_up_coeff)])
    ])

    # safety diag fix (same as your code)
    diag = cb_matrix.diagonal()
    diag_old = diag.copy()
    for node in np.where(np.abs(cb_matrix).sum(axis=1) == 0)[0]:
        diag[node] = 1
    cb_matrix += spr.diags(diag - diag_old)

    # solve
    res = solve_equation(cb_matrix, cb_vector)
    cb = res[:sid.nsq]
    edges.A = res[sid.nsq:sid.nsq + sid.ne]
    edges.B = res[sid.nsq + sid.ne:]

    print(np.max(cb), np.min(cb))
    return cb

def solve_diffusion_da_fix(sid: SimInputData, inc: Incidence, graph: Graph, edges: Edges, cb_vector):
    edges.alpha = 1 * (inc.boundary.T @ (edges.grain > 0)) / 2 * edges.active
    lam_plus_val = sid.Pe / 2 / edges.diams ** 2 * \
        (np.sqrt(np.abs(edges.flow) ** 2 + 4 * edges.alpha * sid.Da * edges.diams ** 3 / sid.Pe) + np.abs(edges.flow))
    lam_plus_val = np.array(np.ma.fix_invalid(lam_plus_val, fill_value = 0))
    lam_minus_val = sid.Pe / 2 / edges.diams ** 2 * \
        (np.sqrt(np.abs(edges.flow) ** 2 + 4 * edges.alpha * sid.Da * edges.diams ** 3 / sid.Pe) - np.abs(edges.flow))
    lam_minus_val = np.array(np.ma.fix_invalid(lam_minus_val, fill_value = 0))
    
    lam_diff_fix = 1 * ((edges.flow == 0) & (edges.diams > 0) &
                (edges.alpha == 0)) # fix where we should solve just diffusion
    lam_exp_fix = 1 * (lam_plus_val > sid.diffusion_exp_limit) * (1 - lam_diff_fix) # fix where flow is so large that diffusion is negligible
    lam_plus_zero = lam_exp_fix + lam_diff_fix
    lam_plus_val = lam_plus_val * (1 - lam_exp_fix - lam_diff_fix)
    lam_minus_val = lam_minus_val * (1 - lam_exp_fix - lam_diff_fix)

    exp_plus = spr.diags(np.exp(lam_plus_val * edges.lens) * (1 - lam_exp_fix - lam_diff_fix))
    exp_plus2 = spr.diags(np.exp(lam_plus_val * edges.lens) * (1 - lam_exp_fix - lam_diff_fix))
    exp_minus = spr.diags(np.exp(-lam_minus_val * edges.lens) * (1 - lam_diff_fix)) 
    exp_minus2 = spr.diags(np.exp(-lam_minus_val * edges.lens) * (1 - lam_exp_fix - lam_diff_fix))
    
    #lam_minus_val = lam_minus_val * (1 - lam_plus_zero)
    

    upstream = 1 * (spr.diags(edges.flow) @ inc.incidence > 0) + 1 * (spr.diags(1 * (edges.flow == 0)) @ inc.incidence < 0)
    downstream = 1 * (spr.diags(edges.flow) @ inc.incidence < 0) + 1 * (spr.diags(1 * (edges.flow == 0)) @ inc.incidence > 0)
    #downstream2 = 1 * ((spr.diags(edges.flow) @ inc.incidence < 0).multiply((1 - lam_exp_fix)[:, np.newaxis])) + 1 * (spr.diags(1 * (edges.flow == 0)) @ inc.incidence > 0)
    
    flux_a = 1 * (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_plus2 @ downstream + (spr.diags(lam_plus_val * edges.diams ** 2)) @ upstream - (exp_plus2 * spr.diags(lam_plus_val * edges.diams ** 2)) @ downstream)
    flux_b = 1 * (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_minus2 @ downstream + (spr.diags(-lam_minus_val * edges.diams ** 2)) @ upstream + (exp_minus2 * spr.diags(lam_minus_val * edges.diams ** 2)) @ downstream)
    exp_pe_fix = np.exp(-edges.alpha * sid.Da * edges.diams * edges.lens / np.abs(edges.flow))
    exp_pe_fix = np.array(np.ma.fix_invalid(exp_pe_fix, fill_value = 0))
    

    flux_b += 1 * (sid.Pe * spr.diags(np.abs(edges.flow)) @ spr.diags(lam_exp_fix * exp_pe_fix) @ downstream)
    flux_b += spr.diags(-edges.diams**2 * lam_diff_fix) @ upstream \
            + spr.diags(edges.diams**2 * lam_diff_fix) @ downstream

    
    flux_a_in = flux_a.T.multiply((1 - graph.in_vec)[:, np.newaxis]) 
    flux_b_in = flux_b.T.multiply((1 - graph.in_vec)[:, np.newaxis]) #- ((exp_minus * spr.diags(lam_minus_val)) @ downstream).T.multiply((graph.out_vec * (1 - cb_pe_fix))[:, np.newaxis]) #- (spr.diags(np.abs(edges.flow)) @ exp_minus @ downstream).T.multiply((graph.out_vec * cb_pe_fix)[:, np.newaxis])
    flow_fix_pe = -sid.Pe * downstream.T @ np.abs(edges.flow)
    flow_fix_pe = flow_fix_pe * (1 - graph.in_vec) + graph.in_vec
    
    
    # cb_matrix = spr.vstack([spr.hstack([spr.diags(flow_fix_pe), flux_a_in, flux_b_in]), \
    #                 spr.hstack([-downstream2, exp_plus, exp_minus2]), \
    #                 spr.hstack([-upstream, spr.diags(np.ones(sid.ne) - lam_plus_zero), spr.diags(np.ones(sid.ne))]) \
    #                 ])
    cb_matrix = spr.vstack([spr.hstack([spr.diags(flow_fix_pe), flux_a_in, flux_b_in]), \
                spr.hstack([-downstream, exp_plus2 + spr.diags(lam_diff_fix + exp_pe_fix * lam_exp_fix), exp_minus2 + spr.diags(edges.lens * lam_diff_fix + exp_pe_fix * lam_exp_fix)]), \
                spr.hstack([-upstream, spr.diags(np.ones(sid.ne)),  spr.diags(np.ones(sid.ne) - lam_diff_fix)]) \
                ])

    diag = cb_matrix.diagonal()
    diag_old = diag.copy()
    for node in np.where(np.abs(cb_matrix).sum(axis = 1) == 0)[0]:
        diag[node] = 1
    cb_matrix += spr.diags(diag - diag_old)
    res = solve_equation(cb_matrix, cb_vector)
    cb = res[:sid.nsq]
    edges.A = res[sid.nsq:sid.nsq+sid.ne]
    edges.B = res[sid.nsq+sid.ne:]
    print(np.max(cb), np.min(cb))
    return cb

def solve_diffusion_fracture(sid: SimInputData, inc: Incidence, graph: Graph, edges: Edges, cb_vector):
    edges.alpha = 1 * (inc.boundary.T @ (edges.grain > 0)) / 2 * edges.active
    lam_plus_val = sid.Pe / 2 / edges.diams ** 2 * \
        (np.sqrt(np.abs(edges.flow) ** 2 + 4 * edges.alpha * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) + np.abs(edges.flow))
    lam_plus_val = np.array(np.ma.fix_invalid(lam_plus_val, fill_value = 0))
    lam_minus_val = sid.Pe / 2 / edges.diams ** 2 * \
        (np.sqrt(np.abs(edges.flow) ** 2 + 4 * edges.alpha * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) - np.abs(edges.flow)) #* (edges.diams <= sid.dmax)
    lam_minus_val = np.array(np.ma.fix_invalid(lam_minus_val, fill_value = 0))
    #lam_plus_zero = 1 * (1 * (lam_plus_val > sid.diffusion_exp_limit) + 1 * (edges.alpha == 0) != 0)
    lam_plus_zero = 1 * (1 * (lam_plus_val > sid.diffusion_exp_limit) + 1 * (edges.diams == 0) > 0)
    lam_plus_val = lam_plus_val * (1 - lam_plus_zero)
    lam_minus_val = lam_minus_val * (1 - lam_plus_zero)

    exp_plus = spr.diags(np.exp(lam_plus_val * edges.lens) * (1 - lam_plus_zero) + lam_plus_zero)
    exp_plus2 = spr.diags(np.exp(lam_plus_val * edges.lens) * (1 - lam_plus_zero))
    exp_minus = spr.diags(np.exp(-lam_minus_val * edges.lens))
    exp_minus2 = spr.diags(np.exp(-lam_minus_val * edges.lens) * (1 - lam_plus_zero))
    
    lam_minus_val = lam_minus_val * (1 - lam_plus_zero)
    print(f'lam plus zero: {np.sum(lam_plus_zero), np.sum(1 - lam_plus_zero)}')
    print(lam_plus_val[np.where((edges.flow ==0) * (lam_plus_val > 0))[0]])
    #print(lam_minus_val[np.where((edges.flow ==0) * (lam_plus_val > 0))[0]])
    # when flow == 0, we need purely diffusive flux
    upstream = 1 * (spr.diags(edges.flow) @ inc.incidence > 0) + 1 * (spr.diags(1 * (edges.flow == 0) * (edges.diams > 0)) @ inc.incidence > 0)
    downstream = 1 * (spr.diags(edges.flow) @ inc.incidence < 0) + 1 * (spr.diags(1 * (edges.flow == 0) * (edges.diams > 0)) @ inc.incidence < 0)
    downstream2 = 1 * (1 * (spr.diags(edges.flow) @ inc.incidence < 0) + 1 * (spr.diags(1 * (edges.flow == 0) * (edges.diams > 0)) @ inc.incidence < 0)).multiply((1 - lam_plus_zero)[:, np.newaxis])
    #cb_pe_fix = 1 * (downstream.T @ (1 - lam_plus_zero) == 0)
    flux_a = 1 * (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_plus2 @ downstream + (spr.diags(lam_plus_val * edges.diams ** 2)) @ upstream - (exp_plus * spr.diags(lam_plus_val * edges.diams ** 2)) @ downstream).multiply((1 - lam_plus_zero)[:, np.newaxis])
    #flux_b = 1 * (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_minus @ downstream + (spr.diags(-lam_minus_val * edges.diams ** 2)) @ upstream + (exp_minus * spr.diags(lam_minus_val * edges.diams ** 2)) @ downstream)
    flux_b = 1 * (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_minus2 @ downstream + (spr.diags(-lam_minus_val * edges.diams ** 2)) @ upstream + (exp_minus2 * spr.diags(lam_minus_val * edges.diams ** 2)) @ downstream).multiply((1 - lam_plus_zero)[:, np.newaxis])
    exp_pe_fix = np.exp(-edges.alpha * sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow))
    exp_pe_fix = np.array(np.ma.fix_invalid(exp_pe_fix, fill_value = 0))
    flux_b += 1 * (sid.Pe * spr.diags(np.abs(edges.flow)) @ spr.diags(lam_plus_zero * exp_pe_fix) @ downstream)
    flux_a_in = flux_a.T.multiply((1 - graph.in_vec)[:, np.newaxis])
    
    flux_b_in = flux_b.T.multiply((1 - graph.in_vec)[:, np.newaxis])

    flow_fix_pe = -sid.Pe * downstream.T @ np.abs(edges.flow)
    flow_fix_pe = flow_fix_pe * (1 - graph.in_vec) + graph.in_vec
    zero_flow_fix = 1 * (edges.flow == 0) * (edges.alpha == 0) * (edges.diams > 0) # where the flow is zero and alpha is zero, we solve a different equation: d2c/dx2 = 0, with c(0) = c_up and c(l) = c_down
    #flow_fix_pe += 1 * (flow_fix_pe == 0) * (downstream.T @ (1 * (edges.diams == 0)) != 0)
    # what are the equations when flow == 0?
    cb_matrix = spr.vstack([spr.hstack([spr.diags(flow_fix_pe), flux_a_in, flux_b_in]), \
                    spr.hstack([-downstream2, exp_plus + spr.diags((edges.lens -  1) * zero_flow_fix), exp_minus2]), \
                    spr.hstack([-upstream, spr.diags(np.ones(sid.ne) - zero_flow_fix), spr.diags(np.ones(sid.ne))]) \
                    ])
    

    cb_matrix += spr.diags(1 * (np.array(np.sum(np.abs(cb_matrix), axis = 1))[:, 0] == 0))

    res = solve_equation(cb_matrix, cb_vector)
    cb = res[:sid.nsq]
    edges.A = res[sid.nsq:sid.nsq+sid.ne]
    edges.B = res[sid.nsq+sid.ne:]
    J_in = np.sum(edges.inlet * (np.abs(edges.flow) * (edges.A * (1 - lam_plus_zero) + edges.B) - (1 - lam_plus_zero) / sid.Pe * edges.diams ** 2 * (lam_plus_val * edges.A - lam_minus_val * edges.B)))
    cb = cb / J_in * sid.cb_0 * sid.Q_in
    edges.A = edges.A / J_in * sid.cb_0 * sid.Q_in
    edges.B = edges.B / J_in * sid.cb_0 * sid.Q_in

    print(np.max(cb), np.min(cb))

    return cb