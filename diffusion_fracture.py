
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
