
from config import SimInputData
from incidence import Incidence
from network import Graph, Edges
from utils import solve_equation

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as spr

def create_vector(sid: SimInputData, graph: Graph) -> spr.csc_matrix:
    """ Create vector result for B concentration calculation.
    """
    return np.concatenate([sid.Jb_in * graph.in_vec, np.zeros(2 * sid.ne)])

def solve_diffusion(sid: SimInputData, inc: Incidence, graph: Graph, edges: Edges, cb_vector):
    lam_plus_val = sid.Pe / 2 * edges.lens / edges.diams ** 2 * \
        (np.sqrt(np.abs(edges.flow) ** 2 + 4 * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) + np.abs(edges.flow))
    lam_plus_val = np.array(np.ma.fix_invalid(lam_plus_val, fill_value = 0))
    lam_minus_val = sid.Pe / 2 * edges.lens / edges.diams ** 2 * \
        (np.sqrt(np.abs(edges.flow) ** 2 + 4 * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) - np.abs(edges.flow))
    lam_minus_val = np.array(np.ma.fix_invalid(lam_minus_val, fill_value = 0))
    exp_plus = spr.diags(np.exp(lam_plus_val))
    exp_minus = spr.diags(np.exp(-lam_minus_val))
    lam_plus = spr.diags(lam_plus_val)
    lam_minus = spr.diags(lam_minus_val)
    upstream = 1 * (spr.diags(edges.flow) @ inc.incidence > 0)
    downstream = 1 * (spr.diags(edges.flow) @ inc.incidence < 0)
    flux_a = 1 * ((spr.diags(lam_plus_val * edges.diams ** 2)) @ upstream - (exp_plus * spr.diags(lam_plus_val * edges.diams ** 2)) @ downstream)
    flux_b = 1 * ((spr.diags(-lam_minus_val * edges.diams ** 2)) @ upstream + (exp_minus * spr.diags(lam_minus_val * edges.diams ** 2)) @ downstream)
    #flux_a_in = flux_a.T.multiply((1 - (graph.in_vec + graph.out_vec))[:, np.newaxis]) + (flux_a / sid.Pe + (spr.diags(np.abs(edges.flow)) @ upstream)).T.multiply(graph.in_vec[:, np.newaxis]) + ((exp_plus * spr.diags(lam_plus_val)) @ downstream).T.multiply(graph.out_vec[:, np.newaxis])
    #flux_a_in = flux_a.T.multiply((1 - (graph.in_vec + graph.out_vec))[:, np.newaxis]) + upstream.T.multiply(graph.in_vec[:, np.newaxis]) + ((exp_plus * spr.diags(lam_plus_val)) @ downstream).T.multiply(graph.out_vec[:, np.newaxis])
    flux_a_in = flux_a.T.multiply((1 - (graph.in_vec + graph.out_vec))[:, np.newaxis]) + ((exp_plus * spr.diags(lam_plus_val)) @ downstream).T.multiply(graph.out_vec[:, np.newaxis])
    #flux_b_in = flux_b.T.multiply((1 - (graph.in_vec + graph.out_vec))[:, np.newaxis]) + (flux_b / sid.Pe + (spr.diags(np.abs(edges.flow)) @ upstream)).T.multiply(graph.in_vec[:, np.newaxis]) - ((exp_minus * spr.diags(lam_minus_val)) @ downstream).T.multiply(graph.out_vec[:, np.newaxis])
    #flux_b_in = flux_b.T.multiply((1 - (graph.in_vec + graph.out_vec))[:, np.newaxis]) + upstream.T.multiply(graph.in_vec[:, np.newaxis]) - ((exp_minus * spr.diags(lam_minus_val)) @ downstream).T.multiply(graph.out_vec[:, np.newaxis])
    flux_b_in = flux_b.T.multiply((1 - (graph.in_vec + graph.out_vec))[:, np.newaxis]) - ((exp_minus * spr.diags(lam_minus_val)) @ downstream).T.multiply(graph.out_vec[:, np.newaxis])
    #print(flux_a_in.shape, flux_b_in.shape)
    cb_matrix = spr.vstack([spr.hstack([spr.diags(graph.in_vec), flux_a_in, flux_b_in]), \
                    spr.hstack([-upstream, spr.diags(np.ones(sid.ne)), spr.diags(np.ones(sid.ne))]), \
                    spr.hstack([-downstream, exp_plus, exp_minus])])
    cb_matrix = spr.diags(1 - inc.merge_vec) @ cb_matrix @ spr.diags(1 - inc.merge_vec) + spr.diags(inc.merge_vec)
    diag = cb_matrix.diagonal()
    diag_old = diag.copy()
    for node in np.where(cb_matrix.sum(axis = 1) == 0)[0]:
        diag[node] = 1
    cb_matrix += spr.diags(diag - diag_old)
    res = solve_equation(cb_matrix, cb_vector)
    cb = res[:sid.nsq]
    edges.A = res[sid.nsq:sid.nsq+sid.ne]
    edges.B = res[sid.nsq+sid.ne:]
    #print(np.where(cb_matrix.sum(axis = 1) == 0))
    #np.savetxt('cbm.txt', cb_matrix.toarray())
    #np.savetxt('cb.txt', cb)
    return cb

def solve_diffusion_pe_fix(sid: SimInputData, inc: Incidence, graph: Graph, edges: Edges, cb_vector):
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
    
    #lam_plus = spr.diags(lam_plus_val)
    #lam_minus = spr.diags(lam_minus_val)
    #print(np.sum(lam_plus_zero))
    #np.savetxt('expp.txt', exp_plus.toarray())
    upstream = 1 * (spr.diags(edges.flow) @ inc.incidence > 0) + 1 * (spr.diags(1 * (edges.flow == 0)) @ inc.incidence > 0)
    downstream = 1 * (spr.diags(edges.flow) @ inc.incidence < 0)
    downstream2 = 1 * ((spr.diags(edges.flow) @ inc.incidence < 0).multiply((1 - lam_plus_zero)[:, np.newaxis]))
    #cb_pe_fix = 1 * (downstream.T @ (1 - lam_plus_zero) == 0)
    flux_a = 1 * (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_plus2 @ downstream + (spr.diags(lam_plus_val * edges.diams ** 2)) @ upstream - (exp_plus * spr.diags(lam_plus_val * edges.diams ** 2)) @ downstream).multiply((1 - lam_plus_zero)[:, np.newaxis])
    flux_b = 1 * (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_minus @ downstream + (spr.diags(-lam_minus_val * edges.diams ** 2)) @ upstream + (exp_minus * spr.diags(lam_minus_val * edges.diams ** 2)) @ downstream)
    #flux_a_in = flux_a.T.multiply((1 - (graph.in_vec + graph.out_vec))[:, np.newaxis]) + (flux_a / sid.Pe + (spr.diags(np.abs(edges.flow)) @ upstream)).T.multiply(graph.in_vec[:, np.newaxis]) + ((exp_plus * spr.diags(lam_plus_val)) @ downstream).T.multiply(graph.out_vec[:, np.newaxis])
    #flux_a_in = flux_a.T.multiply((1 - (graph.in_vec + graph.out_vec))[:, np.newaxis]) + upstream.T.multiply(graph.in_vec[:, np.newaxis]) + ((exp_plus * spr.diags(lam_plus_val)) @ downstream).T.multiply(graph.out_vec[:, np.newaxis])
    #flux_a_in = flux_a.T.multiply((1 - (graph.in_vec + graph.out_vec))[:, np.newaxis]) #+ ((exp_plus * spr.diags(lam_plus_val)) @ downstream).T.multiply((graph.out_vec * (1 - cb_pe_fix))[:, np.newaxis])
    flux_a_in = flux_a.T.multiply((1 - graph.in_vec)[:, np.newaxis]) #+ ((exp_plus * spr.diags(lam_plus_val)) @ downstream).T.multiply((graph.out_vec * (1 - cb_pe_fix))[:, np.newaxis])
    
    
    #flux_b_in = flux_b.T.multiply((1 - (graph.in_vec + graph.out_vec))[:, np.newaxis]) + (flux_b / sid.Pe + (spr.diags(np.abs(edges.flow)) @ upstream)).T.multiply(graph.in_vec[:, np.newaxis]) - ((exp_minus * spr.diags(lam_minus_val)) @ downstream).T.multiply(graph.out_vec[:, np.newaxis])
    #flux_b_in = flux_b.T.multiply((1 - (graph.in_vec + graph.out_vec))[:, np.newaxis]) + upstream.T.multiply(graph.in_vec[:, np.newaxis]) - ((exp_minus * spr.diags(lam_minus_val)) @ downstream).T.multiply(graph.out_vec[:, np.newaxis])
    #flux_b_in = flux_b.T.multiply((1 - (graph.in_vec + graph.out_vec))[:, np.newaxis]) #- ((exp_minus * spr.diags(lam_minus_val)) @ downstream).T.multiply((graph.out_vec * (1 - cb_pe_fix))[:, np.newaxis]) #- (spr.diags(np.abs(edges.flow)) @ exp_minus @ downstream).T.multiply((graph.out_vec * cb_pe_fix)[:, np.newaxis])
    flux_b_in = flux_b.T.multiply((1 - graph.in_vec)[:, np.newaxis]) #- ((exp_minus * spr.diags(lam_minus_val)) @ downstream).T.multiply((graph.out_vec * (1 - cb_pe_fix))[:, np.newaxis]) #- (spr.diags(np.abs(edges.flow)) @ exp_minus @ downstream).T.multiply((graph.out_vec * cb_pe_fix)[:, np.newaxis])
    #flux_a_in = 0 * flux_b_in
    
    #print(flux_a_in.shape, flux_b_in.shape)
    #flow_fix_pe = -sid.Pe * np.abs(inc.incidence.T) @ np.abs(edges.flow) / 2
    #flow_fix_pe = flow_fix_pe * (1 - graph.in_vec + graph.out_vec) + graph.in_vec
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
        print(node)
    cb_matrix += spr.diags(diag - diag_old)
    res = solve_equation(cb_matrix, cb_vector)
    cb = res[:sid.nsq]
    edges.A = res[sid.nsq:sid.nsq+sid.ne]
    edges.B = res[sid.nsq+sid.ne:]
    #np.savetxt('cb.txt', cb)
    #np.savetxt('cbm.txt', cb_matrix.toarray())
    #np.savetxt('lam.txt', lam_plus_zero)
    print(np.max(cb), np.min(cb))
    return cb