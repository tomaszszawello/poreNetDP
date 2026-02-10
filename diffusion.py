
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
    qc_in = sid.Pe * (np.abs(inc.incidence).T @ np.abs(edges.flow)) * np.concatenate([np.array([1]), np.zeros(sid.nsq - 1)]) # shape: nsq
    return np.concatenate([sid.cb_0 * qc_in, np.zeros(2 * sid.ne)])

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
    edges.alpha = 1 * (inc.boundary.T @ (edges.grain > 0)) / 2 * edges.active
    print(np.sum(edges.alpha))
    print(np.sum(edges.active))
    lam_plus_val = sid.Pe / 2 * edges.lens / edges.diams ** 2 * \
        (np.sqrt(np.abs(edges.flow) ** 2 + 4 * edges.alpha * sid.ksi * edges.diams ** 2) + np.abs(edges.flow))
    lam_plus_val = np.array(np.ma.fix_invalid(lam_plus_val, fill_value = 0))
    lam_minus_val = sid.Pe / 2 * edges.lens / edges.diams ** 2 * \
        (np.sqrt(np.abs(edges.flow) ** 2 + 4 * edges.alpha * sid.ksi * edges.diams ** 2) - np.abs(edges.flow))
    lam_minus_val = np.array(np.ma.fix_invalid(lam_minus_val, fill_value = 0))
    lam_plus_zero = 1 * (1 * (lam_plus_val > sid.diffusion_exp_limit) + 1 * (edges.diams == 0) + 1 * (edges.flow == 0) > 0)
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
    upstream = 1 * (spr.diags(edges.flow) @ inc.incidence > 0) + 1 * (spr.diags(1 * (edges.flow == 0)) @ inc.incidence < 0)
    downstream = 1 * (spr.diags(edges.flow) @ inc.incidence < 0) + 1 * (spr.diags(1 * (edges.flow == 0)) @ inc.incidence > 0)
    downstream2 = 1 * ((spr.diags(edges.flow) @ inc.incidence < 0).multiply((1 - lam_plus_zero)[:, np.newaxis])) + 1 * (spr.diags(1 * (edges.flow == 0)) @ inc.incidence > 0)
    #cb_pe_fix = 1 * (downstream.T @ (1 - lam_plus_zero) == 0)
    flux_a = 1 * (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_plus2 @ downstream + (spr.diags(lam_plus_val * edges.diams ** 2)) @ upstream - (exp_plus * spr.diags(lam_plus_val * edges.diams ** 2)) @ downstream).multiply((1 - lam_plus_zero)[:, np.newaxis])
    flux_b = 1 * (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_minus @ downstream + (spr.diags(-lam_minus_val * edges.diams ** 2)) @ upstream + (exp_minus * spr.diags(lam_minus_val * edges.diams ** 2)) @ downstream)
    exp_pe_fix = np.exp(-edges.alpha * sid.ksi * edges.lens / np.abs(edges.flow))
    exp_pe_fix = np.array(np.ma.fix_invalid(exp_pe_fix, fill_value = 0))
    flux_b += 1 * (sid.Pe * spr.diags(np.abs(edges.flow)) @ spr.diags((lam_plus_val > sid.diffusion_exp_limit) * exp_pe_fix) @ downstream)

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
    #cb_matrix = spr.diags(1 - inc.merge_vec) @ cb_matrix @ spr.diags(1 - inc.merge_vec) + spr.diags(inc.merge_vec)
    
    diag = cb_matrix.diagonal()
    diag_old = diag.copy()
    for node in np.where(np.abs(cb_matrix).sum(axis = 1) == 0)[0]:
        diag[node] = 1
        #print(node)
    # for node in np.where(np.abs(cb_matrix).sum(axis = 0) == 0)[1]:
    #     diag[node] = 1
    #     print(node)
    #diag += 1 * (diag == 0)
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

def solve_diffusion_sinks(sid: SimInputData, inc: Incidence, graph: Graph, edges: Edges, cb_vector):
    edges.alpha = 1 * ((inc.boundary.T @ edges.grain) > 0)
    #print(edges.alpha)
    lam_plus_val = sid.Pe * edges.lens / edges.diams ** 2 * np.abs(edges.flow)
    lam_plus_val += np.sqrt(edges.alpha * sid.Da2 * edges.lens ** 2 / edges.diams / (1 + sid.G * edges.diams)) * (edges.flow == 0)
    lam_plus_val = np.array(np.ma.fix_invalid(lam_plus_val, fill_value = 0))    
    lam_minus_val = 0
    lam_minus_val += np.sqrt(edges.alpha * sid.Da2 * edges.lens ** 2 / edges.diams / (1 + sid.G * edges.diams)) * (edges.flow == 0)
    lam_minus_val = np.array(np.ma.fix_invalid(lam_minus_val, fill_value = 0))
    pure_diff = 1 * (edges.alpha == 0) * (inc.boundary.T @ np.ones(sid.ntr) > 0) + 1 * edges.active * (edges.flow == 0)
    #print(pure_diff)
    lam_plus_zero = 1 * (1 * (lam_plus_val > sid.diffusion_exp_limit) + 1 * (edges.diams == 0) + pure_diff > 0)
    lam_plus_val = lam_plus_val * (1 - lam_plus_zero)

    exp_plus = spr.diags(np.exp(lam_plus_val) * (1 - lam_plus_zero) + lam_plus_zero * (1 - pure_diff) + pure_diff * edges.lens)
    exp_plus2 = spr.diags(np.exp(lam_plus_val) * (1 - lam_plus_zero))
    exp_minus = spr.diags(np.exp(-lam_minus_val) * (1 - pure_diff))
    exp_minus2 = spr.diags(np.exp(-lam_minus_val) * (1 - lam_plus_zero) + pure_diff)
    
    lam_minus_val = lam_minus_val * (1 - lam_plus_zero)
    
    upstream = 1 * (spr.diags(edges.flow) @ inc.incidence > 0) + 1 * (spr.diags(1 * (edges.flow == 0)) @ inc.incidence < 0)
    downstream = 1 * (spr.diags(edges.flow) @ inc.incidence < 0) + 1 * (spr.diags(1 * (edges.flow == 0)) @ inc.incidence > 0)
    downstream2 = 1 * ((spr.diags(edges.flow) @ inc.incidence < 0).multiply((1 - lam_plus_zero)[:, np.newaxis])) + 1 * (spr.diags(1 * (edges.flow == 0)) @ inc.incidence > 0)
    #cb_pe_fix = 1 * (downstream.T @ (1 - lam_plus_zero) == 0)
    flux_a = 1 * (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_plus2 @ downstream + (spr.diags(lam_plus_val * edges.diams ** 2)) @ upstream - (exp_plus * spr.diags(lam_plus_val * edges.diams ** 2)) @ downstream).multiply((1 - lam_plus_zero)[:, np.newaxis]) + spr.diags(pure_diff * edges.diams ** 2) @ upstream - spr.diags(pure_diff * edges.diams ** 2) @ downstream
    flux_b = 1 * (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_minus @ downstream + (spr.diags(-lam_minus_val * edges.diams ** 2)) @ upstream + (exp_minus * spr.diags(lam_minus_val * edges.diams ** 2)) @ downstream)
    flux_a_in = flux_a.T.multiply((1 - graph.in_vec)[:, np.newaxis]) #+ ((exp_plus * spr.diags(lam_plus_val)) @ downstream).T.multiply((graph.out_vec * (1 - cb_pe_fix))[:, np.newaxis])
    flux_b_in = flux_b.T.multiply((1 - graph.in_vec)[:, np.newaxis]) #- ((exp_minus * spr.diags(lam_minus_val)) @ downstream).T.multiply((graph.out_vec * (1 - cb_pe_fix))[:, np.newaxis]) #- (spr.diags(np.abs(edges.flow)) @ exp_minus @ downstream).T.multiply((graph.out_vec * cb_pe_fix)[:, np.newaxis])
    flow_fix_pe = -sid.Pe * downstream.T @ np.abs(edges.flow)
    flow_fix_pe = flow_fix_pe * (1 - graph.in_vec) + graph.in_vec
    #print(len(graph.in_vec), len(lam_minus_val), len(lam_plus_val))
    cb_matrix = spr.vstack([spr.hstack([spr.diags(flow_fix_pe), flux_a_in, flux_b_in]), \
                    spr.hstack([-downstream2, exp_plus, exp_minus2]), \
                    spr.hstack([-upstream, spr.diags(np.ones(sid.ne) - lam_plus_zero), spr.diags(np.ones(sid.ne))]) \
                    ])
    #cb_matrix = spr.diags(1 - inc.merge_vec) @ cb_matrix @ spr.diags(1 - inc.merge_vec) + spr.diags(inc.merge_vec)
    
    diag = cb_matrix.diagonal()
    diag_old = diag.copy()
    for node in np.where(np.abs(cb_matrix).sum(axis = 1) == 0)[0]:
        diag[node] = 1
        #print(node)
    # for node in np.where(np.abs(cb_matrix).sum(axis = 0) == 0)[1]:
    #     diag[node] = 1
    #     print(node)
    #diag += 1 * (diag == 0)
    #print(diag)
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

import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix

def find_linearly_dependent_rows_csr(M: csr_matrix, print_details=True):
    """
    Finds all pairs of rows in a CSR matrix M that are direct scalar multiples
    of each other (i.e. linearly dependent pairs).

    Parameters
    ----------
    M : csr_matrix
        Sparse matrix in CSR format.
    print_details : bool
        If True, print out the row indices that match, and the pattern of columns/data.

    Returns
    -------
    dependent_pairs : list of lists
        A list where each element is a list of row indices that share the same signature
        (i.e. all are scalar multiples of each other).
        Example: [[2, 10, 15], [5, 7]] means row 2,10,15 are multiples of each other
        and row 5,7 are multiples of each other, etc.
    """
    #if not isinstance(M, csr_matrix):
    #    raise ValueError("M must be a csr_matrix.")

    row_dict = defaultdict(list)  # signature -> list of row indices
    n_rows = M.shape[0]

    for i in range(n_rows):
        start = M.indptr[i]
        end   = M.indptr[i+1]
        cols = M.indices[start:end]
        vals = M.data[start:end]

        if len(cols) == 0:
            # This row is entirely zero
            signature = ((), ())
        else:
            pivot = vals[0]  # first nonzero
            scaled_vals = vals / pivot
            signature = (tuple(cols), tuple(np.round(scaled_vals, decimals=15)))
            # Rounding can help avoid floating-point comparison issues if your data is floating

        row_dict[signature].append(i)

    # Collect all row-groups that have more than one row
    dependent_pairs = []
    for signature, rows in row_dict.items():
        if len(rows) > 1:
            dependent_pairs.append(rows)
            if print_details:
                print(f"Rows {rows} are scalar multiples of each other.")
                if len(signature[0]) == 0:
                    print("  => They are all-zero rows.")
                else:
                    print(f"  Columns: {signature[0]}")
                    print(f"  Scaled data pattern: {signature[1]}")
                print()

    return dependent_pairs


def solve_diffusion_fracture(sid: SimInputData, inc: Incidence, graph: Graph, edges: Edges, cb_vector):
    edges.alpha = 1 * ((inc.boundary.T @ edges.grain) > 0)
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

def solve_diffusion_vol(sid: SimInputData, inc: Incidence, graph: Graph, edges: Edges, vols: Volumes, cb_vector, data):
    edges.alpha = 1 * ((vols.triangles @ vols.vol_a) > 0)
    print(np.where(edges.alpha != 1))
    # print(np.where(alpha != 1)[0].shape)
    # print(alpha)
    lam_plus_val = sid.Pe / 2 / edges.diams ** 2 * \
        (np.sqrt(np.abs(edges.flow) ** 2 + 4 * edges.alpha * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) + np.abs(edges.flow))
    lam_plus_val = np.array(np.ma.fix_invalid(lam_plus_val, fill_value = 0))
    lam_minus_val = sid.Pe / 2 / edges.diams ** 2 * \
        (np.sqrt(np.abs(edges.flow) ** 2 + 4 * edges.alpha * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) - np.abs(edges.flow)) #* (edges.diams <= sid.dmax)
    lam_minus_val = np.array(np.ma.fix_invalid(lam_minus_val, fill_value = 0))
    #lam_plus_zero = 1 * (1 * (lam_plus_val > sid.diffusion_exp_limit) + 1 * (edges.alpha == 0) != 0)
    lam_plus_zero = 1 * (lam_plus_val > sid.diffusion_exp_limit)
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
    cb_matrix = spr.diags(1 - inc.merge_vec) @ cb_matrix @ spr.diags(1 - inc.merge_vec) + spr.diags(inc.merge_vec)
    #cb_matrix = spr.diags(1 * (edges.diams > 0)) @ cb_matrix @ spr.diags(1 * (edges.diams > 0)) + spr.diags(1 * (edges.diams > 0))
    
    #diag = cb_matrix.diagonal()
    #diag_old = diag.copy()

    #diag += 1 * (diag == 0)
    cb_matrix += spr.diags(1 * (np.array(np.sum(np.abs(cb_matrix), axis = 1))[:, 0] == 0))
    # print(cb_matrix[np.where(cb_matrix.diagonal() == 0)[0]]) # why it does not show anything when the matrix is singular?
    # print(inc.incidence.T[92])
    # print(inc.incidence.T[101])
    # print(vols.triangles[6540])
    # print(vols.triangles[6547])
    res = solve_equation(cb_matrix, cb_vector)
    cb = res[:sid.nsq]
    edges.A = res[sid.nsq:sid.nsq+sid.ne]
    edges.B = res[sid.nsq+sid.ne:]
    J_in = np.sum(edges.inlet * (np.abs(edges.flow) * (edges.A * (1 - lam_plus_zero) + edges.B) - (1 - lam_plus_zero) / sid.Pe * edges.diams ** 2 * (lam_plus_val * edges.A - lam_minus_val * edges.B)))
    cb = cb / J_in * sid.cb_0 * sid.Q_in
    edges.A = edges.A / J_in * sid.cb_0 * sid.Q_in
    edges.B = edges.B / J_in * sid.cb_0 * sid.Q_in
    data.J_in = np.sum(edges.inlet * (np.abs(edges.flow) * (edges.A * (1 - lam_plus_zero) + edges.B) - (1 - lam_plus_zero) / sid.Pe * edges.diams ** 2 * (lam_plus_val * edges.A - lam_minus_val * edges.B)))
    
    data.J_out = np.abs(1 * ( inc.incidence.T @ spr.diags(edges.flow) < 0) @ (np.abs(edges.flow) \
                 * edges.outlet)) @ cb
    # ind = np.where((edges.flow ==0) * (lam_plus_val > 0))[0]
    # if len(ind):
    #     print(edges.A[ind])
    #     print(edges.B[ind])
    #     print((upstream @ cb - (edges.A + edges.B))[ind])
    #     print((np.abs(edges.flow) * (downstream @ cb - (1 - lam_plus_zero) * (edges.A * np.exp(lam_plus_val * edges.lens) + edges.B * np.exp(-lam_minus_val * edges.lens))))[ind])

    # flux = spr.diags(flow_fix_pe) @ cb + flux_a_in @ edges.A + flux_b_in @ edges.B
    # print(f'Flux 92: {flux[92]}')
    # print(f'Flux 101: {flux[101]}')
    # print(f'Flux 101: {(cb_matrix @ (res / J_in * sid.cb_0 * sid.Q_in))[101]}')
    # print(np.where(inc.merge_vec * (np.concatenate([np.zeros(sid.nsq), edges.diams, edges.diams]))))
    #print(f'Flux in: {np.sum(edges.inlet * (np.abs(edges.flow) * (edges.A * (1 - lam_plus_zero) + edges.B) - (1 - lam_plus_zero) / sid.Pe * edges.diams ** 2 * (lam_plus_val * edges.A - lam_minus_val * edges.B)))}')
    # outlet_diff = np.sum(np.abs(edges.flow) * (downstream @ cb - (1 - lam_plus_zero) * (edges.A * np.exp(lam_plus_val * edges.lens) + edges.B * np.exp(-lam_minus_val * edges.lens))))
    # outlet_flow =  -lam_plus_zero * edges.B * np.exp(-edges.alpha * sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow))
    # outlet_flow = np.array(np.ma.fix_invalid(outlet_flow, fill_value = 0))
    # outlet_diff += outlet_flow
    #print(f'inlet difference: {np.sum(upstream @ cb - (edges.A + edges.B))}, outlet difference: {np.sum(outlet_diff)}')
    # print(inc.merge_vec[92], inc.merge_vec[101], inc.merge_vec[sid.nsq + 6540], inc.merge_vec[sid.nsq + 6547])
    print(np.max(cb), np.min(cb))

    return cb



def solve_vol_nr(sid: SimInputData, inc: Incidence, graph: Graph, \
    edges: Edges, vols: Volumes, cb_vector, data) -> np.ndarray:
    """ Calculate B concentration with tracking of A volume.
    """
    alpha = 1 * ((vols.triangles @ vols.vol_a) > 0)
    print(alpha)
    print(f'problems... {np.where(vols.triangles @ (vols.vol_a == 0) != 0)}')
    alpha_prev = np.zeros(sid.ne)
    it_alpha = 0
    alpha_tr = 1 * (vols.vol_a > 0) # vector scaling the reaction constants in
    # triangles according to A availibility
    alpha_tr_prev = np.zeros(sid.ntr)
    #print(np.where(alpha != 1))
    # print(np.where(alpha != 1)[0].shape)
    # print(alpha)
    lam_plus_val = sid.Pe / 2 / edges.diams ** 2 * \
        (np.sqrt(np.abs(edges.flow) ** 2 + 4 * alpha * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) + np.abs(edges.flow))
    lam_plus_val = np.array(np.ma.fix_invalid(lam_plus_val, fill_value = 0))
    lam_minus_val = sid.Pe / 2 / edges.diams ** 2 * \
        (np.sqrt(np.abs(edges.flow) ** 2 + 4 * alpha * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) - np.abs(edges.flow)) #* (edges.diams <= sid.dmax)
    lam_minus_val = np.array(np.ma.fix_invalid(lam_minus_val, fill_value = 0))
    #lam_plus_zero = 1 * (1 * (lam_plus_val > sid.diffusion_exp_limit) + 1 * (edges.alpha == 0) != 0)
    lam_plus_zero = 1 * (lam_plus_val > sid.diffusion_exp_limit)
    print(f'lam plus zero: {np.sum(lam_plus_zero), np.sum(1 - lam_plus_zero)}')
    lam_plus_val = lam_plus_val * (1 - lam_plus_zero)
    lam_minus_val = lam_minus_val * (1 - lam_plus_zero)

    exp_plus = spr.diags(np.exp(lam_plus_val * edges.lens) * (1 - lam_plus_zero) + lam_plus_zero)
    exp_plus2 = spr.diags(np.exp(lam_plus_val * edges.lens) * (1 - lam_plus_zero))
    exp_minus = spr.diags(np.exp(-lam_minus_val * edges.lens))
    exp_minus2 = spr.diags(np.exp(-lam_minus_val * edges.lens) * (1 - lam_plus_zero))
    
    lam_minus_val = lam_minus_val * (1 - lam_plus_zero)

    upstream = 1 * (spr.diags(edges.flow) @ inc.incidence > 0) + 1 * (spr.diags(1 * (edges.flow == 0) * (edges.diams > 0)) @ inc.incidence > 0)
    downstream = 1 * (spr.diags(edges.flow) @ inc.incidence < 0) + 1 * (spr.diags(1 * (edges.flow == 0) * (edges.diams > 0)) @ inc.incidence < 0)
    downstream2 = 1 * (1 * (spr.diags(edges.flow) @ inc.incidence < 0) + 1 * (spr.diags(1 * (edges.flow == 0) * (edges.diams > 0)) @ inc.incidence < 0)).multiply((1 - lam_plus_zero)[:, np.newaxis])
    #cb_pe_fix = 1 * (downstream.T @ (1 - lam_plus_zero) == 0)
    flux_a = 1 * (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_plus2 @ downstream + (spr.diags(lam_plus_val * edges.diams ** 2)) @ upstream - (exp_plus * spr.diags(lam_plus_val * edges.diams ** 2)) @ downstream).multiply((1 - lam_plus_zero)[:, np.newaxis])
    #flux_b = 1 * (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_minus @ downstream + (spr.diags(-lam_minus_val * edges.diams ** 2)) @ upstream + (exp_minus * spr.diags(lam_minus_val * edges.diams ** 2)) @ downstream)
    flux_b = 1 * (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_minus2 @ downstream + (spr.diags(-lam_minus_val * edges.diams ** 2)) @ upstream + (exp_minus2 * spr.diags(lam_minus_val * edges.diams ** 2)) @ downstream).multiply((1 - lam_plus_zero)[:, np.newaxis])
    exp_pe_fix = np.exp(-alpha * sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow))
    exp_pe_fix = np.array(np.ma.fix_invalid(exp_pe_fix, fill_value = 0))
    flux_b += 1 * (sid.Pe * spr.diags(np.abs(edges.flow)) @ spr.diags(lam_plus_zero * exp_pe_fix) @ downstream)
    flux_a_in = flux_a.T.multiply((1 - graph.in_vec)[:, np.newaxis])
    
    flux_b_in = flux_b.T.multiply((1 - graph.in_vec)[:, np.newaxis])

    flow_fix_pe = -sid.Pe * downstream.T @ np.abs(edges.flow)
    flow_fix_pe = flow_fix_pe * (1 - graph.in_vec) + graph.in_vec
    zero_flow_fix = 1 * (edges.flow == 0) * (alpha == 0) # where the flow is zero and alpha is zero, we solve a different equation: d2c/dx2 = 0, with c(0) = c_up and c(l) = c_down

    # what are the equations when flow == 0?
    cb_matrix = spr.vstack([spr.hstack([spr.diags(flow_fix_pe), flux_a_in, flux_b_in]), \
                    spr.hstack([-downstream2, exp_plus + spr.diags((edges.lens -  1) * zero_flow_fix), exp_minus2]), \
                    spr.hstack([-upstream, spr.diags(np.ones(sid.ne) - zero_flow_fix), spr.diags(np.ones(sid.ne))]) \
                    ])
    cb_matrix = spr.diags(1 - inc.merge_vec) @ cb_matrix @ spr.diags(1 - inc.merge_vec) + spr.diags(inc.merge_vec)

    cb_matrix += spr.diags(1 * (np.array(np.sum(np.abs(cb_matrix), axis = 1))[:, 0] == 0))
    res = solve_equation(cb_matrix, cb_vector)
    cb = res[:sid.nsq]
    edges.A = res[sid.nsq:sid.nsq+sid.ne]
    edges.B = res[sid.nsq+sid.ne:]

    J_in = np.sum(edges.inlet * (np.abs(edges.flow) * (edges.A * (1 - lam_plus_zero) + edges.B) - (1 - lam_plus_zero) / sid.Pe * edges.diams ** 2 * (lam_plus_val * edges.A - lam_minus_val * edges.B)))
    cb = cb / J_in * sid.cb_0 * sid.Q_in
    edges.A = edges.A  / J_in * sid.cb_0 * sid.Q_in
    edges.B = edges.B  / J_in * sid.cb_0 * sid.Q_in

    edge_vols = vols.triangles @ vols.vol_a
    #change = (1 - lam_plus_zero) / (sid.Da * edges.lens * edges.diams) * (edges.A * (np.exp(lam_plus_val) - 1) * (np.sqrt(np.abs(edges.flow) ** 2 + 4 * alpha * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) - np.abs(edges.flow)) + edges.B * (1 - np.exp(-lam_minus_val)) * (np.sqrt(np.abs(edges.flow) ** 2 + 4 * alpha * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) + np.abs(edges.flow))) + lam_plus_zero * edges.B * np.abs(edges.flow) / (sid.Da * edges.lens * edges.diams) * (1 - np.exp(-alpha * sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow)))
    change_pe_fix = lam_plus_zero * 2 * edges.B * np.abs(edges.flow) / sid.Da * (1 - np.exp(-alpha * sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow)))
    change_pe_fix = np.array(np.ma.fix_invalid(change_pe_fix, fill_value = 0))
    change = (1 - lam_plus_zero) * 2  * edges.diams ** 2 / (sid.Pe * sid.Da) * (edges.A * (np.exp(lam_plus_val * edges.lens) - 1) * lam_minus_val + edges.B * (1 - np.exp(-lam_minus_val * edges.lens)) * lam_plus_val) + change_pe_fix #+ change_zero_flow_fix

    change = np.array(np.ma.fix_invalid(change, fill_value = 0))
    #vol_a_dissolved = (spr.diags(vols.vol_a) @ vols.triangles.T) @ (change / edge_vols) * sid.dt
    vol_a_dissolved = vols.triangles.T @ (change / edges.triangles) * sid.dt
    f_alpha = vols.vol_a - vol_a_dissolved # check if overdissolved
    f_alpha_check = (f_alpha < 0) * (vols.vol_a > 0)

    # iterate using N-R until alpha_b_tr is the same (up to certain threshold)
    # in consecutive iterations; alpha_b for each edge is a function of
    # alpha_b_tr for the triangles neighbouring the edge, so we use matrix N-R
    # df(alpha i-1) @ delta_alpha = f(alpha i-1)
    # alpha i = alpha i-1 + delta_alpha
    while np.linalg.norm(alpha_tr - alpha_tr_prev) > sid.it_alpha_th:
        print(f'alpha diff: {np.linalg.norm(alpha_tr - alpha_tr_prev)}')
        alpha_tr_prev = alpha_tr.copy()
        
        #df_alpha_pe_fix = lam_plus_zero * 2 * edges.diams * edges.lens * edges.B * np.exp(-alpha * sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow)) * sid.dt / (1 + sid.G * edges.diams) / edges.triangles ** 2
        df_alpha_pe_fix = lam_plus_zero * 2 * edges.B * np.exp(-alpha * sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow)) * sid.dt / (1 + sid.G * edges.diams) / edges.triangles
        df_alpha_pe_fix = np.array(np.ma.fix_invalid(df_alpha_pe_fix, fill_value = 0.)) # fix
        #df_alpha = (-2 * (1 - lam_plus_zero) * edges.diams * edges.lens * (edges.A * np.exp(lam_plus_val * edges.lens) + edges.B * np.exp(-lam_minus_val * edges.lens)) * sid.dt / (1 + sid.G * edges.diams) / edges.triangles ** 2 - df_alpha_pe_fix) * (alpha > 0)
        df_alpha = (-2 * (1 - lam_plus_zero) * (edges.A * np.exp(lam_plus_val * edges.lens) + edges.B * np.exp(-lam_minus_val * edges.lens)) * sid.dt / (1 + sid.G * edges.diams) / edges.triangles - df_alpha_pe_fix) * (alpha > 0)
        df_alpha = np.array(np.ma.fix_invalid(df_alpha, fill_value = 0.)) # fix
        # for zero surface
        # we calculate delta_alpha only where overdissolved, that's why we use
        # f_alpha_check
        df_alpha_matrix = spr.diags(1 * f_alpha_check) @ vols.triangles.T @ \
            spr.diags(df_alpha) @ vols.triangles
        # we set rows without overdissolution to identity
        df_alpha_matrix += spr.diags(1 * (df_alpha_matrix.diagonal() == 0))
        delta_alpha = solve_equation(df_alpha_matrix, -f_alpha * f_alpha_check)
        # we clip the reaction rate to [0,1], as N-R sometimes overshoots and
        # we only want to slow down the reaction, not fasten
        alpha_tr = np.clip(alpha_tr + delta_alpha, 0, 1)

        #alpha_tr = np.array(np.ma.fix_invalid(alpha_tr, fill_value = 0.))
        alpha = np.array(np.ma.fix_invalid((vols.triangles @ (alpha_tr)) / edges.triangles, fill_value = 0.))
        #print(delta_alpha)
        #print(alpha_tr)
        #print(vols.vol_a)
        #print(vol_a_dissolved)
        #print(vols.triangles.T @ df_alpha)
        #print(alpha)
        # if np.sum(delta_alpha):
        #     np.savetxt('df.txt', df_alpha_matrix.toarray())
        #     raise ValueError
        # if alpha_b != identity, we recalculate B concentrations (which
        # change when we change alpha_b) and dissolved volumes and iterate
        # until alpha_b converges
        if np.sum(alpha) != np.sum(edges.diams > 0):
            lam_plus_val = sid.Pe / 2 / edges.diams ** 2 * \
                (np.sqrt(np.abs(edges.flow) ** 2 + 4 * alpha * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) + np.abs(edges.flow))
            lam_plus_val = np.array(np.ma.fix_invalid(lam_plus_val, fill_value = 0))
            lam_minus_val = sid.Pe / 2 / edges.diams ** 2 * \
                (np.sqrt(np.abs(edges.flow) ** 2 + 4 * alpha * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) - np.abs(edges.flow)) #* (edges.diams <= sid.dmax)
            lam_minus_val = np.array(np.ma.fix_invalid(lam_minus_val, fill_value = 0))
            #lam_plus_zero = 1 * (1 * (lam_plus_val > sid.diffusion_exp_limit) + 1 * (edges.alpha == 0) != 0)
            lam_plus_zero = 1 * (lam_plus_val > sid.diffusion_exp_limit)
            lam_plus_val = lam_plus_val * (1 - lam_plus_zero)
            lam_minus_val = lam_minus_val * (1 - lam_plus_zero)

            exp_plus = spr.diags(np.exp(lam_plus_val * edges.lens) * (1 - lam_plus_zero) + lam_plus_zero)
            exp_plus2 = spr.diags(np.exp(lam_plus_val * edges.lens) * (1 - lam_plus_zero))
            exp_minus = spr.diags(np.exp(-lam_minus_val * edges.lens))
            exp_minus2 = spr.diags(np.exp(-lam_minus_val * edges.lens) * (1 - lam_plus_zero))
            
            lam_minus_val = lam_minus_val * (1 - lam_plus_zero)

            upstream = 1 * (spr.diags(edges.flow) @ inc.incidence > 0) + 1 * (spr.diags(1 * (edges.flow == 0) * (edges.diams > 0)) @ inc.incidence > 0)
            downstream = 1 * (spr.diags(edges.flow) @ inc.incidence < 0) + 1 * (spr.diags(1 * (edges.flow == 0) * (edges.diams > 0)) @ inc.incidence < 0)
            downstream2 = 1 * (1 * (spr.diags(edges.flow) @ inc.incidence < 0) + 1 * (spr.diags(1 * (edges.flow == 0) * (edges.diams > 0)) @ inc.incidence < 0)).multiply((1 - lam_plus_zero)[:, np.newaxis])
            #cb_pe_fix = 1 * (downstream.T @ (1 - lam_plus_zero) == 0)
            flux_a = 1 * (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_plus2 @ downstream + (spr.diags(lam_plus_val * edges.diams ** 2)) @ upstream - (exp_plus * spr.diags(lam_plus_val * edges.diams ** 2)) @ downstream).multiply((1 - lam_plus_zero)[:, np.newaxis])
            #flux_b = 1 * (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_minus @ downstream + (spr.diags(-lam_minus_val * edges.diams ** 2)) @ upstream + (exp_minus * spr.diags(lam_minus_val * edges.diams ** 2)) @ downstream)
            flux_b = 1 * (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_minus2 @ downstream + (spr.diags(-lam_minus_val * edges.diams ** 2)) @ upstream + (exp_minus2 * spr.diags(lam_minus_val * edges.diams ** 2)) @ downstream).multiply((1 - lam_plus_zero)[:, np.newaxis])
            exp_pe_fix = np.exp(-alpha * sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow))
            exp_pe_fix = np.array(np.ma.fix_invalid(exp_pe_fix, fill_value = 0))
            flux_b += 1 * (sid.Pe * spr.diags(np.abs(edges.flow)) @ spr.diags(lam_plus_zero * exp_pe_fix) @ downstream)
            flux_a_in = flux_a.T.multiply((1 - graph.in_vec)[:, np.newaxis])
            
            flux_b_in = flux_b.T.multiply((1 - graph.in_vec)[:, np.newaxis])

            flow_fix_pe = -sid.Pe * downstream.T @ np.abs(edges.flow)
            flow_fix_pe = flow_fix_pe * (1 - graph.in_vec) + graph.in_vec
            zero_flow_fix = 1 * (edges.flow == 0) * (alpha == 0) # where the flow is zero and alpha is zero, we solve a different equation: d2c/dx2 = 0, with c(0) = c_up and c(l) = c_down

            # what are the equations when flow == 0?
            cb_matrix = spr.vstack([spr.hstack([spr.diags(flow_fix_pe), flux_a_in, flux_b_in]), \
                            spr.hstack([-downstream2, exp_plus + spr.diags((edges.lens -  1) * zero_flow_fix), exp_minus2]), \
                            spr.hstack([-upstream, spr.diags(np.ones(sid.ne) - zero_flow_fix), spr.diags(np.ones(sid.ne))]) \
                            ])
            cb_matrix = spr.diags(1 - inc.merge_vec) @ cb_matrix @ spr.diags(1 - inc.merge_vec) + spr.diags(inc.merge_vec)

            cb_matrix += spr.diags(1 * (np.array(np.sum(np.abs(cb_matrix), axis = 1))[:, 0] == 0))
            #J_in = np.sum(edges.inlet * (np.abs(edges.flow) * (edges.A * (1 - lam_plus_zero) + edges.B) - (1 - lam_plus_zero) / sid.Pe * edges.diams ** 2 * (lam_plus_val * edges.A - lam_minus_val * edges.B)))
            res = solve_equation(cb_matrix, cb_vector)
            cb = res[:sid.nsq]
            edges.A = res[sid.nsq:sid.nsq+sid.ne]
            edges.B = res[sid.nsq+sid.ne:]
            J_in = np.sum(edges.inlet * (np.abs(edges.flow) * (edges.A * (1 - lam_plus_zero) + edges.B) - (1 - lam_plus_zero) / sid.Pe * edges.diams ** 2 * (lam_plus_val * edges.A - lam_minus_val * edges.B)))
            cb = cb / J_in * sid.cb_0 * sid.Q_in
            edges.A = edges.A  / J_in * sid.cb_0 * sid.Q_in
            edges.B = edges.B  / J_in * sid.cb_0 * sid.Q_in
            change_pe_fix = lam_plus_zero * 2 * edges.B * np.abs(edges.flow) / sid.Da * (1 - np.exp(-alpha * sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow)))
            change_pe_fix = np.array(np.ma.fix_invalid(change_pe_fix, fill_value = 0))
            change = (1 - lam_plus_zero) * 2  * edges.diams ** 2 / (sid.Pe * sid.Da) * (edges.A * (np.exp(lam_plus_val * edges.lens) - 1) * lam_minus_val + edges.B * (1 - np.exp(-lam_minus_val * edges.lens)) * lam_plus_val) + change_pe_fix #+ change_zero_flow_fix

            change = np.array(np.ma.fix_invalid(change, fill_value = 0))
            #vol_a_dissolved = (spr.diags(vols.vol_a) @ vols.triangles.T) @ (change / edge_vols) * sid.dt
            vol_a_dissolved = vols.triangles.T @ (change / edges.triangles) * sid.dt
            f_alpha = vols.vol_a - vol_a_dissolved # check if overdissolved
            f_alpha_check += (f_alpha < 0) * (vols.vol_a > 0)
            it_alpha += 1
        if it_alpha > sid.it_limit:
            raise ValueError("Iterating for dissolution did not converge")

    edges.alpha = alpha
    data.J_in = np.sum(edges.inlet * (np.abs(edges.flow) * (edges.A * (1 - lam_plus_zero) + edges.B) - (1 - lam_plus_zero) / sid.Pe * edges.diams ** 2 * (lam_plus_val * edges.A - lam_minus_val * edges.B)))
    data.J_out = np.abs(1 * ( inc.incidence.T @ spr.diags(edges.flow) < 0) @ (np.abs(edges.flow) \
                 * edges.outlet)) @ cb
    print(np.max(cb), np.min(cb))

    #print(1 * (vols.triangles @ (vols.vol_a == 0)))
    return cb


def solve_vol_chat(sid: SimInputData, inc: Incidence, graph: Graph, \
    edges: Edges, vols: Volumes, cb_vector, data) -> np.ndarray:
    """ Calculate B concentration with tracking of A volume.
    """

    print(f'problems... {np.where(vols.triangles @ (vols.vol_a == 0) != 0)}')
    alpha_prev = np.zeros(sid.ne)
    it_alpha = 0
    alpha_tr = 1 * (vols.vol_a > 0) # vector scaling the reaction constants in
    alpha = np.array(np.ma.fix_invalid((vols.triangles @ (alpha_tr)) / edges.triangles, fill_value = 0.))
    # triangles according to A availibility
    alpha_tr_prev = np.zeros(sid.ntr)
    #print(np.where(alpha != 1))
    # print(np.where(alpha != 1)[0].shape)
    # print(alpha)
    lam_plus_val = sid.Pe / 2 / edges.diams ** 2 * \
        (np.sqrt(np.abs(edges.flow) ** 2 + 4 * alpha * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) + np.abs(edges.flow))
    lam_plus_val = np.array(np.ma.fix_invalid(lam_plus_val, fill_value = 0))
    lam_minus_val = sid.Pe / 2 / edges.diams ** 2 * \
        (np.sqrt(np.abs(edges.flow) ** 2 + 4 * alpha * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) - np.abs(edges.flow)) #* (edges.diams <= sid.dmax)
    lam_minus_val = np.array(np.ma.fix_invalid(lam_minus_val, fill_value = 0))
    #lam_plus_zero = 1 * (1 * (lam_plus_val > sid.diffusion_exp_limit) + 1 * (edges.alpha == 0) != 0)
    lam_plus_zero = 1 * (lam_plus_val > sid.diffusion_exp_limit)
    print(f'lam plus zero: {np.sum(lam_plus_zero), np.sum(1 - lam_plus_zero)}')
    lam_plus_val = lam_plus_val * (1 - lam_plus_zero)
    lam_minus_val = lam_minus_val * (1 - lam_plus_zero)

    exp_plus = spr.diags(np.exp(lam_plus_val * edges.lens) * (1 - lam_plus_zero) + lam_plus_zero)
    exp_plus2 = spr.diags(np.exp(lam_plus_val * edges.lens) * (1 - lam_plus_zero))
    exp_minus = spr.diags(np.exp(-lam_minus_val * edges.lens))
    exp_minus2 = spr.diags(np.exp(-lam_minus_val * edges.lens) * (1 - lam_plus_zero))
    
    lam_minus_val = lam_minus_val * (1 - lam_plus_zero)

    upstream = 1 * (spr.diags(edges.flow) @ inc.incidence > 0) + 1 * (spr.diags(1 * (edges.flow == 0) * (edges.diams > 0)) @ inc.incidence > 0)
    downstream = 1 * (spr.diags(edges.flow) @ inc.incidence < 0) + 1 * (spr.diags(1 * (edges.flow == 0) * (edges.diams > 0)) @ inc.incidence < 0)
    downstream2 = 1 * (1 * (spr.diags(edges.flow) @ inc.incidence < 0) + 1 * (spr.diags(1 * (edges.flow == 0) * (edges.diams > 0)) @ inc.incidence < 0)).multiply((1 - lam_plus_zero)[:, np.newaxis])
    #cb_pe_fix = 1 * (downstream.T @ (1 - lam_plus_zero) == 0)
    flux_a = 1 * (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_plus2 @ downstream + (spr.diags(lam_plus_val * edges.diams ** 2)) @ upstream - (exp_plus * spr.diags(lam_plus_val * edges.diams ** 2)) @ downstream).multiply((1 - lam_plus_zero)[:, np.newaxis])
    #flux_b = 1 * (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_minus @ downstream + (spr.diags(-lam_minus_val * edges.diams ** 2)) @ upstream + (exp_minus * spr.diags(lam_minus_val * edges.diams ** 2)) @ downstream)
    flux_b = 1 * (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_minus2 @ downstream + (spr.diags(-lam_minus_val * edges.diams ** 2)) @ upstream + (exp_minus2 * spr.diags(lam_minus_val * edges.diams ** 2)) @ downstream).multiply((1 - lam_plus_zero)[:, np.newaxis])
    exp_pe_fix = np.exp(-alpha * sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow))
    exp_pe_fix = np.array(np.ma.fix_invalid(exp_pe_fix, fill_value = 0))
    flux_b += 1 * (sid.Pe * spr.diags(np.abs(edges.flow)) @ spr.diags(lam_plus_zero * exp_pe_fix) @ downstream)
    flux_a_in = flux_a.T.multiply((1 - graph.in_vec)[:, np.newaxis])
    
    flux_b_in = flux_b.T.multiply((1 - graph.in_vec)[:, np.newaxis])

    flow_fix_pe = -sid.Pe * downstream.T @ np.abs(edges.flow)
    flow_fix_pe = flow_fix_pe * (1 - graph.in_vec) + graph.in_vec
    zero_flow_fix = 1 * (edges.flow == 0) * (alpha == 0) # where the flow is zero and alpha is zero, we solve a different equation: d2c/dx2 = 0, with c(0) = c_up and c(l) = c_down

    # what are the equations when flow == 0?
    cb_matrix = spr.vstack([spr.hstack([spr.diags(flow_fix_pe), flux_a_in, flux_b_in]), \
                    spr.hstack([-downstream2, exp_plus + spr.diags((edges.lens -  1) * zero_flow_fix), exp_minus2]), \
                    spr.hstack([-upstream, spr.diags(np.ones(sid.ne) - zero_flow_fix), spr.diags(np.ones(sid.ne))]) \
                    ])
    cb_matrix = spr.diags(1 - inc.merge_vec) @ cb_matrix @ spr.diags(1 - inc.merge_vec) + spr.diags(inc.merge_vec)

    rows_empty = (cb_matrix.getnnz(axis=1) == 0)
    if np.any(rows_empty):
        cb_matrix = cb_matrix + spr.diags(rows_empty.astype(float))
    res = solve_equation(cb_matrix, cb_vector)
    cb = res[:sid.nsq]
    edges.A = res[sid.nsq:sid.nsq+sid.ne]
    edges.B = res[sid.nsq+sid.ne:]

    J_in = np.sum(edges.inlet * (np.abs(edges.flow) * (edges.A * (1 - lam_plus_zero) + edges.B) - (1 - lam_plus_zero) / sid.Pe * edges.diams ** 2 * (lam_plus_val * edges.A - lam_minus_val * edges.B)))
    cb = cb / J_in * sid.cb_0 * sid.Q_in
    edges.A = edges.A  / J_in * sid.cb_0 * sid.Q_in
    edges.B = edges.B  / J_in * sid.cb_0 * sid.Q_in

    edge_vols = vols.triangles @ vols.vol_a
    #change = (1 - lam_plus_zero) / (sid.Da * edges.lens * edges.diams) * (edges.A * (np.exp(lam_plus_val) - 1) * (np.sqrt(np.abs(edges.flow) ** 2 + 4 * alpha * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) - np.abs(edges.flow)) + edges.B * (1 - np.exp(-lam_minus_val)) * (np.sqrt(np.abs(edges.flow) ** 2 + 4 * alpha * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) + np.abs(edges.flow))) + lam_plus_zero * edges.B * np.abs(edges.flow) / (sid.Da * edges.lens * edges.diams) * (1 - np.exp(-alpha * sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow)))
    change_pe_fix = lam_plus_zero * 2 * edges.B * np.abs(edges.flow) / sid.Da * (1 - np.exp(-alpha * sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow)))
    change_pe_fix = np.array(np.ma.fix_invalid(change_pe_fix, fill_value = 0))
    change = (1 - lam_plus_zero) * 2  * edges.diams ** 2 / (sid.Pe * sid.Da) * (edges.A * (np.exp(lam_plus_val * edges.lens) - 1) * lam_minus_val + edges.B * (1 - np.exp(-lam_minus_val * edges.lens)) * lam_plus_val) + change_pe_fix #+ change_zero_flow_fix

    change = np.array(np.ma.fix_invalid(change, fill_value = 0))
    #vol_a_dissolved = (spr.diags(vols.vol_a) @ vols.triangles.T) @ (change / edge_vols) * sid.dt
    vol_a_dissolved = vols.triangles.T @ (change / edges.triangles) * sid.dt
    f_alpha = vols.vol_a - vol_a_dissolved # check if overdissolved
    f_alpha_check = (f_alpha < 0) * (vols.vol_a > 0)
    print(f_alpha[np.where(f_alpha < 0)[0]])
    # iterate using N-R until alpha_b_tr is the same (up to certain threshold)
    # in consecutive iterations; alpha_b for each edge is a function of
    # alpha_b_tr for the triangles neighbouring the edge, so we use matrix N-R
    # df(alpha i-1) @ delta_alpha = f(alpha i-1)
    # alpha i = alpha i-1 + delta_alpha
    while np.linalg.norm(alpha_tr - alpha_tr_prev) > sid.it_alpha_th:
        print(f'alpha diff: {np.linalg.norm(alpha_tr - alpha_tr_prev)}')
        alpha_tr_prev = alpha_tr.copy()
        
        #df_alpha_pe_fix = lam_plus_zero * 2 * edges.diams * edges.lens * edges.B * np.exp(-alpha * sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow)) * sid.dt / (1 + sid.G * edges.diams) / edges.triangles ** 2
        df_alpha_pe_fix = lam_plus_zero * 2 * edges.B * np.exp(-alpha * sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow)) * sid.dt / (1 + sid.G * edges.diams) / edges.triangles
        df_alpha_pe_fix = np.array(np.ma.fix_invalid(df_alpha_pe_fix, fill_value = 0.)) # fix
        #df_alpha = (-2 * (1 - lam_plus_zero) * edges.diams * edges.lens * (edges.A * np.exp(lam_plus_val * edges.lens) + edges.B * np.exp(-lam_minus_val * edges.lens)) * sid.dt / (1 + sid.G * edges.diams) / edges.triangles ** 2 - df_alpha_pe_fix) * (alpha > 0)
        df_alpha = (-2 * (1 - lam_plus_zero) * (edges.A * np.exp(lam_plus_val * edges.lens) + edges.B * np.exp(-lam_minus_val * edges.lens)) * sid.dt / (1 + sid.G * edges.diams) / edges.triangles - df_alpha_pe_fix) * (alpha > 0)
        df_alpha = np.array(np.ma.fix_invalid(df_alpha, fill_value = 0.)) # fix
        # for zero surface
        # we calculate delta_alpha only where overdissolved, that's why we use
        # f_alpha_check
        df_alpha_matrix = spr.diags(1 * f_alpha_check) @ vols.triangles.T @ \
            spr.diags(df_alpha) @ vols.triangles
        # we set rows without overdissolution to identity
        df_alpha_matrix += spr.diags(1 * (df_alpha_matrix.diagonal() == 0))
        delta_alpha = solve_equation(df_alpha_matrix, -f_alpha * f_alpha_check)
        # we clip the reaction rate to [0,1], as N-R sometimes overshoots and
        # we only want to slow down the reaction, not fasten
        alpha_tr = np.clip(alpha_tr + delta_alpha, 0, 1)

        #alpha_tr = np.array(np.ma.fix_invalid(alpha_tr, fill_value = 0.))
        alpha = np.array(np.ma.fix_invalid((vols.triangles @ (alpha_tr)) / edges.triangles, fill_value = 0.))
        #print(delta_alpha)
        #print(alpha_tr)
        #print(vols.vol_a)
        #print(vol_a_dissolved)
        #print(vols.triangles.T @ df_alpha)
        #print(alpha)
        # if np.sum(delta_alpha):
        #     np.savetxt('df.txt', df_alpha_matrix.toarray())
        #     raise ValueError
        # if alpha_b != identity, we recalculate B concentrations (which
        # change when we change alpha_b) and dissolved volumes and iterate
        # until alpha_b converges
        if np.sum(alpha) != np.sum(edges.diams > 0):
            lam_plus_val = sid.Pe / 2 / edges.diams ** 2 * \
                (np.sqrt(np.abs(edges.flow) ** 2 + 4 * alpha * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) + np.abs(edges.flow))
            lam_plus_val = np.array(np.ma.fix_invalid(lam_plus_val, fill_value = 0))
            lam_minus_val = sid.Pe / 2 / edges.diams ** 2 * \
                (np.sqrt(np.abs(edges.flow) ** 2 + 4 * alpha * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) - np.abs(edges.flow)) #* (edges.diams <= sid.dmax)
            lam_minus_val = np.array(np.ma.fix_invalid(lam_minus_val, fill_value = 0))
            #lam_plus_zero = 1 * (1 * (lam_plus_val > sid.diffusion_exp_limit) + 1 * (edges.alpha == 0) != 0)
            lam_plus_zero = 1 * (lam_plus_val > sid.diffusion_exp_limit)
            lam_plus_val = lam_plus_val * (1 - lam_plus_zero)
            lam_minus_val = lam_minus_val * (1 - lam_plus_zero)

            exp_plus = spr.diags(np.exp(lam_plus_val * edges.lens) * (1 - lam_plus_zero) + lam_plus_zero)
            exp_plus2 = spr.diags(np.exp(lam_plus_val * edges.lens) * (1 - lam_plus_zero))
            exp_minus = spr.diags(np.exp(-lam_minus_val * edges.lens))
            exp_minus2 = spr.diags(np.exp(-lam_minus_val * edges.lens) * (1 - lam_plus_zero))
            
            lam_minus_val = lam_minus_val * (1 - lam_plus_zero)

            upstream = 1 * (spr.diags(edges.flow) @ inc.incidence > 0) + 1 * (spr.diags(1 * (edges.flow == 0) * (edges.diams > 0)) @ inc.incidence > 0)
            downstream = 1 * (spr.diags(edges.flow) @ inc.incidence < 0) + 1 * (spr.diags(1 * (edges.flow == 0) * (edges.diams > 0)) @ inc.incidence < 0)
            downstream2 = 1 * (1 * (spr.diags(edges.flow) @ inc.incidence < 0) + 1 * (spr.diags(1 * (edges.flow == 0) * (edges.diams > 0)) @ inc.incidence < 0)).multiply((1 - lam_plus_zero)[:, np.newaxis])
            #cb_pe_fix = 1 * (downstream.T @ (1 - lam_plus_zero) == 0)
            flux_a = 1 * (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_plus2 @ downstream + (spr.diags(lam_plus_val * edges.diams ** 2)) @ upstream - (exp_plus * spr.diags(lam_plus_val * edges.diams ** 2)) @ downstream).multiply((1 - lam_plus_zero)[:, np.newaxis])
            #flux_b = 1 * (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_minus @ downstream + (spr.diags(-lam_minus_val * edges.diams ** 2)) @ upstream + (exp_minus * spr.diags(lam_minus_val * edges.diams ** 2)) @ downstream)
            flux_b = 1 * (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_minus2 @ downstream + (spr.diags(-lam_minus_val * edges.diams ** 2)) @ upstream + (exp_minus2 * spr.diags(lam_minus_val * edges.diams ** 2)) @ downstream).multiply((1 - lam_plus_zero)[:, np.newaxis])
            exp_pe_fix = np.exp(-alpha * sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow))
            exp_pe_fix = np.array(np.ma.fix_invalid(exp_pe_fix, fill_value = 0))
            flux_b += 1 * (sid.Pe * spr.diags(np.abs(edges.flow)) @ spr.diags(lam_plus_zero * exp_pe_fix) @ downstream)
            flux_a_in = flux_a.T.multiply((1 - graph.in_vec)[:, np.newaxis])
            
            flux_b_in = flux_b.T.multiply((1 - graph.in_vec)[:, np.newaxis])

            flow_fix_pe = -sid.Pe * downstream.T @ np.abs(edges.flow)
            flow_fix_pe = flow_fix_pe * (1 - graph.in_vec) + graph.in_vec
            zero_flow_fix = 1 * (edges.flow == 0) * (alpha == 0) # where the flow is zero and alpha is zero, we solve a different equation: d2c/dx2 = 0, with c(0) = c_up and c(l) = c_down
            print('Zero flow: ', np.sum(zero_flow_fix))
            # what are the equations when flow == 0?
            cb_matrix = spr.vstack([spr.hstack([spr.diags(flow_fix_pe), flux_a_in, flux_b_in]), \
                            spr.hstack([-downstream2, exp_plus + spr.diags((edges.lens -  1) * zero_flow_fix), exp_minus2]), \
                            spr.hstack([-upstream, spr.diags(np.ones(sid.ne) - zero_flow_fix), spr.diags(np.ones(sid.ne))]) \
                            ])
            cb_matrix = spr.diags(1 - inc.merge_vec) @ cb_matrix @ spr.diags(1 - inc.merge_vec) + spr.diags(inc.merge_vec)
            rows_empty = (cb_matrix.getnnz(axis=1) == 0)
            if np.any(rows_empty):
                cb_matrix = cb_matrix + spr.diags(rows_empty.astype(float))

            #cb_matrix += spr.diags(1 * (np.array(np.sum(np.abs(cb_matrix), axis = 1))[:, 0] == 0))
            #J_in = np.sum(edges.inlet * (np.abs(edges.flow) * (edges.A * (1 - lam_plus_zero) + edges.B) - (1 - lam_plus_zero) / sid.Pe * edges.diams ** 2 * (lam_plus_val * edges.A - lam_minus_val * edges.B)))
            res = solve_equation(cb_matrix, cb_vector)
            cb = res[:sid.nsq]
            edges.A = res[sid.nsq:sid.nsq+sid.ne]
            edges.B = res[sid.nsq+sid.ne:]
            J_in = np.sum(edges.inlet * (np.abs(edges.flow) * (edges.A * (1 - lam_plus_zero) + edges.B) - (1 - lam_plus_zero) / sid.Pe * edges.diams ** 2 * (lam_plus_val * edges.A - lam_minus_val * edges.B)))
            cb = cb / J_in * sid.cb_0 * sid.Q_in
            edges.A = edges.A  / J_in * sid.cb_0 * sid.Q_in
            edges.B = edges.B  / J_in * sid.cb_0 * sid.Q_in
            change_pe_fix = lam_plus_zero * 2 * edges.B * np.abs(edges.flow) / sid.Da * (1 - np.exp(-alpha * sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow)))
            change_pe_fix = np.array(np.ma.fix_invalid(change_pe_fix, fill_value = 0))
            change = (1 - lam_plus_zero) * 2  * edges.diams ** 2 / (sid.Pe * sid.Da) * (edges.A * (np.exp(lam_plus_val * edges.lens) - 1) * lam_minus_val + edges.B * (1 - np.exp(-lam_minus_val * edges.lens)) * lam_plus_val) + change_pe_fix #+ change_zero_flow_fix

            change = np.array(np.ma.fix_invalid(change, fill_value = 0))
            #vol_a_dissolved = (spr.diags(vols.vol_a) @ vols.triangles.T) @ (change / edge_vols) * sid.dt
            vol_a_dissolved = vols.triangles.T @ (change / edges.triangles) * sid.dt
            f_alpha = vols.vol_a - vol_a_dissolved # check if overdissolved
            f_alpha_check = f_alpha_check | ((f_alpha < 0) & (vols.vol_a > 0))
            it_alpha += 1
        if it_alpha > sid.it_limit:
            raise ValueError("Iterating for dissolution did not converge")

    edges.alpha = alpha
    data.J_in = np.sum(edges.inlet * (np.abs(edges.flow) * (edges.A * (1 - lam_plus_zero) + edges.B) - (1 - lam_plus_zero) / sid.Pe * edges.diams ** 2 * (lam_plus_val * edges.A - lam_minus_val * edges.B)))
    data.J_out = np.abs(1 * ( inc.incidence.T @ spr.diags(edges.flow) < 0) @ (np.abs(edges.flow) \
                 * edges.outlet)) @ cb
    print(np.max(cb), np.min(cb))
    print(f'alpha diff: {np.linalg.norm(alpha_tr - alpha_tr_prev)}')
    print(f_alpha[np.where(f_alpha < 0)[0]])
    #print(1 * (vols.triangles @ (vols.vol_a == 0)))
    return cb

def solve_vol_nr_chat(sid: SimInputData, inc: Incidence, graph: Graph,
                 edges: Edges, vols: Volumes, cb_vector, data) -> np.ndarray:
    """Calculate B concentration with tracking of A volume and prevent overdissolution."""

    # --- Helper masks & safe primitives ---
    def to_float_mask(x):
        return x.astype(float)
    edge_vol = vols.triangles @ vols.vol_a
    triangles_w = vols.triangles @ spr.diags(vols.vol_a)
    # Initialize alpha_tr from available solid; compute edge alphas as triangle-average
    alpha_tr = (vols.vol_a > 0).astype(float)          # per-triangle
    alpha = np.array(np.ma.fix_invalid((triangles_w @ alpha_tr) / edge_vol,
                                       fill_value=0.0))  # per-edge

    # Precompute safe inverse |flow| (0 where flow==0)
    inv_abs_flow = np.divide(
        1.0, np.abs(edges.flow),
        out=np.zeros_like(edges.flow, dtype=float),
        where=np.abs(edges.flow) > 0
    )

    # --- Build transport coefficients (first solve) ---
    lam_root = np.sqrt(np.abs(edges.flow) ** 2 +
                       4.0 * alpha * sid.Da / (1.0 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3)

    lam_plus_val = sid.Pe / (2.0 * edges.diams ** 2) * (lam_root + np.abs(edges.flow))
    lam_minus_val = sid.Pe / (2.0 * edges.diams ** 2) * (lam_root - np.abs(edges.flow))

    # Mask diffusion-exp-limit and cast mask to float early
    lam_plus_zero = (lam_plus_val > sid.diffusion_exp_limit).astype(float)
    lam_plus_val = lam_plus_val * (1.0 - lam_plus_zero)
    lam_minus_val = lam_minus_val * (1.0 - lam_plus_zero)

    exp_plus_diag  = np.exp(lam_plus_val * edges.lens) * (1.0 - lam_plus_zero) + lam_plus_zero
    exp_plus2_diag = np.exp(lam_plus_val * edges.lens) * (1.0 - lam_plus_zero)
    exp_minus_diag  = np.exp(-lam_minus_val * edges.lens)              # (not used directly in blocks)
    exp_minus2_diag = np.exp(-lam_minus_val * edges.lens) * (1.0 - lam_plus_zero)

    exp_plus  = spr.diags(exp_plus_diag)
    exp_plus2 = spr.diags(exp_plus2_diag)
    exp_minus2 = spr.diags(exp_minus2_diag)

    # Upstream / downstream incidence with robust OR on zero-flow carriers
    F = spr.diags(edges.flow)
    zero_carrier = spr.diags(((edges.flow == 0) & (edges.diams > 0)).astype(float))

    A_pos = ((F @ inc.incidence) > 0)
    Z_pos = ((zero_carrier @ inc.incidence) > 0)
    A_neg = ((F @ inc.incidence) < 0)
    Z_neg = ((zero_carrier @ inc.incidence) < 0)

    # elementwise logical OR for sparse matrices
    upstream_bool   = A_pos.maximum(Z_pos)
    downstream_bool = A_neg.maximum(Z_neg)

    upstream   = upstream_bool.astype(float)
    downstream = downstream_bool.astype(float)
    downstream2 = downstream.multiply((1.0 - lam_plus_zero)[:, np.newaxis])

    # Flux blocks
    flux_a = (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_plus2 @ downstream
              + spr.diags(lam_plus_val * edges.diams ** 2) @ upstream
              - exp_plus @ spr.diags(lam_plus_val * edges.diams ** 2) @ downstream).multiply(
                  (1.0 - lam_plus_zero)[:, np.newaxis])

    flux_b = (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_minus2 @ downstream
              - spr.diags(lam_minus_val * edges.diams ** 2) @ upstream
              + exp_minus2 @ spr.diags(lam_minus_val * edges.diams ** 2) @ downstream).multiply(
                  (1.0 - lam_plus_zero)[:, np.newaxis])

    # Péclet fix for lam_plus_zero edges (safe division)
    exp_pe_fix = np.exp(-alpha * sid.Da / (1.0 + sid.G * edges.diams) *
                        edges.diams * edges.lens * inv_abs_flow)
    flux_b += sid.Pe * spr.diags(np.abs(edges.flow)) @ spr.diags(lam_plus_zero * exp_pe_fix) @ downstream

    # Inlet/outlet coupling
    flux_a_in = flux_a.T.multiply((1.0 - graph.in_vec)[:, np.newaxis])
    flux_b_in = flux_b.T.multiply((1.0 - graph.in_vec)[:, np.newaxis])

    flow_fix_pe = -sid.Pe * downstream.T @ np.abs(edges.flow)
    flow_fix_pe = flow_fix_pe * (1.0 - graph.in_vec) + graph.in_vec

    # Zero-flow edges that also have no reaction (alpha==0): enforce linear profile via identity
    zero_flow_fix = ((edges.flow == 0) & (alpha == 0)).astype(float)
    # Use identity for those rows instead of the advection-exp block
    exp_plus_eff_diag = exp_plus_diag.copy()
    exp_plus_eff_diag[zero_flow_fix.astype(bool)] = 1.0
    exp_plus_eff = spr.diags(exp_plus_eff_diag)

    # Assemble linear system
    cb_matrix = spr.vstack([
        spr.hstack([spr.diags(flow_fix_pe),           flux_a_in,                                 flux_b_in]),
        spr.hstack([-downstream2,                     exp_plus_eff,                              exp_minus2]),
        spr.hstack([-upstream,                        spr.diags(np.ones(sid.ne) - zero_flow_fix), spr.diags(np.ones(sid.ne))])
    ])

    # Merge handling
    merge_diag = spr.diags((1.0 - inc.merge_vec))
    cb_matrix = merge_diag @ cb_matrix @ merge_diag + spr.diags(inc.merge_vec.astype(float))

    # Ensure no empty rows without densifying
    rows_empty = (cb_matrix.getnnz(axis=1) == 0)
    if np.any(rows_empty):
        cb_matrix = cb_matrix + spr.diags(rows_empty.astype(float))

    # Solve
    res = solve_equation(cb_matrix, cb_vector)
    cb = res[:sid.nsq]
    edges.A = res[sid.nsq:sid.nsq + sid.ne]
    edges.B = res[sid.nsq + sid.ne:]

    # Normalize by inlet flux
    J_in = np.sum(
        edges.inlet *
        (np.abs(edges.flow) * (edges.A * (1.0 - lam_plus_zero) + edges.B)
         - (1.0 - lam_plus_zero) / sid.Pe * edges.diams ** 2 * (lam_plus_val * edges.A - lam_minus_val * edges.B))
    )
    if (not np.isfinite(J_in)) or (J_in <= 0):
        raise ValueError("Non-positive or invalid inlet flux during normalization")

    scale = sid.cb_0 * sid.Q_in / J_in
    cb *= scale
    edges.A *= scale
    edges.B *= scale

    # Dissolution update
    change_pe_fix = lam_plus_zero * 2.0 * edges.B * np.abs(edges.flow) / sid.Da * (
        1.0 - np.exp(-alpha * sid.Da / (1.0 + sid.G * edges.diams) * edges.diams * edges.lens * inv_abs_flow)
    )
    change_pe_fix = np.array(np.ma.fix_invalid(change_pe_fix, fill_value=0.0))

    change = ((1.0 - lam_plus_zero) * 2.0 * edges.diams ** 2 / (sid.Pe * sid.Da) *
              (edges.A * (np.exp(lam_plus_val * edges.lens) - 1.0) * lam_minus_val
               + edges.B * (1.0 - np.exp(-lam_minus_val * edges.lens)) * lam_plus_val)
              + change_pe_fix)
    change = np.array(np.ma.fix_invalid(change, fill_value=0.0))

    vol_a_dissolved = triangles_w.T @ (change / edge_vol) * sid.dt
    f_alpha = vols.vol_a - vol_a_dissolved
    f_alpha_check = (f_alpha < 0) & (vols.vol_a > 0)

    # --- Newton–Raphson loop to rescale reaction where overdissolved ---
    alpha_tr_prev = np.zeros_like(alpha_tr)
    it_alpha = 0
    alpha_prev_for_solve = alpha.copy()  # for re-solve trigger

    while np.linalg.norm(alpha_tr - alpha_tr_prev) > sid.it_alpha_th:
        it_alpha += 1
        if it_alpha > sid.it_limit:
            raise ValueError("Iterating for dissolution did not converge")

        alpha_tr_prev = alpha_tr.copy()

        # Derivatives wrt alpha (safe exp with flow==0 handling)
        exp_pe_fix_for_df = np.exp(-alpha * sid.Da / (1.0 + sid.G * edges.diams) *
                                   edges.diams * edges.lens * inv_abs_flow)

        df_alpha_pe_fix = (lam_plus_zero * 2.0 * edges.B * exp_pe_fix_for_df *
                           sid.dt / (1.0 + sid.G * edges.diams) / edges.triangles)
        df_alpha_pe_fix = np.array(np.ma.fix_invalid(df_alpha_pe_fix, fill_value=0.0))

        df_alpha = (-(1.0 - lam_plus_zero) * 2.0 *
                    (edges.A * np.exp(lam_plus_val * edges.lens) +
                     edges.B * np.exp(-lam_minus_val * edges.lens)) *
                    sid.dt / (1.0 + sid.G * edges.diams) / edges.triangles
                    - df_alpha_pe_fix) * (alpha > 0).astype(float)
        df_alpha = np.array(np.ma.fix_invalid(df_alpha, fill_value=0.0))

        # Build Jacobian over triangles; select only overdissolved rows
        sel = spr.diags(f_alpha_check.astype(float))
        df_alpha_matrix = sel @ vols.triangles.T @ spr.diags(df_alpha) @ vols.triangles

        # Ensure identity on rows with no selection
        diag_zero = (df_alpha_matrix.diagonal() == 0)
        if np.any(diag_zero):
            df_alpha_matrix = df_alpha_matrix + spr.diags(diag_zero.astype(float))

        # Newton step on triangles
        delta_alpha = solve_equation(df_alpha_matrix, -f_alpha * f_alpha_check.astype(float))
        alpha_tr = np.clip(alpha_tr + delta_alpha, 0.0, 1.0)

        # Update edge alphas from triangles (average over incident tris)
        alpha = np.array(np.ma.fix_invalid((triangles_w @ alpha_tr) / edge_vol,
                                        fill_value=0.0))  # per-edge

        # Trigger full transport re-solve if alphas changed anywhere
        needs_resolve = np.any(alpha != alpha_prev_for_solve)
        alpha_prev_for_solve = alpha.copy()

        if needs_resolve:
            # Rebuild transport coefficients with updated alpha
            lam_root = np.sqrt(np.abs(edges.flow) ** 2 +
                               4.0 * alpha * sid.Da / (1.0 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3)
            lam_plus_val = sid.Pe / (2.0 * edges.diams ** 2) * (lam_root + np.abs(edges.flow))
            lam_minus_val = sid.Pe / (2.0 * edges.diams ** 2) * (lam_root - np.abs(edges.flow))

            lam_plus_zero = (lam_plus_val > sid.diffusion_exp_limit).astype(float)
            lam_plus_val = lam_plus_val * (1.0 - lam_plus_zero)
            lam_minus_val = lam_minus_val * (1.0 - lam_plus_zero)

            exp_plus_diag  = np.exp(lam_plus_val * edges.lens) * (1.0 - lam_plus_zero) + lam_plus_zero
            exp_plus2_diag = np.exp(lam_plus_val * edges.lens) * (1.0 - lam_plus_zero)
            exp_minus2_diag = np.exp(-lam_minus_val * edges.lens) * (1.0 - lam_plus_zero)

            exp_plus  = spr.diags(exp_plus_diag)
            exp_plus2 = spr.diags(exp_plus2_diag)
            exp_minus2 = spr.diags(exp_minus2_diag)

            downstream2 = downstream.multiply((1.0 - lam_plus_zero)[:, np.newaxis])

            flux_a = (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_plus2 @ downstream
                      + spr.diags(lam_plus_val * edges.diams ** 2) @ upstream
                      - exp_plus @ spr.diags(lam_plus_val * edges.diams ** 2) @ downstream).multiply(
                          (1.0 - lam_plus_zero)[:, np.newaxis])

            flux_b = (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_minus2 @ downstream
                      - spr.diags(lam_minus_val * edges.diams ** 2) @ upstream
                      + exp_minus2 @ spr.diags(lam_minus_val * edges.diams ** 2) @ downstream).multiply(
                          (1.0 - lam_plus_zero)[:, np.newaxis])

            exp_pe_fix = np.exp(-alpha * sid.Da / (1.0 + sid.G * edges.diams) *
                                edges.diams * edges.lens * inv_abs_flow)
            flux_b += sid.Pe * spr.diags(np.abs(edges.flow)) @ spr.diags(lam_plus_zero * exp_pe_fix) @ downstream

            flux_a_in = flux_a.T.multiply((1.0 - graph.in_vec)[:, np.newaxis])
            flux_b_in = flux_b.T.multiply((1.0 - graph.in_vec)[:, np.newaxis])

            flow_fix_pe = -sid.Pe * downstream.T @ np.abs(edges.flow)
            flow_fix_pe = flow_fix_pe * (1.0 - graph.in_vec) + graph.in_vec

            zero_flow_fix = ((edges.flow == 0) & (alpha == 0)).astype(float)
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

            J_in = np.sum(
                edges.inlet *
                (np.abs(edges.flow) * (edges.A * (1.0 - lam_plus_zero) + edges.B)
                 - (1.0 - lam_plus_zero) / sid.Pe * edges.diams ** 2 * (lam_plus_val * edges.A - lam_minus_val * edges.B))
            )
            if (not np.isfinite(J_in)) or (J_in <= 0):
                raise ValueError("Non-positive or invalid inlet flux during normalization")
            scale = sid.cb_0 * sid.Q_in / J_in
            cb *= scale
            edges.A *= scale
            edges.B *= scale

            change_pe_fix = lam_plus_zero * 2.0 * edges.B * np.abs(edges.flow) / sid.Da * (
                1.0 - np.exp(-alpha * sid.Da / (1.0 + sid.G * edges.diams) * edges.diams * edges.lens * inv_abs_flow)
            )
            change_pe_fix = np.array(np.ma.fix_invalid(change_pe_fix, fill_value=0.0))

            change = ((1.0 - lam_plus_zero) * 2.0 * edges.diams ** 2 / (sid.Pe * sid.Da) *
                      (edges.A * (np.exp(lam_plus_val * edges.lens) - 1.0) * lam_minus_val
                       + edges.B * (1.0 - np.exp(-lam_minus_val * edges.lens)) * lam_plus_val)
                      + change_pe_fix)
            change = np.array(np.ma.fix_invalid(change, fill_value=0.0))

            vol_a_dissolved = triangles_w.T @ (change / edge_vol) * sid.dt
            f_alpha = vols.vol_a - vol_a_dissolved
            print(f_alpha[np.where(f_alpha < 0)[0]])
            # Choose behavior: current-iteration check only (recommended)
            f_alpha_check = (f_alpha < 0) & (vols.vol_a > 0)

    # Final assignments
    edges.alpha = alpha
    print(np.min(cb), np.max(cb))
    print(np.sum(vols.vol_a == 0))
    print(f_alpha[np.where(f_alpha < 0)[0]])
    # Sanity checks
    if np.any(cb < -1e-12):
        raise ValueError("Negative concentration detected")
    if np.any((vols.vol_a - vol_a_dissolved) < -1e-10):
        raise ValueError("Overdissolution remained after NR rescaling")

    # Fluxes
    data.J_in = np.sum(
        edges.inlet *
        (np.abs(edges.flow) * (edges.A * (1.0 - lam_plus_zero) + edges.B)
         - (1.0 - lam_plus_zero) / sid.Pe * edges.diams ** 2 * (lam_plus_val * edges.A - lam_minus_val * edges.B))
    )

    out_sel = ((inc.incidence.T @ spr.diags(edges.flow)) < 0).astype(float)
    data.J_out = np.abs(out_sel @ (np.abs(edges.flow) * edges.outlet)) @ cb

    return cb

def solve_vol_scaling(sid: SimInputData, inc: Incidence, graph: Graph,
                 edges: Edges, vols: Volumes, cb_vector, data) -> np.ndarray:
    """Calculate B concentration with tracking of A volume and prevent overdissolution."""

    # --- Helper masks & safe primitives ---
    def to_float_mask(x):
        return x.astype(float)
    edge_vol = vols.triangles @ vols.vol_a
    triangles_w = vols.triangles @ spr.diags(vols.vol_a)
    # Initialize alpha_tr from available solid; compute edge alphas as triangle-average
    alpha_tr = (vols.vol_a > 0).astype(float)          # per-triangle
    alpha_tr_prev = np.zeros_like(alpha_tr)
    it_alpha = 0

    f_alpha_check = np.ones_like(vols.vol_a)
    f_alpha_check_prev = np.zeros_like(vols.vol_a)
    overdissolved = np.zeros_like(vols.vol_a)
    while np.sum(1 * f_alpha_check) != np.sum(1 * f_alpha_check_prev):
        it_alpha += 1
        if it_alpha > sid.it_limit:
            raise ValueError("Iterating for dissolution did not converge")

        alpha_tr_prev = alpha_tr.copy()
        alpha = np.array(np.ma.fix_invalid((triangles_w @ alpha_tr) / edge_vol,
                                        fill_value=0.0))  # per-edge

        # Precompute safe inverse |flow| (0 where flow==0)
        inv_abs_flow = np.divide(
            1.0, np.abs(edges.flow),
            out=np.zeros_like(edges.flow, dtype=float),
            where=np.abs(edges.flow) > 0
        )

        # --- Build transport coefficients (first solve) ---
        lam_root = np.sqrt(np.abs(edges.flow) ** 2 +
                        4.0 * alpha * sid.Da / (1.0 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3)

        lam_plus_val = sid.Pe / (2.0 * edges.diams ** 2) * (lam_root + np.abs(edges.flow))
        lam_minus_val = sid.Pe / (2.0 * edges.diams ** 2) * (lam_root - np.abs(edges.flow))

        # Mask diffusion-exp-limit and cast mask to float early
        lam_plus_zero = (lam_plus_val > sid.diffusion_exp_limit).astype(float)
        lam_plus_val = lam_plus_val * (1.0 - lam_plus_zero)
        lam_minus_val = lam_minus_val * (1.0 - lam_plus_zero)

        exp_plus_diag  = np.exp(lam_plus_val * edges.lens) * (1.0 - lam_plus_zero) + lam_plus_zero
        exp_plus2_diag = np.exp(lam_plus_val * edges.lens) * (1.0 - lam_plus_zero)
        exp_minus_diag  = np.exp(-lam_minus_val * edges.lens)              # (not used directly in blocks)
        exp_minus2_diag = np.exp(-lam_minus_val * edges.lens) * (1.0 - lam_plus_zero)

        exp_plus  = spr.diags(exp_plus_diag)
        exp_plus2 = spr.diags(exp_plus2_diag)
        exp_minus2 = spr.diags(exp_minus2_diag)

        # Upstream / downstream incidence with robust OR on zero-flow carriers
        F = spr.diags(edges.flow)
        zero_carrier = spr.diags(((edges.flow == 0) & (edges.diams > 0)).astype(float))

        A_pos = ((F @ inc.incidence) > 0)
        Z_pos = ((zero_carrier @ inc.incidence) > 0)
        A_neg = ((F @ inc.incidence) < 0)
        Z_neg = ((zero_carrier @ inc.incidence) < 0)

        # elementwise logical OR for sparse matrices
        upstream_bool   = A_pos.maximum(Z_pos)
        downstream_bool = A_neg.maximum(Z_neg)

        upstream   = upstream_bool.astype(float)
        downstream = downstream_bool.astype(float)
        downstream2 = downstream.multiply((1.0 - lam_plus_zero)[:, np.newaxis])

        # Flux blocks
        flux_a = (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_plus2 @ downstream
                + spr.diags(lam_plus_val * edges.diams ** 2) @ upstream
                - exp_plus @ spr.diags(lam_plus_val * edges.diams ** 2) @ downstream).multiply(
                    (1.0 - lam_plus_zero)[:, np.newaxis])

        flux_b = (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_minus2 @ downstream
                - spr.diags(lam_minus_val * edges.diams ** 2) @ upstream
                + exp_minus2 @ spr.diags(lam_minus_val * edges.diams ** 2) @ downstream).multiply(
                    (1.0 - lam_plus_zero)[:, np.newaxis])

        # Péclet fix for lam_plus_zero edges (safe division)
        exp_pe_fix = np.exp(-alpha * sid.Da / (1.0 + sid.G * edges.diams) *
                            edges.diams * edges.lens * inv_abs_flow)
        flux_b += sid.Pe * spr.diags(np.abs(edges.flow)) @ spr.diags(lam_plus_zero * exp_pe_fix) @ downstream

        # Inlet/outlet coupling
        flux_a_in = flux_a.T.multiply((1.0 - graph.in_vec)[:, np.newaxis])
        flux_b_in = flux_b.T.multiply((1.0 - graph.in_vec)[:, np.newaxis])

        flow_fix_pe = -sid.Pe * downstream.T @ np.abs(edges.flow)
        flow_fix_pe = flow_fix_pe * (1.0 - graph.in_vec) + graph.in_vec

        # Zero-flow edges that also have no reaction (alpha==0): enforce linear profile via identity
        zero_flow_fix = ((edges.flow == 0) & (alpha == 0)).astype(float)
        # Use identity for those rows instead of the advection-exp block
        exp_plus_eff_diag = exp_plus_diag.copy()
        exp_plus_eff_diag[zero_flow_fix.astype(bool)] = 1.0
        exp_plus_eff = spr.diags(exp_plus_eff_diag)

        # Assemble linear system
        cb_matrix = spr.vstack([
            spr.hstack([spr.diags(flow_fix_pe),           flux_a_in,                                 flux_b_in]),
            spr.hstack([-downstream2,                     exp_plus_eff,                              exp_minus2]),
            spr.hstack([-upstream,                        spr.diags(np.ones(sid.ne) - zero_flow_fix), spr.diags(np.ones(sid.ne))])
        ])

        # Merge handling
        merge_diag = spr.diags((1.0 - inc.merge_vec))
        cb_matrix = merge_diag @ cb_matrix @ merge_diag + spr.diags(inc.merge_vec.astype(float))

        # Ensure no empty rows without densifying
        rows_empty = (cb_matrix.getnnz(axis=1) == 0)
        if np.any(rows_empty):
            cb_matrix = cb_matrix + spr.diags(rows_empty.astype(float))

        # Solve
        res = solve_equation(cb_matrix, cb_vector)
        cb = res[:sid.nsq]
        edges.A = res[sid.nsq:sid.nsq + sid.ne]
        edges.B = res[sid.nsq + sid.ne:]

        # Normalize by inlet flux
        J_in = np.sum(
            edges.inlet *
            (np.abs(edges.flow) * (edges.A * (1.0 - lam_plus_zero) + edges.B)
            - (1.0 - lam_plus_zero) / sid.Pe * edges.diams ** 2 * (lam_plus_val * edges.A - lam_minus_val * edges.B))
        )
        if (not np.isfinite(J_in)) or (J_in <= 0):
            raise ValueError("Non-positive or invalid inlet flux during normalization")

        scale = sid.cb_0 * sid.Q_in / J_in
        cb *= scale
        edges.A *= scale
        edges.B *= scale

        # Dissolution update
        change_pe_fix = lam_plus_zero * 2.0 * edges.B * np.abs(edges.flow) / sid.Da * (
            1.0 - np.exp(-alpha * sid.Da / (1.0 + sid.G * edges.diams) * edges.diams * edges.lens * inv_abs_flow)
        )
        change_pe_fix = np.array(np.ma.fix_invalid(change_pe_fix, fill_value=0.0))

        change = ((1.0 - lam_plus_zero) * 2.0 * edges.diams ** 2 / (sid.Pe * sid.Da) *
                (edges.A * (np.exp(lam_plus_val * edges.lens) - 1.0) * lam_minus_val
                + edges.B * (1.0 - np.exp(-lam_minus_val * edges.lens)) * lam_plus_val)
                + change_pe_fix)
        change = np.array(np.ma.fix_invalid(change, fill_value=0.0))

        vol_a_dissolved = triangles_w.T @ (change / edge_vol) * sid.dt
        f_alpha = vols.vol_a - vol_a_dissolved
        f_alpha_check_prev = f_alpha_check.copy()
        f_alpha_check = f_alpha_check + (f_alpha < 0) * (f_alpha_check == 0)
        overdissolved = overdissolved + (f_alpha < 0) * (overdissolved == 0)
        #print(f_alpha[np.where(f_alpha < 0)[0]])
        alpha_tr = np.clip(vols.vol_a / vol_a_dissolved, 0, 1)
    vols.vol_a = vols.vol_a * (1 - overdissolved)
    # Final assignments
    edges.alpha = alpha
    print(np.min(cb), np.max(cb))
    print(np.sum(vols.vol_a == 0))
    print(f_alpha[np.where(f_alpha < 0)[0]])
    # Sanity checks
    if np.any(cb < -1e-12):
        raise ValueError("Negative concentration detected")
    # if np.any((vols.vol_a - vol_a_dissolved) < -1e-10):
    #     raise ValueError("Overdissolution remained after NR rescaling")

    # Fluxes
    data.J_in = np.sum(
        edges.inlet *
        (np.abs(edges.flow) * (edges.A * (1.0 - lam_plus_zero) + edges.B)
         - (1.0 - lam_plus_zero) / sid.Pe * edges.diams ** 2 * (lam_plus_val * edges.A - lam_minus_val * edges.B))
    )

    out_sel = ((inc.incidence.T @ spr.diags(edges.flow)) < 0).astype(float)
    data.J_out = np.abs(out_sel @ (np.abs(edges.flow) * edges.outlet)) @ cb

    return cb

def solve_vol_scaling_chat(sid: SimInputData, inc: Incidence, graph: Graph,
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
    def solve_transport_and_change(alpha_edge):
        # safe inverse |flow|
        M = spr.diags(edges.flow) @ inc.incidence  # edges x nodes

        # if incidence uses -1 at tail and +1 at head:
        out_of_node = (M < 0).astype(float)   # flow leaves node
        into_node   = (M > 0).astype(float)   # flow enters node

        Qout = out_of_node.T @ np.abs(edges.flow)
        Qin  = into_node.T   @ np.abs(edges.flow)

        tol = 1e-14
        dirichlet = (graph.in_vec > 0) & (Qin > Qout + tol)

        in_vec = graph.in_vec #dirichlet.astype(float)
        cb_vector = np.concatenate([sid.cb_0 * in_vec, np.zeros(2 * sid.ne)])


        
        inv_abs_flow = np.divide(1.0, np.abs(edges.flow),
                                 out=np.zeros_like(edges.flow, dtype=float),
                                 where=np.abs(edges.flow) > 0)

        lam_root = np.sqrt(np.abs(edges.flow) ** 2 +
                           4.0 * alpha_edge * sid.Da / (1.0 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3)
        lam_plus_val  = sid.Pe / (2.0 * edges.diams ** 2) * (lam_root + np.abs(edges.flow))
        lam_plus_val = np.array(np.ma.fix_invalid(lam_plus_val, fill_value = 0))
        lam_minus_val = sid.Pe / (2.0 * edges.diams ** 2) * (lam_root - np.abs(edges.flow))
        lam_minus_val = np.array(np.ma.fix_invalid(lam_minus_val, fill_value = 0))

        lam_plus_zero = (lam_plus_val > sid.diffusion_exp_limit).astype(float)
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
        #Z_pos = ((zero_carrier @ inc.incidence) > 0)
        A_neg = ((F @ inc.incidence) < 0)
        #Z_neg = ((zero_carrier @ inc.incidence) < 0)
        #upstream   = A_pos.maximum(Z_pos).astype(float)
        #downstream = A_neg.maximum(Z_neg).astype(float)
        Z_up = ((zero_carrier @ inc.incidence) < 0)   # tail (-1)
        Z_dn = ((zero_carrier @ inc.incidence) > 0)   # head (+1)

        upstream   = A_pos.maximum(Z_up).astype(float)
        downstream = A_neg.maximum(Z_dn).astype(float)

        downstream2 = downstream.multiply((1.0 - lam_plus_zero)[:, np.newaxis])

        # flux blocks
        flux_a = (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_plus2 @ downstream
                  + spr.diags(lam_plus_val * edges.diams ** 2) @ upstream
                  - exp_plus @ spr.diags(lam_plus_val * edges.diams ** 2) @ downstream).multiply(
                      (1.0 - lam_plus_zero)[:, np.newaxis])
        flux_b = (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_minus2 @ downstream
                  - spr.diags(lam_minus_val * edges.diams ** 2) @ upstream
                  + exp_minus2 @ spr.diags(lam_minus_val * edges.diams ** 2) @ downstream).multiply(
                      (1.0 - lam_plus_zero)[:, np.newaxis])

        # Pe fix
        exp_pe_fix = np.exp(-alpha_edge * sid.Da / (1.0 + sid.G * edges.diams) *
                            edges.diams * edges.lens * inv_abs_flow)
        flux_b += sid.Pe * spr.diags(np.abs(edges.flow)) @ spr.diags(lam_plus_zero * exp_pe_fix) @ downstream

        flux_a_in = flux_a.T.multiply((1.0 - in_vec)[:, np.newaxis])
        flux_b_in = flux_b.T.multiply((1.0 - in_vec)[:, np.newaxis])

        flow_fix_pe = -sid.Pe * downstream.T @ np.abs(edges.flow)
        flow_fix_pe = flow_fix_pe * (1.0 - in_vec) + in_vec

        #zero_flow_fix = ((edges.flow == 0) & (alpha_edge == 0)).astype(float)
        zero_flow_fix = ((edges.flow == 0) & (alpha_edge == 0) & (edges.diams > 0)).astype(float)
        flux_b += spr.diags(-edges.diams**2 * zero_flow_fix) @ upstream \
            + spr.diags( edges.diams**2 * zero_flow_fix) @ downstream

        exp_plus_eff_diag = exp_plus_diag.copy()
        exp_plus_eff_diag[zero_flow_fix.astype(bool)] = 1.0
        exp_plus_eff = spr.diags(exp_plus_eff_diag)

        A_up = np.ones(sid.ne)
        B_up = np.ones(sid.ne) - zero_flow_fix         # linear edges: B coeff 0

        A_dn = exp_plus_eff_diag.copy()
        B_dn = exp_minus2_diag.copy()

        # overwrite for linear edges: cb_down = A + B*L
        A_dn[zero_flow_fix.astype(bool)] = 1.0
        B_dn[zero_flow_fix.astype(bool)] = edges.lens[zero_flow_fix.astype(bool)]

        cb_matrix = spr.vstack([
            spr.hstack([spr.diags(flow_fix_pe), flux_a_in, flux_b_in]),
            spr.hstack([-downstream, spr.diags(A_dn), spr.diags(B_dn)]),
            spr.hstack([-upstream,  spr.diags(A_up), spr.diags(B_up)]),
        ])

        # cb_matrix = spr.vstack([
        #     spr.hstack([spr.diags(flow_fix_pe),           flux_a_in,                                 flux_b_in]),
        #     spr.hstack([-downstream2,                     exp_plus_eff,                              exp_minus2]),
        #     spr.hstack([-upstream,                        spr.diags(np.ones(sid.ne) - zero_flow_fix), spr.diags(np.ones(sid.ne))])
        # ])
        merge_diag = spr.diags((1.0 - inc.merge_vec))
        cb_matrix = merge_diag @ cb_matrix @ merge_diag + spr.diags(inc.merge_vec.astype(float))

        rows_empty = (cb_matrix.getnnz(axis=1) == 0)
        if np.any(rows_empty):
            cb_matrix = cb_matrix + spr.diags(rows_empty.astype(float))

        res = solve_equation(cb_matrix, cb_vector)
        cb = res[:sid.nsq]
        edges.A = res[sid.nsq:sid.nsq + sid.ne]
        edges.B = res[sid.nsq + sid.ne:]

        # # normalize
        # J_in = np.sum(
        #     edges.inlet *
        #     (np.abs(edges.flow) * (edges.A * (1.0 - lam_plus_zero) + edges.B)
        #      - (1.0 - lam_plus_zero) / sid.Pe * edges.diams ** 2 * (lam_plus_val * edges.A - lam_minus_val * edges.B))
        # )
        # if (not np.isfinite(J_in)) or (J_in <= 0):
        #     print(J_in)
        #     raise ValueError("Non-positive or invalid inlet flux during normalization")
        # scale = sid.cb_0 * sid.Q_in / J_in
        # cb *= scale; edges.A *= scale; edges.B *= scale

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
        #cb, lam_plus_val, lam_minus_val, lam_plus_zero, change = solve_transport_and_change_danckwerts(alpha)
        cb, lam_plus_val, lam_minus_val, lam_plus_zero, change = solve_transport_and_change(alpha)

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
    vols.vol = vols.vol_a + vols.vol_e

    # Outputs
    edges.alpha = alpha
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

# def solve_transport_and_change_danckwerts(alpha_edge):
#         # safe inverse |flow|
#         inv_abs_flow = np.divide(1.0, np.abs(edges.flow),
#                                  out=np.zeros_like(edges.flow, dtype=float),
#                                  where=np.abs(edges.flow) > 0)

#         lam_root = np.sqrt(np.abs(edges.flow) ** 2 +
#                            4.0 * alpha_edge * sid.Da / (1.0 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3)
#         lam_plus_val  = sid.Pe / (2.0 * edges.diams ** 2) * (lam_root + np.abs(edges.flow))
#         lam_plus_val = np.array(np.ma.fix_invalid(lam_plus_val, fill_value = 0))
#         lam_minus_val = sid.Pe / (2.0 * edges.diams ** 2) * (lam_root - np.abs(edges.flow))
#         lam_minus_val = np.array(np.ma.fix_invalid(lam_minus_val, fill_value = 0))

#         lam_plus_zero = (lam_plus_val * edges.lens > sid.diffusion_exp_limit).astype(float)
#         lam_plus_val  = lam_plus_val  * (1.0 - lam_plus_zero)
#         lam_minus_val = lam_minus_val * (1.0 - lam_plus_zero)

#         exp_plus_diag   = np.exp(lam_plus_val * edges.lens) * (1.0 - lam_plus_zero) + lam_plus_zero
#         exp_plus2_diag  = np.exp(lam_plus_val * edges.lens) * (1.0 - lam_plus_zero)
#         exp_minus2_diag = np.exp(-lam_minus_val * edges.lens) * (1.0 - lam_plus_zero)

#         exp_plus   = spr.diags(exp_plus_diag)
#         exp_plus2  = spr.diags(exp_plus2_diag)
#         exp_minus2 = spr.diags(exp_minus2_diag)

#         # upstream / downstream
#         F = spr.diags(edges.flow)
#         zero_carrier = spr.diags(((edges.flow == 0) & (edges.diams > 0)).astype(float))
#         A_pos = ((F @ inc.incidence) > 0)
#         Z_pos = ((zero_carrier @ inc.incidence) > 0)
#         A_neg = ((F @ inc.incidence) < 0)
#         Z_neg = ((zero_carrier @ inc.incidence) < 0)
#         upstream   = A_pos.maximum(Z_pos).astype(float)
#         downstream = A_neg.maximum(Z_neg).astype(float)
#         downstream2 = downstream.multiply((1.0 - lam_plus_zero)[:, np.newaxis])

#         # flux blocks
#         flux_a = (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_plus2 @ downstream
#                   + spr.diags(lam_plus_val * edges.diams ** 2) @ upstream
#                   - exp_plus2 @ spr.diags(lam_plus_val * edges.diams ** 2) @ downstream).multiply(
#                       (1.0 - lam_plus_zero)[:, np.newaxis])
#         flux_b = (sid.Pe * spr.diags(np.abs(edges.flow)) @ exp_minus2 @ downstream
#                   - spr.diags(lam_minus_val * edges.diams ** 2) @ upstream
#                   + exp_minus2 @ spr.diags(lam_minus_val * edges.diams ** 2) @ downstream).multiply(
#                       (1.0 - lam_plus_zero)[:, np.newaxis])

#         # Pe fix
#         exp_pe_fix = np.exp(-alpha_edge * sid.Da / (1.0 + sid.G * edges.diams) *
#                             edges.diams * edges.lens * inv_abs_flow)
#         flux_b += sid.Pe * spr.diags(np.abs(edges.flow)) @ spr.diags(lam_plus_zero * exp_pe_fix) @ downstream

#         flux_a_in = -flux_a.T#.multiply((1.0 - graph.in_vec)[:, np.newaxis])
#         flux_b_in = -flux_b.T#.multiply((1.0 - graph.in_vec)[:, np.newaxis])

#         flow_fix_pe = sid.Pe * downstream.T @ np.abs(edges.flow)
#         flow_fix_pe = flow_fix_pe * (1.0 - graph.in_vec)
#         flow_fix_pe += sid.Pe * upstream.T @ np.abs(edges.flow) * graph.in_vec

#         zero_flow_fix = ((edges.flow == 0) & (alpha_edge == 0)).astype(float)
#         exp_plus_eff_diag = exp_plus_diag.copy()
#         exp_plus_eff_diag[zero_flow_fix.astype(bool)] = 1.0
#         exp_plus_eff = spr.diags(exp_plus_eff_diag)

#         cb_matrix = spr.vstack([
#             spr.hstack([spr.diags(flow_fix_pe),           flux_a_in,                                 flux_b_in]),
#             spr.hstack([-downstream2,                     exp_plus_eff,                              exp_minus2]),
#             spr.hstack([-upstream,                        spr.diags(np.ones(sid.ne) - zero_flow_fix), spr.diags(np.ones(sid.ne))])
#         ])
#         merge_diag = spr.diags((1.0 - inc.merge_vec))
#         cb_matrix = merge_diag @ cb_matrix @ merge_diag + spr.diags(inc.merge_vec.astype(float))


#         rows_empty = (cb_matrix.getnnz(axis=1) == 0)
#         if np.any(rows_empty):
#             cb_matrix = cb_matrix + spr.diags(rows_empty.astype(float))

#         res = solve_equation(cb_matrix, cb_vector)
#         cb = res[:sid.nsq]
#         edges.A = res[sid.nsq:sid.nsq + sid.ne]
#         edges.B = res[sid.nsq + sid.ne:]

#         # normalize
#         J_in = np.sum(
#             edges.inlet *
#             (np.abs(edges.flow) * (edges.A * (1.0 - lam_plus_zero) + edges.B)
#              - (1.0 - lam_plus_zero) / sid.Pe * edges.diams ** 2 * (lam_plus_val * edges.A - lam_minus_val * edges.B))
#         )
#         print(J_in)
#         if (not np.isfinite(J_in)) or (J_in <= 0):
#             raise ValueError("Non-positive or invalid inlet flux during normalization")
#         #scale = sid.cb_0 * sid.Q_in / J_in
#         #cb *= scale; edges.A *= scale; edges.B *= scale

#         # edge loss rate ("change")
#         change_pe_fix = lam_plus_zero * 2.0 * edges.B * np.abs(edges.flow) / sid.Da * (
#             1.0 - np.exp(-alpha_edge * sid.Da / (1.0 + sid.G * edges.diams) * edges.diams * edges.lens * inv_abs_flow)
#         )
#         change_pe_fix = np.array(np.ma.fix_invalid(change_pe_fix, fill_value=0.0))
#         change = ((1.0 - lam_plus_zero) * 2.0 * edges.diams ** 2 / (sid.Pe * sid.Da) *
#                   (edges.A * (np.exp(lam_plus_val * edges.lens) - 1.0) * lam_minus_val
#                    + edges.B * (1.0 - np.exp(-lam_minus_val * edges.lens)) * lam_plus_val)
#                   + change_pe_fix)
#         change = np.array(np.ma.fix_invalid(change, fill_value=0.0))

#         return cb, lam_plus_val, lam_minus_val, lam_plus_zero, change
