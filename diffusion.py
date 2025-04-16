
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

def solve_diffusion_vol(sid: SimInputData, inc: Incidence, graph: Graph, edges: Edges, vols: Volumes, cb_vector, data):
    edges.alpha = 1 * ((vols.triangles @ vols.vol_a) > 0)
    #print(np.where(alpha != 1))
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
        print(delta_alpha)
        print(alpha_tr)
        print(vols.vol_a)
        print(vol_a_dissolved)
        print(vols.triangles.T @ df_alpha)
        #print(alpha)
        if np.sum(delta_alpha):
            np.savetxt('df.txt', df_alpha_matrix.toarray())
            raise ValueError
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
