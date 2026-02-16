import numpy as np
import scipy.sparse as spr

from config import SimInputData
from incidence import Incidence
from network_hex import Graph, Edges

from utils import solve_equation

def find_alpha_stream(sid: SimInputData, edges: Edges, inc: Incidence):
    ne = sid.ne

    # shift hierarchies: 0 -> -1 ("no partner"), 1..ne -> 0..ne-1
    h0 = edges.hierarchy0 - 1
    h1 = edges.hierarchy1 - 1

    # budgets: remaining capacity (in) and demand (out) per edge
    q_in_to_allocate  = np.abs(edges.flow).astype(float).copy()
    q_out_to_allocate = np.abs(edges.flow).astype(float).copy()

    # --- build directed edge-edge adjacency correctly ---
    # inc.incidence is assumed to be (ne x nnodes): edge–node incidence
    B = inc.incidence.tocsr()              # (ne, nnodes)
    Fdiag = spr.diags(edges.flow)          # (ne, ne)

    # orientation of flow at each node: O[e, n] = flow[e] * sign(e,n)
    O = Fdiag @ B                          # (ne, nnodes)

    # incoming / outgoing flags at each node
    In  = (O < 0).astype(int)              # edge incoming at node
    Out = (O > 0).astype(int)              # edge outgoing at node

    # edge_inc_directed[row, col] > 0  iff
    # there exists a node where row is incoming and col is outgoing
    edge_inc_directed = (In @ Out.T).tocsr()   # (ne, ne)

    # we’ll store *flux* per pair
    flux_rows, flux_cols, flux_data = [], [], []

    L = max(h0.shape[1], h1.shape[1])
    row_id = np.arange(ne)

    def process_level(v, q_in_to_allocate, q_out_to_allocate):
        """Process one hierarchy column v (one rank) for all edges."""
        nonlocal flux_rows, flux_cols, flux_data

        # v: (ne,), partner indices for this rank, -1 = no partner
        valid = v >= 0
        if not np.any(valid):
            return

        rows_valid = row_id[valid]   # receiver candidates
        v_valid    = v[valid]        # source candidates

        # direction filter, pairwise:
        sub = edge_inc_directed[rows_valid][:, v_valid]
        vals = np.asarray(sub.diagonal()).ravel()
        dir_mask = (vals > 0)

        if not np.any(dir_mask):
            return

        rows_dir = rows_valid[dir_mask]   # true receivers (incoming at some node)
        cols_dir = v_valid[dir_mask]      # true sources  (outgoing at that node)

        # local budgets
        q_in  = q_in_to_allocate[rows_dir]
        q_out = q_out_to_allocate[cols_dir]

        # flux along each pair = min(remaining inlet capacity, remaining outlet flux)
        flux = np.minimum(q_in, q_out)

        nz = flux > 0.0
        if not np.any(nz):
            return

        rows_dir = rows_dir[nz]
        cols_dir = cols_dir[nz]
        flux     = flux[nz]

        # update global budgets
        q_in_to_allocate[rows_dir]  -= flux
        q_out_to_allocate[cols_dir] -= flux

        # store flux contributions
        flux_rows.extend(rows_dir.tolist())
        flux_cols.extend(cols_dir.tolist())
        flux_data.extend(flux.tolist())

    for i in range(L):
        if i < h0.shape[1]:
            v0 = h0[:, i]
            process_level(v0, q_in_to_allocate, q_out_to_allocate)
        # print("after h0 level", i, q_in_to_allocate, q_out_to_allocate)

        if i < h1.shape[1]:
            v1 = h1[:, i]
            process_level(v1, q_in_to_allocate, q_out_to_allocate)
        #print("after h1 level", i, q_in_to_allocate, q_out_to_allocate)

        # early exit if no more outlet demand anywhere (except boundaries)
        if not np.any(q_out_to_allocate > 1e-12):
            break

    # Build flux matrix F[k, j] = flux from edge j into edge k (at some node)
    if flux_data:
        F = spr.csr_matrix((flux_data, (flux_rows, flux_cols)), shape=(ne, ne))
    else:
        F = spr.csr_matrix((ne, ne))

    q_abs = np.abs(edges.flow).astype(float)

    inv_q = np.zeros_like(q_abs)
    mask = q_abs > 0
    inv_q[mask] = 1.0 / q_abs[mask]

    # Scale columns by 1/|q_j|
    D_inv = spr.diags(inv_q)
    alpha = F @ D_inv
    return alpha

def find_full_alpha(edges, inc):
    """
    Full-mixing alpha matrix, no Python loops.

    alpha_full[k, j] = fraction of flux of edge j that goes to edge k
    under *full mixing* at each intersection.

    edges.flow      : (ne,)
    inc.incidence   : (ne, nnodes) sparse incidence (±1, 0)
    """

    B = inc.incidence.tocsr()              # (ne, nnodes)
    q_abs = np.abs(edges.flow).astype(float)
    ne, nnodes = B.shape

    # Orientation of flow at each node: O[e, n] = flow[e] * sign(e,n)
    Fdiag = spr.diags(edges.flow)          # (ne, ne)
    O = Fdiag @ B                          # (ne, nnodes)

    # incoming / outgoing flags at each node
    # In[e, n]  = 1 if edge e is incoming at node n
    # Out[e, n] = 1 if edge e is outgoing at node n
    In  = (O > 0).astype(float)            # (ne, nnodes)
    Out = (O < 0).astype(float)            # (ne, nnodes)

    # For each node n: total outgoing flux s_n = sum_k Out[k,n] * |q_k|
    s_n = Out.T @ q_abs                    # (nnodes,)
    inv_s = np.zeros_like(s_n)
    mask = s_n > 0
    inv_s[mask] = 1.0 / s_n[mask]
    S_inv = spr.diags(inv_s)               # (nnodes, nnodes)

    # Out_w[k, n] = Out[k,n] * |q_k| / s_n  (outlet weight p_k at node n)
    Out_w = spr.diags(q_abs) @ Out @ S_inv   # (ne, nnodes)

    # alpha_full[k, j] = sum over nodes n of Out_w[k,n] * In[j,n]
    #   → for the node where j is incoming, this is p_k; otherwise 0
    alpha_full = Out_w @ In.T              # (ne, ne)

    return alpha_full

def find_concentration(sid: SimInputData, edges: Edges, inc: Incidence, graph: Graph, alpha_matrix):
    #edge_inc_upstream = np.abs(inc.incidence) @ (inc.incidence.T @ spr.diags(edges.flow) < 0)
    c_inlet = 1 * (np.abs(inc.incidence) @ graph.in_vec_a > 0)
    c_matrix = alpha_matrix.T @ spr.diags(np.abs(edges.flow)) - spr.diags(np.abs(edges.flow))
    c_matrix = c_matrix.multiply(1 - edges.inlet[:, np.newaxis]) + spr.diags(edges.inlet)
    c = solve_equation(c_matrix, c_inlet)
    return c

def blend_alpha(alpha_stream, alpha_full, w):
    """
    Blend streamline and full-mixing alpha matrices using
    per-edge weights w (per *source* edge / column).

    alpha_stream, alpha_full : (ne x ne) sparse or dense (but sparse is expected)
    w : (ne,) ndarray, 0 <= w <= 1
    """
    ne = w.shape[0]
    W   = spr.diags(w)
    Wc  = spr.diags(1.0 - w)

    # column-wise blend: alpha_eff[:,j] = w[j]*alpha_stream[:,j] + (1-w[j])*alpha_full[:,j]
    #alpha_eff = alpha_stream @ W + alpha_full @ Wc
    alpha_eff = W @ alpha_stream + Wc @ alpha_full
    return alpha_eff

def compute_edge_Pe(edges, sid):
    """
    Compute per-edge Péclet number and mixing weight w(Pe).

    edges.flow      : (ne,)
    edges.diameter  : (ne,)  (same units as length)
    diffusivity     : scalar D_m
    Pe_c            : scalar Pe_c
    """
    q_abs = np.abs(edges.flow).astype(float)
    d = np.asarray(edges.diams, dtype=float)

    # cylindrical cross-section
    A = d**2
    # avoid division by zero
    A[A == 0.0] = np.inf

    u = q_abs / A             # velocity magnitude
    Pe = u * d  # Pe = u * d / D_m

    # w(Pe) = Pe / (Pe + Pe_c)
    w = np.zeros_like(Pe)
    denom = Pe + sid.Pe_c
    valid = denom > 0.0
    w[valid] = Pe[valid] / denom[valid]

    # clamp numerically
    w = np.clip(w, 0.0, 1.0)
    return w

def calculate_node_weights(edges, node_diams, inc, sid):
    """
    Calculates mixing weights w for nodes based on a Node Peclet number.
    
    Parameters:
    - flow: array of edge flows
    - node_diams: array of node diameters
    - incidence: the incidence matrix (nodes x edges)
    - Pe_c: critical Peclet number from sid
    """
    # 1. Calculate Total Throughput per node (Q_node)
    # Total flow through node = 0.5 * sum of absolute flows of all connected edges
    q_abs = np.abs(edges.flow)
    Q_node = 0.5 * (np.abs(inc.incidence.T) @ q_abs)
    
    # 2. Define Characteristic Node Area (cross-section)
    # A_node ~ d_node**2
    d = np.asarray(node_diams, dtype=float)
    A = d**2
    
    # 3. Calculate Node Velocity and Peclet Number
    # u = Q_node / A
    # Pe = u * d  => (Q_node / d**2) * d => Q_node / d
    # We use a small epsilon to avoid division by zero for clogged/isolated nodes
    Pe_node = np.zeros_like(d)
    valid_d = d > 1e-15
    Pe_node[valid_d] = Q_node[valid_d] / d[valid_d]
    
    # 4. w(Pe) = Pe / (Pe + Pe_c)
    # High Pe -> w = 1 (Streamlined/Advection dominated)
    # Low Pe  -> w = 0 (Mixed/Diffusion dominated)
    w_node = np.zeros_like(Pe_node)
    denom = Pe_node + sid.Pe_c
    valid_w = denom > 0.0
    w_node[valid_w] = Pe_node[valid_w] / denom[valid_w]
    
    # 5. Clamp and Return

    w_node = np.clip(w_node, 0.0, 1.0)
    w_edge = (spr.diags(edges.flow) @ inc.incidence > 0) @ w_node
    return w_node, w_edge

# def additional_mixing(alpha_eff, sid, edges):
#     eps = sid.mixing_at_barrier  # 1% artificial mixing between the two interface channels
#     alpha_eff[edges.special[0], edges.special[1]] += eps
#     alpha_eff[edges.special[1], edges.special[0]] += eps

#     # renormalize each column j so sum_k alpha_eff[k, j] = 1
#     col_sums = np.asarray(alpha_eff.sum(axis=0)).ravel()
#     inv_sum = np.zeros_like(col_sums)
#     mask = col_sums > 0
#     inv_sum[mask] = 1.0 / col_sums[mask]
#     alpha_eff = alpha_eff @ spr.diags(inv_sum)
#     return alpha_eff

import numpy as np
import scipy.sparse as spr

def calculate_node_weights_with_dispersion_baked(edges, node_diams, inc, sid, eps=1e-15):
    """
    Same interface idea as your original: returns w_node,
    but now w_node is reduced by an along-edge dispersion measure.

    No new parameters: uses sid.Pe_c as the only scale.
    """

    q_abs = np.abs(edges.flow)

    # --- your original node weight (junction micromixing) ---
    Q_node = 0.5 * (np.abs(inc.incidence.T) @ q_abs)  # (nn,)
    d_node = np.maximum(np.asarray(node_diams, float), eps)

    Pe_node = Q_node / d_node
    w_node = Pe_node / (Pe_node + sid.Pe_c + eps)
    w_node = np.clip(w_node, 0.0, 1.0)

    # --- dispersion accumulated on edges (heuristic, no new params) ---
    d_edge = np.maximum(np.asarray(edges.diams, float), eps)
    Pe_edge = q_abs / d_edge  # consistent with your Pe_node scaling

    # "dispersion strength" in [0,1], increasing with flow and edge length
    s_edge = 1.0 - np.exp(-edges.lens * (Pe_edge / (sid.Pe_c + eps)))
    s_edge = np.clip(s_edge, 0.0, 1.0)

    # map edge dispersion to nodes: average over incident edges
    A = np.abs(inc.incidence.T)                  # (nn x ne)  node-edge adjacency (0/1)
    deg = np.asarray(A @ np.ones_like(s_edge)).ravel()
    deg = np.maximum(deg, 1.0)

    s_node = np.asarray(A @ s_edge).ravel() / deg
    s_node = np.clip(s_node, 0.0, 1.0)

    # effective streamline weight at node after along-edge spreading
    w_node_eff = np.clip(w_node * (1.0 - s_node), 0.0, 1.0)

    return w_node_eff, s_edge, s_node


def find_alpha(sid, edges, inc, node_diams):
    alpha_full = find_full_alpha(edges, inc)
    alpha_stream = find_alpha_stream(sid, edges, inc)
    #w = compute_edge_Pe(edges, sid)
    w_node, w_edge = calculate_node_weights(edges, node_diams, inc, sid)
    #w_node, w_edge, s_node = calculate_node_weights_with_dispersion_baked(edges, node_diams, inc, sid, eps=1e-15)
    w_node = sid.w_node * np.ones(sid.nsq)
    alpha_eff = blend_alpha(alpha_stream, alpha_full, w_edge)
    #alpha_eff = additional_mixing(alpha_eff, sid, edges)
    #return alpha_eff
    return alpha_eff, alpha_full, alpha_stream, w_node

