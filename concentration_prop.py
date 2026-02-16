import numpy as np
import scipy.sparse as spr

from numba import njit

from config import SimInputData
from network import Edges, Graph
from incidence import Incidence

def build_out_csr(nnodes, tail, active_edge):
    act = np.flatnonzero(active_edge & (tail >= 0))
    # sort edges by tail node
    order = np.argsort(tail[act], kind="mergesort")
    edges_sorted = act[order]
    tail_sorted = tail[edges_sorted]

    counts = np.bincount(tail_sorted, minlength=nnodes)
    out_ptr = np.empty(nnodes + 1, dtype=np.int64)
    out_ptr[0] = 0
    np.cumsum(counts, out=out_ptr[1:])
    return out_ptr, edges_sorted.astype(np.int64)

def csr_arrays(A):
    A = A.tocsr(copy=True)
    A.sum_duplicates()
    A.eliminate_zeros()
    return (A.indptr,
            A.indices,
            A.data)


import numpy as np
from numba import njit

@njit
def edge_pfr_analytic_nb(cA_in, cB_in, q, d, Da, G, Ksp, L, eps=1e-14):
    A = float(cA_in) if cA_in > 0.0 else 0.0
    B = float(cB_in) if cB_in > 0.0 else 0.0
    qv, dv, Lv = float(q), float(d), float(L)
    qabs = qv if qv >= 0.0 else -qv

    if qabs < eps or dv <= 0.0 or Da <= 0.0 or Lv <= 0.0:
        return A, B, 0.0
    if A <= 0.0 or B <= 0.0:
        return A, B, 0.0

    Kspv = float(Ksp)
    if Kspv <= 0.0:
        return A, B, 0.0

    # If undersaturated at inlet, no precipitation anywhere on that edge
    if A * B <= Kspv + eps:
        return A, B, 0.0

    # Your hindered prefactor (same as before)
    K = ((Da / (1.0 + G * dv)) * dv) / qabs

    # Normalized-to-inlet drive => divide by (1 - Ksp) (since Omega_ref = 1/Ksp, inlet A=B=1)
    denom = 1.0 - Kspv
    if denom <= eps:
        # degenerate (would mean Ksp ~ 1 in your nondim), just fall back to no reaction
        return A, B, 0.0

    kappa = K / denom  # dξ/dx = kappa * ( (A-ξ)(B-ξ) - Ksp )

    # Roots of (A-ξ)(B-ξ) - Ksp = 0  =>  ξ^2 - (A+B)ξ + (AB - Ksp) = 0
    r = np.sqrt((A - B) * (A - B) + 4.0 * Kspv)
    xi1 = 0.5 * (A + B + r)  # larger root
    xi2 = 0.5 * (A + B - r)  # smaller root (stable equilibrium conversion)

    # Exact solution via partial fractions:
    # (ξ-ξ1)/(ξ-ξ2) = ((0-ξ1)/(0-ξ2)) * exp((ξ1-ξ2)*kappa*x)
    # ξ1-ξ2 = r
    # initial ratio = ξ1/ξ2
    if xi2 <= eps:
        # equilibrium conversion ~0 (shouldn't happen for your regime), do nothing
        return A, B, 0.0

    exp_arg = r * kappa * Lv

    # If exp_arg is huge, ξ -> xi2 (equilibrium) to numerical precision
    if exp_arg > 50.0:
        xi = xi2
    else:
        R = (xi1 / xi2) * np.exp(exp_arg)
        # ξ = (ξ1 - R ξ2) / (1 - R)
        denomR = 1.0 - R
        if np.abs(denomR) < eps:
            xi = xi2
        else:
            xi = (xi1 - R * xi2) / denomR

    # Clamp physically
    if xi < 0.0:
        xi = 0.0
    lim = A if A < B else B
    if xi > lim:
        xi = lim

    Aout = A - xi
    Bout = B - xi

    # Keep your existing “precip volume” convention
    return Aout, Bout, qabs * xi / Da



import numpy as np
from numba import njit

@njit
def node_cstr_analytic_nb(cA_in, cB_in, d_val, q_mix, G_mix, Da, G, Ksp, eps=1e-14):
    A = float(cA_in) if cA_in > 0.0 else 0.0
    B = float(cB_in) if cB_in > 0.0 else 0.0
    dv = float(d_val)
    qv, gval = float(q_mix), float(G_mix)
    qabs = qv if qv >= 0.0 else -qv

    if qabs < eps or gval <= 0.0 or Da <= 0.0 or dv <= 0.0:
        return A, B, 0.0
    if A <= 0.0 or B <= 0.0:
        return A, B, 0.0

    Kspv = float(Ksp)
    if Kspv <= 0.0:
        return A, B, 0.0

    # If undersaturated at inlet, precipitation cannot start in precipitation-only model
    if A * B <= Kspv + eps:
        return A, B, 0.0

    denom = 1.0 - Kspv
    if denom <= eps:
        # Ksp ~ 1 in your nondim -> degenerate for this normalization
        return A, B, 0.0

    # Base CSTR strength (like your original K), with hindering and normalized drive built-in
    K = ((Da / (1.0 + G * dv)) * gval) / qabs
    K /= denom  # corresponds to using (Omega - 1)/(Omega_ref - 1) with Omega_ref=1/Ksp

    # If K is tiny, essentially no reaction
    if K < 1e-30:
        return A, B, 0.0

    delta = B - A  # B_out = A_out + delta

    # Solve: A - x = K*(x*(x+delta) - Ksp)
    # => K x^2 + (1 + K*delta) x - (A + K*Ksp) = 0
    qa = K
    qb = 1.0 + K * delta
    qc = -(A + K * Kspv)

    disc = qb * qb - 4.0 * qa * qc
    if disc < 0.0:
        # numerically shouldn't happen for qa>0, qc<0, but be safe
        return A, B, 0.0

    sqrt_disc = np.sqrt(disc)
    x = (-qb + sqrt_disc) / (2.0 * qa)  # physical root, same branch as your old code

    # Convert to consumption and clamp physically
    xi = A - x
    if xi < 0.0:
        xi = 0.0

    # precipitation-only equilibrium cap: (A-xi)(B-xi) >= Ksp
    # equilibrium conversion solves (A-xi)(B-xi)=Ksp:
    r = np.sqrt((A - B) * (A - B) + 4.0 * Kspv)
    xi_eq = 0.5 * (A + B - r)  # smaller root
    if xi > xi_eq:
        xi = xi_eq

    # stoichiometric limit
    lim = A if A < B else B
    if xi > lim:
        xi = lim

    Aout = A - xi
    Bout = B - xi

    return Aout, Bout, qabs * xi / Da


# --- 2. Main Propagation Loop ---

@njit
def propagate_nb(
    node_order, out_ptr, out_edges,
    head, active_edge, inlet_mask,
    flow, qabs, 
    diams, lens, Da_global, chi0, 
    w_node, node_diam,
    as_indptr, as_indices, as_data,
    af_indptr, af_indices, af_data,
    cA_in, cB_in,
    G_e, G_n, Ksp # Hindering for edges (pipes) and nodes (spheres)
):
    ne, nn = flow.size, node_order.size
    cA_out, cB_out = np.zeros(ne), np.zeros(ne)
    precip_edge_rate, precip_node_rate = np.zeros(ne), np.zeros(node_diam.size)

    for it in range(nn):
        n = node_order[it]
        sN, eN = out_ptr[n], out_ptr[n+1]
        if sN == eN: continue

        wn, Dn = float(w_node[n]), float(node_diam[n])
        phi = 1.0 - wn
        
        # --- 1. Node Hindering ---
        # Applied to the sphere diameter. Usually G_n > G_e.
        
        G_mix_total = Dn**2 * chi0
        L_str_path = Dn * chi0

        Q_node_total = 0.0
        for p in range(sN, eN):
            Q_node_total += qabs[out_edges[p]]
        if Q_node_total < 1e-30: Q_node_total = 1.0

        for p in range(sN, eN):
            k = out_edges[p]
            qk    = qabs[k]
            q_str = wn  * qk
            q_mix = phi * qk
            
            # --- Streamline/Mixed Route Calculation ---
            numA_s = 0.0; numB_s = 0.0; den_s = 0.0
            for t in range(as_indptr[k], as_indptr[k+1]):
                j = as_indices[t]
                if not active_edge[j] or head[j] != n: continue
                f = as_data[t] * qabs[j]
                den_s += f
                numA_s += f * cA_out[j]; numB_s += f * cB_out[j]
            cA_str0 = numA_s / den_s if den_s > 0.0 else 0.0
            cB_str0 = numB_s / den_s if den_s > 0.0 else 0.0

            numA_m = 0.0; numB_m = 0.0; den_m = 0.0
            for t in range(af_indptr[k], af_indptr[k+1]):
                j = af_indices[t]
                if not active_edge[j] or head[j] != n: continue
                f = af_data[t] * qabs[j]
                den_m += f
                numA_m += f * cA_out[j]; numB_m += f * cB_out[j]
            cA_mix0 = numA_m / den_m if den_m > 0.0 else 0.0
            cB_mix0 = numB_m / den_m if den_m > 0.0 else 0.0

            area_frac = qabs[k] / Q_node_total

            # --- Node Reactions (Both hindered by Da_eff_node) ---
            cA_str1, cB_str1, pr_str = edge_pfr_analytic_nb(
                cA_str0, cB_str0, q_str, Dn, Da_global, G_n, Ksp, L_str_path * area_frac
            )
            #cA_str1, cB_str1, pr_str = cA_str0, cB_str0, 0 #edge_pfr_analytic_nb(
                #cA_str0, cB_str0, wn * qabs[k], Dn, Da_eff_node, L_str_path * area_frac
            #)
            cA_mix1, cB_mix1, pr_mix = node_cstr_analytic_nb(
                cA_mix0, cB_mix0, Dn, q_mix, G_mix_total * area_frac, Da_global, G_n, Ksp
            )

            # --- 2. Edge Hindering ---
            dk = float(diams[k])

            if not inlet_mask[k]:
                cA_in[k] = wn * cA_str1 + phi * cA_mix1
                cB_in[k] = wn * cB_str1 + phi * cB_mix1

            # Edge Reaction using pipe-specific hindering
            cA_out[k], cB_out[k], precip_edge_rate[k] = edge_pfr_analytic_nb(
                cA_in[k], cB_in[k], flow[k], dk, Da_global, G_e, Ksp, lens[k]
            )
            precip_node_rate[n] += (pr_str + pr_mix) / chi0

    return cA_out, cB_out, precip_node_rate, precip_edge_rate


from collections import deque


def topo_sort_nodes(nnodes, tail, head, active_edge):
    indeg = np.zeros(nnodes, dtype=int)
    succ = [[] for _ in range(nnodes)]
    for e in range(active_edge.size):
        if not active_edge[e]:
            continue
        u = tail[e]; v = head[e]
        if u < 0 or v < 0:
            continue
        succ[u].append(v)
        indeg[v] += 1

    q = deque(np.where(indeg == 0)[0].tolist())
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if len(order) != nnodes:
        raise ValueError("Cycle in active flow graph (or disconnected due to removals).")
    return order

def build_in_out_edges(nnodes, tail, head, ne, active_edge):
    in_edges  = [[] for _ in range(nnodes)]
    out_edges = [[] for _ in range(nnodes)]
    for e in range(active_edge.size):
        if not active_edge[e]:
            continue
        u = tail[e]; v = head[e]
        if u < 0 or v < 0:
            continue
        out_edges[u].append(e)
        in_edges[v].append(e)
    return in_edges, out_edges



import numpy as np
import scipy.sparse as spr


def flow_aligned_endpoints(incidence, flow, active_edge=None, tol=0.0):
    """
    Fast extraction of flow-aligned tail/head without per-edge getrow().

    Assumes each *active* edge row has exactly two nonzeros: -1 and +1.
    For flow>0: tail = node(-1), head = node(+1)
    For flow<0: tail = node(+1), head = node(-1)
    """
    B = incidence.tocsr() if spr.issparse(incidence) else spr.csr_matrix(incidence)
    flow = np.asarray(flow, float)

    ne, nn = B.shape
    indptr = B.indptr
    indices = B.indices
    data = B.data

    # active edge default: exactly 2 nonzeros
    if active_edge is None:
        row_nnz = np.diff(indptr)
        active_edge = (row_nnz == 2)
    active_edge = np.asarray(active_edge, bool)
    if active_edge.size != ne:
        raise ValueError("active_edge must have length ne")

    # init outputs
    tail = np.full(ne, -1, dtype=np.int64)
    head = np.full(ne, -1, dtype=np.int64)

    # edges we will process
    act = np.flatnonzero(active_edge & (np.abs(flow) > tol))
    if act.size == 0:
        return tail, head

    # For each active row e, grab the slice [indptr[e], indptr[e+1]) which should be length 2
    s = indptr[act]
    # we know nnz==2, so the two entries live at positions s and s+1
    i0 = s
    i1 = s + 1

    n0 = indices[i0]
    n1 = indices[i1]
    v0 = data[i0]
    v1 = data[i1]

    # Identify which is -1 and which is +1.
    # (Be tolerant: values might be floats very close to ±1)
    is0_minus = v0 < 0
    is1_minus = v1 < 0

    # rows are malformed if both entries have same sign (or not exactly ±1)
    good = is0_minus ^ is1_minus
    act = act[good]
    n0 = n0[good]; n1 = n1[good]
    is0_minus = is0_minus[good]

    # u_minus = node where value is -1, u_plus = node where value is +1
    u_minus = np.where(is0_minus, n0, n1).astype(np.int64)
    u_plus  = np.where(is0_minus, n1, n0).astype(np.int64)

    # apply flow sign
    pos = flow[act] > 0
    # flow>0: (-1)->(+1)
    tail[act[pos]] = u_minus[pos]
    head[act[pos]] = u_plus[pos]
    # flow<0: (+1)->(-1)
    tail[act[~pos]] = u_plus[~pos]
    head[act[~pos]] = u_minus[~pos]

    return head, tail

from line_profiler import profile
@profile
def propagate_tubes_balls_blend(incidence,
    flow, diams, lens, node_diam,
    inlet_mask, cA_in, cB_in,                 # full edge-length arrays
    alpha_stream, alpha_full, w_node,         # for blend at nodes
    Da, G, Ksp, chi0, k_node=1.0, tau_node=1.0,
    precip_yield=1.0
    ):
    # --- Python/SciPy: update geometry/flow/alpha ---
    active_edge = (diams > 0)
    ne, nnodes = incidence.shape

    # endpoints
    tail, head = flow_aligned_endpoints(incidence, flow, active_edge)

    # topo order (Python)
    node_order = topo_sort_nodes(nnodes, tail, head, active_edge)
    node_order = np.asarray(node_order, dtype=np.int64)

    # node->out edges CSR (Python/NumPy)
    out_ptr, out_edges = build_out_csr(nnodes, tail, active_edge)

    # alpha CSR arrays (Python/SciPy)
    as_indptr, as_indices, as_data = csr_arrays(alpha_stream)
    af_indptr, af_indices, af_data = csr_arrays(alpha_full)

    inlet_mask = inlet_mask.astype(np.bool_)
    qabs = np.abs(flow).astype(np.float64)

    # --- Numba kernel ---
    cA_out, cB_out, precip_node_rate, precip_edge_rate = propagate_nb(
        node_order, out_ptr, out_edges,
        head, active_edge, inlet_mask,
        flow, qabs, diams, lens, Da, chi0,
        w_node, node_diam,
        as_indptr, as_indices, as_data,
        af_indptr, af_indices, af_data,
        cA_in, cB_in,
        G, G * 2, Ksp)
    return cA_out, cB_out, precip_node_rate, precip_edge_rate



def update_diameters(sid: SimInputData, inc: Incidence, edges: Edges, graph: Graph, \
    prec_node, prec_edge, node_diam) -> tuple[bool, float]:
    """ Update diameters.

    This function updates diameters of edges, calculates the next timestep (if
    adt is used) and checks if the network is dissolved. Based on config, we
    include either dissolution or both dissolution and precipitation.

    Parameters
    -------
    sid : simInputData class object
        all config parameters of the simulation
        include_cc : bool
        dmin : float
        dmin_th : float
        d_break : float
        include_adt : bool
        growth_rate : float
        dt : float
        dt_max : float

    inc : Incidence class object
        matrices of incidence

    edges : Edges class object
        all edges in network and their parameters
        diams : numpy ndarray (ne)
        lens : numpy ndarray (ne)
        outlet : numpy ndarray (ne)

    cb : numpy ndarray (nsq)
        vector of substance B concentration

    cc : numpy ndarray (nsq)
        vector of substance C concentration

    Returns
    -------
    breakthrough : bool
        parameter stating if the system was dissolved (if diameter of output
        edge grew at least to sid.d_break)

    dt_next : float
        new timestep
    """
    change = prec_edge / (edges.diams * edges.lens)
    change = np.array(np.ma.fix_invalid(change, fill_value = 0))
    change_node = prec_node / node_diam ** 2
    change_node = np.array(np.ma.fix_invalid(change_node, fill_value = 0))
    breakthrough = False
    if sid.include_adt:
        change_rate = change / edges.diams * (edges.diams > 0.01)
        change_rate = np.array(np.ma.fix_invalid(change_rate, fill_value = 0))
        rate = np.max(np.abs(change_rate))
        if rate == 0:
            rate = 10
        dt_next = sid.growth_rate / rate
        if dt_next > sid.dt_max:
            dt_next = sid.dt_max
            breakthrough = True
    else:
        dt_next = sid.dt
    diams_new = edges.diams - change * dt_next
    edges.diams_min = diams_new * (diams_new > 0) + edges.diams_min * (diams_new == 0)
    node_diam -= change_node * dt_next
    node_diam *= node_diam > sid.node_diam_min
    diams_new = diams_new * (1 - (np.abs(inc.incidence) @ (node_diam == 0)))
    if np.sum(diams_new == 0) != np.sum(edges.diams == 0):
        print("Edges cut")
        #diams_zero = 1 * (np.abs(inc.incidence) @ (inc.incidence.T @ spr.diags(edges.flow) > 0) @ (diams_new == 0) > 0) * (edges.diams != 0)
        #diams_new = diams_new * (1 - diams_zero)
        inc.incidence = inc.incidence.multiply(1 * (diams_new > 0)[:, np.newaxis])
        zero_nodes = np.array(np.abs(inc.incidence).sum(axis = 0) == 1)[0] * (1 - graph.in_vec - graph.out_vec)
        zero_edges = np.abs(inc.incidence) @ zero_nodes
        print(np.sum(zero_edges))
        diams_new = diams_new * (1 - zero_edges)
        inc.incidence = inc.incidence.multiply(1 * (diams_new > 0)[:, np.newaxis])
        inc.inlet = inc.inlet.multiply(1 * (diams_new > 0)[:, np.newaxis])
        edges.inlet *= diams_new > 0
        edges.outlet *= diams_new > 0         


    # if sid.include_adt:
    #     diams_rate = np.abs((diams_new - edges.diams) / edges.diams)
    #     diams_rate = np.array(np.ma.fix_invalid(diams_rate, fill_value = 0))
    #     dt_next = sid.growth_rate / sid.dt / np.max(diams_rate)
    #     if dt_next > sid.dt_max:
    #         dt_next = sid.dt_max

    edges.diams = diams_new
    # if np.max(edges.diams / edges.diams_initial) > 300:
    #     breakthrough = True
    return breakthrough, dt_next, node_diam

import numpy as np

def compute_drive(cA, cB, Ksp, eps=1e-30):
    """
    Normalized precipitation driving force consistent with your
    (Omega-1)/(Omega_ref-1) choice when c_refA=c_refB=1:

        drive = max( (cA*cB/Ksp) - 1, 0 ) / ( (1/Ksp) - 1 )
              = max( cA*cB - Ksp, 0 ) / (1 - Ksp)

    Returns drive >= 0 (dimensionless).
    """
    Ksp = float(Ksp)
    if Ksp <= 0.0:
        return np.zeros_like(cA, dtype=float)

    denom = max(1.0 - Ksp, eps)
    return np.maximum(cA * cB - Ksp, 0.0) / denom


def compute_f_eff(edges,
                  node_diam,
                  Ksp,
                  cA_edge, cB_edge,
                  cA_node=None, cB_node=None,
                  active_edge=None,
                  active_node=None,
                  include_nodes=True,
                  eps=1e-30):
    """
    Surface-area-weighted mean drive (mixing limitation factor).

    edges: has edges.diams (ne,), edges.lens (ne,), optionally edges.flow
    node_diam: (nn,)
    cA_edge,cB_edge: (ne,) concentrations *at the place you apply edge reaction*
                     (typically edge inlet in your PFR)
    cA_node,cB_node: (nn,) concentrations *at the place you apply node reaction*
                     (typically mixed-in node concentrations). Optional.

    Returns
    -------
    f_eff : float in [0, +inf) (usually <=1 early on)
    info  : dict with edge/node pieces for debugging
    """
    ne = len(edges.diams)
    nn = len(node_diam)

    cA_edge = np.asarray(cA_edge, float).reshape(ne)
    cB_edge = np.asarray(cB_edge, float).reshape(ne)

    # masks
    if active_edge is None:
        active_edge = (np.asarray(edges.diams, float) > 0.0) & (np.asarray(edges.lens, float) > 0.0)
    else:
        active_edge = np.asarray(active_edge, bool).reshape(ne)

    if active_node is None:
        active_node = (np.asarray(node_diam, float) > 0.0)
    else:
        active_node = np.asarray(active_node, bool).reshape(nn)

    # --- edges (tubes): area ~ pi*d*L, pi cancels in ratios ---
    d_e = np.asarray(edges.diams, float)
    L_e = np.asarray(edges.lens, float)
    A_e = (d_e * L_e) * active_edge

    drive_e = compute_drive(cA_edge, cB_edge, Ksp)
    num_e = float(np.sum(A_e * drive_e))
    den_e = float(np.sum(A_e) + eps)

    # --- nodes (balls): area ~ pi*D^2 (or 4*pi*r^2), constant cancels ---
    num_n = 0.0
    den_n = 0.0
    drive_n = None
    if include_nodes and (cA_node is not None) and (cB_node is not None):
        cA_node = np.asarray(cA_node, float).reshape(nn)
        cB_node = np.asarray(cB_node, float).reshape(nn)

        Dn = np.asarray(node_diam, float)
        A_n = (Dn * Dn) * active_node
        drive_n = compute_drive(cA_node, cB_node, Ksp)

        num_n = float(np.sum(A_n * drive_n))
        den_n = float(np.sum(A_n) + eps)

    # combined surface-weighted mean drive
    num = num_e + (num_n if include_nodes else 0.0)
    den = den_e + (den_n if (include_nodes and den_n > 0.0) else 0.0)

    f_eff = num / max(den, eps)


    return f_eff