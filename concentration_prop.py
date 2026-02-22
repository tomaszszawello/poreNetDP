import numpy as np
import scipy.sparse as spr

from numba import njit

from config import SimInputData
from network import Edges, Graph
from incidence import Incidence
from volumes import Volumes

def create_vector_cA(sid: SimInputData, edges: Edges) -> spr.csc_matrix:
    """ Create vector result for B concentration calculation.
    """
    return np.concatenate([sid.cA_in * edges.inlet])

def create_vector_cB(sid: SimInputData, edges: Edges) -> spr.csc_matrix:
    """ Create vector result for B concentration calculation.
    """
    return np.concatenate([sid.cB_in * edges.inlet])

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


@njit
def _edge_prefactor(Da, G, d, qabs, eps=1e-14):
    if Da <= 0.0 or d <= 0.0 or qabs <= eps:
        return 0.0
    return ((Da / (1.0 + G * d)) * d) / qabs

@njit
def _feoh3_xi_eq(H, Fe, Ksp, Kw, eps=1e-14):
    """
    Find xi in [0, Fe] such that:
      (Fe - xi) * (Kw / (H + 3 xi))^3 = Ksp
    assuming initial state is supersaturated.
    """
    if H <= eps or Fe <= eps:
        return 0.0

    OH = Kw / max(H, eps)
    if Fe * OH * OH * OH <= Ksp + eps:
        return 0.0

    lo = 0.0
    hi = Fe
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        Hm = H + 3.0 * mid
        OHm = Kw / max(Hm, eps)
        fm = (Fe - mid) * OHm * OHm * OHm - Ksp
        if fm > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

@njit
def edge_pfr_caco3_diss_feoh3_prec_HFe_nb(
    H_in, Fe_in,
    q, d,
    Da_diss, Da_prec,
    G, Ksp_feoh3, L,
    Kw=1, nsteps=25, eps=1e-14
):
    """
    A = H+, B = Fe3+.

    CaCO3(s) + 2H+ -> ...     (dissolution)   rate ~ k_diss * H
    Fe3+ + ... -> Fe(OH)3(s)  (precip)        rate ~ k_prec * (Fe*(Kw/H)^3 - Ksp)+

    Returns:
      H_out, Fe_out, solid_FeOH3_measure, dissolved_CaCO3_measure
    """
    H  = float(H_in)  if H_in  > 0.0 else 0.0
    Fe = float(Fe_in) if Fe_in > 0.0 else 0.0

    qv, dv, Lv = float(q), float(d), float(L)
    qabs = qv if qv >= 0.0 else -qv

    if qabs < eps or dv <= 0.0 or Lv <= 0.0:
        return H, Fe, 0.0, 0.0

    k_diss = _edge_prefactor(float(Da_diss), float(G), dv, qabs, eps)
    k_prec = _edge_prefactor(float(Da_prec), float(G), dv, qabs, eps)

    if nsteps < 1:
        nsteps = 1
    dx = Lv / float(nsteps)

    xi_diss_tot = 0.0
    xi_prec_tot = 0.0

    for _ in range(nsteps):
        # --- CaCO3 dissolution: exact H update for dH/dx = -2*k_diss*H
        if k_diss > 0.0 and H > eps:
            Hold = H
            H = Hold * np.exp(-2.0 * k_diss * dx)
            if H < 0.0:
                H = 0.0
            # stoich: 2H+ consumed per 1 CaCO3 dissolved
            xi_diss = 0.5 * (Hold - H)
            if xi_diss > 0.0:
                xi_diss_tot += xi_diss

        # --- Fe(OH)3 precipitation
        if k_prec > 0.0 and Fe > eps and H > eps and Ksp_feoh3 > 0.0:
            OH = Kw / max(H, eps)
            ip = Fe * OH * OH * OH
            if ip > Ksp_feoh3 + eps:
                drive = ip - Ksp_feoh3
                dxi = k_prec * drive * dx

                if dxi > Fe:
                    dxi = Fe
                if dxi < 0.0:
                    dxi = 0.0

                # clamp to equilibrium (avoid overshoot)
                xi_eq = _feoh3_xi_eq(H, Fe, Ksp_feoh3, Kw, eps)
                if dxi > xi_eq:
                    dxi = xi_eq

                if dxi > 0.0:
                    Fe -= dxi
                    H  += 3.0 * dxi
                    xi_prec_tot += dxi

    solid_FeOH3 = 0.0
    if Da_prec > 0.0:
        solid_FeOH3 = qabs * xi_prec_tot / float(Da_prec)

    dissolved_CaCO3 = 0.0
    if Da_diss > 0.0:
        dissolved_CaCO3 = qabs * xi_diss_tot / float(Da_diss)

    return H, Fe, solid_FeOH3, dissolved_CaCO3


@njit
def propagate_nb(
    node_order, out_ptr, out_edges,
    head,
    active_edge, inlet_mask,
    flow, qabs,
    diams, lens,
    Da_diss, Da_prec,          # scalars
    cA_in_bc, cB_in_bc,        # BC arrays (read-only)
    G, Ksp_feoh3,              # scalars
    Kw=1.0, nsteps=25, eps=1e-14
):
    ne = flow.size
    nnodes = node_order.size

    # edge outputs
    cA_out = np.zeros(ne, dtype=np.float64)   # H+
    cB_out = np.zeros(ne, dtype=np.float64)   # Fe3+
    precip_edge_rate = np.zeros(ne, dtype=np.float64)
    diss_edge_rate   = np.zeros(ne, dtype=np.float64)

    # node outputs (mixed concentrations)
    cA_node_out = np.zeros(nnodes, dtype=np.float64)
    cB_node_out = np.zeros(nnodes, dtype=np.float64)

    # ------------------------------------------------------------------
    # Build incoming-edge CSR for perfect mixing using head[]
    # ------------------------------------------------------------------
    counts = np.zeros(nnodes, dtype=np.int64)
    for e in range(ne):
        if not active_edge[e]:
            continue
        h = head[e]
        if h >= 0:
            counts[h] += 1

    in_ptr = np.empty(nnodes + 1, dtype=np.int64)
    in_ptr[0] = 0
    for n in range(nnodes):
        in_ptr[n + 1] = in_ptr[n] + counts[n]

    in_edges = np.empty(in_ptr[nnodes], dtype=np.int64)
    cursor = np.empty(nnodes, dtype=np.int64)
    for n in range(nnodes):
        cursor[n] = in_ptr[n]

    for e in range(ne):
        if not active_edge[e]:
            continue
        h = head[e]
        if h < 0:
            continue
        idx = cursor[h]
        in_edges[idx] = e
        cursor[h] += 1

    # ------------------------------------------------------------------
    # Main propagation in topological order
    # ------------------------------------------------------------------
    for it in range(nnodes):
        n = node_order[it]

        # --- perfect mixing at node n from incoming edges ---
        sI = in_ptr[n]
        eI = in_ptr[n + 1]

        sumq = 0.0
        numA = 0.0
        numB = 0.0

        for p in range(sI, eI):
            j = in_edges[p]
            if not active_edge[j]:
                continue
            qj = qabs[j]
            if qj <= 0.0:
                continue

            sumq += qj

            # Important: if an incoming edge is a boundary inlet edge, its outlet
            # may not have been computed elsewhere. For robustness, treat inlet edges'
            # "incoming concentration" as their prescribed BC directly.
            if inlet_mask[j]:
                numA += qj * cA_in_bc[j]
                numB += qj * cB_in_bc[j]
            else:
                numA += qj * cA_out[j]
                numB += qj * cB_out[j]

        if sumq > eps:
            cA_node = numA / sumq
            cB_node = numB / sumq
        else:
            # No incoming flow: define node concentration from outgoing inlet edges' BCs (if any)
            sO_tmp = out_ptr[n]
            eO_tmp = out_ptr[n + 1]
            sumq2 = 0.0
            numA2 = 0.0
            numB2 = 0.0
            for p in range(sO_tmp, eO_tmp):
                k = out_edges[p]
                if not active_edge[k]:
                    continue
                if not inlet_mask[k]:
                    continue
                qk = qabs[k]
                if qk <= 0.0:
                    continue
                sumq2 += qk
                numA2 += qk * cA_in_bc[k]
                numB2 += qk * cB_in_bc[k]
            if sumq2 > eps:
                cA_node = numA2 / sumq2
                cB_node = numB2 / sumq2
            else:
                cA_node = 0.0
                cB_node = 0.0

        # store node concentrations (by node index)
        cA_node_out[n] = cA_node
        cB_node_out[n] = cB_node

        # --- push node concentrations into outgoing edges + edge reactions ---
        sO = out_ptr[n]
        eO = out_ptr[n + 1]
        for p in range(sO, eO):
            k = out_edges[p]
            if not active_edge[k]:
                continue

            # inlet edges keep their BC; other edges use mixed node value
            Hin  = cA_in_bc[k] if inlet_mask[k] else cA_node
            Fein = cB_in_bc[k] if inlet_mask[k] else cB_node

            Hout, Feout, solid_FeOH3, dissolved_CaCO3 = edge_pfr_caco3_diss_feoh3_prec_HFe_nb(
                Hin, Fein,
                float(flow[k]), float(diams[k]),
                Da_diss, Da_prec,
                G, Ksp_feoh3, float(lens[k]),
                Kw=Kw, nsteps=nsteps, eps=eps
            )

            cA_out[k] = Hout
            cB_out[k] = Feout
            precip_edge_rate[k] = solid_FeOH3
            diss_edge_rate[k]   = dissolved_CaCO3

    return cA_out, cB_out, cA_node_out, cB_node_out, diss_edge_rate, precip_edge_rate

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

    return tail, head

import numpy as np

def propagate_tubes_balls_blend(
    incidence,
    flow, diams, lens,
    inlet_mask, cA_in, cB_in,
    Da, G, Ksp, Da_prec,
    Kw=1.0, nsteps=25, eps=1e-14
):
    active_edge = (diams > 0)
    ne, nnodes = incidence.shape

    # endpoints (your function returns head, tail — keep your convention)
    head, tail = flow_aligned_endpoints(incidence, flow, active_edge)

    node_order = topo_sort_nodes(nnodes, tail, head, active_edge)
    node_order = np.asarray(node_order, dtype=np.int64)

    out_ptr, out_edges = build_out_csr(nnodes, tail, active_edge)

    inlet_mask = inlet_mask.astype(np.bool_)
    qabs = np.abs(flow).astype(np.float64)

    cA_in = np.asarray(cA_in, dtype=np.float64)  # BC array
    cB_in = np.asarray(cB_in, dtype=np.float64)  # BC array

    (cA_out, cB_out,
     cA_node, cB_node,
     diss_edge_rate, precip_edge_rate) = propagate_nb(
        node_order, out_ptr, out_edges,
        head,
        active_edge, inlet_mask,
        flow, qabs,
        diams, lens,
        float(Da), float(Da_prec),
        cA_in, cB_in,
        float(G), float(Ksp),
        Kw=float(Kw), nsteps=int(nsteps), eps=float(eps)
    )

    return cA_out, cB_out, cA_node, cB_node, diss_edge_rate, precip_edge_rate


def update_diameters(sid: SimInputData, inc: Incidence, edges: Edges, graph: Graph, \
    vols: Volumes, diss_edge_rate: np.ndarray, precip_edge_rate: np.ndarray, data) -> tuple[bool, float]:
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
    dissolve, precipitate = 0, 0
    change = diss_edge_rate - precip_edge_rate
    breakthrough = False
    if sid.include_adt:
        #change_rate = change / edges.diams
        change_rate = np.abs(change) / edges.diams_initial ** 2 / edges.lens
        change_rate = np.array(np.ma.fix_invalid(change_rate, fill_value = 0))
        #print(change_rate)
        if np.max(change_rate) == 0:
            breakthrough = True
        else:
            dt_next = sid.growth_rate / float(np.max(change_rate))
        #print(dt_next)
        if dt_next > sid.dt_max:
            dt_next = sid.dt_max
    else:
        dt_next = sid.dt
    
        vols.vol_a_prev = vols.vol_a.copy()
    change = change * sid.dt
    dissolve = diss_edge_rate * sid.dt
    precipitate = precip_edge_rate * sid.dt
    #edge_vols = vols.triangles @ vols.vol_a
    #vol_a_dissolved = (spr.diags(vols.vol_a) @ vols.triangles.T) @ (change / edge_vols)
    data.vol_dissolved += np.sum(dissolve)
    data.vol_precipitated += np.sum(precipitate)
    
    vol_a_dissolved = vols.triangles.T @ (dissolve / edges.triangles)
    vol_e_precipitated = vols.triangles.T @ (precipitate / edges.triangles)
    print(f'Dissolved: {np.sum(vol_a_dissolved)}, Precipitated: {np.sum(vol_e_precipitated)}')
    #print(change)
    vol_a_dissolved = np.array(np.ma.fix_invalid(vol_a_dissolved, fill_value = 0))
    vol_e_precipitated = np.array(np.ma.fix_invalid(vol_e_precipitated, fill_value = 0))
    #vol_a_dissolved = np.min([vol_a_dissolved, vols.vol_a], axis = 0)
    vols.vol_a = np.clip(vols.vol_a - np.abs(vol_a_dissolved), 0, None)
    #vols.vol_e = np.clip(vols.vol_e + np.abs(vol_e_precipitated), 0, vols.vol_max - vols.vol_a)
    vols.vol_e = np.clip(vols.vol_e + np.abs(vol_e_precipitated), 0, vols.vol_max - vols.vol_a)
    #change = vols.triangles @ (vol_a_dissolved / np.array(np.sum(vols.triangles.T, axis = 1))[:, 0])
    # print(vol_a_dissolved)
    #print(change)
    diams_new = edges.diams + change / edges.diams / edges.lens / 2
    #diams_new = np.sqrt(edges.diams ** 2 + change / edges.lens)
    diams_new = np.array(np.ma.fix_invalid(diams_new, fill_value = 0))
    # diams_new = diams_new * (diams_new >= sid.dmin) \
    #     + sid.dmin * (diams_new < sid.dmin)
    diams_new = diams_new * (diams_new > sid.dmin)
    if np.sum(diams_new == 0) != np.sum(edges.diams == 0):
        print("Edges cut")
        print(np.where((diams_new == 0) != (edges.diams == 0)))
        for edge in np.where((diams_new == 0) != (edges.diams == 0)):
            for ind in inc.incidence[edge].nonzero()[1]:
                inc.incidence[edge, ind] = 0
            for ind in inc.inlet[edge].nonzero()[1]:
                inc.inlet[edge, ind] = 0
            edges.inlet[edge] = 0
            edges.outlet[edge] = 0

    if np.max(edges.outlet * edges.diams) > sid.d_break:
        breakthrough = True
        print ('Network dissolved.')
    if np.sum((vols.triangles @ (vols.vol_a == 0)) * edges.outlet) > sid.m / 2:
        breakthrough = True
        print ('Network dissolved.')

    edges.diams = diams_new

    edges.diams_draw = diams_new * (diams_new > 0) + edges.diams_draw * (diams_new == 0)

    
    return breakthrough, dt_next