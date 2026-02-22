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
    # equilibrium xi such that (Fe-xi)*(Kw/(H+3xi))^3 = Ksp
    if H <= eps or Fe <= eps:
        return 0.0
    OH = Kw / max(H, eps)
    if Fe * OH * OH * OH <= Ksp + eps:
        return 0.0
    lo, hi = 0.0, Fe
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
    # caps on total extent along this edge in this timestep (per unit fluid volume)
    xi_diss_cap, xi_prec_cap,
    Kw=1.0, nsteps=25, eps=1e-14
):
    """
    Returns:
      H_out, Fe_out, xi_diss, xi_prec
    where xi_* are the ACTUAL extents used (already capped).
    """
    H  = float(H_in)  if H_in  > 0.0 else 0.0
    Fe = float(Fe_in) if Fe_in > 0.0 else 0.0

    qv, dv, Lv = float(q), float(d), float(L)
    qabs = qv if qv >= 0.0 else -qv

    if qabs < eps or dv <= 0.0 or Lv <= 0.0:
        return H, Fe, 0.0, 0.0

    # If caps are exhausted, shut the reaction off safely
    if xi_diss_cap <= 0.0:
        Da_diss = 0.0
    if xi_prec_cap <= 0.0:
        Da_prec = 0.0

    k_diss = _edge_prefactor(float(Da_diss), float(G), dv, qabs, eps)
    k_prec = _edge_prefactor(float(Da_prec), float(G), dv, qabs, eps)

    if nsteps < 1:
        nsteps = 1
    dx = Lv / float(nsteps)

    xi_diss_tot = 0.0
    xi_prec_tot = 0.0

    for _ in range(nsteps):
        # --- dissolution: exact step, then cap remaining extent ---
        if k_diss > 0.0 and H > eps and xi_diss_tot < xi_diss_cap - eps:
            Hold = H
            Hexp = Hold * np.exp(-2.0 * k_diss * dx)
            xi_step = 0.5 * (Hold - Hexp)

            rem = xi_diss_cap - xi_diss_tot
            if xi_step > rem:
                xi_step = rem
                # enforce stoichiometry exactly when we cap mid-step
                H = Hold - 2.0 * xi_step
                if H < 0.0:
                    H = 0.0
            else:
                H = Hexp

            if xi_step > 0.0:
                xi_diss_tot += xi_step

        # --- precipitation: drive-based step, cap remaining extent ---
        if k_prec > 0.0 and Fe > eps and H > eps and Ksp_feoh3 >= 0.0 and xi_prec_tot < xi_prec_cap - eps:
            OH = Kw / max(H, eps)
            ip = Fe * OH * OH * OH
            if ip > Ksp_feoh3 + eps:
                drive = ip - Ksp_feoh3
                dxi = k_prec * drive * dx

                # physical caps
                if dxi > Fe:
                    dxi = Fe
                if dxi < 0.0:
                    dxi = 0.0

                # equilibrium cap (avoid overshoot to undersat)
                xi_eq = _feoh3_xi_eq(H, Fe, Ksp_feoh3, Kw, eps)
                if dxi > xi_eq:
                    dxi = xi_eq

                # timestep capacity cap
                remp = xi_prec_cap - xi_prec_tot
                if dxi > remp:
                    dxi = remp

                if dxi > 0.0:
                    Fe -= dxi
                    H  += 3.0 * dxi
                    xi_prec_tot += dxi

    return H, Fe, xi_diss_tot, xi_prec_tot


@njit
def propagate_nb(
    node_order, out_ptr, out_edges,
    head,
    active_edge, inlet_mask,
    flow, qabs,
    diams, lens,
    edge_grains,
    vol_a, vol_e, vol_max,  # (ng,) grain volumes (updated in-place)
    dt,
    Da_diss, Da_prec,   # scalars
    cA_in_bc, cB_in_bc, # BC arrays per edge (read-only)
    G, Ksp_feoh3,       # scalars
    Gamma,              # Vm_calcite / Vm_precip
    Kw=1.0, nsteps=25, eps=1e-14
):
    ne = flow.size
    nnodes = node_order.size

    # edge outputs
    cA_out = np.zeros(ne, dtype=np.float64)   # H+
    cB_out = np.zeros(ne, dtype=np.float64)   # Fe3+
    xi_diss_edge = np.zeros(ne, dtype=np.float64)
    xi_prec_edge = np.zeros(ne, dtype=np.float64)

    # node outputs
    cA_node_out = np.zeros(nnodes, dtype=np.float64)
    cB_node_out = np.zeros(nnodes, dtype=np.float64)

    # molar volumes in your volume units:
    Vm_cal  = 1.0
    Vm_prec = 1.0 / max(Gamma, eps)

    # ------------------------------------------------------------------
    # Build incoming-edge CSR based on head[]
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

        # --- mix at node from incoming edges ---
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
            # IMPORTANT: ALWAYS use edge OUTLET concentrations for mixing
            # (including inlet edges, which were reacted when their tail node was processed)
            numA += qj * cA_out[j]
            numB += qj * cB_out[j]

        if sumq > eps:
            cA_node = numA / sumq
            cB_node = numB / sumq
        else:
            # SOURCE NODE: no incoming edges.
            # Define node concentration from outgoing inlet edges' BCs (if any), else 0.
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

        cA_node_out[n] = cA_node
        cB_node_out[n] = cB_node

        # --- outgoing edges ---
        sO = out_ptr[n]
        eO = out_ptr[n + 1]
        for p in range(sO, eO):
            k = out_edges[p]
            if not active_edge[k]:
                continue

            Hin  = cA_in_bc[k] if inlet_mask[k] else cA_node
            Fein = cB_in_bc[k] if inlet_mask[k] else cB_node

            # --- compute caps from neighboring grains for THIS timestep ---
            g1 = edge_grains[k, 0]
            g2 = edge_grains[k, 1]

            # available calcite volume near this edge
            vcal = 0.0
            if g1 >= 0:
                vcal += vol_a[g1]
            if g2 >= 0:
                vcal += vol_a[g2]

            # available precipitation space near this edge
            vspace = 0.0
            if g1 >= 0:
                cap1 = vol_max[g1] - (vol_a[g1] + vol_e[g1])
                if cap1 > 0.0:
                    vspace += cap1
            if g2 >= 0:
                cap2 = vol_max[g2] - (vol_a[g2] + vol_e[g2])
                if cap2 > 0.0:
                    vspace += cap2

            qkabs = qabs[k]
            denom = qkabs * dt

            # convert volume limits -> extent caps (per fluid volume)
            if denom > eps and vcal > 0.0:
                xi_diss_cap = (vcal / Vm_cal) / denom
            else:
                xi_diss_cap = 0.0

            if denom > eps and vspace > 0.0:
                xi_prec_cap = (vspace / Vm_prec) / denom
            else:
                xi_prec_cap = 0.0

            # --- edge chemistry with caps ---
            Hout, Feout, xi_diss, xi_prec = edge_pfr_caco3_diss_feoh3_prec_HFe_nb(
                Hin, Fein,
                float(flow[k]), float(diams[k]),
                Da_diss, Da_prec,
                G, Ksp_feoh3, float(lens[k]),
                xi_diss_cap, xi_prec_cap,
                Kw=Kw, nsteps=nsteps, eps=eps
            )

            cA_out[k] = Hout
            cB_out[k] = Feout
            xi_diss_edge[k] = xi_diss
            xi_prec_edge[k] = xi_prec

            # --- update grain volumes in-place using ACTUAL extents ---
            if denom > eps:
                # moles over dt
                n_diss = qkabs * dt * xi_diss
                n_prec = qkabs * dt * xi_prec
                # volumes over dt
                Vd = n_diss * Vm_cal
                Vp = n_prec * Vm_prec

                # dissolve calcite: subtract from vol_a[g1], vol_a[g2]
                if Vd > 0.0:
                    if g1 >= 0 and g2 >= 0:
                        take1 = 0.5 * Vd
                        if take1 > vol_a[g1]:
                            take1 = vol_a[g1]
                        take2 = Vd - take1
                        if take2 > vol_a[g2]:
                            take2 = vol_a[g2]
                        rem = Vd - (take1 + take2)
                        if rem > 0.0:
                            # try to take remainder from whichever still has calcite
                            extra = vol_a[g1] - take1
                            if extra > 0.0:
                                add = rem if rem < extra else extra
                                take1 += add
                                rem -= add
                            if rem > 0.0:
                                extra = vol_a[g2] - take2
                                if extra > 0.0:
                                    add = rem if rem < extra else extra
                                    take2 += add
                                    rem -= add
                        vol_a[g1] -= take1
                        vol_a[g2] -= take2
                    elif g1 >= 0:
                        take = Vd if Vd < vol_a[g1] else vol_a[g1]
                        vol_a[g1] -= take
                    elif g2 >= 0:
                        take = Vd if Vd < vol_a[g2] else vol_a[g2]
                        vol_a[g2] -= take

                # precipitate: add to vol_e[g1], vol_e[g2] but keep vol_a+vol_e<=vol_max
                if Vp > 0.0:
                    if g1 >= 0 and g2 >= 0:
                        cap1 = vol_max[g1] - (vol_a[g1] + vol_e[g1])
                        if cap1 < 0.0:
                            cap1 = 0.0
                        cap2 = vol_max[g2] - (vol_a[g2] + vol_e[g2])
                        if cap2 < 0.0:
                            cap2 = 0.0

                        put1 = 0.5 * Vp
                        if put1 > cap1:
                            put1 = cap1
                        put2 = Vp - put1
                        if put2 > cap2:
                            put2 = cap2
                        rem = Vp - (put1 + put2)
                        if rem > 0.0:
                            extra = cap1 - put1
                            if extra > 0.0:
                                add = rem if rem < extra else extra
                                put1 += add
                                rem -= add
                            if rem > 0.0:
                                extra = cap2 - put2
                                if extra > 0.0:
                                    add = rem if rem < extra else extra
                                    put2 += add
                                    rem -= add

                        vol_e[g1] += put1
                        vol_e[g2] += put2
                    elif g1 >= 0:
                        cap = vol_max[g1] - (vol_a[g1] + vol_e[g1])
                        if cap > 0.0:
                            put = Vp if Vp < cap else cap
                            vol_e[g1] += put
                    elif g2 >= 0:
                        cap = vol_max[g2] - (vol_a[g2] + vol_e[g2])
                        if cap > 0.0:
                            put = Vp if Vp < cap else cap
                            vol_e[g2] += put

    return cA_out, cB_out, cA_node_out, cB_node_out, xi_diss_edge, xi_prec_edge, vol_a, vol_e

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


def extract_edge_grains(triangles_mat, ne=None):
    """
    triangles_mat: shape (ne, ngrains), with 1 where edge neighbors a grain.
                   Can be scipy sparse or dense.
    Returns:
        edge_grains: (ne,2) int64 array, each row [g1,g2] (or -1 if missing)
    """
    if spr.issparse(triangles_mat):
        T = triangles_mat.tocsr()
        ne_ = T.shape[0] if ne is None else ne
        edge_grains = np.full((ne_, 2), -1, dtype=np.int64)

        indptr = T.indptr
        indices = T.indices

        for e in range(ne_):
            s = indptr[e]
            t = indptr[e + 1]
            deg = t - s
            if deg <= 0:
                continue
            if deg == 1:
                edge_grains[e, 0] = indices[s]
            else:
                # assume 2; if more, we take first two (or raise if you prefer)
                edge_grains[e, 0] = indices[s]
                edge_grains[e, 1] = indices[s + 1]
        return edge_grains

    # dense case
    T = np.asarray(triangles_mat)
    ne_ = T.shape[0] if ne is None else ne
    edge_grains = np.full((ne_, 2), -1, dtype=np.int64)
    for e in range(ne_):
        cols = np.flatnonzero(T[e])
        if cols.size >= 1:
            edge_grains[e, 0] = int(cols[0])
        if cols.size >= 2:
            edge_grains[e, 1] = int(cols[1])
    return edge_grains


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
    triangles, vol_a, vol_e, vol_max,
    dt,
    Gamma,
    Kw=1.0, nsteps=25, eps=1e-14
):
    active_edge = (diams > 0)
    ne, nnodes = incidence.shape
    edge_grains = extract_edge_grains(triangles, ne=ne)
    edge_grains = np.asarray(edge_grains, dtype=np.int64)

    # your endpoint function returns (head, tail) in your code
    head, tail = flow_aligned_endpoints(incidence, flow, active_edge)

    node_order = topo_sort_nodes(nnodes, tail, head, active_edge)
    node_order = np.asarray(node_order, dtype=np.int64)

    out_ptr, out_edges = build_out_csr(nnodes, tail, active_edge)

    inlet_mask = inlet_mask.astype(np.bool_)
    qabs = np.abs(flow).astype(np.float64)

    cA_in = np.asarray(cA_in, dtype=np.float64)
    cB_in = np.asarray(cB_in, dtype=np.float64)


    vol_a = np.asarray(vol_a, dtype=np.float64)         # (ng,)
    vol_e = np.asarray(vol_e, dtype=np.float64)         # (ng,)
    vol_max = np.asarray(vol_max, dtype=np.float64)     # (ng,)

    (cA_out, cB_out,
     cA_node, cB_node,
     xi_diss_edge, xi_prec_edge, vol_a, vol_e) = propagate_nb(
        node_order, out_ptr, out_edges,
        head,
        active_edge, inlet_mask,
        flow, qabs,
        diams, lens,
        edge_grains, vol_a, vol_e, vol_max,
        float(dt),
        float(Da), float(Da_prec),
        cA_in, cB_in,
        float(G), float(Ksp),
        float(Gamma),
        Kw=float(Kw), nsteps=int(nsteps), eps=float(eps)
    )

    return cA_out, cB_out, cA_node, cB_node, xi_diss_edge, xi_prec_edge, vol_a, vol_e


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
    #vols.vol_a = np.clip(vols.vol_a - np.abs(vol_a_dissolved), 0, None)
    #vols.vol_e = np.clip(vols.vol_e + np.abs(vol_e_precipitated), 0, vols.vol_max - vols.vol_a)
    #vols.vol_e = np.clip(vols.vol_e + np.abs(vol_e_precipitated), 0, vols.vol_max - vols.vol_a)
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