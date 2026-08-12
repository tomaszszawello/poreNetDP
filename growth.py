""" Updates edges diameters based on dissolution and precipitation.

This module calculates the change od diameters in the network, resulting from
dissolution (and precipitation, if enabled). Based on that change, new
timestep is calculated.

Notable functions
-------
update_diameters(SimInputData, Incidence, Edges, np.ndarray, np.ndarray) \
    -> tuple[bool, float]
    update diameters, calculate timestep and check if network is dissolved
update_diameters_d0_hindering(...)
    drop-in geometry update for the D0-dependent precipitation model
solve_dp_vol_d0_hindering(...)
    edgewise dissolution/precipitation volume rates consistent with the
    D0-dependent transport solver
"""

import numpy as np
import scipy.sparse as spr

from config import SimInputData
from network import Edges, Graph
from incidence import Incidence
from volumes import Volumes

from utils import keep_largest_component

def update_diameters(sid: SimInputData, inc: Incidence, edges: Edges, graph: Graph, \
    vols: Volumes, cb: np.ndarray, cc: np.ndarray, cd: np.ndarray, data,
    use_d0_hindering: bool = True) -> tuple[bool, float]:
    """ Update diameters.

    This function updates diameters of edges, calculates the next timestep (if
    adt is used) and checks if the network is dissolved. Based on config, we
    include either dissolution or both dissolution and precipitation.

    When ``use_d0_hindering`` is true, the precipitation contribution to the
    geometry update is reconstructed with the same concentration-dependent
    transverse-hindering law as
    ``solve_precipitation_safe_d0_hindering``::

        P(D0) = Da*K*d*(D0/Kp) / [1 + G*K*d*(D0/Kp)].

    The default is false so existing simulations retain the legacy growth law.

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

    use_d0_hindering : bool, optional
        If true, use :func:`solve_dp_vol_d0_hindering` for the coupled
        dissolution--precipitation geometry update.  This option requires
        ``include_volumes=True`` and should be paired with
        ``solve_precipitation_safe_d0_hindering``.

    Returns
    -------
    breakthrough : bool
        parameter stating if the system was dissolved (if diameter of output
        edge grew at least to sid.d_break)

    dt_next : float
        new timestep
    """
    dissolve, precipitate = 0, 0
    if sid.include_precipitation:
        if sid.include_volumes:
            if use_d0_hindering:
                change, dissolve, precipitate = solve_dp_vol_d0_hindering(
                    sid, inc, edges, vols, cb, cc, cd
                )
            else:
                change, dissolve, precipitate = solve_dp_vol(
                    sid, inc, edges, vols, cb, cc, cd
                )
        else:
            if use_d0_hindering:
                raise NotImplementedError(
                    "D0-dependent precipitation growth currently requires "
                    "sid.include_volumes=True."
                )
            change = solve_dp(sid, inc, edges, cb, cc)
    else:
        if sid.include_volumes:
            change, dissolve, precipitate = solve_d_vol(sid, inc, edges, vols, cb)
        else:
            change = solve_d(sid, inc, edges, cb)
    breakthrough = False
    if sid.include_adt:
        #change_rate = change / edges.diams
        change_rate = np.abs(change) / edges.diams_initial ** 2 / edges.lens
        change_rate = np.array(np.ma.fix_invalid(change_rate, fill_value = 0))
        #print(change_rate)
        if np.max(change_rate) == 0:
            breakthrough = True
            dt_next = sid.dt
        else:
            dt_next = sid.growth_rate / float(np.max(change_rate))
        #print(dt_next)
        if dt_next > sid.dt_max:
            dt_next = sid.dt_max
    else:
        dt_next = sid.dt
    
    vols.vol_a_prev = vols.vol_a.copy()
    change = change * sid.dt
    dissolve = dissolve * sid.dt
    precipitate = precipitate * sid.dt
    #edge_vols = vols.triangles @ vols.vol_a
    #vol_a_dissolved = (spr.diags(vols.vol_a) @ vols.triangles.T) @ (change / edge_vols)
    vol_a_dissolved = vols.triangles.T @ (dissolve / edges.triangles)
    vol_e_precipitated = vols.triangles.T @ (precipitate / edges.triangles)
    print(f'Dissolved: {np.sum(vol_a_dissolved)}, Precipitated: {np.sum(vol_e_precipitated)}')
    vol_a_dissolved = np.array(np.ma.fix_invalid(vol_a_dissolved, fill_value = 0))
    vol_e_precipitated = np.array(np.ma.fix_invalid(vol_e_precipitated, fill_value = 0))

    # snapshot before clips so the trackers record what actually changed
    vol_a_before = vols.vol_a.copy()
    vol_e_before = vols.vol_e.copy()
    vols.vol_a = np.clip(vols.vol_a - np.abs(vol_a_dissolved), 0, None)
    vols.vol_e = np.clip(vols.vol_e + np.abs(vol_e_precipitated), 0, vols.vol_max - vols.vol_a)
    # increment from actual post-clip changes, not from the raw (potentially over-large) rates
    data.vol_dissolved    += np.sum(vol_a_before - vols.vol_a)
    data.vol_precipitated += np.sum(vols.vol_e   - vol_e_before)
    # accumulate flux-based expectations for cumulative cross-checks in check_mass_balance
    if sid.include_volumes:
        abs_q_g = np.abs(edges.flow)
        dn_g    = 1 * (spr.diags(edges.flow) @ inc.incidence < 0)
        cb_dn_g = np.asarray(dn_g @ cb).ravel()
        B_consumed = (np.sum(edges.inlet * abs_q_g) * sid.cb_in
                      - np.sum(edges.outlet * abs_q_g * cb_dn_g))
        data.A_vol_expected_cumulative += B_consumed / sid.Da * sid.dt
    if sid.include_precipitation:
        # Use the growth formula's pre-clip output: same basis as vol_e_precipitated,
        # eliminates the D-flux vs growth-formula mismatch (Component 2 of mass balance).
        # Residual V_E - E_expect now reflects only hard-clip events (Component 1).
        data.E_vol_expected_cumulative += np.sum(np.abs(vol_e_precipitated))
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
        

        # zero_nodes = (1 - graph.in_vec) * (np.abs(inc.incidence.T) @ edges.inlet == np.abs(inc.incidence.T) @ np.ones(sid.ne)) + (1 - graph.out_vec) * (np.abs(inc.incidence.T) @ edges.outlet == np.abs(inc.incidence.T) @ np.ones(sid.ne))
        # for node in np.where(zero_nodes == 1)[0]:
        #     for ind in inc.incidence.T[node].nonzero()[1]:
        #         inc.incidence[ind, node] = 0
        #     for ind in inc.inlet.T[node].nonzero()[1]:
        #         inc.inlet[ind, node] = 0
        #keep_largest_component(inc)
        #graph.in_vec = 1 * (np.abs(inc.incidence.T) @ edges.inlet == np.abs(inc.incidence.T) @ np.ones(sid.ne))
        #graph.out_vec = 1 * (np.abs(inc.incidence.T) @ edges.outlet == np.abs(inc.incidence.T) @ np.ones(sid.ne))
        print(f'in_vec: {np.sum(graph.in_vec)}, inlet: {np.sum(edges.inlet)}')
        print(f'out_vec: {np.sum(graph.out_vec)}, outlet: {np.sum(edges.outlet)}')
        # inc.incidence = inc.incidence.multiply(1 * (diams_new > 0)[:, np.newaxis])
        # inc.inlet = inc.inlet.multiply(1 * (diams_new > 0)[:, np.newaxis])
        # edges.inlet *= diams_new > 0
        # edges.outlet *= diams_new > 0
    if np.max(edges.outlet * edges.diams) > sid.d_break:
        breakthrough = True
        print ('Network dissolved.')
    if np.sum((vols.triangles @ (vols.vol_a == 0)) * edges.outlet) > sid.m / 2:
        breakthrough = True
        print ('Network dissolved.')
    # if sid.include_adt:
    #     diams_rate = np.abs((diams_new - edges.diams) / edges.diams)
    #     diams_rate = np.array(np.ma.fix_invalid(diams_rate, fill_value = 0))
    #     dt_next = sid.growth_rate / sid.dt / np.max(diams_rate)
    #     if dt_next > sid.dt_max:
    #         dt_next = sid.dt_max
    # if np.sum(edges.inlet) == 0:
    #     breakthrough = True
    #     print ('Network clogged.')
    edges.diams = diams_new
    #print(edges.diams)
    edges.diams_draw = diams_new * (diams_new > 0) + edges.diams_draw * (diams_new == 0)
    # if np.max(edges.diams / edges.diams_initial) > 300:
    #     breakthrough = True

    
    return breakthrough, dt_next


def update_diameters_d0_hindering(
    sid: SimInputData,
    inc: Incidence,
    edges: Edges,
    graph: Graph,
    vols: Volumes,
    cb: np.ndarray,
    cc: np.ndarray,
    cd: np.ndarray,
    data,
) -> tuple[bool, float]:
    """Update diameters with the D0-dependent precipitation growth law.

    This convenience wrapper keeps the legacy :func:`update_diameters`
    behavior unchanged while providing a drop-in counterpart for simulations
    that use ``solve_precipitation_safe_d0_hindering``.
    """
    return update_diameters(
        sid,
        inc,
        edges,
        graph,
        vols,
        cb,
        cc,
        cd,
        data,
        use_d0_hindering=True,
    )

def solve_d(sid: SimInputData, inc: Incidence, edges: Edges, cb: np.ndarray) \
    -> np.ndarray:
    """ Updates diameters in case of dissolution.

    Parameters
    -------
    sid : simInputData class object
        all config parameters of the simulation
        Da : float
        G : float
        dt : float

    inc : Incidence class object
        matrices of incidence
        incidence : scipy sparse csr matrix (ne x nsq)

    edges : Edges class object
        all edges in network and their parameters
        diams : numpy ndarray (ne)
        lens : numpy ndarray (ne)
        flow : numpy ndarray (ne)

    cb : numpy ndarray (nsq)
        vector of substance B concentration

    Returns
    -------
    change : numpy ndarray (ne)
        change of diameter of each edge
    """
    # create list of concentrations which should be used for growth of each
    # edge (upstream one)
    cb_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ cb
    change = cb_in * np.abs(edges.flow) / (sid.Da * edges.lens \
        * edges.diams) * (1 - np.exp(-sid.Da / (1 + sid.G * edges.diams) \
        * edges.diams * edges.lens / np.abs(edges.flow)))
    change = np.array(np.ma.fix_invalid(change, fill_value = 0))
    return change

def solve_d_diff(sid: SimInputData, inc: Incidence, edges: Edges, cb: np.ndarray) \
    -> np.ndarray:
    """ Updates diameters in case of dissolution.

    Parameters
    -------
    sid : simInputData class object
        all config parameters of the simulation
        Da : float
        G : float
        dt : float

    inc : Incidence class object
        matrices of incidence
        incidence : scipy sparse csr matrix (ne x nsq)

    edges : Edges class object
        all edges in network and their parameters
        diams : numpy ndarray (ne)
        lens : numpy ndarray (ne)
        flow : numpy ndarray (ne)

    cb : numpy ndarray (nsq)
        vector of substance B concentration

    Returns
    -------
    change : numpy ndarray (ne)
        change of diameter of each edge
    """
    # create list of concentrations which should be used for growth of each
    # edge (upstream one)
    lam_plus_val = sid.Pe / 2 / edges.diams ** 2 * \
        (np.sqrt(np.abs(edges.flow) ** 2 + 4 * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) + np.abs(edges.flow))
    lam_plus_val = np.array(np.ma.fix_invalid(lam_plus_val, fill_value = 0))
    lam_minus_val = sid.Pe / 2 / edges.diams ** 2 * \
        (np.sqrt(np.abs(edges.flow) ** 2 + 4 * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) - np.abs(edges.flow))
    lam_minus_val = np.array(np.ma.fix_invalid(lam_minus_val, fill_value = 0))
    # change = np.abs(edges.flow) / (sid.Da * edges.lens \
    #       * edges.diams) * (edges.A * (np.exp(lam_plus_val) - 1) / lam_plus_val + edges.B * (1 - np.exp(-lam_minus_val)) / lam_minus_val)
    
    #change = 2 / (1 + sid.G * edges.diams) * (edges.A * (np.exp(lam_plus_val) - 1) / lam_plus_val + edges.B * (1 - np.exp(-lam_minus_val)) / lam_minus_val)
    change = 1 / (sid.Da * edges.lens * edges.diams) * (edges.A * (np.exp(lam_plus_val * edges.lens) - 1) * (np.sqrt(np.abs(edges.flow) ** 2 + 4 * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) - np.abs(edges.flow)) + edges.B * (1 - np.exp(-lam_minus_val * edges.lens)) * (np.sqrt(np.abs(edges.flow) ** 2 + 4 * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) + np.abs(edges.flow)))

    # change = cb_in * np.abs(edges.flow) / (sid.Da * edges.lens \
    #     * edges.diams) * (1 - np.exp(-sid.Da / (1 + sid.G * edges.diams) \
    #     * edges.diams * edges.lens / np.abs(edges.flow)))
    change = np.array(np.ma.fix_invalid(change, fill_value = 0))
    return change

def solve_d_diff_pe_fix(sid: SimInputData, inc: Incidence, edges: Edges, cb: np.ndarray) \
    -> np.ndarray:
    """ Updates diameters in case of dissolution.

    Parameters
    -------
    sid : simInputData class object
        all config parameters of the simulation
        Da : float
        G : float
        dt : float

    inc : Incidence class object
        matrices of incidence
        incidence : scipy sparse csr matrix (ne x nsq)

    edges : Edges class object
        all edges in network and their parameters
        diams : numpy ndarray (ne)
        lens : numpy ndarray (ne)
        flow : numpy ndarray (ne)

    cb : numpy ndarray (nsq)
        vector of substance B concentration

    Returns
    -------
    change : numpy ndarray (ne)
        change of diameter of each edge
    """
    # create list of concentrations which should be used for growth of each
    # edge (upstream one)
    lam_plus_val = sid.Pe / 2 * edges.lens / edges.diams ** 2 * \
        (np.sqrt(np.abs(edges.flow) ** 2 + 4 * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) + np.abs(edges.flow))
    lam_plus_val = np.array(np.ma.fix_invalid(lam_plus_val, fill_value = 0))
    lam_plus_zero = 1 * (lam_plus_val > sid.diffusion_exp_limit)
    lam_plus_val = lam_plus_val * (1 - lam_plus_zero)  
    lam_minus_val = sid.Pe / 2 * edges.lens / edges.diams ** 2 * \
        (np.sqrt(np.abs(edges.flow) ** 2 + 4 * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) - np.abs(edges.flow))
    lam_minus_val = np.array(np.ma.fix_invalid(lam_minus_val, fill_value = 0))
    # change = np.abs(edges.flow) / (sid.Da * edges.lens \
    #       * edges.diams) * (edges.A * (np.exp(lam_plus_val) - 1) / lam_plus_val + edges.B * (1 - np.exp(-lam_minus_val)) / lam_minus_val)
    
    #change = 2 / (1 + sid.G * edges.diams) * (edges.A * (np.exp(lam_plus_val) - 1) / lam_plus_val + edges.B * (1 - np.exp(-lam_minus_val)) / lam_minus_val)
    #change = 1 / (sid.Da * edges.lens * edges.diams) * (edges.A * (np.exp(lam_plus_val) - 1) * (np.sqrt(np.abs(edges.flow) ** 2 + 4 * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) - np.abs(edges.flow)) + edges.B * (1 - np.exp(-lam_minus_val)) * (np.sqrt(np.abs(edges.flow) ** 2 + 4 * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) + np.abs(edges.flow)))
    #change = 1 / (sid.Da * edges.lens * edges.diams) * (edges.A * (np.exp(lam_plus_val) - 1) * (np.sqrt(np.abs(edges.flow) ** 2 + 4 * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) - np.abs(edges.flow)) + edges.B * (1 - np.exp(-lam_minus_val)) * (np.sqrt(np.abs(edges.flow) ** 2 + 4 * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) + np.abs(edges.flow)))
    change = (1 - lam_plus_zero) / (sid.Da * edges.lens * edges.diams) * (edges.A * (np.exp(lam_plus_val) - 1) * (np.sqrt(np.abs(edges.flow) ** 2 + 4 * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) - np.abs(edges.flow)) + edges.B * (1 - np.exp(-lam_minus_val)) * (np.sqrt(np.abs(edges.flow) ** 2 + 4 * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) + np.abs(edges.flow))) + lam_plus_zero * edges.B * np.abs(edges.flow) / (sid.Da * edges.lens * edges.diams) * (1 - np.exp(-sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow)))

    # change = cb_in * np.abs(edges.flow) / (sid.Da * edges.lens \
    #     * edges.diams) * (1 - np.exp(-sid.Da / (1 + sid.G * edges.diams) \
    #     * edges.diams * edges.lens / np.abs(edges.flow)))
    change = np.array(np.ma.fix_invalid(change, fill_value = 0))
    return change

def solve_d_diff_vol(sid: SimInputData, inc: Incidence, edges: Edges, vols: Volumes, cb: np.ndarray) \
    -> np.ndarray:
    """ Updates diameters in case of dissolution.

    Parameters
    -------
    sid : simInputData class object
        all config parameters of the simulation
        Da : float
        G : float
        dt : float

    inc : Incidence class object
        matrices of incidence
        incidence : scipy sparse csr matrix (ne x nsq)

    edges : Edges class object
        all edges in network and their parameters
        diams : numpy ndarray (ne)
        lens : numpy ndarray (ne)
        flow : numpy ndarray (ne)

    cb : numpy ndarray (nsq)
        vector of substance B concentration

    Returns
    -------
    change : numpy ndarray (ne)
        change of diameter of each edge
    """
    # create list of concentrations which should be used for growth of each
    # edge (upstream one)
    lam_plus_val = sid.Pe / 2 / edges.diams ** 2 * \
        (np.sqrt(np.abs(edges.flow) ** 2 + 4 * edges.alpha * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) + np.abs(edges.flow))
    lam_plus_val = np.array(np.ma.fix_invalid(lam_plus_val, fill_value = 0))
    lam_plus_zero = 1 * (lam_plus_val > sid.diffusion_exp_limit)
    lam_plus_val = lam_plus_val * (1 - lam_plus_zero)  
    lam_minus_val = sid.Pe / 2 / edges.diams ** 2 * \
        (np.sqrt(np.abs(edges.flow) ** 2 + 4 * edges.alpha * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) - np.abs(edges.flow))
    lam_minus_val = np.array(np.ma.fix_invalid(lam_minus_val, fill_value = 0))
    # change = np.abs(edges.flow) / (sid.Da * edges.lens \
    #       * edges.diams) * (edges.A * (np.exp(lam_plus_val) - 1) / lam_plus_val + edges.B * (1 - np.exp(-lam_minus_val)) / lam_minus_val)
    
    #change = 2 / (1 + sid.G * edges.diams) * (edges.A * (np.exp(lam_plus_val) - 1) / lam_plus_val + edges.B * (1 - np.exp(-lam_minus_val)) / lam_minus_val)
    #change = 1 / (sid.Da * edges.lens * edges.diams) * (edges.A * (np.exp(lam_plus_val) - 1) * (np.sqrt(np.abs(edges.flow) ** 2 + 4 * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) - np.abs(edges.flow)) + edges.B * (1 - np.exp(-lam_minus_val)) * (np.sqrt(np.abs(edges.flow) ** 2 + 4 * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) + np.abs(edges.flow)))
    #change = 1 / (sid.Da * edges.lens * edges.diams) * (edges.A * (np.exp(lam_plus_val) - 1) * (np.sqrt(np.abs(edges.flow) ** 2 + 4 * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) - np.abs(edges.flow)) + edges.B * (1 - np.exp(-lam_minus_val)) * (np.sqrt(np.abs(edges.flow) ** 2 + 4 * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) + np.abs(edges.flow)))
    
    #change = (1 - lam_plus_zero) / (sid.Da * edges.lens * edges.diams) * (edges.A * (np.exp(lam_plus_val * edges.lens) - 1) * (np.sqrt(np.abs(edges.flow) ** 2 + 4 * edges.alpha * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) - np.abs(edges.flow)) + edges.B * (1 - np.exp(-lam_minus_val * edges.lens)) * (np.sqrt(np.abs(edges.flow) ** 2 + 4 * edges.alpha * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) + np.abs(edges.flow))) + lam_plus_zero * edges.B * np.abs(edges.flow) / (sid.Da * edges.lens * edges.diams) * (1 - np.exp(-edges.alpha * sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow)))
    
    #change = (1 - lam_plus_zero) / sid.Da * (edges.A * (np.exp(lam_plus_val * edges.lens) - 1) * (np.sqrt(np.abs(edges.flow) ** 2 + 4 * edges.alpha * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) - np.abs(edges.flow)) + edges.B * (1 - np.exp(-lam_minus_val * edges.lens)) * (np.sqrt(np.abs(edges.flow) ** 2 + 4 * edges.alpha * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) + np.abs(edges.flow))) + lam_plus_zero * 2 * edges.B * np.abs(edges.flow) / sid.Da * (1 - np.exp(-edges.alpha * sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow)))
    change_pe_fix = lam_plus_zero * 2 * edges.B * np.abs(edges.flow) / sid.Da * (1 - np.exp(-edges.alpha * sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow)))
    change_pe_fix = np.array(np.ma.fix_invalid(change_pe_fix, fill_value = 0))
    #zero_flow_fix = 1 * (np.abs(edges.flow) == 0)
    #lam_zero_flow = np.sqrt(edges.alpha * sid.Da * sid.Pe / ((1 + sid.G * edges.diams) * edges.diams))
    #change_zero_flow_fix = zero_flow_fix * 2 * edges.diams / lam_zero_flow * (edges.A * (np.exp(lam_zero_flow * edges.lens) - 1) + edges.B * (1 - np.exp(-lam_zero_flow * edges.lens)))
    change = (1 - lam_plus_zero) * 2  * edges.diams ** 2 / (sid.Pe * sid.Da) * (edges.A * (np.exp(lam_plus_val * edges.lens) - 1) * lam_minus_val + edges.B * (1 - np.exp(-lam_minus_val * edges.lens)) * lam_plus_val) + change_pe_fix #+ change_zero_flow_fix

    change = np.array(np.ma.fix_invalid(change, fill_value = 0))

    return change


def solve_dp(sid: SimInputData, inc: Incidence, edges: Edges, cb: np.ndarray, \
    cc: np.ndarray) -> np.ndarray:
    """ Updates diameters in case of dissolution + precipitation.

    Parameters
    -------
    sid : simInputData class object
        all config parameters of the simulation
        Da : float
        G : float
        K : float
        Gamma : float
        at

    inc : Incidence class object
        matrices of incidence
        incidence : scipy sparse csr matrix (ne x nsq)

    edges : Edges class object
        all edges in network and their parameters
        diams : numpy ndarray (ne)
        lens : numpy ndarray (ne)
        flow : numpy ndarray (ne)
        alpha_b : numpy ndarray (ne)

    cb : numpy ndarray (nsq)
        vector of substance B concentration

    cc : numpy ndarray (nsq)
        vector of substance C concentration

    Returns
    -------
    change : numpy ndarray (ne)
        change of diameter of each edge
    """
    # create list of concentrations which should be used for
    # growth/shrink of each edge (upstream one)
    growth_matrix = np.abs((spr.diags(edges.flow) @ inc.incidence > 0))
    cb_in = growth_matrix @ cb
    cc_in = growth_matrix @ cc
    growth = cb_in * np.abs(edges.flow)  / (sid.Da * edges.lens * edges.diams) \
        * (1 - np.exp(-sid.Da / (1 + sid.G * edges.diams) * edges.diams \
        * edges.lens / np.abs(edges.flow)))
    growth = np.array(np.ma.fix_invalid(growth, fill_value = 0))
    shrink_cb = cb_in * np.abs(edges.flow) * sid.Gamma / (sid.Da * edges.lens \
        * edges.diams) / (sid.K - 1) * (sid.K * (1 - \
        np.exp(-sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens \
        / np.abs(edges.flow))) - (1 - np.exp(-sid.Da * sid.K / (1 + sid.G \
        * sid.K * edges.diams) * edges.diams * edges.lens \
        / np.abs(edges.flow))))
    shrink_cb = np.array(np.ma.fix_invalid(shrink_cb, fill_value = 0))
    shrink_cc = cc_in * np.abs(edges.flow) * sid.Gamma / (sid.Da * edges.lens \
        * edges.diams) * (1 - np.exp(-sid.Da * sid.K / (1 + sid.G \
        * sid.K * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow)))
    shrink_cc = np.array(np.ma.fix_invalid(shrink_cc, fill_value = 0))
    change = (growth - shrink_cb - shrink_cc)
    return change

def solve_d_vol(sid, inc, edges, vols, cb):
    growth_matrix = np.abs((spr.diags(edges.flow) @ inc.incidence > 0))
    cb_growth = growth_matrix @ cb # choose upstream concentration of B for the
    # calculation of growth
    exp_b = np.exp(-np.abs(sid.Da / (1 + sid.G * edges.diams) * \
                edges.diams * edges.lens / edges.flow))
    dissolve = cb_growth * np.abs(edges.flow) / sid.Da * (1 - exp_b)
    dissolve = np.array(np.ma.fix_invalid(dissolve, fill_value = 0.))
    exp_b2 = np.exp(-np.abs(edges.alpha_b * sid.Da / (1 + sid.G * edges.diams) * \
                edges.diams * edges.lens / edges.flow))
    change = cb_growth * np.abs(edges.flow) / sid.Da * (1 - exp_b2)
    change = np.array(np.ma.fix_invalid(change, fill_value = 0.))
    return change, dissolve, np.zeros_like(dissolve)

def solve_dp_vol(sid: SimInputData, inc: Incidence, edges: Edges, vols: Volumes,
                 cb: np.ndarray, cc: np.ndarray, cd: np.ndarray) -> np.ndarray:
    """ Updates diameters in case of dissolution + precipitation.

    Parameters
    -------
    sid : simInputData class object
        all config parameters of the simulation

    inc : Incidence class object
        matrices of incidence

    edges : Edges class object
        all edges in network and their parameters

    vols : Volumes class object
        triangle volumes (vol_a, vol_e, vol_max) — used for dissolution volume tracking

    cb, cc, cd : numpy ndarray (nsq)
        node concentrations of B, C, D

    Returns
    -------
    change : numpy ndarray (ne)
        signed change of d²L per edge (positive = dissolution, negative = precipitation)

    dissolve : numpy ndarray (ne)
        alpha_b-scaled dissolution volume rate per edge

    precipitate : numpy ndarray (ne)
        alpha_c-scaled precipitation volume rate per edge
    """
    # edges.alpha_c is set by solve_precipitation_safe before this call —
    # use it directly so the growth step is consistent with the concentration solver.
    growth_matrix = np.abs((spr.diags(edges.flow) @ inc.incidence > 0))
    cb_in = growth_matrix @ cb
    cc_in = growth_matrix @ cc
    cd_in = growth_matrix @ cd

    X_rate = cd_in * sid.K / (1 + sid.K * sid.G * edges.diams)
    Y_rate = sid.Kp / (1 + sid.G * edges.diams)
    ksi = edges.alpha_c * X_rate - edges.alpha_b * Y_rate

    exp_p2 = np.exp(-edges.alpha_c * sid.Da * sid.K / (1 + sid.G * sid.K * edges.diams) \
        * cd_in / sid.Kp * edges.diams * edges.lens / np.abs(edges.flow))
    exp_p2 = np.array(np.ma.fix_invalid(exp_p2, fill_value = 0))
    exp_d2 = np.exp(-edges.alpha_b * sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow))
    exp_d2 = np.array(np.ma.fix_invalid(exp_d2, fill_value = 0))

    # alpha_b/alpha_c-scaled: used for diameter change and volume tracking
    growth2   = cb_in * np.abs(edges.flow) / sid.Da * (1 - exp_d2)
    shrink_cc2 = cc_in * np.abs(edges.flow) * sid.Gamma / sid.Da * (1 - exp_p2)
    # Cross-term: D consumed via B→C channel; alpha_c enters both numerator and ksi so
    # the formula matches the NR's particular-solution constant (alpha_c·k·cd - B_pref).
    shrink_cb2 = cb_in * np.abs(edges.flow) * sid.Gamma / sid.Da \
        * (edges.alpha_c * X_rate * (1 - exp_d2) - edges.alpha_b * Y_rate * (1 - exp_p2)) / ksi
    shrink_cc2 = np.array(np.ma.fix_invalid(shrink_cc2, fill_value=0))
    shrink_cb2 = np.array(np.ma.fix_invalid(shrink_cb2, fill_value=0))

    change = growth2 - shrink_cc2 - shrink_cb2
    # Return alpha-scaled volumes consistent with change (no abs needed — both terms ≥ 0).
    return change, growth2, shrink_cc2 + shrink_cb2


def solve_dp_vol_d0_hindering(
    sid: SimInputData,
    inc: Incidence,
    edges: Edges,
    vols: Volumes,
    cb: np.ndarray,
    cc: np.ndarray,
    cd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Update geometry using D0-dependent precipitation hindering.

    This is the growth counterpart of
    ``solve_precipitation_nr9_vxx_d0_hindering`` and
    ``solve_precipitation_safe_d0_hindering``.  The upstream concentration of
    species D is treated as constant on each edge and enters both the
    pseudo-first-order precipitation rate and its transverse mass-transfer
    correction,

    .. math::

        P(D_0) = \frac{\mathrm{Da}\,K\,d\,(D_0/K_p)}
                      {1 + G K d (D_0/K_p)}.

    The edge outlet concentrations are reconstructed with the same analytical
    solution and the same finite-D cap as the nonlinear transport solver.  The
    precipitated solid-volume rate is then obtained from the stoichiometric
    consumption of D,

    .. math::

        \dot V_E = |q|\,\Delta D\,\Gamma/\mathrm{Da}.

    Parameters are the same as for :func:`solve_dp_vol`.  ``vols`` is retained
    in the signature for API compatibility; the availability factors already
    stored in ``edges.alpha_b`` and ``edges.alpha_c`` are used directly.

    Returns
    -------
    change : ndarray
        Signed change rate of ``d^2 L`` on every edge.  Positive values denote
        net dissolution and negative values net precipitation.
    dissolve : ndarray
        Dissolved-primary solid-volume rate on every edge.
    precipitate : ndarray
        Precipitated-secondary solid-volume rate on every edge, including the
        molar-volume factor ``Gamma``.
    """
    del vols  # availability is represented by edges.alpha_b/alpha_c here

    if sid.Kp <= 0:
        raise ValueError("sid.Kp must be positive")

    da = float(sid.Da)
    if abs(da) <= 1e-30:
        zeros = np.zeros_like(np.asarray(edges.flow, dtype=float))
        return zeros.copy(), zeros.copy(), zeros.copy()

    d = np.asarray(edges.diams, dtype=float)
    length = np.asarray(edges.lens, dtype=float)
    abs_q = np.abs(np.asarray(edges.flow, dtype=float))
    active = abs_q > 1e-12
    q_safe = np.where(active, abs_q, 1.0)

    # One upstream node per flowing edge.  This is identical to the selector
    # used by the concentration solver and the safe availability wrapper.
    upstream = (
        spr.diags(edges.flow) @ inc.incidence > 0
    ).astype(float).tocsr()
    cb_in = np.asarray(upstream @ cb, dtype=float).ravel()
    cc_in = np.asarray(upstream @ cc, dtype=float).ravel()
    cd_in = np.asarray(upstream @ cd, dtype=float).ravel()
    cd_rate = np.maximum(cd_in, 0.0)

    residence = length / q_safe

    def safe_exp(exponent: np.ndarray) -> np.ndarray:
        return np.exp(np.clip(exponent, -700.0, 700.0))

    # Alpha_b-scaled dissolution, consistent with the B transport equation.
    b_rate = edges.alpha_b * da * d / (1.0 + sid.G * d)
    exp_b = safe_exp(-np.abs(b_rate * residence))
    cb_out = cb_in * exp_b

    # Concentration-aware precipitation coefficient and alpha_c-scaled sink.
    d0_over_kp = cd_rate / sid.Kp
    hindering = 1.0 + sid.G * sid.K * d * d0_over_kp
    p_rate = da * sid.K * d * d0_over_kp / hindering
    sink_rate = np.asarray(edges.alpha_c, dtype=float) * p_rate
    exp_p = safe_exp(-np.abs(sink_rate * residence))

    # Stable evaluation of
    #   C1 = C0 exp(-s x) + A [exp(-r x)-exp(-s x)]/(s-r),
    # including the finite limit when s -> r.  This mirrors the implementation
    # in the D0-dependent Newton solver and safe wrapper.
    denominator = sink_rate - b_rate
    y = denominator * residence
    near_resonance = np.abs(y) < 1e-6

    quotient = np.empty_like(denominator)
    regular = ~near_resonance
    quotient[regular] = (
        exp_b[regular] - exp_p[regular]
    ) / denominator[regular]

    yr = y[near_resonance]
    xr = residence[near_resonance]
    er = exp_b[near_resonance]
    quotient[near_resonance] = er * xr * (
        1.0
        - 0.5 * yr
        + yr**2 / 6.0
        - yr**3 / 24.0
        + yr**4 / 120.0
    )

    source_amplitude = b_rate * cb_in
    cc_out = cc_in * exp_p + source_amplitude * quotient

    cb_out = np.where(np.isfinite(cb_out), cb_out, cb_in)
    cc_out = np.where(np.isfinite(cc_out), cc_out, cc_in)

    # Stoichiometric D consumption.  The cap is the same one used in the
    # nonlinear edge solver when the unconstrained solution would give D1 < 0.
    d_consumed = (cb_in - cb_out) + (cc_in - cc_out)
    d_consumed = np.minimum(np.maximum(d_consumed, 0.0), cd_rate)

    dissolve = abs_q * np.maximum(cb_in - cb_out, 0.0) / da
    precipitate = abs_q * d_consumed * sid.Gamma / da

    dissolve[~active] = 0.0
    precipitate[~active] = 0.0

    dissolve = np.asarray(
        np.ma.fix_invalid(dissolve, fill_value=0.0), dtype=float
    )
    precipitate = np.asarray(
        np.ma.fix_invalid(precipitate, fill_value=0.0), dtype=float
    )
    change = dissolve - precipitate

    return change, dissolve, precipitate

def solve_dp_kp(sid: SimInputData, inc: Incidence, edges: Edges, cb: np.ndarray, \
    cc: np.ndarray, cd: np.ndarray) -> np.ndarray:
    """ Updates diameters in case of dissolution + precipitation.

    Parameters
    -------
    sid : simInputData class object
        all config parameters of the simulation
        Da : float
        G : float
        K : float
        Gamma : float
        at

    inc : Incidence class object
        matrices of incidence
        incidence : scipy sparse csr matrix (ne x nsq)

    edges : Edges class object
        all edges in network and their parameters
        diams : numpy ndarray (ne)
        lens : numpy ndarray (ne)
        flow : numpy ndarray (ne)
        alpha_b : numpy ndarray (ne)

    cb : numpy ndarray (nsq)
        vector of substance B concentration

    cc : numpy ndarray (nsq)
        vector of substance C concentration

    Returns
    -------
    change : numpy ndarray (ne)
        change of diameter of each edge
    """
    # create list of concentrations which should be used for
    # growth/shrink of each edge (upstream one)
    growth_matrix = np.abs((spr.diags(edges.flow) @ inc.incidence > 0))
    cb_in = growth_matrix @ cb
    cc_in = growth_matrix @ cc
    cd_in = growth_matrix @ cd
    # upstream saturation ratio minus 1
    epsS = 1e-3  # same order you used in Newton
    Sminus1 = (cc_in * cd_in / sid.Kp) - 1.0
    H = 0.5 * (1.0 + Sminus1 / np.sqrt(Sminus1*Sminus1 + epsS*epsS))  # ~0 undersat, ~1 supersat

    # safest: gate precipitation as a whole
    H = np.array(np.ma.fix_invalid(H, fill_value=0.0))
    H = np.clip(H, 0.0, 1.0)

    # use effective D for precipitation kinetics
    cd_eff = cd_in * H
    
    ksi = cd_eff * sid.K / (1 + sid.K * sid.G * edges.diams) - sid.Kp / (1 + sid.G * edges.diams)
    exp_p  = np.exp(-sid.Da * sid.K / (1 + sid.G * sid.K * edges.diams)
                * (cd_eff / sid.Kp) * edges.diams * edges.lens / np.abs(edges.flow))
    exp_p = np.array(np.ma.fix_invalid(exp_p, fill_value = 0))
    exp_d = np.exp(-sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow))
    exp_d = np.array(np.ma.fix_invalid(exp_d, fill_value = 0))
    exp_p2 = np.exp(-edges.alpha_c * sid.Da * sid.K / (1 + sid.G * sid.K * edges.diams)
                * (cd_eff / sid.Kp) * edges.diams * edges.lens / np.abs(edges.flow))
    exp_p2 = np.array(np.ma.fix_invalid(exp_p2, fill_value = 0))
    exp_d2 = np.exp(-edges.alpha_b * sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow))
    exp_d2 = np.array(np.ma.fix_invalid(exp_d2, fill_value = 0))        
    # growth = cb_in * np.abs(edges.flow)  / (sid.Da * edges.lens * edges.diams) \
    #     * (1 - np.exp(-sid.Da / (1 + sid.G * edges.diams) * edges.diams \
    #     * edges.lens / np.abs(edges.flow)))
    # growth = np.array(np.ma.fix_invalid(growth, fill_value = 0))
    # shrink_cc = cc_in * sid.Kp / cd_in * np.abs(edges.flow)  / (sid.Da * edges.lens \
    #     * edges.diams * sid.Gamma) * (1 - np.exp(-sid.Da * sid.K / (1 + sid.G \
    #     * sid.K * edges.diams) * cd_in / sid.Kp * edges.diams * edges.lens / np.abs(edges.flow)))
    growth = cb_in * np.abs(edges.flow)  / sid.Da * (1 - exp_d)
    shrink_cc = cc_in * np.abs(edges.flow) * sid.Gamma / sid.Da * (1 - exp_p)
    # shrink_cc = cc_in * sid.Kp / cd_in * np.abs(edges.flow)  / (sid.Da * edges.lens \
     #    * edges.diams * sid.Gamma) * (1 - exp_p)
    shrink_cb = edges.alpha_b * cb_in * cd_in * sid.K * np.abs(edges.flow) * sid.Gamma / sid.Da \
        * ((1 - exp_d) / (1 + sid.G * edges.diams * sid.K) - (1 - exp_p) * sid.Kp / (sid.K * cd_in * (1 + sid.G * edges.diams))) / ksi
    shrink_cc = np.array(np.ma.fix_invalid(shrink_cc, fill_value = 0)) 
    shrink_cb = np.array(np.ma.fix_invalid(shrink_cb, fill_value = 0)) 
    growth2 = cb_in * np.abs(edges.flow)  / sid.Da * (1 - exp_d2)
    shrink_cc2 = cc_in * np.abs(edges.flow) * sid.Gamma / sid.Da * (1 - exp_p2)
    # shrink_cc = cc_in * sid.Kp / cd_in * np.abs(edges.flow)  / (sid.Da * edges.lens \
     #    * edges.diams * sid.Gamma) * (1 - exp_p)
    shrink_cb2 = edges.alpha_b * cb_in * cd_in * sid.K * np.abs(edges.flow) * sid.Gamma / sid.Da \
        * ((1 - exp_d2) / (1 + sid.G * edges.diams * sid.K) - (1 - exp_p2) * sid.Kp / (sid.K * cd_in * (1 + sid.G * edges.diams))) / ksi
    shrink_cc2 = np.array(np.ma.fix_invalid(shrink_cc2, fill_value = 0)) 
    shrink_cb2 = np.array(np.ma.fix_invalid(shrink_cb2, fill_value = 0)) 
    shrink_cc  *= H
    shrink_cb  *= H
    shrink_cc2 *= H
    shrink_cb2 *= H
    change = growth2 - np.abs(shrink_cc2) - np.abs(shrink_cb2)
    #print(np.sum(inc.incidence @ (cc - cd + cb)))
    return change, growth, np.abs(shrink_cc) + np.abs(shrink_cb)