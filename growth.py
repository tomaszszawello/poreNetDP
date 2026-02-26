""" Updates edges diameters based on dissolution and precipitation.

This module calculates the change od diameters in the network, resulting from
dissolution (and precipitation, if enabled). Based on that change, new
timestep is calculated.

Notable functions
-------
update_diameters(SimInputData, Incidence, Edges, np.ndarray, np.ndarray) \
    -> tuple[bool, float]
    update diameters, calculate timestep and check if network is dissolved
"""

import numpy as np
import scipy.sparse as spr

from config import SimInputData
from network import Edges, Graph
from incidence import Incidence
from volumes import Volumes

from utils import keep_largest_component

def update_diameters(sid: SimInputData, inc: Incidence, edges: Edges, graph: Graph, \
    vols: Volumes, cb: np.ndarray, cc: np.ndarray, cd: np.ndarray, data) -> tuple[bool, float]:
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
    if sid.include_precipitation:
        if sid.include_volumes:
            change, dissolve, precipitate = solve_dp_vol(sid, inc, edges, cb, cc, cd)            
        else:
            change = solve_dp(sid, inc, edges, cb, cc)
    else:
        if sid.include_diffusion:
            if sid.include_volumes:
                change = solve_d_diff_vol(sid, inc, edges, vols, cb)
            else:
                change = solve_d_diff_pe_fix(sid, inc, edges, cb)
            #change = solve_d(sid, inc, edges, cb)
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

def solve_dp_vol(sid: SimInputData, inc: Incidence, edges: Edges, cb: np.ndarray, \
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
    ksi = cd_in * sid.K / (1 + sid.K * sid.G * edges.diams) - sid.Kp / (1 + sid.G * edges.diams)
    exp_p = np.exp(-sid.Da * sid.K / (1 + sid.G * sid.K * edges.diams) \
        * cd_in / sid.Kp * edges.diams * edges.lens / np.abs(edges.flow))
    exp_p = np.array(np.ma.fix_invalid(exp_p, fill_value = 0))
    exp_d = np.exp(-sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow))
    exp_d = np.array(np.ma.fix_invalid(exp_d, fill_value = 0))
    exp_p2 = np.exp(-edges.alpha_c * sid.Da * sid.K / (1 + sid.G * sid.K * edges.diams) \
        * cd_in / sid.Kp * edges.diams * edges.lens / np.abs(edges.flow))
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
    change = growth2 - np.abs(shrink_cc2) - np.abs(shrink_cb2)
    #print(np.sum(inc.incidence @ (cc - cd + cb)))
    return change, growth, np.abs(shrink_cc) + np.abs(shrink_cb)

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