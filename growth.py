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
from network import Graph, Edges
from incidence import Incidence
from volumes import Volumes


def update_diameters(sid: SimInputData, inc: Incidence, graph: Graph, edges: Edges, \
    vols: Volumes, cb: np.ndarray, cc: np.ndarray) -> tuple[bool, float]:
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
    if sid.include_cc:
        change = solve_dp(sid, inc, edges, cb, cc)
    else:
        if sid.include_diffusion:
            if sid.include_volumes:
                change, change2 = solve_d_diff_vol(sid, inc, edges, vols, cb)
            else:
                #change = solve_d_diff_pe_fix(sid, inc, edges, cb)
                change = solve_d_diff(sid, inc, edges, cb)
            #change = solve_d(sid, inc, edges, cb)
        else:
            if sid.include_volumes:
                change = solve_d_vol(sid, inc, edges, vols, cb)
            else:
                change, change2 = solve_d(sid, inc, edges, cb)
    breakthrough = False
    if sid.include_adt:
        #change_rate = change / edges.diams
        change_rate = change / edges.diams
        change_rate = np.array(np.ma.fix_invalid(change_rate, fill_value = 0))
        if float(np.max(change_rate)) == 0:
            breakthrough = True
            print ('Network dissolved, no more change.')
            return breakthrough, 0
        dt_next = sid.growth_rate / float(np.max(change_rate))
        if dt_next > sid.dt_max:
            dt_next = sid.dt_max
    else:
        dt_next = sid.dt
    #edges.grain -= inc.boundary @ (change * edges.diams * edges.lens / 4)
    edges.grain -= inc.boundary @ (change * edges.diams * edges.lens / 4) * dt_next
    #print(np.sum(edges.grain <= 0))
    #edges.active += 1 * ((1 * (inc.center.T @ (edges.grain <= 0)) + 1 * (inc.boundary.T @ (edges.grain <= 0))) > 0) * (edges.active == 0)
    edges.active = 1 * ((1 * (np.abs(inc.incidence) @ graph.in_vec > 0) + 1 * (inc.center.T @ (edges.grain <= 0)) + 1 * (inc.boundary.T @ (edges.grain <= 0))) > 0)
    edges.alpha = 1 * (inc.boundary.T @ (edges.grain > 0)) / 2
    #print('grains: ', edges.grain)
    #print(edges.active, edges.alpha)
    edges.diams = edges.diams * (1 - edges.active) + sid.dmax * edges.active
    
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
    # change = cb_in * np.abs(edges.flow) / (sid.Da * edges.lens \
    #     * edges.diams) * (1 - np.exp(-sid.Da / (1 + sid.G * edges.diams) \
    #     * edges.diams * edges.lens / np.abs(edges.flow)))
    change = cb_in * np.abs(edges.flow) / sid.Da * (1 - np.exp(-sid.Da / (1 + sid.G * edges.diams) \
        * edges.diams * edges.lens / np.abs(edges.flow)))
    change = np.array(np.ma.fix_invalid(change, fill_value = 0))
    return change, change

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
    edges.alpha = 1 * (inc.boundary.T @ (edges.grain > 0)) / 2 * edges.active
    lam_plus_val = sid.Pe / 2 / edges.diams ** 2 * \
        (np.sqrt(np.abs(edges.flow) ** 2 + 4 * edges.alpha * sid.ksi * edges.diams ** 2) + np.abs(edges.flow))
    lam_plus_val = np.array(np.ma.fix_invalid(lam_plus_val, fill_value = 0))
    lam_minus_val = sid.Pe / 2 / edges.diams ** 2 * \
        (np.sqrt(np.abs(edges.flow) ** 2 + 4 * edges.alpha * sid.ksi * edges.diams ** 2) - np.abs(edges.flow))
    lam_minus_val = np.array(np.ma.fix_invalid(lam_minus_val, fill_value = 0))
    #change = 2 / (1 + sid.G * edges.diams) * (edges.A * (np.exp(lam_plus_val) - 1) / lam_plus_val + edges.B * (1 - np.exp(-lam_minus_val)) / lam_minus_val)
    lam_plus_zero = 1 * (1 * (lam_plus_val > sid.diffusion_exp_limit) + 1 * (edges.diams == 0) + 1 * (edges.flow == 0) > 0)
    #change = (1 - lam_plus_zero) * edges.alpha / (sid.ksi * edges.lens * edges.diams) * (edges.A * (np.exp(lam_plus_val * edges.lens) - 1) * lam_minus_val + edges.B * (1 - np.exp(-lam_minus_val * edges.lens)) * lam_plus_val)
    change_pe_fix = (lam_plus_val > sid.diffusion_exp_limit) * edges.alpha * edges.B * np.abs(edges.flow) / (sid.ksi * edges.diams * edges.lens) * (1 - np.exp(-edges.alpha / sid.ksi  * edges.lens / np.abs(edges.flow)))
    change = (1 - lam_plus_zero) * edges.alpha / (sid.ksi * edges.lens * edges.diams) * (edges.A * (np.exp(lam_plus_val * edges.lens) - 1) * lam_minus_val + edges.B * (1 - np.exp(-lam_minus_val * edges.lens)) * lam_plus_val) + change_pe_fix
    print(len(lam_plus_zero), np.sum(lam_plus_zero), np.sum(edges.alpha))
    print(np.sum(change > 0))
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
    edges.alpha = 1 * ((inc.boundary.T @ edges.grain) > 0)
    lam_plus_val = np.sqrt(edges.alpha * sid.Da2 * edges.lens ** 2 / edges.diams / (1 + sid.G * edges.diams)) * (edges.flow == 0)
    lam_plus_val = np.array(np.ma.fix_invalid(lam_plus_val, fill_value = 0))    
    lam_minus_val = np.sqrt(edges.alpha * sid.Da2 * edges.lens ** 2 / edges.diams / (1 + sid.G * edges.diams)) * (edges.flow == 0)
    lam_minus_val = np.array(np.ma.fix_invalid(lam_minus_val, fill_value = 0))
    change = edges.diams / (np.sqrt(sid.Da2) * edges.lens) * (edges.A * (np.exp(lam_plus_val) - 1)  + edges.B * (1 - np.exp(-lam_minus_val)))
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


    lam_plus_val = sid.Pe / 2 / edges.diams ** 2 * \
        (np.sqrt(np.abs(edges.flow) ** 2 + 4 * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) + np.abs(edges.flow))
    lam_plus_val = np.array(np.ma.fix_invalid(lam_plus_val, fill_value = 0))
    lam_plus_zero = 1 * (lam_plus_val > sid.diffusion_exp_limit)
    lam_plus_val = lam_plus_val * (1 - lam_plus_zero)  
    lam_minus_val = sid.Pe / 2 / edges.diams ** 2 * \
        (np.sqrt(np.abs(edges.flow) ** 2 + 4 * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) - np.abs(edges.flow))
    lam_minus_val = np.array(np.ma.fix_invalid(lam_minus_val, fill_value = 0))
    # change = np.abs(edges.flow) / (sid.Da * edges.lens \
    #       * edges.diams) * (edges.A * (np.exp(lam_plus_val) - 1) / lam_plus_val + edges.B * (1 - np.exp(-lam_minus_val)) / lam_minus_val)
    
    #change = 2 / (1 + sid.G * edges.diams) * (edges.A * (np.exp(lam_plus_val) - 1) / lam_plus_val + edges.B * (1 - np.exp(-lam_minus_val)) / lam_minus_val)
    #change = 1 / (sid.Da * edges.lens * edges.diams) * (edges.A * (np.exp(lam_plus_val) - 1) * (np.sqrt(np.abs(edges.flow) ** 2 + 4 * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) - np.abs(edges.flow)) + edges.B * (1 - np.exp(-lam_minus_val)) * (np.sqrt(np.abs(edges.flow) ** 2 + 4 * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) + np.abs(edges.flow)))
    #change = 1 / (sid.Da * edges.lens * edges.diams) * (edges.A * (np.exp(lam_plus_val) - 1) * (np.sqrt(np.abs(edges.flow) ** 2 + 4 * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) - np.abs(edges.flow)) + edges.B * (1 - np.exp(-lam_minus_val)) * (np.sqrt(np.abs(edges.flow) ** 2 + 4 * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) + np.abs(edges.flow)))
    
    #change = (1 - lam_plus_zero) / (sid.Da * edges.lens * edges.diams) * (edges.A * (np.exp(lam_plus_val * edges.lens) - 1) * (np.sqrt(np.abs(edges.flow) ** 2 + 4 * edges.alpha * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) - np.abs(edges.flow)) + edges.B * (1 - np.exp(-lam_minus_val * edges.lens)) * (np.sqrt(np.abs(edges.flow) ** 2 + 4 * edges.alpha * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) + np.abs(edges.flow))) + lam_plus_zero * edges.B * np.abs(edges.flow) / (sid.Da * edges.lens * edges.diams) * (1 - np.exp(-edges.alpha * sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow)))
    
    #change = (1 - lam_plus_zero) / sid.Da * (edges.A * (np.exp(lam_plus_val * edges.lens) - 1) * (np.sqrt(np.abs(edges.flow) ** 2 + 4 * edges.alpha * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) - np.abs(edges.flow)) + edges.B * (1 - np.exp(-lam_minus_val * edges.lens)) * (np.sqrt(np.abs(edges.flow) ** 2 + 4 * edges.alpha * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) + np.abs(edges.flow))) + lam_plus_zero * 2 * edges.B * np.abs(edges.flow) / sid.Da * (1 - np.exp(-edges.alpha * sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow)))
    change_pe_fix = lam_plus_zero * 2 * edges.B * np.abs(edges.flow) / sid.Da * (1 - np.exp(-sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow)))
    change_pe_fix = np.array(np.ma.fix_invalid(change_pe_fix, fill_value = 0))
    #zero_flow_fix = 1 * (np.abs(edges.flow) == 0)
    #lam_zero_flow = np.sqrt(edges.alpha * sid.Da * sid.Pe / ((1 + sid.G * edges.diams) * edges.diams))
    #change_zero_flow_fix = zero_flow_fix * 2 * edges.diams / lam_zero_flow * (edges.A * (np.exp(lam_zero_flow * edges.lens) - 1) + edges.B * (1 - np.exp(-lam_zero_flow * edges.lens)))
    change2 = (1 - lam_plus_zero) * 2  * edges.diams ** 2 / (sid.Pe * sid.Da) * (edges.A * (np.exp(lam_plus_val * edges.lens) - 1) * lam_minus_val + edges.B * (1 - np.exp(-lam_minus_val * edges.lens)) * lam_plus_val) + change_pe_fix #+ change_zero_flow_fix

    change2 = np.array(np.ma.fix_invalid(change2, fill_value = 0))


    return change, change2


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
    shrink_cb = cb_in * np.abs(edges.flow)  / (sid.Da * edges.lens \
        * edges.diams * sid.Gamma) / (sid.K - 1) * (sid.K * (1 - \
        np.exp(-sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens \
        / np.abs(edges.flow))) - (1 - np.exp(-sid.Da * sid.K / (1 + sid.G \
        * sid.K * edges.diams) * edges.diams * edges.lens \
        / np.abs(edges.flow))))
    shrink_cb = np.array(np.ma.fix_invalid(shrink_cb, fill_value = 0))
    shrink_cc = cc_in * np.abs(edges.flow)  / (sid.Da * edges.lens \
        * edges.diams * sid.Gamma) * (1 - np.exp(-sid.Da * sid.K / (1 + sid.G \
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
    growth = cb_growth * np.abs(edges.flow) / sid.Da * (1 - exp_b)
    growth = np.array(np.ma.fix_invalid(growth, fill_value = 0.))
    return growth