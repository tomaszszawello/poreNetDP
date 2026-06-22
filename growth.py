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
from data import Data
from network import Edges
from incidence import Incidence
from volumes import Volumes


def update_diameters(sid: SimInputData, inc: Incidence, edges: Edges, \
    vols: Volumes, data: Data, cb: np.ndarray, cc: np.ndarray, cd: np.ndarray) -> tuple[bool, float]:
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
    if sid.include_precipitation:
        if sid.include_volumes:
            change, dissolve, precipitate = solve_dp_vol(sid, inc, edges, cb, cc, cd)
        else:
            change = solve_dp(sid, inc, edges, cb, cc)
    else:
        if sid.include_diffusion:
            if sid.include_volumes:
                change, change2 = solve_d_diff_vol(sid, inc, edges, vols, cb)
            else:
                change = solve_d_diff(sid, inc, edges, cb)
        else:
            if sid.include_volumes:
                change, dissolve = solve_d_vol(sid, inc, edges, vols, cb)
            else:
                change = solve_d(sid, inc, edges, cb)
    state = False
    if sid.include_adt:
        if sid.include_volumes:
            change_rate = change / edges.diams ** 2 / edges.lens
        else:
            change_rate = change / edges.diams
        change_rate = np.array(np.ma.fix_invalid(change_rate, fill_value = 0))
        if float(np.max(change_rate)) == 0:
            state = True
            print ('No mineral left to dissolve')
            return state, 0
        dt_next = sid.growth_rate / float(np.max(change_rate))
        if dt_next > sid.dt_max:
            dt_next = sid.dt_max
    else:
        dt_next = sid.dt
    # update diameters - using exact formula with sqrt is more precise,
    # but slightly inconsistent with the dissolved volume calculation
    if sid.include_volumes:
        if sid.include_precipitation:
            vols.vol_a_prev = vols.vol_a.copy()
            change = change * sid.dt
            dissolve = dissolve * sid.dt
            precipitate = precipitate * sid.dt

            data.vol_dissolved += np.sum(dissolve)
            data.vol_precipitated += np.sum(precipitate)
            
            vol_a_dissolved = vols.triangles.T @ (dissolve / edges.triangles)
            vol_e_precipitated = vols.triangles.T @ (precipitate / edges.triangles)
            vol_a_dissolved = np.array(np.ma.fix_invalid(vol_a_dissolved, fill_value = 0))
            vol_e_precipitated = np.array(np.ma.fix_invalid(vol_e_precipitated, fill_value = 0))
            
            vols.vol_a = np.clip(vols.vol_a - np.abs(vol_a_dissolved), 0, None)
            
            vols.vol_e = np.clip(vols.vol_e + np.abs(vol_e_precipitated), 0, vols.vol_max - vols.vol_a)

            diams_new = edges.diams + change / edges.diams / edges.lens / 2

            diams_new = np.array(np.ma.fix_invalid(diams_new, fill_value = 0))

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
        else:
            #diams_new = edges.diams + change * sid.dt / edges.diams / edges.lens / 2
            diams_new = np.sqrt(edges.diams ** 2 + change * sid.dt / edges.lens)
            diams_new = np.array(np.ma.fix_invalid(diams_new, fill_value = 0))
            diams_new = diams_new * (diams_new >= sid.dmin) \
                + sid.dmin * (diams_new < sid.dmin)
            vols.vol_a_prev = vols.vol_a.copy()
            edge_vol = vols.triangles @ vols.vol_a
            triangles_w = vols.triangles @ spr.diags(vols.vol_a)
            vol_a_dissolved = triangles_w.T @ (change2 / edge_vol)
            vol_a_dissolved = np.array(np.ma.fix_invalid(vol_a_dissolved, fill_value = 0))
    else:
        #diams_new = edges.diams + change * sid.dt / edges.diams / edges.lens / 2
        diams_new = np.sqrt(edges.diams ** 2 + change * sid.dt / edges.lens)
        diams_new = np.array(np.ma.fix_invalid(diams_new, fill_value = 0))
        diams_new = diams_new * (diams_new >= sid.dmin) \
            + sid.dmin * (diams_new < sid.dmin)
    edges.diams = diams_new
    return state, dt_next

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
        change of volume of each edge [L^3 / T]
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
    lam_plus_val = sid.Pe / 2 * edges.lens / edges.diams ** 2 * \
        (np.sqrt(np.abs(edges.flow) ** 2 + 4 * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) + np.abs(edges.flow))
    lam_plus_val = np.array(np.ma.fix_invalid(lam_plus_val, fill_value = 0))
    lam_plus_zero = 1 * (lam_plus_val > sid.diffusion_exp_limit)
    lam_plus_val = lam_plus_val * (1 - lam_plus_zero)  
    lam_minus_val = sid.Pe / 2 * edges.lens / edges.diams ** 2 * \
        (np.sqrt(np.abs(edges.flow) ** 2 + 4 * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) - np.abs(edges.flow))
    lam_minus_val = np.array(np.ma.fix_invalid(lam_minus_val, fill_value = 0))
    change = (1 - lam_plus_zero) / (sid.Da * edges.lens * edges.diams) * (edges.A * (np.exp(lam_plus_val) - 1) * (np.sqrt(np.abs(edges.flow) ** 2 + 4 * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) - np.abs(edges.flow)) + edges.B * (1 - np.exp(-lam_minus_val)) * (np.sqrt(np.abs(edges.flow) ** 2 + 4 * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) + np.abs(edges.flow))) + lam_plus_zero * edges.B * np.abs(edges.flow) / (sid.Da * edges.lens * edges.diams) * (1 - np.exp(-sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow)))
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
        (np.sqrt(np.abs(edges.flow) ** 2 + 4 * edges.alpha_b * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) + np.abs(edges.flow))
    lam_plus_val = np.array(np.ma.fix_invalid(lam_plus_val, fill_value = 0))
    lam_plus_zero = 1 * (lam_plus_val > sid.diffusion_exp_limit)
    lam_plus_val = lam_plus_val * (1 - lam_plus_zero)  
    lam_minus_val = sid.Pe / 2 / edges.diams ** 2 * \
        (np.sqrt(np.abs(edges.flow) ** 2 + 4 * edges.alpha_b * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) - np.abs(edges.flow))
    lam_minus_val = np.array(np.ma.fix_invalid(lam_minus_val, fill_value = 0))
    change_pe_fix = lam_plus_zero * 2 * edges.B * np.abs(edges.flow) / sid.Da * (1 - np.exp(-edges.alpha_b * sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow)))
    change_pe_fix = np.array(np.ma.fix_invalid(change_pe_fix, fill_value = 0))
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
    change_pe_fix = lam_plus_zero * 2 * edges.B * np.abs(edges.flow) / sid.Da * (1 - np.exp(-sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow)))
    change_pe_fix = np.array(np.ma.fix_invalid(change_pe_fix, fill_value = 0))
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
        change of volume of each edge [L^3/T]
    """
    # create list of concentrations which should be used for
    # growth/shrink of each edge (upstream one)
    growth_matrix = np.abs((spr.diags(edges.flow) @ inc.incidence > 0))
    cb_in = growth_matrix @ cb
    cc_in = growth_matrix @ cc
    growth = cb_in * np.abs(edges.flow)  / (sid.Da) \
        * (1 - np.exp(-sid.Da / (1 + sid.G * edges.diams) * edges.diams \
        * edges.lens / np.abs(edges.flow)))
    growth = np.array(np.ma.fix_invalid(growth, fill_value = 0))

    K_pref = ((1 + sid.G * edges.diams) / (1 + sid.G * sid.K * edges.diams)) 
    shrink_cb = cb_in * np.abs(edges.flow)  / (sid.Da * sid.Gamma) \
            / (K_pref*sid.K - 1) * (K_pref*sid.K * (1 - \
        np.exp(-sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens \
        / np.abs(edges.flow))) - (1 - np.exp(-sid.Da * sid.K / (1 + sid.G \
        * sid.K * edges.diams) * edges.diams * edges.lens \
        / np.abs(edges.flow))))
    shrink_cb = np.array(np.ma.fix_invalid(shrink_cb, fill_value = 0))
    shrink_cc = cc_in * np.abs(edges.flow)  / (sid.Da * sid.Gamma) \
        * (1 - np.exp(-sid.Da * sid.K / (1 + sid.G \
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
    return change, dissolve

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
    growth = cb_in * np.abs(edges.flow)  / sid.Da * (1 - exp_d)
    shrink_cc = cc_in * np.abs(edges.flow) * sid.Gamma / sid.Da * (1 - exp_p)
    shrink_cb = edges.alpha_b * cb_in * cd_in * sid.K * np.abs(edges.flow) * sid.Gamma / sid.Da \
        * ((1 - exp_d) / (1 + sid.G * edges.diams * sid.K) - (1 - exp_p) * sid.Kp / (sid.K * cd_in * (1 + sid.G * edges.diams))) / ksi
    shrink_cc = np.array(np.ma.fix_invalid(shrink_cc, fill_value = 0)) 
    shrink_cb = np.array(np.ma.fix_invalid(shrink_cb, fill_value = 0)) 
    growth2 = cb_in * np.abs(edges.flow)  / sid.Da * (1 - exp_d2)
    shrink_cc2 = cc_in * np.abs(edges.flow) * sid.Gamma / sid.Da * (1 - exp_p2)
    shrink_cb2 = edges.alpha_b * cb_in * cd_in * sid.K * np.abs(edges.flow) * sid.Gamma / sid.Da \
        * ((1 - exp_d2) / (1 + sid.G * edges.diams * sid.K) - (1 - exp_p2) * sid.Kp / (sid.K * cd_in * (1 + sid.G * edges.diams))) / ksi
    shrink_cc2 = np.array(np.ma.fix_invalid(shrink_cc2, fill_value = 0)) 
    shrink_cb2 = np.array(np.ma.fix_invalid(shrink_cb2, fill_value = 0)) 
    change = growth2 - np.abs(shrink_cc2) - np.abs(shrink_cb2)
    return change, growth, np.abs(shrink_cc) + np.abs(shrink_cb)
