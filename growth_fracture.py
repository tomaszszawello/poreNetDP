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
    vols: Volumes, cb: np.ndarray) -> tuple[bool, float]:
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
    change = solve_d_da(sid, inc, edges, cb)
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
    edges.grain -= inc.boundary @ (change * edges.diams * edges.lens / 4) * dt_next
    edges.active = 1 * ((1 * (np.abs(inc.incidence) @ graph.in_vec > 0) + 1 * (inc.center.T @ (edges.grain <= 0)) + 1 * (inc.boundary.T @ (edges.grain <= 0))) > 0)
    edges.alpha = 1 * (inc.boundary.T @ (edges.grain > 0)) / 2
    edges.diams = edges.diams * (1 - edges.active) + sid.dmax * edges.active
    
    return breakthrough, dt_next

def solve_d(sid: SimInputData, inc: Incidence, edges: Edges, cb: np.ndarray) \
    -> np.ndarray:
    
    edges.alpha = 1 * (inc.boundary.T @ (edges.grain > 0)) / 2 * edges.active
    lam_plus_val = sid.Pe / 2 / edges.diams ** 2 * \
        (np.sqrt(np.abs(edges.flow) ** 2 + 4 * edges.alpha * sid.ksi * edges.diams ** 2 / sid.Pe ** 2) + np.abs(edges.flow))
    lam_plus_val = np.array(np.ma.fix_invalid(lam_plus_val, fill_value = 0))
    lam_minus_val = sid.Pe / 2 / edges.diams ** 2 * \
        (np.sqrt(np.abs(edges.flow) ** 2 + 4 * edges.alpha * sid.ksi * edges.diams ** 2 / sid.Pe ** 2) - np.abs(edges.flow))
    lam_minus_val = np.array(np.ma.fix_invalid(lam_minus_val, fill_value = 0))
    lam_plus_zero = ((lam_plus_val > sid.diffusion_exp_limit) |
                (edges.diams == 0) |
                (edges.flow == 0))
    lam_plus_zero = lam_plus_zero.astype(int)
    change_pe_fix = (lam_plus_val > sid.diffusion_exp_limit) * edges.alpha * edges.B * np.abs(edges.flow) / (sid.ksi * edges.diams * edges.lens) * (1 - np.exp(-edges.alpha * sid.ksi / sid.Pe * edges.lens / np.abs(edges.flow)))
    change = (1 - lam_plus_zero) * edges.alpha * sid.Pe / (sid.ksi * edges.lens * edges.diams) * (edges.A * (np.exp(lam_plus_val * edges.lens) - 1) * lam_minus_val + edges.B * (1 - np.exp(-lam_minus_val * edges.lens)) * lam_plus_val) + change_pe_fix
    change = np.array(np.ma.fix_invalid(change, fill_value = 0))
    return change

def solve_d_da(sid: SimInputData, inc: Incidence, edges: Edges, cb: np.ndarray) \
    -> np.ndarray:
    
    edges.alpha = 1 * (inc.boundary.T @ (edges.grain > 0)) / 2 * edges.active
    lam_plus_val = sid.Pe / 2 / edges.diams ** 2 * \
        (np.sqrt(np.abs(edges.flow) ** 2 + 4 * edges.alpha * sid.Da * edges.diams ** 3 / sid.Pe) + np.abs(edges.flow))
    lam_plus_val = np.array(np.ma.fix_invalid(lam_plus_val, fill_value = 0))
    lam_minus_val = sid.Pe / 2 / edges.diams ** 2 * \
        (np.sqrt(np.abs(edges.flow) ** 2 + 4 * edges.alpha * sid.Da * edges.diams ** 3 / sid.Pe) - np.abs(edges.flow))
    lam_minus_val = np.array(np.ma.fix_invalid(lam_minus_val, fill_value = 0))
    lam_plus_zero = ((lam_plus_val > sid.diffusion_exp_limit) |
                (edges.diams == 0) |
                (edges.flow == 0))
    lam_plus_zero = lam_plus_zero.astype(int)
    change_pe_fix = (lam_plus_val > sid.diffusion_exp_limit) * edges.alpha * edges.B * np.abs(edges.flow) / (sid.Da * edges.diams * edges.lens) * (1 - np.exp(-edges.alpha * sid.Da * edges.diams * edges.lens / np.abs(edges.flow)))
    change = (1 - lam_plus_zero) * edges.alpha / (sid.Da * edges.lens * edges.diams) * (edges.A * (np.exp(lam_plus_val * edges.lens) - 1) * lam_minus_val + edges.B * (1 - np.exp(-lam_minus_val * edges.lens)) * lam_plus_val) + change_pe_fix
    change = np.array(np.ma.fix_invalid(change, fill_value = 0))
    return change
