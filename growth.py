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
from network import Edges
from incidence import Incidence


def update_diameters(sid: SimInputData, inc: Incidence, edges: Edges, \
    cb: np.ndarray, cc: np.ndarray) -> tuple[bool, float]:
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
    change = solve_d(sid, inc, edges, cb, cc)
    breakthrough = False
    if sid.include_adt:
        change_rate = change / edges.diams * (edges.diams > 0.5)
        change_rate = np.array(np.ma.fix_invalid(change_rate, fill_value = 0))
        rate = np.max(np.abs(change_rate))
        if rate == 0:
            rate = 10
        dt_next = sid.growth_rate / rate
        if dt_next > sid.dt_max:
            dt_next = sid.dt_max
    else:
        dt_next = sid.dt
    diams_new = edges.diams + change * dt_next
    if sid.cut_edges:
        diams_new = diams_new * (diams_new > sid.dmin)
        if np.sum(diams_new == 0) != np.sum(edges.diams == 0):
            print("Edges cut")
        inc.incidence = inc.incidence.multiply(1 * (diams_new > 0)[:, np.newaxis])
        inc.inlet = inc.inlet.multiply(1 * (diams_new > 0)[:, np.newaxis])
        edges.inlet *= diams_new > 0
        edges.outlet *= diams_new > 0
    else:
        diams_new = diams_new * (diams_new >= sid.dmin) \
            + sid.dmin * (diams_new < sid.dmin)

    # if sid.include_adt:
    #     diams_rate = np.abs((diams_new - edges.diams) / edges.diams)
    #     diams_rate = np.array(np.ma.fix_invalid(diams_rate, fill_value = 0))
    #     dt_next = sid.growth_rate / sid.dt / np.max(diams_rate)
    #     if dt_next > sid.dt_max:
    #         dt_next = sid.dt_max

    edges.diams = diams_new
    # if np.max(edges.diams / edges.diams_initial) > 300:
    #     breakthrough = True
    return breakthrough, dt_next

def solve_d(sid: SimInputData, inc: Incidence, edges: Edges, cb: np.ndarray, cc: np.ndarray) \
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
    cb_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ np.abs(cb)
    cc_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ np.abs(cc)
    # change = -cb_in * np.abs(edges.flow) / (sid.Da * edges.lens \
    #     * edges.diams) * (1 - np.exp(-cc_in / sid.c_eq * sid.Da / (1 + sid.G * edges.diams) \
    #     * edges.diams * edges.lens / np.abs(edges.flow)))
    # change = -sid.c_eq * np.abs(edges.flow) / (sid.Da * edges.lens \
    #     * edges.diams) * np.log(np.abs((-cc_in + cb_in * np.exp((cb_in - cc_in) / sid.c_eq * sid.Da / (1 + sid.G * edges.diams) \
    #     * edges.diams * edges.lens / np.abs(edges.flow))) / (cb_in - cc_in)))
    exp_a = np.exp((cb_in - cc_in) / sid.c_eq * sid.Da / (1 + sid.G * edges.diams) \
         * edges.diams * edges.lens / np.abs(edges.flow))
    exp_a = np.array(np.ma.fix_invalid(exp_a, fill_value = 0))
    change = -cb_in * cc_in * (-1 + exp_a) / (-cc_in + cb_in * exp_a) * np.abs(edges.flow) / (sid.Da * edges.lens \
         * edges.diams)

    change = np.array(np.ma.fix_invalid(change, fill_value = 1))
    change_fix = -cb_in * cc_in / (1 + cb_in / sid.c_eq * sid.Da / (1 + sid.G * edges.diams) \
         * edges.diams * edges.lens / np.abs(edges.flow)) * np.abs(edges.flow) / (sid.Da * edges.lens \
         * edges.diams)
    change_fix = np.array(np.ma.fix_invalid(change_fix, fill_value = 0))
    change = change * (change != 1) + change_fix * (change == 1)
    change *= (edges.diams > sid.dmin)
    # change = change * (change != 1) -cb_in * cc_in / (1 + cb_in / sid.c_eq * sid.Da / (1 + sid.G * edges.diams) \
    #      * edges.diams * edges.lens / np.abs(edges.flow)) * (change == 1) * np.abs(edges.flow) / (sid.Da * edges.lens \
    #      * edges.diams)
    if np.sum(change > 1e-2):
        np.savetxt('change.txt', change)
        raise ValueError('change > 0')
    return change
