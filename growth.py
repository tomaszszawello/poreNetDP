""" Updates edges diameters based on dissolution.

This module calculates the change od diameters in the network, resulting from
dissolution. Based on that change, new timestep is calculated.

Notable functions
-------
update_diameters(SimInputData, Incidence, Edges, numpy ndarray) \
    -> tuple[bool, float]
    update diameters, calculate timestep and check if network is dissolved
"""

import numpy as np
import scipy.sparse as spr

from config import SimInputData
from incidence import Edges, Incidence


def update_apertures(sid: SimInputData, inc: Incidence, edges: Edges, \
    concentration: np.ndarray) -> tuple[bool, float]:
    """ Update diameters.

    This function updates diameters of edges, calculates the next timestep (if
    adt is used) and checks if the network is dissolved.

    Parameters
    -------
    sid : simInputData class object
        all config parameters of the simulation

    inc : Incidence class object
        matrices of incidence

    edges : Edges class object
        all edges in network and their parameters

    concentration : numpy ndarray
        vector of solvent concentration

    Returns
    -------
    breakthrough : bool
        parameter stating if the system was dissolved (if diameter of outlet
        edge grew at least sid.d_break times)

    dt_next : float
        new timestep
    """

    breakthrough = False
    # create list of concentrations which should be used for growth of each
    # edge (upstream one)
    concentration_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) \
        @ concentration
    # calculate growth of each aperture
    aperture_change = concentration_in * np.abs(edges.flow)  / (sid.Da \
        * edges.lens) * (1 - np.exp(-np.abs(sid.Da / (1 + sid.G \
        * edges.apertures) * edges.lens / edges.flow)))
    # recalculate timestep (if adt is used), so that maximum growth of aperture
    # is by exactly sid.growth_rate
    if sid.include_adt:
        dt = sid.growth_rate / float(np.max(aperture_change / edges.apertures))
        # clip calculated timestep to sid.dt_max
        if dt > sid.dt_max:
            dt = sid.dt_max
    else:
        # if no adt, just use timestep from config
        dt = sid.dt
    # changes apertures using recalculated timestep
    edges.apertures += dt * aperture_change
    # check if network is dissolved
    if np.max(edges.outlet * edges.apertures) > sid.b_break \
        and sid.include_breakthrough:
        breakthrough = True
        print ('Network dissolved.')
    return breakthrough, dt
