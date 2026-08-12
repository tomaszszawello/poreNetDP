""" Various utilities for other modules.

This module contains different utilities used by multiple other modules, e.g.
for solving matrix equations, initializing and updating iterators in the main
loop and creating simulation directory.

Notable functions
-------
solve_equation(spr.csr_matrix, spr.csc_matrix) -> spr.csc_matrix
    Solve for x matrix equation A * x = b.
"""

import os
import scipy.sparse as spr
import scipy.sparse.linalg as sprlin

from config import SimInputData


def solve_equation(A: spr.csr_matrix, b: spr.csc_matrix) -> spr.csc_matrix:
    """ Solve matrix equation A * x = b.

    Parameters
    -------
    A : scipy sparse matrix
        matrix A from equation

    b : scipy sparse vector
        result b from equation

    Returns
    -------
    scipy sparse vector
        result x from equation
    """
    return sprlin.spsolve(A, b)

def initialize_iterators(sid: SimInputData) -> tuple[int, float, int, float, \
    bool]:
    """ Creates iterators for simulation steps, time and other conditions.

    Parameters
    -------
    sid : SimInputData class object
        all config parameters of the simulation
        iters - max no. of iterations of new simulation
        tmax - max time of new simulation
        old_iters - no. of previous iterations (if loaded from saved file)
        old_t - time of previous simulation (if loaded from saved file)

    Returns
    -------
    iters : int
        max no. of new iterations

    tmax : float
        max new time

    i : int
        iterator in range from old iterations to sum of old and new

    t : float
        time iterator in range from old time to sum of old and new

    breakthrough : bool
        parameter stating if the system was dissolved (if diameter of output
        edge grew at least to sid.d_break)
    """
    iters = sid.old_iters + sid.iters
    tmax = sid.old_t + sid.tmax
    i = sid.old_iters
    t = sid.old_t
    breakthrough = False
    return iters, tmax, i, t, breakthrough

def update_iterators(sid: SimInputData, i: int, t: float, dt_next: float) -> \
    tuple[int, float]:
    """ Updates iterators in simulation and in configuration class.

    Parameters
    -------
    sid : simInputData class object
        all config parameters of the simulation
        old_iters - no. of previous iterations (if loaded from saved file)
        old_t - time of previous simulation (if loaded from saved file)

    i : int
        current iteration

    t : float
        current time

    dt_next : float
        new timestep

    Returns
    -------
    i : int
        current iteration

    t : float
        current time
    """
    i += 1
    sid.old_iters += 1 # update simulation iterations in configuration class
    t += sid.dt
    sid.old_t += sid.dt # update simulation time in configuration class
    sid.dt = dt_next
    return i, t

def make_dir(sid: SimInputData) -> None:
    """ Create directory for the simulation.

    Create directory named with lowest unoccupied index in directory
    corresponding to simulation data (geometry + size / Da_eff + G).

    Parameters
    -------
    sid : simInputData class object
        all config parameters of the simulation
        dirname - directory of the simulation
    """
    i = 0
    while True:
        candidate = sid.dirname + "/" + str(i)
        try:
            os.makedirs(candidate)   # atomic: raises FileExistsError if already taken
            sid.dirname = candidate
            return
        except FileExistsError:
            i += 1

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

def keep_largest_component(inc):
    """
    Parameters
    ----------
    inc : scipy.sparse.csc_matrix (n_nodes × n_edges)
        Signed incidence matrix (+1/−1) of an undirected network.

    Returns
    -------
    inc_pruned : csc_matrix
        Copy of `inc` where rows (nodes) outside the largest
        connected component have been zeroed.
    keep_mask : ndarray, bool
        True for nodes kept; False for zeroed rows.
    """
    # 1. Build an undirected node-adjacency matrix:  A = |inc| · |inc|ᵀ
    A = abs(inc.incidence).astype(bool).astype(int)     # 0/1 matrix keeps multiplication cheap
    G = A @ A.T
    G.setdiag(0)                              # remove self-loops
    G.eliminate_zeros()

    # 2. Connected components of the node graph
    ncomp, labels = connected_components(G, directed=False)

    if ncomp <= 1:                            # already one component → return unchanged
        return None

    # 3. Largest component
    largest = np.argmax(np.bincount(labels))
    keep_mask = labels == largest

    # 4. Zero rows outside that component
    inc_pruned = inc.incidence.copy().tolil()           # easier row assignment
    inc_pruned[~keep_mask, :] = 0             # zero unwanted nodes
    inc.incidence = inc_pruned.tocsc()
    inlet_pruned = inc.inlet.copy().tolil()
    inlet_pruned[~keep_mask, :] = 0 
    inc.inlet = inlet_pruned.tocsc()
    return None