""" Various utilities for other modules.

This module contains different utilities used by multiple other modules, e.g.
for solving matrix equations, initializing and updating iterators in the main
loop and creating simulation directory.

Notable functions
-------
solve_equation(spr.csr_matrix, spr.csc_matrix) -> spr.csc_matrix
    Solves for x matrix equation A * x = b.
"""

import numpy as np
import os
import scipy.sparse as spr
import scipy.sparse.linalg as sprlin

from config import SimInputData


def solve_equation(A: spr.csr_matrix, b: spr.csc_matrix) -> np.ndarray:
    """ Solves matrix equation A * x = b.

    Parameters
    -------
    A : scipy sparse matrix
        matrix A from equation

    b : scipy sparse vector
        result b from equation

    Returns
    -------
    numpy ndarray
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
    """ Creates directory for the simulation.

    Creates directory named with lowest unoccupied index in directory
    corresponding to simulation data (network name / Da_eff + G / index).

    Parameters
    -------
    sid : simInputData class object
        all config parameters of the simulation
    """
    i = 0
    dirname2 = sid.dirname
    while sid.dirname == dirname2:
        if not os.path.isdir(sid.dirname + "/" + str(i)):
            sid.dirname = sid.dirname + "/" + str(i)
        else:
            i += 1
    if not os.path.isdir(sid.dirname):
        os.makedirs(sid.dirname)

def save_config(sid:SimInputData) -> None:
    """ Saves config file od the simulation.

    Parameters
    -------
    sid : SimInputData
        all config parameters of the simulation
    """
    f = open(sid.dirname + '/config.txt', 'w')
    for key, val in sid.__class__.__dict__.items():
        # workaround to get all class attributes and at the same time get the
        # values of those updated during simulation
        if key not in sid.__dict__.keys():
            f.write(f'{key} = {val} \r')
        else:
            f.write(f'{key} = {sid.__dict__[key]} \r')
    f.close()

def create_bins(vals, num_bins, spacing = "log", x=[], weights=None, a_low=None, a_high=None, bin_edge = "center"):
    if a_low == None:
        a_low = 0.95 * np.min(vals)
    if a_high == None:
        a_high = np.max(vals)

    # Create bins 

    if spacing == "linear":
        x = np.linspace(a_low,a_high,num_bins+1)
    elif spacing == "log":
        if min(a_low, a_high) > 0:
            x = np.logspace(np.log10(a_low), np.log10(a_high), num_bins+1)
        else:
            x = np.logspace(-2, 2, num_bins+1, endpoint=False)
            A = np.max(x)
            B = np.min(x)
            x = (x-A) * (a_low - a_high) / (B-A) + a_high
    else: 
        print("Unknown spacing type. Using Linear spacing")
        x = np.linspace(a_low,a_high,num_bins+1)
    return x

    
def create_pdf(vals, num_bins, spacing = "log", x = [], weights=None, a_low=None, a_high=None, bin_edge = "center"):
    """  create pdf of vals 

    Parameters
    ----------
        vals : array
           array of values to be binned
        num_bins : int
            Number of bins in the pdf
        spacing : string 
            spacing for the pdf, options are linear and log
        x : array
            array of bin edges
        weights :array
            weights corresponding to vals to be used to create a weighted pdf
        a_low : float
            lower value of bin range. If no value provided 0.95*min(vals) is used
        a_high : float
            upper value of bin range. If no value is provided max(vals) is used
        bin_edge: string
            which bin edge is returned. options are left, center, and right

    Returns
    -------
        bx : array
            bin edges or centers (x values of the pdf)
        pdf : array
            values of the pdf, normalized so the Riemann sum(pdf*dx) = 1.
    """

    # Pick bin range 
    if a_low == None:
        a_low = 0.95 * np.min(vals)
    if a_high == None:
        a_high = np.max(vals)

    # Create bins 

    if spacing == "linear":
        x = np.linspace(a_low,a_high,num_bins+1)
    elif spacing == "log":
        if min(a_low, a_high) > 0:
            x = np.logspace(np.log10(a_low), np.log10(a_high), num_bins+1)
        else:
            x = np.logspace(-2, 2, num_bins+1, endpoint=False)
            A = np.max(x)
            B = np.min(x)
            x = (x-A) * (a_low - a_high) / (B-A) + a_high
    else: 
        print("Unknown spacing type. Using Linear spacing")
        x = np.linspace(a_low,a_high,num_bins+1)

    # Create PDF
    pdf, bin_edges = np.histogram(vals, bins=x, weights=weights, density=True)

    # Return arrays of the same size
    if bin_edge == "left":
        return bin_edges[:-1],pdf

    elif bin_edge == "right":
        return bin_edges[1:],pdf

    elif bin_edge == "center":
        bx = bin_edges[:-1] + 0.5*np.diff(bin_edges)
        return bx, pdf

    else: 
        print("Unknown bin edge type {0}. Returning left edges".format(bin_edge))
        return bin_edge[:-1],pdf
