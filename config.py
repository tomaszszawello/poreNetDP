""" Initial parameters of the simulation.

This module contains all parameters set before the simulation. Class
SimInputData is used in nearly all functions. Most of the parameters (apart
from VARIOUS section) are set by the user before starting the simulation.
Most notable parameters are: n - network size, iters/tmax - simulation length,
Da_eff, G, K, Gamma - dissolution/precipitation parameters, include_cc - turn
on precipitation, load - build a new network or load a previous one.

TO DO: fix own geometry
"""

import numpy as np


class SimInputData:
    ''' Configuration class for the whole simulation.
    '''
    # GENERAL
    n: int = 50
    "network size"
    iters: int = 1000000
    "maximum number of iterations"
    tmax: float = 500.
    "maximum time"
    dissolved_v_max: float = 10
    "maximum dissolved volume (in terms of initial pore volume)"
    plot_every: int = 10
    "frequency of plotting the results"
    plotting_mode: str = 'time' # 'volume', 'iters'
    "time measure used for plotting"

    track_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    "list of time measures in which tracking is performed"

    # DISSOLUTION & PRECIPITATION
    Da_eff: float = 0.1
    "effective Damkohler number"
    G: float = 5.
    "diffusion to reaction ratio"
    Da: float = Da_eff * (1 + G)
    "Damkohler number"
    chi0: float = 0.01
    "diameter scale to length scale ratio for merging"

    # INCLUDE
    include_adt: bool = True
    "include adaptive timestep"
    include_merging: bool = False
    "include pore merging"
    tracking_mode = 'time'

    # INITIAL CONDITIONS
    qin: float = 1.
    "characteristic flow for inlet edge"
    cb_in: float = 1.
    "inlet B concentration"
    cc_in: float = 2.
    "inlet C concentration"
    initial_merging: int = 5
    "number of initial merging iterations"
    c_eq = 2.
    c_th = 1e-3
    solve_type = "simple" # "full"

    q_rate = 1
    q_amp = 0.1
    q_period = 50

    # TIME
    dt: float = 0.01
    "initial timestep (if no adaptive timestep, timestep for whole simulation)"
    growth_rate: float = 0.05
    ("maximum percentage growth of an edges (used for finding adaptive \
     timestep)")
    dt_max: float = 5.
    "maximum timestep (for adaptive)"

    # DIAMETERS
    noise: str = 'file_lognormal_k' # 'gaussian', 'lognormal', 'klognormal'
    # 'file_lognormal_d', 'file_lognormal_k'
    "type of noise in diameters distribution"
    noise_filename: str = 'n100lam10r01.dat'
    "name of file with initial diameters if noise == file_"
    d0: float = 1.
    "initial dimensionless mean diameter"
    sigma_d0: float = 0.1
    "initial diameter standard deviation"
    dmin: float = 0.1
    "minimum diameter"
    dmax: float = 1000.
    "maximum diameter"
    d_breakthrough: float = 4.
    "minimal diameter of outlet edge for network to be dissolved"

    # DRAWING
    figsize: float = 10.
    "figure size"
    qdrawconst: float = 0.3
    "constant for improving flow drawing"
    ddrawconst: float = 2.
    "constant for improving diameter drawing"
    
    # INITIALIZATION
    load: int = 0
    ("type of loading: 0 - build new network based on config and start new \
     simulation, 1 - load previous network from load_name and continue \
     simulation, 2 - load template network from load_name and start new \
     simulation")
    load_name: str = 'singurindy/G5.00Daeff0.10/17'
    "name of loaded network"
    dirname: str = f'singurindy/G{G:.2f}Daeff{Da_eff:.2f}'
    "directory of simulation"

    # GEOMETRY
    geo: str = "rect" # WARNING - own is deprecated
    ("type of geometry: 'rect' - rectangular, 'own' - custom inlet and outlet \
     nodes, set in in/out_nodes_own")
    periodic: str = 'none'
    ("periodic boundary condition: 'none' - no PBC, 'top' - up and down, \
     'side' - left and right, 'all' - PBC everywhere")
    in_nodes_own: np.ndarray = np.array([[20, 50]]) / 100 * n
    "custom outlet for 'own' geometry"
    out_nodes_own: np.ndarray = np.array([[80, 50], [70, 25], [70, 75]]) \
        / 100 * n
    "custom outlet for 'own' geometry"

    # VARIOUS (updated during simulation)
    ne: int = 0
    "number of edges"
    nsq: int = n ** 2
    "number of nodes"
    old_iters: int = 0
    "total iterations of simulation"
    old_t: float = 0.
    "total time of simulation"
    Q_in = 1.
    "total inlet flow"