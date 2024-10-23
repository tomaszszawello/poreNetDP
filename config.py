""" Initial parameters of the simulation.

This module contains all parameters set before the simulation. Class
SimInputData is used in nearly all functions. Most of the parameters (apart
from VARIOUS section) are set by the user before starting the simulation.
Most notable parameters are: n - network size, iters/tmax - simulation length,
Da_eff, G, K, Gamma - dissolution/precipitation parameters, include_cc - turn
on precipitation, load - build a new network or load a previous one.
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
    tmax: float = 100000.
    "maximum time"
    dissolved_v_max: float = 20
    "maximum dissolved pore volume"
    plot_every: int = 1
    "frequency of plotting the results"
    track_every: int = 2
    "frequency of checking channelization"
    track_list = [1, 2, 5, 10]
    "times of checking channelization"

    # DISSOLUTION & PRECIPITATION
    Da_eff: float = 0.5
    "effective Damkohler number"
    G: float = 5
    "diffusion to reaction ratio"
    Da: float = Da_eff * (1 + G)
    "Damkohler number"
    K: float = 0.5
    "precipitation to dissolution reaction rate"
    Gamma: float = 2.
    "precipitation to dissolution acid capacity number"
    merge_length: float = 100.
    "diameter scale to length scale ratio for merging"
    n_tracking = 2000

    initial_pipe = False
    pipe_diam = 5
    pipe_width = 2

    # INCLUDE
    include_adt: bool = True
    "include adaptive timestep"
    include_cc: bool = False
    "include precipitation"
    include_merging: bool = False
    "include pore merging"

    # INITIAL CONDITIONS
    qin: float = 1.
    "characteristic flow for inlet edge"
    cb_in: float = 1.
    "inlet B concentration"
    cc_in: float = 0.
    "inlet C concentration"

    # TIME
    dt: float = 0.01
    "initial timestep (if no adaptive timestep, timestep for whole simulation)"
    growth_rate: float = 0.01
    ("maximum percentage growth of an edges (used for finding adaptive \
     timestep)")
    dt_max: float = 5.
    "maximum timestep (for adaptive)"

    # DIAMETERS
    noise: str = 'gaussian'
    ("type of noise in diameters distribution: 'gaussian', 'lognormal', \
    'klognormal', 'file_lognormal_d', 'file_lognormal_k'")
    noise_filename: str = 'n100lam10r01.dat'
    "name of file with initial diameters if noise == file_"
    d0: float = 1.
    "initial dimensionless mean diameter"
    sigma_d0: float = 0.
    "initial diameter standard deviation"
    dmin: float = 0
    "minimum diameter"
    dmax: float = 1000.
    "maximum diameter"
    d_break: float = 4.
    "minimal diameter of outlet edge for network to be dissolved"

    # DRAWING
    figsize: float = 10.
    "figure size"
    qdrawconst: float = 0.3
    "constant for improving flow drawing"
    ddrawconst: float = 0.1
    "constant for improving diameter drawing"
    draw_th_q: float = 3
    "threshold for drawing of flow"
    draw_th_d: float = 3
    "threshold for drawing of diameters"

    # INITIALIZATION
    load: int = 0
    ("type of loading: 0 - build new network based on config and start new \
     simulation, 1 - load previous network from load_name and continue \
     simulation, 2 - load template network from load_name and start new \
     simulation")
    load_name: str = 'rect100/G5.00Daeff0.05/1'
    "name of loaded network"

    # GEOMETRY
    geo: str = "rect" # WARNING - own is deprecated
    ("type of geometry: 'rect' - rectangular, 'own' - custom inlet and outlet \
     nodes, set in in/out_nodes_own")
    periodic: str = 'top'
    ("periodic boundary condition: 'none' - no PBC, 'top' - up and down, \
     'side' - left and right, 'all' - PBC everywhere")
    in_nodes_own: np.ndarray = np.array([[20, 50]]) / 100 * n
    "custom outlet for 'own' geometry"
    out_nodes_own: np.ndarray = np.array([[80, 50], [70, 25], [70, 75]]) \
        / 100 * n
    "custom outlet for 'own' geometry"

    # VARIOUS
    ne: int = 0
    "number of edges (updated later)"
    ntr: int = 0
    "number of triangles (updated later)"
    nsq: int = n ** 2
    "number of nodes"
    old_iters: int = 0
    "total iterations of simulation"
    old_t: float = 0.
    "total time of simulation"
    Q_in = 1.
    "total inlet flow (updated later)"
    #dirname: str = geo + str(n) + '/' + f'G{G:.2f}Daeff{Da_eff:.2f}'
    dirname: str = 'tracking/' + f'G{G:.2f}Daeff{Da_eff:.2f}'
    "directory of simulation"
    initial_merging: int = 5
    "number of initial merging iterations"
    