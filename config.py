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
    m: int = 50
    "network size"
    iters: int = 10000
    "maximum number of iterations"
    tmax: float = 10000000.
    "maximum time"
    dissolved_v_max: float = 100
    "maximum dissolved pore volume"
    plot_every: int = 100000
    "frequency of plotting the results"
    track_every: int = dissolved_v_max / 10
    "frequency of checking channelization"
    track_list = [1, 2, 5, 10]
    "times of checking channelization"

    # DISSOLUTION & PRECIPITATION
    Da_L = 100
    Pe_L = 1
    load_name: str = 'diffusion/Pe1000.00Da1000.00/1/template/71'
    # Da: float = 100#0.67 * 10 ** -1
    # "effective Damkohler number"
    # Pe = 100.
    chi0 = 0.1
    Sh = 4
    include_diffusion = True
    Da = Da_L * np.pi * 2  * chi0 / (m)
    Pe = Pe_L * 2 / (np.pi * chi0 ** 2)

    G: float = Da * Pe / Sh * chi0 ** 2 / 4
    "diffusion to reaction ratio"
    Da_eff: float = Da / (1 + G)
    "Damkohler number"
    # G = 50
    # Da_eff = 5
    # Da = Da_eff * (1 + G)
    K: float = 0.5
    "precipitation to dissolution reaction rate"
    Gamma: float = 2.
    "precipitation to dissolution acid capacity number"
    merge_length: float = 1 / chi0
    "diameter scale to length scale ratio for merging"
    n_tracking = 2000

    Jb_in = 1
    diffusion_exp_limit = 10
    "threshold above which we set the lambda+ solution for concentration to zero"

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
    growth_rate: float = 0.05
    ("maximum percentage growth of an edges (used for finding adaptive \
     timestep)")
    dt_max: float = 50000.
    "maximum timestep (for adaptive)"

    # DIAMETERS
    noise: str = 'file_lognormal_k'
    ("type of noise in diameters distribution: 'gaussian', 'lognormal', \
    'klognormal', 'file_lognormal_d', 'file_lognormal_k'")
    noise_filename: str = 'n100m300lam30r01.dat' #'n200lam20r1.dat'
    "name of file with initial diameters if noise == file_"
    d0: float = 1.
    "initial dimensionless mean diameter"
    sigma_d0: float = 0
    "initial diameter standard deviation"
    dmin: float = 0
    "minimum diameter"
    dmax: float = n
    "maximum diameter"
    d_break: float = 4.
    "minimal diameter of outlet edge for network to be dissolved"

    # DRAWING
    figsize: float = 30.
    "figure size"
    qdrawconst: float = 10 / n
    "constant for improving flow drawing"
    ddrawconst: float = 2400 / n * chi0 #10 / n
    "constant for improving diameter drawing"
    draw_th_q: float = 3
    "threshold for drawing of flow"
    draw_th_d: float = 2
    "threshold for drawing of diameters"

    # INITIALIZATION
    load: int = 0
    ("type of loading: 0 - build new network based on config and start new \
     simulation, 1 - load previous network from load_name and continue \
     simulation, 2 - load template network from load_name and start new \
     simulation")
    #load_name: str = 'tracking/Pe50.00Da20.00/1'
    
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
    nsq: int = n * m #n ** 2
    "number of nodes"
    old_iters: int = 0
    "total iterations of simulation"
    old_t: float = 0.
    "total time of simulation"
    Q_in = 1.
    "total inlet flow (updated later)"
    #dirname: str = geo + str(n) + '/' + f'G{G:.2f}Daeff{Da_eff:.2f}'
    dirname: str = 'diffusion/' + f'Pe{Pe_L:.2f}Da{Da_L:.2f}'
    "directory of simulation"
    initial_merging: int = 5
    "number of initial merging iterations"
    