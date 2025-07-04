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
    n: int = 100
    "network size along y (transverse to the flow)"
    m: int = 50
    "network size along x (parallel to the flow)"
    iters: int = 1000000
    "maximum number of iterations"
    tmax: float = 50.
    "maximum time"
    dissolved_v_max: float = 100
    "maximum dissolved pore volume"
    plot_every: int = 100000
    "frequency of plotting the results"
    track_every: int = tmax / 10 #dissolved_v_max / 10
    "frequency of checking channelization"
    track_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #track_list = [1, 2, 5, 10]
    "times of checking channelization"

    # DISSOLUTION & PRECIPITATION
    Da_L = 200
    Pe_L = 20
    load_name: str = 'diffusion/Pe1000.00Da1000.00/1/template/71'
    # Da: float = 100#0.67 * 10 ** -1
    # "effective Damkohler number"
    # Pe = 100.
    phi = 0.4
    #chi0 = (1 - (1 - phi) ** (1/3)) / np.sqrt(3)
    #chi0 = (1 / (1 - phi) ** (1/3) - 1) / np.sqrt(3)
    chi0 = 4 * np.sqrt(phi) / np.pi
    Sh = 4
    include_diffusion = True
    Da = Da_L * np.pi * chi0 / (m)
    #Pe = Pe_L * 2 / (np.pi * chi0 ** 2)
    Pe = Pe_L * 4 / (np.pi * chi0 ** 2)
    
    G: float = Da * Pe / Sh * chi0 ** 2 / 4
    "diffusion to reaction ratio"
    Da_eff: float = Da / (1 + G)
    "Damkohler number"
    # G = 50
    # Da_eff = 5
    # Da = Da_eff * (1 + G)
    #V_tot = (1 / chi0) ** 2 * np.sqrt(6) / 6 / np.pi
    V_tot = (1 / chi0) ** 2 * 3 / 4 / np.pi

    debug = False

    K: float = 0.5
    "precipitation to dissolution reaction rate"
    Gamma: float = 2.
    "precipitation to dissolution acid capacity number"
    merge_length: float = 1 / chi0
    "diameter scale to length scale ratio for merging"
    n_tracking = 2000

    cb_0 = 1
    diffusion_exp_limit = 20
    "threshold above which we set the lambda+ solution for concentration to zero"

    initial_pipe = False
    pipe_diam = 5
    pipe_width = 2
    phi_max = 1

    # INCLUDE
    include_adt: bool = True
    "include adaptive timestep"
    include_cc: bool = False
    "include precipitation"
    include_merging: bool = True
    "include pore merging"
    include_volumes: bool = True
    "include pore volume tracking"

    # INITIAL CONDITIONS
    qin: float = 1.
    "characteristic flow for inlet edge"
    cb_in: float = 1.
    "inlet B concentration"
    cc_in: float = 0.
    "inlet C concentration"

    # TIME
    dt: float = 0.000001
    "initial timestep (if no adaptive timestep, timestep for whole simulation)"
    growth_rate: float = 0.01
    ("maximum percentage growth of an edges (used for finding adaptive \
     timestep)")
    dt_max: float = 50000.
    "maximum timestep (for adaptive)"

    it_alpha_th = 1e-2
    it_limit = 100

    # DIAMETERS
    noise: str = 'file_lognormal_k'
    ("type of noise in diameters distribution: 'gaussian', 'lognormal', \
    'klognormal', 'file_lognormal_d', 'file_lognormal_k'")
    #noise_filename: str = 'n200lam20r1.dat' #'n200lam20r1.dat'
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
    ddrawconst: float = 1#2400 / n * chi0 #10 / n
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
    #load_name: str = 'diffusion/Pe0.00Da100.00/17'
    load_name: str = 'diffusion/Pe1.00Da100.00/40'#/template/13'
    
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
    