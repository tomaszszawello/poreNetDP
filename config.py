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
    n: int = 200
    "network size along y (transverse to the flow)"
    m: int = 200
    "network size along x (parallel to the flow)"
    iters: int = 10000000
    "maximum number of iterations"
    
    "maximum time"
    phi = 0.001
    dissolved_v_max: float = 100000000.
    "maximum dissolved pore volume"
    plot_every: int = 100000
    "frequency of plotting the results"
    
    "frequency of checking channelization"
    track_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #track_list = [1, 2, 5, 10]
    "times of checking channelization"

    p_in = 1
    frac_diam = 100

    # DISSOLUTION & PRECIPITATION
    Da_L = 100
    Pe_L = 10
    #0.67 * 10 ** -1
    # "effective Damkohler number"
    # Pe = 100.

    #chi0 = (1 - (1 - phi) ** (1/3)) / np.sqrt(3)
    #chi0 = (1 / (1 - phi) ** (1/3) - 1) / np.sqrt(3)
    #chi0 = 4 * np.sqrt(phi) / np.pi
    chi0 = 2 * np.sqrt(phi) / np.pi / np.sqrt(3)
    Sh = 4
    include_diffusion = True
    ksi = 4.364 * 100 # Sh / chi ^ 2
    #Pe = Pe_L * 2 / (np.pi * chi0 ** 2)
    Pe = 0.01 #Pe_L * 4 / (np.pi * chi0 ** 2)
    Da2: float = 1
    Da: float = Da2 / Pe
    grain_vol = 1

    tmax: float = 1000.
    track_every: int = tmax / 10
    inert_fraction = 0.9

    M = 0.1
    cosm_in = 1.25
    cosm_out = 0.75

    G: float = 0#Da * Pe / Sh * chi0 ** 2 / 4
    "diffusion to reaction ratio"
    Da_eff: float = Da / (1 + G)
    "Damkohler number"
    # G = 50
    # Da_eff = 5
    #Da = Da_eff * (1 + G)
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

    c_th = 0.01

    initial_pipe = False
    pipe_diam = 5
    pipe_width = 2
    phi_max = 1

    # INCLUDE
    include_adt: bool = True
    "include adaptive timestep"
    include_cc: bool = False
    "include precipitation"
    include_merging: bool = False
    "include pore merging"
    include_volumes: bool = True
    "include pore volume tracking"
    include_electroosmosis: bool = False

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
    growth_rate: float = 0.2
    ("maximum percentage growth of an edges (used for finding adaptive \
     timestep)")
    dt_max: float = 50000.
    "maximum timestep (for adaptive)"

    it_alpha_th = 1e-5
    it_limit = 100

    # DIAMETERS
    noise: str = 'file_lognormal_k'
    ("type of noise in diameters distribution: 'gaussian', 'lognormal', \
    'klognormal', 'file_lognormal_d', 'file_lognormal_k'")
    noise_filename: str = 'n200lam20r1.dat' #'n200lam20r1.dat'
    #noise_filename: str = 'n100m300lam30r01.dat' #'n200lam20r1.dat'
    rock_filename: str = 'n200lam20r2.dat'
    "name of file with initial diameters if noise == file_"
    d0: float = 1.
    "initial dimensionless mean diameter"
    sigma_d0: float = 0.1
    "initial diameter standard deviation"
    dmin: float = 0
    "minimum diameter"
    dmax: float = 1
    "maximum diameter"
    d_break: float = 4.
    "minimal diameter of outlet edge for network to be dissolved"
    sigma_phi: float = 0#.2

    # DRAWING
    figsize: float = 20.
    "figure size"
    qdrawconst: float = 0.05
    "constant for improving flow drawing"
    ddrawconst: float = 0.05 #2400 / n * chi0 #10 / n
    "constant for improving diameter drawing"
    draw_th_q: float = 0.1
    "threshold for drawing of flow"
    draw_th_d: float = 0.1
    "threshold for drawing of diameters"

    # INITIALIZATION
    load: int = 2
    ("type of loading: 0 - build new network based on config and start new \
     simulation, 1 - load previous network from load_name and continue \
     simulation, 2 - load template network from load_name and start new \
     simulation")
    #load_name: str = 'electro/Pe1.00Da1.00/5'
    #load_name: str = 'paper_diff/Pe1.00Da1.00/4'
    load_name: str = 'fracture_diffusion/Pe1.00Da1.00/5'
    
    "name of loaded network"

    # GEOMETRY
    #geo: str = "rect" # WARNING - own is deprecated
    geo: str = "own"
    ("type of geometry: 'rect' - rectangular, 'own' - custom inlet and outlet \
     nodes, set in in/out_nodes_own")
    periodic: str = 'none'
    ("periodic boundary condition: 'none' - no PBC, 'top' - up and down, \
     'side' - left and right, 'all' - PBC everywhere")
    in_nodes_own: np.ndarray = np.array([[0, 0]]) / 100 * n
    "custom outlet for 'own' geometry"
    out_nodes_own: np.ndarray = np.array([[0, 100], [100, 0]]) \
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
    dirname: str = 'fracture_diffusion/' + f'Pe{Pe:.2f}Da{Da:.2f}'
    "directory of simulation"
    initial_merging: int = 5
    "number of initial merging iterations"
    