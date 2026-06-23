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
    ''' Configuration class for the simulation.
    '''
    # GENERAL
    n: int = 50
    "network size along y (transverse to the flow)"
    m: int = 50
    "network size along x (parallel to the flow)"
    iters: int = 10000000
    "maximum number of iterations"
    tmax: float = 1000.
    "maximum time"
    
    dissolved_v_max: float = 10.
    "maximum dissolved pore volume"
    plot_every: int = 100000
    "frequency of plotting the results"
    track_every: int = 1.0 # tmax / 2
    "frequency of checking channelization"
    track_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #track_list = [1, 2, 5, 10]
    "times of checking channelization"
    track_type = "time" # "dissolved, "time" or "iterate"

    # DISSOLUTION & PRECIPITATION
    Da_eff: float = 10
    "effective Damkohler number"
    G: float = 0.
    "transport parameter (reaction to transverse diffusion)"
    
    Pe = 100.
    "Peclet number (only if include_diffusion = 1)"
    phi = 0.23
    "porosity (physical if include_volumes = 1; otherwise just setting chi0)"
    chi0 = 2 * np.sqrt(phi) / np.pi / np.sqrt(3)
    "pore aspect ratio"
    Da: float = Da_eff * (1 + G)
    "Damkohler number"
    inert_fraction = 0.27
    "average fraction of inert mineral in grains"
    V_tot = (1 / chi0) ** 2 * 3 / 4 / np.pi
    "grain total volume"
    K: float = 10
    "precipitation to dissolution reaction rate"
    Gamma: float = 1
    "precipitation to dissolution molar volume / acid capacity number"
    A: float = 1000
    "nucleation rate pre-factor"
    c_sat: float = 1/10000.
    "saturation limit"
    merge_length: float = 1 / chi0
    "diameter scale to length scale ratio for merging"
    n_tracking = 2000


    debug = False

    min_perm = 1e-3

    cb_0 = 1
    diffusion_exp_limit = 20
    "threshold above which we set the lambda+ solution for concentration to zero"


    # INCLUDE
    include_adt: bool = False
    "include adaptive timestep"
    include_diffusion = False
    "include diffusion for dissolution"
    include_precipitation: bool = True
    "include precipitation"
    include_merging: bool = False
    "include pore merging"
    include_volumes: bool = False
    "include pore volume tracking"
    include_nucleation: bool = True
    "include nucleation (requires precipitation)"

    flow_bc: str = "q" # "p"
    "flow boundary condition: constant total flow rate or constant pressure"

    # INITIAL CONDITIONS
    qin: float = 1.
    "characteristic flow for inlet edge"
    cb_in: float = 1.
    "inlet B concentration"
    cc_in: float = 0.
    "inlet C concentration"
    cd_in: float = 1.
    "inlet D concentration"

    # TIME
    dt: float = 0.0001
    "initial timestep (if no adaptive timestep, timestep for whole simulation)"
    growth_rate: float = 0.05
    ("maximum percentage growth of an edges (used for finding adaptive \
     timestep)")
    dt_max: float = 0.01
    "maximum timestep (for adaptive)"

    it_alpha_th = 1e-5
    it_limit = 100
    c_th = 1e-2
    Kp = 1#1e-5#0.01

    # DIAMETERS
    noise: str = 'lognormal'
    ("type of noise in diameters distribution: 'gaussian', 'lognormal', \
    'klognormal', 'file_lognormal_d', 'file_lognormal_k'")
    noise_filename: str = 'n200lam20r1.dat' #'n200lam20r1.dat'
    #noise_filename: str = 'n100m300lam30r01.dat' #'n200lam20r1.dat'
    rock_filename: str = 'n200lam20r1.dat'
    "name of file with initial diameters if noise == file_"
    d0: float = 1.
    "initial dimensionless mean diameter"
    sigma_d0: float = 0.1
    "initial diameter standard deviation"
    dmin: float = 0
    "minimum diameter"
    dmax: float = n
    "maximum diameter"
    d_break: float = 4.
    "minimal diameter of outlet edge for network to be dissolved"
    sigma_phi: float = 0.5
    "initial porosity lognormal deviation"

    # DRAWING
    figsize: float = 10.
    "figure size"
    qdrawconst: float = 50 / n
    "constant for improving flow drawing"
    ddrawconst: float = 1 #2400 / n * chi0 #10 / n
    "constant for improving diameter drawing"
    draw_th_q: float = 0.1
    "threshold for drawing of flow"
    draw_th_d: float = 0.1
    "threshold for drawing of diameters"

    # INITIALIZATION
    load: int = 0
    ("type of loading: 0 - build new network based on config and start new \
     simulation, 1 - load previous network from load_name and continue \
     simulation, 2 - load template network from load_name and start new \
     simulation")
    #load_name: str = 'electro/Pe1.00Da1.00/5'
    #load_name: str = 'paper_diff/Pe1.00Da1.00/4'
    load_name: str = 'paper_diff/Pe1.00Da1.00/3/template/37'
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
    p_in = 0.
    "inlet pressure (updated later, if flow_bc is constant pressure)"
    #dirname: str = geo + str(n) + '/' + f'G{G:.2f}Daeff{Da_eff:.2f}'
    dirname: str = 'integration/' + f'A{A:.2f}Da{Da:.2f}K{K:.2f}Gamma{Gamma:.2f}'
    "directory of simulation"
    initial_merging: int = 5
    "number of initial merging iterations"
    
