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

    m = 20
    n: int = 80
    "network size"
    iters: int = 100000000
    "maximum number of iterations"
    tmax: float = 36 * 4
    "maximum time"
    dissolved_v_max: float = 10
    "maximum dissolved volume (in terms of initial pore volume)"
    plot_every: int = tmax // 24
    "frequency of plotting the results"
    plotting_mode: str = 'time' # 'volume', 'iters'
    "time measure used for plotting"
    track_every: int = 1000000

    bound_x = n/5 #100 / 3 * np.sqrt(3) / 3 - 0.5
    bound_y = (m - 1)/2 #m * np.sqrt(3) / 2 - 1.2
    y_max = (2 * m) * np.sqrt(3) / 2
    y_min = 0

    diams_y_min = y_max / 2 - 10 * np.sqrt(3)
    diams_y_max = y_max / 2 + 10 * np.sqrt(3)

    track_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    "list of time measures in which tracking is performed"

    # DISSOLUTION & PRECIPITATION
    Da = 0.073 / 2
    #Da_eff: float = 0.026
    "effective Damkohler number"
    G: float = 1.05
    "diffusion to reaction ratio"
    #Da: float = Da_eff * (1 + G)
    Da_eff = Da / (1 + G)
    "Damkohler number"
    chi0: float = 0.63 # 0.3
    "diameter scale to length scale ratio for merging"

    # INCLUDE
    include_adt: bool = True
    "include adaptive timestep"
    include_merging: bool = False
    "include pore merging"
    tracking_mode = 'time'

    cut = 'intersections_up' # 'edges'
    pore_diam = 1.
    # INITIAL CONDITIONS
    qin: float = 4.
    "characteristic flow for inlet edge"
    cb_in: float = 1
    "inlet B concentration"
    cc_in: float = cb_in
    "inlet C concentration"
    initial_merging: int = 5
    "number of initial merging iterations"
    c_eq = 1.
    c_th = 1e-2
    solve_type = "full" # "full"
    Ksp = 7.78e-5

    v0 = 1
    #dmin = np.sqrt(1 / (1 + c_eq / (cb_in * Da_eff))) #* 0.5
    #dmin = 0.3 * np.sqrt(1 / (1 + 1 / Da_eff))
    #dmin = 0.1#0.42 * np.sqrt(1 / (1 + 1 / Da_eff))
    dmin = 1 / (1 + 1 / (Da_eff * cb_in))
    node_diam_min = 0#.1
    cond_weight = 0.5
    w_node = 0.7

    Pe_c = 0.01
    "critical Pe for mixing; 0 for streamlined, infinity for full"
    alpha_disp = 0#0.1
    "additional mixing due to mechanical dispersion"
    mixing_at_barrier = 0#0.05
    "additional mixing right behind barrier"

    q_rate = 1
    q_amp = 0.048 # 0.10
    q_period = tmax / 24#150
    q_trans = 0#q_period / 10

    # TIME
    dt: float = 1e-6
    "initial timestep (if no adaptive timestep, timestep for whole simulation)"
    growth_rate: float = 0.01
    ("maximum percentage growth of an edges (used for finding adaptive \
     timestep)")
    dt_max: float = 200.
    "maximum timestep (for adaptive)"

    # DIAMETERS
    noise: str = 'klognormal' # 'gaussian', 'lognormal', 'klognormal'
    # 'file_lognormal_d', 'file_lognormal_k'
    "type of noise in diameters distribution"
    noise_filename: str = 'n100lam103.dat'
    "name of file with initial diameters if noise == file_"
    d0: float = 1.
    "initial dimensionless mean diameter"
    sigma_d0: float = 0#0.0001
    "initial diameter standard deviation"
    #dmin: float = 0.3
    "minimum diameter"
    dmax: float = 1000.
    "maximum diameter"
    d_breakthrough: float = 4.
    "minimal diameter of outlet edge for network to be dissolved"

    # DRAWING
    figsize: float = 20.
    "figure size"
    qdrawconst: float = 1
    "constant for improving flow drawing"
    ddrawconst: float = 1.
    "constant for improving diameter drawing"
    
    # INITIALIZATION
    load: int = 0
    ("type of loading: 0 - build new network based on config and start new \
     simulation, 1 - load previous network from load_name and continue \
     simulation, 2 - load template network from load_name and start new \
     simulation")
    #load_name: str = 'Daeff0.25/25'
    load_name: str = 'Daeff0.56/2'
    "name of loaded network"
    dirname: str = f'mip/Daeff{Da_eff:.2f}'
    #dirname: str = f'Daeff{Da_eff:.2f}'
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
