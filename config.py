""" Initial parameters of the simulation.

This module contains all parameters set before the simulation. Class
SimInputData is used in nearly all functions. Most of the parameters (apart
from VARIOUS section) are set by the user before starting the simulation.
Most notable parameters are: iters/tmax - simulation length, Da_eff, G - 
dissolution parameters.
"""


class SimInputData:
    """ Configuration class for the whole simulation.
    """
    # GENERAL
    iters: int = 10000000
    "maximum number of iterations"
    tmax: float = 10000000.
    "maximum time"
    save_every: int = 10000
    "frequency (in iterations) of plotting the results"
    collect_every: int = 10
    "frequency (in iterations) of collecting data"
    track_every: int = 0.1
    "frequency (in time) of performing tracking and slice check"
    load_name: str = 'samples/carbonate_x01'
    "name of loaded network"
    dissolved_v_max = 1
    track_list = [1, 2, 5, 10]
    dissolved_v = 0


    flow_focusing_profile = True
    aperture_focusing_profile = True
    # DISSOLUTION & PRECIPITATION
    Da_eff: float = 0.0002
    "effective Damkohler number"
    G: float = 5
    "diffusion to reaction ratio"
    Da: float = Da_eff * (1 + G)
    "Damkohler number"

    # INCLUDE
    include_adt: bool = True
    "include adaptive timestep"
    include_breakthrough: bool = False
    ("stop simulation when network is dissolved (i.e. diameter of edge \
     connected to the outlet grew b_break times)")

    # INITIAL CONDITIONS
    q_in: float = 1.
    "characteristic flow for inlet edge (dimensionless)"
    concentration_in: float = 1.
    "inlet solvent concentration (dimensionless)"
    b_break: float = 4.
    ("minimal ratio of aperture of outlet fracture to its initial aperture for \
     network to be dissolved")

    # TIME
    dt: float = 0.01
    "initial timestep (if no adaptive timestep, timestep for whole simulation)"
    growth_rate: float = 0.05
    ("maximum percentage growth of an edges (used for finding adaptive \
     timestep)")
    dt_max: float = 10000.
    "maximum timestep (for adaptive)"

    # VARIOUS
    n_nodes: int = 0
    "number of nodes (updated later)"
    n_edges: int = 0
    "number of edges (updated later)"
    b0: float = 1.
    "initial mean aperture (updated later)"
    l0: float = 1.
    "initial mean fracture length (updated later)"
    w0: float = 1.
    "initial mean fracture width (updated later)" 
    old_iters: int = 0
    "total iterations of simulation"
    old_t: float = 0.
    "total time of simulation"
    #dirname: str = f'{load_name}/G{G:.2f}Daeff{Da_eff:.2f}'
    dirname: str = f'{load_name}/G{G:.4f}Daeff{Da_eff:.4f}'
    "directory of simulation"
