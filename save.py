""" Utilities for saving and loading network template and simulation results.

This module contains functions for saving the network template of each
simulation (to be able to start a new simulation with exactly the same initial
conditions) and the result of the simulation (to continue simulation in the
future) and for loading simulation in both cases. It also contains function for
saving the config file (to know simulation parameters).
"""

import dill

from config import SimInputData
from incidence import Incidence
from network import Edges, Graph


def save(name: str, sid: SimInputData, graph: Graph, inc: Incidence, \
    edges: Edges) -> None:
    """ Save all simulation data.

    This function saves all data necessary to either continue simulation from
    a given point or start a new simulation with the same initial conditions.

    Parameters
    -------
    sid : simInputData class object
        all config parameters of the simulation

    graph : Graph class object
        network and all its properties
    """
    data = [sid, graph, inc, edges]
    with open(sid.dirname+name, 'wb') as file:
        dill.dump(data, file)

def load(name: str) -> tuple[SimInputData, Graph, Incidence, Edges]:
    """ Load all simulation data.

    This function loads data necessary to recreate the simulation or continue
    it.

    Parameters
    -------
    name : string
        name of the file from which simulation is recreated (usually
        directory + save.dill or template.dill)

    Returns
    -------
    sid : simInputData class object
        all config parameters of the simulation

    graph : Graph class object
        network and all its properties
    """
    with open(name, 'rb') as file:
        data = dill.load(file)
    sid = data[0]
    graph = data[1]
    inc = data[2]
    edges = data[3]
    return  sid, graph, inc, edges

def save_config(sid: SimInputData) -> None:
    """ Save configuration of simulation to text file.

    Parameters
    -------
    sid : SimInputData
        all config parameters of the simulation
        dirname - directory of simulation
    """
    f = open(sid.dirname+'/config.txt', 'w', encoding = "utf-8")
    for key, val in sid.__class__.__dict__.items():
        f.write(f'{key} = {val} \r')
    f.close()
