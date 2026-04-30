""" Initialize main simulation classes depending on config data.

This module creates instances of classes necessary for simulation. Based mostly
on load parameter from config, it builds a new network and starts a new
simulation, loads an evolved network and continues simulation or loads some
template network and starts a new simulation on it.

Notable functions
-------
build(None) -> tuple[SimInputData, In.Incidence, De.Graph, In.Edges, Data]
    create class objects and initialize their parameters
"""

import network as Ne
import incidence as In
import save as Sv

from config import SimInputData
from data import Data
from utils import make_dir
from volumes import Volumes

import numpy as np

def build() -> tuple[SimInputData, In.Incidence, Ne.Graph, In.Edges, Volumes, Ne.Triangles, Data]:
    ''' Initialize main classes used in simulation based on config file.

    Create class objects and initialize their parameters. Make a simulation
    directory and save there a config file and a template.

    Parameters
    -------
    None

    Returns
    -------
    sid : SimInputData
        all config parameters of the simulation

    inc : Incidence class object
        matrices of incidence

    graph : Graph class object
        network and all its properties

    edges : Edges class object
        all edges in network and their parameters

    data : Data class object
        physical properties of the network measured during simulation
    '''
    # 0 - load config from SimInputData, build Delaunay graph and based on it
    # create incidence matrices, edges etc., save template of the simulation
    # and the configuration
    if SimInputData.load == 0:
        print('Load 0: building new network')
        sid = SimInputData()
        make_dir(sid)
        inc = In.Incidence()
        graph, edges, triangles = Ne.build_delaunay_net(sid, inc)
        Ne.set_geometry(sid, graph)
        In.create_matrices(sid, graph, inc, edges, triangles)
        vols = Volumes(sid, inc, edges, triangles)
        data = Data(sid, edges)
        Sv.save('/template.dill', sid, graph, inc, edges, triangles, vols)
        Sv.save_config(sid)
    # 1 - load config and network from data saved at the end of previous
    # simulation, from directory specified by load_name; based on that recreate
    # incidence and edges (with saved diameters), continue simulation
    elif SimInputData.load == 1:
        print(f'Load 1: continuing simulation from {SimInputData.load_name}')
        sid, graph, inc, edges, triangles, vols = Sv.load(SimInputData.load_name+'/save.dill')
        data = Data(sid, edges)
        #data.load_data()
    # 2 - load config from SimInputData, but use graph from a template saved in
    # the directory specified by load_name; based on that create incidence and
    # edges (with initial diameters), also update data in config corresponding
    # to the geometry of the graph; save simulation in the load_name directory,
    # but in an additional folder named template, save new config there
    elif SimInputData.load == 2:
        print(f'Load 2: new simulation from template {SimInputData.load_name}')
        sid = SimInputData()
        sid2, graph, inc, edges, triangles, vols \
            = Sv.load(SimInputData.load_name+'/template.dill')
        sid.Q_in = sid.qin * 2 * len(graph.in_nodes)
        sid.m = sid2.m
        sid.n = sid2.n
        sid.nsq = sid2.nsq
        sid.ne = sid2.ne
        sid.ntr = sid2.ntr
        sid.dirname = sid2.dirname + '/template'
        make_dir(sid)
        data = Data(sid, edges)
        Sv.save_config(sid)
    else:
        raise ValueError(f"Unknown load type: {SimInputData.load}")
    return sid, inc, graph, edges, vols, triangles, data
