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

import delaunay as De
import network_pump2 as Ne
import incidence as In
import save as Sv

from config import SimInputData
from data import Data
from utils import make_dir


def build() -> tuple[SimInputData, In.Incidence, De.Graph, In.Edges, Data]:
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
    sid = SimInputData()
    # 0 - load config from SimInputData, build Delaunay graph and based on it
    # create incidence matrices, edges etc., save template of the simulation
    # and the configuration
    if sid.load == 0:
        print('load 0: building a network')
        #sid = SimInputData()
        make_dir(sid)
        inc = In.Incidence()
        graph, edges = Ne.build_delaunay_net(sid, inc)
        Ne.set_geometry(sid, graph)
        In.create_matrices(sid, graph, inc, edges)
        data = Data(sid, edges)
        Sv.save('/template.dill', sid, graph, inc, edges)
        Sv.save_config(sid)
    # 1 - load config and network from data saved at the end of previous
    # simulation, from directory specified by load_name; based on that recreate
    # incidence and edges (with saved diameters), continue simulation
    elif sid.load == 1:
        print(f'load 1: loading a network from {sid.load_name}')
        sid, graph, inc, edges = Sv.load(sid.load_name+'/save.dill')
        data = Data(sid, edges)
        data.load_data()
    # 2 - load config from SimInputData, but use graph from a template saved in
    # the directory specified by load_name; based on that create incidence and
    # edges (with initial diameters), also update data in config corresponding
    # to the geometry of the graph; save simulation in the load_name directory,
    # but in an additional folder named template, save new config there
    elif sid.load == 2:
        print(f'load 2: loading a network from template {sid.load_name}')
        #sid = SimInputData()
        sid2, graph, inc, edges \
            = Sv.load(SimInputData.load_name+'/template.dill')
        sid.Q_in = sid.qin * 2 * len(graph.in_nodes)
        sid.n = sid2.n
        sid.nsq = sid2.nsq
        sid.ne = sid2.ne
        sid.dirname = sid2.dirname + '/template'
        make_dir(sid)
        data = Data(sid, edges)
        Sv.save_config(sid)
    else:
        raise ValueError(f'load type unknown: {sid.load}')
    return sid, inc, graph, edges, data
