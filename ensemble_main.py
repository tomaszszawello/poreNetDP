#!/usr/bin/env python3
""" Start simulation based on parameters from config.

This module performs the whole simulation. It should be started after all
parameters in config file are set (most importantly load_name - loaded network,
iters/tmax - simulation length, Da_eff, G - dissolution parameters.
After starting, directory consisting of network name / G + Damkohler number
/ simulation index will be created. Outputs of the network and other data will
be saved there.
"""
import numpy as np

import dissolution as Di
import growth as Gr
import pressure as Pr

from build import build
from config import SimInputData
from utils import initialize_iterators, update_iterators, make_dir
from utils_vtk import save_vtk, save_vtk_nodes
# sid = SimInputData()
# sid.load_name = f'carbonate_x01'
# sid.dirname = f'check/G{sid.G:.2f}Daeff{sid.Da_eff:.2f}/{sid.load_name}'
# graph, graph_real, inc, edges, data = build(sid)

#from copy import deepcopy

for sample in range(0, 30):
    print(f'carbonate_x{sample+1:02}')
    # initialize simulation data from config

    #sid = deepcopy(sid2)
    sid = SimInputData()
    sid.load_name = f'carbonate_x{sample+1:02}'
    sid.dirname = f'check_new_Da/G{sid.G:.5f}Daeff{sid.Da_eff:.5f}/{sid.load_name}'
    make_dir(sid)
    sid.old_t = 0
    # initialize main classes
    #graph2, graph_real2, inc2, edges2, data2 = build(sid)
    #graph, graph_real, inc, edges, data = deepcopy(graph2), deepcopy(graph_real2), deepcopy(inc2), deepcopy(edges2), deepcopy(data2)
    graph, graph_real, inc, edges, data = build(sid)
    # initialize constant vectors (result vectors for matrix equations)
    pressure_b = Pr.create_result_vector(sid, graph)
    concentration_b = Di.create_result_vector(sid, graph)
    # initialize variable vectors
    pressure = Pr.create_vector(sid)
    concentration = Di.create_vector(sid)
    # initialize iterators
    iters, tmax, i, t, breakthrough = initialize_iterators(sid)
    vol_init = np.sum(edges.apertures * edges.fracture_lens * edges.lens)
    dissolved_v = 0
    iterator_dissolved = 0
    t = 0
    data.slices = []
    data.slice_times = []
    data.dirname = sid.dirname
    # main loop
    # runs until we reach iteration limi5t or time limit or network is dissolved
    while t < tmax and i < iters and not breakthrough and dissolved_v < sid.dissolved_v_max:
        print(f'Iter {i + 1}/{iters} Time {t:.2f}/{tmax:.2f}, Dissolved {dissolved_v:.2f}/{sid.dissolved_v_max:.2f}')
        # find pressure and update flow in edges
        #print('Solving pressure')
        pressure = Pr.solve_flow(sid, inc, graph, edges, pressure_b)
        # find solv concentration
        #print("Solving concentration")
        concentration = Di.solve_dissolution(sid, inc, graph, edges, \
            concentration_b)
        # update physical parameters in data
        if i % sid.collect_every == 0:
            data.collect(sid, graph, inc, edges, pressure)
        # calculate slice channelization
        if t == 0:
            print('0 save')
            #data.check_init_slice_channelization(graph, inc, edges)
            #data.check_slice_channelization(graph, inc, edges, dissolved_v)
            graph_real.dump_json_graph(sid, edges)
        # elif sid.old_t // sid.track_every != (sid.old_t + sid.dt) \
        #     // sid.track_every:
        elif dissolved_v // sid.track_every > iterator_dissolved:
            iterator_dissolved += 1
            if iterator_dissolved in sid.track_list:
                #data.check_slice_channelization(graph, inc, edges, dissolved_v)
                graph_real.dump_json_graph(sid, edges)
        # save network in JSON
        if i % sid.save_every == 0:
            data.check(edges)
            # save_vtk(sid, graph, edges, pressure, concentration)
            # graph_real.dump_json_graph(sid, edges)
        # grow apertures and update them in edges, check if network dissolved,
        # find new timestep
        breakthrough, dt = Gr.update_apertures(sid, inc, edges, concentration)
        i, t = update_iterators(sid, i, t, dt)
        dissolved_v = (np.sum(edges.apertures * edges.fracture_lens * edges.lens) - vol_init) / vol_init
        sid.dissolved_v = dissolved_v
    # save network and data from the last iteration of simulation and plot them
    #if i != 1:
    data.check(edges)
    #graph_real.dump_json_graph(sid, edges)
    #save_vtk(sid, graph, edges, pressure, concentration)
    #save_vtk_nodes(sid, graph)
    data.collect(sid, graph, inc, edges, pressure)
    #data.check_slice_channelization(graph, inc, edges, dissolved_v)
    graph_real.dump_json_graph(sid, edges)
    #data.save()
    #data.plot_slice_channelization(graph)
    del sid, graph, graph_real, inc, edges, data, pressure_b, concentration_b, pressure, concentration