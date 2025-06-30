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
from utils import initialize_iterators, update_iterators
from utils_vtk import save_vtk, save_vtk_nodes


# initialize simulation data from config
sid = SimInputData()
# initialize main classes
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

iterator_dissolved = 0
# main loop
# runs until we reach iteration limi5t or time limit or network is dissolved
while t < tmax and i < iters and not breakthrough and sid.dissolved_v < sid.dissolved_v_max:
    print(f'Iter {i + 1}/{iters} Time {t:.2f}/{tmax:.2f}, Dissolved {sid.dissolved_v:.2f}/{sid.dissolved_v_max:.2f}')
    # find pressure and update flow in edges
    pressure = Pr.solve_flow(sid, inc, graph, edges, pressure_b)
    # find solv concentration
    concentration = Di.solve_dissolution(sid, inc, graph, edges, \
        concentration_b)
    # update physical parameters in data
    if i % sid.collect_every == 0:
        data.collect(sid, graph, inc, edges, pressure)
    # calculate slice channelization
    if t == 0:
        data.collect_initial_data(sid, graph, inc, edges, concentration)
        #data.collect_data(sid, graph, inc, edges)
        #graph_real.dump_json_graph(sid, edges)
        save_vtk(sid, graph, edges, pressure, concentration)
    # elif sid.old_t // sid.track_every != (sid.old_t + sid.dt) \
    #     // sid.track_every:
    elif sid.dissolved_v // sid.track_every > iterator_dissolved:
        iterator_dissolved += 1
        if iterator_dissolved in sid.track_list:
            data.collect_data(sid, graph, inc, edges, concentration)
            #graph_real.dump_json_graph(sid, edges)
            save_vtk(sid, graph, edges, pressure, concentration)
    # save network in JSON
    if i % sid.save_every == 0:
        data.check(edges)
        # graph_real.dump_json_graph(sid, edges)
    # grow apertures and update them in edges, check if network dissolved,
    # find new timestep
    breakthrough, dt = Gr.update_apertures(sid, inc, edges, concentration)
    i, t = update_iterators(sid, i, t, dt)
    sid.dissolved_v = (np.sum(edges.apertures * edges.fracture_lens * edges.lens) - vol_init) / vol_init
# save network and data from the last iteration of simulation and plot them
if i != 1:
    data.check(edges)
    #graph_real.dump_json_graph(sid, edges)
    #save_vtk_nodes(sid, graph)
    data.collect(sid, graph, inc, edges, pressure)
    data.collect_data(sid, graph, inc, edges, concentration)
    data.plot_data(graph)
    graph_real.dump_json_graph(sid, edges)
    save_vtk(sid, graph, edges, pressure, concentration)
    data.save()
