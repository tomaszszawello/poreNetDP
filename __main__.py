#!/usr/bin/env python3
""" Start simulation based on parameters from config.

This module performs the whole simulation. It should be started after all
parameters in config file are set (most importantly n - network size,
iters/tmax - simulation length, Da_eff, G, K, Gamma - dissolution/precipitation
parameters, include_cc - turn on precipitation, load - build a new network
or load a previous one). After starting, directory consisting of
geometry + network size / G + Damkohler number / simulation index
will be created. Plots of the network and other data will be saved there.
"""

import dissolution as Di
import draw_net as Dr
import growth as Gr
import merging as Me
import precipitation as Pi
import pressure as Pr
import save as Sv

from build import build
from utils import initialize_iterators, update_iterators
from utils_vtk import save_VTK

# initialize main classes
sid, inc, graph, edges, data = build()

iters, tmax, i, t, breakthrough = initialize_iterators(sid)
iterator_dissolved = 0


# initial merging
if sid.include_merging:
    for initial_i in range(sid.initial_merging):
        Me.solve_merging(sid, inc, graph, edges, 'initial')

import numpy as np
from network import Edges

# main loop
# runs until we reach iteration limit or time limit or network is dissolved
while t < tmax and i < iters and data.dissolved_v < sid.dissolved_v_max:
    print((f'Iter {i + 1}/{iters} Time {t:.2f}/{tmax:.2f} \
        Dissolved {data.dissolved_v:.2f}/{sid.dissolved_v_max:.2f}'))
    # initialize vectors
    pressure_b = Pr.create_vector(sid, graph)
    cb_b = Di.create_vector(sid, graph)
    # find pressure and update flow in edges
    print ('Solving pressure')
    pressure = Pr.solve_flow(sid, inc, graph, edges, pressure_b)
    
    # find B concentration
    print ('Solving concentration')
    cb = Di.solve_dissolution(sid, inc, graph, edges, cb_b)
    # find C concentration
    cc = Pi.solve_precipitation(sid, inc, graph, edges, cb)
    # calculate ffp, draw figures
    if t == 0:
        data.check_data(edges)
        data.check_init_slice_channelization(graph, inc, edges)
        data.check_slice_channelization(graph, inc, edges, t)
        Dr.draw_flow(sid, graph, edges, f'q_{data.dissolved_v:.2f}.jpg', 'q')
        Dr.draw_flow(sid, graph, edges, f'd_{data.dissolved_v:.2f}.jpg', 'd')
        save_VTK(sid, graph, edges, pressure, cb, \
            f'network_{data.dissolved_v:.2f}.vtk')
    else:
        if data.dissolved_v // sid.track_every > iterator_dissolved:
            print('Drawing')
            iterator_dissolved += 1
            if iterator_dissolved in sid.track_list:
                Dr.draw_flow(sid, graph, edges, \
                    f'q_{data.dissolved_v:.2f}.jpg', 'q')
                save_VTK(sid, graph, edges, pressure, cb, \
                    f'network_{data.dissolved_v:.2f}.vtk')
                data.check_data(edges)
                data.check_slice_channelization(graph, inc, edges, \
                    data.dissolved_v)
    # grow/shrink diameters and update them in edges, update volumes with
    # dissolved/precipitated values, check if network dissolved, find new
    # timestep
    print ('Updating diameters')
    breakthrough, dt_next = Gr.update_diameters(sid, inc, edges, cb, cc)
    # merge edges
    if sid.include_merging:
        Me.solve_merging(sid, inc, graph, edges)
    # update physical parameters in data
    data.collect_data(sid, inc, edges, pressure, cb, cc)
    i, t = update_iterators(sid, i, t, dt_next)


# save data from the last iteration of simulation, save the whole simulation
# to be able to continue it later
if i != 1 and sid.load != 1:
    data.check_data(edges)
    data.check_slice_channelization(graph, inc, edges, data.dissolved_v)
    #data.plot_slice_channelization_v2(sid, graph)
    Dr.draw_flow_profile(sid, graph, edges, data, \
        f'focusing_q_{data.dissolved_v:.2f}.jpg', 'q')
    Dr.draw_diams_profile(sid, graph, edges, data, \
        f'focusing_d_{data.dissolved_v:.2f}.jpg', 'd')
    save_VTK(sid, graph, edges, pressure, cb, \
        f'network_{data.dissolved_v:.2f}.vtk')
    Sv.save('/save.dill', sid, graph, inc, edges)
    data.save_data()
