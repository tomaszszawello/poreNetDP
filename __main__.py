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
import pressure_pump as Pr

from build import build
from save import save
from utils import initialize_iterators, update_iterators
from utils_vtk import save_VTK

import numpy as np
import scipy.sparse as spr

# initialize main classes
sid, inc, graph, edges, data = build()

iter_max, t_max, i, t, draw_i = initialize_iterators(sid)

# initialize vectors
pressure_b = Pr.create_vector(sid, graph)
cb_b, cc_b = Di.create_vector(sid, graph)

# main loop
# runs until we reach iteration limit or time limit
while t < t_max and i < iter_max:
    sid.q_rate = 1 + sid.q_amp * np.sin(2 * np.pi * t / sid.q_period)
    print(f'Iter {i + 1}/{iter_max} Time {t:.2f}/{t_max:.2f}')
    # find pressure and update flow in edges
    print ('Solving pressure')
    pressure = Pr.solve_flow(sid, inc, graph, edges, pressure_b)
    # find B  and C concentration
    print ('Solving concentration')
    if t == 0:
        cb, cc = Di.solve_dissolution_nr(sid, inc, graph, edges, cb_b, cc_b)
    elif sid.solve_type == "full":
        cb, cc = Di.solve_dissolution_nr(sid, inc, graph, edges, cb_b, cc_b)
    elif sid.solve_type == "simple":
        cb, cc = Di.solve_dissolution(sid, inc, graph, edges, cb_b, cc_b, cb, cc)
    #print(np.average(cb), np.average(cc))
    if sid.tracking_mode == 'time':
        if t // sid.plot_every > draw_i:
            print('Drawing')
            draw_i += 1
            cb_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ cb
            cc_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ cc
            Dr.draw(sid, graph, edges, f'q{t:05}.jpg', 'q')
            Dr.draw_c(sid, graph, edges, f'd{t:05}.jpg', 'd', cb_in, cc_in)
            #save_VTK(sid, graph, edges, pressure, cb, f'network_{t:.2f}.vtk')
            data.check_data(sid, edges, inc, cb, cc)
    # grow/shrink diameters and update them in edges, update volumes with
    # dissolved/precipitated values, check if network dissolved, find new
    # timestep
    print ('Updating diameters')
    breakthrough, dt_next = Gr.update_diameters(sid, inc, edges, cb, cc)

    # update physical parameters in data
    data.collect_data(sid, inc, edges, pressure, cb, cc)
    i, t = update_iterators(sid, i, t, dt_next)


# save data from the last iteration of simulation, save the whole simulation
# to be able to continue it later
if i != 1 and sid.load != 1:
    data.check_data(sid, edges, inc, cb, cc)
    Dr.draw(sid, graph, edges, f'q{t:05}.jpg', 'q')
    Dr.draw_c(sid, graph, edges, f'd{t:05}.jpg', 'd', cb_in, cc_in)
    #save_VTK(sid, graph, edges, pressure, cb, f'network_{t:.2f}.vtk')
    save("save.dill", sid, graph, inc, edges)