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

import clogging as Cl
import dissolution as Di
import draw_net as Dr
#import growth_dharm as Gr
import rectangle as Pr

from build import build
from save import save
from utils import initialize_iterators, update_iterators
from utils_vtk import save_VTK

import numpy as np
import scipy.sparse as spr

# initialize main classes
sid, inc, graph, edges, data = build()

iter_max, t_max, i, t, draw_i = initialize_iterators(sid)
track_i = 0

# initialize vectors
pressure_b = Pr.create_vector(sid, graph)
node_clogging = Cl.create_vector(sid, graph)
cb_b, cc_b = Di.create_vector(sid, graph)

# main loop
# runs until we reach iteration limit or time limit
breakthrough = False
while t < t_max and i < iter_max and not breakthrough:
    #sid.q_rate = 1 + sid.q_amp * np.sin(2 * np.pi * t / sid.q_period)
    sid.q_rate = sid.q_amp * (-1) ** (t // sid.q_period)
    # if t % sid.q_period < sid.q_trans:
    #     sid.q_rate = 0
    print(f'Iter {i + 1}/{iter_max} Time {t:.2f}/{t_max:.2f}')
    # find pressure and update flow in edges
    print ('Solving pressure')
    pressure = Pr.solve_flow(sid, inc, graph, edges, pressure_b)
    Q_in = np.sum(edges.inlet * np.abs(edges.flow))
    Q_out = np.sum(edges.outlet * np.abs(edges.flow))
    print('Q_in =', Q_in, 'Q_out =', Q_out)
    if np.abs(np.abs(Q_in) - np.abs(Q_out)) > 0.1:
        # Dr.draw(sid, graph, edges, f'q{t:05}.jpg', 'q')
        # Dr.draw(sid, graph, edges, f'd{t:05}.jpg', 'd')
        # zero_nodes = np.array(np.abs(inc.incidence).sum(axis = 0) == 1)[0] * (1 - graph.in_vec - graph.out_vec)
        # zero_edges = np.abs(inc.incidence) @ zero_nodes
        # print(np.sum(zero_edges))
        # import pressure as Pr2
        # pressure = Pr2.solve_flow(sid, inc, graph, edges, pressure_b)
        # Q_in = np.sum(edges.inlet * np.abs(edges.flow))
        # Q_out = np.sum(edges.outlet * np.abs(edges.flow))
        # print('Q_in =', Q_in, 'Q_out =', Q_out)
        # Dr.draw_nodes(sid, graph, edges, pressure, f'p{t:05}.jpg', 'q')
        # save("/save.dill", sid, graph, inc, edges)
        # Dr.draw_labels(sid, graph, edges, 'labels.png', 'd')
        # node_to_index = {node: idx for idx, node in enumerate(graph.nodes)}
        # print(node_to_index[(9, 3)], node_to_index[(9, 4)])
        # print(inc.incidence.tocsr()[155].nonzero())
        # print(inc.incidence.tocsr()[176].nonzero())
        raise ValueError('Flow not matching!')
    # find B  and C concentration
    print ('Solving concentration')
    if t == 0:
        cb, cc = Di.solve_dissolution_an(sid, inc, graph, edges, cb_b, cc_b)
    elif sid.solve_type == "full":
        cb, cc = Di.solve_dissolution_v2(sid, inc, graph, edges, cb_b, cc_b, cb, cc)
    elif sid.solve_type == "simple":
        cb, cc = Di.solve_dissolution(sid, inc, graph, edges, cb_b, cc_b, cb, cc)
    if np.sum(cb < -1e-3) or np.sum(cc < -1e-3):
        print(np.min(cb), np.min(cc))
        Dr.draw_nodes(sid, graph, edges, 1 * (cb < -1e-3) + (cc < -1e-3), f'cb{t:05}.jpg', 'q')
        raise ValueError('Negative concentration')
    print(np.sum(graph.out_vec * cb), np.sum(graph.out_vec * cc))
    #print(np.average(cb), np.average(cc))
    if sid.tracking_mode == 'time':
        if t // sid.plot_every > draw_i:
            print('Drawing')
            if t == 0:
                data.check_init_slice_channelization(sid, graph, inc, edges)
            if t // sid.track_every > track_i:
                data.check_slice_channelization(sid, graph, inc, edges, t)
                track_i += 1
            draw_i += 1
            cb_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ cb
            cc_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ cc
            Dr.draw(sid, graph, edges, f'q{t:05}.jpg', 'q')
            Dr.draw(sid, graph, edges, f'd{t:05}.jpg', 'd')
            Dr.draw_c(sid, graph, edges, f'color{t:05}.jpg', 'd', cb_in, cc_in)
            Dr.draw_c_product(sid, graph, edges, f'conc{t:05}.jpg', 'd', cb_in, cc_in)
            Dr.draw_conc(sid, graph, edges, f'cb{t:05}.jpg', 'd', cb_in)
            Dr.draw_conc(sid, graph, edges, f'cc{t:05}.jpg', 'd', cc_in)
            # Dr.draw_nodes(sid, graph, edges, cb, f'cb{t:05}.jpg', 'q')
            # Dr.draw_nodes(sid, graph, edges, cc, f'cc{t:05}.jpg', 'q')
            #save_VTK(sid, graph, edges, pressure, cb, f'network_{t:.2f}.vtk')
            #data.check_data(sid, edges, inc, cb, cc)
    # grow/shrink diameters and update them in edges, update volumes with
    # dissolved/precipitated values, check if network dissolved, find new
    # timestep
    print ('Updating diameters')
    breakthrough, dt_next, node_clogging = Cl.update_diameters(sid, inc, edges, graph, cb, cc, node_clogging)
    
    # update physical parameters in data
    data.collect_data(sid, inc, edges, graph, pressure, cb, cc)
    i, t = update_iterators(sid, i, t, dt_next)


# save data from the last iteration of simulation, save the whole simulation
# to be able to continue it later
if i != 1 and sid.load != 1:
    #data.check_data(sid, edges, inc, cb, cc)
    Dr.draw(sid, graph, edges, f'q{t:05}.jpg', 'q')
    Dr.draw(sid, graph, edges, f'd{t:05}.jpg', 'd')
    Dr.draw_c(sid, graph, edges, f'color{t:05}.jpg', 'd', cb_in, cc_in)
    #Dr.draw_c(sid, graph, edges, f'd{t:05}.jpg', 'd', cb_in, cc_in)
    #save_VTK(sid, graph, edges, pressure, cb, f'network_{t:.2f}.vtk')
    data.check_slice_channelization(sid, graph, inc, edges, t)
    data.plot_slice_channelization_v2(sid, graph)
    data.plot_pressure()
    data.plot_pressure_diff()
    save("/save.dill", sid, graph, inc, edges)
