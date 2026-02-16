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
import concentration_prop as Cp
#import concentration_prop_node as Cp
import dissolution as Di
import draw_net as Dr
import growth as Gr
import mixing as Mi
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
track_i = 0

# initialize vectors
pressure_b = Pr.create_vector(sid, graph)
cb_b, cc_b = Di.create_vector(sid, graph)
node_clogging = Cl.create_vector(sid, graph)
data.vol_init += 2 / 3 * np.sum(node_clogging ** 3)

inlet_a = 1. * (np.abs(inc.incidence) @ graph.in_vec_a > 0)
inlet_b = 1. * (np.abs(inc.incidence) @ graph.in_vec_b > 0)
print(np.max(inlet_a))

# main loop
# runs until we reach iteration limit or time limit
breakthrough = False
while t < t_max and i < iter_max and not breakthrough:
    # if t > t_max / 4 and flag == 0:
    #     sid.q_amp += 0.048
    #     flag = 1
    # if t > t_max / 2 and flag == 1:
    #     sid.q_amp += 0.048
    #     flag = 2
    # if t > 3 * t_max / 4 and flag == 2:
    #     sid.q_amp += 0.048
    #     flag = 3
    #sid.q_rate = 1 + sid.q_amp * np.sin(2 * np.pi * t / sid.q_period)
    sid.q_rate = sid.q_amp * (-1) ** (t // sid.q_period)
    # if t % sid.q_period < sid.q_trans:
    #     sid.q_rate = 0
    print(f'Iter {i + 1}/{iter_max} Time {t:.2f}/{t_max:.2f}')
    # find pressure and update flow in edges
    print ('Solving pressure')
    pressure = Pr.solve_flow_nodes(sid, inc, graph, edges, data, pressure_b, node_clogging)
    Q_in = np.sum(edges.inlet * np.abs(edges.flow))
    Q_out = np.sum(edges.outlet * np.abs(edges.flow))
    print('Q_in =', Q_in, 'Q_out =', Q_out)
    # if np.abs(np.abs(Q_in) - np.abs(Q_out)) > 0.1 or np.isnan(Q_in):
    #     raise ValueError('Flow not matching!')
    # find B  and C concentration
    print ('Solving concentration')
    # if t == 0:
    #     cb, cc = Di.solve_dissolution_an(sid, inc, graph, edges, cb_b, cc_b)
    # elif sid.solve_type == "full":
    #     cb, cc = Di.solve_dissolution_v2(sid, inc, graph, edges, cb_b, cc_b, cb, cc)
    # elif sid.solve_type == "simple":
    #     cb, cc = Di.solve_dissolution(sid, inc, graph, edges, cb_b, cc_b, cb, cc)
    #alpha_matrix = Mi.find_alpha(sid, edges, inc)
    alpha_eff, alpha_full, alpha_stream, w = Mi.find_alpha(sid, edges, inc, node_clogging)
    print('w average: ', np.average(w))
    #cb, cc = Di.solve_dissolution_mixing(sid, edges, inc, graph, alpha_matrix)
    # if t == 0:
    #     cb, cc = Di.solve_dissolution_edges_with_alpha(sid, edges, alpha_matrix, inlet_a, inlet_b, edges.inlet)
    # else:
    #     cb, cc = Di.solve_dissolution_edges_with_alpha(sid, edges, alpha_matrix, cb, cc, edges.inlet)
    
    #cb, cc = Cp.propagate_chemistry(alpha_matrix.T, edges.flow, inc.incidence, edges.inlet, inlet_a, inlet_b)
    cb, cc, prec_node, prec_edge = Cp.propagate_tubes_balls_blend(inc.incidence, edges.flow, edges.diams, edges.lens, node_clogging, edges.inlet, inlet_a, inlet_b, alpha_stream.T, alpha_full.T, w, sid.Da_eff, sid.G, sid.Ksp, sid.chi0)
    # edges.diams -= prec_edge
    # node_clogging -= prec_node



    if np.sum(cb < -1e-3) or np.sum(cc < -1e-3):
        print(np.min(cb), np.min(cc))
        Dr.draw_nodes(sid, graph, edges, 1 * (cb < -1e-3) + (cc < -1e-3), f'cb{t:05}.jpg', 'q')
        raise ValueError('Negative concentration')
    #print(np.sum(edges.outlet * cb), np.sum(edges.outlet * cc))
    #print('cb: ', np.min(cb), np.max(cb), np.average(cb), ' cc: ', np.min(cc), np.max(cc), np.average(cc))
    #print(cb[edges.special[0]], cb[edges.special[1]], cc[edges.special[0]], cc[edges.special[1]])
    
    #print(np.array(np.sum(alpha_matrix, axis = 1))[:, 0])
    #alpha_matrix = (alpha_matrix > 0) * 0.5
    # if t == 0:
    
    #     f_eff = Cp.compute_f_eff(edges, node_clogging, sid.Ksp, cb, cc)
    #     print(f_eff)
    #     sid.tmax /= f_eff
    #     sid.q_period = sid.tmax / 24
    #     sid.plot_every = sid.tmax / 24
    #     t_max = sid.tmax

    if sid.tracking_mode == 'time':
        if t // sid.plot_every > draw_i:
            print('Drawing')
            if t == 0:
                data.check_init_slice_channelization(sid, graph, inc, edges)
                edges.c = Mi.find_concentration(sid, edges, inc, graph, alpha_eff)
                Dr.draw(sid, graph, edges, node_clogging, f'c{t:05}.jpg', 'c')
                edges.cb = cb
                Dr.draw(sid, graph, edges, node_clogging, f'cb_init{t:05}.jpg', 'cb')
                #Dr.draw_nodes(sid, graph, edges, graph.vol_nodes, f'nodevol{t:05}.jpg', 'd')
                data.compare_conc(sid, inc, graph, cb)
            if t // sid.track_every > track_i:
                data.check_slice_channelization(sid, graph, inc, edges, t)
                track_i += 1
            draw_i += 1

            
            #np.savetxt("c.txt", edges.c)
            Dr.draw(sid, graph, edges, node_clogging, f'q{t:05}.jpg', 'q')
            Dr.draw(sid, graph, edges, node_clogging, f'd{t:05}.jpg', 'd')
            #Dr.draw(sid, graph, edges, f'k{t:05}.jpg', 'k')
            Dr.draw_d_diff(sid, graph, edges, f'd_diff{t:05}.jpg')
            Dr.draw_c(sid, graph, edges, f'color{t:05}.jpg', 'd', cb, cc)
            #Dr.draw_nodes(sid, graph, edges, graph.vol_nodes, f'nodevol{t:05}.jpg', 'd')
            # Dr.draw_nodes(sid, graph, edges, cb, f'cb{t:05}.jpg', 'q')
            # Dr.draw_nodes(sid, graph, edges, cc, f'cc{t:05}.jpg', 'q')
            #save_VTK(sid, graph, edges, pressure, cb, f'network_{t:.2f}.vtk')
            #data.check_data(sid, edges, inc, cb, cc)
    # grow/shrink diameters and update them in edges, update volumes with
    # dissolved/precipitated values, check if network dissolved, find new
    # timestep
    print ('Updating diameters')
    #breakthrough, dt_next = Gr.update_diameters(sid, inc, edges, graph, cb, cc)
    #breakthrough, dt_next, node_clogging = Cl.update_diameters(sid, inc, edges, graph, cb, cc, node_clogging)
    breakthrough, dt_next, node_clogging = Cp.update_diameters(sid, inc, edges, graph, prec_node, prec_edge, node_clogging)
    print(np.average(edges.diams))
    # update physical parameters in data
    data.collect_data(sid, inc, edges, graph, pressure, cb, cc, node_clogging)
    i, t = update_iterators(sid, i, t, dt_next)


# save data from the last iteration of simulation, save the whole simulation
# to be able to continue it later
if i != 1 and sid.load != 1:
    #data.check_data(sid, edges, inc, cb, cc)
    Dr.draw(sid, graph, edges, node_clogging, f'q{t:05}.jpg', 'q')
    Dr.draw(sid, graph, edges, node_clogging, f'd{t:05}.jpg', 'd')
    Dr.draw_d_diff(sid, graph, edges, f'd_diff{t:05}.jpg')
    Dr.draw_c(sid, graph, edges, f'color{t:05}.jpg', 'd', cb, cc)
    #Dr.draw_nodes(sid, graph, edges, graph.vol_nodes, f'nodevol{t:05}.jpg', 'd')
    #Dr.draw_c(sid, graph, edges, f'd{t:05}.jpg', 'd', cb_in, cc_in)
    #save_VTK(sid, graph, edges, pressure, cb, f'network_{t:.2f}.vtk')
    data.check_slice_channelization(sid, graph, inc, edges, t)
    data.plot_slice_channelization_v2(sid, graph)
    data.plot_pressure()
    data.plot_pressure_diff()
    data.plot_precipitate()
    data.plot_conductivity()
    data.plot_interface_width()
    save("/save.dill", sid, graph, inc, edges)
