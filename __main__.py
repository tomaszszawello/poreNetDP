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
import diffusion as Dif
import draw_net as Dr
import growth as Gr
import merging as Me
import precipitation as Pi
import pressure as Pr
import save as Sv
import tracking as Tr

from build import build
from utils import initialize_iterators, update_iterators
from utils_vtk import save_VTK



# initialize main classes
sid, inc, graph, edges, vols, triangles, data = build()

iters, tmax, i, t, breakthrough = initialize_iterators(sid)
iterator_dissolved = 0
clogged = False


import numpy as np


flag_s = 1


# main loop
# runs until we reach iteration limit or time limit or network is dissolved
while t < tmax and i < iters and not breakthrough and not clogged:
    print((f'Iter {i + 1}/{iters} Time {t:.2f}/{tmax:.2f}'))
    print(np.max(edges.diams))
    # initialize vectors
    pressure_b = Pr.create_vector(sid, graph)
    cb_b = Di.create_vector(sid, graph)

    # find pressure and update flow in edges
    print ('Solving pressure')
    pressure = Pr.solve_flow(sid, inc, graph, edges, pressure_b)

    # if t == 0:
    #     q_in = np.abs(np.sum(edges.diams ** 4 / edges.lens * (inc.inlet \
    #         @ pressure)))
    #     pressure *= sid.Q_in / q_in
    #     # update flow
    #     edges.flow = edges.diams ** 4 / edges.lens * (inc.incidence @ pressure)
    #     sid.p0 = np.max(pressure)
    #     pressure_b = Pr.create_vector(sid, graph)

    Q_in = np.sum(edges.inlet * np.abs(edges.flow))
    Q_out = np.sum(edges.outlet * np.abs(edges.flow))
    print('Q_in =', Q_in, 'Q_out =', Q_out, 'p_in = ', np.max(pressure))
    # find B concentration
    print ('Solving concentration')
    # find C concentration
    if sid.include_volumes:
        cb = Di.solve_dissolution_safe(sid, inc, graph, edges, vols, cb_b)
    else:
        cb = Di.solve_dissolution(sid, inc, graph, edges, cb_b)
    print('cb: ', np.min(cb), np.max(cb))
    if np.max(cb) > 1.1:
        print(cb)
        print(pressure)
        raise ValueError
    if sid.include_precipitation:
        if t == 0 or sid.load == 1:
            cc, cd = Pi.create_vector_nr(sid, graph, inc, edges, cb)
        cc, cd = Pi.solve_precipitation_safe(sid, inc, graph, edges, vols, cb, cc, cd)
    else:
        cc, cd = np.zeros(sid.nsq), np.zeros(sid.nsq)
    if np.max(cc) == -1 and i != 0:
        Sv.save('/save.dill', sid, graph, inc, edges, triangles, vols)
        raise ValueError('NR didnt converge')

    print(f'alpha_b: {np.sum(edges.alpha_b == 0)}, alpha_c: {np.sum(edges.alpha_c == 0)}')
    print(f'cc: {np.min(cc)}, {np.max(cc)}, cd: {np.min(cd)}, {np.max(cd)}')
    #cc = Pi.solve_precipitation(sid, inc, graph, edges, cb)
    # calculate ffp, draw figures
    if t == 0 and not sid.debug:
        int_part  = int(t)
        frac_part = int(100 * (t - int_part))
        name = f"{int_part:04d}_{frac_part:02d}.jpg"
        data.check_data(edges)
        data.check_mass_balance(sid, inc, edges, vols, cb, cc, cd)
        data.check_slice_porosity(graph, inc, edges, triangles, vols, int(t))
        data.check_init_slice_channelization(graph, inc, edges)
        data.check_slice_channelization(graph, inc, edges, t)
        #Tr.track(sid, graph, inc, edges, data, pressure)
        Dr.draw_flow(sid, graph, edges, f'q_' + name, 'q')
        #Dr.draw_flow(sid, graph, edges, f'd_{data.dissolved_v:.2f}.jpg', 'd')
        Dr.draw_triangles(sid, triangles, edges, graph, vols, f'tri_' + name)
        Dr.uniform_hist(sid, graph, edges, vols, cb, f'dreal_' + name, 'd')
        #Dr.draw_colored_edges(sid, graph, edges, vols, f'dc_{data.dissolved_v:.2f}.jpg')
        #Dr.draw_nodes(sid, graph, edges, cb, f'c_{data.dissolved_v:.2f}.jpg', 'q')
        # save_VTK(sid, graph, edges, pressure, cb, \
        #     f'network_{data.dissolved_v:.2f}.vtk')
    else:
        #if data.dissolved_v // sid.track_every > iterator_dissolved:
        if t // sid.track_every > iterator_dissolved:
            print('Drawing')
            iterator_dissolved += 1
            int_part  = int(t)
            frac_part = int(100 * (t - int_part))
            name = f"{int_part:04d}_{frac_part:02d}.jpg"
            # if iterator_dissolved in sid.track_list:
            Dr.draw_flow(sid, graph, edges, \
                f'q_' + name, 'q')
            #Dr.draw_flow(sid, graph, edges, f'd_{data.dissolved_v:.2f}.jpg', 'd')
            Dr.draw_triangles(sid, triangles, edges, graph, vols, f'tri_' + name)
            Dr.uniform_hist(sid, graph, edges, vols, cb, f'dreal_' + name, 'd')
            #Dr.draw_colored_edges(sid, graph, edges, vols, f'dc_{data.dissolved_v:.2f}.jpg')
            #Dr.draw_nodes(sid, graph, edges, cb, f'c_{data.dissolved_v:.2f}.jpg', 'q')
            # save_VTK(sid, graph, edges, pressure, cb, \
            #     f'network_{data.dissolved_v:.2f}.vtk')
            if iterator_dissolved in sid.track_list:
                data.check_data(edges)
                data.check_mass_balance(sid, inc, edges, vols, cb, cc, cd)               
                data.check_slice_porosity(graph, inc, edges, triangles, vols, int(t))
                data.check_slice_channelization(graph, inc, edges, \
                    data.dissolved_v)
            data.save_data()
            #if iterator_dissolved == 3:
            #    Sv.save('/save.dill', sid, graph, inc, edges, triangles, vols)
            #Tr.track(sid, graph, inc, edges, data, pressure)
    # grow/shrink diameters and update them in edges, update volumes with
    # dissolved/precipitated values, check if network dissolved, find new
    # timestep
    print('Updating diameters')
    breakthrough, dt_next = Gr.update_diameters(sid, inc, edges, graph, vols, cb, cc, cd, data)
    # if breakthrough:
    #     break
    #data.check_mass_balance(sid, inc, edges, vols, cb, cc, cd, verbose=False, tol=0.5)
    data.collect_data(sid, inc, edges, vols, pressure, cb, cc, cd)
    if np.max(pressure) > data.pressure[0] / sid.min_perm:
        print('Network clogged.')
        clogged = True
    # merge edges
    if sid.include_merging:
        print ('Merging')
        if sid.include_volumes:
            Me.solve_merging_vols(sid, inc, graph, vols, triangles, edges)
        else:
            try:
                Me.solve_merging(sid, inc, graph, edges)
            except:
                Dr.draw_nodes(sid, graph, edges, cb, f'c_{data.dissolved_v:.2f}.jpg', 'q')
                raise ValueError
            #Me.fix_connections(sid, inc, graph, edges)
    # update physical parameters in data

    i, t = update_iterators(sid, i, t, dt_next)
    if np.abs(Q_in - Q_out) > 1:
        #Dr.draw_nodes(sid, graph, edges, cb, f'c_{data.dissolved_v:.2f}.jpg', 'q')
        #Sv.save('/save.dill', sid, graph, inc, edges)
        Dr.draw_flow(sid, graph, edges, f'd_{data.dissolved_v:.2f}.jpg', 'd')
        raise ValueError('Flow not matching!')

    # if i == 1110:
    #     Sv.save('/save.dill', sid, graph, inc, edges, triangles, vols)
    # if np.sum((np.array((inc.merge != 0).sum(axis = 0))[0] == 0) * (edges.diams != 0)):

    #     print((inc.merge != 0).sum(axis = 0))
    #     print(edges.diams)
    #     raise ValueError
#Dr.draw_nodes(sid, graph, edges, cb, f'c2_{data.dissolved_v:.2f}.jpg', 'q')
# save data from the last iteration of simulation, save the whole simulation
# to be able to continue it later

Dr.draw_triangles(sid, triangles, edges, graph, vols, f'tri_final.png')

if i != 1 and sid.load != 1 and not sid.debug:
    #data.check_data(edges)
    data.check_slice_channelization(graph, inc, edges, data.dissolved_v)
    #Tr.track(sid, graph, inc, edges, data, pressure)
    # Dr.draw_diams_profile(sid, graph, edges, data, \
    #     f'focusing_d_{data.dissolved_v:.2f}.jpg', 'd')
    # save_VTK(sid, graph, edges, pressure, cb, \
    #     f'network_{data.dissolved_v:.2f}.vtk')
    if clogged == True:
        sid.qdrawconst = 0
    data.save_data()
    data.plot_things(sid)
    data.check_mass_balance(sid, inc, edges, vols, cb, cc, cd)       
    data.check_slice_porosity(graph, inc, edges, triangles, vols, int(t))
    data.plot_vol_profile(graph)
    data.plot_profile(graph)
    #Tr.plot_tracking(data, 100)
    # Dr.draw_flow_profile(sid, graph, edges, data, \
    #         f'focusing_q_{data.dissolved_v:.2f}.jpg', 'q')
    int_part  = int(t)
    frac_part = int(100 * (t - int_part))
    name = f"{int_part:04d}_{frac_part:02d}.jpg"
    Dr.draw_flow(sid, graph, edges, f'q_' + name, 'q')
    #Dr.draw_flow(sid, graph, edges, f'd_{data.dissolved_v:.2f}.jpg', 'd')
    Dr.draw_triangles(sid, triangles, edges, graph, vols, f'tri_' + name)
    Dr.uniform_hist(sid, graph, edges, vols, cb, f'dreal_' + name, 'd')
    #Dr.draw_colored_edges(sid, graph, edges, vols, f'dc_{data.dissolved_v:.2f}.jpg')
    #Dr.draw_nodes(sid, graph, edges, cb, f'c_{data.dissolved_v:.2f}.jpg', 'q')
    Sv.save('/save.dill', sid, graph, inc, edges, triangles, vols)
