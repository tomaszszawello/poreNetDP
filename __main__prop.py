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
import numpy as np
import scipy.sparse as spr

import conc_prop_vol as Cp
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

cA_in = Cp.create_vector_cA(sid, edges)
cB_in = Cp.create_vector_cB(sid, edges)

print(sid.ne)
# main loop
# runs until we reach iteration limit or time limit or network is dissolved
while t < tmax and i < iters and data.dissolved_v < sid.dissolved_v_max and not breakthrough and not clogged:
    print((f'Iter {i + 1}/{iters} Time {t:.2f}/{tmax:.2f} \
        Dissolved {data.dissolved_v:.2f}/{sid.dissolved_v_max:.2f}'))
    # initialize vectors
    pressure_b = Pr.create_vector(sid, graph)
    # find pressure and update flow in edges
    print ('Solving pressure')
    pressure = Pr.solve_flow(sid, inc, graph, edges, pressure_b)
    #data.check_data(edges)
    Q_in = np.sum(edges.inlet * np.abs(edges.flow))
    Q_out = np.sum(edges.outlet * np.abs(edges.flow))
    print('Q_in =', Q_in, 'Q_out =', Q_out, 'p_in = ', np.max(pressure))

    # find B concentration
    print ('Solving concentration')
    cA_out, cB_out, cA_node, cB_node, diss_edge_rate, precip_edge_rate, vol_a, vol_e = Cp.propagate_tubes_balls_blend(inc.incidence,
    edges.flow, edges.diams, edges.lens,
    edges.inlet, cA_in, cB_in,
    sid.Da, sid.G, sid.Ksp, sid.K * sid.Da,
    vols.triangles, vols.vol_a, vols.vol_e, vols.vol_max, sid.dt, sid.Gamma)
    inlet = edges.inlet.astype(bool)
    vols.vol_a = vol_a
    vols.vol_e = vol_e

    print("max B_out:", float(np.max(cB_out)))
    print("max B_out on inlet edges:", float(np.max(cB_out[inlet])))
    print("max B_in on inlet edges:", float(np.max(cB_in[inlet])))

    print("max A_out:", float(np.max(cA_out)))
    print("max A_out on inlet edges:", float(np.max(cA_out[inlet])))
    print("max A_in on inlet edges:", float(np.max(cA_in[inlet])))
    # calculate ffp, draw figures
    if t == 0 and not sid.debug:
        int_part  = int(t)
        frac_part = int(100 * (t - int_part))
        name = f"{int_part:04d}_{frac_part:02d}.jpg"
        data.check_data(edges)
        data.check_slice_porosity(graph, inc, edges, triangles, vols, int(t))
        data.check_init_slice_channelization(graph, inc, edges)
        data.check_slice_channelization(graph, inc, edges, t)
        #Tr.track(sid, graph, inc, edges, data, pressure)
        Dr.draw_flow(sid, graph, edges, f'q_' + name, 'q')
        #Dr.draw_flow(sid, graph, edges, f'd_{data.dissolved_v:.2f}.jpg', 'd')
        Dr.draw_triangles(sid, triangles, edges, graph, vols, f'tri_' + name)
        Dr.uniform_hist(sid, graph, edges, vols, cA_out, f'dreal_' + name, 'd')
        #Dr.draw_colored_edges(sid, graph, edges, vols, f'dc_{data.dissolved_v:.2f}.jpg')
        Dr.draw_nodes(sid, graph, edges, cA_node, f'cA_' + name, 'q')
        Dr.draw_nodes(sid, graph, edges, cB_node, f'cB_' + name, 'q')
        # save_VTK(sid, graph, edges, pressure, cb, \
        #     f'network_{data.dissolved_v:.2f}.vtk')
    else:
        #if data.dissolved_v // sid.track_every > iterator_dissolved:
        if t // sid.track_every > iterator_dissolved and iterator_dissolved + 1 in sid.track_list:
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
            Dr.uniform_hist(sid, graph, edges, vols, cA_out, f'dreal_' + name, 'd')
            Dr.draw_nodes(sid, graph, edges, cA_node, f'cA_' + name, 'q')
            Dr.draw_nodes(sid, graph, edges, cB_node, f'cB_' + name, 'q')
            #Dr.draw_colored_edges(sid, graph, edges, vols, f'dc_{data.dissolved_v:.2f}.jpg')
            #Dr.draw_nodes(sid, graph, edges, cb, f'c_{data.dissolved_v:.2f}.jpg', 'q')
            # save_VTK(sid, graph, edges, pressure, cb, \
            #     f'network_{data.dissolved_v:.2f}.vtk')
            data.check_data(edges)
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
    breakthrough, dt_next = Cp.update_diameters(sid, inc, edges, graph, vols, diss_edge_rate, precip_edge_rate, data)
    # if breakthrough:
    #     break
    data.collect_data(sid, inc, edges, vols, pressure, cA_out, cB_out)
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
                Dr.draw_nodes(sid, graph, edges, cA_out, f'c_{data.dissolved_v:.2f}.jpg', 'q')
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
    Dr.uniform_hist(sid, graph, edges, vols, cA_out, f'dreal_' + name, 'd')
    Dr.draw_nodes(sid, graph, edges, cA_node, f'cA_' + name, 'q')
    Dr.draw_nodes(sid, graph, edges, cB_node, f'cB_' + name, 'q')
    #Dr.draw_colored_edges(sid, graph, edges, vols, f'dc_{data.dissolved_v:.2f}.jpg')
    #Dr.draw_nodes(sid, graph, edges, cb, f'c_{data.dissolved_v:.2f}.jpg', 'q')
    Sv.save('/save.dill', sid, graph, inc, edges, triangles, vols)
