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


# initial merging
# if sid.include_merging:
#     for initial_i in range(sid.initial_merging):
#         Me.solve_merging_vols(sid, inc, graph, vols, edges, 'initial')
#     edges.diams_initial = edges.diams.copy()
#     data.vol_init = np.sum(edges.diams ** 2 * edges.lens)
    #Me.fix_connections(sid, inc, graph, edges)

flag_s = 1

import scipy.sparse as spr
print(sid.ne)
# main loop
# runs until we reach iteration limit or time limit or network is dissolved
while t < tmax and i < iters and data.dissolved_v < sid.dissolved_v_max and not breakthrough and not clogged:
    print((f'Iter {i + 1}/{iters} Time {t:.2f}/{tmax:.2f} \
        Dissolved {data.dissolved_v:.2f}/{sid.dissolved_v_max:.2f}'))
    print(np.max(edges.diams))
    # initialize vectors
    pressure_b = Pr.create_vector(sid, graph)
    if sid.include_diffusion:
        cb_b = Dif.create_vector(sid, graph)
    else:
        cb_b = Di.create_vector(sid, graph)

    # find pressure and update flow in edges
    print ('Solving pressure')
    #print(np.sum(edges.diams ** 2 * edges.lens))
        #Me.fix_connections(sid, inc, graph, edges)
    pressure = Pr.solve_flow(sid, inc, graph, edges, pressure_b)
    #data.check_data(edges)
    Q_in = np.sum(edges.inlet * np.abs(edges.flow))
    Q_out = np.sum(edges.outlet * np.abs(edges.flow))
    print('Q_in =', Q_in, 'Q_out =', Q_out, 'p_in = ', np.max(pressure))

    # find B concentration
    print ('Solving concentration')
    #print(np.sum((edges.flow == 0) * (edges.diams > 0)))
    # find C concentration
    if sid.include_diffusion:
        if sid.include_volumes:
            cb = Dif.solve_vol_nr(sid, inc, graph, edges, vols, cb_b, data)
            #cb = Dif.solve_diffusion_vol(sid, inc, graph, edges, vols, cb_b, data)
        else:
            cb = Dif.solve_diffusion_pe_fix(sid, inc, graph, edges, cb_b)
    else:
        if sid.include_volumes:
            cb = Di.solve_dissolution_safe(sid, inc, graph, edges, vols, cb_b)
        else:
            cb = Di.solve_dissolution(sid, inc, graph, edges, cb_b)
    print('cb: ', np.min(cb), np.max(cb))
    if np.max(cb) > 1.1:
        print(cb)
        print(pressure)
        raise ValueError
    # node = 0
    # if len(np.where(cb < -1e-2)[0]):
    #     node = np.where(cb < -1e-2)[0][0]
    # elif len(np.where(cb > 1.01)[0]):
    #     node = np.where(cb > 1.01)[0][0]
    # if node:
    #     print(node)
    #     print(1 * (spr.diags(edges.flow) @ inc.incidence > 0).T[node].nonzero()[1])
    #     Dr.draw_nodes(sid, graph, edges, cb, f'c_{data.dissolved_v:.2f}.jpg', 'q')
    #     print(pressure[node], np.max(pressure))
    #     for edge in inc.incidence.T[node].nonzero()[1]:
    #         print(edge)
    #         print(edges.flow[edge], edges.diams[edge], edges.A[edge], edges.B[edge], inc.merge_vec[sid.nsq + edge])
    #         for node in inc.incidence[edge].nonzero()[1]:
    #             print(node, pressure[node], cb[node])
    #     #Sv.save('/save.dill', sid, graph, inc, edges, triangles, vols)
    #     raise ValueError("cb")
    #cc, cd = Pi.create_vector_nr(sid, graph, inc, edges, cb)
    if sid.include_precipitation:
        if t == 0 or sid.load == 1:
            #cd = sid.cd_in* np.ones(sid.nsq)#sid.cd_in * (sid.cb_in - cb)
            #cc = 0.001 * np.ones(sid.nsq)#sid.cb_in - cb + sid.cc_in
            # cc, cd = Pi.create_vector_nr(sid, graph, inc, edges, cb)
            # cc2, cd2 = Pi.solve_precipitation_nr2_vxx(sid, inc, graph, edges, vols, cb, cc, cd)
            # print(f'cc: {np.min(cc2)}, {np.max(cc2)}, cd: {np.min(cd2)}, {np.max(cd2)}')
            cc, cd = Pi.create_vector_nr(sid, graph, inc, edges, cb)
            cc, cd = Pi.solve_precipitation_nr9_vxx(sid, inc, graph, edges, vols, cb, cc, cd)             
            #print(np.sum(np.abs(cc - cc2)), np.sum(np.abs(cd - cd2)))
            #np.savetxt('cc.txt', cc)
            #np.savetxt('cc2.txt', cc2)
        else:
            cc, cd = Pi.solve_precipitation_nr9_vxx(sid, inc, graph, edges, vols, cb, cc, cd)
            #print(np.sum(np.abs(cc - cc2)), np.sum(np.abs(cd - cd2)))
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
        data.check_slice_porosity(graph, inc, edges, triangles, vols, int(t))
        data.check_init_slice_channelization(graph, inc, edges)
        data.check_slice_channelization(graph, inc, edges, t)
        #Tr.track(sid, graph, inc, edges, data, pressure)
<<<<<<< Updated upstream
        Dr.draw_flow(sid, graph, edges, f'q_' + name, 'q')
        #Dr.draw_flow(sid, graph, edges, f'd_{data.dissolved_v:.2f}.jpg', 'd')
        Dr.draw_triangles(sid, triangles, edges, graph, vols, f'tri_' + name)
        Dr.uniform_hist(sid, graph, edges, vols, cb, f'dreal_' + name, 'd')
        #Dr.draw_colored_edges(sid, graph, edges, vols, f'dc_{data.dissolved_v:.2f}.jpg')
        #Dr.draw_nodes(sid, graph, edges, cb, f'c_{data.dissolved_v:.2f}.jpg', 'q')
=======
        Dr.draw_flow(sid, graph, edges, f'q_{t:.1f}.jpg', 'q')
        Dr.draw_flow(sid, graph, edges, f'd_{t:.1f}.jpg', 'd')
        #Dr.draw_triangles(sid, triangles, graph, vols, f'tri_{t:.1f}.jpg')
        #Dr.uniform_hist(sid, graph, edges, vols, cb, f'dreal_{t:.1f}.jpg', 'd')
        #Dr.draw_nodes(sid, graph, edges, cosm, f'c_{t:.1f}.jpg', 'q')
        #Dr.draw_nodes(sid, graph, edges, pressure, f'p_{t:.1f}.jpg', 'q')
>>>>>>> Stashed changes
        # save_VTK(sid, graph, edges, pressure, cb, \
        #     f'network_{data.dissolved_v:.2f}.vtk')
    else:
        #if data.dissolved_v // sid.track_every > iterator_dissolved:
        if t // sid.track_every > iterator_dissolved and iterator_dissolved + 1 in sid.track_list:
            # print(edges.diams)
            # print(np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ cb)
            # print(np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ cc)
            # print(np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) @ cd)
            # print(vols.vol_a)
            # print(vols.vol_e)
            
            print('Drawing')
            iterator_dissolved += 1
<<<<<<< Updated upstream
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
            data.check_data(edges)
            data.check_slice_porosity(graph, inc, edges, triangles, vols, int(t))
            data.check_slice_channelization(graph, inc, edges, \
               data.dissolved_v)
            data.save_data()
            #if iterator_dissolved == 3:
            #    Sv.save('/save.dill', sid, graph, inc, edges, triangles, vols)
            #Tr.track(sid, graph, inc, edges, data, pressure)
=======
            if iterator_dissolved in sid.track_list:
                #Dr.draw_flow(sid, graph, edges, \
                #    f'q_{t:.1f}.jpg', 'q')
                Dr.draw_flow(sid, graph, edges, f'q_{t:.1f}.jpg', 'q')
                Dr.draw_flow(sid, graph, edges, f'd_{t:.1f}.jpg', 'd')
                #Dr.draw_triangles(sid, triangles, graph, vols, f'tri_{t:.1f}.jpg')
                #Dr.uniform_hist(sid, graph, edges, vols, cb, f'dreal_{t:.1f}.jpg', 'd')
                #Dr.draw_nodes(sid, graph, edges, cosm, f'c_{t:.1f}.jpg', 'q')
                #Dr.draw_nodes(sid, graph, edges, cb, f'c_{t:.1f}.jpg', 'q')
                #Dr.draw_nodes(sid, graph, edges, pressure, f'p_{t:.1f}.jpg', 'q')
                # save_VTK(sid, graph, edges, pressure, cb, \
                #     f'network_{t:.1f}.vtk')
                data.check_data(edges)
                data.check_slice_channelization(graph, inc, edges, \
                    t)
                #Tr.track(sid, graph, inc, edges, data, pressure)
>>>>>>> Stashed changes
    # grow/shrink diameters and update them in edges, update volumes with
    # dissolved/precipitated values, check if network dissolved, find new
    # timestep
    print('Updating diameters')
    breakthrough, dt_next = Gr.update_diameters(sid, inc, edges, graph, vols, cb, cc, cd, data)
    # if breakthrough:
    #     break
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
<<<<<<< Updated upstream
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
=======
    #         f'focusing_q_{t:.1f}.jpg', 'q')
    Dr.draw_flow(sid, graph, edges, f'q_{t:.1f}.jpg', 'q')
    Dr.draw_flow(sid, graph, edges, f'd_{t:.1f}.jpg', 'd')
    #Dr.draw_triangles(sid, triangles, graph, vols, f'tri_{t:.1f}.jpg')
    #Dr.draw_nodes(sid, graph, edges, cosm, f'c_{t:.1f}.jpg', 'q')
    #Dr.uniform_hist(sid, graph, edges, vols, cb, f'dreal_{t:.1f}.jpg', 'd')
    #Dr.draw_nodes(sid, graph, edges, cb, f'c_{t:.1f}.jpg', 'q')
    np.savetxt(sid.dirname + '/tau_h.txt', np.array([np.sum(np.abs(edges.flow) * edges.lens) / (np.abs(Q_in) * sid.m)]))
    #Sv.save('/save.dill', sid, graph, inc, edges, triangles, vols)
    #data.plot_things(sid)

Dr.draw_triangles(sid, triangles, graph, vols, f'tri_final_{t:.1f}.jpg')
>>>>>>> Stashed changes
