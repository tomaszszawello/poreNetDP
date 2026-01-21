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
import diffusion_fracture as Dif
import draw_net as Dr
import growth_fracture as Gr
import pressure as Pr
import save as Sv

from build_fracture import build
from utils import initialize_iterators, update_iterators


vols = 0
# initialize main classes
sid, inc, graph, edges, data = build()

iters, tmax, i, t, breakthrough = initialize_iterators(sid)
iterator_dissolved = 0


import numpy as np


# main loop
# runs until we reach iteration limit or time limit or network is dissolved
while t < tmax and i < iters and data.dissolved_v < sid.dissolved_v_max and not breakthrough:
    print((f'Iter {i + 1}/{iters} Time {t:.2f}/{tmax:.2f} \
        Dissolved {data.dissolved_v:.2f}/{sid.dissolved_v_max:.2f}'))
    
    pressure_b = Pr.create_vector(sid, graph)
    cb_b = Dif.create_vector(sid, graph)

    # find pressure and update flow in edges
    print ('Solving pressure')
    pressure = Pr.solve_flow(sid, inc, graph, edges, pressure_b)

    # find B concentration
    print ('Solving concentration')
   
    cb = Dif.solve_diffusion_da_fix(sid, inc, graph, edges, cb_b)

    # calculate ffp, draw figures
    if t == 0 and not sid.debug:
        data.check_data(edges)

        Dr.draw_flow(sid, graph, edges, f'q_{t:.1f}.jpg', 'q')
        Dr.draw_flow(sid, graph, edges, f'd_{t:.1f}.jpg', 'd')

        Dr.draw_nodes(sid, graph, edges, cb, f'c_{t:.1f}.jpg', 'q')
        Dr.draw_nodes(sid, graph, edges, pressure, f'p_{t:.1f}.jpg', 'q')

    else:
        if t // sid.track_every > iterator_dissolved:
        #if t // sid.track_every > iterator_dissolved:
            print('Drawing')
            iterator_dissolved += 1
            if iterator_dissolved in sid.track_list:
                #Dr.draw_flow(sid, graph, edges, \
                #    f'q_{t:.1f}.jpg', 'q')
                Dr.draw_flow(sid, graph, edges, f'q_{t:.1f}.jpg', 'q')
                Dr.draw_flow(sid, graph, edges, f'd_{t:.1f}.jpg', 'd')
                #Dr.draw_triangles(sid, triangles, graph, vols, f'tri_{t:.1f}.jpg')
                #Dr.uniform_hist(sid, graph, edges, vols, cb, f'dreal_{t:.1f}.jpg', 'd')
                #Dr.draw_nodes(sid, graph, edges, cosm, f'c_{t:.1f}.jpg', 'q')
                Dr.draw_nodes(sid, graph, edges, cb, f'c_{t:.1f}.jpg', 'q')
                Dr.draw_nodes(sid, graph, edges, pressure, f'p_{t:.1f}.jpg', 'q')
                # save_VTK(sid, graph, edges, pressure, cb, \
                #     f'network_{t:.1f}.vtk')
                data.check_data(edges)
                #Tr.track(sid, graph, inc, edges, data, pressure)
    # grow/shrink diameters and update them in edges, update volumes with
    # dissolved/precipitated values, check if network dissolved, find new
    # timestep
    print ('Updating diameters')
    breakthrough, dt_next = Gr.update_diameters(sid, inc, graph, edges, vols, cb)

    i, t = update_iterators(sid, i, t, dt_next)
    

# save data from the last iteration of simulation, save the whole simulation
# to be able to continue it later
if i != 1 and sid.load != 1 and not sid.debug:
    #data.check_data(edges)


    #Tr.plot_tracking(data, 100)
    # Dr.draw_flow_profile(sid, graph, edges, data, \
    #         f'focusing_q_{t:.1f}.jpg', 'q')
    Dr.draw_flow(sid, graph, edges, f'q_{t:.1f}.jpg', 'q')
    Dr.draw_flow(sid, graph, edges, f'd_{t:.1f}.jpg', 'd')
    #Dr.draw_triangles(sid, triangles, graph, vols, f'tri_{t:.1f}.jpg')
    #Dr.draw_nodes(sid, graph, edges, cosm, f'c_{t:.1f}.jpg', 'q')
    #Dr.uniform_hist(sid, graph, edges, vols, cb, f'dreal_{t:.1f}.jpg', 'd')
    Dr.draw_nodes(sid, graph, edges, cb, f'c_{t:.1f}.jpg', 'q')
    #np.savetxt(sid.dirname + '/tau_h.txt', np.array([np.sum(np.abs(edges.flow) * edges.lens) / (np.abs(Q_in) * sid.m)]))
    #Sv.save('/save.dill', sid, graph, inc, edges, triangles, vols)
    #data.plot_things(sid)

#Dr.draw_triangles(sid, triangles, graph, vols, f'tri_final_{t:.1f}.jpg')