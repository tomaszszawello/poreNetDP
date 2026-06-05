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
import nucleation as Nu
import data as Da # DELME
import probe as Pro

from build import build
from utils import initialize_iterators, update_iterators, stop_condition
#from utils_vtk import save_VTK

import numpy as np
import pandas as pd

# initialize main classes
sid, inc, graph, edges, vols, triangles, data = build()

# initial merging to remove very small spaces between pores
if sid.include_merging and not sid.include_volumes:
    for initial_i in range(sid.initial_merging):
        Me.solve_merging(sid, inc, graph, edges, 'initial')
    edges.diams_initial = edges.diams.copy()
    data.vol_init = np.sum(edges.diams ** 2 * edges.lens)

iters, tmax, i, t, state = initialize_iterators(sid)
iterator_dissolved = 0

if sid.include_diffusion and sid.include_precipitation:
    raise ValueError("Unsupported: this combination is not yet available")
if sid.include_nucleation and not sid.include_precipitation:
    raise ValueError("Unsupported: sid.include_nucleation requires sid.include_precipitation")
if sid.include_nucleation and (sid.include_diffusion or sid.include_volumes):
    raise ValueError("Unsupported: TODO: enable sid.include_volumes")

# probe initialisation
#path_probe = Da.Probe(sid, edges, graph, snode=8, opts="path", total_nodes=10)
#path_probe.show_probes(sid, edges, graph)

grid_probe = Pro.Probe(sid, edges, graph, snode=8, opts="grid", total_nodes=9)
grid_probe.show_probes(sid, edges, graph, labels=False)
#grid_probe.record(
#    t = t,
#    node_data = {'cb': np.ones_like(edges.diams), 'cc': np.zeros_like(edges.diams)},
#    edge_data = {'f': edges.ftrans}
#)


# main loop
# runs until we reach iteration limit or time limit or network is dissolved
while t < tmax and i < iters and not state:
    print(f'Iter {i + 1}/{iters} Time {t:.2f}/{tmax:.2f}')

    # find pressure and update flow in edges
    print ('Solving pressure and flow')
    pressure_b = Pr.create_vector(graph)
    pressure = Pr.solve_flow(sid, inc, graph, edges, pressure_b)
    
    if sid.include_diffusion:
        if sid.include_volumes:
            cb_b = Dif.create_vector_danckwerts(sid, inc, graph, edges)
        else:
            cb_b = Dif.create_vector(sid, graph)
    else:
        cb_b = Di.create_vector(sid, graph)
    
    # find B concentration
    print ('Solving concentration')
    # find C concentration
    if sid.include_diffusion:
        if sid.include_volumes:
            cb = Dif.solve_diffuson_vol(sid, inc, graph, edges, vols, cb_b, data)
        else:
            cb = Dif.solve_diffusion(sid, inc, graph, edges, cb_b)
    else:
        if sid.include_volumes:
            cb = Di.solve_dissolution_nr(sid, inc, graph, edges, vols, cb_b)
        else:
            if sid.include_nucleation:
                cb = Di.solve_dissolution_nucleation(sid, inc, graph, edges, cb_b)
            else:
                cb = Di.solve_dissolution(sid, inc, graph, edges, cb_b)

    if sid.include_precipitation:
        if sid.include_volumes:
            if t == 0 or sid.load == 1:
                cc, cd = Pi.create_vector_nr(sid, graph, inc, edges, cb)
                cc, cd = Pi.solve_precipitation_nr(sid, inc, graph, edges, cb, cc, cd)             
            else:
                cc, cd = Pi.solve_precipitation_nr(sid, inc, graph, edges, cb, cc, cd)
        else:
            cc, cd = Pi.solve_precipitation(sid, inc, graph, edges, cb), np.zeros(sid.nsq)
    else:
        cc, cd = np.zeros(sid.nsq), np.zeros(sid.nsq)


    # calculate ffp, draw figures
    if t == 0 and not sid.debug:
        data.check_init_slice_channelization(graph, inc, edges)
        data.check_slice_channelization(graph, inc, edges, t)
        Dr.draw(sid, graph, edges, triangles, vols, cb, t)
        # save_VTK(sid, graph, edges, pressure, cb, \
        #     f'network_{t:.1f}.vtk')
        #Tr.track(sid, graph, inc, edges, data, pressure)
    else:
        if stop_condition(sid, t, i, iterator_dissolved):
            print(f'Drawing at (i, t, dissolved) = ({i}, {t:.2f}, {iterator_dissolved:.2f})')
            iterator_dissolved += 1
            if sid.track_type == "dissolved" and iterator_dissolved in sid.track_list:
                data.check_slice_channelization(graph, inc, edges, t)
                Dr.draw(sid, graph, edges, triangles, vols, cb, t)
                # save_VTK(sid, graph, edges, pressure, cb, \
                #     f'network_{t:.1f}.vtk')
                #Tr.track(sid, graph, inc, edges, data, pressure)
            else:
                live_tit = f"\n$t =$ {t:.1f}, $Vol. Diss. =$ {data.dissolved_v:.1f}" + \
                        f" Mean $f = ${np.mean(edges.ftrans):.3f}, Mean $d = ${np.mean(edges.diams):.3f}"
                Dr.draw_flow_both(sid, graph, edges, f"t{t:.2f}.png", live_tit)
                grid_probe.plot_time_series_data(sid, edges, graph)

    # grow/shrink diameters and update them in edges, update volumes with
    # dissolved/precipitated values, check if network dissolved, find new
    # timestep
    print ('Updating diameters')
    state, dt_next = Gr.update_diameters(sid, inc, edges, vols, data, cb, cc, cd)

    old_ftrans = edges.ftrans # scope..
    if sid.include_nucleation:
        print ('Updating transformed fraction')
        ccr = Nu.reconstruct_cC_profiles(sid, edges, inc, cb, cc, n_pts=100)
        avg_nr, avg_vel = Nu.get_average_rates(sid, edges, ccr)
        Nu.update_frac_transformed_explicit(sid, edges, ccr, avg_nr, avg_vel, sid.dt)
        #edges.ftrans = np.clip(edges.ftrans, 1e-18, 0.99999)
        # probe data 
        grid_probe.record(
            t = t,
            node_data = {'cb': cb, 'cc': cc, 'pressure': pressure},
            edge_data = {'f': edges.ftrans, 'nucleation_rate_avg': avg_nr, 
                         'growth_vel_avg': avg_vel, 'diams': edges.diams, 'flow': edges.flow}
        )

    data.collect_data(sid, inc, edges, vols, pressure, cb, cc)
    data.check_timescale_sep(sid, inc, edges, old_ftrans)
    data.summarise_data(sid, edges, pressure, cb, cc, cd, state)
    state = data.check_data(sid, edges, pressure, cb, cc, cd, state)
    
    # merge edges
    if sid.include_merging:
        print ('Merging')
        if sid.include_volumes:
            Me.solve_merging_vols(sid, inc, graph, vols, triangles, edges)
        else:
            Me.solve_merging(sid, inc, graph, edges)
    # update physical parameters in data
    i, t = update_iterators(sid, i, t, dt_next)

# save data from the last iteration of simulation, save the whole simulation
# to be able to continue it later
if i != 1 and sid.load != 1 and not sid.debug:
    data.check_slice_channelization(graph, inc, edges, t)
    #Tr.track(sid, graph, inc, edges, data, pressure)
    # save_VTK(sid, graph, edges, pressure, cb, \
    #     f'network_{t:.1f}.vtk')
    data.save_data()
    data.plot_profile(graph)
    #Tr.plot_tracking(data, 100)
    Dr.draw(sid, graph, edges, triangles, vols, cb, t)
    Sv.save('/save.dill', sid, graph, inc, edges, triangles, vols)
    #data.plot_things(sid)
