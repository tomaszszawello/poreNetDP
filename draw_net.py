""" Plot evolved network.

This module contains functions for plotting the evolved network with diameters
or flow as edge width as well as some additional data to better understand the
simulation.

Notable functions
-------
uniform_hist(SimInputData, Graph, Edges, np.ndarray, np.ndarray, str, str) \
    -> None
    plot the network and histograms of key data
"""

from matplotlib import gridspec
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import matplotlib.colors as mcolors
#matplotlib.use('Agg')

from config import SimInputData
from data import Data
from network import Edges, Graph
from incidence import Incidence
from volumes import Volumes

font = {#'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 10}

matplotlib.rc('font', **font)

def draw(sid, graph, edges, triangles, vols, cb, t):
    int_part  = int(t)
    frac_part = int(100 * (t - int_part))
    name = f"{int_part:04d}_{frac_part:02d}.jpg"
    # draw_nodes(sid, graph, edges, cb, f'c_' + name, 'q')
    # uniform_hist(sid, graph, edges, vols, cb, f'dreal_' + name, 'd')
    if sid.include_volumes:
        draw_triangles(sid, triangles, graph, vols, f'tri_' + name)
    else:
        draw_flow(sid, graph, edges, f'q_' + name, 'q')
        if sid.include_precipitation:
            draw_colored_edges(sid, graph, edges, f'd_' + name)
        else:
            draw_flow(sid, graph, edges, f'd_' + name, 'd')

def draw_net_and_avg_props(sid: SimInputData, graph: Graph, edges: Edges, \
        name: str, title: str, data_obj) -> None:
    """ Draws network above plot of averaged node or edge data 
        
    Parameters
    -------
    data_obj : Data class object
        Used for plotting averaged edge or node property 
    """
    #fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, 
    fig, (ax1, ax2) = plt.subplots(2, 1, 
        figsize=(sid.figsize * 2, sid.figsize * 1),
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.00005})
    fig.suptitle(title, fontsize=15)

    pos = nx.get_node_attributes(graph, 'pos')
    edge_colors = plt.cm.copper_r(mcolors.Normalize(0, 1)(edges.ftrans)) 
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    # Pre-calculate shared node coordinates
    xi, yi = zip(*[pos[n] for n in graph.in_nodes])
    xo, yo = zip(*[pos[n] for n in graph.out_nodes])

    for ax, plot_option in zip([ax1], ['d']): #zip([ax1, ax2], ['d', 'q']):
        ax.set_aspect('equal')
        ax.set_axis_off() # Disabling axes is faster than styling them
        ax.scatter(xi, yi, s=1000/sid.n, fc='white', ec='black', zorder=3)
        ax.scatter(xo, yo, s=1000/sid.n, fc='black', ec='white', zorder=3)
        if plot_option == 'q':
            qs = (1 - edges.boundary_list) * np.abs(edges.flow)
            w = sid.qdrawconst * np.array(qs)
        else:
            qs = (1 - edges.boundary_list) * (edges.diams * (edges.diams > 0))
            w = sid.ddrawconst * np.array(qs)
        nx.draw_networkx_edges(graph, pos, edgelist=edges.edge_list, 
                               edge_color=edge_colors, width=w, ax=ax)

    # plot the concentration profile along the network
    data_obj.plot_avg_node_props(sid, current_time=True, ax=ax2)
    #data_obj.plot_avg_node_props(sid, current_time=True, ax=ax4)

    ax2.set_xlim(ax1.get_xlim())

    # Align the axes
    plt.tight_layout() 
    fig.canvas.draw()
    pos_net = ax1.get_position()
    pos_line = ax2.get_position()
    ax2.set_position([pos_net.x0, pos_line.y0, pos_net.width, pos_line.height])
    fig.savefig(f"{sid.dirname}/{name}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    #gc.collect() # import gc

def draw_flow(sid: SimInputData, graph: Graph, edges: Edges, \
    name: str, plot_type: str) -> None:
    """ Draw the network with diameters/flow as edge width.

    This function plots the network with one of parameters as edge width, as
    well as some histograms of key data.

    Parameters
    -------
    sid : SimInputData
        all config parameters of the simulation
        figsize - size of the plot (~resolution)
        ddrawconst - scaling parameter to improve visibility when drawing
        diameters
        qdrawconst - scaling parameter to improve visibility when drawing flow
        dirname - directory of the simulation

    graph : Graph class object
        network and all its properties
        in_nodes - list of inlet nodes
        out_nodes - list of outlet nodes

    edges : Edges class object
        all edges in network and their parameters
        diams - diameters of edges
        diams_initial - initial diameters of edges
        flow - flow in edges
        boundary_list - edges assuring PBC (to be excluded from drawing)

    cb : numpy ndarray
        vector of current B concentration

    cc : numpy ndarray
        vector of current C concentration

    name : str
        name of the saved file with the plot

    data : str
        parameter taken as edge width (diameter or flow)
    """
    plt.figure(figsize=(sid.figsize * sid.m / sid.n, sid.figsize))
    #plt.suptitle(f'G = {sid.G:.2f}', fontsize = 1, color='white')
    spec = gridspec.GridSpec(ncols = 2, nrows = 1, width_ratios=[100, 1])
    # draw first panel for the network
    ax1 = plt.subplot(spec[0])
    plt.axis('equal')
    ax1.set_axis_on()
    ax1.tick_params(bottom=True)
    plt.xlim(0, sid.m)
    plt.ylim(0, sid.n)
    #plt.xlabel('x', fontsize = 60, style = 'italic')
    #plt.ylabel('flow focusing index', fontsize = 40)
    #plt.yticks([], [])
    #ax1.yaxis.label.set_color('white')
    #ax1.tick_params(axis = 'y', colors='white')
    #ax1.xaxis.label.set_color('white')
    #ax1.tick_params(axis = 'x', colors='white')
    pos = nx.get_node_attributes(graph, 'pos')
    # draw inlet and outlet nodes
    x_in, y_in = [], []
    for node in graph.in_nodes:
        x_in.append(pos[node][0])
        y_in.append(pos[node][1])
    x_out, y_out = [], []
    for node in graph.out_nodes:
        x_out.append(pos[node][0])
        y_out.append(pos[node][1])
    plt.scatter(x_in, y_in, s = 1000 / sid.n, facecolors = 'white', \
        edgecolors = 'black')
    plt.scatter(x_out, y_out, s = 1000 / sid.n, facecolors = 'black', \
        edgecolors = 'white')
    if plot_type == 'q':
        qs = (1 - edges.boundary_list) * np.abs(edges.flow) #/ edges.diams ** 2
        draw_const = sid.qdrawconst
    else:
        #qs = (1 - edges.boundary_list) * np.abs(edges.diams_draw - edges.diams_initial) * (edges.diams_initial > 0)
        #qs = (1 - edges.boundary_list) * edges.diams_draw * (edges.diams_draw - edges.diams_initial > 0.1) * (edges.diams_initial > 0)
        #qs = (1 - edges.boundary_list) * edges.diams_draw * (edges.diams_initial > 0)
        qs = (1 - edges.boundary_list) * edges.diams
        #qs = (1 - edges.boundary_list) * edges.diams * (edges.diams > 3 * edges.diams_initial)
        draw_const = sid.ddrawconst
    nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'k', \
        width = draw_const * np.array(qs))
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(sid.dirname + "/" + name, bbox_inches="tight", dpi = 600)
    plt.close()

def draw_flow_profile(sid: SimInputData, graph: Graph, edges: Edges, \
    data: Data, name: str, plot_type: str) -> None:
    """ Draw the network with diameters/flow as edge width.

    This function plots the network with one of parameters as edge width, as
    well as some histograms of key data.

    Parameters
    -------
    sid : SimInputData
        all config parameters of the simulation
        figsize - size of the plot (~resolution)
        ddrawconst - scaling parameter to improve visibility when drawing
        diameters
        qdrawconst - scaling parameter to improve visibility when drawing flow
        dirname - directory of the simulation

    graph : Graph class object
        network and all its properties
        in_nodes - list of inlet nodes
        out_nodes - list of outlet nodes

    edges : Edges class object
        all edges in network and their parameters
        diams - diameters of edges
        diams_initial - initial diameters of edges
        flow - flow in edges
        boundary_list - edges assuring PBC (to be excluded from drawing)

    cb : numpy ndarray
        vector of current B concentration

    cc : numpy ndarray
        vector of current C concentration

    name : str
        name of the saved file with the plot

    data : str
        parameter taken as edge width (diameter or flow)
    """
    fig = plt.figure(figsize=(sid.figsize, sid.figsize * 1.5))
    spec = gridspec.GridSpec(ncols = 1, nrows = 2, height_ratios = [2, 1])
    # draw first panel for the network
    ax1 = plt.subplot(spec[0])
    plt.axis('equal')
    plt.xlim(0, sid.m)
    plt.ylim(-1.1, sid.n + 1.1)
    plt.yticks([],[])
    pos = nx.get_node_attributes(graph, 'pos')
    # draw inlet and outlet nodes
    x_in, y_in = [], []
    for node in graph.in_nodes:
        x_in.append(pos[node][0])
        y_in.append(pos[node][1])
    x_out, y_out = [], []
    for node in graph.out_nodes:
        x_out.append(pos[node][0])
        y_out.append(pos[node][1])
    plt.scatter(x_in, y_in, s = 1000 / sid.n, facecolors = 'white', \
        edgecolors = 'black')
    plt.scatter(x_out, y_out, s = 1000 / sid.n, facecolors = 'black', \
        edgecolors = 'white')
    if plot_type == 'q':
        qs = (1 - edges.boundary_list) * np.abs(edges.flow)
        draw_const = sid.qdrawconst
    else:
        qs = (1 - edges.boundary_list) * (edges.diams - edges.diams_initial) \
            * (edges.diams > 0)
        draw_const = sid.ddrawconst
    nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'k', \
        width = draw_const * np.array(qs))
    ax2 = plt.subplot(spec[1], sharex = ax1)
    pos_x = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,0]
    slices = np.linspace(np.min(pos_x), np.max(pos_x), 102)[1:-1]
    edge_number  = np.array(data.slices[0])
    plt.plot([], [], ' ', label=' ')
    plt.plot([], [], ' ', label=' ')
    plt.plot([], [], ' ', label=' ')
    plt.plot(slices, np.array((edge_number - 2 * np.array(data.slices[1])) \
        / edge_number), color = 'black', label = '0.0', linewidth = 5)
    colors = ['C0', 'C1', 'C2', 'C3']
    for i, channeling in enumerate(data.slices[2:]):
        plt.plot(slices, (edge_number - 2 * np.array(channeling)) \
            / edge_number, label = data.slice_times[i+1], color = colors[i], linewidth = 5)
    
    plt.ylim(0, 1.05)
    plt.xlabel('x', fontsize = 60, style = 'italic')
    #ax2.xaxis.label.set_color('white')
    #ax2.tick_params(axis = 'x', colors='white')
    #plt.xticks([],[])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(tight = True)
    #plt.ylabel('flow focusing index', fontsize = 50)
    plt.yticks([],[])
    #plt.yticks([0, 0.5, 1],['0', '0.5', '1'])
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0,4,1,5,2,6,3,7]

    legend = plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc="lower center", mode = "expand", ncol = 4, prop={'size': 40}, handlelength = 1, frameon=False, borderpad = 0, handletextpad = 0.4)
    for legobj in legend.legend_handles:
        legobj.set_linewidth(10.0)
    plt.savefig(sid.dirname + "/" + name, bbox_inches="tight")
    plt.close()

def draw_diams_profile(sid: SimInputData, graph: Graph, edges: Edges, \
    data: Data, name: str, plot_type: str) -> None:
    """ Draw the network with diameters/flow as edge width.

    This function plots the network with one of parameters as edge width, as
    well as some histograms of key data.

    Parameters
    -------
    sid : SimInputData
        all config parameters of the simulation
        figsize - size of the plot (~resolution)
        ddrawconst - scaling parameter to improve visibility when drawing
        diameters
        qdrawconst - scaling parameter to improve visibility when drawing flow
        dirname - directory of the simulation

    graph : Graph class object
        network and all its properties
        in_nodes - list of inlet nodes
        out_nodes - list of outlet nodes

    edges : Edges class object
        all edges in network and their parameters
        diams - diameters of edges
        diams_initial - initial diameters of edges
        flow - flow in edges
        boundary_list - edges assuring PBC (to be excluded from drawing)

    cb : numpy ndarray
        vector of current B concentration

    cc : numpy ndarray
        vector of current C concentration

    name : str
        name of the saved file with the plot

    data : str
        parameter taken as edge width (diameter or flow)
    """
    fig = plt.figure(figsize=(sid.figsize, sid.figsize * 1.5))
    spec = gridspec.GridSpec(ncols = 1, nrows = 2, height_ratios = [2, 1])
    # draw first panel for the network
    ax1 = plt.subplot(spec[0])
    plt.axis('equal')
    plt.xlim(0, sid.n)
    plt.ylim(-1.1, sid.n + 1.1)
    plt.yticks([],[])
    pos = nx.get_node_attributes(graph, 'pos')
    # draw inlet and outlet nodes
    x_in, y_in = [], []
    for node in graph.in_nodes:
        x_in.append(pos[node][0])
        y_in.append(pos[node][1])
    x_out, y_out = [], []
    for node in graph.out_nodes:
        x_out.append(pos[node][0])
        y_out.append(pos[node][1])
    plt.scatter(x_in, y_in, s = 1000 / sid.n, facecolors = 'white', \
        edgecolors = 'black')
    plt.scatter(x_out, y_out, s = 1000 / sid.n, facecolors = 'black', \
        edgecolors = 'white')
    if plot_type == 'q':
        qs = (1 - edges.boundary_list) * np.abs(edges.flow)
        draw_const = sid.qdrawconst
    else:
        qs = (1 - edges.boundary_list) * (edges.diams - edges.diams_initial) \
            * (edges.diams > 0)
        draw_const = sid.ddrawconst
    nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'k', \
        width = draw_const * np.array(qs))
    ax2 = plt.subplot(spec[1], sharex = ax1)
    pos_x = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,0]
    slices = np.linspace(np.min(pos_x), np.max(pos_x), 102)[1:-1]
    edge_number  = np.array(data.slices[0])
    plt.plot([], [], ' ')
    plt.plot(slices, np.array((edge_number - 2 * np.array(data.slices[1])) \
        / edge_number), linewidth = 5, color = 'black', label = '0.0')
    for i, channeling in enumerate(data.slices[2:]):
        plt.plot(slices, (edge_number - 2 * np.array(channeling)) \
            / edge_number, label = data.slice_times[i+1], linewidth = 5)
    plt.ylim(0, 1.05)
    plt.xlabel('x', fontsize = 60, style = 'italic')
    # ax2.xaxis.label.set_color('white')
    # ax2.tick_params(axis = 'x', colors='white')
    #plt.xticks([],[])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(tight = True)
    plt.ylabel('flow focusing index', fontsize = 50)
    #plt.yticks([],[])
    plt.yticks([0, 0.5, 1],['0', '0.5', '1'])
    spine_color = 'blue'
    for spine in ax1.spines.values():
        spine.set_linewidth(5)
        spine.set_edgecolor(spine_color)
    for spine in ax2.spines.values():
        spine.set_linewidth(5)
        spine.set_edgecolor(spine_color)
    # save file in the directory
    plt.savefig(sid.dirname + "/" + name, bbox_inches="tight")
    plt.close()

def draw_particles(sid: SimInputData, graph: Graph, edges: Edges, locations: list, \
    name: str, plot_type: str) -> None:
    """ Draw the network with diameters/flow as edge width.

    This function plots the network with one of parameters as edge width, as
    well as some histograms of key data.

    Parameters
    -------
    sid : SimInputData
        all config parameters of the simulation
        figsize - size of the plot (~resolution)
        ddrawconst - scaling parameter to improve visibility when drawing
        diameters
        qdrawconst - scaling parameter to improve visibility when drawing flow
        dirname - directory of the simulation

    graph : Graph class object
        network and all its properties
        in_nodes - list of inlet nodes
        out_nodes - list of outlet nodes

    edges : Edges class object
        all edges in network and their parameters
        diams - diameters of edges
        diams_initial - initial diameters of edges
        flow - flow in edges
        boundary_list - edges assuring PBC (to be excluded from drawing)

    cb : numpy ndarray
        vector of current B concentration

    cc : numpy ndarray
        vector of current C concentration

    name : str
        name of the saved file with the plot

    data : str
        parameter taken as edge width (diameter or flow)
    """
    plt.figure(figsize=(sid.figsize, sid.figsize))
    plt.suptitle(f'G = {sid.G:.2f}', fontsize = 1, color='white')
    spec = gridspec.GridSpec(ncols = 2, nrows = 1, width_ratios=[100, 1])
    # draw first panel for the network
    ax1 = plt.subplot(spec[0])
    plt.axis('equal')
    ax1.set_axis_on()
    ax1.tick_params(bottom=True)
    plt.xlim(0, sid.n)
    plt.ylim(0, sid.n)
    plt.xlabel('x', fontsize = 60, style = 'italic')
    #plt.ylabel('flow focusing index', fontsize = 40)
    plt.yticks([], [])
    #ax1.yaxis.label.set_color('white')
    #ax1.tick_params(axis = 'y', colors='white')
    ax1.xaxis.label.set_color('white')
    ax1.tick_params(axis = 'x', colors='white')
    pos = nx.get_node_attributes(graph, 'pos')
    # draw inlet and outlet nodes
    x_in, y_in = [], []
    for node in graph.in_nodes:
        x_in.append(pos[node][0])
        y_in.append(pos[node][1])
    x_out, y_out = [], []
    for node in graph.out_nodes:
        x_out.append(pos[node][0])
        y_out.append(pos[node][1])
    x_part, y_part = [], []
    for node in locations:
        x_part.append(pos[node][0])
        y_part.append(pos[node][1])
    plt.scatter(x_in, y_in, s = 1000 / sid.n, facecolors = 'white', \
        edgecolors = 'black')
    plt.scatter(x_out, y_out, s = 1000 / sid.n, facecolors = 'black', \
        edgecolors = 'white')

    if plot_type == 'q':
        qs = (1 - edges.boundary_list) * np.abs(edges.flow)
        draw_const = sid.qdrawconst
    else:
        qs = (1 - edges.boundary_list) * np.abs(edges.diams) * (edges.diams > 0)
        draw_const = sid.ddrawconst
    qs = np.clip(qs / np.median(qs), None, 2)
    nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'k', \
        width = draw_const * 3 * np.array(qs))
    plt.scatter(x_part, y_part, s = 1000 / sid.n, facecolors = 'red', \
        edgecolors = 'white')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(sid.dirname + "/" + name, bbox_inches="tight")
    plt.close()

def draw_nodes(sid: SimInputData, graph: Graph, edges: Edges, cb, \
    name: str, data: str) -> None:
    """ Draw the network with diameters/flow as edge width.

    This function plots the network with one of parameters as edge width, as
    well as some histograms of key data.

    Parameters
    -------
    sid : SimInputData
        all config parameters of the simulation
        figsize - size of the plot (~resolution)
        ddrawconst - scaling parameter to improve visibility when drawing
        diameters
        qdrawconst - scaling parameter to improve visibility when drawing flow
        dirname - directory of the simulation

    graph : Graph class object
        network and all its properties
        in_nodes - list of inlet nodes
        out_nodes - list of outlet nodes

    edges : Edges class object
        all edges in network and their parameters
        diams - diameters of edges
        diams_initial - initial diameters of edges
        flow - flow in edges
        boundary_list - edges assuring PBC (to be excluded from drawing)

    name : str
        name of the saved file with the plot

    data : str
        parameter taken as edge width (diameter or flow)
    """
    # draw first panel for the network
    plt.axis('equal')
    
    plt.figure(figsize=(sid.figsize, sid.figsize))
    pos = nx.get_node_attributes(graph, 'pos')
    # draw inlet and outlet nodes
    x_in, y_in = [], []
    for node in graph.in_nodes:
        x_in.append(pos[node][0])
        y_in.append(pos[node][1])
    x_out, y_out = [], []
    for node in graph.out_nodes:
        x_out.append(pos[node][0])
        y_out.append(pos[node][1])
    x_zero, y_zero = [], []
    for node in graph.zero_nodes:
        x_zero.append(pos[node][0])
        y_zero.append(pos[node][1])
    #print (x_zero, y_zero)
    qs = (1 - edges.boundary_list) * np.abs(edges.flow) * (edges.diams > 0)
    #qs = (1 - edges.boundary_list) * np.abs(edges.diams) * (edges.diams > 0)
    draw_const = sid.qdrawconst

    # nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'k', \
    #     width = draw_const * np.array(qs))

    #print(list(zip(edges.edge_list, np.arange(0, len(edges.edge_list)))))
    # for key, val in dict(zip(edges.edge_list, np.arange(0, len(edges.edge_list)))):
    #     print(key, val)
    # print(pos)
    #nx.draw_networkx_edge_labels(graph, pos, edge_labels=dict(zip(edges.edge_list, np.arange(0, len(edges.edge_list)))), font_size = 10)
    #pathcollection = nx.draw_networkx_nodes(graph, pos, nodelist = np.where(cb > 0)[0], node_color = cb[np.where(cb > 0)], node_size = 1000 / sid.n, vmin = 0, vmax = 1)
    pathcollection = nx.draw_networkx_nodes(graph, pos, node_color = cb + np.min(cb), node_size = 1000 / sid.n)
    # pathcollection = nx.draw_networkx_nodes(graph, pos, node_color = (cb + np.min(cb)) * (1 * (cb < 0) + 1 * (cb > 1)) , node_size = 20)
    plt.colorbar(pathcollection)
    #nx.draw_networkx_labels(graph, pos, labels=dict(zip(graph.nodes(), graph.nodes())), font_size=5)
    plt.scatter(x_in, y_in, s = 30, facecolors = 'white', edgecolors = 'black')
    plt.scatter(x_out, y_out, s = 30, facecolors = 'black', \
        edgecolors = 'white')
    #plt.scatter(x_zero, y_zero, s = 60, facecolors = 'blue', edgecolors = 'black')
    # save file in the directory
    plt.axis('off')
    plt.savefig(sid.dirname + "/" + name)
    plt.close()

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib as mpl

def draw_triangles(sid, triangles, graph, volumes, name):

    pos = np.array(list(nx.get_node_attributes(graph, 'pos').values()))
    verts1 = []
    verts2 = []
    # scale1 = (1 - triangles.boundary) * volumes.vol_a / volumes.vol_max
    # scale1 = np.array(np.ma.fix_invalid(scale1, fill_value = 0))
    # for i, nodes in enumerate(triangles.tlist):
    #     n1, n2, n3 = nodes
    #     pi = triangles.centers[i]
    #     scalei = scale1[i]
    #     p1 = (pos[n1] - pi) * scalei + pi
    #     p2 = (pos[n2] - pi) * scalei + pi
    #     p3 = (pos[n3] - pi) * scalei + pi
    #     verts1.append((p1, p2, p3))
    scale1 = np.array(np.ma.fix_invalid((1 - triangles.boundary) * volumes.vol_a / volumes.vol_max,
                                    fill_value=0.0))
    scale2 = np.array(np.ma.fix_invalid((1 - triangles.boundary) * volumes.vol_e / volumes.vol_max,
                                    fill_value=0.0))

    for (n1, n2, n3), pi, s1, s2 in zip(triangles.tlist, triangles.centers, scale1, scale2):
        # current centroid from positions
        C = (pos[n1] + pos[n2] + pos[n3]) / 3.0
        p1 = (pos[n1] - C) * s1 + C
        p2 = (pos[n2] - C) * s1 + C
        p3 = (pos[n3] - C) * s1 + C
        verts1.append((p1, p2, p3))
        if s1 == 0:
            C = (pos[n1] + pos[n2] + pos[n3]) / 3.0
            p1 = (pos[n1] - C) * s2 + C
            p2 = (pos[n2] - C) * s2 + C
            p3 = (pos[n3] - C) * s2 + C
            verts2.append((p1, p2, p3))
        
    
    plt.figure(figsize=(sid.figsize * sid.m / sid.n, sid.figsize))
    spec = gridspec.GridSpec(ncols = 2, nrows = 1, width_ratios=[100, 1])
    # draw first panel for the network
    ax = plt.subplot(spec[0])
    # Make the collection and add it to the plot.
    # coll = PolyCollection(verts2, facecolors = 'c')
    # #coll.set_sizes(z, dpi = 300)
    # #coll.set_alpha(volumes.vol_a / volumes.vol_max)
    # ax.add_collection(coll)
    plt.axis('equal')
    coll1 = PolyCollection(verts1, facecolors = 'black')
    ax.add_collection(coll1)
    coll2 = PolyCollection(verts2, facecolors = 'yellow')
    ax.add_collection(coll2)
    ax.set_xlim(0, sid.m)
    ax.set_ylim(0, sid.n)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.tick_params(axis='both', which='both',
    #             bottom=False, top=False, left=False, right=False,
    #             labelbottom=False, labelleft=False)
    # #ax.autoscale_view()
    # plt.axis('equal')
    # ax.set_axis_on()
    # ax.tick_params(bottom=False, left = False)
    plt.axis('off')
    # Add a colorbar for the PolyCollection
    #fig.colorbar(coll, ax=ax)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(sid.dirname + "/" + name, bbox_inches="tight", dpi = 600)
    plt.close()

def uniform_hist(sid: SimInputData, graph: Graph, edges: Edges, vols: Volumes, \
    cb: np.ndarray, name: str, data: str) -> None:
    """ Draw the network with diameters/flow as edge width.

    This function plots the network with one of parameters as edge width, as
    well as some histograms of key data.

    Parameters
    -------
    sid : SimInputData
        all config parameters of the simulation
        figsize - size of the plot (~resolution)
        ddrawconst - scaling parameter to improve visibility when drawing
        diameters
        qdrawconst - scaling parameter to improve visibility when drawing flow
        dirname - directory of the simulation

    graph : Graph class object
        network and all its properties
        in_nodes - list of inlet nodes
        out_nodes - list of outlet nodes

    edges : Edges class object
        all edges in network and their parameters
        diams - diameters of edges
        diams_initial - initial diameters of edges
        flow - flow in edges
        boundary_list - edges assuring PBC (to be excluded from drawing)
    
    vols : Volumes class object
        volumes of substances in network triangles and their properties
        vol_a - volume of substance A (dissolved)
        vol_e - volume of substance E (precipitated)

    cb : numpy ndarray
        vector of current B concentration

    cc : numpy ndarray
        vector of current C concentration

    name : str
        name of the saved file with the plot

    data : str
        parameter taken as edge width (diameter or flow)
    """
    cols = 6
    plt.figure(figsize=(sid.figsize * 1.25, sid.figsize))
    spec = gridspec.GridSpec(ncols = cols, nrows = 2, height_ratios = [5, 1])
    # draw first panel for the network
    plt.subplot(spec.new_subplotspec((0, 0), colspan = cols))
    plt.axis('equal')
    pos = nx.get_node_attributes(graph, 'pos')
    # draw inlet and outlet nodes
    x_in, y_in = [], []
    for node in graph.in_nodes:
        x_in.append(pos[node][0])
        y_in.append(pos[node][1])
    x_out, y_out = [], []
    for node in graph.out_nodes:
        x_out.append(pos[node][0])
        y_out.append(pos[node][1])
    plt.scatter(x_in, y_in, s = 60, facecolors = 'white', edgecolors = 'black')
    plt.scatter(x_out, y_out, s = 60, facecolors = 'black', \
        edgecolors = 'white')
    # draw triangles with a certain volume
    # x_tr, y_tr = [], []
    # for node in triangles_pos:
    #     x_tr.append(node[0])
    #     y_tr.append(node[1])
    # plt.scatter(x_tr, y_tr, s = 1 * (vols.vol_a < 9), facecolors = 'red', \
    #     edgecolors = 'black')
    #
    # if drawing diameters, we mark clogged pores (here defined as pores with
    # diameter reduced at least twice) as red
    if data == 'd':
        qs2 = (1 - edges.boundary_list) * edges.diams
        nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'k', \
            width = sid.ddrawconst * np.array(qs2))
    elif data == 'q':
        qs = (1 - edges.boundary_list) * np.abs(edges.flow)
        nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'k', \
            width = sid.qdrawconst * np.array(qs))
    # draw histograms with data below the network
    plt.subplot(spec[cols]).set_title('Diameter')
    plt.hist(edges.diams, bins = 50)
    plt.yscale("log")
    plt.subplot(spec[cols + 1]).set_title('Flow')
    plt.hist(np.abs(edges.flow), bins = 50)
    plt.yscale("log")
    plt.subplot(spec[cols + 2]).set_title('cb')
    plt.hist(cb, bins = 50)
    plt.yscale("log")
    plt.subplot(spec[cols + 3]).set_title('cc')
    plt.hist(cb, bins = 50)
    plt.yscale("log")
    plt.subplot(spec[cols + 4]).set_title('vola')
    plt.hist(vols.vol_a, bins = 50)
    plt.yscale("log")
    plt.subplot(spec[cols + 5]).set_title('vole')
    plt.hist(vols.vol_a, bins = 50)
    plt.yscale("log")
    # save file in the directory
    plt.savefig(sid.dirname + "/" + name)
    plt.close()

from matplotlib.colors import LinearSegmentedColormap

# bright endpoints that stay vivid on a dark slide / projector
cmap_custom = LinearSegmentedColormap.from_list(
    "cyan_magenta_bright",
    ["red",   # 0.0 → full-saturation cyan
     "grey"]   # 1.0 → full-saturation magenta
)

def draw_colored_edges(sid: SimInputData, graph: Graph, edges: Edges, name: str) -> None:
    """ Draw the network with diameters/flow as edge width.

    """
    # draw first panel for the network
    plt.figure(figsize=(sid.figsize, sid.figsize))
    spec = gridspec.GridSpec(ncols = 2, nrows = 1, width_ratios=[100, 1])
    pos = nx.get_node_attributes(graph, 'pos')
    ax1 = plt.subplot(spec[0])
    plt.axis('equal')
    plt.xlim(0, sid.n)
    plt.ylim(0, sid.n)
    # draw inlet and outlet nodes
    x_in, y_in = [], []
    for node in graph.in_nodes:
        x_in.append(pos[node][0])
        y_in.append(pos[node][1])
    x_out, y_out = [], []
    for node in graph.out_nodes:
        x_out.append(pos[node][0])
        y_out.append(pos[node][1])
    plt.scatter(x_in, y_in, s = 1000 / sid.n, facecolors = 'white', \
        edgecolors = 'black')
    plt.scatter(x_out, y_out, s = 1000 / sid.n, facecolors = 'black', \
        edgecolors = 'white')
    qs1 = (1 - edges.boundary_list) * edges.diams_initial * 0.9 \
        * (edges.diams <= edges.diams_initial / 5)
    qs2 = (1 - edges.boundary_list) * edges.diams_initial * 0.9 \
        * (edges.diams <= edges.diams_initial * 0.9) \
        * (edges.diams > edges.diams_initial / 5)
    qs3 = (1 - edges.boundary_list) * edges.diams \
        * (edges.diams <= edges.diams_initial * 1.1) \
        * (edges.diams > edges.diams_initial * 0.9)
    qs4 = (1 - edges.boundary_list) * edges.diams \
        * (edges.diams > edges.diams_initial * 1.1)
    colors = np.clip(edges.diams / edges.diams_initial, 0, 1) #1 - (vols.triangles @ (vols.vol_e / vols.vol_max)) / edges.triangles
    nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = colors, \
        edge_cmap=cmap_custom, edge_vmin=0, edge_vmax=1, width = sid.ddrawconst * np.array(qs1))
    nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = colors, \
        edge_cmap=cmap_custom, edge_vmin=0, edge_vmax=1, width = sid.ddrawconst * np.array(qs2))
    nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = colors, \
        edge_cmap=cmap_custom, edge_vmin=0, edge_vmax=1, width = sid.ddrawconst * np.array(qs3))
    nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'k', \
        width = sid.ddrawconst * np.array(qs4))
    #nx.draw_networkx_nodes(graph, pos, node_color = cd)
    plt.subplots_adjust(wspace=0, hspace=0)
    # save file in the directory
    plt.savefig(sid.dirname + "/" + name, bbox_inches="tight")
    plt.close()
