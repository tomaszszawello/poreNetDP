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
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as spr

from config import SimInputData
from data import Data
from network import Edges, Graph
from incidence import Incidence



import matplotlib
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 50}

matplotlib.rc('font', **font)


def draw_fig1a(sid: SimInputData, graph: Graph, inc: Incidence, edges: Edges, \
    data: Data, name: str, plot_type: str, colored: bool) -> None:
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
    #plt.suptitle(f'G = {sid.G:.2f} Daeff = {sid.Da_eff:.2f} noise = {sid.noise_filename}')
    spec = gridspec.GridSpec(ncols = 2, nrows = 1, width_ratios=[100, 1])
    # draw first panel for the network
    ax1 = plt.subplot(spec[0])
    #ax1.set_title(f'time: {sid.old_t:.2f} dissolved: {data.dissolved_v:.2f}')
    plt.axis('equal')
    plt.xlim(0, sid.n)
    plt.ylim(0, sid.n)
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
    
    plt.scatter(x_in, y_in, s = 1000 / sid.n, facecolors = 'white', edgecolors = 'black')
    plt.scatter(x_out, y_out, s = 1000 / sid.n, facecolors = 'black', \
        edgecolors = 'white')
    if plot_type == 'q':
        qs = (1 - edges.boundary_list) * np.abs(edges.flow)
    else:
        qs = (1 - edges.boundary_list) * np.abs(edges.diams) * (edges.diams > 0)

    if colored:
        slice_x = sid.n / 2
        plt.vlines(slice_x, -1.1, sid.n + 1.1, 'black', 'dashed')
        pos_x = np.array(list(pos.values()))[:,0]
        slice_edges = (spr.diags(edges.flow) @ inc.incidence > 0) \
            @ (pos_x <= slice_x) * np.abs(inc.incidence @ (pos_x > slice_x)) \
            - (spr.diags(edges.flow) @ inc.incidence > 0) @ (pos_x > slice_x) \
            * np.abs(inc.incidence @ (pos_x <= slice_x))
        slice_flow = np.array(sorted(slice_edges * np.abs(edges.flow), reverse = True))
        fraction_flow = 0
        total_flow = np.sum(slice_flow)
        for i, edge_flow in enumerate(slice_flow):
            fraction_flow += edge_flow
            if fraction_flow > total_flow / 2:
                edge_number = i + 1
                break
        colored_edges_r = np.zeros(sid.ne)
        colored_edges_g = np.zeros(sid.ne)
        for i, edge_flow in enumerate(slice_flow):
            if i < edge_number:
                colored_edges_r[np.where(np.abs(edges.flow) == slice_flow[i])[0][0]] = 1
            elif edge_flow > 0:
                colored_edges_g[np.where(np.abs(edges.flow) == slice_flow[i])[0][0]] = 1

        nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'k', \
            width = sid.qdrawconst * np.array(qs) * (1 - colored_edges_r - colored_edges_g))
        nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'r', \
            width = sid.qdrawconst * np.array(qs) * colored_edges_r)
        nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'b', \
            width = sid.qdrawconst * np.array(qs) * colored_edges_g)
    else:
        #qs = np.ma.fix_invalid(np.log(1+qs), fill_value=0)
        nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'k', \
            width = sid.qdrawconst * np.array(qs), alpha = np.array(qs)/np.max(qs))
    # nx.draw_networkx_edge_labels(graph, pos, edge_labels=dict(zip(edges.edge_list, np.arange(0, len(edges.edge_list)))), font_size = 5)
    plt.subplots_adjust(wspace=0, hspace=0)
    # save file in the directory
    plt.savefig(sid.dirname + "/" + name, bbox_inches="tight")
    plt.close()


def draw_fig1c(sid: SimInputData, graph: Graph, inc: Incidence, edges: Edges, \
    data: Data, name: str, plot_type: str, colored: bool) -> None:
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
    #plt.suptitle(f'G = {sid.G:.2f} Daeff = {sid.Da_eff:.2f} noise = {sid.noise_filename}')
    spec = gridspec.GridSpec(ncols = 1, nrows = 2, height_ratios = [2, 1])
    # draw first panel for the network
    ax1 = plt.subplot(spec[0])
    #ax1.set_title(f'time: {sid.old_t:.2f} dissolved: {data.dissolved_v:.2f}')

    #plt.title(f'G = {sid.G}', fontsize = 70)

    plt.axis('equal')
    plt.xlim(0, sid.n)
    plt.ylim(-1.1, sid.n + 1.1)
    plt.yticks([],[])
    #plt.ylabel(r'Da$_{eff}$ = ' + f'{sid.Da_eff}', fontsize = 70)

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
    
    plt.scatter(x_in, y_in, s = 1000 / sid.n, facecolors = 'white', edgecolors = 'black')
    plt.scatter(x_out, y_out, s = 1000 / sid.n, facecolors = 'black', \
        edgecolors = 'white')
    qs = (1 - edges.boundary_list) * np.abs(edges.flow)

    slice_x = sid.n / 2
    plt.vlines(slice_x, -1.1, sid.n + 1.1, 'black', 'dashed')
    pos_x = np.array(list(pos.values()))[:,0]
    slice_edges = (spr.diags(edges.flow) @ inc.incidence > 0) \
        @ (pos_x <= slice_x) * np.abs(inc.incidence @ (pos_x > slice_x)) \
        - (spr.diags(edges.flow) @ inc.incidence > 0) @ (pos_x > slice_x) \
        * np.abs(inc.incidence @ (pos_x <= slice_x))
    slice_flow = np.array(sorted(slice_edges * np.abs(edges.flow), reverse = True))
    fraction_flow = 0
    total_flow = np.sum(slice_flow)
    for i, edge_flow in enumerate(slice_flow):
        fraction_flow += edge_flow
        if fraction_flow > total_flow / 2:
            edge_number = i + 1
            break
    colored_edges_r = np.zeros(sid.ne)
    colored_edges_g = np.zeros(sid.ne)
    for i, edge_flow in enumerate(slice_flow):
        if i < edge_number:
            colored_edges_r[np.where(np.abs(edges.flow) == slice_flow[i])[0][0]] = 1
        elif edge_flow > 0:
            colored_edges_g[np.where(np.abs(edges.flow) == slice_flow[i])[0][0]] = 1

    nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'k', \
        width = sid.qdrawconst * np.array(qs) * (1 - colored_edges_r - colored_edges_g))
    nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'r', \
        width = sid.qdrawconst * np.array(qs) * colored_edges_r)
    nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'b', \
        width = sid.qdrawconst * np.array(qs) * colored_edges_g)

    ax2 = plt.subplot(spec[1], sharex = ax1)
    pos_x = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,0]
    slices = np.linspace(np.min(pos_x), np.max(pos_x), 102)[1:-1]
    edge_number  = np.array(data.slices[0])
    plt.plot(slices, np.array((edge_number - 2 * np.array(data.slices[1])) / edge_number), color = 'black', linewidth = 5, label = '0.0')
    plt.plot(slices, np.array((edge_number - 2 * np.array(data.slices[-1])) / edge_number), color = 'red', linewidth = 5, label = data.slice_times[-1])
    # plt.plot(slices, np.array((edge_number - 2 * np.array(data.slices_d[-1])) / edge_number), color = 'red')
    # plt.plot(slices, np.array((edge_number - 2 * np.array(data.slices_d[1])) / edge_number), '--', color = 'red')
    # plt.plot(slices, np.array((edge_number - 2 * np.array(data.slices_s[-1])) / edge_number), color = 'yellow')
    # plt.plot(slices, np.array((edge_number - 2 * np.array(data.slices_s[1])) / edge_number), '--', color = 'yellow')
    plt.ylim(0, 1)
    if len(data.slices) == 3:
        plt.ylabel('flow focusing index', fontsize = 50)
        plt.yticks([0, 0.5, 1],['0', '0.5', '1'])
    else:
        plt.yticks([], [])
    #plt.plot([], [], ' ', label=' ')
    plt.ylim(0, 1.05)
    plt.vlines(slice_x, 0, 1, 'black', 'dashed')
    plt.xlabel('x', fontsize = 60, style = 'italic')
    #ax2.xaxis.label.set_color('white')
    #ax2.tick_params(axis = 'x', colors='white')
    plt.xticks([0, 25, 50],['0', '25', '50'])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(tight = True)
    #plt.ylabel('flow focusing index', fontsize = 40)
    
    legend = plt.legend(loc="lower center", mode = "expand", ncols = 4, prop={'size': 40}, handlelength = 1, frameon=False, borderpad = 0)
    #legend = plt.legend(loc="lower right", ncols = 4, prop={'size': 40}, handlelength = 1, frameon=False, borderpad = 0)
    legend._legend_box.sep = 3
    for legobj in legend.legendHandles:
        legobj.set_linewidth(10.0)
    #fig.align_ylabels()
    # save file in the directory
    plt.savefig(sid.dirname + "/" + name, bbox_inches="tight")
    plt.close()


def draw_fig2(sid: SimInputData, graph: Graph, inc: Incidence, edges: Edges, \
    data: Data, name: str, plot_type: str, colored: bool) -> None:
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
    #ax1.set_title(f'time: {sid.old_t:.2f} dissolved: {data.dissolved_v:.2f}')
    #if len(data.slices) == 2:
    #    plt.suptitle(f'G = {sid.G:.2f}', fontsize = 1, color='white')
    #    plt.title(r'$\sigma^2$ = 0.1', fontsize = 70)
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
    
    plt.scatter(x_in, y_in, s = 1000 / sid.n, facecolors = 'white', edgecolors = 'black')
    plt.scatter(x_out, y_out, s = 1000 / sid.n, facecolors = 'black', \
        edgecolors = 'white')
    if plot_type == 'q':
        qs = (1 - edges.boundary_list) * np.abs(edges.flow)
        draw_const = sid.qdrawconst
    else:
        qs = (1 - edges.boundary_list) * np.abs(edges.diams) * (edges.diams > 0)
        draw_const = sid.ddrawconst
    nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'k', \
        width = draw_const * np.array(qs), hide_ticks = False)
    # nx.draw_networkx_edge_labels(graph, pos, edge_labels=dict(zip(edges.edge_list, np.arange(0, len(edges.edge_list)))), font_size = 5)
    plt.subplots_adjust(wspace=0, hspace=0)
    # save file in the directory
    plt.savefig(sid.dirname + "/" + name, bbox_inches="tight")
    plt.close()


def draw_fig2c(sid: SimInputData, graph: Graph, inc: Incidence, edges: Edges, \
    data: Data, name: str, plot_type: str, colored: bool) -> None:
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
    #plt.suptitle(f'G = {sid.G:.2f} Daeff = {sid.Da_eff:.2f} noise = {sid.noise_filename}')
    spec = gridspec.GridSpec(ncols = 1, nrows = 2, height_ratios = [2, 1])
    # draw first panel for the network
    ax1 = plt.subplot(spec[0])
    #ax1.set_title(f'time: {sid.old_t:.2f} dissolved: {data.dissolved_v:.2f}')

    #plt.title(f'G = {sid.G}', fontsize = 70)

    plt.axis('equal')
    plt.xlim(0, sid.n)
    plt.ylim(-1.1, sid.n + 1.1)
    plt.yticks([],[])
    #plt.ylabel(r'Da$_{eff}$ = ' + f'{sid.Da_eff}', fontsize = 70)

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
    
    plt.scatter(x_in, y_in, s = 1000 / sid.n, facecolors = 'white', edgecolors = 'black')
    plt.scatter(x_out, y_out, s = 1000 / sid.n, facecolors = 'black', \
        edgecolors = 'white')
    if plot_type == 'q':
        qs = (1 - edges.boundary_list) * np.abs(edges.flow)
        draw_const = sid.qdrawconst
    else:
        qs = (1 - edges.boundary_list) * (edges.diams - edges.diams_initial) * (edges.diams > 0)
        draw_const = sid.ddrawconst
        #qs = (1 - edges.boundary_list) * edges.diams
    # nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'k', \
    #     width = sid.qdrawconst * np.array(qs), alpha = (0.2 + 0.8*np.clip(np.array(qs)/np.max(qs), 0, 1)))
    nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'k', \
        width = draw_const * np.array(qs))
    # nx.draw_networkx_edge_labels(graph, pos, edge_labels=dict(zip(edges.edge_list, np.arange(0, len(edges.edge_list)))), font_size = 5)
    # draw histograms with data below the network
    ax2 = plt.subplot(spec[1], sharex = ax1)
    pos_x = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,0]
    slices = np.linspace(np.min(pos_x), np.max(pos_x), 102)[1:-1]
    edge_number  = np.array(data.slices[0])
    #i_division = sid.dissolved_v_max // sid.track_every // 3
    # plt.plot([], [], ' ', label = ' ')
    # plt.plot([], [], ' ', label = ' ')
    # plt.plot([], [], ' ', label = ' ')
    plt.plot(slices, np.array((edge_number - 2 * np.array(data.slices[1])) / edge_number), color = 'black', label = '0.0', linewidth = 5)
    colors = ['C0', 'C1', 'C2', 'C3']
    for i, channeling in enumerate(data.slices[2:]):
        #if i % i_division == 0 and i > 0:
        plt.plot(slices, (edge_number - 2 * np.array(channeling)) / edge_number, \
                label = data.slice_times[i+1], linewidth = 5, color = colors[i])
    
    #plt.plot([], [], ' ', label=' ')
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

    # handles, labels = plt.gca().get_legend_handles_labels()
    # order = [0,4,1,5,2,6,3,7]

    # legend = plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc="lower center", mode = "expand", ncols = 4, prop={'size': 40}, handlelength = 1, frameon=False, borderpad = 0, handletextpad = 0.4)
    # for legobj in legend.legendHandles:
    #     legobj.set_linewidth(10.0)
    legend = plt.legend(loc="lower center", mode = "expand", ncols = 4, prop={'size': 40}, handlelength = 1, frameon=False, borderpad = 0, handletextpad = 0.4)
    for legobj in legend.legendHandles:
        legobj.set_linewidth(10.0)


    #fig.align_ylabels()
    # save file in the directory
    plt.savefig(sid.dirname + "/" + name, bbox_inches="tight")
    plt.close()

def draw_fig3(sid: SimInputData, graph: Graph, inc: Incidence, edges: Edges, \
    data: Data, name: str, plot_type: str, colored: bool) -> None:
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
    #plt.suptitle(f'G = {sid.G:.2f} Daeff = {sid.Da_eff:.2f} noise = {sid.noise_filename}')
    spec = gridspec.GridSpec(ncols = 1, nrows = 2, height_ratios = [2, 1])
    # draw first panel for the network
    ax1 = plt.subplot(spec[0])
    #ax1.set_title(f'time: {sid.old_t:.2f} dissolved: {data.dissolved_v:.2f}')

    #plt.title(f'G = {sid.G}', fontsize = 70)

    plt.axis('equal')
    plt.xlim(0, sid.n)
    plt.ylim(-1.1, sid.n + 1.1)
    plt.yticks([],[])
    plt.ylabel('$\mathregular{Da}^L_\mathregular{eff}}$ = ' + f'{100*sid.Da_eff}', fontsize = 70)

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
    
    plt.scatter(x_in, y_in, s = 1000 / sid.n, facecolors = 'white', edgecolors = 'black')
    plt.scatter(x_out, y_out, s = 1000 / sid.n, facecolors = 'black', \
        edgecolors = 'white')
    if plot_type == 'q':
        qs = (1 - edges.boundary_list) * np.abs(edges.flow)
        draw_const = sid.qdrawconst
    else:
        qs = (1 - edges.boundary_list) * (edges.diams - edges.diams_initial) * (edges.diams > 0)
        draw_const = sid.ddrawconst
        #qs = (1 - edges.boundary_list) * edges.diams
    # nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'k', \
    #     width = sid.qdrawconst * np.array(qs), alpha = (0.2 + 0.8*np.clip(np.array(qs)/np.max(qs), 0, 1)))
    nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'k', \
        width = draw_const * np.array(qs))
    # nx.draw_networkx_edge_labels(graph, pos, edge_labels=dict(zip(edges.edge_list, np.arange(0, len(edges.edge_list)))), font_size = 5)
    # draw histograms with data below the network
    ax2 = plt.subplot(spec[1], sharex = ax1)
    pos_x = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,0]
    slices = np.linspace(np.min(pos_x), np.max(pos_x), 102)[1:-1]
    edge_number  = np.array(data.slices[0])
    #i_division = sid.dissolved_v_max // sid.track_every // 3
    plt.plot([], [], ' ')
    plt.plot(slices, np.array((edge_number - 2 * np.array(data.slices[1])) / edge_number), linewidth = 5, color = 'black', label = '0.0')
    for i, channeling in enumerate(data.slices[2:]):
        #if i % i_division == 0 and i > 0:
        plt.plot(slices, (edge_number - 2 * np.array(channeling)) / edge_number, \
                label = data.slice_times[i+1], linewidth = 5)
    
    #plt.plot([], [], ' ', label=' ')
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
    # legend = plt.legend(loc="lower center", mode = "expand", ncols = 4, prop={'size': 40}, handlelength = 1, frameon=False, borderpad = 0, handletextpad = 0.4)
    # legend._legend_box.sep = 3
    # for legobj in legend.legendHandles:
    #     legobj.set_linewidth(10.0)

    spine_color = 'blue'
    for spine in ax1.spines.values():
        spine.set_linewidth(5)
        spine.set_edgecolor(spine_color)
    for spine in ax2.spines.values():
        spine.set_linewidth(5)
        spine.set_edgecolor(spine_color)
    #fig.align_ylabels()
    # save file in the directory
    plt.savefig(sid.dirname + "/" + name, bbox_inches="tight")
    plt.close()


def draw_anim(sid: SimInputData, graph: Graph, inc: Incidence, edges: Edges, \
    data: Data, name: str, plot_type: str, colored: bool) -> None:
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
    #plt.suptitle(f'G = {sid.G:.2f} Daeff = {sid.Da_eff:.2f} noise = {sid.noise_filename}')
    spec = gridspec.GridSpec(ncols = 1, nrows = 2, height_ratios = [2, 1])
    # draw first panel for the network
    ax1 = plt.subplot(spec[0])
    #ax1.set_title(f'time: {sid.old_t:.2f} dissolved: {data.dissolved_v:.2f}')

    plt.title(f'wormholing', fontsize = 70)

    plt.axis('equal')
    plt.xlim(0, sid.n)
    plt.ylim(-1.1, sid.n + 1.1)
    plt.yticks([],[])
    #plt.ylabel(r'Da$_{eff}$ = ' + f'{sid.Da_eff}', fontsize = 70)

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
    
    plt.scatter(x_in, y_in, s = 1000 / sid.n, facecolors = 'white', edgecolors = 'black')
    plt.scatter(x_out, y_out, s = 1000 / sid.n, facecolors = 'black', \
        edgecolors = 'white')
    if plot_type == 'q':
        qs = (1 - edges.boundary_list) * np.abs(edges.flow)
        draw_const = sid.qdrawconst
    else:
        qs = (1 - edges.boundary_list) * (edges.diams - edges.diams_initial) * (edges.diams > 0)
        draw_const = sid.ddrawconst
        #qs = (1 - edges.boundary_list) * edges.diams
    # nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'k', \
    #     width = sid.qdrawconst * np.array(qs), alpha = (0.2 + 0.8*np.clip(np.array(qs)/np.max(qs), 0, 1)))
    nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'k', \
        width = draw_const * np.array(qs))
    # nx.draw_networkx_edge_labels(graph, pos, edge_labels=dict(zip(edges.edge_list, np.arange(0, len(edges.edge_list)))), font_size = 5)
    # draw histograms with data below the network
    ax2 = plt.subplot(spec[1], sharex = ax1)
    pos_x = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,0]
    slices = np.linspace(np.min(pos_x), np.max(pos_x), 102)[1:-1]
    edge_number  = np.array(data.slices[0])
    #i_division = sid.dissolved_v_max // sid.track_every // 3
    # plt.plot([], [], ' ', label = ' ')
    # plt.plot([], [], ' ', label = ' ')
    # plt.plot([], [], ' ', label = ' ')
    plt.plot(slices, np.array((edge_number - 2 * np.array(data.slices[1])) / edge_number), color = 'black', label = 'initial', linewidth = 5)
    plt.plot(slices, (edge_number - 2 * np.array(data.slices[-1])) / edge_number, \
            label = 'current', linewidth = 5, color = 'red')
    
    #plt.plot([], [], ' ', label=' ')
    plt.ylim(0, 1.05)

    plt.xlabel('x', fontsize = 60, style = 'italic')
    #ax2.xaxis.label.set_color('white')
    #ax2.tick_params(axis = 'x', colors='white')
    #plt.xticks([],[])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(tight = True)
    plt.ylabel('flow focusing index', fontsize = 50)
    #plt.yticks([],[])
    plt.yticks([0, 0.5, 1],['0', '0.5', '1'])

    # handles, labels = plt.gca().get_legend_handles_labels()
    # order = [0,4,1,5,2,6,3,7]

    # legend = plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc="lower center", mode = "expand", ncols = 4, prop={'size': 40}, handlelength = 1, frameon=False, borderpad = 0, handletextpad = 0.4)
    # for legobj in legend.legendHandles:
    #     legobj.set_linewidth(10.0)
    legend = plt.legend(loc="lower center", mode = "expand", ncols = 4, prop={'size': 40}, handlelength = 1, frameon=False, borderpad = 0, handletextpad = 0.4)
    for legobj in legend.legendHandles:
        legobj.set_linewidth(10.0)


    #fig.align_ylabels()
    # save file in the directory
    plt.savefig(sid.dirname + "/" + name, bbox_inches="tight")
    plt.close()