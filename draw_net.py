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

from config import SimInputData
from data import Data
from network import Edges, Graph
from incidence import Incidence

font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 50}

matplotlib.rc('font', **font)

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
    nx.draw_networkx_edges(graph, pos, edges.edge_list, edge_color = 'k', \
        width = draw_const * np.array(qs), hide_ticks = False)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(sid.dirname + "/" + name, bbox_inches="tight")
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
    for legobj in legend.legendHandles:
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
