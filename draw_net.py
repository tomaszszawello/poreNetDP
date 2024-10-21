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

def uniform_hist(sid: SimInputData, graph: Graph, edges: Edges, \
    cb: np.ndarray, cc: np.ndarray, name: str, data: str) -> None:
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
    cols = 4
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
    if data == 'd':
        qs1 = (1 - edges.boundary_list) * edges.diams \
            * (edges.diams < edges.diams_initial / 2)
        qs2 = (1 - edges.boundary_list) * edges.diams \
            * (edges.diams >= edges.diams_initial / 2)
        nx.draw_networkx_edges(graph, pos, edges.edge_list_draw, edge_color = 'r', \
            width = sid.ddrawconst * np.array(qs1))
        nx.draw_networkx_edges(graph, pos, edges.edge_list_draw, edge_color = 'k', \
            width = sid.ddrawconst * np.array(qs2))
    elif data == 'q':
        qs = (1 - edges.boundary_list) * np.abs(edges.flow)
        nx.draw_networkx_edges(graph, pos, edges.edge_list_draw, edge_color = 'k', \
            width = sid.ddrawconst * np.array(qs))
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
    plt.hist(cc, bins = 50)
    plt.yscale("log")
    # save file in the directory
    plt.savefig(sid.dirname + "/" + name)
    plt.close()


def draw(sid: SimInputData, graph: Graph, edges: Edges, \
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

    name : str
        name of the saved file with the plot

    data : str
        parameter taken as edge width (diameter or flow)
    """
    # draw first panel for the network
    plt.figure(figsize=(sid.figsize, sid.figsize * sid.m / sid.n), dpi = 300)
    spec = gridspec.GridSpec(ncols = 2, nrows = 1, width_ratios=[100, 1])
    pos = nx.get_node_attributes(graph, 'pos')
    ax1 = plt.subplot(spec[0])
    #plt.axis('equal')
    plt.xlim(np.min(np.array(list(pos.values()))[:, 0]), np.max(np.array(list(pos.values()))[:, 0]))
    plt.ylim(np.min(np.array(list(pos.values()))[:, 1]), np.max(np.array(list(pos.values()))[:, 1]))
    # draw inlet and outlet nodes
    x_in, y_in = [], []
    nodes = list(graph.nodes())
    index_to_node = {idx: node for idx, node in enumerate(nodes)}
    for node in graph.in_nodes:
        x_in.append(pos[index_to_node[node]][0])
        y_in.append(pos[index_to_node[node]][1])
    x_out, y_out = [], []
    for node in graph.out_nodes:
        x_out.append(pos[index_to_node[node]][0])
        y_out.append(pos[index_to_node[node]][1])
    plt.scatter(x_in, y_in, s = 1000 / sid.n, facecolors = 'white', \
        edgecolors = 'black')
    plt.scatter(x_out, y_out, s = 1000 / sid.n, facecolors = 'white', \
        edgecolors = 'black')
    if plot_type == 'q':
        qs = (1 - edges.boundary_list) * np.abs(edges.flow)
        nx.draw_networkx_edges(graph, pos, edges.edge_list_draw, edge_color = 'k', \
            width = sid.qdrawconst * np.array(qs))
    else:
        qs1 = (1 - edges.boundary_list) * edges.diams_initial * 0.9 \
            * (edges.diams <= edges.diams_initial / 5)
        qs2 = (1 - edges.boundary_list) * edges.diams_initial * 0.9 \
            * (edges.diams <= edges.diams_initial * 0.9) \
            * (edges.diams > edges.diams_initial / 5)
        qs3 = (1 - edges.boundary_list) * edges.diams \
            * (edges.diams <= edges.diams_initial * 1.5) \
            * (edges.diams > edges.diams_initial * 0.9)
        qs4 = (1 - edges.boundary_list) * edges.diams \
            * (edges.diams > edges.diams_initial * 1.5)
        nx.draw_networkx_edges(graph, pos, edges.edge_list_draw, edge_color = 'r', \
            width = sid.ddrawconst * np.array(qs1))
        nx.draw_networkx_edges(graph, pos, edges.edge_list_draw, edge_color = 'y', \
            width = sid.ddrawconst * np.array(qs2))
        nx.draw_networkx_edges(graph, pos, edges.edge_list_draw, edge_color = 'grey', \
            width = sid.ddrawconst * np.array(qs3))
        nx.draw_networkx_edges(graph, pos, edges.edge_list_draw, edge_color = 'k', \
            width = sid.ddrawconst * np.array(qs4))
    #nx.draw_networkx_nodes(graph, pos, node_color = cd)
    plt.subplots_adjust(wspace=0, hspace=0)
    # save file in the directory
    plt.savefig(sid.dirname + "/" + name, bbox_inches="tight")
    plt.close()


def draw_c(sid: SimInputData, graph: Graph, edges: Edges, \
    name: str, plot_type: str, cb_in, cc_in) -> None:
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
    plt.figure(figsize=(sid.figsize, sid.figsize * sid.m / sid.n), dpi = 300)
    spec = gridspec.GridSpec(ncols = 2, nrows = 1, width_ratios=[100, 1])
    pos = nx.get_node_attributes(graph, 'pos')
    ax1 = plt.subplot(spec[0])
    #plt.axis('equal')
    plt.xlim(np.min(np.array(list(pos.values()))[:, 0]), np.max(np.array(list(pos.values()))[:, 0]))
    plt.ylim(np.min(np.array(list(pos.values()))[:, 1]), np.max(np.array(list(pos.values()))[:, 1]))
    # draw inlet and outlet nodes
    x_in, y_in = [], []
    nodes = list(graph.nodes())
    index_to_node = {idx: node for idx, node in enumerate(nodes)}
    for node in graph.in_nodes:
        x_in.append(pos[index_to_node[node]][0])
        y_in.append(pos[index_to_node[node]][1])
    x_out, y_out = [], []
    for node in graph.out_nodes:
        x_out.append(pos[index_to_node[node]][0])
        y_out.append(pos[index_to_node[node]][1])
    plt.scatter(x_in, y_in, s = 1000 / sid.n, facecolors = 'white', \
        edgecolors = 'black')
    plt.scatter(x_out, y_out, s = 1000 / sid.n, facecolors = 'black', \
        edgecolors = 'white')
    if plot_type == 'q':
        qs = (1 - edges.boundary_list) * np.abs(edges.flow)
        nx.draw_networkx_edges(graph, pos, edges.edge_list_draw, edge_color = 'k', \
            width = sid.qdrawconst * np.array(qs))
    else:
        qs1 = (1 - edges.boundary_list) * edges.diams \
            * (edges.diams < edges.diams_initial / 2)
        qs2 = (1 - edges.boundary_list) * edges.diams \
            * (edges.diams >= edges.diams_initial / 2) * (cb_in > cc_in)
        qs3 = (1 - edges.boundary_list) * edges.diams \
            * (edges.diams >= edges.diams_initial / 2) * (cb_in <= cc_in)
        nx.draw_networkx_edges(graph, pos, edges.edge_list_draw, edge_color = 'r', \
            width = sid.ddrawconst * np.array(qs1))
        nx.draw_networkx_edges(graph, pos, edges.edge_list_draw, edge_color = 'y', \
            width = sid.ddrawconst * np.array(qs2))
        nx.draw_networkx_edges(graph, pos, edges.edge_list_draw, edge_color = 'b', \
            width = sid.ddrawconst * np.array(qs3))
    plt.subplots_adjust(wspace=0, hspace=0)
    # save file in the directory
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
    pathcollection = nx.draw_networkx_nodes(graph, pos, node_color = cb * (cb >= 0))
    plt.colorbar(pathcollection)
    #nx.draw_networkx_labels(graph, pos, labels=dict(zip(graph.nodes(), graph.nodes())), font_size=5)
    plt.scatter(x_in, y_in, s = 30, facecolors = 'white', edgecolors = 'black')
    plt.scatter(x_out, y_out, s = 30, facecolors = 'black', \
        edgecolors = 'white')
    plt.scatter(x_zero, y_zero, s = 60, facecolors = 'blue', edgecolors = 'black')
    # save file in the directory
    plt.axis('off')
    plt.savefig(sid.dirname + "/" + name)
    plt.close()
    
def draw_labels(sid: SimInputData, graph: Graph, edges: Edges, \
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
    plt.scatter(x_in, y_in, s = 30, facecolors = 'white', edgecolors = 'black')
    plt.scatter(x_out, y_out, s = 30, facecolors = 'black', \
        edgecolors = 'white')
    
    if data == 'd':
        qs1 = (1 - edges.boundary_list) * edges.diams \
            * (edges.diams < edges.diams_initial / 2)
        qs2 = (1 - edges.boundary_list) * edges.diams \
            * (edges.diams >= edges.diams_initial / 2)
        nx.draw_networkx_edges(graph, pos, edges.edge_list_draw, edge_color = 'r', \
            width = sid.ddrawconst * np.array(qs1))
        nx.draw_networkx_edges(graph, pos, edges.edge_list_draw, edge_color = 'k', \
            width = sid.ddrawconst * np.array(qs2))
    elif data == 'q':
        qs = (1 - edges.boundary_list) * np.abs(edges.flow)
        nx.draw_networkx_edges(graph, pos, edges.edge_list_draw, edge_color = 'k', \
            width = sid.qdrawconst * np.array(qs))
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=dict(zip(edges.edge_list_draw, np.arange(0, len(edges.edge_list_draw)))), font_size = 5)
    #plt.scatter(x_zero, y_zero, s = 60, facecolors = 'blue', edgecolors = 'black')
    # save file in the directory
    plt.axis('off')
    plt.savefig(sid.dirname + "/" + name)
    plt.close()

def draw_data(sid: SimInputData, graph: Graph, edges: Edges, \
    data: Data, name: str, plot_data: str) -> None:
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
    plt.figure(figsize=(sid.figsize * 1.25, sid.figsize))
    plt.suptitle(f'G = {sid.G:.2f} Daeff = {sid.Da_eff:.2f} noise = {sid.noise_filename}')
    spec = gridspec.GridSpec(ncols = 2, nrows = 2, width_ratios = [3, 1])
    # draw first panel for the network
    plt.subplot(spec.new_subplotspec((0, 0), rowspan = 2)).set_title(f'time: {sid.old_t:.2f} dissolved: {data.dissolved_v:.2f}')
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
    if plot_data == 'd':
        qs1 = (1 - edges.boundary_list) * edges.diams \
            * (edges.diams < edges.diams_initial / 2)
        qs2 = (1 - edges.boundary_list) * edges.diams \
            * (edges.diams >= edges.diams_initial / 2)
        nx.draw_networkx_edges(graph, pos, edges.edge_list_draw, edge_color = 'r', \
            width = sid.ddrawconst * np.array(qs1))
        nx.draw_networkx_edges(graph, pos, edges.edge_list_draw, edge_color = 'k', \
            width = sid.ddrawconst * np.array(qs2))
    elif plot_data == 'q':
        qs = (1 - edges.boundary_list) * np.abs(edges.flow)
        nx.draw_networkx_edges(graph, pos, edges.edge_list_draw, edge_color = 'k', \
            width = sid.ddrawconst * np.array(qs))
    # draw histograms with data below the network
    plt.subplot(spec[1]).set_title(f'Slice t: {data.slice_times[-1]}')
    pos_x = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,0]
    slices = np.linspace(np.min(pos_x), np.max(pos_x), 120)[10:-10]
    edge_number  = np.array(data.slices[0])
    plt.plot(slices, np.array(data.slices[1]) / edge_number, color = 'red')
    plt.plot(slices, np.array(data.slices[-1]) / edge_number)

    plt.ylim(0, 1)
    plt.xlabel('x')
    plt.ylabel('channeling [%]')
    plt.subplot(spec[3]).set_title('Participation ratio')
    plt.plot(data.dissolved_v_list/data.vol_init, data.participation_ratio)
    plt.ylim(0, 1)
    plt.xlim(0, sid.dissolved_v_max/data.vol_init)
    plt.xlabel('dissolved volume')
    # ax_p = plt.subplot(spec[3])
    # ax_p.set_title('Participation ratio')
    # ax_p.set_ylim(0, 1)
    # ax_p.set_xlim(0, sid.tmax)
    # ax_p.set_xlabel('time')
    # ax_p.set_ylabel('participation ratio')
    # ax_p2 = ax_p.twinx()
    # ax_p2.plot(data.t, data.participation_ratio_nom, label = "pi", color='green', linestyle='dashed')
    # ax_p2.plot(data.t, data.participation_ratio_denom, label = "pi'", color='red', linestyle='dashed')
    # ax_p.plot(data.t, data.participation_ratio)
    # save file in the directory
    plt.savefig(sid.dirname + "/" + name)
    plt.close()
    
    
def draw_focusing(sid: SimInputData, graph: Graph, inc: Incidence, edges: Edges, \
    data: Data, name: str, plot_data: str) -> None:
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
    plt.figure(figsize=(sid.figsize, sid.figsize * 1.5))
    plt.suptitle(f'G = {sid.G:.2f} Daeff = {sid.Da_eff:.2f} noise = {sid.noise_filename}')
    spec = gridspec.GridSpec(ncols = 1, nrows = 2, height_ratios = [2, 1])
    # draw first panel for the network
    #plt.subplot(spec.new_subplotspec((0, 0), rowspan = 2)).set_title(f'time: {sid.old_t:.2f} dissolved: {data.dissolved_v / data.vol_init:.2f}')
    ax1 = plt.subplot(spec[0])
    ax1.set_title(f'time: {sid.old_t:.2f} dissolved: {data.dissolved_v:.2f}')
    plt.axis('equal')
    plt.xlim(0, sid.n)
    plt.ylim(-1.1, sid.n + 1.1)
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
        edgecolors = 'black')
    merged_number = np.asarray(inc.plot.sum(axis = 0)).flatten()
    if np.sum(merged_number == 0) > 0:
        raise ValueError("Merge number zero")
    if np.sum((inc.plot @ edges.flow) * (edges.flow != 0) != edges.flow) > 0:
        raise ValueError("Merge number zero")
    merged_number = merged_number * (merged_number > 0) + 1 * (merged_number == 0)
    diams = inc.plot @ edges.diams / merged_number
    flow = inc.plot @ np.abs(edges.flow) / merged_number
    # in_flow = np.abs(inc.incidence.T) @ np.abs(edges.flow) / 2
    # transverse_flow = np.max(in_flow[edges.edge_list_draw], axis = 1) * edges.transversed
    # flow2 = inc.plot @ np.abs(transverse_flow) / merged_number
    # flow = np.max([flow, flow2], axis = 0)
    qs = (1 - edges.boundary_list) * np.abs(flow)
    #q2 = (1 - edges.boundary_list) * np.abs(flow) * edges.transversed
    # connectionstyle_list = []
    # for i in range(len(q1)):
    #     if q1[i]:
    #         connectionstyle_list.append("arc3,rad=0")
    #     else:
    #         connectionstyle_list.append("arc3,rad=2")
    # if np.sum(q2) != 0:
    #     print('Plotting curved')
    #print(np.max(merged_number))
    # print(graph.merged_triangles)
    verts = []
    for edge, n1, n2, n3 in graph.merged_triangles:
        if plot_data == 'd':
            if edges.diams[edge] - edges.diams_initial[edge] > sid.draw_th_d and edges.diams_initial[edge] != 0:
                verts.append((pos[n1], pos[n2], pos[n3]))
        if plot_data == 'q':
            if np.abs(flow[edge]) > sid.draw_th_q:
                verts.append((pos[n1], pos[n2], pos[n3]))
    coll = PolyCollection(verts, facecolors = 'k')
    ax1.add_collection(coll)
    if plot_data == 'd':
        qs = (1 - edges.boundary_list) * (diams - edges.diams_initial) * (diams - edges.diams_initial > 0)
        nx.draw_networkx_edges(graph, pos, edges.edge_list_draw, edge_color = 'k', \
            width = sid.ddrawconst * np.array(qs))
    elif plot_data == 'q':
        # qs = (1 - edges.boundary_list) * np.abs(flow)
        # nx.draw_networkx_edges(graph, pos, edges.edge_list_draw, edge_color = 'k', \
        #     width = sid.ddrawconst * q1, connectionstyle=connectionstyle_list, arrows = False)
        #arc_rad = 0.1
        nx.draw_networkx_edges(graph, pos, edges.edge_list_draw, edge_color = 'k', \
            width = sid.qdrawconst * qs)
    # nx.draw_networkx_edge_labels(graph, pos, edge_labels=dict(zip(edges.edge_list_draw, np.arange(0, len(edges.edge_list_draw)))), font_size = 5)
    # draw histograms with data below the network
    plt.subplot(spec[1], sharex = ax1)
    pos_x = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,0]
    slices = np.linspace(np.min(pos_x), np.max(pos_x), 102)[1:-1]
    edge_number  = np.array(data.slices[0])
    plt.plot(slices, np.array((edge_number - 2 * np.array(data.slices[1])) / edge_number), color = 'red')
    plt.plot(slices, np.array((edge_number - 2 * np.array(data.slices[-1])) / edge_number))
    plt.ylim(0, 1)
    plt.xlabel('x')
    plt.ylabel('flow focusing index')
    plt.subplots_adjust(wspace=0, hspace=0)
    # save file in the directory
    plt.savefig(sid.dirname + "/" + name)
    plt.close()


def draw_focusing2(sid: SimInputData, graph: Graph, inc: Incidence, edges: Edges, \
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
    plt.figure(figsize=(sid.figsize, sid.figsize * 1.5))
    #plt.suptitle(f'G = {sid.G:.2f} Daeff = {sid.Da_eff:.2f} noise = {sid.noise_filename}')
    spec = gridspec.GridSpec(ncols = 1, nrows = 2, height_ratios = [2, 1])
    # draw first panel for the network
    ax1 = plt.subplot(spec[0])
    #ax1.set_title(f'time: {sid.old_t:.2f} dissolved: {data.dissolved_v:.2f}')
    plt.axis('equal')
    plt.xlim(0, sid.n)
    plt.ylim(-1.1, sid.n + 1.1)
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
        qs = (1 - edges.boundary_list) * np.abs(edges.diams - edges.diams_initial) * (edges.diams > 0)

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

        nx.draw_networkx_edges(graph, pos, edges.edge_list_draw, edge_color = 'k', \
            width = sid.qdrawconst * np.array(qs) * (1 - colored_edges_r - colored_edges_g))
        nx.draw_networkx_edges(graph, pos, edges.edge_list_draw, edge_color = 'r', \
            width = sid.qdrawconst * np.array(qs) * colored_edges_r)
        nx.draw_networkx_edges(graph, pos, edges.edge_list_draw, edge_color = 'b', \
            width = sid.qdrawconst * np.array(qs) * colored_edges_g)
    else:
        nx.draw_networkx_edges(graph, pos, edges.edge_list_draw, edge_color = 'k', \
            width = sid.qdrawconst * np.array(qs))
    # nx.draw_networkx_edge_labels(graph, pos, edge_labels=dict(zip(edges.edge_list_draw, np.arange(0, len(edges.edge_list_draw)))), font_size = 5)
    # draw histograms with data below the network
    plt.subplot(spec[1], sharex = ax1)
    pos_x = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,0]
    slices = np.linspace(np.min(pos_x), np.max(pos_x), 102)[1:-1]
    edge_number  = np.array(data.slices[0])
    plt.plot(slices, np.array((edge_number - 2 * np.array(data.slices[-1])) / edge_number), color = 'red')
    plt.plot(slices, np.array((edge_number - 2 * np.array(data.slices[1])) / edge_number), '--', color = 'black')
    # plt.plot(slices, np.array((edge_number - 2 * np.array(data.slices_d[-1])) / edge_number), color = 'red')
    # plt.plot(slices, np.array((edge_number - 2 * np.array(data.slices_d[1])) / edge_number), '--', color = 'red')
    # plt.plot(slices, np.array((edge_number - 2 * np.array(data.slices_s[-1])) / edge_number), color = 'yellow')
    # plt.plot(slices, np.array((edge_number - 2 * np.array(data.slices_s[1])) / edge_number), '--', color = 'yellow')
    if colored:
        plt.vlines(slice_x, 0, 1, 'black', 'dashed')
    plt.ylim(0, 1)
    if len(data.slices) == 3:
        plt.ylabel('flow focusing index')
    else:
        plt.yticks([], [])
    plt.xlabel('x', fontsize = 60, style = 'italic')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(tight = True)
    # save file in the directory
    plt.savefig(sid.dirname + "/" + name, bbox_inches="tight")
    plt.close()

def draw_final_focusing(sid: SimInputData, graph: Graph, inc: Incidence, edges: Edges, \
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
    plt.figure(figsize=(sid.figsize, sid.figsize * 1.5))
    #plt.suptitle(f'G = {sid.G:.2f} Daeff = {sid.Da_eff:.2f} noise = {sid.noise_filename}')
    spec = gridspec.GridSpec(ncols = 1, nrows = 2, height_ratios = [2, 1])
    # draw first panel for the network
    ax1 = plt.subplot(spec[0])
    #ax1.set_title(f'time: {sid.old_t:.2f} dissolved: {data.dissolved_v:.2f}')
    plt.axis('equal')
    plt.xlim(0, sid.n)
    plt.ylim(-1.1, sid.n + 1.1)
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
        qs = (1 - edges.boundary_list) * np.abs(edges.diams - edges.diams_initial) * (edges.diams > 0)


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

        nx.draw_networkx_edges(graph, pos, edges.edge_list_draw, edge_color = 'k', \
            width = sid.qdrawconst * np.array(qs) * (1 - colored_edges_r - colored_edges_g))
        nx.draw_networkx_edges(graph, pos, edges.edge_list_draw, edge_color = 'r', \
            width = sid.qdrawconst * np.array(qs) * colored_edges_r)
        nx.draw_networkx_edges(graph, pos, edges.edge_list_draw, edge_color = 'b', \
            width = sid.qdrawconst * np.array(qs) * colored_edges_g)
    else:
        # qs = np.ma.fix_invalid(np.log(1+qs), fill_value=0)
        nx.draw_networkx_edges(graph, pos, edges.edge_list_draw, edge_color = 'k', \
            width = sid.qdrawconst * np.array(qs), alpha = np.array(qs)/np.max(qs))
    # nx.draw_networkx_edge_labels(graph, pos, edge_labels=dict(zip(edges.edge_list_draw, np.arange(0, len(edges.edge_list_draw)))), font_size = 5)
    # draw histograms with data below the network
    ax2 = plt.subplot(spec[1], sharex = ax1)
    pos_x = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,0]
    slices = np.linspace(np.min(pos_x), np.max(pos_x), 102)[1:-1]
    edge_number  = np.array(data.slices[0])
    #i_division = sid.dissolved_v_max // sid.track_every // 3
    for i, channeling in enumerate(data.slices[2:]):
        #if i % i_division == 0 and i > 0:
        plt.plot(slices, (edge_number - 2 * np.array(channeling)) / edge_number, \
                label = data.slice_times[i+1])
    plt.plot(slices, np.array((edge_number - 2 * np.array(data.slices[1])) / edge_number), '--', color = 'black')
    if colored:
        plt.vlines(slice_x, 0, 1, 'black', 'dashed')
    plt.ylim(0, 1)
    # plt.xlabel('x', fontsize = 60, style = 'italic')
    plt.xticks([],[])
    plt.ylabel('flow focusing index')
    #plt.yticks([],[])
    plt.legend(loc="lower center", mode = "expand", ncol = 5)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(tight = True)
    for spine in ax1.spines.values():
        spine.set_linewidth(5)
        spine.set_edgecolor('green')
    for spine in ax2.spines.values():
        spine.set_linewidth(5)
        spine.set_edgecolor('green')
    # save file in the directory
    plt.savefig(sid.dirname + "/" + name, bbox_inches="tight")
    plt.close()
