#%%
from networkx.readwrite import json_graph
import json
import networkx as nx
import numpy as np
from network import Graph
import scipy.sparse as spr
import matplotlib.pyplot as plt
import matplotlib
from utils import solve_equation

font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 50}

matplotlib.rc('font', **font)


def create_matrices(graph: Graph):
    """ Create incidence matrices and edges class for graph parameters.

    This function takes the network and based on its properties creates
    matrices of connections for different types of nodes and 
    It later updates the matrices in Incidence class and returns Edges class
    for easy access to the parameters of edges in the network.

    Parameters
    -------
    sid : SimInputData class object
        all config parameters of the simulation

    inc : Incidence class object
        matrices of incidence

    graph : Graph class object
        network and all its properties

    Returns
    -------
    edges : Edges class object
        all edges in network and their parameters
    """




def check_channelization(graph: Graph, incidence, flow, fracture_lens, \
        slice_x: float) -> tuple[int, float]:
        """ Calculate channelization parameter for a slice of the network.

        This function calculates the channelization parameter for a slice of
        the network perpendicular to the main direction of the flow. It checks
        how many edges take half of the total flow going through the slice. The
        function returns the exact number of edges and that number divided by
        the total number of edges in a given slice (so the percentage of edges
        taking half of the total flow in the slice).

        Parameters
        -------
        graph : Graph class object
            network and all its properties

        inc : Incidence class object
            matrices of incidence

        edges : Edges class object
            all edges in network and their parameters

        slice_x : float
            position of the slice

        Returns
        -------
        int
            number of edges taking half of the flow in the slice

        float
            percentage of edges taking half of the flow in the slice
        """
        pos_x = np.array(list(nx.get_node_attributes(graph, 'x').values()))
        # find edges crossing the given slice and their orientation - if edge
        # crosses the slice from left to right, it is marked with 1, if from
        # right to left - -1, if it doesn't cross - 0
        slice_edges = (spr.diags(flow) @ incidence > 0) \
            @ (pos_x <= slice_x) * np.abs(incidence @ (pos_x > slice_x)) \
            - (spr.diags(flow) @ incidence > 0) @ (pos_x > slice_x) \
            * np.abs(incidence @ (pos_x <= slice_x))
        # sort edges from maximum flow to minimum (taking into account
        # their orientation)
        slice_flow = np.array(sorted(slice_edges * fracture_lens \
            * np.abs(flow), reverse = True))
        fraction_flow = 0
        total_flow = np.sum(slice_flow)
        # calculate how many edges take half of the flow
        for i, edge_flow in enumerate(slice_flow):
            fraction_flow += edge_flow
            if fraction_flow > total_flow / 2:
                return (i + 1, (i + 1) / np.sum(slice_flow != 0))
        # if calculation failed, raise an error (it never should happen...)
        raise ValueError("Impossible")


def check_init_slice_channelization(graph: Graph, inc, \
    flow, fracture_lens) -> None:
    pos_x = np.array(list(nx.get_node_attributes(graph, 'x').values()))
    # slices = np.linspace(np.min(pos_x), np.max(pos_x), 120)[10:-10]
    slices = np.linspace(np.min(pos_x), np.max(pos_x), 102)[1:-1]
    channels_tab = []
    for x in slices:
        res = check_channelization(graph, inc, flow, fracture_lens, x)
        channels_tab.append(res[0] / res[1])
    return channels_tab

def check_slice_channelization(graph: Graph, inc, \
    flow, fracture_lens) -> None:
    pos_x = np.array(list(nx.get_node_attributes(graph, 'x').values()))
    # slices = np.linspace(np.min(pos_x), np.max(pos_x), 120)[10:-10]
    slices = np.linspace(np.min(pos_x), np.max(pos_x), 102)[1:-1]
    channels_tab = []
    for x in slices:
        res = check_channelization(graph, inc, flow, fracture_lens, x)
        channels_tab.append(res[0])
    return channels_tab

def plot_slice_channelization(dirname, graph: Graph, slice_tab, networks) -> None:
        """ Plots slice data from text file.

        This function loads the data from text file slices.txt and plots them
        to files slices.png, slices_no_div.png, slices_norm.png.
        """
        pos_x = np.array(list(nx.get_node_attributes(graph, 'x').values()))
        # slices = np.linspace(np.min(pos_x), np.max(pos_x), 120)[10:-10]
        slices = np.linspace(0, 1, 102)[1:-1]
        edge_number  = np.array(slice_tab[0])
        colors = ['C0', 'C1', 'C2', 'C3']
        plt.figure(figsize = (18, 5))
        plt.plot([], [], ' ', label=' ')
        plt.plot([], [], ' ', label=' ')
        plt.plot([], [], ' ', label=' ')
        plt.plot(slices, np.array((edge_number - 2 * np.array(slice_tab[1])) \
            / edge_number), linewidth = 5, color = 'black', label = '0.0')
        for i, channeling in enumerate(slice_tab[2:]):
            plt.plot(slices, (edge_number - 2 * np.array(channeling)) \
                / edge_number, label = networks[i+1][:-1], linewidth = 5, color = colors[i])
        plt.ylim(0, 1.05)
        plt.xlim(0, 1)
        plt.xlabel('x / L', fontsize = 60, style = 'italic')
        # ax2.xaxis.label.set_color('white')
        # ax2.tick_params(axis = 'x', colors='white')
        plt.title(" ")
        plt.xticks([0, 0.5, 1],[0, 0.5, 1])
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.margins(tight = True)
        #plt.ylabel('flow focusing index', fontsize = 50)
        plt.yticks([],[])
        #plt.yticks([0, 0.5, 1],['0', '0.5', '1'])
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [0,4,1,5,2,6,3,7]
        #legend = plt.legend(loc='bottom', prop={'size': 40}, mode = 'expand', frameon=False, handlelength = 0.2, borderpad = 0, handletextpad = 0.4)
        #for legobj in legend.legendHandles:
        #    legobj.set_markersize(24.0)
        legend = plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc="lower center", mode = "expand", ncol = 4, prop={'size': 40}, handlelength = 1, frameon=False, borderpad = 0, handletextpad = 0.4)
        for legobj in legend.legend_handles:
            legobj.set_linewidth(10.0)
        #spine_color = 'blue'
        # for spine in ax1.spines.values():
        #     spine.set_linewidth(5)
        #     spine.set_edgecolor(spine_color)
        # for spine in ax2.spines.values():
        #     spine.set_linewidth(5)
        #     spine.set_edgecolor(spine_color)
        # save file in the directory
        plt.savefig(dirname + "/profile.png", bbox_inches="tight", dpi = 300)
        #plt.show()
        plt.close()


def load_graph(name):
    graph = Graph.from_json_file(name)
    # copy for saving network exactly same as initial, but with evolving
    # apertures and permeabilities
    # get rid of additional edges with inf permeability from inlet/outlet node
    # (we keep them in graph_real)
    for edge in graph.edges():
        n1, n2 = edge
        #pos = list(zip(nx.get_node_attributes(graph, 'x').values(), \
        #    nx.get_node_attributes(graph, 'y').values(), \
        #    nx.get_node_attributes(graph, 'z').values()))
        if isinstance(n1, str) or isinstance(n2, str):
            if n1 == 's':
                graph.in_nodes.append(n2)
            elif n1 == 't':
                graph.out_nodes.append(n2)
            if n2 == 's':
                graph.in_nodes.append(n1)
            elif n2 == 't':
                graph.out_nodes.append(n1)
            graph.remove_edge(n1, n2)

    remove_nodes = []
    for node in graph.nodes():
        if isinstance(node, str):
            remove_nodes.append(node)
        #elif len(list(graph.neighbors(node))) < 2:
        #    remove_nodes.append(node)
        #    print ('!!!')
    for node in remove_nodes:
        graph.remove_node(node)
    # update parameters in sid based on loaded graph
    #flow = nx.get_edge_attributes(graph, 'q').values()
    lens = nx.get_edge_attributes(graph, 'length').values()
    l0 = 25#sum(lens) / len(lens)
    n_edges = len(graph.edges())
    n_nodes = len(graph.nodes())
    graph.in_vec = np.zeros(n_nodes)
    graph.out_vec = np.zeros(n_nodes)
    for node in graph.in_nodes:
        graph.in_vec[node] = 1
    for node in graph.out_nodes:
        graph.out_vec[node] = 1
    # data for standard incidence matrix (ne x nsq)
    data, row, col = [], [], []
    data_in, row_in, col_in = [], [], []
    # vectors of edges parameters (ne)
    apertures, lens, fracture_lens = [], [], []
    # data for matrix keeping connections of only middle nodes (nsq x nsq)
    for i, e in enumerate(graph.edges()):
        n1, n2 = e
        b = graph[n1][n2]['b']
        l = graph[n1][n2]['length']
        #q = graph[n1][n2]['q']
        data.append(-1)
        row.append(i)
        col.append(n1)
        data.append(1)
        row.append(i)
        col.append(n2)
        fracture_lens.append(graph[n1][n2]['area'] / b)
        apertures.append(b)
        lens.append(l)
        if n1 not in graph.in_nodes and n2 in graph.in_nodes:
            data_in.append(1)
            row_in.append(i)
            col_in.append(n1)
            data_in.append(-1)
            row_in.append(i)
            col_in.append(n2)
        elif n1 in graph.in_nodes and n2 not in graph.in_nodes:
            data_in.append(1)
            row_in.append(i)
            col_in.append(n2)
            data_in.append(-1)
            row_in.append(i)
            col_in.append(n1)
    fracture_lens = np.array(fracture_lens)
    fracture_lens /= np.average(fracture_lens)
    lens = np.array(lens) / l0
    apertures = np.array(apertures)
    b0 = sum(apertures) / len(apertures)
    incidence = spr.csr_matrix((data, (row, col)), shape=(n_edges, \
        n_nodes))
    inlet = spr.csr_matrix((data_in, (row_in, col_in)), \
        shape = (n_edges, n_nodes))
    
    return graph, incidence, fracture_lens, b0, lens, inlet

def find_flow(graph, incidence, b0, fracture_lens, lens, inlet, name):
    graph2 = Graph.from_json_file(name)
    apertures = []
    for edge in graph2.edges():
        n1, n2 = edge
        #pos = list(zip(nx.get_node_attributes(graph, 'x').values(), \
        #    nx.get_node_attributes(graph, 'y').values(), \
        #    nx.get_node_attributes(graph, 'z').values()))
        if isinstance(n1, str) or isinstance(n2, str):
            graph2.remove_edge(n1, n2)
    for i, e in enumerate(graph2.edges()):
        n1, n2 = e
        b = graph2[n1][n2]['b']
        apertures.append(b)
    apertures = np.array(apertures) / b0
    apertures = np.ones(len(apertures))
    p_matrix = incidence.T @ spr.diags(fracture_lens * apertures ** 3 / lens) @ incidence
    p_matrix = p_matrix.multiply((1 - graph.in_vec - graph.out_vec)[:, np.newaxis]) + spr.diags(graph.in_vec + graph.out_vec)
    pressure = solve_equation(p_matrix, graph.in_vec)
    q_in = np.abs(np.sum(fracture_lens * apertures ** 3 \
        / lens * (inlet @ pressure)))
    #pressure *= sid.q_in * np.sum(edges.inlet) / q_in
    pressure /= q_in
    flow = apertures ** 3 / lens * (incidence @ pressure)
    return flow

def find_flow2(graph, incidence, b0, fracture_lens, lens, inlet, name):
    graph2 = Graph.from_json_file(name)
    apertures = []
    for edge in graph2.edges():
        n1, n2 = edge
        #pos = list(zip(nx.get_node_attributes(graph, 'x').values(), \
        #    nx.get_node_attributes(graph, 'y').values(), \
        #    nx.get_node_attributes(graph, 'z').values()))
        if isinstance(n1, str) or isinstance(n2, str):
            graph2.remove_edge(n1, n2)
    for i, e in enumerate(graph2.edges()):
        n1, n2 = e
        b = graph2[n1][n2]['b']
        apertures.append(b)
    apertures = np.array(apertures) / b0
    p_matrix = incidence.T @ spr.diags(fracture_lens * apertures ** 3 / lens) @ incidence
    p_matrix = p_matrix.multiply((1 - graph.in_vec - graph.out_vec)[:, np.newaxis]) + spr.diags(graph.in_vec + graph.out_vec)
    pressure = solve_equation(p_matrix, graph.in_vec)
    q_in = np.abs(np.sum(fracture_lens * apertures ** 3 \
        / lens * (inlet @ pressure)))
    #pressure *= sid.q_in * np.sum(edges.inlet) / q_in
    pressure /= q_in
    flow = apertures ** 3 / lens * (incidence @ pressure)
    return flow
# #%%
# slices = []
# dirname = 'check/G1.00Daeff0.05/carbonate_x02/0/0/'
# networks = ['0.00', '0.10', '0.20', '0.51', '1.01']

# for net in networks:
#     name = dirname + 'network_' + net + '.json'    
#     if net == '0.00':
#         graph, incidence, fracture_lens, b0, lens, inlet = load_graph(name)
#         flow = find_flow2(graph, incidence, b0, fracture_lens, lens, inlet, name)
#         slices.append(check_init_slice_channelization(graph, incidence, flow, fracture_lens).copy())
#     else:
#         flow = find_flow2(graph, incidence, b0, fracture_lens, lens, inlet, name)
#     slices.append(check_slice_channelization(graph, incidence, flow, fracture_lens).copy())
# np.savetxt(dirname + 'slices.txt', slices)
# plot_slice_channelization(dirname, graph, slices, networks)
#%%
import os
for i in range(1, 31):
    print(i)
    slices = []
    networks = []
    dirname = f'check_new_Da/G5.00000Daeff0.00020/carbonate_x{i:02}/'
    for name in os.listdir(dirname):
        if name[:3] == 'net':
            networks.append(name[8:12])

    for net in networks:
        name = dirname + 'network_' + net + '.json'    
        if net == '0.00':
            graph, incidence, fracture_lens, b0, lens, inlet = load_graph(name)
            flow = find_flow2(graph, incidence, b0, fracture_lens, lens, inlet, name)
            slices.append(check_init_slice_channelization(graph, incidence, flow, fracture_lens).copy())
        else:
            flow = find_flow2(graph, incidence, b0, fracture_lens, lens, inlet, name)
        slices.append(check_slice_channelization(graph, incidence, flow, fracture_lens).copy())
    np.savetxt(dirname + 'slices.txt', slices)
    #plot_slice_channelization(dirname, graph, slices, networks)
# %%
import os
for i in range(1, 2):
    print(i)
    slices = []
    networks = []
    dirname = f'check_new_Da/G5.00000Daeff0.02000/carbonate_x{i:02}/'
    for name in os.listdir(dirname):
        if name[:3] == 'net':
            networks.append(name[8:12])

    for net in networks:
        name = dirname + 'network_' + net + '.json'    
        if net == '0.00':
            graph, incidence, fracture_lens, b0, lens, inlet = load_graph(name)
            flow = find_flow2(graph, incidence, b0, fracture_lens, lens, inlet, name)
            slices.append(check_init_slice_channelization(graph, incidence, flow, fracture_lens).copy())
        else:
            flow = find_flow2(graph, incidence, b0, fracture_lens, lens, inlet, name)
        slices.append(check_slice_channelization(graph, incidence, flow, fracture_lens).copy())
    #np.savetxt(dirname + 'slices.txt', slices)
    plot_slice_channelization(dirname, graph, slices, networks)
# %%
