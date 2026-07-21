from networkx.readwrite import json_graph
import json
import networkx as nx
import numpy as np
from network import Graph
import scipy.sparse as spr
import matplotlib.pyplot as plt
import matplotlib
from utils import solve_equation
import numba
from numba import njit, types, typed, prange

font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 50}

matplotlib.rc('font', **font)

def load_graph(name):
    graph = Graph.from_json_file_draw(name)
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
    fracture_lens = np.array(fracture_lens) #/ l0
    fracture_lens /= np.average(fracture_lens)
    lens = np.array(lens) / l0
    apertures = np.array(apertures)
    b0 = sum(apertures) / len(apertures)
    incidence = spr.csr_matrix((data, (row, col)), shape=(n_edges, \
        n_nodes))
    inlet = spr.csr_matrix((data_in, (row_in, col_in)), \
        shape = (n_edges, n_nodes))
    
    return graph, incidence, fracture_lens, b0, l0, lens, inlet

def find_flow(graph, incidence, b0, fracture_lens, lens, inlet, name):
    graph2 = Graph.from_json_file_draw(name)
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
    return apertures, pressure, flow

@njit
def track_particle(neigh_inc, tot_flow, inlet_flow, edge_list, out_nodes, tot_time, exp, pressure):
    time = 0
    conc = 1
    # choose inlet edge to introduce particle
    #in_edge = np.random.choice(len(inlet_flow), p = inlet_flow)

    cumulative_distribution = np.cumsum(inlet_flow)
    cumulative_distribution /= cumulative_distribution[-1]
    uniform_samples = np.random.rand(len(inlet_flow))
    index = np.searchsorted(cumulative_distribution, uniform_samples, side="right")[0]

    #n1, n2 = edge_list[in_edge]
    new_nodes = edge_list[index]
    n1 = new_nodes[0]
    n2 = new_nodes[1]
    # put particle on the end of inlet edge with lower pressure

    
    if pressure[n1] > pressure[n2]:
        node = n2
    else:
        node = n1

    # travel until particle reaches an outlet node
    while node not in out_nodes:

        # prob = np.array(prob) / np.sum(prob)


        prob = np.array([tot_flow[edge] for edge in neigh_inc[node].nonzero()[-1]])
        cumulative_distribution = np.cumsum(prob)
        cumulative_distribution /= cumulative_distribution[-1]
        uniform_samples = np.random.rand(len(prob))
        index = np.searchsorted(cumulative_distribution, uniform_samples, side="right")[0]

        # choose neighbor with probability dependent on flow
        #edge = neigh_edges[np.random.choice(len(prob), p = prob)]
        edge = neigh_inc[node].nonzero()[-1][index]
        # increase time and decrease concentration
        time += tot_time[edge]
        conc *= exp[edge]
        # if concentration is too low, reduce it to 0 (for plotting)
        if conc < 1e-40:
            conc = 0
        # change node to the chosen one
        new_nodes = edge_list[edge]
        n1 = new_nodes[0]
        n2 = new_nodes[1]
        if n1 == node:
            node = n2
        else:
            node = n1
        return time, conc



@njit
def track_particle2(neigh_inc, tot_flow, inlet_flow, edge_list, out_nodes, tot_time, exp, lens, pressure, max_neighbors):
    time = 0.0
    path_len = 0.0
    conc = 1.0

    # Select inlet edge
    cumulative_distribution = np.cumsum(inlet_flow)
    cumulative_distribution /= cumulative_distribution[-1]
    uniform_sample = np.random.rand()
    in_edge = np.searchsorted(cumulative_distribution, uniform_sample, side="right")

    # Determine starting node
    n1, n2 = edge_list[in_edge]
    node = n2 if pressure[n1] > pressure[n2] else n1

    # Buffer arrays for probabilities to avoid dynamic allocation
    prob = np.empty(max_neighbors)
    cumulative_distribution = np.empty(max_neighbors)

    time += tot_time[in_edge]
    conc *= exp[in_edge]
    path_len += lens[in_edge]

    # Begin tracking
    while node not in out_nodes:
        # Get neighboring edges and compute flow-based probabilities
        neighbors = neigh_inc[node].nonzero()[-1]
        num_neighbors = len(neighbors)
        
        # Compute probabilities for neighbors
        total_prob = 0.0
        for i in range(num_neighbors):
            prob[i] = tot_flow[neighbors[i]]
            total_prob += prob[i]
        for i in range(num_neighbors):
            prob[i] /= total_prob  # Normalize

        # Create cumulative distribution manually
        cumulative_distribution[0] = prob[0]
        for i in range(1, num_neighbors):
            cumulative_distribution[i] = cumulative_distribution[i-1] + prob[i]

        # Generate a uniform sample and select next edge
        uniform_sample = np.random.rand()
        edge_index = 0
        while edge_index < num_neighbors and uniform_sample > cumulative_distribution[edge_index]:
            edge_index += 1

        edge = neighbors[edge_index]  # Selected edge
        
        # Update time and concentration
        time += tot_time[edge]
        conc *= exp[edge]
        path_len += lens[edge]

        # Determine next node
        n1, n2 = edge_list[edge]
        node = n2 if n1 == node else n1

    return time, conc, path_len

@njit(parallel=True)
def parallel_tracking(neigh_inc, tot_flow, inlet_flow, edge_list, out_nodes, tot_time, exp, lens, pressure, max_neighbors, n_part, seeds):
    breakthrough_times = np.zeros(n_part)
    concentrations = np.zeros(n_part)
    path_lens = np.zeros(n_part)
    for i in range(n_part):
        np.random.seed(seeds[i])
        time, conc, path_len = track_particle2(neigh_inc, tot_flow, inlet_flow, edge_list, out_nodes, tot_time, exp, lens, pressure, max_neighbors)
        breakthrough_times[i] = time
        concentrations[i] = conc
        path_lens[i] = path_len
    
    return breakthrough_times, concentrations, path_lens

def track(graph, incidence, inlet, flow, fracture_lens, lens, apertures, pressure, G, Da, n_part, seeds) -> None:
    """ Perform particle tracking and collect flow/velocity data.

    This function performs particle tracking and saves them in Data class to
    later create breakthrough curves. Depending on config parameters, it
    performs standard tracking, concentration weighted tracking and tracking
    with removing particles due to reactions (where in each edge, we remove
    a tracked particle with probability dependent on reaction term in a given
    edge - we calculate how much the concentration changes in this edge
    c_out / c_in = exp(- 2 Da / (1 + G b) L / q)) and we remove the particle
    with probability p = 1 - c_out / c_in). We also collect the flow and
    velocity in the whole network.
    
    Parameters
    -------
    sid : simInputData class object
        all config parameters of the simulation

    inc : Incidence class object
        matrices of incidence

    graph : Graph class object
        network and all its properties

    edges : Edges class object
        all edges in network and their parameters
    
    data : Data class object
        physical properties of the network measured during simulation
        
    pressure : numpy ndarray
        vector of pressure in nodes
    """
    edge_list = np.array(list(graph.edges()))
    # find upstream neighbours
    neigh_inc = (spr.diags(flow) @ incidence > 0).T
    tot_flow = np.abs(flow * fracture_lens)
    tot_velocity = np.abs(flow / apertures)
    # collect data for flow and velocity
    #velocities.append(tot_velocity)
    #vol_flow.append(tot_flow)
    # calculate travel time through each edge
    tot_time = np.abs(lens / tot_velocity)
    # we introduce a particle to an inlet edge with probability proportional to
    # the flow in that edge
    inlet_flow = fracture_lens * apertures ** 3 \
        / lens * (inlet @ pressure)
    inlet_flow /= np.sum(inlet_flow)
    # standard and concentration weighted tracking
    
    # neigh_edges = neigh_inc.nonzero()[1]
    # node_idx = neigh_inc.nonzero()[0]
    # neigh_edges_idx = []
    # for node in graph.nodes():
    #     edge_idx = np.where(node_idx == node)[0][0]
    #     neigh_edges_idx.append(edge_idx)
        
    #neigh_edges = np.array(neigh_edges)
    # reaction term for calculation of concentration drop during tracking
    exp = np.exp(-np.abs(Da / (1 + G * apertures) \
        * lens / flow))
    # loop for tracking particles
    max_neighbors = np.max((neigh_inc != 0).sum(axis = 1))

    breakthrough_times, concentrations, path_lens = parallel_tracking(neigh_inc.toarray(), tot_flow, inlet_flow, edge_list, graph.out_nodes, tot_time, exp, lens, pressure, max_neighbors, n_part, seeds)
    return breakthrough_times, concentrations, path_lens


def create_bins(vals, num_bins, spacing = "log", x=[], weights=None, a_low=None, a_high=None, bin_edge = "center"):
    if a_low == None:
        a_low = 0.95 * np.min(vals)
    if a_high == None:
        a_high = np.max(vals)

    # Create bins 

    if spacing == "linear":
        x = np.linspace(a_low,a_high,num_bins+1)
    elif spacing == "log":
        if min(a_low, a_high) > 0:
            x = np.logspace(np.log10(a_low), np.log10(a_high), num_bins+1)
        else:
            x = np.logspace(-2, 2, num_bins+1, endpoint=False)
            A = np.max(x)
            B = np.min(x)
            x = (x-A) * (a_low - a_high) / (B-A) + a_high
    else: 
        print("Unknown spacing type. Using Linear spacing")
        x = np.linspace(a_low,a_high,num_bins+1)
    return x

    
def create_pdf(vals, num_bins, spacing = "log", x = [], weights=None, a_low=None, a_high=None, bin_edge = "center"):
    """  create pdf of vals 

    Parameters
    ----------
        vals : array
           array of values to be binned
        num_bins : int
            Number of bins in the pdf
        spacing : string 
            spacing for the pdf, options are linear and log
        x : array
            array of bin edges
        weights :array
            weights corresponding to vals to be used to create a weighted pdf
        a_low : float
            lower value of bin range. If no value provided 0.95*min(vals) is used
        a_high : float
            upper value of bin range. If no value is provided max(vals) is used
        bin_edge: string
            which bin edge is returned. options are left, center, and right

    Returns
    -------
        bx : array
            bin edges or centers (x values of the pdf)
        pdf : array
            values of the pdf, normalized so the Riemann sum(pdf*dx) = 1.
    """

    # Pick bin range 
    if a_low == None:
        a_low = 0.95 * np.min(vals)
    if a_high == None:
        a_high = np.max(vals)

    # Create bins 

    if spacing == "linear":
        x = np.linspace(a_low,a_high,num_bins+1)
    elif spacing == "log":
        if min(a_low, a_high) > 0:
            x = np.logspace(np.log10(a_low), np.log10(a_high), num_bins+1)
        else:
            x = np.logspace(-2, 2, num_bins+1, endpoint=False)
            A = np.max(x)
            B = np.min(x)
            x = (x-A) * (a_low - a_high) / (B-A) + a_high
    else: 
        print("Unknown spacing type. Using Linear spacing")
        x = np.linspace(a_low,a_high,num_bins+1)

    # Create PDF
    pdf, bin_edges = np.histogram(vals, bins=x, weights=weights, density=True)

    # Return arrays of the same size
    if bin_edge == "left":
        return bin_edges[:-1],pdf

    elif bin_edge == "right":
        return bin_edges[1:],pdf

    elif bin_edge == "center":
        bx = bin_edges[:-1] + 0.5*np.diff(bin_edges)
        return bx, pdf

    else: 
        print("Unknown bin edge type {0}. Returning left edges".format(bin_edge))
        return bin_edge[:-1],pdf



def plot_tracking_average(tracking, concentration_tracking, reactive_tracking, dirname, networks, nbins) -> None:
    ''' Plot data from text file params.txt and save the plot to params.png.

    Parameters
    -------
    sid : SimInputData class object
        all config parameters of the simulation, here we use attributes:
        dirname - directory of current simulation
    '''
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (45, 15))
    ax1.set_title('BTC')
    ax2.set_title('BTC concentration-weighted')
    ax3.set_title('BTC with particle removal')
    ax1.set_xlabel('time / average time')
    ax2.set_xlabel('time / average time')
    ax3.set_xlabel('time / average time')
    ax1.set_ylabel('probability density')
    bx1 = create_bins(tracking[-1] / np.average(tracking[-1]), nbins)
    bx3 = create_bins(reactive_tracking[-1] / np.average(reactive_tracking[-1]), nbins)
    colors = ['black', 'C0', 'C1', 'C2', 'C3', 'C4']
    ymin = 10 ** (-7)
    ymax = 10 ** (1)
    for i, time in enumerate(networks):
        bx11, pdf = create_pdf(tracking[i] / np.average(tracking[i]), nbins, x = bx1)
        ax1.loglog(bx11, pdf, "o", alpha = 1, markersize=12, label = time[:-1], color = colors[i])
        ax1.set_ylim(ymin, ymax)
        bx12, pdf = create_pdf(tracking[i] / np.average(tracking[i]), nbins, x = bx1, \
            weights = concentration_tracking[i])
        ax2.loglog(bx12, pdf, "o", alpha = 1, markersize=12, label = time[:-1], color = colors[i])
        ax2.set_ylim(ymin, ymax)
        bx13, pdf = create_pdf(reactive_tracking[i] / np.average(reactive_tracking[i]), nbins, x = bx3)
        ax3.loglog(bx13, pdf, "o", alpha = 1, markersize=12, label = time[:-1], color = colors[i])
        ax3.set_ylim(ymin, ymax)
    #legend = ax3.legend(loc='center right', bbox_to_anchor=(1.05, 0.5), prop={'size': 40}, frameon=False, handlelength = 0.2, borderpad = 0, handletextpad = 0.4)
    legend = ax3.legend(loc='lower center', prop={'size': 40}, mode = 'expand', ncol = 5, frameon=False, handlelength = 0.2, borderpad = 0, handletextpad = 0.4)
    for legobj in legend.legendHandles:
        legobj.set_markersize(24.0)
    plt.savefig(dirname + f'/track.png', bbox_inches="tight", transparent = False)
    plt.close()

def plot_tracking(tracking, concentration_tracking, dirname, networks, nbins) -> None:
    ''' Plot data from text file params.txt and save the plot to params.png.

    Parameters
    -------
    sid : SimInputData class object
        all config parameters of the simulation, here we use attributes:
        dirname - directory of current simulation
    '''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (30, 15))
    ax1.set_title('BTC')
    ax2.set_title('BTC concentration-weighted')
    ax1.set_xlabel('time')
    ax2.set_xlabel('time')
    ax1.set_ylabel('probability density')
    bx1 = create_bins(tracking[-1] / np.average(tracking[-1]), nbins)
    colors = ['black', 'C0', 'C1', 'C2', 'C3', 'C4']
    ymin = 10 ** (-7)
    ymax = 10 ** (1)
    xmin = 10 ** (-2)
    xmax = 10 ** (2)
    for i, time in enumerate(networks):
        bx11, pdf = create_pdf(tracking[i], nbins, x = bx1)
        ax1.loglog(bx11, pdf, "o", alpha = 1, markersize=12, label = time[:-1], color = colors[i])
        #ax1.set_xlim(xmin, xmax)
        ax1.set_ylim(ymin, ymax)
        bx12, pdf = create_pdf(tracking[i], nbins, x = bx1, \
            weights = concentration_tracking[i])
        ax2.loglog(bx12, pdf, "o", alpha = 1, markersize=12, label = time[:-1], color = colors[i])
        #ax2.set_xlim(xmin, xmax)
        ax2.set_ylim(ymin, ymax)
    #legend = ax3.legend(loc='center right', bbox_to_anchor=(1.05, 0.5), prop={'size': 40}, frameon=False, handlelength = 0.2, borderpad = 0, handletextpad = 0.4)
    legend = ax1.legend(loc='lower center', prop={'size': 40}, mode = 'expand', ncol = 5, frameon=False, handlelength = 0.2, borderpad = 0, handletextpad = 0.4)
    for legobj in legend.legendHandles:
        legobj.set_markersize(24.0)
    plt.savefig(dirname + f'/track_num.png', bbox_inches="tight", transparent = False)
    plt.close()

def plot_tracking_standard(tracking, concentration_tracking, dirname, networks, nbins) -> None:
    ''' Plot data from text file params.txt and save the plot to params.png.

    Parameters
    -------
    sid : SimInputData class object
        all config parameters of the simulation, here we use attributes:
        dirname - directory of current simulation
    '''
    fig, ax  = plt.subplots(1, 1, figsize = (10, 10))
    #ax.set_title('BTC')
    ax.set_xlabel('time / peak arrival time')
    #ax2.set_xlabel('time')
    ax.set_ylabel('probability density')
    bx1 = create_bins(tracking[-1], nbins)
    colors = ['black', 'C3']
    ymin = 10 ** (-7)
    ymax = 10 ** (1)
    xmin = 10 ** (-2)
    xmax = 10 ** (2)

    bx11, pdf = create_pdf(tracking[0], nbins, x = bx1)
    ax.loglog(bx11, pdf, "o", alpha = 1, markersize=12, label = networks[0][:-1], color = colors[0])
    bx11, pdf = create_pdf(tracking[-1], nbins, x = bx1)
    ax.loglog(bx11, pdf, "o", alpha = 1, markersize=12, label = networks[-1][:-1], color = colors[-1])
    #plt.plot(bx11, 100000 * bx11 ** (-3))
    #legend = ax3.legend(loc='center right', bbox_to_anchor=(1.05, 0.5), prop={'size': 40}, frameon=False, handlelength = 0.2, borderpad = 0, handletextpad = 0.4)
    legend = ax.legend(loc='lower left', prop={'size': 40}, mode = 'expand', frameon=False, handlelength = 0.2, borderpad = 0, handletextpad = 0.4)
    for legobj in legend.legendHandles:
        legobj.set_markersize(24.0)
    plt.savefig(dirname + f'/track_num2.png', bbox_inches="tight", transparent = False)
    plt.close()



def find_flow2(graph, incidence, b0, fracture_lens, lens, inlet, name):
    graph2 = Graph.from_json_file_draw(name)
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
    return apertures, pressure, flow


# import os
# n_parts = 100000
# np.random.seed(1234)
# seeds = np.random.randint(0, n_parts, size = n_parts)
# for i in range(1, 31):
#     print(i)
#     networks = []
#     dirname = f'check_new_Da/G0.10000Daeff0.02000/carbonate_x{i:02}/'
#     for name in os.listdir(dirname):
#         if name[:3] == 'net':
#             networks.append(name[8:12])
#     tracking_list = []
#     c_tracking_list = []
#     r_tracking_list = []
#     pl_tracking_list = []
#     nbins = 100
#     G = 0.1
#     Da = 0.02

#     for net in networks:
#         name = dirname + 'network_' + net + '.json'    
#         if net == '0.00':
#             graph, incidence, fracture_lens, b0, l0, lens, inlet = load_graph(name)
#             apertures, pressure, flow = find_flow(graph, incidence, b0, fracture_lens, lens, inlet, name)
#         else:
#             apertures, pressure, flow = find_flow(graph, incidence, b0, fracture_lens, lens, inlet, name)
#         tracking, c_tracking, path_lens = track(graph, incidence, inlet, flow, fracture_lens, lens, apertures, pressure, G, Da, n_parts, seeds)
#         tracking_list.append(tracking)
#         c_tracking_list.append(c_tracking)
#         pl_tracking_list.append(path_lens * l0)
#     np.savetxt(dirname + 'tracks_num.txt', tracking_list)
#     np.savetxt(dirname + 'c_tracks_num.txt', c_tracking_list)
#     np.savetxt(dirname + 'pl_tracks_num.txt', pl_tracking_list)

# import os
# n_parts = 100000
# np.random.seed(1234)
# seeds = np.random.randint(0, n_parts, size = n_parts)
# #for i in range(1, 31):
#     print(i)
#     networks = []
#     dirname = f'check_new_Da/G0.10000Daeff0.02000/carbonate_x{i:02}/'
#     for name in os.listdir(dirname):
#         if name[:3] == 'net':
#             networks.append(name[8:12])
#     tracking_list = []
#     c_tracking_list = []
#     r_tracking_list = []
#     pl_tracking_list = []
#     nbins = 100
#     G = 0.1
#     Da = 0.02

#     for net in networks:
#         name = dirname + 'network_' + net + '.json'    
#         if net == '0.00':
#             graph, incidence, fracture_lens, b0, l0, lens, inlet = load_graph(name)
#             apertures, pressure, flow = find_flow(graph, incidence, b0, fracture_lens, lens, inlet, name)
#         else:
#             apertures, pressure, flow = find_flow(graph, incidence, b0, fracture_lens, lens, inlet, name)
#         tracking, c_tracking, path_lens = track(graph, incidence, inlet, flow, fracture_lens, lens, apertures, pressure, G, Da, n_parts, seeds)
#         tracking_list.append(tracking)
#         c_tracking_list.append(c_tracking)
#         pl_tracking_list.append(path_lens * l0)
#     np.savetxt(dirname + 'tracks_num.txt', tracking_list)
#     np.savetxt(dirname + 'c_tracks_num.txt', c_tracking_list)
#     np.savetxt(dirname + 'pl_tracks_num.txt', pl_tracking_list)

import os
n_parts = 100000
np.random.seed(1234)
seeds = np.random.randint(0, n_parts, size = n_parts)
G = 5.
Da = 0.0002
dirname = f'oman_dfn_v3/G5.0000Daeff0.0002/1/'
network = "network_0.10"
name = dirname + network + ".json"
graph, incidence, fracture_lens, b0, l0, lens, inlet = load_graph(name)
apertures, pressure, flow = find_flow(graph, incidence, b0, fracture_lens, lens, inlet, name)
tracking, c_tracking, path_lens = track(graph, incidence, inlet, flow, fracture_lens, lens, apertures, pressure, G, Da, n_parts, seeds)
np.savetxt(dirname + network + '_tracking.txt', tracking)
np.savetxt(dirname + network + '_c_tracking.txt', c_tracking)
np.savetxt(dirname + network + '_pl_tracking.txt', path_lens)
