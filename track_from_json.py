#%%
from networkx.readwrite import json_graph
import json
import networkx as nx
import numpy as np
from network import Graph
import scipy.sparse as spr
import scipy.sparse.linalg as sprlin
import matplotlib.pyplot as plt
import matplotlib
from utils import solve_equation

font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 50}

matplotlib.rc('font', **font)

"""
This scripts allows to reproduce figures from publication "Quantifying
Dissolution Dynamics in Porous Media Using a Spatial Flow Focusing Profile".
Enter parameters for a given figure (most notably network_name and profile_name)
and run the script.
"""
import networkx as nx
import vtk
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from matplotlib import gridspec
from vtk.numpy_interface import dataset_adapter as dsa

# name of the VTK file with the network
network_name = 'fig3c1'
# name of the TXT file with the profiles
profile_name = 'fig3c1'
# type of the plot - flow or diameters
plot_type = 'q' # 'd'
# scaling of edge widths (for better visualization)
scaling = 1 # 1/3, 3


def load_VTK(file_name: str) -> dict:
    """
    Load fields from VTK file into dictionary.
    """
    scalars = {} # dictionary where the data of VTK file
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(file_name)
    reader.Update()
    data = dsa.WrapDataObject(reader.GetOutput())
    points = np.array(data.Points)
    pores = np.array(data.GetLines().GetData())
    no_of_pores = data.GetNumberOfCells()
    pores_array = np.array(pores)
    list_of_pore_connection = []
    for i in range(no_of_pores):
        list_of_pore_connection.append([pores_array[j] \
            for j in range(i*3, i*3+3)])
    scalars['Cell_Nodes'] = list_of_pore_connection;
    no_of_fields = len(data.CellData.keys())
    scalars['Points'] = points
    for i in range(no_of_fields): 
        name_of_field = data.CellData.keys()[i]
        scalars[name_of_field] = np.array(data.CellData[name_of_field])
    return scalars

def load_graph(network_name) -> Graph:
    """
    Build network from VTK file and draw it.
    """
    scalars = load_VTK(network_name + '.vtk')
    nkw = len(scalars['Points'])
    n = int(nkw ** 0.5)
    in_nodes = list(range(n))
    out_nodes = list(range(nkw - n, nkw))
    G_edges, diams, flow, lens = [], [], [], [] 
    # load edge data into network

    for i,e in enumerate(scalars['Cell_Nodes']):
        n1 = e[1]
        n2 = e[2]
        d = scalars['d'][i]
        q = scalars['q'][i]
        l = scalars['l'][i]
        d_init = scalars['d0'][i]
        if d != 0:
            # do not include boundary edges
            if not ((n1 < n and n2 >= nkw - n) or (n2 < n and n1 >= nkw - n)):
                G_edges.append((n1, n2))
                diams.append(d)
                flow.append(q)
                lens.append(l)
    pos = []
    for node in scalars['Points']:
        pos.append((node[0], node[1]))
    diams, flow, lens = np.array(diams), np.array(flow), np.array(lens)
    graph = Graph()
    graph.add_nodes_from(list(range(nkw)))
    graph.add_edges_from(G_edges)
    for node in graph.nodes:
        graph.nodes[node]["pos"] = pos[node]
    for i, edge in enumerate(graph.edges()):
        graph[edge[0]][edge[1]]['d'] = diams[i]
        graph[edge[0]][edge[1]]['l'] = lens[i]
        graph[edge[0]][edge[1]]['q'] = flow[i]
    graph.in_nodes = in_nodes
    graph.out_nodes = out_nodes
    graph.in_vec = np.zeros(nkw)
    graph.out_vec = np.zeros(nkw)
    for node in graph.in_nodes:
        graph.in_vec[node] = 1
    for node in graph.out_nodes:
        graph.out_vec[node] = 1
    return graph

def load_all(name):
    graph = load_graph(name)
    n_nodes = len(graph.nodes())
    n_edges = len(graph.edges())
    # data for standard incidence matrix (ne x nsq)
    data, row, col = [], [], []
    data_in, row_in, col_in = [], [], []
    # vectors of edges parameters (ne)
    diams, lens, flows = [], [], []
    # data for matrix keeping connections of only middle nodes (nsq x nsq)
    in_edges = np.zeros(n_edges)
    for i, e in enumerate(graph.edges()):
        n1, n2 = e
        d = graph[n1][n2]['d']
        l = graph[n1][n2]['l']
        q = graph[n1][n2]['q']
        data.append(-1)
        row.append(i)
        col.append(n1)
        data.append(1)
        row.append(i)
        col.append(n2)
        flows.append(q)
        diams.append(d)
        lens.append(l)
        if n1 not in graph.in_nodes and n2 in graph.in_nodes:
            data_in.append(1)
            row_in.append(i)
            col_in.append(n1)
            data_in.append(-1)
            row_in.append(i)
            col_in.append(n2)
            in_edges[i] = 1
        elif n1 in graph.in_nodes and n2 not in graph.in_nodes:
            data_in.append(1)
            row_in.append(i)
            col_in.append(n2)
            data_in.append(-1)
            row_in.append(i)
            col_in.append(n1)
            in_edges[i] = 1
    lens = np.array(lens)
    diams = np.array(diams)
    flows = np.array(flows)
    incidence = spr.csr_matrix((data, (row, col)), shape=(n_edges, \
        n_nodes))
    inlet = spr.csr_matrix((data_in, (row_in, col_in)), \
        shape = (n_edges, n_nodes))
    
    p_matrix = incidence.T @ spr.diags(diams ** 4 / lens) \
        @ incidence
    p_matrix = p_matrix.multiply((1 - graph.in_vec - graph.out_vec)[:, np.newaxis]) + spr.diags(graph.in_vec + graph.out_vec)
    diag = p_matrix.diagonal()
    # fix for nodes with no connections
    for i, node in enumerate(diag):
        if node == 0:
            diag[i] = 1
    # replace diagonal
    p_matrix.setdiag(diag)
    # solve matrix @ pressure = pressure_b
    pressure = sprlin.spsolve(p_matrix, graph.in_vec)
    # normalize pressure in inlet nodes to match condition for constant inlet
    # flow
    q_in = np.abs(np.sum(diams ** 4 / lens * (inlet \
        @ pressure)))
    pressure *= 100 / q_in
    flows = diams ** 4 / lens * (incidence @ pressure)
    return graph, incidence, diams, lens, flows, in_edges, pressure

def track(graph, incidence, flow, diams, lens, in_edges, pressure, G, Da, n_parts, track_type) -> None:
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
    edge_list = list(graph.edges())
    breakthrough_times = []
    concentrations = []
    tracking = []
    concentration_tracking = []
    # find upstream neighbours
    neigh_inc = (spr.diags(flow) @ incidence > 0).T
    tot_flow = np.abs(flow)
    tot_velocity = np.abs(flow / diams ** 2)
    # collect data for flow and velocity
    #velocities.append(tot_velocity)
    #vol_flow.append(tot_flow)
    # calculate travel time through each edge
    tot_time = np.abs(lens / tot_velocity)
    # we introduce a particle to an inlet edge with probability proportional to
    # the flow in that edge
    if track_type == 'uniform':
        inlet_flow = in_edges * (flow != 0)
    else:
        inlet_flow = in_edges * np.abs(flow)
    inlet_flow /= np.sum(inlet_flow)
    # standard and concentration weighted tracking
    
    # reaction term for calculation of concentration drop during tracking
    exp = np.exp(-np.abs(Da / (1 + G * diams) \
        * diams * lens / flow))
    # loop for tracking particles
    for _ in range(n_parts):
        time = 0
        conc = 1
        # choose inlet edge to introduce particle
        in_edge = np.random.choice(len(inlet_flow), p = inlet_flow)
        n1, n2 = edge_list[in_edge]
        # put particle on the end of inlet edge with lower pressure
        if pressure[n1] > pressure[n2]:
            node = n2
        else:
            node = n1
        # travel until particle reaches an outlet node
        while node not in graph.out_nodes:
            prob = []
            # get edge indices to neighbors of nodes
            neigh_edges = neigh_inc[node].nonzero()[1]
            for edge in neigh_edges:
                prob.append(tot_flow[edge])
            prob = np.array(prob) / np.sum(prob)
            # choose neighbor with probability dependent on flow
            edge = neigh_edges[np.random.choice(len(prob), p = prob)]
            # increase time and decrease concentration
            time += tot_time[edge]
            conc *= exp[edge]
            # if concentration is too low, reduce it to 0 (for plotting)
            if conc < 1e-40:
                conc = 0
            # change node to the chosen one
            n1, n2 = edge_list[edge]
            if n1 == node:
                node = n2
            else:
                node = n1
        breakthrough_times.append(time)
        concentrations.append(conc)

    #reactive tracking
    # scale exponent so as not to kill particles too fast
    exp = np.exp(-np.abs(Da / (1 + G * diams) \
        * diams * lens / flow) / 10)
    reactive_breakthrough_times = []
    # loop for tracking particles
    while len(reactive_breakthrough_times) < n_parts:
        time = 0
        conc = 1
        # choose inlet edge to introduce particle
        in_edge = np.random.choice(len(inlet_flow), p = inlet_flow)
        n1, n2 = edge_list[in_edge]
        # put particle on the end of inlet edge with lower pressure
        if pressure[n1] > pressure[n2]:
            node = n2
        else:
            node = n1
        # travel until particle is killed or reaches an outlet node
        flag = True
        while flag:
            prob = []
            # get edge indices to neighbors of nodes
            neigh_edges = neigh_inc[node].nonzero()[1]
            for edge in neigh_edges:
                prob.append(tot_flow[edge])
            prob = np.array(prob) / np.sum(prob)
            # choose neighbor with probability dependent on flow
            edge = neigh_edges[np.random.choice(len(prob), p = prob)]
            time += tot_time[edge]
            
            # kill particle with probability depending on amount of
            # reaction in a given edge
            if np.random.rand() < conc * (1 - exp[edge]):
                # if particle is killed, break loop and skip it in data
                break
            conc *= exp[edge]
            n1, n2 = edge_list[edge]
            # change node to the chosen one
            if n1 == node:
                node = n2
            else:
                node = n1
            # if particle reached outlet, end loop
            if node in graph.out_nodes:
                flag = False
        # if particle reached outlet, include it in data
        if not flag:
            reactive_breakthrough_times.append(time)
    # collect reactive tracking times
    
    return breakthrough_times, concentrations, reactive_breakthrough_times

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



def plot_tracking(tracking, concentration_tracking, reactive_tracking, dirname, networks, nbins) -> None:
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
    ax1.set_xlabel('time')
    ax2.set_xlabel('time')
    ax3.set_xlabel('time')
    ax1.set_ylabel('probability density')
    bx1 = create_bins(tracking[-1], nbins)
    bx3 = create_bins(reactive_tracking[-1], nbins)
    colors = ['black', 'C0', 'C1', 'C2', 'C3', 'C4']
    #ymin = 10 ** (-10)
    #ymax = 10 ** (-1)
    for i, time in enumerate(networks):
        bx11, pdf = create_pdf(tracking[i], nbins, x = bx1)
        ax1.loglog(bx11, pdf, "o", alpha = 1, markersize=12, label = time, color = colors[i])
        #ax1.set_ylim(ymin, ymax)
        bx12, pdf = create_pdf(tracking[i], nbins, x = bx1, \
            weights = concentration_tracking[i])
        ax2.loglog(bx12, pdf, "o", alpha = 1, markersize=12, label = time, color = colors[i])
        #ax2.set_ylim(ymin, ymax)
        bx13, pdf = create_pdf(reactive_tracking[i], nbins, x = bx3)
        ax3.loglog(bx13, pdf, "o", alpha = 1, markersize=12, label = time, color = colors[i])
        #ax3.set_ylim(ymin, ymax)
    #legend = ax3.legend(loc='center right', bbox_to_anchor=(1.05, 0.5), prop={'size': 40}, frameon=False, handlelength = 0.2, borderpad = 0, handletextpad = 0.4)
    legend = ax3.legend(loc='lower center', prop={'size': 40}, mode = 'expand', ncol = 5, frameon=False, handlelength = 0.2, borderpad = 0, handletextpad = 0.4)
    for legobj in legend.legend_handles:
        legobj.set_markersize(24.0)
    plt.savefig(dirname + f'/track.png', bbox_inches="tight")
    plt.close()


#%%
dirname = 'rect100/G5.00Daeff0.01/1/'
networks = ['0.00', '1.00', '2.01', '5.03', '10.11']
tracking_list = []
c_tracking_list = []
r_tracking_list = []
nbins = 100
G = 5.00
Da = 0.005
#%%

name = dirname + 'network'
graph, incidence, diams, lens, flow, in_edges, pressure = load_all(name)
#%%
n_parts = 2000
for net in networks:
    name = dirname + 'network_' + net
    graph, incidence, diams, lens, flow, in_edges, pressure = load_all(name)
    tracking, c_tracking, r_tracking = track(graph, incidence, flow, diams, lens, in_edges, pressure, G, Da, n_parts, 'uniform')
    tracking_list.append(tracking)
    c_tracking_list.append(c_tracking)
    r_tracking_list.append(r_tracking)
np.savetxt(dirname + 'tracks_u.txt', tracking_list)
np.savetxt(dirname + 'c_tracks_u.txt', c_tracking_list)
np.savetxt(dirname + 'r_tracks_u.txt', r_tracking_list)

#%%    
nbins = 100
plot_tracking2(tracking_list, c_tracking_list, r_tracking_list, dirname, networks, nbins)
#%%
import os
for i in range(1, 31):
    print(i)
    networks = []
    dirname = f'check/G5.00Daeff0.01/carbonate_x{i:02}/'
    for name in os.listdir(dirname):
        if name[:3] == 'net':
            networks.append(name[8:12])
    tracking_list = []
    c_tracking_list = []
    r_tracking_list = []
    nbins = 100
    G = 5.00
    Da = 0.005

    for net in networks:
        name = dirname + 'network_' + net + '.json'    
        if net == '0.00':
            graph, incidence, fracture_lens, b0, lens, inlet = load_graph(name)
            apertures, pressure, flow = find_flow(graph, incidence, b0, fracture_lens, lens, inlet, name)
        else:
            apertures, pressure, flow = find_flow(graph, incidence, b0, fracture_lens, lens, inlet, name)
        tracking, c_tracking, r_tracking = track(graph, incidence, inlet, flow, fracture_lens, lens, apertures, pressure, G, Da)
        tracking_list.append(tracking)
        c_tracking_list.append(c_tracking)
        r_tracking_list.append(r_tracking)
    np.savetxt(dirname + 'tracks.txt', tracking_list)
    np.savetxt(dirname + 'c_tracks.txt', c_tracking_list)
    #np.savetxt(dirname + 'r_tracks.txt', r_tracking_list)


# %%
tracking_list_tot = np.empty((3, 0)).tolist()
c_tracking_list_tot = np.empty((3, 0)).tolist()
r_tracking_list_tot = np.empty((3, 0)).tolist()
for i in range(1, 2):
    dirname = f'grl/c/'
    #dirname = f'check/G5.00Daeff0.50/carbonate_x{i:02}/'
    f = open(dirname + 'tracks.txt', 'r')
    data = np.loadtxt(f)
    f = open(dirname + 'c_tracks.txt', 'r')
    data_c = np.loadtxt(f)
    f = open(dirname + 'r_tracks.txt', 'r')
    data_r = np.loadtxt(f)
    for j, _ in enumerate(data):
        tracking_list_tot[j].extend(data[j] / np.average(data[j]))
        c_tracking_list_tot[j].extend(data_c[j] / np.average(data_c[j]))
        r_tracking_list_tot[j].extend(data_r[j] / np.average(data_r[j]))
#%%
nbins = 100
#networks = ['0.00', '0.10', '0.20', '0.50', '1.00']
#dirname = f'check/G5.00Daeff0.50/'
plot_tracking(tracking_list_tot, c_tracking_list_tot, r_tracking_list_tot, dirname, networks, nbins)
# %%
def plot_tracking2(tracking, concentration_tracking, reactive_tracking, dirname, networks, nbins) -> None:
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
    bx1 = create_bins(tracking[-1], nbins)
    bx3 = create_bins(reactive_tracking[-1], nbins)
    colors = ['black', 'C0', 'C1', 'C2', 'C3', 'C4']
    #ymin = 10 ** (-10)
    #ymax = 10 ** (-1)
    for i, time in enumerate(networks):
        data = np.array(tracking[i]) / np.average(tracking[i])
        data_r = np.array(reactive_tracking[i]) / np.average(reactive_tracking[i])
        bx11, pdf = create_pdf(data, nbins, x = bx1)
        ax1.loglog(bx11, pdf, "o", alpha = 1, markersize=12, label = time, color = colors[i])
        #ax1.set_ylim(ymin, ymax)
        bx12, pdf = create_pdf(data, nbins, x = bx1, \
            weights = concentration_tracking[i])
        ax2.loglog(bx12, pdf, "o", alpha = 1, markersize=12, label = time, color = colors[i])
        #ax2.set_ylim(ymin, ymax)
        bx13, pdf = create_pdf(data_r, nbins, x = bx3)
        ax3.loglog(bx13, pdf, "o", alpha = 1, markersize=12, label = time, color = colors[i])
        #ax3.set_ylim(ymin, ymax)
    #legend = ax3.legend(loc='center right', bbox_to_anchor=(1.05, 0.5), prop={'size': 40}, frameon=False, handlelength = 0.2, borderpad = 0, handletextpad = 0.4)
    legend = ax3.legend(loc='lower center', prop={'size': 40}, mode = 'expand', ncol = 5, frameon=False, handlelength = 0.2, borderpad = 0, handletextpad = 0.4)
    for legobj in legend.legend_handles:
        legobj.set_markersize(24.0)
    plt.savefig(dirname + f'/track2.png', bbox_inches="tight")
    plt.close()

# %%
dirname = f'rect100/G5.00Daeff0.05/1/template/10/'
networks = ['0.0', '1.0', '2.0', '5.0', '10.0']
#dirname = f'check/G5.00Daeff0.50/carbonate_x{i:02}/'
f = open(dirname + 'track.txt', 'r')
data = np.loadtxt(f)
f = open(dirname + 'c_track.txt', 'r')
data_c = np.loadtxt(f)
f = open(dirname + 'r_track.txt', 'r')
data_r = np.loadtxt(f)
nbins = 50
plot_tracking2(data, data_c, data_r, dirname, networks, nbins)
# %%