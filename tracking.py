""" Perform particle tracking.

This module contains function for particle tracking - standard, weighted by
the concentration of reactant or with removing particles due to reaction.

Notable functions
-------
track(SimInputData, Incidence, Graph, Edges, Data, numpy ndarray) -> None
    perform tracking and collect relative data
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as spr


from config import SimInputData
from data import Data
from draw_net import draw_particles
from network import Graph, Edges
from incidence import Incidence

font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 50}

matplotlib.rc('font', **font)

def track(sid: SimInputData, graph: Graph, inc: Incidence, edges: Edges, data: Data, pressure: np.ndarray) -> None:
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
    breakthrough_times = []
    n_part = sid.n_tracking
    concentrations = []
    # find upstream neighbours
    neigh_inc = (spr.diags(edges.flow) @ inc.incidence > 0).T
    tot_flow = np.abs(edges.flow)
    tot_velocity = np.abs(edges.flow / edges.diams ** 2)
    # collect data for flow and velocity
    #velocities.append(tot_velocity)
    #vol_flow.append(tot_flow)
    # calculate travel time through each edge
    tot_time = np.abs(edges.lens / tot_velocity)
    # we introduce a particle to an inlet edge with probability proportional to
    # the flow in that edge
    #inlet_flow = edges.inlet * np.abs(edges.flow)
    inlet_flow = edges.inlet * np.ones_like(edges.flow)
    inlet_flow /= np.sum(inlet_flow)
    # standard and concentration weighted tracking
    
    # reaction term for calculation of concentration drop during tracking
    exp = np.exp(-np.abs(sid.Da / (1 + sid.G * edges.diams) \
        * edges.diams * edges.lens / edges.flow))
    # loop for tracking particles
    flag = True
    locations = []
    for _ in range(n_part):
        if _ > sid.n_tracking / 10 and flag:
            time_loc = np.average(breakthrough_times) / 3
            flag = 0
        time = 0
        conc = 1
        # choose inlet edge to introduce particle
        in_edge = np.random.choice(len(inlet_flow), p = inlet_flow)
        n1, n2 = edges.edge_list[in_edge]
        # put particle on the end of inlet edge with lower pressure
        if pressure[n1] > pressure[n2]:
            node = n2
        else:
            node = n1
        # travel until particle reaches an outlet node
        flag2 = True
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
            if _ > sid.n_tracking / 10:
                if flag2 and time > time_loc:
                    locations.append(node)
                    flag2 = False
            conc *= exp[edge]
            # if concentration is too low, reduce it to 0 (for plotting)
            if conc < 1e-40:
                conc = 0
            # change node to the chosen one
            n1, n2 = edges.edge_list[edge]
            if n1 == node:
                node = n2
            else:
                node = n1
        breakthrough_times.append(time)
        concentrations.append(conc)

    #reactive tracking
    # scale exponent so as not to kill particles too fast
    exp = np.exp(-np.abs(sid.Da / (1 + sid.G * edges.diams) \
        * edges.diams * edges.lens / edges.flow) / 10)
    reactive_breakthrough_times = []
    # loop for tracking particles
    # while len(reactive_breakthrough_times) < n_part:
    #     time = 0
    #     conc = 1
    #     # choose inlet edge to introduce particle
    #     in_edge = np.random.choice(len(inlet_flow), p = inlet_flow)
    #     n1, n2 = edges.edge_list[in_edge]
    #     # put particle on the end of inlet edge with lower pressure
    #     if pressure[n1] > pressure[n2]:
    #         node = n2
    #     else:
    #         node = n1
    #     # travel until particle is killed or reaches an outlet node
    #     flag = True
    #     while flag:
    #         prob = []
    #         # get edge indices to neighbors of nodes
    #         neigh_edges = neigh_inc[node].nonzero()[1]
    #         for edge in neigh_edges:
    #             prob.append(tot_flow[edge])
    #         prob = np.array(prob) / np.sum(prob)
    #         # choose neighbor with probability dependent on flow
    #         edge = neigh_edges[np.random.choice(len(prob), p = prob)]
    #         time += tot_time[edge]
            
    #         # kill particle with probability depending on amount of
    #         # reaction in a given edge
    #         if np.random.rand() < conc * (1 - exp[edge]):
    #             # if particle is killed, break loop and skip it in data
    #             break
    #         conc *= exp[edge]
    #         n1, n2 = edges.edge_list[edge]
    #         # change node to the chosen one
    #         if n1 == node:
    #             node = n2
    #         else:
    #             node = n1
    #         # if particle reached outlet, end loop
    #         if node in graph.out_nodes:
    #             flag = False
    #     # if particle reached outlet, include it in data
    #     if not flag:
    #         reactive_breakthrough_times.append(time)
    # collect reactive tracking times
    
    data.breakthrough_times.append(breakthrough_times)
    data.concentrations.append(concentrations)
    data.reactive_breakthrough_times.append(reactive_breakthrough_times)
    data.track_times.append("{0}".format(str(round(data.dissolved_v, 1) if data.dissolved_v % 1 else int(data.dissolved_v))))
    draw_particles(sid, graph, edges, locations, \
                    f'q_particles_{data.dissolved_v:.1f}.jpg', 'q')


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



def plot_tracking(data: Data, nbins) -> None:
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
    ax1.set_xlabel('normalized time')
    ax2.set_xlabel('normalized time')
    ax3.set_xlabel('normalized time')
    ax1.set_ylabel('probability density')
    bx1 = create_bins(data.breakthrough_times[-1] / np.median(data.breakthrough_times[-1]), nbins)
    #bx3 = create_bins(data.reactive_breakthrough_times[-1] / np.median(data.reactive_breakthrough_times[-1]), nbins)
    colors = ['black', 'C0', 'C1', 'C2', 'C3', 'C4']
    #ymin = 10 ** (-10)
    #ymax = 10 ** (-1)
    for i, time in enumerate(data.track_times):
        bx11, pdf = create_pdf(data.breakthrough_times[i] / np.median(data.breakthrough_times[i]), nbins, x = bx1)
        #avr = bx11[np.where(pdf == pdf.max())]
        ax1.loglog(bx11, pdf, "o", alpha = 1, markersize=12, label = time, color = colors[i])
        #ax1.set_ylim(ymin, ymax)
        bx12, pdf = create_pdf(data.breakthrough_times[i] / np.median(data.breakthrough_times[i]), nbins, x = bx1, \
            weights = data.concentrations[i])
        ax2.loglog(bx12, pdf, "o", alpha = 1, markersize=12, label = time, color = colors[i])
        #ax2.set_ylim(ymin, ymax)
        # bx13, pdf = create_pdf(data.reactive_breakthrough_times[i] / np.median(data.reactive_breakthrough_times[i]), nbins, x = bx3)
        # ax3.loglog(bx13, pdf, "o", alpha = 1, markersize=12, label = time, color = colors[i])
        # #ax3.set_ylim(ymin, ymax)
    #legend = ax3.legend(loc='center right', bbox_to_anchor=(1.05, 0.5), prop={'size': 40}, frameon=False, handlelength = 0.2, borderpad = 0, handletextpad = 0.4)
    legend = ax1.legend(loc='lower center', prop={'size': 40}, mode = 'expand', ncol = 5, frameon=False, handlelength = 0.2, borderpad = 0, handletextpad = 0.4)
    for legobj in legend.legendHandles:
        legobj.set_markersize(24.0)
    plt.savefig(data.dirname + f'/track.png', bbox_inches="tight")
    plt.close()
