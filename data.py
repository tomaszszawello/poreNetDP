""" Collect physical data from the simulation and save/plot them.

This module initializes Data class, storing information about physical data in
the simulation. It stores the data during simulation and afterwards saves them
in a text file and plots them. For now the data are: pressure difference
between input and output (1 / permeability) and quantities of substance B and C
that flowed out of the system.

Notable classes
-------
Data
    container for physical data collected during simulation

TO DO: name data on plots, maybe collect permeability explicitly
"""

from matplotlib import gridspec
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as spr

import random
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from config import SimInputData
from network import Edges, Graph
from incidence import Incidence

font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 50}

matplotlib.rc('font', **font)

class Data():
    """ Contains data collected during the simulation.

    Attributes
    -------
    t : list
        elapsed time of the simulation

    pressure : list
        pressure difference between inlet and outlet

    cb_out : list
        difference of inflow and outflow of substance B in the system

    cb_out : list
        difference of inflow and outflow of substance C in the system

    delta_b : float
        current difference of inflow and outflow of substance B in the system

    delta_c : float
        current difference of inflow and outflow of substance C in the system
    """
    t = []
    pressure = []
    porosity = []
    cb_out = []
    cc_out = []
    delta_b = 0.
    delta_c = 0.
    dissolved_v = 0.
    dissolved_v_list = []
    slices: list = []
    slices_d: list = []
    slices_s: list = []
    "channelization for slices through the whole system in a given time"
    slice_times: list = []
    "list of times of checking slice channelization"
    breakthrough_times: list = []
    concentrations: list = []
    reactive_breakthrough_times: list = []
    track_times: list = []
    vol_dissolved: float = 0.
    vol_precipitated: float = 0.
    flush_to_passivation_ratio = [] # 

    def __init__(self, sid: SimInputData, edges: Edges):
        self.dirname = sid.dirname
        self.vol_init = np.sum(edges.diams ** 2 * edges.lens)

    def save_data(self) -> None:
        """ Save data to text file.

        This function saves the collected data to text file params.txt in
        columns. If the simulation is continued from saved parameters, new data
        is appended to that previously collected.
        """
        is_saved = False
        while not is_saved: # prevents problems with opening text file
            try:
                file = open(self.dirname + '/params.txt', 'w', \
                    encoding = "utf-8")
                np.savetxt(file, np.array([self.t, self.dissolved_v_list, self.pressure, self.porosity], dtype = float).T)
                file.close()
                is_saved = True
            except PermissionError:
                pass
        # self slice data to slices.txt
        is_saved = False
        while not is_saved: # prevents problems with opening text file
            try:
                file = open(self.dirname + '/profiles.txt', 'w', \
                    encoding = "utf-8")
                np.savetxt(file, self.slices)
                file.close()
                is_saved = True
            except PermissionError:
                pass

    def load_data(self) -> None:
        data = np.loadtxt(self.dirname + '/params.txt').T
        self.t, self.dissolved_v_list, self.pressure, self.porosity, self.participation_ratio, self.cb_out, \
                    self.cc_out = list(data[0]), list(data[1]), list(data[2]), list(data[3]), list(data[4]), list(data[5]), list(data[6])
        self.dissolved_v = self.dissolved_v_list[-1]

    def check_data(self, sid: SimInputData, edges: Edges, pressure, cb, cc, cd, state) -> None:
        """ Check the key physical parameters of the simulation.

        This function calculates and checks if basic physical properties of the
        simulation are valied, i.e. if inflow is equal to outflow.

        Parameters
        -------
        edges : Edges class object
            all edges in network and their parameters
            flow - flow in edges
            inlet - edges connected to inlet nodes
            outlet - edges connected to outlet nodes
        """
        Q_in = np.abs(np.sum(edges.inlet * edges.flow))
        Q_out = np.abs(np.sum(edges.outlet * edges.flow))
        if np.abs(Q_in - Q_out) > 1e-2:
            raise ValueError("Flow continuity error")
        # TO DO: fix mass balance check from collect_data
        if np.max(edges.outlet * edges.diams) > sid.d_break:
            state = True
            print("Network dissolved")
        if np.max(pressure) > self.pressure[0] / sid.min_perm:
            state = True
            print("Network clogged")
        return state

    def summarise_data(self, sid: SimInputData, edges: Edges, pressure, cb, cc, cd, state):
        """ Pretty format for runtime variables
        """
        print(f"  Params   | [Da_eff :  K   :  G   : Gamma]")
        print(f"           | [ {sid.Da_eff:.2f}  : {sid.K:.2f} : {sid.G:.2f} : {sid.Gamma:.2f}]")
        Q_in = np.abs(np.sum(edges.inlet * edges.flow))
        Q_out = np.abs(np.sum(edges.outlet * edges.flow))
        print(f"  Flow           | Q_in: {Q_in:<10.4f} Q_out: {Q_out:.4f}")
        print(f"  Min/max conc.  | cb:  [{np.min(cb):.4f}, {np.max(cb):.4f}]")
        if sid.include_precipitation:
            print(f"             | cc:  [{np.min(cc):.4f}, {np.max(cc):.4f}]")
            print(f"             | cd:  [{np.min(cd):.4f}, {np.max(cd):.4f}]")
        if sid.include_nucleation:
            N = edges.N_tot * 2. * np.pi * edges.diams * edges.lens
            N_preview = np.array2string(N, precision=2, separator=', ', edgeitems=2, threshold=5)
            print(f"  Nuclei     | A     : {sid.A:.4g}")
            print(f"             | Bounds: [{np.min(N):.2f}, {np.max(N):.2f}]")
            print(f"             | Mean #: {np.mean(N):.2f}")
            print(f"             | N     : {N_preview}")
            F_preview = np.array2string(edges.ftrans, precision=2, separator=', ', edgeitems=2, threshold=5)
            print(f"  Passiv.    | Bounds: [{np.min(edges.ftrans):.2f}, {np.max(edges.ftrans):.2f}]")
            print(f"             | Mean  : {np.mean(edges.ftrans):.2f}")
            print(f"             | f(t)  : {F_preview}")
    

    def collect_data(self, sid: SimInputData, inc: Incidence, edges: Edges, vols, \
        p: np.ndarray, cb: np.ndarray, cc: np.ndarray) -> None:
        """ Collect data from different vectors.

        This function extracts information such as permeability, quantity of
        substances flowing out of the system etc. and saves them in the data
        class.

        Parameters
        -------
        sid : SimInputData class object
            all config parameters of the simulation
            old_t - total time of simulation
            dt - current timestep

        inc : Incidence class object
            matrices of incidence
            incidence - connections of all edges with all nodes

        edges : Edges class object
            all edges in network and their parameters
            flow - flow in edges
            inlet - edges connected to inlet nodes
            outlet - edges connected to outlet nodes

        p : numpy ndarray
            vector of current pressure

        cb : numpy ndarray
            vector of current substance B concentration

        cc : numpy ndarray
            vector of current substance C concentration
        """
        self.t.append(sid.old_t)
        self.pressure.append(np.max(p))
        self.porosity.append(1 - np.sum(vols.vol_a) / np.sum(vols.vol_max))
        self.dissolved_v = (np.sum(edges.diams ** 2 * edges.lens) - self.vol_init) / self.vol_init
        self.dissolved_v_list.append(self.dissolved_v)
        


    def plot_data(self) -> None:
        """ Plot data from text file.

        This function loads the data from text file params.txt and plots them
        to file params.png.
        """
        f = open(self.dirname + '/params.txt', 'r', encoding = "utf-8")
        data = np.loadtxt(f)
        n_data = data.shape[1]
        t = data[:, 0]
        plt.figure(figsize = (15, 5))
        plt.suptitle('Parameters')
        spec = gridspec.GridSpec(ncols = n_data - 1, nrows = 1)
        for i_data in range(n_data - 1):
            plt.subplot(spec[i_data]).set_title(f'Data {i_data}')
            #plt.plot(t, data[:, i_data + 1] / data[0, i_data + 1])
            plt.plot(t, data[:, i_data + 1])
            plt.yscale('log')
            plt.xlabel('simulation time')
        plt.savefig(self.dirname + '/params.png')
        plt.close()

    def check_channelization(self, graph: Graph, inc: Incidence, edges: Edges, \
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
        pos_x = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,0]
        # find edges crossing the given slice and their orientation - if edge
        # crosses the slice from left to right, it is marked with 1, if from
        # right to left - -1, if it doesn't cross - 0
        slice_edges = (spr.diags(edges.flow) @ inc.incidence > 0) \
            @ (pos_x <= slice_x) * np.abs(inc.incidence @ (pos_x > slice_x)) \
            - (spr.diags(edges.flow) @ inc.incidence > 0) @ (pos_x > slice_x) \
            * np.abs(inc.incidence @ (pos_x <= slice_x))
        # sort edges from maximum flow to minimum (taking into account
        # their orientation)
        slice_flow = np.array(sorted(slice_edges * np.abs(edges.flow), reverse = True))
        fraction_flow = 0
        total_flow = np.sum(slice_flow)
        #print(total_flow)
        # calculate how many edges take half of the flow
        for i, edge_flow in enumerate(slice_flow):
            fraction_flow += edge_flow
            if fraction_flow > total_flow / 2:
                flow_50 = i + 1
                break
        slice_diams = np.array(sorted(slice_edges * np.abs(edges.diams), reverse = True))
        fraction_diams = 0
        total_diams = np.sum(slice_diams)
        # calculate how many edges take half of the flow
        for i, edge_diam in enumerate(slice_diams):
            fraction_diams += edge_diam
            if fraction_diams > total_diams / 2:
                diams_50 = i + 1
                break
        slice_surface = np.array(sorted(slice_edges * np.abs(edges.diams ** 2), reverse = True))
        fraction_surface = 0
        total_surface = np.sum(slice_surface)
        # calculate how many edges take half of the flow
        for i, edge_surface in enumerate(slice_surface):
            fraction_surface += edge_surface
            if fraction_surface > total_surface / 2:
                surface_50 = i + 1
                break
        return (flow_50, np.sum(slice_flow != 0), diams_50, np.sum(slice_diams != 0), surface_50, np.sum(surface_50 != 0))

    def check_init_slice_channelization(self, graph: Graph, inc: Incidence, \
        edges: Edges) -> None:
        pos_x = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,0]
        slices = np.linspace(np.min(pos_x), np.max(pos_x), 102)[1:-1]
        channels_tab = []
        diams_tab = []
        surface_tab = []
        for x in slices:
            res = self.check_channelization(graph, inc, edges, x)
            channels_tab.append(res[1])
            diams_tab.append(res[3])
            surface_tab.append(res[5])
        self.slices.append(channels_tab)
        self.slices_d.append(diams_tab)
        self.slices_s.append(surface_tab)

    def check_slice_channelization(self, graph: Graph, inc: Incidence, \
        edges: Edges, time: float) -> None:
        pos_x = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,0]
        slices = np.linspace(np.min(pos_x), np.max(pos_x), 102)[1:-1]
        channels_tab = []
        diams_tab = []
        surface_tab = []
        for x in slices:
            res = self.check_channelization(graph, inc, edges, x)
            channels_tab.append(res[0])
            diams_tab.append(res[2])
            surface_tab.append(res[4])
        self.slices.append(channels_tab)
        self.slices_d.append(diams_tab)
        self.slices_s.append(surface_tab)
        self.slice_times.append("{0}".format(str(round(time, 1) if time % 1 else int(time))))

    def plot_profile(self, graph: Graph) -> None:
        """ Plots slice data from text file.

        This function loads the data from text file slices.txt and plots them
        to files slices.png, slices_no_div.png, slices_norm.png.
        """
        pos_x = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,0]
        # slices = np.linspace(np.min(pos_x), np.max(pos_x), 120)[10:-10]
        slices = np.linspace(np.min(pos_x), np.max(pos_x), 102)[1:-1]
        edge_number  = np.array(self.slices[0])
        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
        plt.figure(figsize = (15, 10))
        plt.plot(slices, np.array((edge_number - 2 * np.array(self.slices[1])) \
            / edge_number), linewidth = 5, color = 'black', label = '0.0')
        for i, channeling in enumerate(self.slices[2:]):
            plt.plot(slices, (edge_number - 2 * np.array(channeling)) \
                / edge_number, label = self.slice_times[i+1], color = colors[i], linewidth = 5)
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
        handles, labels = plt.gca().get_legend_handles_labels()
        n_labels = len(labels)
        ncol = max(1, int(np.ceil(n_labels / 2)))
        nrow = int(np.ceil(n_labels / ncol))

        order = []
        for j in range(ncol):
            for i in range(nrow):
                idx = i * ncol + j
                if idx < n_labels:
                    order.append(idx)

        legend = plt.legend(
            [handles[idx] for idx in order],
            [labels[idx] for idx in order],
            loc = "lower right",
            ncol = ncol,
            prop = {'size': 30},
            handlelength = 1,
            frameon = False,
            borderpad = 0,
            handletextpad = 0.4,
            columnspacing = 0.8
        )
        for legobj in legend.legend_handles:
            legobj.set_linewidth(10.0)
        plt.savefig(self.dirname + "/profile.png", bbox_inches="tight")
        plt.close()

    def plot_things(self, sid: SimInputData):
        plt.figure(figsize = (15, 10))
        plt.title('Permeability')
        plt.plot(self.t, self.pressure[0] / self.pressure, linewidth = 5, color = 'black')
        plt.xlabel(r'simulation time', fontsize = 50)
        #plt.subplots_adjust(wspace=0, hspace=0)
        plt.margins(tight = True)
        plt.ylabel(r'$\kappa / \kappa_0$', fontsize = 50)
        plt.savefig(self.dirname + '/permeability.png', bbox_inches="tight")
        plt.close()
        plt.figure(figsize = (15, 10))
        plt.title('Porosity')
        plt.plot(self.t, self.porosity, linewidth = 5, color = 'black')
        plt.xlabel(r'simulation time', fontsize = 50)
        #plt.subplots_adjust(wspace=0, hspace=0)
        plt.margins(tight = True)
        plt.ylabel(r'$\phi$', fontsize = 50)
        plt.savefig(self.dirname + '/porosity.png', bbox_inches="tight")
        plt.close()

    def check_timescale_sep(self, sid: SimInputData, inc: Incidence, edges: Edges, old_ftrans):
        """ Checks for time scale separation between passivation flush time
        Parameters
        -------
            old_ftrans : np.ndarray
                f(t-1)
            edge_probe_idxs : np.ndarray
            TODO: move local ratio elsewhere 
        """
        V_pore_initial = np.sum(((edges.diams / 2.0)**2) * edges.lens)
        dt = sid.dt
        if len(self.t) > 2:
            dt = self.t[-1] - self.t[-2]

        tau_porevol = V_pore_initial / sid.Q_in
        tau_porevol_i = (edges.lens * (edges.diams/ 2.)**2)  / np.abs(edges.flow + 1e-25)
        df_dt = (edges.ftrans - old_ftrans) / dt
        # Time it would take to fully passivate at the current rate df/dt
        tau_pass_i = (1 - edges.ftrans) / np.abs(df_dt)
        tau_ratio = tau_porevol_i / tau_pass_i # ratio per edge
        # Number of edges which transmit less than 1 pore volume during the passivation time
        global_ratio = len(np.where(tau_ratio > 1.)[0])/len(edges.diams)
        self.flush_to_passivation_ratio.append(global_ratio)

        print(f"Initial Pore Volume:  {V_pore_initial:.4f}")
        print(f"Fluid Flush Time:     {tau_porevol:.4f} units")
        #print(f"Ratio series: {self.t[-1]:5f}, {tau_ratio[edge_probe_idxs]}")
        print(f"Ratio data: {self.t[-1]:5f}, {sid.Da_eff}, {sid.K}, {sid.Gamma}")
    
class Probe: 
    """ Visualisation and debug tool.
        Easy access and visualisation for local network data at (for now) pre-defined locations
        
        Basic idea is that we might want to look at concentraion profile in a particular edge,
        or the passivated fraction in an edge, node pressure etc etc. often as a function of time
        Moreover we don't want to track and plot this data for every edge/node in the network

        TODO: add ratio data time series plots
    """

    start_node = 0
    probe_type = ""
    edge_probe_idxs = []
    node_probe_idxs = []
    edge_probe_tuples = []  
    def __init__(self, sid, edges: Edges, graph: Graph, snode=20, opts=None, total_nodes=None):
        """ 
        Parameters
        -------
        ...
        snode : np.int
            index of an inlet node
        opts : string
            path 
                random oriented path through the network
            grid
                square grid of nodes with one attached edge   
        """
        self.start_node = snode
        self.probe_type = opts
        
        if opts == "path": 

            lr_nodes, lr_path, lr_idx = self.get_left_to_right_path(sid, edges, graph)
            
            # TODO: fix this - len_subset is not length but increment of the sequence
            len_subset = total_nodes
            if total_nodes is None:
                len_subset = int(0.1*len(lr_nodes)) # default 10%
            probe_sl = slice(1, len(lr_nodes), len_subset)

            self.node_probe_idxs = np.array(lr_nodes[probe_sl])
            self.edge_probe_tuples = np.array(lr_path[probe_sl]) 
            self.edge_probe_idxs = np.array(lr_idx[probe_sl])
        elif opts == "grid":
            # if total_nodes is not a square number, make it square
            print("TODO")
            # TODO: Want an approximately square grid of nodes on the network 
            # (assume a square network, sid.n = sid.m). 
            # e.g. 9 nodes that are approx. equi-distant apart and then 9 edges 
            # that are attached to these nodes, such that each edge has the
            # same flow direction , i.e. flow in each edge is +ve
            # since the network is formed (delaunay) from uniform random points on
            # sid.n x sid.m, i suggest finding nodes on the network that are cllosest to 
            # points on a square integer grid. 
        else:
            raise ValueError('ERROR: should this be an erorr?')

    def get_left_to_right_path(self, sid, edges: Edges, graph: Graph):
        """ Gets a random path from inlet to outlet that moves +ve horiz. direction
        TODO: Paths passing through PBC not checked
        Parameters:
        --------
        ...
        Returns:
        --------
        path_nodes : list
            List of node indices forming the path
        path_edges : list
            List of edge tuples [(n0, n1), (n1, n2), ...] 
                - used with nx's edgelist parameter
        path_edge_idxs : list
            List of edge indices 
                - explicit index to edges.<property>
        """
    
        if self.start_node > sid.n:
            #TODO: Choose a random valid start instead
            raise ValueError("ERROR @find_left_to_right_path(): Invalid starting node.")
        pos = nx.get_node_attributes(graph, 'pos')
        path_nodes = [self.start_node]
        path_edges = []     # edge between node indices
        path_edge_idxs = [] # the edge itself
        snode = self.start_node 

        while snode not in graph.out_nodes:
            # Get all neighbors
            neighbors = list(graph.neighbors(snode))
            valid_next = []
            snode_x = pos[snode][0]
            #snode = pos[snode][0]
            # Pick the neighbour that moves us in + x dir
            for n in neighbors:
                x_next = pos[n][0]
                if x_next > snode_x:
                    valid_next.append(n)
            
            if not valid_next:
                break
            # Choose one random forward neighbor
            # TODO: Could expand for more 'specific' path types
            #       e.g. largest incrememnt instead of random valid
            next_node = random.choice(valid_next)
            path_nodes.append(next_node)
            path_edges.append((snode, next_node))
    
            u, v = snode, next_node
            if (u, v) in edges.edge_list:
                path_edge_idxs.append(edges.edge_list.index((u, v)))
            elif (v, u) in edges.edge_list:
                path_edge_idxs.append(edges.edge_list.index((v, u)))
    
            snode = next_node
        return path_nodes, path_edges, path_edge_idxs

    def show_probes(self, sid: SimInputData, edges: Edges, graph: Graph):
        """ Displays the network with probes highlighted  

        TODO: Path that wraps across P.B. won't be drawn correctly 
        TODO: Does not adapt to merging 
        TODO: Add optional args to a relevant draw_net.py function instead of code dupe.
        """
        fig, axs = plt.subplots(figsize=(sid.figsize, sid.figsize))
        title=""
        fig.suptitle(title, fontsize=15)

        pos = nx.get_node_attributes(graph, 'pos')
        if sid.include_nucleation:
            edge_colors = plt.cm.copper_r(mcolors.Normalize(0, 1)(edges.ftrans)) 
            norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
        else:
            edge_colors = 'k'
        cmap = plt.get_cmap('tab10') 
        probe_colors = {node_idx: cmap(i % 10) for i, node_idx in enumerate(self.node_probe_idxs)}

        # Pre-calculate shared node coordinates
        xi, yi = zip(*[pos[n] for n in graph.in_nodes])
        xo, yo = zip(*[pos[n] for n in graph.out_nodes])
        axs.set_aspect('equal')
        axs.set_xlim([-0.5, sid.n + 0.5])
        axs.set_ylim([-0.5, sid.n + 0.5])
        axs.set_axis_off() # Disabling axs is faster than styling them
        axs.scatter(xi, yi, s=1000/sid.n, fc='white', ec='black', zorder=3)
        axs.scatter(xo, yo, s=1000/sid.n, fc='black', ec='white', zorder=3)
        axs.margins(0)
        qs = (1 - edges.boundary_list) * (edges.diams * (edges.diams > 0))
        w = sid.ddrawconst * np.array(qs)

        # Draw the full network
        nx.draw_networkx_edges(graph, pos, edgelist=edges.edge_list, 
                               edge_color=edge_colors, width=w, ax=axs)
        # Highlight selected edges along a path in the network
        nx.draw_networkx_edges(graph, pos, edgelist = self.edge_probe_tuples, edge_color='red', \
                width = w *2, alpha=1.0, ax=axs)
        # Highlight selected nodes along path
        all_pos = nx.get_node_attributes(graph, 'pos')
        for node_idx in self.node_probe_idxs:
            x, y = all_pos[node_idx]
            c = probe_colors[node_idx]
            axs.scatter(x, y, s=120, color=c, alpha=1)#, label=f'Node {node_idx}')
            axs.text(x, y + (sid.n * 0.05), str(node_idx), 
                        fontsize=9, fontweight='bold', ha='center', color='red')

        plt.tight_layout()
        plt.show()
        plt.close()

