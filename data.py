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

import math
import random
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from config import SimInputData
from network import Edges, Graph
from incidence import Incidence

font = {#'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 10}

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
    # General 
    t = []
    pressure = []
    porosity = []
    cb_out = []
    cc_out = []
    delta_b = 0.
    delta_c = 0.
    dissolved_v = 0.
    dissolved_v_list = []

    # Channelisation
    slices: list = []
    slices_d: list = []
    slices_s: list = [] # channelization for slices through the whole system in a given time
    slice_times: list = [] # list of times of checking slice channelization

    # Time-dependent array data / slice data
    x_eval = None
    cb_network_avg = [] # avg. conc. parallel to flow
    cc_network_avg = [] # 
    flush_to_passivation_ratio = [] # 

    #breakthrough_times: list = []          # UNUSED / DEPRECATED? 
    #concentrations: list = []              # UNUSED / DEPRECATED? 
    #reactive_breakthrough_times: list = [] # UNUSED / DEPRECATED? 
    #track_times: list = []                 # UNUSED / DEPRECATED? 
    #vol_dissolved: float = 0.              # UNUSED / DEPRECATED? 
    #vol_precipitated: float = 0.           # UNUSED / DEPRECATED? 

    def __init__(self, sid: SimInputData, edges: Edges):
        self.dirname = sid.dirname
        self.vol_init = np.sum(edges.diams ** 2 * edges.lens)

    def collect_data(self, sid: SimInputData, inc: Incidence, graph: Graph, edges: Edges, \
                     vols, p: np.ndarray, cb: np.ndarray, cc: np.ndarray) -> None:
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

    def collect_slice_data(self, sid: SimInputData, inc: Incidence, graph: Graph, edges: Edges, \
                     vols, p: np.ndarray, cb: np.ndarray, cc: np.ndarray) -> None:
        """ Collects 'slice' data - a slice is a bin taken perpendicular to the flow.
            In each slice the average is taken over either a node or edge property

        """

        ## TODO: the following might be better handled in a separate update function
        ##       and with the time check in the main loop, i.e. via utils::stop_condition(...)
        #val = self.t[-1] / sid.track_every
        #if math.isclose(val, round(val), rel_tol=1e-9, abs_tol=1e-9):
        # average node concentration across network
        x, cbavg = self.get_slice_avg_node_prop(sid, graph, cb)
        if self.x_eval is None:
            self.x_eval = x
        self.cb_network_avg.append(cbavg)
        _, ccavg = self.get_slice_avg_node_prop(sid, graph, cc)
        self.cc_network_avg.append(ccavg)

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
        self.t, self.dissolved_v_list, self.pressure = list(data[0]), list(data[1]), list(data[2])
        self.porosity, self.participation_ratio = list(data[3]), list(data[4]), 
        self.cb_out, self.cc_out = list(data[5]), list(data[6])
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
        print(f"  Flow          | Q_in: {Q_in:<10.4f} Q_out: {Q_out:.4f}")
        print(f"  Min/max conc. | cb:  [{np.min(cb):.4f}, {np.max(cb):.4f}]")
        if sid.include_precipitation:
            print(f"                | cc:  [{np.min(cc):.4f}, {np.max(cc):.4f}]")
            print(f"                | cd:  [{np.min(cd):.4f}, {np.max(cd):.4f}]\n")
            if sid.include_nucleation:
                N = edges.N_tot * 2. * np.pi * edges.diams * edges.lens
                N_ = np.array2string(N, precision=2, separator=', ', \
                        edgeitems=2, threshold=5)
                print(f"  Nuclei stats  | A     : {sid.A:.4g}")
                print(f"                | Bounds: [{np.min(N):.2f}, {np.max(N):.2f}]")
                print(f"                | Mean #: {np.mean(N):.2f}")
                print(f"                | N     : {N_}\n")
                F_ = np.array2string(edges.ftrans, precision=2, separator=', ', \
                        edgeitems=2, threshold=5)
                print(f"  Passivation   | Bounds: [{np.min(edges.ftrans):.2f}, \
                      {np.max(edges.ftrans):.2f}]")
                print(f"                | Mean  : {np.mean(edges.ftrans):.2f}")
                print(f"                | f(t)  : {F_}\n")
    

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
        # sort edges from maximum flow to minimum (taking into account their orientation)
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

    def check_init_slice_channelization(self, graph: Graph, inc: Incidence, edges: Edges) -> None:
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

    def check_passivation_time(self, sid: SimInputData, inc: Incidence, edges: Edges, old_ftrans):
        """ Checks the time to passivate relative to advection time
        Parameters
        -------
            old_ftrans : np.ndarray
                f(t-1)
            edge_probe_idxs : np.ndarray
            TODO: move local ratio elsewhere 
        """
        dt = sid.dt
        if len(self.t) > 2:
            dt = self.t[-1] - self.t[-2] #?
        #V_pore_initial = np.sum(((edges.diams / 2.0)**2) * edges.lens)
        #tau_adv = V_pore_initial / sid.Q_in
        tau_adv_i = (edges.lens * (edges.diams/ 2.)**2)  / np.abs(edges.flow + 1e-25)
        df_dt = (edges.ftrans - old_ftrans) / dt
        # Time it would take to fully passivate at the current rate df/dt
        tau_pass_i = (1 - edges.ftrans) / np.abs(df_dt)
        tau_ratio = tau_adv_i / tau_pass_i # ratio per edge
        # Number of edges which transmit less than 1 pore volume during the passivation time
        global_ratio = len(np.where(tau_ratio > 1.)[0])/len(edges.diams)
        self.flush_to_passivation_ratio.append(global_ratio) # function of time
    
    def get_slice_avg_node_prop(self, sid: SimInputData, graph: Graph, node_prop, npoints=200):
        """ Gets average of a node property in slices perpendicular to flow direction
        Parameters
        -------
        npoints : int
            number of points to interpolate along (fixed number is used
            in case merging changes the resolution)
        Returns
        -------
        x_eval : np.ndarray
            bin centres
        avg : np.ndarray
            average value of node property at `x_eval`
        """
        #TODO: does merging doesn't change m,n?
        if node_prop.shape[0] != sid.m * sid.n:
            raise ValueError("ERROR: @Data::get_slice_avg_node_prop() \
                    Attempting to average a non-node property.")
        pos_x = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,0]
        bins = np.linspace(0, sid.m, npoints)
        x_eval, avg = [], []
        for i in range(len(bins)-1):
            mask = (pos_x >= bins[i]) & (pos_x < bins[i+1])
            x_eval.append((bins[i] + bins[i+1]) / 2)
            if np.any(mask):
                avg.append(np.mean(node_prop[mask]))
            else:
                avg.append(np.nan)
        return np.array(x_eval), np.array(avg)

    def get_slice_avg_edge_prop(self, sid, graph, edges, inc, edge_prop, npoints=100):
        """ Gets average of an edge property in slices perpendicular to flow direction
        Parameters
        -------
        npoints : int
            number of points to interpolate along (fixed number is used
            in case merging changes the resolution)
        Returns
        -------
        x_eval : np.ndarray
            bin centres
        avg : np.ndarray
            average value of edge property at `x_eval`
        """
        if len(edge_prop) != inc.incidence.shape[0]: 
            raise ValueError("ERROR: @Data::get_slice_avg_edge_prop() \
                    Attempting to average a non-edge property.")

        #pos_x = np.array([pos[0] for pos in nx.get_node_attributes(graph, 'pos').values()])
        pos_x = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,0]
        slices = np.linspace(0, sid.m, npoints)
        x_eval, avg = []
        for slice_x in slices:
            # Create a 1D vector: 1 if node is Right of slice, 0 if Left
            #nodes_right = (pos_x > slice_x).astype(int)
            # Multiply incidence matrix. Result is +/- 1 if nodes are on opposite sides.
            # abs() forces it to a boolean intersection mask.
            #crossing_edges_mask = np.abs(inc.incidence @ nodes_right) == 1
            crossing_edges_mask = np.abs(inc.incidence @ (pos_x > slice_x)) == 1
            x_eval.append(slice_x)
            if np.any(crossing_edges_mask):
                avg.append(np.mean(edge_prop[crossing_edges_mask]))
            else:
                avg.append(np.nan) # Preserve array shape if no edges cross this exact line
        return np.array(x_eval), np.array(avg)


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

    def plot_avg_node_props(self, sid, current_time=False, ax=None):
        """ Plot average of node property at different times
        TODO: generalise plotting, similar to Probe::plot_time_series_data 
              e.g. need to define and iterate through a set of node properties
        Parameters
        -------
        current_time : bool
            If true, plots all averaged node properties on a single panel at
            the current time

        TODO: add nucleation rate, number of nuclei as average edge properties, plot under network
        evolution
        """

        times = np.array(self.t)
        cb_avg = self.cb_network_avg
        cc_avg = self.cc_network_avg
        if not current_time:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            for i in range(len(cb_avg)):
                lab = f"t = {sid.track_every*i:.2f}"
                ax1.plot(self.x_eval, cb_avg[i], alpha=0.8, lw=1.5, label=lab)#, color=color)
                ax2.plot(self.x_eval, cc_avg[i], alpha=0.8, lw=1.5, label=lab)#, color=color)
            ax1.grid(True, linestyle='--', alpha=0.5)
            ax2.grid(True, linestyle='--', alpha=0.5)
            ax1.set_ylabel(r"Network average $c_B$")
            ax2.set_ylabel(r"Network average $c_C$")
            #ax1.set_xlabel(r"Horiz. span $x$")
            ax2.set_xlabel(r"Horiz. span $x$")
            ax1.legend(loc='center right', bbox_to_anchor=(-0.05, 0.5), 
                frameon=True, fontsize=10, alignment='right')
            plt.show()
        else:
            show_plot = False
            if ax is None:
                fig, ax = plt.subplots(figsize=(12, 8))
                show_plot = True
            fig, ax1 = plt.subplots(figsize=(12, 8))
            ax.plot(self.x_eval, cb_avg[-1], alpha=0.8, lw=1.5, label=r"$\overline{c}_B$")
            ax.plot(self.x_eval, cc_avg[-1], alpha=0.8, lw=1.5, label=r"$\overline{c}_C$")
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_ylabel(r"Spatial average")
            ax.set_xlabel(r"Horiz. span $x$")
            ax.legend(loc='center right', frameon=False, fontsize=10, alignment='right')
            if show_plot:
                plt.show()
            return ax
