""" Collect physical data from the simulation and save/plot them.

This module contains Data class, storing information about physical data in
the simulation. It stores the data during simulation and afterwards saves them
in a text file and plots them. For now the data are: permeability and
channelization (take a slice of the system in a given x-coordinate and check
how many of the edges contain half of the total flow through the slice) for
3 slices of the system (at 1/4, 1/2 and 3/4).

Notable classes
-------
Data
    container for physical data collected during simulation

TO DO: name data on plots, fix no values on y-axis for permeability in
channelization plot, maybe add plots of effluent concentration and dissolved
volume, include tracking plots in data?
"""

from matplotlib import gridspec
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as spr

from config import SimInputData
from incidence import Edges, Incidence
from network import Graph
from utils import create_bins, create_pdf

font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 50}

matplotlib.rc('font', **font)


class Data():
    """ Contains data collected during the simulation.

    This class is a container for all data collected during simulation, such
    as permeability, channelization etc. Part of the data
    is saved in params.txt file, the rest is plotted and saved as figures.
    """
    t: list = []
    "elapsed time of the simulation"
    perm: list = []
    "permeability between inlet and outlet"
    order: list = []
    "order parameter ((N - sum(q^2)^2/sum(q^4)) / (N - 1))"
    channels_1: list = []
    "channelization in slice 1 (defaultly 1/4 of system)"
    channels_2: list = []
    "channelization in slice 2 (defaultly 1/2 of system)"
    channels_3: list = []
    "channelization in slice 3 (defaultly 3/4 of system)"
    slices: list = []
    "channelization for slices through the whole system in a given time"
    slice_times: list = []
    "list of times of checking slice channelization"
    diams: list = []
    labels: list = []
    diams_slices: list = []
    flux_slices: list = []
    concentration_slices: list = []

    def __init__(self, sid: SimInputData, graph: Graph):
        self.dirname = sid.dirname
        # set positions of 3 default slices measured vs time
        pos_x = np.array(list(nx.get_node_attributes(graph, 'x').values()), \
            dtype = float)
        self.slice_x1 = float((np.min(pos_x) + np.average(pos_x)) / 2)
        self.slice_x2 = float(np.average(pos_x))
        self.slice_x3 = float((np.max(pos_x) + np.average(pos_x)) / 2)

    def collect(self, sid: SimInputData, graph: Graph, inc: Incidence, \
        edges: Edges, pressure: np.ndarray) -> None:
        """ Collect data from different vectors.

        This function extracts information such as permeability, channelization
        etc. and saves them in the data class.

        Parameters
        -------
        sid : SimInputData class object
            all config parameters of the simulation

        inc : Incidence class object
            matrices of incidence

        edges : Edges class object
            all edges in network and their parameters

        graph : Graph class object
            network and all its properties

        pressure : numpy ndarray
            vector of current pressure
        """
        # simulation time
        self.t.append(sid.old_t)
        # permeability (dimensionless)
        self.perm.append(1 / np.max(pressure))
        # flow focusing parameter
        self.order.append((np.sum(edges.flow != 0) \
            - np.sum(np.abs(edges.fracture_lens * edges.flow) ** 2) ** 2 \
            / np.sum(np.abs(edges.fracture_lens * edges.flow) ** 4)) \
            / (np.sum(edges.flow != 0) - 1))
        # channelization
        # self.channels_1.append(self.check_channelization(graph, inc, edges, \
        #     self.slice_x1)[1])
        # self.channels_2.append(self.check_channelization(graph, inc, edges, \
        #     self.slice_x2)[1])
        # self.channels_3.append(self.check_channelization(graph, inc, edges, \
        #     self.slice_x3)[1])
        
    def plot(self) -> None:
        """ Save all data and plot them.        
        """
        self.save()
        self.plot_params()
        self.plot_channelization()

    def save(self) -> None:
        """ Saves data to text file.

        This function saves the collected data to text file params.txt in
        columns. If the simulation is continued from saved parameters, new data
        is appended to that previously collected.
        """
        # save basic data to params.txt
        is_saved = False
        while not is_saved: # prevents problems with opening text file
            try:
                file = open(self.dirname + '/params.txt', 'a', \
                    encoding = "utf-8")
                np.savetxt(file, np.array([self.t, self.perm, self.order], \
                    dtype = float).T)
                file.close()
                is_saved = True
            except PermissionError:
                pass
        # self slice data to slices.txt
        is_saved = False
        while not is_saved: # prevents problems with opening text file
            try:
                file = open(self.dirname + '/slices.txt', 'a', \
                    encoding = "utf-8")
                np.savetxt(file, self.slices)
                file.close()
                is_saved = True
            except PermissionError:
                pass

    def check(self, edges: Edges) -> None:
        """ Check if flow in the system is valid.

        This function calculates and prints inflow and outflow to check if they
        are equal.

        Parameters
        -------
        edges : Edges class object
            all edges in network and their parameters
        """
        Q_in = np.sum(edges.inlet * np.abs(edges.fracture_lens * edges.flow))
        Q_out = np.sum(edges.outlet * np.abs(edges.fracture_lens * edges.flow))
        print('Q_in =', Q_in, 'Q_out =', Q_out)

    def plot_params(self) -> None:
        """ Plots data from text file.

        This function loads the data from text file params.txt and plots them
        to file params.png.
        """
        f = open(self.dirname + '/params.txt', 'r', encoding = "utf-8")
        data = np.loadtxt(f)
        n_data = data.shape[1]
        # first column is time
        t = data[:, 0]
        plt.figure(figsize = (15, 5))
        plt.suptitle('Parameters')
        spec = gridspec.GridSpec(ncols = n_data - 1, nrows = 1)
        for i_data in range(n_data - 1):
            # TO DO: name data columns
            plt.subplot(spec[i_data]).set_title(f'Data {i_data}')
            plt.plot(t, data[:, i_data + 1])
            plt.xlabel('simulation time')
        plt.savefig(self.dirname + '/params.png')
        plt.close()

    def plot_channelization(self) -> None:
        """ Plot channelization data from text file.

        This function loads the data from text file params.txt and plots those
        corresponding to channelization vs permeability.
        """
        f = open(self.dirname+'/params.txt', 'r')
        data = np.loadtxt(f)
        n_data = data.shape[1]
        t = data[:, 0]
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('simulation time')
        ax2 = ax1.twinx()
        ax1.set_yscale('log')
        ax1.set_ylabel('permeability')
        ax2.set_ylabel('channelization')
        ax1.plot(t, data[:, 1], 'k', '-', label = r'$\kappa$')
        for i_data in range(3, n_data):
            ax2.plot(t, data[:, i_data], label = f'slice {i_data - 2} / 4')
        ax2.legend()    
        plt.savefig(self.dirname + '/channelization.png')
        plt.close()


    def check_channelization(self, graph: Graph, inc: Incidence, edges: Edges, \
        concentration: np.ndarray, slice_x: float) -> tuple[int, float]:
        """
        """
        pos_x = np.array(list(nx.get_node_attributes(graph, 'x').values()))
        # find edges crossing the given slice and their orientation - if edge
        # crosses the slice from left to right, it is marked with 1, if from
        # right to left - -1, if it doesn't cross - 0
        slice_edges = (spr.diags(edges.flow) @ inc.incidence > 0) \
            @ (pos_x <= slice_x) * np.abs(inc.incidence @ (pos_x > slice_x)) \
            - (spr.diags(edges.flow) @ inc.incidence > 0) @ (pos_x > slice_x) \
            * np.abs(inc.incidence @ (pos_x <= slice_x))
        # sort edges from maximum flow to minimum (taking into account
        # their orientation)
        slice_flow = np.array(sorted(slice_edges * edges.fracture_lens \
            * np.abs(edges.flow), reverse = True))
        edge_number = np.sum(slice_flow != 0)
        fraction_flow = 0
        total_flow = np.sum(slice_flow)
        # calculate how many edges take half of the flow
        for i, edge_flow in enumerate(slice_flow):
            fraction_flow += edge_flow
            if fraction_flow > total_flow / 2:
                flow_edge = i + 1
                break
        slice_apertures = np.array(sorted(np.abs(slice_edges) \
            * edges.apertures, reverse = True))
        fraction_apertures = 0
        total_apertures = np.sum(slice_apertures)
        # calculate how many edges take half of the flow
        for i, edge_aperture in enumerate(slice_apertures):
            fraction_apertures += edge_aperture
            if fraction_apertures > total_apertures / 2:
                aperture_edge = i + 1
                break
        concentration_in = np.abs((spr.diags(edges.flow) @ inc.incidence > 0)) \
            @ concentration
        slice_concentration = np.array(sorted(slice_edges \
            * concentration_in * edges.apertures * edges.fracture_lens, reverse = True))
        fraction_concentration = 0
        total_concentration = np.sum(slice_concentration)
        # calculate how many edges take half of the flow
        if total_concentration > 0:
            for i, edge_aperture in enumerate(slice_concentration):
                fraction_concentration += edge_aperture
                if fraction_concentration > total_concentration / 2:
                    concentration_edge = i + 1
                    break
        else:
            concentration_edge = edge_number
        slice_flux = np.array(sorted(slice_edges \
            * np.abs(edges.flow) * edges.fracture_lens * concentration_in, reverse = True))
        fraction_flux = 0
        total_flux = np.sum(slice_flux)
        # calculate how many edges take half of the flow
        if total_flux > 0:
            for i, edge_aperture in enumerate(slice_flux):
                fraction_flux += edge_aperture
                if fraction_flux > total_flux / 2:
                    flux_edge = i + 1
                    break
        else:
            flux_edge = edge_number
        return flow_edge, aperture_edge, flux_edge, concentration_edge, edge_number

    def check_init_channelization(self, graph: Graph, inc, \
        edges, concentration) -> None:
        pos_x = np.array(list(nx.get_node_attributes(graph, 'x').values()))
        # slices = np.linspace(np.min(pos_x), np.max(pos_x), 120)[10:-10]
        slices = np.linspace(np.min(pos_x), np.max(pos_x), 102)[1:-1]
        channels_tab = []
        for x in slices:
            res = self.check_channelization(graph, inc, edges, concentration, x)[4]
            channels_tab.append(res)
        self.slices.append(channels_tab)
        self.diams_slices.append(channels_tab)
        self.flux_slices.append(channels_tab)
        self.concentration_slices.append(channels_tab)

    def check_slice_channelization(self, graph: Graph, inc, \
        edges, concentration) -> None:
        pos_x = np.array(list(nx.get_node_attributes(graph, 'x').values()))
        # slices = np.linspace(np.min(pos_x), np.max(pos_x), 120)[10:-10]
        slices = np.linspace(np.min(pos_x), np.max(pos_x), 102)[1:-1]
        channels_tab = []
        d_channels_tab = []
        f_channels_tab = []
        c_channels_tab = []
        for x in slices:
            res = self.check_channelization(graph, inc, edges, concentration, x)
            channels_tab.append(res[0])
            d_channels_tab.append(res[1])
            f_channels_tab.append(res[2])
            c_channels_tab.append(res[3])
        self.slices.append(channels_tab)
        self.diams_slices.append(d_channels_tab)
        self.flux_slices.append(f_channels_tab)
        self.concentration_slices.append(c_channels_tab)

    def plot_diams_histogram(self, nbins) -> None:
        plt.figure(figsize = (15, 15))
        plt.title('diameter distribution')
        plt.xlabel('normalized diameter', fontsize = 50)
        plt.ylabel('probability density', fontsize = 50)
        bx1 = create_bins(self.diams[0] / np.average(self.diams[0]), nbins, spacing = 'linear')
        colors = ['black', 'C0', 'C1', 'C2', 'C3', 'C4']
        for i, data in enumerate(self.diams):
            if i == 0 or i == 2 or i == 4:
                bx11, pdf = create_pdf(data / np.average(data), nbins, spacing = 'linear', x = bx1)
                plt.plot(bx11, pdf, "o", alpha = 1, markersize=12, label = self.labels[i], color = colors[i])
                plt.yscale('log')
            else:
                plt.plot([], [], ' ', label=' ')
        #legend = ax3.legend(loc='center right', bbox_to_anchor=(1.05, 0.5), prop={'size': 40}, frameon=False, handlelength = 0.2, borderpad = 0, handletextpad = 0.4)
        legend = plt.legend(loc='lower center', prop={'size': 40}, mode = 'expand', ncol = 5, frameon=False, handlelength = 0.2, borderpad = 0, handletextpad = 0.4)
        for legobj in legend.legend_handles:
            legobj.set_markersize(24.0)
        plt.savefig(self.dirname + f'/aperture_hist.png', bbox_inches="tight", transparent = False)
        plt.close()

    def plot_slice_channelization(self, graph: Graph, data_type) -> None:
        """ Plots slice data from text file.

        This function loads the data from text file slices.txt and plots them
        to files slices.png, slices_no_div.png, slices_norm.png.
        """
        pos_x = np.array(list(nx.get_node_attributes(graph, 'x').values()))
        # slices = np.linspace(np.min(pos_x), np.max(pos_x), 120)[10:-10]
        slices = np.linspace(np.min(pos_x), np.max(pos_x), 102)[1:-1]
        
        
        colors = ['C0', 'C1', 'C2', 'C3']
        plt.figure(figsize = (15, 10))
        edge_number  = np.array(self.slices[0])
        if data_type == 'flow':
            slice_data = self.slices
            plt.ylabel('flow focusing index', fontsize = 50)
            save_name = 'profile'
        if data_type == 'aperture':
            slice_data = self.diams_slices
            plt.ylabel('aperture focusing index', fontsize = 50)
            save_name = 'aperture_profile'
        if data_type == 'flux':
            slice_data = self.flux_slices
            plt.ylabel('flux focusing index', fontsize = 50)
            save_name = 'flux_profile'
        if data_type == 'concentration':
            slice_data = self.concentration_slices
            plt.ylabel('concentration focusing index', fontsize = 47)
            save_name = 'concentration_profile'
        
        plt.plot([], [], ' ', label=' ')
        plt.plot([], [], ' ', label=' ')
        plt.plot([], [], ' ', label=' ')
        plt.plot(slices, np.array((edge_number - 2 * np.array(slice_data[1])) \
            / edge_number), linewidth = 5, color = 'black', label = '0.0')
        for i, channeling in enumerate(slice_data[2:]):
            plt.plot(slices, (edge_number - 2 * np.array(channeling)) \
                / edge_number, label = self.labels[i+1], linewidth = 5, color = colors[i])
        plt.ylim(0, 1.05)
        plt.xlabel('x', fontsize = 60, style = 'italic')
        # ax2.xaxis.label.set_color('white')
        # ax2.tick_params(axis = 'x', colors='white')
        #plt.xticks([],[])
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.margins(tight = True)
        #plt.ylabel('aperture variations index', fontsize = 50)
        #plt.yticks([],[])
        plt.yticks([0, 0.5, 1],['0', '0.5', '1'])
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [0,4,1,5,2,6,3,7]

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
        plt.savefig(self.dirname + "/" + save_name + ".png", bbox_inches="tight")
        plt.close()

    def collect_initial_data(self, sid, graph, inc, edges, concentration):
            if sid.flow_focusing_profile:
                self.check_init_channelization(graph, inc, edges, concentration)
                self.check_slice_channelization(graph, inc, edges, concentration)
            self.diams.append(edges.apertures.copy())
            self.labels.append(f'{sid.dissolved_v:.2f}')

    def collect_data(self, sid, graph, inc, edges, concentration):
            if sid.flow_focusing_profile:
                self.check_slice_channelization(graph, inc, edges, concentration)
            self.diams.append(edges.apertures.copy())
            self.labels.append(f'{sid.dissolved_v:.2f}')

    def plot_data(self, graph):
        self.plot_slice_channelization(graph, 'flow')
        self.plot_slice_channelization(graph, 'aperture')
        self.plot_slice_channelization(graph, 'flux')
        self.plot_slice_channelization(graph, 'concentration')
        self.plot_diams_histogram(100)