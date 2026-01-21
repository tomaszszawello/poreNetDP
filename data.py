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
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as spr

from config import SimInputData
from network import Edges, Graph
from incidence import Incidence

import matplotlib
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
    pressure_diff = []
    order = []
    participation_ratio = []
    participation_ratio_nom = []
    participation_ratio_denom = []
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
    cond_ratio_cc: list = []
    cond_ratio_cb: list = []
    interface_width: list = []
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
                np.savetxt(file, np.array([self.t, self.pressure, self.participation_ratio, self.cb_out, \
                    self.cc_out], dtype = float).T)
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
        self.t, self.pressure, self.participation_ratio, self.cb_out, \
            self.cc_out = list(data[0]), list(data[1]), list(data[2]), list(data[3]), list(data[4])
        self.slices = list(np.loadtxt(self.dirname + '/slices.txt'))

    def check_data(self, sid:SimInputData, edges: Edges, inc: Incidence, cb: np.ndarray, cc) -> None:
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
        Q_in = np.sum(edges.inlet * np.abs(edges.flow))
        Q_out = np.sum(edges.outlet * np.abs(edges.flow))
        print('Q_in =', Q_in, 'Q_out =', Q_out)
        if np.abs(np.abs(Q_in) - np.abs(Q_out)) > 0.1 or np.isnan(Q_in) or np.isnan(Q_out):
            raise ValueError('Flow not matching!')
        # delta = np.abs((np.abs(inc.incidence.T < 0) @ (np.abs(edges.flow) \
        #     * edges.inlet) - np.abs(inc.incidence.T > 0) @ (np.abs(edges.flow) \
        #     * edges.outlet)) @ cb * sid.dt)
        # self.delta_b += delta
        # delta = np.abs((np.abs(inc.incidence.T < 0) @ (np.abs(edges.flow) \
        #     * edges.inlet) - np.abs(inc.incidence.T > 0) @ (np.abs(edges.flow) \
        #     * edges.outlet)) @ cc * sid.dt)
        # self.delta_c += delta
        #vol = np.sum(edges.lens * (edges.diams_min ** 2 - edges.diams_initial ** 2))
        #print ('Difference =', (self.delta_b - self.delta_c) / sid.Gamma - self.delta_b + sid.Da * vol / 2)

    def collect_data(self, sid: SimInputData, inc: Incidence, edges: Edges, graph: Graph, \
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
        self.pressure_diff.append(np.max(p * graph.in_vec_a) - np.max(p * graph.in_vec_b))
        self.order.append((sid.ne - np.sum(edges.flow ** 2) ** 2 \
            / np.sum(edges.flow ** 4)) / (sid.ne - 1))
        pi = np.sum(edges.diams ** 2 * np.abs(edges.flow)) ** 2 / np.sum(edges.diams ** 2 \
            * np.abs(edges.flow) ** 2) / sid.nsq
        pi_prime = np.sum(edges.diams ** 2) / sid.nsq
        self.participation_ratio_nom.append(pi)
        self.participation_ratio_denom.append(pi_prime)
        self.participation_ratio.append(pi / pi_prime)
        # calculate the difference between inflow and outflow of each substance
        # delta = np.abs((np.abs(inc.incidence.T < 0) @ (np.abs(edges.flow) \
        #     * edges.inlet) - np.abs(inc.incidence.T > 0) @ (np.abs(edges.flow) \
        #     * edges.outlet)) @ cb * sid.dt)
        # self.delta_b += delta
        # self.cb_out.append(self.delta_b)
        # delta = np.abs((np.abs(inc.incidence.T < 0) @ (np.abs(edges.flow) \
        #     * edges.inlet) - np.abs(inc.incidence.T > 0) @ (np.abs(edges.flow) \
        #     * edges.outlet)) @ cc * sid.dt)
        # self.delta_c += delta
        # self.cc_out.append(self.delta_c)
        self.dissolved_v = (np.sum(edges.diams_min ** 2 * edges.lens) - self.vol_init) / self.vol_init
        self.dissolved_v_list.append(self.dissolved_v)
        interface_nodes = np.abs(inc.incidence.T) @ (edges.diams_initial - edges.diams > 0.1)
        pos_y = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,1]
        n_max = np.max(pos_y * interface_nodes)
        try:
            n_min = np.min(pos_y * interface_nodes[np.nonzero(pos_y * interface_nodes)])
        except:
            n_min = 0
        self.interface_width.append(n_max - n_min)

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
        # plt.figure(figsize = (10, 10))
        # plt.title('Participation ratio')
        # plt.plot(t, data[:, 2])
        # plt.xlabel('simulation time')
        # plt.savefig(self.dirname + '/participation_ratio.png')
        # plt.close()

    def plot_pressure(self):
        plt.figure(figsize = (10, 7))
        plt.yscale('log')
        plt.plot(self.t, self.pressure / self.pressure[0], 'r')
        #plt.xlim(0, 3000)
        plt.xlabel('simulation time')
        plt.ylabel(f'$p / p_0$')
        plt.margins(tight = True)
        plt.savefig(self.dirname + '/pressure.png', bbox_inches="tight")

    def plot_pressure_diff(self):
        plt.figure(figsize = (10, 7))
        #plt.yscale('log')
        plt.plot(self.t, self.pressure_diff / self.pressure[0], 'r')
        #plt.xlim(0, 3000)
        plt.xlabel('simulation time')
        plt.ylabel(f'$p / p_0$')
        plt.margins(tight = True)
        plt.savefig(self.dirname + '/pressure_diff.png', bbox_inches="tight")
    
    def plot_perm(self):
        #plt.figure(figsize = (10, 7))
        plt.yscale('log')
        plt.plot(self.t, self.pressure[0] / self.pressure, 'r')
        #plt.xlim(0, 3000)
        plt.xlabel('simulation time')
        plt.ylabel(f'$\kappa / \kappa_0$')
        plt.margins(tight = True)
        plt.savefig(self.dirname + '/perm.png', bbox_inches="tight")

    def check_diams(self, sid: SimInputData, graph: Graph, inc: Incidence, edges: Edges, \
        slice_y: float) -> tuple[int, float]:
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
        pos_y = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,1]
        #np.savetxt('x.txt', pos_x)
        # find edges crossing the given slice and their orientation - if edge
        # crosses the slice from left to right, it is marked with 1, if from
        # right to left - -1, if it doesn't cross - 0
        slice_edges = (spr.diags(edges.flow) @ inc.incidence > 0) \
            @ (pos_y <= slice_y) * np.abs(inc.incidence @ (pos_y > slice_y)) \
            - (spr.diags(edges.flow) @ inc.incidence > 0) @ (pos_y > slice_y) \
            * np.abs(inc.incidence @ (pos_y <= slice_y))
        if np.sum(slice_edges) == 0:
            pos_y += 0.0001
            slice_edges = (spr.diags(edges.flow) @ inc.incidence > 0) \
                @ (pos_y <= slice_y) * np.abs(inc.incidence @ (pos_y > slice_y)) \
                - (spr.diags(edges.flow) @ inc.incidence > 0) @ (pos_y > slice_y) \
                * np.abs(inc.incidence @ (pos_y <= slice_y))
        return np.sum(np.abs(slice_edges) * (edges.diams - sid.dmin * (edges.diams > sid.dmin)))

    def check_conc(self, sid, graph, cb):
        pos_x = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,0]
        pos_y = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,1]
        x_slice = np.linspace(sid.bound_x + 2, np.max(pos_x) - 2, 5)
        x_correct = []
        for x in x_slice:
           idx = (np.abs(pos_x - x)).argmin()
           x_correct.append(pos_x[idx])
        
        y_slice = []
        c_slice = []
        for x in x_correct:
            nodes_x = np.where(pos_x == x)
            c_slice.append(cb[nodes_x])
            y_slice.append(pos_y[np.where(pos_x == x)])

        plt.figure(figsize = (10, 7))
        #plt.yscale('log')
        for i, c in enumerate(c_slice):
            plt.plot(y_slice[i], c, label = x_correct[i])
        #plt.xlim(0, 3000)
        plt.xlabel('y')
        plt.ylabel(f'concentration')
        plt.margins(tight = True)
        plt.legend()
        plt.savefig(self.dirname + '/concentration_profile.png', bbox_inches="tight")

    def check_conc2(self, sid, inc, graph, cb):
        # Get nodes in the same order as used in cb (this is important!)
        nodes = list(graph.nodes())     # must match the order used in building incidence / cb
        pos = nx.get_node_attributes(graph, 'pos')

        # Build coordinate arrays aligned with `nodes` and `cb`
        pos_x = np.array([pos[n][0] for n in nodes])
        pos_y = np.array([pos[n][1] for n in nodes])

        cb = np.abs(inc.incidence).T @ cb            # ensure it's a 1D numpy array

        # Choose x positions where you want vertical slices
        x_slice = np.linspace(sid.bound_x + 2, np.max(pos_x) - 2, 5)

        # Snap them to actual node x's
        x_correct = []
        for x in x_slice:
            idx = np.argmin(np.abs(pos_x - x))
            x_correct.append(pos_x[idx])
        x_correct = np.array(x_correct)

        y_slice = []
        c_slice = []

        # Tolerance for "same x" (in case of FP noise)
        atol = 1e-8

        for x in x_correct:
            # nodes at this x (within tolerance)
            mask = np.isclose(pos_x, x, atol=atol)
            if not np.any(mask):
                # no nodes at that x -> skip this slice
                continue

            y_vals  = pos_y[mask]
            c_vals  = cb[mask]

            # sort by y so profile is monotonic in vertical direction
            order   = np.argsort(y_vals)
            y_slice.append(y_vals[order])
            c_slice.append(c_vals[order])

        # Plot
        plt.figure(figsize=(20, 15))
        for x_val, y_vals, c_vals in zip(x_correct, y_slice, c_slice):
            plt.plot(y_vals, c_vals, label=f"x = {x_val:.2f}")

        plt.xlabel("y")
        plt.ylabel("concentration")
        plt.margins(tight=True)
        plt.legend(fontsize = 30)
        plt.savefig(self.dirname + "/concentration_profile.png", bbox_inches="tight")
        plt.close()

    def compare_conc(self, sid, inc, graph, cb):

        conc = np.loadtxt("conc_prof.txt")

        # Nodes must match order used in incidence / cb
        nodes = list(graph.nodes())
        pos = nx.get_node_attributes(graph, "pos")

        pos_x = np.array([pos[n][0] for n in nodes], dtype=float)
        pos_y = np.array([pos[n][1] for n in nodes], dtype=float)

        # Make cb a 1D vector aligned with `nodes`
        cb_vec = (np.abs(inc.incidence).T @ cb).ravel()

        # Choose x positions where you want vertical slices
        x_slice = np.array([-200, 1899, 3797, 5583, 7370]) / 7400 * (np.max(pos_x) - sid.bound_x) + sid.bound_x
        #x_slice = np.linspace(sid.bound_x + 2, np.max(pos_x) - 2, 5)

        # Snap to actual node x's (keep one snapped value per requested slice)
        atol = 1e-8
        x_correct = np.array([pos_x[np.argmin(np.abs(pos_x - x))] for x in x_slice], dtype=float)

        # Prepare figure with exactly len(x_slice) panels
        n_panels = len(x_slice)
        fig, axes = plt.subplots(
            1, n_panels,
            figsize=(15 * n_panels, 10),
            sharey=True
        )
        if n_panels == 1:
            axes = [axes]

        # Normalize reference x-axis (conc[:,0]) once
        conc_x = conc[:, 0]
        conc_x_norm = conc_x / np.max(conc_x) if np.max(conc_x) != 0 else conc_x

        # If you want to normalize your graph y the same way as conc[:,0], do it here.
        # Usually conc[:,0] is a vertical coordinate/profile axis, so normalize pos_y similarly:
        pos_y_norm_denom = np.max(pos_y) if np.max(pos_y) != 0 else 1.0

        for i, (ax, x_val) in enumerate(zip(axes, x_correct)):
            mask = np.isclose(pos_x, x_val, atol=atol)

            # --- Plot graph-based slice (y vs c) ---
            if np.any(mask):
                y_vals = pos_y[mask]
                c_vals = cb_vec[mask]

                order = np.argsort(y_vals)
                y_vals = y_vals[order]
                c_vals = c_vals[order]

                # Normalize y like conc[:,0] is normalized; normalize c to compare shapes
                y_plot = y_vals / pos_y_norm_denom
                c_denom = np.max(np.abs(c_vals))
                c_plot = c_vals / c_denom if c_denom != 0 else c_vals

                ax.plot(y_plot, c_plot, label="graph slice")
            else:
                ax.text(0.5, 0.5, "No nodes at this x", ha="center", va="center", transform=ax.transAxes)

            # --- Plot reference profile from conc_prof.txt ---
            # panel i uses conc column i+1 (since col 0 is the axis)
            ref_col = i + 1
            if conc.ndim == 2 and conc.shape[1] > ref_col:
                ref_y = conc[:, ref_col]
                ref_denom = np.average(ref_y[:len(ref_y) // 2]) #np.max(np.abs(ref_y))
                ref_y_norm = ref_y / ref_denom if ref_denom != 0 else ref_y
                ax.plot(conc_x_norm, ref_y_norm, "--", label=f"conc_prof col {ref_col}")
            else:
                ax.text(
                    0.5, 0.1,
                    f"conc_prof.txt missing column {ref_col}",
                    ha="center", va="center", transform=ax.transAxes
                )

            ax.set_title(f"x = {x_val:.2f} (panel {i})")
            ax.set_xlabel("normalized profile axis")
            ax.margins(tight=True)
            ax.legend(fontsize=10)

        axes[0].set_ylabel("normalized concentration")

        fig.tight_layout()
        fig.savefig(self.dirname + "/concentration_profile.png", bbox_inches="tight", dpi=200)
        plt.close(fig)



    def check_init_slice_channelization(self, sid:SimInputData, graph: Graph, inc: Incidence, \
        edges: Edges) -> None:
        pos_y = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,1]
        slices = np.linspace(sid.diams_y_min, sid.diams_y_max, 102)[1:-1]
        channels_tab = []
        for y in slices:
            res = self.check_diams(sid, graph, inc, edges, y)
            channels_tab.append(res)
        self.slices.append(channels_tab)

    def check_slice_channelization(self, sid:SimInputData, graph: Graph, inc: Incidence, \
        edges: Edges, time: float) -> None:
        pos_y = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,1]
        slices = np.linspace(sid.diams_y_min, sid.diams_y_max, 102)[1:-1]
        channels_tab = []
        for y in slices:
            res = self.check_diams(sid, graph, inc, edges, y)
            channels_tab.append(res)
        self.slices.append(channels_tab)
        #self.slice_times.append(f'{time:.2f}')
        self.slice_times.append("{0}".format(str(round(time, 1) if time % 1 else int(time))))

    def plot_slice_channelization(self, graph: Graph) -> None:
        """ Plots slice data from text file.

        This function loads the data from text file slices.txt and plots them
        to files slices.png, slices_no_div.png, slices_norm.png.
        """
        pos_x = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,0]
        slices = np.linspace(np.min(pos_x), np.max(pos_x), 102)[1:-1]
        edge_number  = np.array(self.slices[0])
        plt.figure(figsize = (10, 10))
        for i, channeling in enumerate(self.slices[1:]):
            plt.plot(slices, np.array(channeling) / edge_number, \
                    label = self.slice_times[i])
        plt.xlabel('x')
        plt.ylabel('channeling [%]')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(self.dirname + '/slices.png')
        plt.close()
        plt.figure(figsize = (10, 10))
        for i, channeling in enumerate(self.slices[1:]):
            plt.plot(slices, np.array(channeling) / np.array(self.slices[1]), \
                label = self.slice_times[i])
        plt.xlabel('x')
        plt.ylabel('channeling [initial]')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(self.dirname + '/slices_norm.png')
        plt.close()
        plt.figure(figsize = (10, 10))
        for i, channeling in enumerate(self.slices[1:]):
            plt.plot(slices, channeling, label = self.slice_times[i])
        plt.xlabel('x')
        plt.ylabel('channeling [edge number]')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(self.dirname + '/slices_no_div.png')
        plt.close()

    def plot_slice_channelization_v2(self, sid: SimInputData, graph: Graph) -> None:
        """ Plots slice data from text file.

        This function loads the data from text file slices.txt and plots them
        to files slices.png, slices_no_div.png, slices_norm.png.
        """
        pos_y = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,1]
        #slices = np.linspace(sid.diams_y_min, sid.diams_y_max, 102)[1:-1] - (sid.diams_y_max + sid.diams_y_min) / 2
        slices = np.linspace(0, 1, 102)[1:-1]
        initial_diams  = np.array(self.slices[0])
        # plt.figure(figsize = (10, 10))
        # for i, channeling in enumerate(self.slices[1:]):
        #     if i < i_start:
        #         plt.plot(slices, 1 - np.array(channeling) / initial_diams, \
        #                 label = self.slice_times[i])
        # plt.xlabel('y')
        # plt.ylabel('clogging index')
        # plt.ylim(0, 1)
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.savefig(self.dirname + '/slices_start.png')
        # plt.close()
        colors = ['black', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
        plt.figure(figsize = (15, 10))
        for i, channeling in enumerate(self.slices[1:]):
            label = f'{(i+1) * sid.track_every}'
            plt.plot(slices, 1 - np.array(channeling) / initial_diams, \
                        linewidth = 5, color = colors[i], label = label)
        plt.xlabel('y', fontsize = 60, style = 'italic')
        plt.ylabel('clogging index', fontsize = 50)
        plt.ylim(0, 1)
        legend = plt.legend(loc="upper center", mode = "expand", ncol = 5, prop={'size': 40}, handlelength = 1, frameon=False, borderpad = 0, handletextpad = 0.4)
        for legobj in legend.legend_handles:
            legobj.set_linewidth(10.0)
        plt.vlines([1/2 + sid.q_amp * sid.m / 20 / 2, 1/2 - sid.q_amp * sid.m / 20 / 2], 0, 1, 'r')
        plt.margins(tight = True)
        plt.savefig(self.dirname + '/slices.png', bbox_inches="tight")
        plt.close()

    def plot_participation(self, sid: SimInputData):
        plt.figure(figsize = (10, 10))
        plt.title('Participation ratio')
        ax_p = plt.subplot()
        ax_p.set_title('Participation ratio')
        ax_p.set_ylim(0, 1)
        ax_p.set_xlim(0, sid.dissolved_v_max / self.vol_init)
        ax_p.set_xlabel('dissolved v')
        ax_p.set_ylabel('participation ratio')
        ax_p2 = ax_p.twinx()
        x = np.linspace(0, sid.dissolved_v_max / self.vol_init, len(self.participation_ratio))
        ax_p2.plot(x, self.participation_ratio_nom, label = "pi", color='green', linestyle='dashed')
        ax_p2.plot(x, self.participation_ratio_denom, label = "pi'", color='red', linestyle='dashed')
        ax_p.plot(x, self.participation_ratio)
        ax_p2.legend()
        plt.savefig(self.dirname + '/participation_ratio.pdf')
        plt.close()


    def plot_precipitate(self):
        plt.figure(figsize = (20, 15))
        #plt.yscale('log')
        plt.plot(self.t, np.abs(self.dissolved_v_list), 'r')
        #plt.xlim(0, 3000)
        plt.xlabel('simulation time')
        plt.ylabel(f'precipitate fraction')
        plt.margins(tight = True)
        plt.savefig(self.dirname + '/precipitate.png', bbox_inches="tight")

    def plot_conductivity(self):
        plt.figure(figsize = (10, 7))
        #plt.yscale('log')
        plt.plot(self.t, self.cond_ratio_cb, 'k', label = 'upper channel')
        plt.plot(self.t, self.cond_ratio_cc, 'r', label = 'lower channel')
        #plt.xlim(0, 3000)
        plt.xlabel('simulation time')
        plt.ylabel(f'conductivity ratio')
        plt.margins(tight = True)
        plt.legend(fontsize = 15)
        plt.savefig(self.dirname + '/conductivity.png', bbox_inches="tight")

    def plot_interface_width(self):
        plt.figure(figsize = (10, 7))
        #plt.yscale('log')
        plt.plot(self.t, self.interface_width, 'k')
        #plt.xlim(0, 3000)
        plt.xlabel('simulation time')
        plt.ylabel(f'interface width')
        plt.margins(tight = True)
        plt.savefig(self.dirname + '/interface_width.png', bbox_inches="tight")