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

from config import SimInputData
from network import Edges, Graph, Triangles
from incidence import Incidence
from volumes import Volumes

font = {'family' : 'Liberation Serif',
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
    order = []
    participation_ratio = []
    participation_ratio_nom = []
    participation_ratio_denom = []
    porosity = []
    replaced = []
    cb_out = []
    cc_out = []
    delta_b = 0.
    delta_c = 0.
    delta_d = 0.
    delta_d_list = []
    dissolved_v = 0.
    dissolved_v_list = []
    slices: list = []
    slices_d: list = []
    slices_s: list = []
    slices_phi: list = []
    "channelization for slices through the whole system in a given time"
    slice_times: list = []
    "list of times of checking slice channelization"
    breakthrough_times: list = []
    concentrations: list = []
    reactive_breakthrough_times: list = []
    track_times: list = []
    vol_dissolved = 0.
    vol_precipitated = 0.
    vol_dissolved_list = []
    vol_precipitated_list = []

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
                np.savetxt(file, np.array([self.t, self.dissolved_v_list, self.pressure, self.porosity, self.replaced, self.cb_out, \
                    self.cc_out, self.vol_dissolved_list, self.vol_precipitated_list, self.delta_d_list], dtype = float).T)
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
        is_saved = False
        while not is_saved: # prevents problems with opening text file
            try:
                file = open(self.dirname + '/profiles_phi.txt', 'w', \
                    encoding = "utf-8")
                np.savetxt(file, self.slices_phi)
                file.close()
                is_saved = True
            except PermissionError:
                pass
        # is_saved = False
        # while not is_saved: # prevents problems with opening text file
        #     try:
        #         file = open(self.dirname + '/track.txt', 'w', \
        #             encoding = "utf-8")
        #         np.savetxt(file, self.breakthrough_times)
        #         file.close()
        #         file = open(self.dirname + '/c_track.txt', 'w', \
        #             encoding = "utf-8")
        #         np.savetxt(file, self.concentrations)
        #         file.close()
        #         file = open(self.dirname + '/r_track.txt', 'w', \
        #             encoding = "utf-8")
        #         np.savetxt(file, self.reactive_breakthrough_times)
        #         file.close()
        #         is_saved = True
        #     except PermissionError:
        #         pass

    def load_data(self) -> None:
        data = np.loadtxt(self.dirname + '/params.txt').T
        self.t, self.pressure, self.participation_ratio, self.cb_out, \
            self.cc_out = list(data[0]), list(data[1]), list(data[2]), list(data[3]), list(data[4])
        self.slices = list(np.loadtxt(self.dirname + '/slices.txt'))

    def check_data(self, edges: Edges) -> None:
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
        if np.abs(Q_in - Q_out) > 1:
            raise ValueError('Flow not matching!')
        # delta = np.abs((np.abs(inc.incidence.T < 0) @ (np.abs(edges.flow) \
        #     * edges.inlet) - np.abs(inc.incidence.T > 0) @ (np.abs(edges.flow) \
        #     * edges.outlet)) @ cb * sid.dt)


    def collect_data(self, sid: SimInputData, inc: Incidence, edges: Edges, vols, \
        p: np.ndarray, cb: np.ndarray, cc: np.ndarray, cd: np.ndarray) -> None:
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
        self.t.append(sid.old_t + sid.dt)

        self.pressure.append(np.max(p))
        self.order.append((sid.ne - np.sum(edges.flow ** 2) ** 2 \
            / np.sum(edges.flow ** 4)) / (sid.ne - 1))
        pi = np.sum(edges.diams ** 2 * np.abs(edges.flow)) ** 2 / np.sum(edges.diams ** 2 \
            * np.abs(edges.flow) ** 2) / sid.nsq
        pi_prime = np.sum(edges.diams ** 2) / sid.nsq
        self.participation_ratio_nom.append(pi)
        self.participation_ratio_denom.append(pi_prime)
        self.participation_ratio.append(pi / pi_prime)
        # calculate the difference between inflow and outflow of each substance

        # if sid.include_diffusion:

        #     lam_plus_val = sid.Pe / 2 / edges.diams ** 2 * \
        #         (np.sqrt(np.abs(edges.flow) ** 2 + 4 * edges.alpha * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) + np.abs(edges.flow))
        #     lam_plus_val = np.array(np.ma.fix_invalid(lam_plus_val, fill_value = 0))
        #     lam_plus_zero = 1 * (lam_plus_val > sid.diffusion_exp_limit)
        #     lam_plus_val = lam_plus_val * (1 - lam_plus_zero)
        #     lam_minus_val = sid.Pe / 2 / edges.diams ** 2 * \
        #         (np.sqrt(np.abs(edges.flow) ** 2 + 4 * edges.alpha * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) - np.abs(edges.flow))
        #     lam_minus_val = np.array(np.ma.fix_invalid(lam_minus_val, fill_value = 0))
        #     # Not sure how to calculate J_out - should it be just q_out c_out (as we set dc/dx = 0 at the outlet? - do we for 100%?)
        #     # But no matter how I calculate, I end up with a small error, there could be some small bug
        #     #J_in2 = np.sum(edges.inlet * (np.abs(edges.flow) * (edges.A * (1 - lam_plus_zero) + edges.B) - (1 - lam_plus_zero) / sid.Pe * edges.diams ** 2 * (lam_plus_val * edges.A - lam_minus_val * edges.B)))
        #     #J_in = np.sum(edges.inlet * (np.abs(edges.flow) * (edges.A + edges.B) - 1 / sid.Pe * edges.diams ** 2 *  (lam_plus_val * edges.A - lam_minus_val * edges.B)))
        #     #J_out2 = np.sum((1 - lam_plus_zero) * edges.outlet * (np.abs(edges.flow) * (edges.A * np.exp(lam_plus_val * edges.lens) + edges.B * np.exp(-lam_minus_val * edges.lens)) - 1 / sid.Pe * edges.diams ** 2 *(lam_plus_val * edges.A * np.exp(lam_plus_val * edges.lens) - lam_minus_val * edges.B * np.exp(-lam_minus_val * edges.lens))) + lam_plus_zero * edges.outlet * edges.B * np.abs(edges.flow) * np.exp(-edges.alpha * sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow)))
        #     #J_out = np.sum(edges.outlet * (np.abs(edges.flow) * (edges.A * np.exp(lam_plus_val * edges.lens) + edges.B * np.exp(-lam_minus_val * edges.lens)) - 1 / sid.Pe * edges.diams ** 2 *(lam_plus_val * edges.A * np.exp(lam_plus_val * edges.lens) - lam_minus_val * edges.B * np.exp(-lam_minus_val * edges.lens))))
        #     #J_out = np.sum(edges.outlet * (np.abs(edges.flow) * (edges.A * np.exp(lam_plus_val * edges.lens) + edges.B * np.exp(-lam_minus_val * edges.lens)) - 1 / sid.Pe * edges.diams ** 2 *(lam_plus_val * edges.A * np.exp(lam_plus_val * edges.lens) - lam_minus_val * edges.B * np.exp(-lam_minus_val * edges.lens))))
        #     J_in = np.sum(edges.inlet * (np.abs(edges.flow) * (edges.A * (1 - lam_plus_zero) + edges.B) - (1 - lam_plus_zero) / sid.Pe * edges.diams ** 2 * (lam_plus_val * edges.A - lam_minus_val * edges.B)))
        #     J_out = np.abs(np.abs(inc.incidence.T > 0) @ (np.abs(edges.flow) \
        #          * edges.outlet)) @ cb
        
        # J_out3 = np.abs(np.abs(inc.incidence.T > 0) @ (np.abs(edges.flow) \
        #     * edges.outlet)) @ cb
        # print(f'lam plus zero: {np.sum(lam_plus_zero), np.sum(1 - lam_plus_zero)}')
        
        # print(f'J_in2: {J_in2 * sid.dt}, J_out2: {J_out2 * sid.dt}')
        # print(f'delta J_adv: {(np.sum(edges.inlet * (np.abs(edges.flow) * (edges.A + edges.B)))-np.sum(edges.outlet * (np.abs(edges.flow) * (edges.A * np.exp(lam_plus_val * edges.lens) + edges.B * np.exp(-lam_minus_val * edges.lens))))) * sid.dt}')
        # print(f'delta J_diff: {(np.sum(edges.inlet * (- 1 / sid.Pe * edges.diams ** 2 *  (lam_plus_val * edges.A - lam_minus_val * edges.B)))-np.sum(edges.outlet * (-1 / sid.Pe * edges.diams ** 2 *(lam_plus_val * edges.A * np.exp(lam_plus_val * edges.lens) - lam_minus_val * edges.B * np.exp(-lam_minus_val * edges.lens))))) * sid.dt}')

        # delta = np.abs((np.abs(inc.incidence.T < 0) @ (np.abs(edges.flow) \
        #     * edges.inlet) - np.abs(inc.incidence.T > 0) @ (np.abs(edges.flow) \
        #     * edges.outlet)) @ cb * sid.dt)
        if sid.include_diffusion:
            print(f'J_in: {self.J_in}, J_out: {self.J_out}')
            delta = (self.J_in - self.J_out) * sid.dt
        else:
            # delta = np.abs((np.abs(inc.incidence.T < 0) @ (np.abs(edges.flow) \
            #     * edges.inlet) - np.abs(inc.incidence.T > 0) @ (np.abs(edges.flow) \
            #     * edges.outlet)) @ cb * sid.dt)
            delta = np.abs(np.abs(1 * ( inc.incidence.T @ spr.diags(edges.flow) > 0) @ (np.abs(edges.flow) \
                 * edges.inlet)) @ cb - np.abs(1 * ( inc.incidence.T @ spr.diags(edges.flow) < 0) @ (np.abs(edges.flow) \
                 * edges.outlet)) @ cb) * sid.dt
        vol_dissolved = np.sum(edges.diams ** 2 * edges.lens) - self.vol_init
        vol_a = np.sum(vols.vol_a_0 - vols.vol_a)
        print(f'Zero volume: {np.sum(vols.vol_a == 0)}')
        self.delta_b += delta
        # delta2 = np.abs((np.abs(inc.incidence.T < 0) @ (np.abs(edges.flow) \
        #         * edges.inlet) - np.abs(inc.incidence.T > 0) @ (np.abs(edges.flow) \
        #         * edges.outlet)) @ cb * sid.dt)
        # print(f'Delta2: {delta2}')
        # print(f'Delta3: {(J_in - J_out2) * sid.dt}')
        # print(f'Delta4: {(J_in - J_out3) * sid.dt}')
        print(f'Used concentration: {self.delta_b}, Dissolved volume: {sid.Da * vol_dissolved}, Dissolved volume A: {sid.Da * vol_a}')
        #print(f'c - V: {(self.delta_b - sid.Da * vol_dissolved / 2) / self.delta_b}, c - V_A: {(self.delta_b - sid.Da * vol_a / 2) / self.delta_b}, V - V_A {(vol_dissolved - vol_a) / vol_dissolved}')
        #print(f'c - V: {(self.delta_b - sid.Da * vol_dissolved)}, c - V_A: {(self.delta_b - sid.Da * vol_a)}, V - V_A {(vol_dissolved - vol_a)}')
        self.cb_out.append(self.delta_b)
        delta_c = np.abs(np.abs(1 * ( inc.incidence.T @ spr.diags(edges.flow) > 0) @ (np.abs(edges.flow) \
                 * edges.inlet)) @ cc - np.abs(1 * ( inc.incidence.T @ spr.diags(edges.flow) < 0) @ (np.abs(edges.flow) \
                 * edges.outlet)) @ cc) * sid.dt
        self.delta_c += delta_c
        delta_d = np.abs(np.abs(1 * ( inc.incidence.T @ spr.diags(edges.flow) > 0) @ (np.abs(edges.flow) \
                 * edges.inlet)) @ cd - np.abs(1 * ( inc.incidence.T @ spr.diags(edges.flow) < 0) @ (np.abs(edges.flow) \
                 * edges.outlet)) @ cd) * sid.dt
        self.delta_d += delta_d
        self.delta_d_list.append(self.delta_d)
        # self.injected_d += np.abs(np.abs(1 * ( inc.incidence.T @ spr.diags(edges.flow) > 0) @ (np.abs(edges.flow) \
        #          * edges.inlet)) @ cd) * sid.dt
        # self.injected_list.append(self.injected_d)
        print(f'Delta c_B: {self.delta_b}, Delta c_C: {self.delta_c}, Delta c_D: {self.delta_d}')
        print(f'Delta c_B + Delta c_C - Delta c_D: {(delta - delta_c - delta_d)}')
        # if np.abs(delta - delta_c - delta_d) > 1e-3:
        #     raise ValueError
        self.cc_out.append(self.delta_c)
        self.dissolved_v = np.sum(vols.vol_a) / np.sum(vols.vol_max)
        self.dissolved_v_list.append(self.dissolved_v)
        self.porosity.append(1 - np.sum(vols.vol_a + vols.vol_e) / np.sum(vols.vol_max))
        self.replaced.append(np.sum(vols.vol_e) / np.sum(vols.vol_max))
        self.vol_dissolved_list.append(self.vol_dissolved)
        self.vol_precipitated_list.append(self.vol_precipitated)
        

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
        pos_x = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,0]
        slices = np.linspace(np.min(pos_x), np.max(pos_x), 102)[1:-1]
        edge_number  = np.array(self.slices[0])
        i_start = 5
        i_division = sid.dissolved_v_max // sid.track_every // 5
        plt.figure(figsize = (10, 10))
        for i, channeling in enumerate(self.slices[1:]):
            if i < i_start:
                plt.plot(slices, (edge_number - 2 * np.array(channeling)) / edge_number, \
                        label = self.slice_times[i])
        plt.xlabel('x')
        plt.ylabel('flow focusing index')
        plt.ylim(0, 1)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(self.dirname + '/slices_start.png')
        plt.close()
        plt.figure(figsize = (10, 10))
        for i, channeling in enumerate(self.slices[1:]):
            if i % i_division == 0:
                plt.plot(slices, (edge_number - 2 * np.array(channeling)) / edge_number, \
                        label = self.slice_times[i])
        plt.xlabel('x')
        plt.ylabel('flow focusing index')
        plt.ylim(0, 1)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(self.dirname + '/slices.png')
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
        plt.plot([], [], ' ', label=' ')
        plt.plot([], [], ' ', label=' ')
        plt.plot([], [], ' ', label=' ')
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
        #order = [0,4,1,5,2,6,3,7]
        order = []
        for i in range(len(handles) // 2):
            order.append(i)
            if i == len(handles) // 2 - 1:
                if len(handles) % 2 == 0:
                    order.append(len(handles) // 2 + i)
            else:
                order.append(len(handles) // 2 + i)
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
        plt.savefig(self.dirname + "/profile.png", bbox_inches="tight")
        plt.close()

    def plot_things(self, sid: SimInputData):
        plt.figure(figsize = (15, 10))
        x = sid.cb_in * np.array(self.t) * sid.Q_in / (2 * sid.Da * sid.ne * sid.phi / (1 - sid.phi))
        plt.title('Permeability')
        plt.grid()
        plt.plot(x, self.pressure[0] / self.pressure, linewidth = 5, color = 'black')
        plt.yscale('log')
        plt.xlabel(r'injected B $\nu_A / V^0_A$', fontsize = 50)
        #plt.subplots_adjust(wspace=0, hspace=0)
        plt.margins(tight = True)
        plt.ylabel(r'$\kappa / \kappa_0$', fontsize = 50)
        plt.savefig(self.dirname + '/permeability.png', bbox_inches="tight")
        plt.close()
        plt.figure(figsize = (15, 10))
        plt.title('Porosity')
        plt.grid()
        plt.plot(x, self.porosity, linewidth = 5, color = 'black')
        plt.xlabel(r'injected B $\nu_A / V^0_A$', fontsize = 50)
        #plt.subplots_adjust(wspace=0, hspace=0)
        plt.margins(tight = True)
        plt.ylabel(r'$\phi$', fontsize = 50)
        plt.yscale('log')
        plt.savefig(self.dirname + '/porosity.png', bbox_inches="tight")
        plt.close()
        plt.figure(figsize = (15, 10))
        plt.title('Mineral evolution')
        plt.plot(x, self.dissolved_v_list, linewidth = 5, color = 'black', label = 'A')
        # plt.xlabel(r'$\Delta V_A / V^\text{init}_A$', fontsize = 50)
        # #plt.subplots_adjust(wspace=0, hspace=0)
        # plt.margins(tight = True)
        # plt.ylabel(r'injected B', fontsize = 50)
        # plt.savefig(self.dirname + '/replaced.png', bbox_inches="tight")
        # plt.close()
        # plt.figure(figsize = (15, 10))
        # plt.title('Replaced volume')
        plt.grid()
        plt.plot(x, self.replaced, '--', linewidth = 5, color = 'black', label = 'E')
        plt.ylabel(r'$\Delta V / V^\text{tot}$', fontsize = 50)
        #plt.subplots_adjust(wspace=0, hspace=0)
        plt.margins(tight = True)
        plt.legend()
        plt.xlabel(r'injected B $\nu_A / V^0_A$', fontsize = 50)
        plt.savefig(self.dirname + '/replaced.png', bbox_inches="tight")
        plt.close()
        plt.figure(figsize = (15, 10))
        plt.title('Reacted D')
        plt.plot(sid.cb_in * np.array(self.t) * sid.Q_in / (2 * sid.Da * sid.ne * sid.phi / (1 - sid.phi)), np.array(self.delta_d_list) / (2 * sid.Da * sid.ne * sid.phi / (1 - sid.phi)), linewidth = 5, color = 'black')
        plt.grid()
        plt.ylabel(r'reacted  D $\nu_A / V^0_A$', fontsize = 50)
        #plt.subplots_adjust(wspace=0, hspace=0)
        plt.margins(tight = True)
        plt.xlabel(r'injected B $\nu_A / V^0_A$', fontsize = 50)
        plt.savefig(self.dirname + '/reacted_d.png', bbox_inches="tight")
        plt.close()

    def check_porosity_profile(self, graph: Graph, inc: Incidence, edges: Edges, triangles: Triangles, \
        vols: Volumes, slice_x: float) -> tuple[int, float]:
        
        pos_x = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,0]
        slice_triangles = ((triangles.node_incidence @ (pos_x >= slice_x)) \
            * (triangles.node_incidence @ (pos_x <= slice_x))) > 0
        slice_porosity = 1 - (np.sum(slice_triangles * (vols.vol_a + vols.vol_e))) / (np.sum(slice_triangles * vols.vol_max))
        return slice_porosity, np.sum(slice_triangles)

    def check_slice_porosity(self, graph: Graph, inc: Incidence, \
        edges: Edges, triangles: Triangles, vols: Volumes, time) -> None:
        pos_x = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,0]
        slices = np.linspace(np.min(pos_x), np.max(pos_x), 102)[1:-1]
        channels_tab = []
        for x in slices:
            res = self.check_porosity_profile(graph, inc, edges, triangles, vols, x)
            channels_tab.append(res[0])
        self.slice_times.append(time)
        self.slices_phi.append(channels_tab)

    def plot_porosity_profile(self, graph: Graph) -> None:
        """ Plots slice data from text file.

        This function loads the data from text file slices.txt and plots them
        to files slices.png, slices_no_div.png, slices_norm.png.
        """
        pos_x = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,0]
        # slices = np.linspace(np.min(pos_x), np.max(pos_x), 120)[10:-10]
        slices = np.linspace(np.min(pos_x), np.max(pos_x), 102)[1:-1]
        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
        plt.figure(figsize = (15, 10))
        for i, channeling in enumerate(self.slices_phi):
            plt.plot(slices, self.slices_phi[i], label = self.slice_times[i], color = colors[i], linewidth = 5)
        #plt.ylim(0, 1.05)
        plt.xlabel('x', fontsize = 60, style = 'italic')
        # ax2.xaxis.label.set_color('white')
        # ax2.tick_params(axis = 'x', colors='white')
        #plt.xticks([],[])
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.margins(tight = True)
        plt.ylabel('porosity', fontsize = 50)
        #plt.yticks([],[])
        plt.yticks([0, 0.5, 1],['0', '0.5', '1'])
        legend = plt.legend(loc="lower center", mode = "expand", ncol = 4, prop={'size': 40}, handlelength = 1, frameon=False, borderpad = 0, handletextpad = 0.4)
        for legobj in legend.legend_handles:
            legobj.set_linewidth(10.0)
        plt.savefig(self.dirname + "/porosity_profile.png", bbox_inches="tight")
        plt.close()