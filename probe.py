from matplotlib import gridspec
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import scipy.sparse as spr
import numpy as np
import random
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from config import SimInputData
from network import Edges, Graph
from incidence import Incidence

class Probe: 
    """ Debug tool. View local time series data at pre-defined locations.

        Usage in __main__:
            path_probe = Da.Probe(sid, edges, graph, snode=8, opts="path", total_nodes=10)
            path_probe.show_probes(sid, edges, graph)
            ...
            while(t < T)
            ...
                path_probe.record(
                    t = current_time,
                    node_data = {'cb': cb, 'pressure': p},
                    edge_data = {'f': edges.f, 'flow': edges.flow, 'diams': edges.diams}
                )
        TODO: 'live' probes? e.g. hotspots on the network 

    """

    def __init__(self, sid, edges, graph, snode=20, opts=None, total_nodes=None):
        self.start_node = snode
        self.probe_type = opts
        self.edge_probe_idxs = []
        self.node_probe_idxs = []
        self.edge_probe_tuples = []  
        
        self.history = {'t': []}
        self.node_history = {} # {'cb': [] ...}
        self.edge_history = {} # {'flow': [] ...]}
       

        if opts == "path": 
            lr_nodes, lr_path, lr_idx = self.get_left_to_right_path(sid, edges, graph)
            
            len_subset = total_nodes
            if total_nodes is None:
                len_subset = int(0.1 * len(lr_nodes)) # default 10%
            step = max(1, len(lr_nodes) // max(1, len_subset))
            probe_sl = slice(1, len(lr_nodes)-1, step)

            self.node_probe_idxs = np.array(lr_nodes[probe_sl])
            self.edge_probe_tuples = np.array(lr_path[probe_sl]) 
            self.edge_probe_idxs = np.array(lr_idx[probe_sl])
            
        elif opts == "grid":
            grid_nodes, grid_path, grid_idx = self.get_grid_probes(sid, edges, graph, total_nodes)
            self.node_probe_idxs = grid_nodes
            self.edge_probe_tuples = grid_path
            self.edge_probe_idxs = grid_idx
        else:
            raise ValueError(f"ERROR: Invalid probe type '{opts}'. Expected 'path' or 'grid'.")
        
        #  
        palette = plt.get_cmap('tab20').colors + plt.get_cmap('tab20b').colors + plt.get_cmap('tab20c').colors
        self.probe_colors = {node_idx: palette[i % 60] for i, node_idx in enumerate(self.node_probe_idxs)}

    def record(self, t, globals_data=None, node_data=None, edge_data=None):
        """Record network observables at speicfied location 
        Parameters:
        -----------
        t : float
        globals_data : dict
            Dictionary of scalars
        node_data : dict, optional
            Dictionary of node arrays (cb, etc)
        edge_data : dict, optional
            Dictionary of edge arrays (flow, etc)
        """
        self.history['t'].append(t)
        # Global scalars 
        if globals_data:
            for key, val in globals_data.items():
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(val)
        # Node data
        if node_data:
            for key, array in node_data.items():
                if key not in self.node_history:
                    self.node_history[key] = []
                self.node_history[key].append(np.array(array)[self.node_probe_idxs])
        # Edge data
        if edge_data:
            for key, array in edge_data.items():
                if key not in self.edge_history:
                    self.edge_history[key] = []
                self.edge_history[key].append(np.array(array)[self.edge_probe_idxs])

    def plot_time_series_data(self, sid, edges, graph):
        """ Generic plot of all recorded edge / node data against time
        """

        # Setup square plot grid
        keys_node = list(self.node_history.keys())
        keys_edge = list(self.edge_history.keys())
        total_plots = 1 + len(keys_node) + len(keys_edge)
        grid_dim = int(np.ceil(np.sqrt(total_plots)))
        fig, ax = plt.subplots(grid_dim, grid_dim, figsize=(grid_dim*4, grid_dim*4))
        axs = ax.flatten() if total_plots > 1 else [ax]

        # Top left: network with probes overlay
        self.show_probes(sid, edges, graph, ax=axs[0])

        cmap = plt.get_cmap('tab20')
        plot_idx = 1 

        t = np.array(self.history['t'])
        # Node data
        for key in keys_node:
            ax = axs[plot_idx]
            data = np.array(self.node_history[key]) 
            
            # Plotting node data
            for p_idx, node_idx in enumerate(self.node_probe_idxs):
                c = self.probe_colors[self.node_probe_idxs[p_idx]]
                ax.plot(t, data[:, p_idx], color=c)#, label=f'Node {node_idx}')
            ax.set_ylabel(key)
            ax.set_xlabel('t')
            #ax.set_title(f'{key}')
            ax.grid(True, linestyle='--', alpha=0.6)
            plot_idx += 1

        # Edge data
        for key in keys_edge:
            #if data.ndim > 2:
            #    continue
            ax = axs[plot_idx]
            data = np.array(self.edge_history[key])
            
            # Assumes vector data on an edge is spatial average
            if data.ndim == 3: 
                data = np.mean(data, axis=2) 

            # Plotting edge data
            for p_idx, edge_tuple in enumerate(self.edge_probe_tuples):
                #c = cmap(p_idx % 10)
                c = self.probe_colors[self.node_probe_idxs[p_idx]]
                ax.plot(t, data[:, p_idx], color=c)#, label=f'Edge {edge_tuple}')
            ax.set_ylabel(key)
            ax.set_xlabel('t')
            ax.grid(True, linestyle='--', alpha=0.6)
            plot_idx += 1

        for i in range(plot_idx, len(axs)):
            axs[i].set_visible(False)

        plt.tight_layout()
        plt.show()

    def get_left_to_right_path(self, sid, edges: Edges, graph: Graph):
        """ Gets a random path from inlet to outlet that moves +ve horiz. direction
        TODO: Paths passing through PBC not nice
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
            # TODO: Could have more 'specific' path types
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

    def get_grid_probes(self, sid, edges, graph, total_nodes=None):
        """Gets a square grid of nodes by defining target (x,y) grid
        Parameters:
        --------
            total_nodes : int
                Square number of nodes to form a grid of probes    
        """
        if total_nodes is None:
            total_nodes = 9  

        n_side = int(np.round(np.sqrt(total_nodes)))

        # create x-y target grid 
        x_targets = np.linspace(0, sid.m, n_side + 2)[1:-1]
        y_targets = np.linspace(0, sid.n, n_side + 2)[1:-1]

        # Node positions minus inlet / outlet nodes
        pos = nx.get_node_attributes(graph, 'pos')
        valid_nodes = [n for n in graph.nodes if n not in graph.in_nodes and n not in graph.out_nodes]

        grid_nodes = []
        grid_edges = []
        grid_edge_idxs = []

        for xt in x_targets:
            for yt in y_targets:
                # Closest node to the current point
                closest_node = None
                min_dist = float('inf')
                for n in valid_nodes:
                    nx_pos, ny_pos = pos[n]
                    dist = (nx_pos - xt)**2 + (ny_pos - yt)**2 
                    if dist < min_dist:
                        min_dist = dist
                        closest_node = n
                
                u = closest_node
                # Get a random edge connected to closest node
                # TODO: Get edge with positive flow or forward direction?
                neighbors = list(graph.neighbors(u))
                v = random.choice(neighbors)
                if (u, v) in edges.edge_list:
                    e_idx = edges.edge_list.index((u, v))
                elif (v, u) in edges.edge_list:
                    e_idx = edges.edge_list.index((v, u))
                else:
                    print(f"Probe Warning: Edge between {u} and {v} not found. Skipping.")
                    continue 
                grid_nodes.append(u)
                grid_edges.append((u, v))
                grid_edge_idxs.append(e_idx)
        return np.array(grid_nodes), np.array(grid_edges), np.array(grid_edge_idxs)

    def show_probes(self, sid, edges, graph, labels=True, ax=None):
        """ Displays the network with probes highlighted  

        TODO: Path that wraps across P.B. won't be drawn correctly 
        TODO: Probably does not adapt to merging ?
        """
        show_plot = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(sid.figsize, sid.figsize))
            fig.suptitle("Probe Locations", fontsize=15)
            show_plot = True

        pos = nx.get_node_attributes(graph, 'pos')
        if sid.include_nucleation:
            edge_colors = plt.cm.copper_r(mcolors.Normalize(0, 1)(edges.ftrans)) 
        else:
            edge_colors = 'k'
            
        # Pre-calculate shared node coordinates
        xi, yi = zip(*[pos[n] for n in graph.in_nodes])
        xo, yo = zip(*[pos[n] for n in graph.out_nodes])
        ax.set_aspect('equal')
        ax.set_xlim([-0.5, sid.n + 0.5])
        ax.set_ylim([-0.5, sid.n + 0.5])
        ax.set_axis_off() 
        
        ax.scatter(xi, yi, s=1000/sid.n, fc='white', ec='black', zorder=3)
        ax.scatter(xo, yo, s=1000/sid.n, fc='black', ec='white', zorder=3)
        ax.margins(0)
        qs = (1 - edges.boundary_list) * (edges.diams * (edges.diams > 0))
        w = sid.ddrawconst * np.array(qs)

        # Draw the full network
        nx.draw_networkx_edges(graph, pos, edgelist=edges.edge_list, 
                               edge_color=edge_colors, width=w, ax=ax)
        # Highlight selected edges along a path in the network
        nx.draw_networkx_edges(graph, pos, edgelist=self.edge_probe_tuples, edge_color='red', 
                width=w*2, alpha=1.0, ax=ax)
        
        # Highlight selected nodes
        all_pos = nx.get_node_attributes(graph, 'pos')
        for node_idx in self.node_probe_idxs:
            x, y = all_pos[node_idx]
            c = self.probe_colors[node_idx]
            ax.scatter(x, y, s=120, color=c, alpha=1, label=f'Node {node_idx}')
            if labels:
                ax.text(x, y + (sid.n * 0.05), str(node_idx), 
                    fontsize=12, fontweight='bold', ha='center', color='blue')
        ax.legend(loc='center right', bbox_to_anchor=(-0.05, 0.5), 
            frameon=True, fontsize=10, alignment='left')
        if show_plot:
            plt.tight_layout()
            plt.show()
            plt.close()
        return ax
