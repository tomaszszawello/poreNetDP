""" Merge pores in the network.

This module merges pores in the network when the sum of their diameters exceed
a threshold fraction of the distance between them. Nodes from which the edges
originate are merged into one with new position being a diameter-weighted
average of the positions of original nodes. However lenghts of other edges
connected to that new effective node remain unchanged, despite the new
position. After merging entries in the incidence matrices are changed...

Conserve pore space? What with the 3rd edge (connecting merged nodes)?

"""
import networkx as nx
import numpy as np
import scipy.sparse as spr

from config import SimInputData
from network import Edges, Graph
from incidence import Incidence

import draw_net as Dr



def solve_merging(sid: SimInputData, inc: Incidence, graph: Graph, \
    edges: Edges, merging_type: str = 'standard'):
    """ Find edges which should be merged.

    """
    pos = nx.get_node_attributes(graph, 'pos')
    merge_edges = ((inc.merge @ spr.diags(1 - edges.merged) > 0) \
        @ spr.diags(edges.diams) + ((inc.merge.T @ spr.diags(1 - edges.merged) \
            > 0) @ spr.diags(edges.diams)).T) / 2 > inc.merge
    merged, zeroed, transversed = [], [], []
    merged_diams, zeroed_diams = [], []
    merged_nodes = []
    # take coordinates of nonzero matrix elements - these are indices of edges
    # that we want to merge
    for edge_pair in list(zip(merge_edges.nonzero()[0], \
        merge_edges.nonzero()[1])):
        # choose which edge will remain - the one with larger diameter
        if (edges.inlet[edge_pair[1]] and edges.inlet[edge_pair[0]]) or \
            (edges.outlet[edge_pair[1]] and edges.outlet[edge_pair[0]]):
            if edges.diams[edge_pair[1]] > edges.diams[edge_pair[0]]:
                merge_i = edge_pair[1]
                zero_i = edge_pair[0]
            else:
                merge_i = edge_pair[0]
                zero_i = edge_pair[1]
        elif edges.inlet[edge_pair[1]] or edges.outlet[edge_pair[1]]:
            merge_i = edge_pair[1]
            zero_i = edge_pair[0]
        elif edges.inlet[edge_pair[0]] or edges.outlet[edge_pair[0]]:
            merge_i = edge_pair[0]
            zero_i = edge_pair[1]
        elif edges.diams[edge_pair[1]] > edges.diams[edge_pair[0]]:
            merge_i = edge_pair[1]
            zero_i = edge_pair[0]
        else:
            merge_i = edge_pair[0]
            zero_i = edge_pair[1]
        transversed_flat = [t_edge for transverse_list in transversed \
            for t_edge in transverse_list]
        if merge_i in merged + zeroed + transversed_flat or zero_i in merged + \
            zeroed + transversed_flat:
            continue
        n1, n2 = inc.incidence[merge_i].nonzero()[1]
        n3, n4 = inc.incidence[zero_i].nonzero()[1]
        if (n1 in graph.in_nodes and n2 in graph.in_nodes) or \
            (n1 in graph.out_nodes and n2 in graph.out_nodes):
            continue
        if (n3 in graph.in_nodes and n4 in graph.in_nodes) or \
            (n3 in graph.out_nodes and n4 in graph.out_nodes):
            continue
        if n1 == n3:
            merge_node = n2
            zero_node = n4
        elif n1 == n4:
            merge_node = n2
            zero_node = n3
        elif n2 == n3:
            merge_node = n1
            zero_node = n4
        elif n2 == n4:
            merge_node = n1
            zero_node = n3
        else:
            raise ValueError("Wrong merging!")
        merge_nodes_flat = np.reshape(merged_nodes, 2 * len(merged_nodes))
        if merge_node in merge_nodes_flat or zero_node in merge_nodes_flat:
            continue
        # we need the third edge taking part in merging
        transverse_i_list = []
        if merge_node != zero_node:
            for i in inc.incidence.T[merge_node].nonzero()[1]:
                if inc.incidence[i, zero_node] != 0:
                    transverse_i_list.append(i)
        flag = 0
        for transverse_i in transverse_i_list:
            if transverse_i in merged + zeroed + transversed_flat:
                flag = 1
            if edges.diams[transverse_i] > edges.diams[merge_i]:
                flag = 1
        if flag:
            continue
        print (f"Merging {merge_i} {zero_i} {transverse_i_list} Nodes \
            {merge_node} {zero_node}")
        merged.append(merge_i)
        zeroed.append(zero_i)
        transversed.append(transverse_i_list)
        # for transverse_i in transverse_i_list:
        #     transversed.append(transverse_i)
        # TO DO: consider how merged diameter should be calculated
        d1, d2, l1, l2 = edges.diams[merge_i], edges.diams[zero_i], \
            edges.lens[merge_i], edges.lens[zero_i]
        edges.diams[merge_i] = d1 + d2
        edges.diams_initial[merge_i] += edges.diams_initial[zero_i]
        edges.diams_initial[zero_i] = 0
        edges.lens[merge_i] = d1 * l1 + d2 * l2
        edges.diams[zero_i] = 0
        for transverse_i in transverse_i_list:
            di, li = edges.diams[transverse_i], edges.lens[transverse_i]
            edges.diams[merge_i] += di
            edges.diams_initial[merge_i] += edges.diams_initial[transverse_i]
            edges.diams_initial[transverse_i] = 0
            edges.diams[transverse_i] = 0
            edges.lens[merge_i] += di * li
            edges.lens[transverse_i] = 1
            if edges.inlet[transverse_i]:
                edges.inlet[merge_i] = 1
            if edges.outlet[transverse_i]:
                edges.outlet[merge_i] = 1
            edges.inlet[transverse_i] = 0
            edges.outlet[transverse_i] = 0
            if edges.boundary_list[transverse_i]:
                edges.boundary_list[merge_i] = 1
        merged_diams.append((edges.diams[merge_i] - d1) / 2)
        zeroed_diams.append((edges.diams[merge_i] - d2) / 2)
        # TO DO: does it matter how we set lenghts of merged edges?
        edges.lens[merge_i] /= edges.diams[merge_i]
        # edges.lens[merge_i] = edges.diams[merge_i] ** 4 \
        # / (d1 ** 4 / l1 + d2 ** 4 / l2)
        edges.lens[zero_i] = 1
        edges.inlet[zero_i] = 0
        edges.outlet[zero_i] = 0
        if edges.boundary_list[zero_i]:
            edges.boundary_list[merge_i] = 1
        if merge_node != zero_node:
            merged_nodes.append((zero_node, merge_node))
            if merging_type != 'initial':
                if n3 != merge_node and n3 != zero_node:
                    if merge_node not in graph.boundary_nodes and \
                        zero_node not in graph.boundary_nodes and \
                        n3 not in graph.boundary_nodes:
                        graph.merged_triangles.append((merge_i, merge_node, \
                            zero_node, n3))
                else:
                    if merge_node not in graph.boundary_nodes and \
                        zero_node not in graph.boundary_nodes and \
                        n3 not in graph.boundary_nodes:
                        graph.merged_triangles.append((merge_i, merge_node, \
                            zero_node, n4))
            if merge_node not in graph.boundary_nodes and \
                zero_node not in graph.boundary_nodes:
                graph.nodes[zero_node]['pos'] = graph.nodes[merge_node]['pos']
            graph.zero_nodes.append(zero_node)
            # check if one of inlet/outlet nodes was merged
            if merge_node in graph.in_nodes and zero_node in graph.in_nodes:
                graph.in_vec[zero_node] = 0
            elif merge_node in graph.out_nodes and zero_node in graph.out_nodes:
                graph.out_vec[zero_node] = 0
            else:
                if zero_node in graph.out_nodes:
                    print (zero_node)
        elif merging_type != 'initial':
            n1, n2 = edges.edge_list[merge_i]
            n3, n4 = edges.edge_list[zero_i]
            if n3 != n1 and n3 != n2:
                if n1 not in graph.boundary_nodes and n2 not in \
                    graph.boundary_nodes and n3 not in graph.boundary_nodes:
                    graph.merged_triangles.append((merge_i, n1, n2, n3))
            else:
                if n1 not in graph.boundary_nodes and n2 not in \
                    graph.boundary_nodes and n4 not in graph.boundary_nodes:
                    graph.merged_triangles.append((merge_i, n1, n2, n4))
        # add edges that should be omitted to merged list
        edges.merged[zero_i] = 1
        for transverse_i in transverse_i_list:
            edges.merged[transverse_i] = 1
    if len(merged) > 0:
        # fix merge matrix
        plot_fix = spr.csr_matrix(spr.diags(np.ones(sid.ne)))
        merge_fix_edges = np.ones(sid.ne)
        merge_fix_diams = np.zeros(sid.ne)
        for i, edge in enumerate(merged):
            merge_fix_edges[edge] = 0
            merge_fix_diams[edge] = merged_diams[i]
            merge_fix_edges[zeroed[i]] = 0
            merge_fix_diams[zeroed[i]] = zeroed_diams[i]
            plot_fix[edge, zeroed[i]] = 1
            plot_fix[zeroed[i], edge] = 1
            for t_edge in transversed[i]:
                merge_fix_edges[t_edge] = 0
        inc.plot = 1 * (plot_fix @ inc.plot @ plot_fix != 0)
        merge_fix = spr.diags(merge_fix_edges) @ (inc.merge > 0) @ \
            spr.diags(merge_fix_diams)
        inc.merge += merge_fix + merge_fix.T
        diag_nodes = spr.csr_matrix(spr.diags(np.ones(sid.nsq)))
        for n1, n2 in merged_nodes:
            diag_nodes[n1, n1] = 0
            diag_nodes[n1, n2] = 1
        merged_edges = np.ones(sid.ne)
        for edge in zeroed:
            merged_edges[edge] = 0
            edges.transversed[edge] = 1
        for transversed_list in transversed:
            for edge in transversed_list:
                merged_edges[edge] = 0
        diag_edges = spr.diags(merged_edges)
        inc.incidence = diag_edges @ (inc.incidence @ diag_nodes)
        sign_fix2 = 2 * inc.incidence @ graph.in_vec + edges.inlet + 2 * \
            inc.incidence @ graph.out_vec + edges.outlet
        sign_fix = 1 * (sign_fix2 == -1) - 1 * (sign_fix2 == 3) + \
            1 * (sign_fix2 == 0)
        diag_edges *= spr.diags(1 * (sign_fix != 0))
        edges.inlet *= (sign_fix != 0)
        edges.outlet *= (sign_fix != 0)
        inc.incidence = spr.diags(sign_fix) @ inc.incidence
        inc.inlet = spr.diags(edges.inlet) @ inc.incidence
        inc.middle = spr.diags(1 - graph.in_vec - graph.out_vec) @ \
            ((inc.incidence.T @ inc.incidence) != 0)
        inc.boundary = spr.diags(graph.in_vec + graph.out_vec)
        inc.merge = diag_edges @ inc.merge @ diag_edges
