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
from network import Edges, Graph, Triangles, find_node
from incidence import Incidence
from volumes import Volumes

import draw_net as Dr

def find_closest_node(graph: Graph, node0: int) -> int:
    """ Find node in the graph closest to the given position.

    Parameters
    -------
    graph : Graph class object
        network and all its properties

    pos : tuple
        approximate position of the wanted node

    Returns
    -------
    n_min : int
        index of the node closest to the given position
    """
    x0, y0 = graph.nodes[node0]['pos']
    def r_squared(node):
        x, y = graph.nodes[node]['pos']
        r_sqr = (x - x0) ** 2 + (y - y0) ** 2
        return r_sqr
    r_min = len(graph.nodes())
    n_min = 0
    for node in graph.nodes():
        r = r_squared(node)
        if r < r_min and node != node0:
            r_min = r
            n_min = node
    return n_min


def fix_merging(sid: SimInputData, inc: Incidence, graph: Graph, \
    edges: Edges):
    merge_fix_edge = np.where(((np.array((inc.merge != 0).sum(axis = 0))[0] == 0) * (edges.diams != 0) * (1 - edges.inlet - edges.outlet)) != 0)[0]
    print(merge_fix_edge)
    for edge in merge_fix_edge:
        print(edge)
        print(graph.in_vec[:10])
        try:
            n1, n2 = inc.incidence[edge].nonzero()[1]
        except:
            np.savetxt('inc.txt', inc.incidence.toarray())
            print(edges.diams[edge])
            print(edges.edge_list[edge])

        edges_n1 = inc.incidence.T[n1].nonzero()[1]
        edges_n2 = inc.incidence.T[n2].nonzero()[1]
        r_min = sid.n
        edge_index = -1
        for new_edge in edges_n1:
            if new_edge != edge:
                n3, n4 = inc.incidence[new_edge].nonzero()[1]
                if n3 == n1:
                    n3, n4 = n4, n3
                x1, y1 = graph.nodes[n2]['pos']
                x2, y2 = graph.nodes[n3]['pos']
                r = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                if r < r_min:
                    r_min = r
                    edge_index = new_edge
        # inc.merge[edge, edge_index] = r
        # inc.merge[edge_index, edge] = r
        # r_min = sid.n
        # edge_index = -1
        for new_edge in edges_n2:
            if new_edge != edge:
                n3, n4 = inc.incidence[new_edge].nonzero()[1]
                if n3 == n2:
                    n3, n4 = n4, n3
                x1, y1 = graph.nodes[n1]['pos']
                x2, y2 = graph.nodes[n3]['pos']
                r = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                if r < r_min:
                    r_min = r
                    edge_index = new_edge
        if edge_index != -1:
            inc.merge[edge, edge_index] = r
            inc.merge[edge_index, edge] = r
            print('added possible merging ', edge, edge_index)
        # edges_n2 = inc.incidence.T[n2].nonzero()[1]
        # for new_edge in edges_n1:
        #     print(new_edge)
        #     n3, n4 = inc.incidence[new_edge].nonzero()[1]
        #     if n3 == n1:
        #         n3, n4 = n4, n3
        #     if n3 in edges_n2.flatten():
        #         print('Fixing! ', n3, n2)
        #         x1, y1 = graph.nodes[n2]['pos']
        #         x2, y2 = graph.nodes[n3]['pos']
        #         r = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        #         inc.merge[edge, new_edge] = r
        #         inc.merge[new_edge, edge] = r
        # for new_edge in edges_n2:
        #     print(new_edge)
        #     n3, n4 = inc.incidence[new_edge].nonzero()[1]
        #     if n3 == n2:
        #         n3, n4 = n4, n3
        #     if n3 in edges_n1.flatten():
        #         print('Fixing! ', n3, n1)
        #         x1, y1 = graph.nodes[n1]['pos']
        #         x2, y2 = graph.nodes[n3]['pos']
        #         r = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        #         inc.merge[edge, new_edge] = r
        #         inc.merge[new_edge, edge] = r

def solve_merging(sid: SimInputData, inc: Incidence, graph: Graph, \
    edges: Edges, merging_type: str = 'standard'):
    """ Find edges which should be merged.

    """
    pos = nx.get_node_attributes(graph, 'pos')
    merge_edges = ((inc.merge @ spr.diags(1 - edges.merged) > 0) \
        @ spr.diags(edges.diams) + ((inc.merge.T @ spr.diags(1 - edges.merged) \
            > 0) @ spr.diags(edges.diams)).T) / 2 > inc.merge
    # if np.sum(merge_edges) == inc.prev_merge:
    #     return None
    # else:
    #     inc.prev_merge = np.sum(merge_edges)
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
        try:
            n1, n2 = inc.incidence[merge_i].nonzero()[1]
        except:
            np.savetxt('inc.txt', inc.incidence.toarray())
            print(merge_i, zero_i)
            #print(n1, n2)
            print(inc.incidence[zero_i].nonzero()[1])
            print(edges.edge_list[merge_i])
            print(edges.edge_list[zero_i])
            import draw_net as Dr
            Dr.draw_nodes(sid, graph, edges, np.ones(sid.nsq), \
                'labels.png', 'd')
            raise ValueError('n1')
        # if (n1 in graph.in_nodes and n2 in graph.in_nodes) or \
        #     (n1 in graph.out_nodes and n2 in graph.out_nodes):
        #     continue
        try:
            n3, n4 = inc.incidence[zero_i].nonzero()[1]
        except:
            np.savetxt('inc.txt', inc.incidence.toarray())
            print(merge_i, zero_i)
            print(n1, n2)
            print(inc.incidence[zero_i].nonzero()[1])
            print(edges.edge_list[merge_i])
            print(edges.edge_list[zero_i])
            import draw_net as Dr
            Dr.draw_nodes(sid, graph, edges, np.ones(sid.nsq), \
                'labels.png', 'd')
            raise ValueError('n3')

        # if (n3 in graph.in_nodes and n4 in graph.in_nodes) or \
        #     (n3 in graph.out_nodes and n4 in graph.out_nodes):
        #     continue
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
            inc.merge[merge_i, zero_i] = 0
            inc.merge[zero_i, merge_i] = 0
            continue
        if np.abs(pos[merge_node][0] - pos[zero_node][0]) > 1:
            inc.merge[merge_i, zero_i] = 0
            inc.merge[zero_i, merge_i] = 0
            continue
            #raise ValueError("Wrong merging!")
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
        if merge_node != zero_node:
            if merge_node in graph.in_nodes:
                for edge in inc.incidence.T[zero_node].nonzero()[1]:
                    if edges.inlet[edge]:
                        transverse_i_list.append(edge)
            if merge_node in graph.out_nodes:
                for edge in inc.incidence.T[zero_node].nonzero()[1]:
                    if edges.outlet[edge]:
                        transverse_i_list.append(edge)
            
        merged.append(merge_i)
        zeroed.append(zero_i)
        transversed.append(transverse_i_list)
        # for transverse_i in transverse_i_list:
        #     transversed.append(transverse_i)
        # TO DO: consider how merged diameter should be calculated
        d1, d2, l1, l2 = edges.diams[merge_i], edges.diams[zero_i], \
            edges.lens[merge_i], edges.lens[zero_i]
        if merging_type == 'standard':
            edges.diams[merge_i] = d1 + d2
            edges.diams_initial[merge_i] += edges.diams_initial[zero_i]
            edges.lens[merge_i] = d1 * l1 + d2 * l2
        #edges.diams_initial[zero_i] = 0
        edges.diams[zero_i] = 0
        for transverse_i in transverse_i_list:
            if merging_type == 'standard':
                di, li = edges.diams[transverse_i], edges.lens[transverse_i]
                edges.diams[merge_i] += di
                edges.diams_initial[merge_i] += edges.diams_initial[transverse_i]
                edges.lens[merge_i] += di * li
            #edges.diams_initial[transverse_i] = 0
            edges.diams[transverse_i] = 0
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
        if merge_node in graph.in_nodes:
            for edge in inc.incidence.T[zero_node].nonzero()[1]:
                edges.inlet[edge] = 1
        if merge_node in graph.out_nodes:
            for edge in inc.incidence.T[zero_node].nonzero()[1]:
                edges.outlet[edge] = 1
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
            # if merge_node not in graph.boundary_nodes and \
            #     zero_node not in graph.boundary_nodes:
            #     graph.nodes[zero_node]['pos'] = graph.nodes[merge_node]['pos']
            graph.zero_nodes.append(zero_node)
            # check if one of inlet/outlet nodes was merged
            if merge_node in graph.in_nodes and zero_node in graph.in_nodes:
                graph.in_vec[zero_node] = 0
            elif merge_node in graph.out_nodes and zero_node in graph.out_nodes:
                graph.out_vec[zero_node] = 0
            else:
                if zero_node in graph.out_nodes:
                    print (zero_node)
        else:
            merged_nodes.append((merge_node, merge_node))
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
            if n1 != n2:
                diag_nodes[n1, n1] = 0
                diag_nodes[n1, n2] = 1
                inc.merge_vec[n1] = 1
        merged_edges = np.ones(sid.ne)
        for edge in zeroed:
            merged_edges[edge] = 0
            edges.transversed[edge] = 1
            inc.merge_vec[sid.nsq + edge] = 1
            inc.merge_vec[sid.nsq + sid.ne + edge] = 1
        for transversed_list in transversed:
            for edge in transversed_list:
                merged_edges[edge] = 0
                inc.merge_vec[sid.nsq + edge] = 1
                inc.merge_vec[sid.nsq + sid.ne + edge] = 1
        diag_edges = spr.diags(merged_edges)
        inc.incidence = diag_edges @ (inc.incidence @ diag_nodes)
        #inc.incidence = ((diag_edges @ (inc.incidence @ diag_nodes)).T @ diag_edges).T
        sign_fix2 = 2 * inc.incidence @ graph.in_vec + edges.inlet + 2 * \
            inc.incidence @ graph.out_vec + edges.outlet
        sign_fix = 1 * (sign_fix2 == -1) - 1 * (sign_fix2 == 3) + \
            1 * (sign_fix2 == 0)
        
        inc.merge = diag_edges @ inc.merge @ diag_edges
        diag_edges *= spr.diags(1 * (sign_fix != 0))
        edges.inlet *= (sign_fix != 0)
        edges.outlet *= (sign_fix != 0)
        inc.incidence = spr.diags(sign_fix) @ inc.incidence
        inc.inlet = spr.diags(edges.inlet) @ inc.incidence
        inc.middle = spr.diags(1 - graph.in_vec - graph.out_vec) @ \
            ((inc.incidence.T @ inc.incidence) != 0)
        inc.boundary = spr.diags(graph.in_vec + graph.out_vec)
        diag = inc.merge.diagonal()
        inc.merge -= spr.diags(diag)
        #inc.merge = diag_edges @ inc.merge @ diag_edges
        #fix_merging(sid, inc, graph, edges)
        # np.savetxt('inc.txt', inc.incidence.toarray())
        # np.savetxt('inl.txt', inc.inlet.toarray())

def solve_merging_vols(sid: SimInputData, inc: Incidence, graph: Graph, vols: Volumes, \
    triangles: Triangles, edges: Edges, merging_type: str = 'standard'):
    """ Find edges which should be merged.

    """
    while np.sum(1 * (vols.triangles @ (vols.vol_a == 0))):
    #if True:
        print(np.sum(1 * (vols.triangles @ (vols.vol_a == 0))))
        pos = nx.get_node_attributes(graph, 'pos')
        merge_edges = []
        merge_triangles = []
        merge_matrix = spr.diags(1 * (vols.vol_a == 0)) @ vols.triangles.T
        if merge_matrix.sum():
            rows = list(set(merge_matrix.nonzero()[0]))
            for row in rows:
                elements = merge_matrix[row].nonzero()[1]
                if len(elements) > 1:
                    merge_edges.append(elements)
                    merge_triangles.append(row)
        if len(merge_triangles) == 0:
            break

        merged, zeroed, transversed = [], [], []
        merged_diams, zeroed_diams = [], []
        merged_nodes = []
        merged_triangles = []
        # take coordinates of nonzero matrix elements - these are indices of edges
        # that we want to merge
        for tr_i, edge_list_tri in enumerate(merge_edges):
            transversed_flat = [t_edge for transverse_list in transversed \
                for t_edge in transverse_list]
            skip_list = merged + zeroed + transversed_flat
            flag = 0
            for edge_i in edge_list_tri:
                if edge_i in skip_list:
                    flag = 1
                    print('already merged')
                if edges.diams[edge_i] == 0:
                    print('diam = 0')
                    flag = 1
            if flag:
                continue
            edge_list_sorted = sorted(edge_list_tri, key=lambda i: (edges.inlet[i], edges.outlet[i], edges.diams[i]), reverse=True)
            merge_i = edge_list_sorted[0]
            zero_i = edge_list_sorted[1]
            # if len(edge_list_sorted) > 2:
            #     transverse_i_list = edge_list_sorted[2:]
            # else:
            #     transverse_i_list = []
            
            n1, n2 = inc.incidence[merge_i].nonzero()[1]
            try:
                n3, n4 = inc.incidence[zero_i].nonzero()[1]
            except:
                print(n1, n2)
                print(edge_list_sorted)
                print(merge_triangles[tr_i])
                print(vols.triangles.T[merge_triangles[tr_i]].nonzero())
                print(edges.outlet[merge_i])
                print(edges.outlet[zero_i])
                print(edges.diams[zero_i])
                raise ValueError

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
                for edge in edge_list_tri:
                    vols.triangles[edge, merge_triangles[tr_i]] = 0
                print('something wrong with nodes', n1, n2, n3, n4)
                continue
            if np.abs(pos[merge_node][0] - pos[zero_node][0]) > 1:
                for edge in edge_list_tri:
                    vols.triangles[edge, merge_triangles[tr_i]] = 0
                print('too far away')
                continue
                #raise ValueError("Wrong merging!")
            merge_nodes_flat = np.reshape(merged_nodes, 2 * len(merged_nodes))
            if merge_node in merge_nodes_flat or zero_node in merge_nodes_flat:
                print('node in merged')
                continue

            transverse_i_list = []
            if merge_node != zero_node:
                for i in inc.incidence.T[merge_node].nonzero()[1]:
                    if inc.incidence[i, zero_node] != 0:
                        transverse_i_list.append(i)
            flag = 0
            for transverse_i in transverse_i_list:
                if transverse_i in merged + zeroed + transversed_flat:
                    flag = 1
                    print('transverse in merged')
                if edges.diams[transverse_i] > edges.diams[merge_i]:
                    for edge in edge_list_tri:
                        vols.triangles[edge, merge_triangles[tr_i]] = 0
                    flag = 1
                    print('diam < transverse')
            if flag:
                continue

            print (f"Merging {merge_i} {zero_i} {transverse_i_list} Nodes \
                {merge_node} {zero_node}")
            # if merge_node != zero_node:
            #     if merge_node in graph.in_nodes:
            #         for edge in inc.incidence.T[zero_node].nonzero()[1]:
            #             if edges.inlet[edge] and edge != zero_i:
            #                 transverse_i_list.append(edge)
            #     if merge_node in graph.out_nodes:
            #         for edge in inc.incidence.T[zero_node].nonzero()[1]:
            #             if edges.outlet[edge] and edge != zero_i:
            #                 transverse_i_list.append(edge)
                
            merged.append(merge_i)
            zeroed.append(zero_i)
            transversed.append(transverse_i_list)
            merged_triangles.append(merge_triangles[tr_i])
            # for transverse_i in transverse_i_list:
            #     transversed.append(transverse_i)
            # TO DO: consider how merged diameter should be calculated
            d1, d2, l1, l2 = edges.diams[merge_i], edges.diams[zero_i], \
                edges.lens[merge_i], edges.lens[zero_i]
            if merging_type == 'standard':
                dsum = d1 + d2
                dlsum = d1 * l1 + d2 * l2
                dl2sum = d1 ** 2 * l1 + d2 ** 2 * l2
                #edges.diams[merge_i] = d1 + d2
                edges.diams_initial[merge_i] += edges.diams_initial[zero_i]
                #edges.lens[merge_i] = d1 * l1 + d2 * l2
            #edges.diams_initial[zero_i] = 0
            edges.diams[zero_i] = 0
            for transverse_i in transverse_i_list:
                if merging_type == 'standard':
                    di, li = edges.diams[transverse_i], edges.lens[transverse_i]
                    dsum += di
                    dlsum += di * li
                    dl2sum += di ** 2 * li
                    #edges.diams[merge_i] += di
                    #edges.diams_initial[merge_i] += edges.diams_initial[transverse_i]
                    #edges.lens[merge_i] += di * li
                #edges.diams_initial[transverse_i] = 0
                edges.diams[transverse_i] = 0
                edges.lens[transverse_i] = 1
                if edges.inlet[transverse_i]:
                    edges.inlet[merge_i] = 1
                if edges.outlet[transverse_i]:
                    edges.outlet[merge_i] = 1
                edges.inlet[transverse_i] = 0
                edges.outlet[transverse_i] = 0
                if edges.boundary_list[transverse_i]:
                    edges.boundary_list[merge_i] = 1
            edges.diams[merge_i] = np.sqrt(dl2sum * dsum / dlsum)
            edges.lens[merge_i] = dlsum / dsum
            merged_diams.append((edges.diams[merge_i] - d1) / 2)
            zeroed_diams.append((edges.diams[merge_i] - d2) / 2)
            # TO DO: does it matter how we set lenghts of merged edges?
            #edges.lens[merge_i] /= edges.diams[merge_i]
            # edges.lens[merge_i] = edges.diams[merge_i] ** 4 \
            # / (d1 ** 4 / l1 + d2 ** 4 / l2)
            edges.lens[zero_i] = 1
            if merge_node in graph.in_nodes:
                for edge in inc.incidence.T[zero_node].nonzero()[1]:
                    edges.inlet[edge] = 1
            if merge_node in graph.out_nodes:
                for edge in inc.incidence.T[zero_node].nonzero()[1]:
                    edges.outlet[edge] = 1
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
                # if merge_node not in graph.boundary_nodes and \
                #     zero_node not in graph.boundary_nodes:
                #     graph.nodes[zero_node]['pos'] = graph.nodes[merge_node]['pos']
                graph.zero_nodes.append(zero_node)
                # check if one of inlet/outlet nodes was merged
                if merge_node in graph.in_nodes and zero_node in graph.in_nodes:
                    graph.in_vec[zero_node] = 0
                elif merge_node in graph.out_nodes and zero_node in graph.out_nodes:
                    graph.out_vec[zero_node] = 0
                else:
                    if zero_node in graph.out_nodes:
                        print (zero_node)
            else:
                merged_nodes.append((merge_node, merge_node))
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
                if n1 != n2:
                    diag_nodes[n1, n1] = 0
                    diag_nodes[n1, n2] = 1
                    inc.merge_vec[n1] = 1
            merged_edges = np.ones(sid.ne)
            for edge in zeroed:
                merged_edges[edge] = 0
                edges.transversed[edge] = 1
                inc.merge_vec[sid.nsq + edge] = 1
                inc.merge_vec[sid.nsq + sid.ne + edge] = 1
            for transversed_list in transversed:
                for edge in transversed_list:
                    merged_edges[edge] = 0
                    inc.merge_vec[sid.nsq + edge] = 1
                    inc.merge_vec[sid.nsq + sid.ne + edge] = 1
            diag_edges = spr.diags(merged_edges)
            inc.incidence = diag_edges @ (inc.incidence @ diag_nodes)
            #inc.incidence = ((diag_edges @ (inc.incidence @ diag_nodes)).T @ diag_edges).T
            sign_fix2 = 2 * inc.incidence @ graph.in_vec + edges.inlet + 2 * \
                inc.incidence @ graph.out_vec + edges.outlet
            sign_fix = 1 * (sign_fix2 == -1) - 1 * (sign_fix2 == 3) + \
                1 * (sign_fix2 == 0)
            diag_tr_vec = np.ones(sid.ntr)
            for tr in merged_triangles:
                diag_tr_vec[tr] = 0
            diag_tr = spr.diags(diag_tr_vec)
            inc.merge = diag_edges @ inc.merge @ diag_edges
            #inc.triangles = diag_edges @ inc.triangles @ diag_tr
            diag_edges = spr.csr_matrix(diag_edges)
            for i, edge in enumerate(merged):
                diag_edges[edge, zeroed[i]] = 1
            vols.triangles = (diag_tr @ (diag_edges @ vols.triangles @ diag_tr).T).T
            diag_edges *= spr.diags(1 * (sign_fix != 0))
            edges.inlet *= (sign_fix != 0)
            edges.outlet *= (sign_fix != 0)
            inc.incidence = spr.diags(sign_fix) @ inc.incidence
            inc.inlet = spr.diags(edges.inlet) @ inc.incidence
            inc.middle = spr.diags(1 - graph.in_vec - graph.out_vec) @ \
                ((inc.incidence.T @ inc.incidence) != 0)
            inc.boundary = spr.diags(graph.in_vec + graph.out_vec)
            diag = inc.merge.diagonal()
            inc.merge -= spr.diags(diag)
            fix_lonely_triangles(sid, inc, graph, edges, triangles, vols)
            edges.triangles = np.array(np.sum(vols.triangles, axis = 1))[:, 0]
            
            #inc.merge = diag_edges @ inc.merge @ diag_edges
            #fix_merging(sid, inc, graph, edges)
            # np.savetxt('inc.txt', inc.incidence.toarray())
            # np.savetxt('inl.txt', inc.inlet.toarray())
                    

def fix_connections(sid: SimInputData, inc: Incidence, graph: Graph, \
    edges: Edges):
    lonely_nodes = np.array(((1 * (inc.incidence.T @ inc.incidence != 0)).sum(axis = 0) == 2))[0] * (1 - graph.in_vec - graph.out_vec)
    while(np.sum(lonely_nodes)):
        lonely_edges = inc.incidence @ lonely_nodes
        diag_edges = spr.diags(1 - lonely_edges)
        diag_nodes = spr.diags(1 - lonely_nodes)
        inc.incidence = (diag_nodes @ (diag_edges @ (inc.incidence @ diag_nodes)).T @ diag_edges).T
        edges.inlet *= (1 - lonely_edges)
        edges.outlet *= (1 - lonely_edges)
        inc.merge = diag_edges @ inc.merge @ diag_edges
        inc.merge_vec += np.concatenate([lonely_nodes, lonely_edges, lonely_edges])
        print(np.where(lonely_nodes > 0))
        lonely_nodes = np.array(((1 * (inc.incidence.T @ inc.incidence != 0)).sum(axis = 0) == 2))[0] * (1 - graph.in_vec - graph.out_vec)
        print("Lonely nodes removed")
        print(np.where(lonely_edges > 0))
        #raise ValueError('Alone...')


def find_closest_edge(graph: Graph, inc: Incidence, pos):
    """
    Given:
      - G: a networkx Graph whose nodes have 'pos' attribute = (x, y) coordinates,
      - inc: the incidence matrix of shape (num_nodes, num_edges),
      - pos: a tuple (x, y) for the query position,
    returns:
      - (u, v): the pair of node IDs defining the closest edge,
      - min_dist: the distance from 'pos' to that edge.
    """
    # Helper function: distance from point (px, py) to segment A->B
    def point_segment_distance(px, py, ax, ay, bx, by):
        ABx = bx - ax
        ABy = by - ay
        APx = px - ax
        APy = py - ay

        # Length squared of segment AB
        AB_len_sq = ABx*ABx + ABy*ABy

        # If the two edge endpoints coincide, just return distance to that point
        if AB_len_sq == 0:
            return np.hypot(APx, APy)

        # Projection parameter t of AP onto AB (0 <= t <= 1 for points "within" segment)
        t = (APx*ABx + APy*ABy) / AB_len_sq

        # Closest to A
        if t < 0:
            return np.hypot(APx, APy)
        # Closest to B
        elif t > 1:
            return np.hypot(px - bx, py - by)
        else:
            # Closest to interior point on segment
            cx = ax + t * ABx
            cy = ay + t * ABy
            return np.hypot(px - cx, py - cy)

    px, py = pos
    min_dist = float('inf')
    closest_edge = None

    # Each column of inc corresponds to an edge.
    # For an undirected graph, there should be exactly two non-zero entries (in rows u and v).
    edge_list = list(set(inc.incidence.nonzero()[0]))

    for e in edge_list:
        # Find the nodes that this column (edge) connects
        nodes = inc.incidence[e].nonzero()[1]

        u, v = nodes
        ax, ay = graph.nodes[u]['pos']
        bx, by = graph.nodes[v]['pos']

        dist = point_segment_distance(px, py, ax, ay, bx, by)
        if dist < min_dist:
            min_dist = dist
            closest_edge = (u, v)

    return closest_edge, min_dist

def fix_lonely_triangles(sid: SimInputData, inc: Incidence, graph: Graph, \
    edges: Edges, triangles: Triangles, vols: Volumes):
    """
    During merging on irregular network, sometimes edges can cross and cause
    some cells not to have any neighboring edges, despite having non-zero volume.
    In such a case (i.e. if triangles incidence has no entries for a triangle
    with non-zero volume) we look for an edge closest to the triangle and set
    it as the triangle's neighbor.
    """
    lonely_triangles =  np.where((np.array(np.sum(vols.triangles.T, axis = 1))[:, 0] < 2) * (vols.vol_a > 0))[0]
    pos = np.array(list(nx.get_node_attributes(graph, 'pos').values()))
    for tr in lonely_triangles:
        #node  = find_node(graph, triangles.centers[tr])
        #edge_i = sorted(inc.incidence.T[node].nonzero()[1], key=lambda i: edges.diams[i], reverse=True)[0]
        #edge_i = find_closest_edge(graph, inc, triangles.centers[tr])[0]
        distances = np.linalg.norm(pos - triangles.centers[tr], axis=1)
        flag = 1
        edge_i = 0
        k = 20
        while flag:
            nodes = np.argpartition(distances, k)[:k]
            diams_max = 0
            for node in nodes:
                edge_list = sorted(inc.incidence.T[node].nonzero()[1], key=lambda i: edges.diams[i], reverse=True)
                if edge_list:
                    if edges.diams[edge_list[0]] > diams_max:
                        edge_i = edge_list[0]
                        diams_max = edges.diams[edge_i]
            if edge_i:
                vols.triangles[edge_i, tr] = 1
                flag = 0
            else:
                k *= 2
            
                
        print("Lonely triangles connected")
        #raise ValueError('Alone...')

# def solve_merging_vols(sid: SimInputData, inc: Incidence, graph: Graph, vols: Volumes, \
#     edges: Edges, merging_type: str = 'standard'):
#     """ Find edges which should be merged.

#     """
#     pos = nx.get_node_attributes(graph, 'pos')
#     merged, zeroed, transversed = [], [], []
#     merged_nodes = []

#     vols_merged = 1 * ((1 - (vols.vol_a + vols.vol_e) / vols.vol_max) >= sid.phi_max)
#     vols_merged = np.array(np.ma.fix_invalid(vols_merged, fill_value = 0))
#     #print(vols_merged)
#     #print(vols.vol_a)
#     #print(vols.vol_max)
#     merge_matrix = inc.triangles.T.multiply(vols_merged[:, np.newaxis]).tocsr()
#     #np.savetxt('mm.txt', merge_matrix.toarray())
#     merged_triangles = merge_matrix.sum(axis = 1).nonzero()[0]
#     print(merged_triangles)
#     print(sid.ne, sid.ntr)
#     for i, triangle in enumerate(merged_triangles):
#         print(triangle)
#         print(merge_matrix[triangle].nonzero())
#         edge_indices = merge_matrix[triangle].nonzero()[1]
#         edge_sorted = [edge for _, edge in sorted(zip(edges.diams[edge_indices], edge_indices))]
#         transversed_flat = [t_edge for transverse_list in transversed for t_edge in transverse_list]
#         flag = 0
#         for edge in edge_sorted:
#             if edge in merged + zeroed + transversed_flat:
#                 flag = 1
#         if flag:
#             continue
#         if len(edge_sorted) == 1:
#             continue
#         edge_pair0 = edge_sorted[-1]
#         edge_pair1 = edge_sorted[-2]
#         transverse_i_list = edge_sorted[:-2]
#         diam_total = np.sum(edges.diams[edge_indices])
#         # choose which edge will remain - the one with larger diameter
#         if (edges.inlet[edge_pair1] and edges.inlet[edge_pair0]) or (edges.outlet[edge_pair1] and edges.outlet[edge_pair0]):
#             if edges.diams[edge_pair1] > edges.diams[edge_pair0]:
#                 merge_i = edge_pair1
#                 zero_i = edge_pair0
#             else:
#                 merge_i = edge_pair0
#                 zero_i = edge_pair1
#         elif edges.inlet[edge_pair1] or edges.outlet[edge_pair1]:
#             merge_i = edge_pair1
#             zero_i = edge_pair0
#         elif edges.inlet[edge_pair0] or edges.outlet[edge_pair0]:
#             merge_i = edge_pair0
#             zero_i = edge_pair1
#         elif edges.diams[edge_pair1] > edges.diams[edge_pair0]:
#             merge_i = edge_pair1
#             zero_i = edge_pair0
#         else:
#             merge_i = edge_pair0
#             zero_i = edge_pair1
#         n1, n2 = inc.incidence[merge_i].nonzero()[1]
#         n3, n4 = inc.incidence[zero_i].nonzero()[1]
#         if (n1 in graph.in_nodes and n2 in graph.in_nodes) or (n1 in graph.out_nodes and n2 in graph.out_nodes):
#             continue
#         if (n3 in graph.in_nodes and n4 in graph.in_nodes) or (n3 in graph.out_nodes and n4 in graph.out_nodes):
#             continue
#         if n1 == n3:
#             merge_node = n2
#             zero_node = n4
#         elif n1 == n4:
#             merge_node = n2
#             zero_node = n3
#         elif n2 == n3:
#             merge_node = n1
#             zero_node = n4
#         elif n2 == n4:
#             merge_node = n1
#             zero_node = n3
#         else:
#             raise ValueError("Wrong merging!")
#         # merge_nodes_flat = np.reshape(merged_nodes, 2 * len(merged_nodes))
#         # if merge_node in merge_nodes_flat or zero_node in merge_nodes_flat:
#         #     continue
        
#         print (f"Merging {merge_i} {zero_i} {transverse_i_list} Nodes {merge_node} {zero_node}")
#         merged.append(merge_i)
#         zeroed.append(zero_i)
#         transversed.append(transverse_i_list)
#         vols.vol_a[triangle] = 0


#         # for transverse_i in transverse_i_list:
#         #     transversed.append(transverse_i)
#         # TO DO: consider how merged diameter should be calculated
#         # d1, d2, l1, l2 = edges.diams[merge_i], edges.diams[zero_i], \
#         #     edges.lens[merge_i], edges.lens[zero_i]
#         # edges.diams[merge_i] = d1 + d2
#         # edges.diams_initial[merge_i] += edges.diams_initial[zero_i]
#         # edges.diams_initial[zero_i] = 0
#         # edges.lens[merge_i] = d1 * l1 + d2 * l2
#         # edges.diams[zero_i] = 0
#         # for transverse_i in transverse_i_list:
#         #     di, li = edges.diams[transverse_i], edges.lens[transverse_i]
#         #     edges.diams[merge_i] += di
#         #     edges.diams_initial[merge_i] += edges.diams_initial[transverse_i]
#         #     edges.diams_initial[transverse_i] = 0
#         #     edges.diams[transverse_i] = 0
#         #     edges.lens[merge_i] += di * li
#         #     edges.lens[transverse_i] = 1
#         #     if edges.inlet[transverse_i]:
#         #         edges.inlet[merge_i] = 1
#         #     if edges.outlet[transverse_i]:
#         #         edges.outlet[merge_i] = 1
#         #     edges.inlet[transverse_i] = 0
#         #     edges.outlet[transverse_i] = 0
#         #     if edges.boundary_list[transverse_i]:
#         #         edges.boundary_list[merge_i] = 1
#         # merged_diams.append((edges.diams[merge_i] - d1) / 2)
#         # zeroed_diams.append((edges.diams[merge_i] - d2) / 2)
#         # # TO DO: does it matter how we set lenghts of merged edges?
#         # edges.lens[merge_i] /= edges.diams[merge_i]
#         # edges.lens[merge_i] = edges.diams[merge_i] ** 4 / (d1 ** 4 / l1 + d2 ** 4 / l2)
#         edges.lens[zero_i] = 1
#         edges.inlet[zero_i] = 0
#         edges.outlet[zero_i] = 0
#         if edges.boundary_list[zero_i]:
#             edges.boundary_list[merge_i] = 1
#         if merge_node != zero_node:
#             merged_nodes.append((zero_node, merge_node))
#             if merging_type != 'initial':
#                 if n3 != merge_node and n3 != zero_node:
#                     if merge_node not in graph.boundary_nodes and zero_node not in graph.boundary_nodes and n3 not in graph.boundary_nodes:
#                         graph.merged_triangles.append((merge_i, merge_node, zero_node, n3))
#                 else:
#                     if merge_node not in graph.boundary_nodes and zero_node not in graph.boundary_nodes and n3 not in graph.boundary_nodes:
#                         graph.merged_triangles.append((merge_i, merge_node, zero_node, n4))
#             if merge_node not in graph.boundary_nodes and zero_node not in graph.boundary_nodes:
#                 graph.nodes[zero_node]['pos'] = graph.nodes[merge_node]['pos']
#             #edges.edge_list[zero_i] = edges.edge_list[merge_i]
#             graph.zero_nodes.append(zero_node)
#             # if n3 != merge_node and n3 != zero_node:
#             #     graph.merged_triangles.append((pos[merge_node], pos[zero_node], pos[n3]))
#             # else:
#             #     graph.merged_triangles.append((pos[merge_node], pos[zero_node], pos[n4]))
#             # check if one of inlet/outlet nodes was merged
#             if merge_node in graph.in_nodes and zero_node in graph.in_nodes:
#                 # graph.in_nodes = np.delete(graph.in_nodes, \
#                 #     np.where(graph.in_nodes == zero_node)[0])
#                 graph.in_vec[zero_node] = 0
#             elif merge_node in graph.out_nodes and zero_node in graph.out_nodes:
#                 # graph.out_nodes = np.delete(graph.out_nodes, \
#                 #     np.where(graph.out_nodes == zero_node)[0])
#                 graph.out_vec[zero_node] = 0
#             else:
#                 if zero_node in graph.out_nodes:
#                     print (zero_node)
#         elif merging_type != 'initial':
#             n1, n2 = edges.edge_list[merge_i]
#             n3, n4 = edges.edge_list[zero_i]
#             if n3 != n1 and n3 != n2:
#                 if n1 not in graph.boundary_nodes and n2 not in graph.boundary_nodes and n3 not in graph.boundary_nodes:
#                     graph.merged_triangles.append((merge_i, n1, n2, n3))
#             else:
#                 if n1 not in graph.boundary_nodes and n2 not in graph.boundary_nodes and n4 not in graph.boundary_nodes:
#                     graph.merged_triangles.append((merge_i, n1, n2, n4))
#         # add edges that should be omitted to merged list
#         edges.merged[zero_i] = 1
#         for transverse_i in transverse_i_list:
#             edges.merged[transverse_i] = 1
#     if len(merged) > 0:
#         diag_nodes = spr.csr_matrix(spr.diags(np.ones(sid.nsq)))
#         diag_edges = spr.csr_matrix(spr.diags(np.ones(sid.ne)))
#         diag_tr_vec = np.ones(sid.ntr)
#         for tr in merged_triangles:
#             diag_tr_vec[tr] = 0
#         diag_tr = spr.diags(diag_tr_vec)
#         for n1, n2 in merged_nodes:
#             diag_nodes[n1, n1] = 0
#             diag_nodes[n1, n2] = 1
#         merged_edges = np.ones(sid.ne)
#         for i, edge in enumerate(zeroed):
#             merged_edges[edge] = 0
#             diag_edges[merged[i], edge] = 1
#             diag_edges[edge, edge] = 0
#         for transversed_list in transversed:
#             for edge in transversed_list:     
#                 merged_edges[edge] = 0
#                 diag_edges[edge, edge] = 0
#         diag_edges2 = spr.diags(merged_edges)
#         inc.incidence = diag_edges2 @ (inc.incidence @ diag_nodes)
#         sign_fix2 = 2 * inc.incidence @ graph.in_vec + edges.inlet + 2 * inc.incidence @ graph.out_vec + edges.outlet
#         sign_fix = 1 * (sign_fix2 == -1) - 1 * (sign_fix2 == 3) + 1 * (sign_fix2 == 0)
#         diag_edges *= spr.diags(1 * (sign_fix != 0))
#         edges.inlet *= (sign_fix != 0)
#         edges.outlet *= (sign_fix != 0)
#         inc.incidence = spr.diags(sign_fix) @ inc.incidence
#         inc.inlet = spr.diags(edges.inlet) @ inc.incidence
#         inc.middle = spr.diags(1 - graph.in_vec - graph.out_vec) @ ((inc.incidence.T @ inc.incidence) != 0)
#         inc.boundary = spr.diags(graph.in_vec + graph.out_vec)
#         inc.triangles = diag_edges @ inc.triangles @ diag_tr
