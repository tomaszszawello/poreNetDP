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

def draw(network_name: str, profile_name: str, plot_type: str, \
    scaling: float) -> None:
    """
    Build network from VTK file and draw it.
    """
    scalars = load_VTK(network_name + '.vtk')
    nkw = len(scalars['Points'])
    n = int(nkw ** 0.5)
    in_nodes = list(range(n))
    out_nodes = list(range(nkw - n, nkw))
    G_edges, diams, flow, diams_initial = [], [], [], [] 
    # load edge data into network
    for i,e in enumerate(scalars['Cell_Nodes']):
        n1 = e[1]
        n2 = e[2]
        d = scalars['d'][i]
        q = scalars['q'][i]
        d_init = scalars['d0'][i]
        if d != 0:
            # do not include boundary edges
            if not ((n1 < n and n2 >= nkw - n) or (n2 < n and n1 >= nkw - n)):
                G_edges.append((n1, n2))
                diams.append(d)
                flow.append(q)
                diams_initial.append(d_init)
    pos = []
    for node in scalars['Points']:
        pos.append((node[0], node[1]))
    diams, flow, diams_initial = np.array(diams), np.array(flow), np.array(diams_initial)
    G = nx.Graph()
    G.add_nodes_from(list(range(nkw)))
    G.add_edges_from(G_edges)
    for node in G.nodes:
        G.nodes[node]["pos"] = pos[node]
    for i, edge in enumerate(G.edges()):
        G[edge[0]][edge[1]]['d'] = diams[i]
    
    # draw network and flow focusing profiles
    font = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 50}
    matplotlib.rc('font', **font)
    plt.figure(figsize=(10, 15))
    spec = gridspec.GridSpec(ncols = 1, nrows = 2, height_ratios = [2, 1])
    # draw first panel for the network
    ax1 = plt.subplot(spec[0])
    plt.axis('equal')
    plt.xlim(0, n)
    plt.ylim(0, n)
    # draw inlet and outlet nodes
    x_in, y_in = [], []
    for node in in_nodes:
        x_in.append(pos[node][0])
        y_in.append(pos[node][1])
    x_out, y_out = [], []
    for node in out_nodes:
        x_out.append(pos[node][0])
        y_out.append(pos[node][1])    
    plt.scatter(x_in, y_in, s = 1000 / n, facecolors = 'white', \
        edgecolors = 'black')
    plt.scatter(x_out, y_out, s = 1000 / n, facecolors = 'black', \
        edgecolors = 'white')
    if plot_type == 'q':
        nx.draw_networkx_edges(G, pos, G_edges, edge_color = 'k', \
            width = 0.3 * scaling * np.abs(flow))
    else:
        nx.draw_networkx_edges(G, pos, G_edges, edge_color = 'k', \
            width = 0.1 * scaling * (diams - diams_initial))
    # draw second panel for the profiles
    ax2 = plt.subplot(spec[1], sharex = ax1)
    pos_x = np.array(list(nx.get_node_attributes(G, 'pos').values()))[:,0]
    slice_x = np.linspace(np.min(pos_x), np.max(pos_x), 102)[1:-1]
    slices = np.loadtxt(profile_name + '.txt')
    edge_number  = np.array(slices[0])
    plt.plot(slice_x, np.array((edge_number - 2 * np.array(slices[1])) \
        / edge_number), linewidth = 5, color = 'black', label = '0.0')
    for i, channeling in enumerate(slices[2:]):
        plt.plot(slice_x, (edge_number - 2 * np.array(channeling)) \
            / edge_number, linewidth = 5)    
    plt.ylim(0, 1.05)
    plt.xlabel('x', fontsize = 60, style = 'italic')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(tight = True)
    plt.ylabel('flow focusing index', fontsize = 50)
    plt.yticks([0, 0.5, 1],['0', '0.5', '1'])
    # save file in the directory
    plt.savefig(network_name + '.png', bbox_inches="tight")
    plt.close()    

#draw(network_name, profile_name, plot_type, scaling)
#print(f'Figure plotted and saved to "{network_name}.png"')
