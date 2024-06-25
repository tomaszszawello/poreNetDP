import networkx as nx
import vtk
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import gridspec

from vtk.numpy_interface import dataset_adapter as dsa

from config import SimInputData


def save_VTK(sid, graph, edges, pressure, cb, name): 
    """
    This function is written using VTK module
    Input:
        1. Network graph G
        2. isolated_nodes from "Remove_isolated_nodes" method to avoid the data error by fillinf zeros there
        3. Suffix (string) to save name of files

    """

    np={} 

    pos = nx.get_node_attributes(graph, 'pos')

    for node in pos:
        pos2d = pos[node]
        pos[node] = (pos2d[0], pos2d[1], 0)

    node_pos = pos

    for n in graph.nodes(): 
        try: 
            np[n]=node_pos[n] 
        except KeyError: 
            raise nx.NetworkXError

        # Generate the polyline for the spline. 
    points = vtk.vtkPoints() 
    edgeData = vtk.vtkPolyData() 

        # Edges 

    lines = vtk.vtkCellArray()          
        

    point_data = vtk.vtkDoubleArray()
    point_data.SetNumberOfComponents(2)
    point_data.SetComponentName(0, 'p')
    point_data.SetComponentName(1, 'cb')
    for n in graph.nodes():
        (x,y,z) = node_pos[n]
        points.InsertPoint(n,x,y,z)
        point_data.InsertNextTuple([pressure[n], cb[n]])
    #Filling zeros at deleted nodes
    # try:
    #     for i in self.isolated_nodes:
    #         points.InsertPoint(int(i),0,0,0)
    # except:
    #     pass
    
    cell_data_d = vtk.vtkDoubleArray()
    cell_data_d.SetNumberOfComponents(1)
    cell_data_d.SetName('d')
    
    cell_data_l = vtk.vtkDoubleArray()
    cell_data_l.SetNumberOfComponents(1)
    cell_data_l.SetName('l')

    cell_data_q = vtk.vtkDoubleArray()
    cell_data_q.SetNumberOfComponents(1)
    cell_data_q.SetName('q')

    cell_data_d_init = vtk.vtkDoubleArray()
    cell_data_d_init.SetNumberOfComponents(1)
    cell_data_d_init.SetName('d0')

    tmp_u = []; tmp_v = [];
    for i, e in enumerate(edges.edge_list):
        u=e[0] 
        v=e[1]
        if not edges.boundary_list[i]:
            lines.InsertNextCell(2)  
            lines.InsertCellPoint(u) 
            lines.InsertCellPoint(v)
            cell_data_d.InsertNextTuple([edges.diams[i]])
            cell_data_l.InsertNextTuple([edges.lens[i]])
            cell_data_q.InsertNextTuple([edges.flow[i]])
            cell_data_d_init.InsertNextTuple([edges.diams_initial[i]])

    edgeData.GetCellData().AddArray(cell_data_d)
    edgeData.GetCellData().AddArray(cell_data_l)
    edgeData.GetCellData().AddArray(cell_data_q)
    edgeData.GetCellData().AddArray(cell_data_d_init)

    edgeData.SetPoints(points) 
    edgeData.SetLines(lines)

    writer = vtk.vtkXMLPolyDataWriter();
    writer.SetFileName(sid.dirname + '/' + name);

    writer.SetInputData(edgeData)
    writer.Write()

def load_VTK(file_name):
    
    """
    
        This function reads the Network model output VTK files
        ==========================
        "Without Pore-Merging"
        ==========================
        and returns a Dictionary containing points and scalars in the file
        
        INPUT:
            VTK file saved during simulations
           
        OUTPUT:
            A Dictionary containing cordinates of network points and all scalar fields
               
    """
    #Dictionary where the data of VTK file
    scalars = {}
    
    #reader = vtk.vtkXMLGenericDataObjectReader()
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(file_name)
    #reader.ReadAllScalarsOn()
    reader.Update()

    data = dsa.WrapDataObject(reader.GetOutput())

    points = np.array(data.Points)
    pores = np.array(data.GetLines().GetData())

    #Extracting Cell bounds or pore nodes
    #CellArray = data.GetCells()
    #pores = CellArray.GetData()
    no_of_pores = data.GetNumberOfCells()
    
    pores_array = np.array(pores)
    list_of_pore_connection = []

    for i in range(no_of_pores):
        list_of_pore_connection.append([pores_array[j] for j in range(i*3, i*3+3)])
    scalars['Cell_Nodes'] = list_of_pore_connection;
    
    #No of properties saved in VTK files
    no_of_fields = len(data.CellData.keys())

    scalars['Points'] = points;
    for i in range(no_of_fields): 
        name_of_field = data.CellData.keys()[i]
        scalars[name_of_field] = np.array(data.CellData[name_of_field])
                
    return scalars


def build_VTK(name):
    scalars = load_VTK(name + '.vtk')
    nkw = len(scalars['Points'])
    n = int(nkw ** 0.5)

    in_nodes = list(range(n))
    out_nodes = list(range(nkw - n, nkw))
    G_edges = []
    diams = []
    lens = []
    flow = []
    diams_initial = []
    boundary_edges = []
 
    for i,e in enumerate(scalars['Cell_Nodes']):
        n1 = e[1]
        n2 = e[2]
        d = scalars['d'][i]
        q = scalars['q'][i]
        d_init = scalars['d0'][i]
        l = scalars['l'][i]
        if d != 0:
            if not ((n1 < n and n2 >= nkw - n) or (n2 < n and n1 >= nkw - n)):
                G_edges.append((n1, n2))
                diams.append(d)
                lens.append(l)
                flow.append(q)
                diams_initial.append(d_init)
            if (n1 % n == n-1 and n2 % n == 0) or (n2 % n == n-1 and n1 % n == 0):
                boundary_edges.append((n1, n2))
    
    pos = []
    for node in scalars['Points']:
        pos.append((node[0], node[1]))
    diams, lens, flow, diams_initial = np.array(diams), np.array(lens), np.array(flow), np.array(diams_initial)
    G = nx.Graph()
    G.add_nodes_from(list(range(nkw)))

    G.add_edges_from(G_edges)

    for node in G.nodes:
        G.nodes[node]["pos"] = pos[node]

    for i, edge in enumerate(G.edges()):
        G[edge[0]][edge[1]]['d'] = diams[i]
        G[edge[0]][edge[1]]['l'] = lens[i]

    import matplotlib
    font = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 50}

    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(10, 15))
    #plt.suptitle(f'G = {sid.G:.2f}', fontsize = 1, color='white')
    spec = gridspec.GridSpec(ncols = 1, nrows = 2, height_ratios = [2, 1])
    # draw first panel for the network
    ax1 = plt.subplot(spec[0])
    #ax1.set_title(f'time: {sid.old_t:.2f} dissolved: {data.dissolved_v:.2f}')
    plt.axis('equal')
    ax1.set_axis_on()
    ax1.tick_params(bottom=True)
    plt.xlim(0, n)
    plt.ylim(0, n)
    plt.xlabel('x', fontsize = 60)
    #plt.ylabel('flow focusing index', fontsize = 40)
    plt.yticks([], [])
    #ax1.yaxis.label.set_color('white')
    #ax1.tick_params(axis = 'y', colors='white')
    ax1.xaxis.label.set_color('white')
    ax1.tick_params(axis = 'x', colors='white')
    # draw inlet and outlet nodes
    x_in, y_in = [], []
    for node in in_nodes:
        x_in.append(pos[node][0])
        y_in.append(pos[node][1])
    x_out, y_out = [], []
    for node in out_nodes:
        x_out.append(pos[node][0])
        y_out.append(pos[node][1])
    
    plt.scatter(x_in, y_in, s = 1000 / n, facecolors = 'white', edgecolors = 'black')
    plt.scatter(x_out, y_out, s = 1000 / n, facecolors = 'black', \
        edgecolors = 'white')
    plot_type = 'q'
    if plot_type == 'q':
        qs = np.abs(flow)
    else:
        qs = diams - diams_initial
    nx.draw_networkx_edges(G, pos, G_edges, edge_color = 'k', \
        width = 0.3 * np.array(qs), hide_ticks = False)
    # nx.draw_networkx_edge_labels(graph, pos, edge_labels=dict(zip(edges.edge_list, np.arange(0, len(edges.edge_list)))), font_size = 5)
    ax2 = plt.subplot(spec[1], sharex = ax1)
    pos_x = np.array(list(nx.get_node_attributes(G, 'pos').values()))[:,0]
    slice_x = np.linspace(np.min(pos_x), np.max(pos_x), 102)[1:-1]
    slices = np.loadtxt('fig4/G5.00Daeff0.05/18/slices.txt')
    #slices = np.loadtxt(name+'.txt')
    edge_number  = np.array(slices[0])
    #i_division = sid.dissolved_v_max // sid.track_every // 3
    plt.plot(slice_x, np.array((edge_number - 2 * np.array(slices[1])) / edge_number), linewidth = 5, color = 'black', label = '0.0')
    for i, channeling in enumerate(slices[2:]):
        #if i % i_division == 0 and i > 0:
        plt.plot(slice_x, (edge_number - 2 * np.array(channeling)) / edge_number, linewidth = 5)    
    plt.ylim(0, 1.05)
    plt.xlabel('x', fontsize = 60, style = 'italic')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(tight = True)
    plt.ylabel('flow focusing index', fontsize = 50)
    plt.yticks([0, 0.5, 1],['0', '0.5', '1'])
    # save file in the directory
    plt.savefig('vtk.png', bbox_inches="tight")
    plt.close()    


#    return G, in_nodes, out_nodes, boundary_edges
    