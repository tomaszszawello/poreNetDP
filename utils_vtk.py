""" Save network to VTK files.

This module contains functions saving the network data (the whole network with
node and edge parameters and the boundary nodes positions) to VTK files to
allow visualization in ParaView.
"""

import networkx as nx
import vtk
import numpy as np

from config import SimInputData
from incidence import Edges
from network import Graph


def save_vtk_nodes(sid: SimInputData, graph: Graph) -> None:
    """ Saves positions of inlet and outlet nodes to VTK file.
    
    Parameters
    -------
    sid : SimInputData class object
        all config parameters of the simulation
    
    graph : Graph class object
        network and all its properties
    """
    pos = list(zip(nx.get_node_attributes(graph, 'x').values(), \
        nx.get_node_attributes(graph, 'y').values(), \
        nx.get_node_attributes(graph, 'z').values()))
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(sid.dirname + '/boundary_nodes.vtk')
    data = vtk.vtkAppendPolyData()
    for node in graph.in_nodes:
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(pos[node][0], pos[node][1], pos[node][2])
        sphere.SetRadius(1)
        sphere.Update()
        data.AddInputData(sphere.GetOutput())
    for node in graph.out_nodes:
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(pos[node][0], pos[node][1], pos[node][2])
        sphere.SetRadius(1)
        sphere.Update()
        data.AddInputData(sphere.GetOutput())
    data.Update()
    writer.SetInputData(data.GetOutput())
    writer.Update()

def save_vtk(sid: SimInputData, graph: Graph, edges: Edges, \
    pressure: np.ndarray, concentration: np.ndarray) -> None:
    """ Saves network to VTK file.

    This function saves the whole network to VTK file with the following data:
    in nodes pressure and concentration of B, in edges aperture, length,
    fracture length, flow, inlet concentration of solvent, outlet concentration
    of solvent.

    Parameters
    -------
    sid : simInputData class object
        all config parameters of the simulation

    graph : Graph class object
        network and all its properties

    edges : Edges class object
        all edges in network and their parameters
    
    pressure : numpy ndarray
        vector of pressure in nodes

    cb : numpy ndarray
        vector of solvent concentration in nodes
    """
    pos = list(zip(nx.get_node_attributes(graph, 'x').values(), \
        nx.get_node_attributes(graph, 'y').values(), \
        nx.get_node_attributes(graph, 'z').values()))
    points = vtk.vtkPoints() 
    edgeData = vtk.vtkPolyData() 
    lines = vtk.vtkCellArray()          
    # save data in nodes
    point_data = vtk.vtkDoubleArray()
    point_data.SetNumberOfComponents(2)
    point_data.SetComponentName(0, 'p')
    point_data.SetComponentName(1, 'cb')
    for n in graph.nodes():
        (x,y,z) = pos[n]
        points.InsertPoint(n, x, y, z)
        point_data.InsertNextTuple([pressure[n], concentration[n]])
    # build arrays for data in edges
    # aperture
    cell_data_b = vtk.vtkDoubleArray()
    cell_data_b.SetNumberOfComponents(1)
    cell_data_b.SetName('b')
    # length
    cell_data_l = vtk.vtkDoubleArray()
    cell_data_l.SetNumberOfComponents(1)
    cell_data_l.SetName('l')
    # fracture length
    cell_data_fl = vtk.vtkDoubleArray()
    cell_data_fl.SetNumberOfComponents(1)
    cell_data_fl.SetName('fl')
    # flow
    cell_data_q = vtk.vtkDoubleArray()
    cell_data_q.SetNumberOfComponents(1)
    cell_data_q.SetName('q')
    # inlet concentration of solvent
    cell_data_concentration_in = vtk.vtkDoubleArray()
    cell_data_concentration_in.SetNumberOfComponents(1)
    cell_data_concentration_in.SetName('c_in')
    # outlet concentration of solvent
    cell_data_concentration_out = vtk.vtkDoubleArray()
    cell_data_concentration_out.SetNumberOfComponents(1)
    cell_data_concentration_out.SetName('c_out')
    # save data in edges

    aperture_avr = np.average(edges.apertures)

    for i, e in enumerate(graph.edges()):
        u = e[0] 
        v = e[1]
        lines.InsertNextCell(2)
        lines.InsertCellPoint(u)
        lines.InsertCellPoint(v)
        #cell_data_b.InsertNextTuple([edges.apertures[i] * sid.b0])
        cell_data_b.InsertNextTuple([edges.apertures[i] / aperture_avr])
        cell_data_l.InsertNextTuple([edges.lens[i] * sid.l0])
        cell_data_fl.InsertNextTuple([edges.fracture_lens[i] * sid.l0])
        cell_data_q.InsertNextTuple([edges.flow[i]])
        cell_data_concentration_in.InsertNextTuple([concentration[u]])
        cell_data_concentration_out.InsertNextTuple([concentration[v]])
    edgeData.GetCellData().AddArray(cell_data_b)
    edgeData.GetCellData().AddArray(cell_data_l)
    edgeData.GetCellData().AddArray(cell_data_fl)
    edgeData.GetCellData().AddArray(cell_data_q)
    edgeData.GetCellData().AddArray(cell_data_concentration_in)
    edgeData.GetCellData().AddArray(cell_data_concentration_out)
    edgeData.SetPoints(points) 
    edgeData.SetLines(lines)
    # save VTK data to file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(sid.dirname + f'/network_{sid.old_t:04f}.vtk')
    writer.SetInputData(edgeData)
    writer.Write()
