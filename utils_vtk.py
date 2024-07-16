""" Save network data to VTK files. Docs missing.
"""

import networkx as nx
import numpy as np
import vtk

from config import SimInputData
from network import Graph, Edges

def save_VTK(sid: SimInputData, graph: Graph, edges: Edges, \
    pressure: np.ndarray, cb: np.ndarray, name: str) -> None:
    """ Save data from the simulation into VTK file readable with paraView.
    
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
    points = vtk.vtkPoints()
    edgeData = vtk.vtkPolyData()
    lines = vtk.vtkCellArray()
    point_data = vtk.vtkDoubleArray()
    point_data.SetNumberOfComponents(2)
    point_data.SetComponentName(0, 'p')
    point_data.SetComponentName(1, 'cb')
    for n in graph.nodes():
        (x,y,z) = node_pos[n]
        points.InsertPoint(n,x,y,z)
        point_data.InsertNextTuple([pressure[n], cb[n]])
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
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(sid.dirname + '/' + name)
    writer.SetInputData(edgeData)
    writer.Write()
