import numpy as np
import scipy.sparse as spr

from config import SimInputData
from network import Edges, Triangles
from incidence import Incidence


class Volumes():
    ''' Store and track volumes of ingredients of reactions in triangles of the
    network.

    Attributes
    -------
    triangles : scipy sparse csr matrix
        assignment of edges to neighbouring triangles (each edges usually
        belongs to 2 triangles and each triangle usually consists of 3 edges)

    vol_d_min : numpy ndarray
        minimal volume that emptiness must take in each triangle

    vol_a : numpy ndarray
        volume of ingredient A (dissolved) (ntr)

    vol_e : numpy ndarray
        volume of ingredient E (precipitated) in a triangle

    vol_max : numpy ndarray
        maximum volume of each triangle (vol_max = vol_a + vol_e +
        sum(pi d**2 l / 4))

    vol_a_dissolved : numpy ndarray
        volume of substance A dissolved in current iteration

    vol_e_precipitated : numpy ndarray
        volume of substance E precipitated in current iteration
    '''
    def __init__(self, sid: SimInputData, inc: Incidence, edges: Edges, triangles: Triangles):
        self.triangles: spr.csr_matrix = triangles.incidence
        ("assignment of edges to neighbouring triangles (each edges usually \
         belongs to 2 triangles and each triangle usually consists of 3 edges)")
        self.edge_triangles = np.array(np.sum(self.triangles, axis = 0))[0,:]
        self.vol_d_min = self.triangles.T @ (sid.dmin ** 2 \
            * edges.lens / edges.triangles)
        "minimal volume that emptiness must take in each triangle"
        self.vol_a = (1 - sid.phi) * triangles.volume
        "volume of substance A (dissolved) (ntr)"
        self.vol_a_0 = self.vol_a.copy()
        "initial volume of substance A"
        self.vol_e = np.zeros(sid.ntr)
        "volume of substance E (precipitated) (ntr)"
        self.vol_max = triangles.volume
        "maximum volume of each triangle (ntr)"
        self.vol_a_dissolved = np.zeros(sid.ntr)
        "volume of substance A dissolved in current iteration"
        self.vol_e_precipitated = np.zeros(sid.ntr)
        "volume of substance E precipitated in current iteration"
        self.surface_inc = spr.csr_matrix(0)
        "assignment of edges to neighbouring triangles (updated later)"
        self.vol_a_prev = np.zeros(sid.ntr)
        "volume of substance A from the previous iteration (for merging)"


    def find_edge_surface(self, edges):
        surface = edges.diams
        return surface

    def find_reactive_surace(self, edges):
        surface = self.find_edge_surface(edges)
        return self.triangles.T @ surface / self.edge_triangles
        
        