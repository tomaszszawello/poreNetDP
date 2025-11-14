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
        normal = np.random.randn(len(triangles.volume))
        phi_var = np.exp(1+ sid.sigma_phi * normal)
        phi_var /= np.average(phi_var)
        
        self.vol = (1 - np.clip(sid.phi * phi_var, 0.01, 0.9)) * triangles.volume
        #self.vol = (1 - sid.phi) * triangles.volume
        self.initialize_rocks_from_file(sid, triangles)
        "total volume of rock in the triangle"
        #self.vol_a = self.vol.copy()
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
        self.tri_contact = self.triangles.copy()
        #self.initialize_rocks(sid)
        

    def initialize_rocks(self, sid):
        inert = np.random.randint(0, sid.ntr, size=int(sid.inert_fraction * sid.ntr))
        self.vol_a[inert] = 0
        self.vol_e[inert] = self.vol[inert]

    def initialize_rocks_from_file(self, sid, triangles):
        mineral = np.loadtxt(sid.rock_filename)
        threshold = np.quantile(mineral, 1 - sid.inert_fraction)
        mineral_array = 1 * (mineral > threshold)
        mineral_fraction = []
        for n1, n2, n3 in triangles.tlist:
            mineral_fraction.append((mineral_array[n1 // sid.n, n1 % sid.n] + \
                mineral_array[n2 // sid.n, n2 % sid.n] + mineral_array[n3 // sid.n, n3 % sid.n]) / 3)
        mineral_fraction = np.array(mineral_fraction)
        self.vol_a = self.vol * (1 - mineral_fraction)
        self.vol_e = self.vol * mineral_fraction


    def find_edge_surface(self, edges):
        surface = edges.diams
        return surface

    def find_reactive_surace(self, edges):
        surface = self.find_edge_surface(edges)
        return self.triangles.T @ surface / self.edge_triangles
        
        