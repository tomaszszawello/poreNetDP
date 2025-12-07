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
        if sid.noise == 'file_lognormal_k':
            raw = np.loadtxt(sid.noise_filename).T
            z = (raw - raw.mean()) / raw.std()
            z = np.clip(z, -1, 1)  # avoid extreme tails
            phi = []
            for n1, n2, n3 in triangles.tlist:
                phi.append((z[n1 // sid.n, n1 % sid.n] + \
                    z[n2 // sid.n, n2 % sid.n] + z[n3 // sid.n, n3 % sid.n]) / 3)
            phi = np.array(phi)
            phi_min = 0.01
            phi_max = 0.99
            phi_field = phi_min + (phi_max - phi_min) * (phi - phi.min()) / (phi.max() - phi.min())
            self.vol_a = (1 - phi_field) * triangles.volume
        elif sid.noise == 'lognormal':
            normal = np.random.randn(len(triangles.volume))
            phi_var = np.exp(1+ sid.sigma_phi * normal)
            phi_var /= np.average(phi_var)            
            self.vol_a = (1 - np.clip(sid.phi * phi_var, 0.01, 0.9)) * triangles.volume
        else:
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
        
        