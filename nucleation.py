""" Functions used for kinetic model of nucleation and growth in the network

    Assumes mass balance of the form q cB ~ (1-f) cB and q cC ~ (1 - f) cB - f cC 
    where 0 < f < 1 is a scalar function of time

    
"""
import numpy as np
import scipy.sparse as spr

def reconstruct_cB_profiles(sid, edges, inc, cb, n_pts=20):
    """
    Reconstructs analytical cB profile in each edge. 
    
    Returns:
    --------
    profiles : np.ndarray 
    """
    
    # upstream nodes to edges
    #cb_inc = 1 * (inc.incidence.T @ (spr.diags(edges.flow) @ inc.incidence > 0) != 0)
    inlet_matrix = 1 * ((spr.diags(edges.flow) @ inc.incidence) > 0) 
    cB0 = (inlet_matrix @ cb)[:, np.newaxis]
    q_eps = np.abs(edges.flow) + 1e-25 # Everything breaks without this
    E1 = (1 - edges.f) * sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / q_eps
    E1 = E1[:, np.newaxis]
    # Reconstruct... 
    x_hat = np.linspace(0, 1, n_pts)[np.newaxis, :] 
    return cB0 * np.exp(-E1 * x_hat)

def reconstruct_cC_profiles(sid, edges, inc, cb, cc, n_pts=20):
    """ Reconstructs analytical cC profile in each edge:
    c_C(x) = c_B0 / (E_2 / E_1 - 1) * (exp(-E_1 x) -  exp(-E_2 x)) + c_C0 * exp(-E2 x)

    Parameters
    ----------
    n_pts : number of points to reconstruct 
    ... 
    Returns
    --------
    profiles : np.ndarray 
    """
    # upstream nodes to edges map
    inlet_matrix = 1 * ((spr.diags(edges.flow) @ inc.incidence) > 0)
    
    # Concentrations at the inlets of all edges (N_edges x 1)
    cB0 = (inlet_matrix @ cb)[:, np.newaxis]
    cC0 = (inlet_matrix @ cc)[:, np.newaxis]
    
    q_eps = np.abs(edges.flow) + 1e-25 # Everything breaks without this
    E1 = (1 - edges.ftrans) * sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / q_eps
    E1 = E1[:, np.newaxis]
    E2 = edges.ftrans * (sid.Da * sid.K) / (1 + sid.G * sid.K * edges.diams) * edges.diams * edges.lens / q_eps
    E2 = E2[:, np.newaxis]

    x_hat = np.linspace(0., 1., n_pts)[np.newaxis, :] 

    # Decay of entering C -> c_C0 * exp(-E_2 * x)
    decay_profile = cC0 * np.exp(-E2 * x_hat)

    # Production of C from B -> c_B0 * [E_1 / (E_2 - E_1)] * (exp(-E_1 * x) - exp(-E_2 * x))
    prod_profile = (E1 * (np.exp(-E1 * x_hat) - np.exp(-E2 * x_hat))) / (E2 - E1)
    return decay_profile + cB0 * prod_profile

def get_average_rates(sid, edges, cC_profiles):
    """Gets averaged C concentration, nucleation rate and ('crystal') growth velocity 

    Returns
    --------
    avg_nucleation_rate : np.ndarray 
        average nucleation rate in each edge

    avg_velocity : np.ndarray 
        average growth velocity in each edge
    """
    H_growth = sid.K / sid.Gamma  
    S = np.maximum(cC_profiles, 0) / sid.c_sat
    J_nucl = np.zeros_like(S)
    mask_super = S > (1.0 + 1e-9)
    if np.any(mask_super):
        log_S = np.log(S[mask_super])
        J_nucl[mask_super] = sid.A * np.exp(-1. / (log_S**2))
    xi_grid = np.linspace(0, 1, cC_profiles.shape[1])
    avg_nucleation_rate = np.trapezoid(J_nucl, x=xi_grid, axis=1)

    driving_force = cC_profiles / (1 + sid.G * sid.K * edges.diams[:,np.newaxis])
    avg_driving_force = np.trapezoid(driving_force, x=xi_grid, axis=1)
    avg_velocity = H_growth * avg_driving_force 

    return avg_nucleation_rate, avg_velocity 

# ============================================================================================
# ================== VARIOUS IMPLEMENTATIONS AND GENERALISATIONS OF AVRAMI ===================
# ============================================================================================
def update_frac_transformed_explicit(sid, edges, cC_profiles, avg_nucleation_rate, avg_velocity, dt):
    """ Direct application of the KJMA kinetic model to each edge of the network

    Returns
    --------
    f : np.ndarray 
        fraction of transformed area in each edge

    """
    edges.N_tot += avg_nucleation_rate * dt
    edges.P_ext += (edges.N_tot * avg_velocity) * dt
    d_A_ext = (2 * np.pi * edges.P_ext * avg_velocity) * dt
    edges.A_ext += d_A_ext
    edges.ftrans = 1.0 - np.exp(-edges.A_ext)

def update_frac_transformed_isotropic(sid, edges, cC_profiles, old_diams, avg_nucleation_rate, avg_velocity, dt):
    """ Extends KJMA model to account for nucleation and growth on a deformable substrate:
        Assumes a pore edge distorts only in one dimension and that grains stretch with the
        substrate itself

        Parameters
        ----------
        old_diams : np.ndarray
            edge diameters at the previous time step

        Returns
        --------
        f : np.ndarray 
            fraction of transformed area in each edge
    """
    # Dilution factor
    A_pore_old = 2 * np.pi * old_diams * edges.lens  # Old area
    A_pore = 2 * np.pi * edges.diams * edges.lens    # Curr. area
    scale_factor = A_pore_old / A_pore

    # Integrating factor 
    edges.N_tot = (edges.N_tot + avg_nucleation_rate * dt) * scale_factor
    edges.P_ext = (edges.P_ext + 2*np.pi*avg_velocity*edges.N_tot*dt) * np.sqrt(scale_factor)
    edges.A_ext = (edges.A_ext + avg_velocity*edges.P_ext*dt)

    N = edges.N_tot * 2. * np.pi * edges.diams * edges.lens
    print(f"    Nuclei number stats: {sid.A}, {np.min(N):.2f}, {np.max(N):.2f}, {np.mean(N):.2f}, {N}, {avg_velocity}") 

    return 1 - np.exp(-edges.A_ext)
