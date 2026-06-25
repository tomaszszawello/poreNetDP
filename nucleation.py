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
    velocity = cC_profiles / (1 + sid.G * sid.K * edges.diams[:,np.newaxis])
    avg_velocity = H_growth * np.trapezoid(velocity, x=xi_grid, axis=1)
    return avg_nucleation_rate, avg_velocity 

# ============================================================================================
# ================== VARIOUS IMPLEMENTATIONS AND GENERALISATIONS OF AVRAMI ===================
# ============================================================================================
def update_frac_transformed_explicit(sid, edges, cC_profiles, \
        avg_nucleation_rate, avg_velocity, dt):
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

def update_frac_transformed_isotropic(sid, edges, cC_profiles, old_diams, \
        avg_nucleation_rate, avg_velocity, dt):
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

def update_frac_transformed_fixed_grain(sid, edges, cC_profiles, old_diams, avg_nucleation_rate, avg_velocity, dt):
    """ Extends KJMA model to account for nucleation and growth on a deformable substrate:
        Assumes that the change in area of a pore adds or subtracts from the boundary
        of the KJMA domain. When dA/dt < 0, df/dt = [(1-f)/A]*dS_ext/dt, which is the
        classical KJMA result. When dA/dt > 0, df/dt = [(1-f)/A]*dS_ext/dt - [f/A_p] dA_p/dt
        where the subtrahend is a dilution term due to the creation of new area
        Small range of validitiy overall since homogeneity condition is ruined as pore shrinks

        Parameters
        ----------
        old_diams : np.ndarray
            edge diameters at the previous time step

        Returns
        --------
        f : np.ndarray 
            fraction of transformed area in each edge
    """

    # Epsilon calculation
    A_p_prev = 2 * np.pi * old_diams * edges.lens 
    A_p_curr = 2 * np.pi * edges.diams * edges.lens
    A_ratio = A_p_curr / A_p_prev 
    epsilon = np.log(A_ratio) / dt
    eps_pos = np.maximum(0, epsilon) 
    dilution_factor = 1.0 + eps_pos * dt

    # Implicit state spdate
    edges.N_tot = (edges.N_tot + avg_nucleation_rate * dt) / dilution_factor
    edges.P_ext = (edges.P_ext + avg_velocity * edges.N_tot * dt) / dilution_factor
    edges.A_ext = (edges.A_ext + 2 * np.pi * avg_velocity * edges.P_ext * dt) / dilution_factor
    
    K = 2 * np.pi * avg_velocity * edges.P_ext
    #N = edges.N_tot * A_p_curr
    #print(f"    Nuclei number stats: {sid.A:3f}, {np.min(N):.2f}, {np.max(N):.2f}, {np.mean(N):.2f}, {N}") 
    edges.f = (edges.f + K * dt) / (1.0 + (K + eps_pos) * dt)

def update_frac_transformed_global_ellipse(sid, edges, cC_profiles, dr, \
        old_diams, avg_nucleation_rate, avg_velocity, dt):
    """ Same as the isotropic case  / stretchy grains except we allow grains to 
        distort into ellipses, under the assumption that edges only change in radius
        and not length
        Overall is probably more honest to the geometric picture but makes little difference
        when compared with the plain isotropic case (circular disks)
    """
    # Dilution
    A_pore_old = 2 * np.pi * old_diams * edges.lens  # Old area
    A_pore = 2 * np.pi * edges.diams * edges.lens    # Curr. area
    scale_factor = A_pore_old / A_pore
    print(f"SCALE FACTOR = {np.min(scale_factor)}, {np.max(scale_factor)}, {np.mean(scale_factor)}") 
    if not hasattr(edges, 'N_rho'):
        edges.N_rho = np.zeros_like(diams) 
        edges.rho_v = np.zeros_like(diams)
        edges.R_h = np.zeros_like(diams)
    edges.N_rho = (edges.N_rho + avg_nucleation_rate * dt) * scale_factor
    growth_v = (avg_velocity / A_pore_old) * edges.N_rho * dt
    edges.rho_v = (edges.rho_v + growth_v) * scale_factor
    growth_h = avg_velocity * (edges.N_rho * A_pore_old) * dt
    edges.R_h = edges.R_h + growth_h
    term_v = np.pi * avg_velocity * A_pore_old * edges.rho_v
    term_h = np.pi * (avg_velocity / A_pore_old) * edges.R_h
    edges.A_ext = edges.A_ext + (term_v + term_h) * dt
    #N = edges.N_rho * 2. * np.pi * edges.diams * edges.lens
    #print(f"    Nuclei number stats: {sid.A}, {np.min(N):.2f}, {np.max(N):.2f}, {np.mean(N):.2f}, {N}, {avg_velocity}") 

    edges.f = 1 - np.exp(-edges.A_ext)
