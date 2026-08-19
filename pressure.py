""" Calculate pressure and flow in the system.

This module contains functions for solving the flow and continuity equations
for pressure and flow. It assumes constant inflow (or constant pressure drop)
boundary condition. It constructs a result vector for the matrix equation
(constant throughout the simulation) and the matrix with coefficients
corresponding to aforementioned equations. Function solve_equation from module
utils is used to solve the equations for flow.

Two constitutive relations for the edge conductivity are available (chosen with
include_dw in config):

- Hagen-Poiseuille (include_dw = False), where the conductivity of an edge is
  simply d ** 4 / l and the resulting problem is linear;
- Darcy-Weisbach (include_dw = True), where the conductivity is corrected by
  the factor 64 / (f * Re), with the Darcy friction factor f taken from the
  Churchill correlation (valid in the laminar, transitional and turbulent
  regime). Since f depends on the flow, the problem becomes nonlinear and is
  solved iteratively (Picard iteration on the conductivity, with an exact
  per-edge root find for the Reynolds number in each iteration).

Nondimensionalization
-------
All quantities are dimensionless, with diameters scaled by the characteristic
diameter d0 and flows by the characteristic flow q0. Hence the Reynolds number
of an edge is Re = Re0 * q / d, where Re0 = 4 * rho * q0 / (pi * d0 * mu) is
the Reynolds number of an edge with the characteristic diameter carrying the
characteristic flow (set in config). Re0 = 0 recovers the Hagen-Poiseuille
limit. The wall roughness (config roughness) is likewise given relative to d0.

Notable functions
-------
solve_flow(SimInputData, Incidence, Graph, Edges, spr.csc_matrix) \
    -> np.ndarray
    calculate pressure and update flow in network edges

conductivity(SimInputData, Edges, np.ndarray) -> np.ndarray
    conductivity of the edges for a given Reynolds number

friction_factor(np.ndarray, np.ndarray) -> np.ndarray
    Darcy friction factor from the Churchill correlation
"""

import numpy as np
import scipy.sparse as spr

from config import SimInputData
from network import Edges, Graph
from incidence import Incidence
from utils import keep_largest_component, solve_equation


RE_MIN: float = 1e-6
"Reynolds number below which the flow is treated as purely laminar"
RE_MAX: float = 1e18
"Reynolds number above which we cut off the friction factor (to avoid overflow)"
X_TOL: float = 1e-13
"relative tolerance of the per-edge root find for the Reynolds number"


def create_vector(graph: Graph) -> spr.csc_matrix:
    """ Creates vector result for pressure calculation.

    For inlet and outlet nodes elements of the vector correspond explicitly
    to the pressure in nodes, for regular nodes elements of the vector equal
    0 correspond to flow continuity.

    Parameters
    -------
    graph : Graph class object
        network and all its properties
        in_nodes - inlet nodes

    Returns
    -------
    scipy sparse vector
        result vector for pressure calculation
    """
    return graph.in_vec

def friction_re(re: np.ndarray, rel_rough: np.ndarray) -> np.ndarray:
    """ Calculates the product of the friction factor and Reynolds number.

    Uses the Churchill correlation, which reproduces the laminar (f = 64 / Re),
    transitional and turbulent (Colebrook) regimes with a single formula. The
    product f * Re is calculated instead of f itself, since it stays finite
    (equal 64) in the laminar limit and thus can be evaluated for arbitrarily
    small Re without overflow.

    Parameters
    -------
    re : numpy ndarray
        Reynolds number of the edges

    rel_rough : numpy ndarray
        wall roughness of the edges relative to their diameter

    Returns
    -------
    numpy ndarray
        product of the Darcy friction factor and the Reynolds number
    """
    # for very small Re the correlation is indistinguishable from the laminar
    # result, so we cut it off (as well as for unphysically large Re) to avoid
    # overflow in the powers below
    re_cut = np.clip(np.abs(re), RE_MIN, RE_MAX)
    c = (7 / re_cut) ** 0.9 + 0.27 * rel_rough
    a = (-2.457 * np.log(c)) ** 16
    b = (37530 / re_cut) ** 16
    # f * Re = 8 * (8 ** 12 + Re ** 12 / (A + B) ** 1.5) ** (1 / 12)
    return 8 * (8 ** 12 + re_cut ** 12 / (a + b) ** 1.5) ** (1 / 12)

def friction_factor(re: np.ndarray, rel_rough: np.ndarray) -> np.ndarray:
    """ Calculates the Darcy friction factor (Churchill correlation).

    Parameters
    -------
    re : numpy ndarray
        Reynolds number of the edges

    rel_rough : numpy ndarray
        wall roughness of the edges relative to their diameter

    Returns
    -------
    numpy ndarray
        Darcy friction factor of the edges (infinite for zero flow)
    """
    re_abs = np.abs(re)
    return np.divide(friction_re(re, rel_rough), re_abs, \
        out = np.full_like(re_abs, np.inf), where = re_abs > 0)

def conductivity_factor(re: np.ndarray, rel_rough: np.ndarray) -> np.ndarray:
    """ Calculates the correction of the conductivity due to inertial effects.

    The Darcy-Weisbach conductivity is the Hagen-Poiseuille one multiplied by
    64 / (f * Re). The factor equals 1 in the laminar limit and decreases
    monotonically with Re.

    Parameters
    -------
    re : numpy ndarray
        Reynolds number of the edges

    rel_rough : numpy ndarray
        wall roughness of the edges relative to their diameter

    Returns
    -------
    numpy ndarray
        conductivity correction factor of the edges (in range (0, 1])
    """
    # clip the roundoff in the laminar limit, so that the factor never exceeds
    # 1 (the root find below relies on that)
    return np.minimum(64 / friction_re(re, rel_rough), 1)

def relative_roughness(sid: SimInputData, edges: Edges) -> np.ndarray:
    """ Calculates the wall roughness of edges relative to their diameter.

    Parameters
    -------
    sid : SimInputData class object
        all config parameters of the simulation
        roughness - wall roughness relative to the characteristic diameter

    edges : Edges class object
        all edges in network and their parameters
        diams - diameters

    Returns
    -------
    numpy ndarray
        roughness of the edges relative to their diameter
    """
    return np.divide(sid.roughness, edges.diams, \
        out = np.zeros_like(edges.diams), where = edges.diams > 0)

def conductivity(sid: SimInputData, edges: Edges, re: np.ndarray = None) \
    -> np.ndarray:
    """ Calculates the conductivity of the edges.

    For the Hagen-Poiseuille relation (or zero Reynolds number) it is simply
    d ** 4 / l, for the Darcy-Weisbach one it is additionally multiplied by the
    inertial correction factor.

    Parameters
    -------
    sid : SimInputData class object
        all config parameters of the simulation
        include_dw - use Darcy-Weisbach instead of Hagen-Poiseuille

    edges : Edges class object
        all edges in network and their parameters
        diams - diameters
        lens - lengths

    re : numpy ndarray, optional
        Reynolds number of the edges (laminar flow if not given)

    Returns
    -------
    numpy ndarray
        conductivity of the edges
    """
    cond = edges.diams ** 4 / edges.lens
    if not sid.include_dw or re is None:
        return cond
    return cond * conductivity_factor(re, relative_roughness(sid, edges))

def solve_reynolds(sid: SimInputData, edges: Edges, pressure_drop: np.ndarray) \
    -> np.ndarray:
    """ Calculates the Reynolds number of the edges for a given pressure drop.

    For a fixed pressure drop the Darcy-Weisbach relation gives an implicit
    equation for the Reynolds number of each edge

        Re = Re_lam * g(Re),    Re_lam = Re0 * d ** 3 * |dp| / l,

    with g = 64 / (f * Re) being the conductivity correction factor. Since g is
    positive, decreasing and equal 1 in the laminar limit, the solution
    x = Re / Re_lam always lies in (0, 1] and the root is bracketed from the
    start. We find it with the Illinois (modified regula falsi) method, which
    keeps the bracket and converges superlinearly.

    Parameters
    -------
    sid : SimInputData class object
        all config parameters of the simulation
        Re0 - Reynolds number of a characteristic edge
        dw_re_iter - maximum number of iterations of the root find

    edges : Edges class object
        all edges in network and their parameters
        diams - diameters
        lens - lengths

    pressure_drop : numpy ndarray
        pressure drop along the edges

    Returns
    -------
    numpy ndarray
        Reynolds number of the edges
    """
    rel_rough = relative_roughness(sid, edges)
    # Reynolds number the edge would have if the flow stayed laminar - an upper
    # bound for the actual one
    re_lam = sid.Re0 * edges.diams ** 3 * np.abs(pressure_drop) / edges.lens
    # edges which are laminar even for that upper bound (in a typical network
    # the vast majority) are solved exactly by Re = Re_lam and are skipped
    inertial = conductivity_factor(re_lam, rel_rough) < 1
    if not inertial.any():
        return re_lam
    re_iner, rough_iner = re_lam[inertial], rel_rough[inertial]
    # bracket of the rescaled solution x = Re / Re_lam; the residual
    # r(x) = g(x * Re_lam) - x is decreasing, with r(0) = 1 and r(1) < 0
    x_lo, x_hi = np.zeros_like(re_iner), np.ones_like(re_iner)
    r_lo = np.ones_like(re_iner)
    r_hi = conductivity_factor(re_iner, rough_iner) - 1
    x = np.ones_like(re_iner)
    for _ in range(sid.dw_re_iter):
        denom = r_hi - r_lo
        # denominator vanishes only for edges which are already converged
        x = np.where(denom == 0, x_hi, \
            (x_lo * r_hi - x_hi * r_lo) / np.where(denom == 0, 1, denom))
        r_x = conductivity_factor(x * re_iner, rough_iner) - x
        # the residual is compared with x itself, since for strongly inertial
        # edges the solution is a small fraction of the laminar estimate and
        # we need it with a relative, not absolute accuracy
        if np.max(np.abs(r_x) / np.maximum(x, 1e-300)) < X_TOL:
            break
        # if the residual is positive, the root lies above x, so x becomes the
        # new lower end of the bracket; the retained end has its residual
        # halved (Illinois), which prevents the bracket from stalling
        below = r_x > 0
        x_lo, r_lo, x_hi, r_hi = np.where(below, x, x_lo), \
            np.where(below, r_x, 0.5 * r_lo), np.where(below, x_hi, x), \
            np.where(below, 0.5 * r_hi, r_x)
    re = re_lam.copy()
    re[inertial] = x * re_iner
    return re

def build_matrix(inc: Incidence, graph: Graph, cond: np.ndarray) \
    -> spr.csr_matrix:
    """ Builds the matrix for the pressure equation.

    Parameters
    -------
    inc : Incidence class object
        matrices of incidence
        incidence - incidence of all nodes and edges

    graph : Graph class object
        network and all its properties
        in_vec - vector with ones for inlet nodes
        out_vec - vector with ones for outlet nodes

    cond : numpy ndarray
        conductivity of the edges

    Returns
    -------
    scipy sparse matrix
        matrix of the pressure equation
    """
    # create matrix (nsq x nsq) for solving equations for pressure and flow
    # to find pressure in each node
    p_matrix = inc.incidence.T @ spr.diags(cond) @ inc.incidence
    # for all inlet nodes we set the same pressure, for outlet nodes we set
    # zero pressure; so for boundary nodes we zero the elements of p_matrix
    # and add identity for those rows
    p_matrix = p_matrix.multiply((1 - graph.in_vec \
        - graph.out_vec)[:, np.newaxis]) + spr.diags(graph.in_vec \
        + graph.out_vec)
    diag = p_matrix.diagonal()
    diag_old = diag.copy()
    # fix for nodes with no connections
    diag += 1 * (diag == 0)
    p_matrix += spr.diags(diag - diag_old)
    return p_matrix

def solve_pressure(sid: SimInputData, inc: Incidence, graph: Graph, \
    pressure_b: spr.csc_matrix, cond: np.ndarray, normalize_flow: bool) \
    -> np.ndarray:
    """ Solves the linear pressure equation for a fixed conductivity.

    Parameters
    -------
    sid : SimInputData class object
        all config parameters of the simulation
        flow_bc - type of the flow boundary condition
        Q_in - total inlet flow
        p_in - inlet pressure

    inc : Incidence class object
        matrices of incidence
        incidence - incidence of all nodes and edges
        inlet - incidence of nodes and edges for inlet

    graph : Graph class object
        network and all its properties

    pressure_b : scipy sparse vector
        result vector for pressure equation

    cond : numpy ndarray
        conductivity of the edges

    normalize_flow : bool
        normalize the solution to the constant total inlet flow (instead of
        the constant inlet pressure)

    Returns
    -------
    pressure : numpy ndarray
        vector of pressure in nodes
    """
    p_matrix = build_matrix(inc, graph, cond)
    # solve matrix @ pressure = pressure_b
    pressure = solve_equation(p_matrix, pressure_b)
    # if pressure solve doesn't work, this can be caused by changes in network
    # topology that leave an isolated component - check against that
    if np.isnan(pressure).any():
        keep_largest_component(inc)
        p_matrix = build_matrix(inc, graph, cond)
        pressure = solve_equation(p_matrix, pressure_b)
        if np.isnan(pressure).any():
            raise ValueError("Pressure solve error")
    # normalize pressure in inlet nodes to match condition for constant inlet
    # flow (for a fixed conductivity the equation is linear, so rescaling the
    # solution is exact)
    if normalize_flow:
        q_in = np.abs(np.sum(cond * (inc.inlet @ pressure)))
        pressure *= sid.Q_in / q_in
    else:
        pressure *= sid.p_in
    return pressure

def solve_flow(sid: SimInputData, inc: Incidence, graph: Graph, edges: Edges, \
    pressure_b: spr.csc_matrix) -> np.ndarray:
    """ Calculates pressure and flow.

    For the Hagen-Poiseuille conductivity a single linear solve is performed.
    For the Darcy-Weisbach one the conductivity depends on the flow, so we
    iterate: with the conductivity frozen we solve the linear system for
    pressure, then for the resulting pressure drops we find the Reynolds number
    of each edge (and thus the new conductivity) and repeat until the
    conductivity stops changing (by dw_rtol). The update can be relaxed with
    dw_relax. Note that the flow is always calculated with the conductivity
    used in the last pressure solve, so it satisfies the continuity equation
    exactly, and the Darcy-Weisbach relation up to dw_rtol.

    Parameters
    -------
    sid : SimInputData class object
        all config parameters of the simulation
        include_dw - use Darcy-Weisbach instead of Hagen-Poiseuille
        Re0 - Reynolds number of a characteristic edge
        dw_rtol - relative tolerance of the nonlinear solve
        dw_max_iter - maximum number of iterations of the nonlinear solve
        dw_relax - relaxation factor of the conductivity update
        flow_bc - type of the flow boundary condition
        qin - characteristic flow for inlet edge

    inc : Incidence class object
        matrices of incidence; here all of shape (ne x nsq)
        incidence - incidence of all nodes and edges
        middle - incidence of nodes and edges for all but inlet and outlet
        boundary - incidence of nodes and edges for inlet and outlet
        inlet - incidence of nodes and edges for inlet

    graph : Graph class object
        network and all its properties
        in_nodes - inlet nodes

    edges : Edges class object
        all edges in network and their parameters
        diams - diameters
        lens - lengths

    pressure_b : scipy sparse vector
        result vector for pressure equation

    Returns
    -------
    pressure : numpy ndarray
        vector of pressure in nodes
    """
    if sid.flow_bc not in ('q', 'p'):
        raise ValueError(f"Unknown flow BC: {sid.flow_bc}")
    # with constant pressure BC we still normalize the flow in the first
    # iteration of the simulation, to find the corresponding inlet pressure
    normalize_flow = (sid.flow_bc == 'q' or sid.p_in == 0)
    if not sid.include_dw:
        cond = conductivity(sid, edges)
        pressure = solve_pressure(sid, inc, graph, pressure_b, cond, \
            normalize_flow)
        edges.flow = cond * (inc.incidence @ pressure)
        edges.conductivity = cond
        edges.re = np.zeros_like(cond)
        if sid.flow_bc == 'p' and sid.p_in == 0:
            sid.p_in = np.max(pressure)
        return pressure
    # warm start from the Reynolds numbers of the previous iteration of the
    # simulation (diameters change slowly, so it saves a few iterations)
    re = getattr(edges, 're', None)
    if re is None or np.shape(re) != np.shape(edges.diams):
        re = np.zeros_like(edges.diams)
    cond = conductivity(sid, edges, re)
    # the Hagen-Poiseuille conductivity is the upper bound of the physical one
    cond_lam = conductivity(sid, edges)
    pressure, pressure_old, pressure_drop = None, None, None
    cond_used = cond
    for it in range(sid.dw_max_iter):
        # solve the linear problem for the frozen conductivity
        cond_used = cond
        pressure = solve_pressure(sid, inc, graph, pressure_b, cond_used, \
            normalize_flow)
        pressure_drop = inc.incidence @ pressure
        # find the Reynolds number consistent with the new pressure drops and
        # the conductivity it implies
        re = solve_reynolds(sid, edges, pressure_drop)
        cond_new = conductivity(sid, edges, re)
        # the relative change of the conductivity is the residual of the
        # nonlinear system - it is exactly by how much the flow calculated
        # below misses the Darcy-Weisbach relation
        error = np.max(np.abs(cond_new - cond_used) \
            / np.where(cond_used > 0, cond_used, 1))
        if sid.dw_verbose:
            change = 0. if pressure_old is None else \
                np.linalg.norm(pressure - pressure_old) \
                / np.linalg.norm(pressure)
            print(f'  Darcy-Weisbach iteration {it}: residual = {error:.2e}, \
pressure change = {change:.2e}')
        pressure_old = pressure
        if error < sid.dw_rtol:
            break
        # update the conductivity; the relaxation is done on the logarithm of
        # the conductivity, which keeps it positive for any dw_relax (in
        # particular it allows overrelaxation with dw_relax > 1, which can
        # speed up the convergence, as the iteration is monotonic); we also cut
        # the update at the Hagen-Poiseuille value, so that an overrelaxed step
        # cannot leave the physical range
        ratio = np.divide(cond_new, cond_used, out = np.ones_like(cond_new), \
            where = cond_used > 0)
        cond = np.minimum(cond_used * ratio ** sid.dw_relax, cond_lam)
    else:
        print(f'Warning: Darcy-Weisbach solve did not converge in \
{sid.dw_max_iter} iterations')
    # update flow; we use the conductivity of the last pressure solve, so that
    # the flow is exactly divergence free
    edges.flow = cond_used * pressure_drop
    edges.conductivity = cond_used
    edges.re = re
    if sid.flow_bc == 'p' and sid.p_in == 0:
        sid.p_in = np.max(pressure)
    return pressure
