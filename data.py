""" Collect physical data from the simulation and save/plot them.

This module initializes Data class, storing information about physical data in
the simulation. It stores the data during simulation and afterwards saves them
in a text file and plots them. For now the data are: pressure difference
between input and output (1 / permeability) and quantities of substance B and C
that flowed out of the system.

Notable classes
-------
Data
    container for physical data collected during simulation

TO DO: name data on plots, maybe collect permeability explicitly
"""

from matplotlib import gridspec
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as spr
from matplotlib.lines import Line2D

from config import SimInputData
from network import Edges, Graph, Triangles
from incidence import Incidence
from volumes import Volumes

font = {'family' : 'Liberation Serif',
        'weight' : 'normal',
        'size'   : 50}

matplotlib.rc('font', **font)

class Data():
    """ Contains data collected during the simulation.

    Attributes
    -------
    t : list
        elapsed time of the simulation

    pressure : list
        pressure difference between inlet and outlet

    cb_out : list
        difference of inflow and outflow of substance B in the system

    cb_out : list
        difference of inflow and outflow of substance C in the system

    delta_b : float
        current difference of inflow and outflow of substance B in the system

    delta_c : float
        current difference of inflow and outflow of substance C in the system
    """
    t = []
    pressure = []
    order = []
    participation_ratio = []
    participation_ratio_nom = []
    participation_ratio_denom = []
    porosity = []
    replaced = []
    cb_out = []
    cc_out = []
    delta_b = 0.
    delta_c = 0.
    delta_d = 0.
    delta_d_list = []
    dissolved_v = 0.
    dissolved_v_list = []
    slices: list = []
    slices_d: list = []
    slices_s: list = []
    slices_phi: list = []
    slices_a: list = []
    slices_e: list = []
    "channelization for slices through the whole system in a given time"
    slice_times: list = []
    "list of times of checking slice channelization"
    breakthrough_times: list = []
    concentrations: list = []
    reactive_breakthrough_times: list = []
    track_times: list = []
    vol_dissolved = 0.
    vol_precipitated = 0.
    vol_dissolved_list = []
    vol_precipitated_list = []
    A_vol_expected_cumulative = 0.0   # Σ_t (B_consumed × dt / Da), updated every growth step
    E_vol_expected_cumulative = 0.0   # Σ_t (Γ/Da × D_consumed × dt), updated every growth step

    def __init__(self, sid: SimInputData, edges: Edges):
        self.dirname = sid.dirname
        self.vol_init = np.sum(edges.diams ** 2 * edges.lens)

    def save_data(self) -> None:
        """ Save data to text file.

        This function saves the collected data to text file params.txt in
        columns. If the simulation is continued from saved parameters, new data
        is appended to that previously collected.
        """
        is_saved = False
        while not is_saved: # prevents problems with opening text file
            try:
                file = open(self.dirname + '/params.txt', 'w', \
                    encoding = "utf-8")
                np.savetxt(file, np.array([self.t, self.dissolved_v_list, self.pressure, self.porosity, self.replaced, self.cb_out, \
                    self.cc_out, self.vol_dissolved_list, self.vol_precipitated_list, self.delta_d_list], dtype = float).T)
                file.close()
                is_saved = True
            except PermissionError:
                pass
        # self slice data to slices.txt
        is_saved = False
        while not is_saved: # prevents problems with opening text file
            try:
                file = open(self.dirname + '/profiles.txt', 'w', \
                    encoding = "utf-8")
                np.savetxt(file, self.slices)
                file.close()
                is_saved = True
            except PermissionError:
                pass
        is_saved = False
        while not is_saved: # prevents problems with opening text file
            try:
                file = open(self.dirname + '/profiles_phi.txt', 'w', \
                    encoding = "utf-8")
                np.savetxt(file, self.slices_phi)
                file.close()
                is_saved = True
            except PermissionError:
                pass
        # is_saved = False
        # while not is_saved: # prevents problems with opening text file
        #     try:
        #         file = open(self.dirname + '/track.txt', 'w', \
        #             encoding = "utf-8")
        #         np.savetxt(file, self.breakthrough_times)
        #         file.close()
        #         file = open(self.dirname + '/c_track.txt', 'w', \
        #             encoding = "utf-8")
        #         np.savetxt(file, self.concentrations)
        #         file.close()
        #         file = open(self.dirname + '/r_track.txt', 'w', \
        #             encoding = "utf-8")
        #         np.savetxt(file, self.reactive_breakthrough_times)
        #         file.close()
        #         is_saved = True
        #     except PermissionError:
        #         pass

    def load_data(self) -> None:
        data = np.loadtxt(self.dirname + '/params.txt').T
        self.t, self.pressure, self.participation_ratio, self.cb_out, \
            self.cc_out = list(data[0]), list(data[1]), list(data[2]), list(data[3]), list(data[4])
        self.slices = list(np.loadtxt(self.dirname + '/slices.txt'))

    def check_data(self, edges: Edges) -> None:
        """ Check the key physical parameters of the simulation.

        This function calculates and checks if basic physical properties of the
        simulation are valied, i.e. if inflow is equal to outflow.

        Parameters
        -------
        edges : Edges class object
            all edges in network and their parameters
            flow - flow in edges
            inlet - edges connected to inlet nodes
            outlet - edges connected to outlet nodes
        """
        Q_in = np.sum(edges.inlet * np.abs(edges.flow))
        Q_out = np.sum(edges.outlet * np.abs(edges.flow))
        print('Q_in =', Q_in, 'Q_out =', Q_out)
        if np.abs(Q_in - Q_out) > 1:
            raise ValueError('Flow not matching!')
        # delta = np.abs((np.abs(inc.incidence.T < 0) @ (np.abs(edges.flow) \
        #     * edges.inlet) - np.abs(inc.incidence.T > 0) @ (np.abs(edges.flow) \
        #     * edges.outlet)) @ cb * sid.dt)


    def collect_data(self, sid: SimInputData, inc: Incidence, edges: Edges, vols, \
        p: np.ndarray, cb: np.ndarray, cc: np.ndarray, cd: np.ndarray) -> None:
        """ Collect data from different vectors.

        This function extracts information such as permeability, quantity of
        substances flowing out of the system etc. and saves them in the data
        class.

        Parameters
        -------
        sid : SimInputData class object
            all config parameters of the simulation
            old_t - total time of simulation
            dt - current timestep

        inc : Incidence class object
            matrices of incidence
            incidence - connections of all edges with all nodes

        edges : Edges class object
            all edges in network and their parameters
            flow - flow in edges
            inlet - edges connected to inlet nodes
            outlet - edges connected to outlet nodes

        p : numpy ndarray
            vector of current pressure

        cb : numpy ndarray
            vector of current substance B concentration

        cc : numpy ndarray
            vector of current substance C concentration
        """
        self.t.append(sid.old_t + sid.dt)

        self.pressure.append(np.max(p))
        self.order.append((sid.ne - np.sum(edges.flow ** 2) ** 2 \
            / np.sum(edges.flow ** 4)) / (sid.ne - 1))
        pi = np.sum(edges.diams ** 2 * np.abs(edges.flow)) ** 2 / np.sum(edges.diams ** 2 \
            * np.abs(edges.flow) ** 2) / sid.nsq
        pi_prime = np.sum(edges.diams ** 2) / sid.nsq
        self.participation_ratio_nom.append(pi)
        self.participation_ratio_denom.append(pi_prime)
        self.participation_ratio.append(pi / pi_prime)
        # calculate the difference between inflow and outflow of each substance

        # if sid.include_diffusion:

        #     lam_plus_val = sid.Pe / 2 / edges.diams ** 2 * \
        #         (np.sqrt(np.abs(edges.flow) ** 2 + 4 * edges.alpha * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) + np.abs(edges.flow))
        #     lam_plus_val = np.array(np.ma.fix_invalid(lam_plus_val, fill_value = 0))
        #     lam_plus_zero = 1 * (lam_plus_val > sid.diffusion_exp_limit)
        #     lam_plus_val = lam_plus_val * (1 - lam_plus_zero)
        #     lam_minus_val = sid.Pe / 2 / edges.diams ** 2 * \
        #         (np.sqrt(np.abs(edges.flow) ** 2 + 4 * edges.alpha * sid.Da / (1 + sid.G * edges.diams) / sid.Pe * edges.diams ** 3) - np.abs(edges.flow))
        #     lam_minus_val = np.array(np.ma.fix_invalid(lam_minus_val, fill_value = 0))
        #     # Not sure how to calculate J_out - should it be just q_out c_out (as we set dc/dx = 0 at the outlet? - do we for 100%?)
        #     # But no matter how I calculate, I end up with a small error, there could be some small bug
        #     #J_in2 = np.sum(edges.inlet * (np.abs(edges.flow) * (edges.A * (1 - lam_plus_zero) + edges.B) - (1 - lam_plus_zero) / sid.Pe * edges.diams ** 2 * (lam_plus_val * edges.A - lam_minus_val * edges.B)))
        #     #J_in = np.sum(edges.inlet * (np.abs(edges.flow) * (edges.A + edges.B) - 1 / sid.Pe * edges.diams ** 2 *  (lam_plus_val * edges.A - lam_minus_val * edges.B)))
        #     #J_out2 = np.sum((1 - lam_plus_zero) * edges.outlet * (np.abs(edges.flow) * (edges.A * np.exp(lam_plus_val * edges.lens) + edges.B * np.exp(-lam_minus_val * edges.lens)) - 1 / sid.Pe * edges.diams ** 2 *(lam_plus_val * edges.A * np.exp(lam_plus_val * edges.lens) - lam_minus_val * edges.B * np.exp(-lam_minus_val * edges.lens))) + lam_plus_zero * edges.outlet * edges.B * np.abs(edges.flow) * np.exp(-edges.alpha * sid.Da / (1 + sid.G * edges.diams) * edges.diams * edges.lens / np.abs(edges.flow)))
        #     #J_out = np.sum(edges.outlet * (np.abs(edges.flow) * (edges.A * np.exp(lam_plus_val * edges.lens) + edges.B * np.exp(-lam_minus_val * edges.lens)) - 1 / sid.Pe * edges.diams ** 2 *(lam_plus_val * edges.A * np.exp(lam_plus_val * edges.lens) - lam_minus_val * edges.B * np.exp(-lam_minus_val * edges.lens))))
        #     #J_out = np.sum(edges.outlet * (np.abs(edges.flow) * (edges.A * np.exp(lam_plus_val * edges.lens) + edges.B * np.exp(-lam_minus_val * edges.lens)) - 1 / sid.Pe * edges.diams ** 2 *(lam_plus_val * edges.A * np.exp(lam_plus_val * edges.lens) - lam_minus_val * edges.B * np.exp(-lam_minus_val * edges.lens))))
        #     J_in = np.sum(edges.inlet * (np.abs(edges.flow) * (edges.A * (1 - lam_plus_zero) + edges.B) - (1 - lam_plus_zero) / sid.Pe * edges.diams ** 2 * (lam_plus_val * edges.A - lam_minus_val * edges.B)))
        #     J_out = np.abs(np.abs(inc.incidence.T > 0) @ (np.abs(edges.flow) \
        #          * edges.outlet)) @ cb
        
        # J_out3 = np.abs(np.abs(inc.incidence.T > 0) @ (np.abs(edges.flow) \
        #     * edges.outlet)) @ cb
        # print(f'lam plus zero: {np.sum(lam_plus_zero), np.sum(1 - lam_plus_zero)}')
        
        # print(f'J_in2: {J_in2 * sid.dt}, J_out2: {J_out2 * sid.dt}')
        # print(f'delta J_adv: {(np.sum(edges.inlet * (np.abs(edges.flow) * (edges.A + edges.B)))-np.sum(edges.outlet * (np.abs(edges.flow) * (edges.A * np.exp(lam_plus_val * edges.lens) + edges.B * np.exp(-lam_minus_val * edges.lens))))) * sid.dt}')
        # print(f'delta J_diff: {(np.sum(edges.inlet * (- 1 / sid.Pe * edges.diams ** 2 *  (lam_plus_val * edges.A - lam_minus_val * edges.B)))-np.sum(edges.outlet * (-1 / sid.Pe * edges.diams ** 2 *(lam_plus_val * edges.A * np.exp(lam_plus_val * edges.lens) - lam_minus_val * edges.B * np.exp(-lam_minus_val * edges.lens))))) * sid.dt}')

        # delta = np.abs((np.abs(inc.incidence.T < 0) @ (np.abs(edges.flow) \
        #     * edges.inlet) - np.abs(inc.incidence.T > 0) @ (np.abs(edges.flow) \
        #     * edges.outlet)) @ cb * sid.dt)
        if sid.include_diffusion:
            print(f'J_in: {self.J_in}, J_out: {self.J_out}')
            delta = (self.J_in - self.J_out) * sid.dt
        else:
            # delta = np.abs((np.abs(inc.incidence.T < 0) @ (np.abs(edges.flow) \
            #     * edges.inlet) - np.abs(inc.incidence.T > 0) @ (np.abs(edges.flow) \
            #     * edges.outlet)) @ cb * sid.dt)
            delta = np.abs(np.abs(1 * ( inc.incidence.T @ spr.diags(edges.flow) > 0) @ (np.abs(edges.flow) \
                 * edges.inlet)) @ cb - np.abs(1 * ( inc.incidence.T @ spr.diags(edges.flow) < 0) @ (np.abs(edges.flow) \
                 * edges.outlet)) @ cb) * sid.dt
        vol_dissolved = np.sum(edges.diams ** 2 * edges.lens) - self.vol_init
        vol_a = np.sum(vols.vol_a_0 - vols.vol_a)
        print(f'Zero volume: {np.sum(vols.vol_a == 0)}')
        self.delta_b += delta
        # delta2 = np.abs((np.abs(inc.incidence.T < 0) @ (np.abs(edges.flow) \
        #         * edges.inlet) - np.abs(inc.incidence.T > 0) @ (np.abs(edges.flow) \
        #         * edges.outlet)) @ cb * sid.dt)
        # print(f'Delta2: {delta2}')
        # print(f'Delta3: {(J_in - J_out2) * sid.dt}')
        # print(f'Delta4: {(J_in - J_out3) * sid.dt}')
        print(f'Used concentration: {self.delta_b}, Dissolved volume: {sid.Da * vol_dissolved}, Dissolved volume A: {sid.Da * vol_a}')
        #print(f'c - V: {(self.delta_b - sid.Da * vol_dissolved / 2) / self.delta_b}, c - V_A: {(self.delta_b - sid.Da * vol_a / 2) / self.delta_b}, V - V_A {(vol_dissolved - vol_a) / vol_dissolved}')
        #print(f'c - V: {(self.delta_b - sid.Da * vol_dissolved)}, c - V_A: {(self.delta_b - sid.Da * vol_a)}, V - V_A {(vol_dissolved - vol_a)}')
        self.cb_out.append(self.delta_b)
        delta_c = np.abs(np.abs(1 * ( inc.incidence.T @ spr.diags(edges.flow) > 0) @ (np.abs(edges.flow) \
                 * edges.inlet)) @ cc - np.abs(1 * ( inc.incidence.T @ spr.diags(edges.flow) < 0) @ (np.abs(edges.flow) \
                 * edges.outlet)) @ cc) * sid.dt
        self.delta_c += delta_c
        delta_d = np.abs(np.abs(1 * ( inc.incidence.T @ spr.diags(edges.flow) > 0) @ (np.abs(edges.flow) \
                 * edges.inlet)) @ cd - np.abs(1 * ( inc.incidence.T @ spr.diags(edges.flow) < 0) @ (np.abs(edges.flow) \
                 * edges.outlet)) @ cd) * sid.dt
        self.delta_d += delta_d
        self.delta_d_list.append(self.delta_d)
        # self.injected_d += np.abs(np.abs(1 * ( inc.incidence.T @ spr.diags(edges.flow) > 0) @ (np.abs(edges.flow) \
        #          * edges.inlet)) @ cd) * sid.dt
        # self.injected_list.append(self.injected_d)
        print(f'Delta c_B: {self.delta_b}, Delta c_C: {self.delta_c}, Delta c_D: {self.delta_d}')
        print(f'Delta c_B + Delta c_C - Delta c_D: {(delta - delta_c - delta_d)}')
        # if np.abs(delta - delta_c - delta_d) > 1e-3:
        #     raise ValueError
        self.cc_out.append(self.delta_c)
        self.dissolved_v = np.sum(vols.vol_a) / np.sum(vols.vol_max)
        self.dissolved_v_list.append(self.dissolved_v)
        self.porosity.append(1 - np.sum(vols.vol_a + vols.vol_e) / np.sum(vols.vol_max))
        self.replaced.append(np.sum(vols.vol_e) / np.sum(vols.vol_max))
        self.vol_dissolved_list.append(self.vol_dissolved)
        self.vol_precipitated_list.append(self.vol_precipitated)

    def check_mass_balance(self, sid, inc, edges, vols, cb, cc, cd,
                           verbose: bool = True, tol: float = None) -> dict:
        """Four global mass-balance checks.

        Physical model
        --------------
        Dissolution:    A(s) + B(aq) → C(aq) + products    rate ∝ Da/(1+G·d)
        Precipitation:  C(aq) + D(aq) → E(s)               rate ∝ Da·K·c_D/(Kp·(1+G·K·d))

        Checks
        ------
        (1) Solid volume:  Σ(d²L) + ΣV_A + ΣV_E  =  const
            Pore + rock A + mineral E must sum to initial total volume.
            Violated when diameter update and volume bookkeeping are inconsistent.

        (2) B flux vs dissolution:  B_in − B_out  ≈  Σ_e c_B↑ |Q_e| (1 − exp(−Da·d·L/(|Q|(1+G·d))))
            Net acid consumed equals the sum of local dissolution sinks.
            Residual ≠ 0 signals that the concentration solver breaks global B balance.

        (3) D flux vs precipitation:  D_in − D_out  ≈  Σ_e c_C↑ |Q_e| (1 − exp(−Da·K·c_D↑·d·L/(Kp·|Q|·(1+G·K·d))))
            C and D are consumed 1-for-1 in precipitation, so D consumed equals
            the total C sink.  E formation rate = (Γ/Da) × (D_in − D_out).

        (4) Cumulative tracker drift:
            vol_dissolved  −  (ΣV_A0 − ΣV_A)  — non-zero when alpha_b < 1
                (growth.py uses unscaled growth for vol_a, but alpha_b-scaled growth for diameters)
            vol_precipitated  −  ΣV_E          — should always be 0

        All residuals should be small relative to their reference fluxes.
        """
        abs_q  = np.abs(edges.flow)
        q_safe = np.where(abs_q > 1e-12, abs_q, 1.0)
        flowing = abs_q > 1e-12
        upstream   = np.abs(spr.diags(edges.flow) @ inc.incidence > 0)
        downstream = 1 * (spr.diags(edges.flow) @ inc.incidence < 0)

        cb_up = upstream @ cb
        cc_up = upstream @ cc
        cd_up = upstream @ cd
        # downstream concentrations: concentration *leaving* each edge after wall reaction.
        # For outlet edges this is what actually exits the domain, not the interior-node value.
        cb_dn = downstream @ cb
        cc_dn = downstream @ cc
        cd_dn = downstream @ cd

        # ── (1) Solid volume conservation ────────────────────────────────
        edge_vol   = np.sum(edges.diams ** 2 * edges.lens)
        vol_a_sum  = np.sum(vols.vol_a)
        vol_e_sum  = np.sum(vols.vol_e)
        vol_a0_sum = np.sum(vols.vol_a_0)

        # Σ(d²L) + ΣV_A + ΣV_E  should equal  vol_init + ΣV_A0  (constant)
        total_vol_residual = (edge_vol + vol_a_sum + vol_e_sum) - (self.vol_init + vol_a0_sum)

        delta_vol_a    = vol_a0_sum - vol_a_sum          # solid A dissolved  (≥ 0)
        delta_edge_vol = edge_vol   - self.vol_init       # pore volume gained (> 0 during dissolution)
        # Should hold: Δ(d²L) = ΔV_A − V_E
        solid_balance_residual = delta_edge_vol - (delta_vol_a - vol_e_sum)

        # ── (2) B flux vs local dissolution ──────────────────────────────
        Q_in_total = np.sum(edges.inlet  * abs_q)
        B_in  = Q_in_total * sid.cb_in
        # cb_dn = concentration *leaving* each edge after wall reaction;
        # for outlet edges this is what exits the domain, not the upstream interior value.
        B_out = np.sum(edges.outlet * abs_q * cb_dn)

        # alpha_b-scaled exp: matches solve_dissolution_safe exactly
        exp_d = np.where(flowing,
                         np.exp(-sid.Da * edges.alpha_b / (1.0 + sid.G * edges.diams)
                                * edges.diams * edges.lens / q_safe),
                         0.0)
        # diss_flux = Σ c_B↑ |Q| (1−exp_d) = B consumed = C produced (by dissolution)
        diss_flux  = np.sum(cb_up * abs_q * (1.0 - exp_d))
        B_residual = (B_in - B_out) - diss_flux
        B_rel      = B_residual / max(B_in, 1e-14)

        # ── (3) D flux vs precipitation ───────────────────────────────────
        D_in  = Q_in_total * sid.cd_in
        D_out = np.sum(edges.outlet * abs_q * cd_dn)
        C_in  = Q_in_total * sid.cc_in
        C_out = np.sum(edges.outlet * abs_q * cc_dn)

        D_consumed = D_in - D_out   # exact D consumed by precipitation solver

        # Stoichiometric check (exact, no growth-formula approximation):
        #   A+B→C  ⟹  C produced = B consumed = B_in−B_out
        #   C+D→E  ⟹  D consumed = C consumed = (C produced) − (C net exported)
        #             = (B_in−B_out) − (C_out−C_in)
        # If the solver is stoichiometrically consistent, D_consumed should equal this.
        C_net_exported   = C_out - C_in
        D_stoich_expect  = (B_in - B_out) - C_net_exported
        stoich_residual  = D_consumed - D_stoich_expect
        stoich_rel       = stoich_residual / max(D_in, 1e-14)

        # Growth-formula approximation (for comparison only):
        exp_p = np.where(flowing,
                         np.exp(-sid.Da * sid.K / (1.0 + sid.G * sid.K * edges.diams)
                                * cd_up / sid.Kp * edges.diams * edges.lens / q_safe),
                         0.0)
        # prec_flux ≈ D consumed (growth formula, treats cd as constant along edge)
        prec_flux  = np.sum(cc_up * abs_q * (1.0 - exp_p))
        D_residual = D_consumed - prec_flux
        D_rel      = D_residual / max(D_in, 1e-14)

        # ── (2b) A volume vs B moles cross-check (cumulative) ────────────
        # A_vol_expected_cumulative = Σ_t (B_consumed_t × dt / Da), accumulated
        # every growth step in growth.py.  delta_vol_a is the ground-truth dissolved volume.
        # Residual reveals whether the dissolution rate (cb solver) is consistent with
        # the actual vol_a change (including alpha_b clipping and vol_a → 0 clipping).
        A_vol_expected = self.A_vol_expected_cumulative
        A_vol_residual = delta_vol_a - A_vol_expected
        A_vol_rel      = A_vol_residual / max(A_vol_expected, 1e-14)

        # ── (3b) E volume vs D moles cross-check (cumulative) ────────────
        # E_vol_expected_cumulative = Σ_t (Γ/Da × D_consumed_t × dt), accumulated
        # every growth step in growth.py.  vol_e_sum is the actual precipitated volume.
        # Residual reveals inconsistency between the concentration solver's D bookkeeping
        # and the growth step's vol_e accounting (e.g. alpha_c clipping the solver ignores).
        E_vol_expected = self.E_vol_expected_cumulative
        E_vol_residual = vol_e_sum - E_vol_expected
        E_vol_rel      = E_vol_residual / max(E_vol_expected, 1e-14)

        # ── (4) Cumulative tracker drift ──────────────────────────────────
        # vol_dissolved/precipitated accumulate actual post-clip changes in growth.py;
        # delta_vol_a and vol_e_sum are the ground-truth totals from vols — both should match.
        tracker_A_drift = self.vol_dissolved    - delta_vol_a
        tracker_E_drift = self.vol_precipitated - vol_e_sum

        if verbose:
            print(
                f"\n{'─'*60}\n"
                f" MASS BALANCE CHECK\n"
                f"{'─'*60}\n"
                f" (1) Solid volume conservation\n"
                f"     Σ(d²L)+ΣV_A+ΣV_E − const  (should be 0):  {total_vol_residual:+.4e}\n"
                f"     Δ(d²L) − (ΔV_A − V_E)     (should be 0):  {solid_balance_residual:+.4e}\n"
                f"     ΔV_A (solid dissolved):                    {delta_vol_a:.4e}\n"
                f"     V_E  (solid precipitated):                 {vol_e_sum:.4e}\n"
                f"     Δ(d²L) (pore volume gain):                 {delta_edge_vol:.4e}\n"
                f" (2) B flux / dissolution check  [uses α_b-scaled Da]\n"
                f"     B_in − B_out:                              {B_in - B_out:.4e}\n"
                f"     diss_flux Σ c_B↑|Q|(1−exp_d):             {diss_flux:.4e}\n"
                f"     residual abs / rel:                        {B_residual:+.4e} / {B_rel:+.4e}\n"
                f"     A_vol expected Σ(B_consumed×dt/Da):        {A_vol_expected:.4e}\n"
                f"     V_A dissolved actual (ΣV_A0−ΣV_A):         {delta_vol_a:.4e}\n"
                f"     V_A − A_expect residual / rel:             {A_vol_residual:+.4e} / {A_vol_rel:+.4e}\n"
                f" (3) D flux / precipitation check\n"
                f"     D_in − D_out  (exact D consumed):          {D_consumed:.4e}\n"
                f"     Stoichiometric expect  (B_in−B_out)−(C_out−C_in): {D_stoich_expect:.4e}\n"
                f"     Stoich residual abs / rel (should be 0):  {stoich_residual:+.4e} / {stoich_rel:+.4e}\n"
                f"     C net exported (C_out−C_in):               {C_net_exported:.4e}\n"
                f"     E expect cumul Σ(Γ/Da×D_consumed×dt):      {E_vol_expected:.4e}\n"
                f"     V_E actual cumul (ΣV_E from vols):          {vol_e_sum:.4e}\n"
                f"     V_E − E_expect residual / rel:             {E_vol_residual:+.4e} / {E_vol_rel:+.4e}\n"
                f"     [approx] prec_flux Σ c_C↑|Q|(1−exp_p):    {prec_flux:.4e}\n"
                f"     [approx] D−prec_flux residual / rel:      {D_residual:+.4e} / {D_rel:+.4e}\n"
                f" (4) Cumulative tracker drift\n"
                f"     vol_dissolved − (ΣV_A0−ΣV_A) [α_b effect]:{tracker_A_drift:+.4e}\n"
                f"     vol_precipitated − ΣV_E       (should be 0):{tracker_E_drift:+.4e}\n"
                f"{'─'*60}\n"
            )
        else:
            print(
                f" MB| B_rel={B_rel:+.2e}  "
                f"D_stoich_rel={stoich_rel:+.2e}  "
                f"E_vol_rel={E_vol_rel:+.2e}  "
                f"vol_resid={solid_balance_residual:+.2e}"
            )

        if tol is not None:
            if abs(B_rel) > tol:
                raise ValueError(
                    f"Mass balance violated: B_rel={B_rel:+.4e} exceeds tol={tol:.2e}"
                )
            if abs(stoich_rel) > tol:
                raise ValueError(
                    f"Mass balance violated: D_stoich_rel={stoich_rel:+.4e} exceeds tol={tol:.2e} "
                    f"(Dissolved={delta_vol_a:.4e}, Precipitated={vol_e_sum:.4e})"
                )

        return {
            'total_vol_residual':      total_vol_residual,
            'solid_balance_residual':  solid_balance_residual,
            'B_residual_abs':          B_residual,
            'B_residual_rel':          B_rel,
            'A_vol_cumul_residual_abs': A_vol_residual,
            'A_vol_cumul_residual_rel': A_vol_rel,
            'D_consumed':              D_consumed,
            'E_vol_cumul_residual_abs': E_vol_residual,
            'E_vol_cumul_residual_rel': E_vol_rel,
            'prec_flux':               prec_flux,
            'D_residual_abs':          D_residual,
            'D_residual_rel':          D_rel,
            'C_net_exported':          C_out - C_in,
            'tracker_A_drift':         tracker_A_drift,
            'tracker_E_drift':         tracker_E_drift,
        }

    def plot_data(self) -> None:
        """ Plot data from text file.

        This function loads the data from text file params.txt and plots them
        to file params.png.
        """
        f = open(self.dirname + '/params.txt', 'r', encoding = "utf-8")
        data = np.loadtxt(f)
        n_data = data.shape[1]
        t = data[:, 0]
        plt.figure(figsize = (15, 5))
        plt.suptitle('Parameters')
        spec = gridspec.GridSpec(ncols = n_data - 1, nrows = 1)
        for i_data in range(n_data - 1):
            plt.subplot(spec[i_data]).set_title(f'Data {i_data}')
            #plt.plot(t, data[:, i_data + 1] / data[0, i_data + 1])
            plt.plot(t, data[:, i_data + 1])
            plt.yscale('log')
            plt.xlabel('simulation time')
        plt.savefig(self.dirname + '/params.png')
        plt.close()

    def check_channelization(self, graph: Graph, inc: Incidence, edges: Edges, \
        slice_x: float) -> tuple[int, float]:
        """ Calculate channelization parameter for a slice of the network.

        This function calculates the channelization parameter for a slice of
        the network perpendicular to the main direction of the flow. It checks
        how many edges take half of the total flow going through the slice. The
        function returns the exact number of edges and that number divided by
        the total number of edges in a given slice (so the percentage of edges
        taking half of the total flow in the slice).

        Parameters
        -------
        graph : Graph class object
            network and all its properties

        inc : Incidence class object
            matrices of incidence

        edges : Edges class object
            all edges in network and their parameters

        slice_x : float
            position of the slice

        Returns
        -------
        int
            number of edges taking half of the flow in the slice

        float
            percentage of edges taking half of the flow in the slice
        """
        pos_x = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,0]
        # find edges crossing the given slice and their orientation - if edge
        # crosses the slice from left to right, it is marked with 1, if from
        # right to left - -1, if it doesn't cross - 0
        slice_edges = (spr.diags(edges.flow) @ inc.incidence > 0) \
            @ (pos_x <= slice_x) * np.abs(inc.incidence @ (pos_x > slice_x)) \
            - (spr.diags(edges.flow) @ inc.incidence > 0) @ (pos_x > slice_x) \
            * np.abs(inc.incidence @ (pos_x <= slice_x))
        # sort edges from maximum flow to minimum (taking into account
        # their orientation)
        slice_flow = np.array(sorted(slice_edges * np.abs(edges.flow), reverse = True))
        fraction_flow = 0
        total_flow = np.sum(slice_flow)
        # calculate how many edges take half of the flow
        for i, edge_flow in enumerate(slice_flow):
            fraction_flow += edge_flow
            if fraction_flow > total_flow / 2:
                flow_50 = i + 1
                break
        slice_diams = np.array(sorted(slice_edges * np.abs(edges.diams), reverse = True))
        fraction_diams = 0
        total_diams = np.sum(slice_diams)
        # calculate how many edges take half of the flow
        for i, edge_diam in enumerate(slice_diams):
            fraction_diams += edge_diam
            if fraction_diams > total_diams / 2:
                diams_50 = i + 1
                break
        slice_surface = np.array(sorted(slice_edges * np.abs(edges.diams ** 2), reverse = True))
        fraction_surface = 0
        total_surface = np.sum(slice_surface)
        # calculate how many edges take half of the flow
        for i, edge_surface in enumerate(slice_surface):
            fraction_surface += edge_surface
            if fraction_surface > total_surface / 2:
                surface_50 = i + 1
                break
        return (flow_50, np.sum(slice_flow != 0), diams_50, np.sum(slice_diams != 0), surface_50, np.sum(surface_50 != 0))

    def check_init_slice_channelization(self, graph: Graph, inc: Incidence, \
        edges: Edges) -> None:
        pos_x = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,0]
        slices = np.linspace(np.min(pos_x), np.max(pos_x), 102)[1:-1]
        channels_tab = []
        diams_tab = []
        surface_tab = []
        for x in slices:
            res = self.check_channelization(graph, inc, edges, x)
            channels_tab.append(res[1])
            diams_tab.append(res[3])
            surface_tab.append(res[5])
        self.slices.append(channels_tab)
        self.slices_d.append(diams_tab)
        self.slices_s.append(surface_tab)

    def check_slice_channelization(self, graph: Graph, inc: Incidence, \
        edges: Edges, time: float) -> None:
        pos_x = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,0]
        slices = np.linspace(np.min(pos_x), np.max(pos_x), 102)[1:-1]
        channels_tab = []
        diams_tab = []
        surface_tab = []
        for x in slices:
            res = self.check_channelization(graph, inc, edges, x)
            channels_tab.append(res[0])
            diams_tab.append(res[2])
            surface_tab.append(res[4])
        self.slices.append(channels_tab)
        self.slices_d.append(diams_tab)
        self.slices_s.append(surface_tab)
        #self.slice_times.append("{0}".format(str(round(time, 1) if time % 1 else int(time))))

    def plot_slice_channelization(self, graph: Graph) -> None:
        """ Plots slice data from text file.

        This function loads the data from text file slices.txt and plots them
        to files slices.png, slices_no_div.png, slices_norm.png.
        """
        pos_x = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,0]
        slices = np.linspace(np.min(pos_x), np.max(pos_x), 102)[1:-1]
        edge_number  = np.array(self.slices[0])
        plt.figure(figsize = (10, 10))
        for i, channeling in enumerate(self.slices[1:]):
            plt.plot(slices, np.array(channeling) / edge_number, \
                    label = self.slice_times[i])
        plt.xlabel('x')
        plt.ylabel('channeling [%]')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(self.dirname + '/slices.png')
        plt.close()
        plt.figure(figsize = (10, 10))
        for i, channeling in enumerate(self.slices[1:]):
            plt.plot(slices, np.array(channeling) / np.array(self.slices[1]), \
                label = self.slice_times[i])
        plt.xlabel('x')
        plt.ylabel('channeling [initial]')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(self.dirname + '/slices_norm.png')
        plt.close()
        plt.figure(figsize = (10, 10))
        for i, channeling in enumerate(self.slices[1:]):
            plt.plot(slices, channeling, label = self.slice_times[i])
        plt.xlabel('x')
        plt.ylabel('channeling [edge number]')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(self.dirname + '/slices_no_div.png')
        plt.close()

    def plot_slice_channelization_v2(self, sid: SimInputData, graph: Graph) -> None:
        """ Plots slice data from text file.

        This function loads the data from text file slices.txt and plots them
        to files slices.png, slices_no_div.png, slices_norm.png.
        """
        pos_x = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,0]
        slices = np.linspace(np.min(pos_x), np.max(pos_x), 102)[1:-1]
        edge_number  = np.array(self.slices[0])
        i_start = 5
        i_division = sid.dissolved_v_max // sid.track_every // 5
        plt.figure(figsize = (10, 10))
        for i, channeling in enumerate(self.slices[1:]):
            if i < i_start:
                plt.plot(slices, (edge_number - 2 * np.array(channeling)) / edge_number, \
                        label = self.slice_times[i])
        plt.xlabel('x')
        plt.ylabel('flow focusing index')
        plt.ylim(0, 1)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(self.dirname + '/slices_start.png')
        plt.close()
        plt.figure(figsize = (10, 10))
        for i, channeling in enumerate(self.slices[1:]):
            if i % i_division == 0:
                plt.plot(slices, (edge_number - 2 * np.array(channeling)) / edge_number, \
                        label = self.slice_times[i])
        plt.xlabel('x')
        plt.ylabel('flow focusing index')
        plt.ylim(0, 1)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(self.dirname + '/slices.png')
        plt.close()

    def plot_participation(self, sid: SimInputData):
        plt.figure(figsize = (10, 10))
        plt.title('Participation ratio')
        ax_p = plt.subplot()
        ax_p.set_title('Participation ratio')
        ax_p.set_ylim(0, 1)
        ax_p.set_xlim(0, sid.dissolved_v_max / self.vol_init)
        ax_p.set_xlabel('dissolved v')
        ax_p.set_ylabel('participation ratio')
        ax_p2 = ax_p.twinx()
        x = np.linspace(0, sid.dissolved_v_max / self.vol_init, len(self.participation_ratio))
        ax_p2.plot(x, self.participation_ratio_nom, label = "pi", color='green', linestyle='dashed')
        ax_p2.plot(x, self.participation_ratio_denom, label = "pi'", color='red', linestyle='dashed')
        ax_p.plot(x, self.participation_ratio)
        ax_p2.legend()
        plt.savefig(self.dirname + '/participation_ratio.pdf')
        plt.close()

    def plot_profile(self, graph: Graph) -> None:
        """ Plots slice data from text file.

        This function loads the data from text file slices.txt and plots them
        to files slices.png, slices_no_div.png, slices_norm.png.
        """
        pos_x = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,0]
        # slices = np.linspace(np.min(pos_x), np.max(pos_x), 120)[10:-10]
        slices = np.linspace(np.min(pos_x), np.max(pos_x), 102)[1:-1]
        edge_number  = np.array(self.slices[0])
        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
        plt.figure(figsize = (15, 10))
        plt.plot(slices / np.max(pos_x), np.array((edge_number - 2 * np.array(self.slices[1])) \
            / edge_number), linewidth = 5, color = 'black', label = '0.0')
        for i, channeling in enumerate(self.slices[2:]):
            plt.plot(slices, (edge_number - 2 * np.array(channeling)) \
                / edge_number, label = self.slice_times[i+1], color = colors[i], linewidth = 5)
        plt.ylim(0, 1.05)
        plt.xlabel(r'$x$ / L', fontsize = 60)
        plt.xticks([0, 0.5, 1], ['0', '0.5', '1'])
        # ax2.xaxis.label.set_color('white')
        # ax2.tick_params(axis = 'x', colors='white')
        #plt.xticks([],[])
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.margins(tight = True)
        plt.ylabel('flow focusing index', fontsize = 50)
        #plt.yticks([],[])
        plt.yticks([0, 0.5, 1],['0', '0.5', '1'])
        handles, labels = plt.gca().get_legend_handles_labels()
        n = len(handles)

        cols = 4
        rows = 2
        slots = rows * cols

        # Dummy handle/label (invisible entry)
        
        dummy_handle = Line2D([], [], linestyle='none', marker='', color='none')
        dummy_label = ''

        slot_handles = [dummy_handle] * slots
        slot_labels  = [dummy_label]  * slots

        if n > 0:
            # put label "0" (handles[0]) in top-right corner: row 0, col cols-1
            slot_handles[2 * cols - 2] = handles[0]
            slot_labels[2 * cols - 2]  = labels[0]

        # put the remaining labels (1..n-1) in bottom row, from left to right
        for j, (h, lab) in enumerate(zip(handles[1:], labels[1:])):
            if j >= cols:    # we only support up to 4 "other" labels
                break
            r, c = 1, j      # second row, columns 0..3
            idx = r + c * rows
            slot_handles[idx] = h
            slot_labels[idx]  = lab

        legend = plt.legend(
            slot_handles,
            slot_labels,
            loc="lower center",
            mode="expand",
            ncol=cols,
            prop={'size': 40},
            handlelength=1,
            frameon=False,
            borderpad=0,
            handletextpad=0.4,
        )

        for legobj in legend.legend_handles:
            legobj.set_linewidth(10.0)
        #spine_color = 'blue'
        # for spine in ax1.spines.values():
        #     spine.set_linewidth(5)
        #     spine.set_edgecolor(spine_color)
        # for spine in ax2.spines.values():
        #     spine.set_linewidth(5)
        #     spine.set_edgecolor(spine_color)
        # save file in the directory
        plt.savefig(self.dirname + "/profile.png", bbox_inches="tight")
        plt.close()

    def plot_things(self, sid: SimInputData):
        plt.figure(figsize = (15, 10))
        x = sid.cb_in * np.array(self.t) * sid.Q_in / (2 * sid.Da * sid.ne * sid.phi / (1 - sid.phi))
        plt.title('Conductivity')
        plt.grid()
        plt.plot(x, self.pressure[0] / self.pressure, linewidth = 5, color = 'black')
        plt.yscale('log')
        plt.xlabel(r'injected B $\nu_\text{A} / V^0_\text{A}$', fontsize = 50)
        #plt.subplots_adjust(wspace=0, hspace=0)
        plt.margins(tight = True)
        plt.ylabel(r'$K / K_0$', fontsize = 50)
        plt.savefig(self.dirname + '/conductivity.png', bbox_inches="tight")
        plt.close()
        plt.figure(figsize = (15, 10))
        plt.title('Porosity')
        plt.grid()
        plt.plot(x, self.porosity, linewidth = 5, color = 'black')
        plt.xlabel(r'injected B $\nu_\text{A} / V^0_\text{A}$', fontsize = 50)
        #plt.subplots_adjust(wspace=0, hspace=0)
        plt.margins(tight = True)
        plt.ylabel(r'$\varphi$', fontsize = 50)
        plt.yscale('log')
        plt.savefig(self.dirname + '/porosity.png', bbox_inches="tight")
        plt.close()
        plt.figure(figsize = (15, 10))
        plt.title('Mineral evolution')
        plt.plot(x, self.dissolved_v_list, linewidth = 5, color = 'black', label = 'A')
        # plt.xlabel(r'$\Delta V_A / V^\text{init}_A$', fontsize = 50)
        # #plt.subplots_adjust(wspace=0, hspace=0)
        # plt.margins(tight = True)
        # plt.ylabel(r'injected B', fontsize = 50)
        # plt.savefig(self.dirname + '/replaced.png', bbox_inches="tight")
        # plt.close()
        # plt.figure(figsize = (15, 10))
        # plt.title('Replaced volume')
        plt.grid()
        plt.plot(x, self.replaced, '--', linewidth = 5, color = 'black', label = 'E')
        plt.ylabel(r'$\Delta V / V^\text{tot}$', fontsize = 50)
        #plt.subplots_adjust(wspace=0, hspace=0)
        plt.margins(tight = True)
        plt.legend()
        plt.xlabel(r'injected B $\nu_\text{A} / V^0_\text{A}$', fontsize = 50)
        plt.savefig(self.dirname + '/replaced.png', bbox_inches="tight")
        plt.close()
        plt.figure(figsize = (15, 10))
        plt.title('Reacted D')
        plt.plot(sid.cb_in * np.array(self.t) * sid.Q_in / (2 * sid.Da * sid.ne * sid.phi / (1 - sid.phi)), np.array(self.delta_d_list) / (2 * sid.Da * sid.ne * sid.phi / (1 - sid.phi)), linewidth = 5, color = 'black')
        plt.grid()
        plt.ylabel(r'reacted  D $\nu_\text{A} / V^0_\text{A}$', fontsize = 50)
        #plt.subplots_adjust(wspace=0, hspace=0)
        plt.margins(tight = True)
        plt.xlabel(r'injected B $\nu_\text{A} / V^0_\text{A}$', fontsize = 50)
        plt.savefig(self.dirname + '/reacted_d.png', bbox_inches="tight")
        plt.close()

    def check_porosity_profile(self, graph: Graph, inc: Incidence, edges: Edges, triangles: Triangles, \
        vols: Volumes, slice_x: float) -> tuple[int, float]:
        
        pos_x = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,0]
        slice_triangles = ((triangles.node_incidence @ (pos_x >= slice_x)) \
            * (triangles.node_incidence @ (pos_x <= slice_x))) > 0
        slice_porosity = 1 - (np.sum(slice_triangles * (vols.vol_a + vols.vol_e))) / (np.sum(slice_triangles * vols.vol_max))
        slice_vol_a = (np.sum(slice_triangles * vols.vol_a)) / (np.sum(slice_triangles * vols.vol_max))
        slice_vol_e = (np.sum(slice_triangles * vols.vol_e)) / (np.sum(slice_triangles * vols.vol_max))
        return slice_porosity, np.sum(slice_triangles), slice_vol_a, slice_vol_e

    def check_slice_porosity(self, graph: Graph, inc: Incidence, \
        edges: Edges, triangles: Triangles, vols: Volumes, time) -> None:
        pos_x = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,0]
        slices = np.linspace(np.min(pos_x), np.max(pos_x), 102)[1:-1]
        channels_tab, vol_a_tab, vol_e_tab = [], [], []
        for x in slices:
            res = self.check_porosity_profile(graph, inc, edges, triangles, vols, x)
            channels_tab.append(res[0])
            vol_a_tab.append(res[2])
            vol_e_tab.append(res[3])
        self.slice_times.append(time)
        self.slices_phi.append(channels_tab)
        self.slices_a.append(vol_a_tab)
        self.slices_e.append(vol_e_tab)

    def plot_porosity_profile(self, graph: Graph) -> None:
        """ Plots slice data from text file.

        This function loads the data from text file slices.txt and plots them
        to files slices.png, slices_no_div.png, slices_norm.png.
        """
        pos_x = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:,0]
        # slices = np.linspace(np.min(pos_x), np.max(pos_x), 120)[10:-10]
        slices = np.linspace(np.min(pos_x), np.max(pos_x), 102)[1:-1]
        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
        plt.figure(figsize = (15, 10))
        for i, channeling in enumerate(self.slices_phi):
            plt.plot(slices, self.slices_phi[i], label = self.slice_times[i], color = colors[i], linewidth = 5)
        #plt.ylim(0, 1.05)
        plt.xlabel('x', fontsize = 60, style = 'italic')
        # ax2.xaxis.label.set_color('white')
        # ax2.tick_params(axis = 'x', colors='white')
        #plt.xticks([],[])
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.margins(tight = True)
        plt.ylabel('porosity', fontsize = 50)
        #plt.yticks([],[])
        plt.yticks([0, 0.5, 1],['0', '0.5', '1'])
        legend = plt.legend(loc="lower center", mode = "expand", ncol = 4, prop={'size': 40}, handlelength = 1, frameon=False, borderpad = 0, handletextpad = 0.4)
        for legobj in legend.legend_handles:
            legobj.set_linewidth(10.0)
        plt.savefig(self.dirname + "/porosity_profile.png", bbox_inches="tight")
        plt.close()

    def plot_vol_profile(self, graph: Graph) -> None:
        """Plot profiles of phi, V_A and V_E in 3 subplots with common x-axis."""
        pos_x = np.array(list(nx.get_node_attributes(graph, 'pos').values()))[:, 0]
        slices = np.linspace(np.min(pos_x), np.max(pos_x), 102)[1:-1]

        colors = ['black', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
                'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

        fig, axes = plt.subplots(
            3, 1,
            sharex=True,
            figsize=(15, 12)
        )

        # 1) porosity phi
        ax_phi = axes[0]
        for i, phi in enumerate(self.slices_phi):
            ax_phi.plot(
                slices,
                phi,
                color=colors[i],
                linewidth=5
            )
        ax_phi.set_ylabel(r'$\varphi$', fontsize = 50)
        ax_phi.set_ylim(0, 2 * np.max(self.slices_phi[0]))

        # legend = ax_phi.legend(
        #     loc="lower center",
        #     mode="expand",
        #     ncol=4,
        #     prop={'size': 40},
        #     handlelength=1,
        #     frameon=False,
        #     borderpad=0,
        #     handletextpad=0.4
        # )
        # for legobj in legend.legend_handles:
        #     legobj.set_linewidth(10.0)

        # 2) volume A (V_A)
        ax_a = axes[1]
        for i, va in enumerate(self.slices_a):
            ax_a.plot(
                slices,
                va,
                color=colors[i],
                linewidth=5
            )
        ax_a.set_ylabel(r'$V_\text{A}$ / $V^\text{tot}$', fontsize=50)
        ax_a.set_ylim(0, 1.05)
        ax_a.set_yticks([0, 0.5],['0', '0.5'])

        # 3) volume E (V_E)
        ax_e = axes[2]
        # ax_e.plot([], [], ' ', label=' ')
        # ax_e.plot([], [], ' ', label=' ')
        # ax_e.plot([], [], ' ', label=' ')
        for i, ve in enumerate(self.slices_e):
            ax_e.plot(
                slices,
                ve,
                label=self.slice_times[i],
                color=colors[i],
                linewidth=5
            )
        ax_e.set_ylabel(r'$V_\text{E}$ / $V^\text{tot}$', fontsize=50)
        ax_e.set_xlabel('x', fontsize=60, style='italic')
        ax_e.set_ylim(0, 1.05)
        ax_e.set_yticks([0, 0.5],['0', '0.5'])
        handles, labels = ax_e.get_legend_handles_labels()
        #order = [0,4,1,5,2,6,3,7]
        n = len(handles)

        cols = 4
        rows = 2
        slots = rows * cols

        # Dummy handle/label (invisible entry)
        dummy_handle = Line2D([], [], linestyle='none', marker='', color='none')
        dummy_label = ''

        slot_handles = [dummy_handle] * slots
        slot_labels  = [dummy_label]  * slots

        if n > 0:
            # put label "0" (handles[0]) in top-right corner: row 0, col cols-1
            slot_handles[2 * cols - 2] = handles[0]
            slot_labels[2 * cols - 2]  = labels[0]

        # put the remaining labels (1..n-1) in bottom row, from left to right
        for j, (h, lab) in enumerate(zip(handles[1:], labels[1:])):
            if j >= cols:    # we only support up to 4 "other" labels
                break
            r, c = 1, j      # second row, columns 0..3
            idx = r + c * rows
            slot_handles[idx] = h
            slot_labels[idx]  = lab

        legend = ax_e.legend(
            slot_handles,
            slot_labels,
            loc="upper center",
            mode="expand",
            ncol=cols,
            prop={'size': 40},
            handlelength=1,
            frameon=False,
            borderpad=0,
            handletextpad=0.4,
        )

        for legobj in legend.legend_handles:
            legobj.set_linewidth(10.0)

        # Layout tweaks
        for ax in axes[:-1]:
            ax.tick_params(labelbottom=False)  # hide x tick labels on top two

        fig.subplots_adjust(hspace=0, wspace=0)
        plt.margins(tight=True)

        plt.savefig(self.dirname + "/vol_profiles.png", bbox_inches="tight")
        plt.close()
