""" Active surface area fraction vs dissolved volume for dfn_het_apertures.

For a given network snapshot, each edge belongs to exactly one fracture
('frac' attribute) and connects two fracture-intersection nodes, so the flow
carried by an edge is by construction flow exchanged between that fracture
and a neighboring one. Aggregating per fracture f gives a surface S_f (sum of
width * length over the fracture's edges) and an exchanged flow Q_f (sum of
|flow| over the fracture's edges). The active surface fraction is the
participation-ratio-style metric

    (sum_f S_f Q_f)^2 / (sum_f S_f Q_f^2) / (sum_f S_f)

which is 1 when Q_f / S_f is uniform across fractures (all surface area
equally active) and shrinks as flow concentrates onto a small subset of
fractures.

Loading/flow-solve logic mirrors tracking.py's load_graph/find_flow: graph
topology (and b0) is taken from each combination's first snapshot and reused,
since all snapshots in a run share the same topology and only apertures
evolve.
"""
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as spr

from network import Graph
from utils import solve_equation

ROOT_DIR = 'dfn_het_apertures'
L0 = 25.0

SIGMAS = [0.00, 0.10, 0.20, 0.50, 1.00]
GS = [0.1, 1.0, 5.0]
DAEFFS = [0.002, 0.02, 0.2]

# ordinal sequential blue ramp (dataviz skill, steps 250/350/450/550/650),
# light -> dark for increasing sigma
SIGMA_COLORS = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281']


def load_topology(name: str):
    """ Load graph topology, incidence matrices and per-edge fracture ids
    from a single network_*.json snapshot.
    """
    graph = Graph.from_json_file(name)
    for n1, n2 in list(graph.edges()):
        if isinstance(n1, str) or isinstance(n2, str):
            if n1 == 's':
                graph.in_nodes.append(n2)
            elif n1 == 't':
                graph.out_nodes.append(n2)
            if n2 == 's':
                graph.in_nodes.append(n1)
            elif n2 == 't':
                graph.out_nodes.append(n1)
            graph.remove_edge(n1, n2)
    for node in [n for n in graph.nodes() if isinstance(n, str)]:
        graph.remove_node(node)

    n_edges = len(graph.edges())
    n_nodes = len(graph.nodes())
    graph.in_vec = np.zeros(n_nodes)
    graph.out_vec = np.zeros(n_nodes)
    for node in graph.in_nodes:
        graph.in_vec[node] = 1
    for node in graph.out_nodes:
        graph.out_vec[node] = 1

    data, row, col = [], [], []
    data_in, row_in, col_in = [], [], []
    apertures, lens, fracture_lens, frac_id = [], [], [], []
    for i, (n1, n2) in enumerate(graph.edges()):
        edge = graph[n1][n2]
        b = edge['b']
        data += [-1, 1]
        row += [i, i]
        col += [n1, n2]
        fracture_lens.append(edge['area'] / b)
        apertures.append(b)
        lens.append(edge['length'])
        frac_id.append(edge['frac'])
        if n1 not in graph.in_nodes and n2 in graph.in_nodes:
            data_in += [1, -1]
            row_in += [i, i]
            col_in += [n1, n2]
        elif n1 in graph.in_nodes and n2 not in graph.in_nodes:
            data_in += [1, -1]
            row_in += [i, i]
            col_in += [n2, n1]

    fracture_lens = np.array(fracture_lens)
    fracture_lens /= np.average(fracture_lens)
    lens = np.array(lens) / L0
    apertures = np.array(apertures)
    b0 = np.average(apertures)
    frac_id = np.array(frac_id)

    incidence = spr.csr_matrix((data, (row, col)), shape=(n_edges, n_nodes))
    inlet = spr.csr_matrix((data_in, (row_in, col_in)), shape=(n_edges, n_nodes))
    return graph, incidence, fracture_lens, lens, inlet, b0, frac_id


def solve_flow(graph, incidence, fracture_lens, lens, inlet, b0, name):
    """ Solve pressure/flow for a snapshot's apertures, reusing the topology
    (incidence, inlet, in/out node vectors) built by load_topology.
    """
    graph2 = Graph.from_json_file(name)
    for n1, n2 in list(graph2.edges()):
        if isinstance(n1, str) or isinstance(n2, str):
            graph2.remove_edge(n1, n2)
    apertures = np.array([graph2[n1][n2]['b'] for n1, n2 in graph2.edges()]) / b0

    p_matrix = incidence.T @ spr.diags(fracture_lens * apertures ** 3 / lens) @ incidence
    p_matrix = p_matrix.multiply((1 - graph.in_vec - graph.out_vec)[:, np.newaxis]) \
        + spr.diags(graph.in_vec + graph.out_vec)
    pressure = solve_equation(p_matrix, graph.in_vec)
    q_in = np.abs(np.sum(fracture_lens * apertures ** 3 / lens * (inlet @ pressure)))
    pressure /= q_in
    return apertures ** 3 / lens * (incidence @ pressure)


def active_surface_fraction(fracture_lens, lens, frac_id, flow) -> float:
    """ (sum_f S_f Q_f)^2 / (sum_f S_f Q_f^2) / (sum_f S_f). """
    surface = fracture_lens * lens
    n_frac = int(frac_id.max()) + 1
    s_f = np.bincount(frac_id, weights=surface, minlength=n_frac)
    q_f = np.bincount(frac_id, weights=np.abs(flow), minlength=n_frac)
    mask = s_f > 0
    s_f, q_f = s_f[mask], q_f[mask]
    return float(np.sum(s_f * q_f) ** 2 / (np.sum(s_f * q_f ** 2) * np.sum(s_f)))


def parse_dissolved_v(path: str) -> float:
    match = re.search(r'network_([0-9.]+)\.json$', os.path.basename(path))
    return float(match.group(1))


def process_run(dirname: str) -> list[tuple[float, float]]:
    """ Compute (dissolved_v, active_surface_fraction) for every snapshot in
    a single sigma/G/Daeff/0 directory.
    """
    files = sorted(
        glob.glob(os.path.join(dirname, 'network_*.json')),
        key=parse_dissolved_v)
    if not files:
        return []
    graph, incidence, fracture_lens, lens, inlet, b0, frac_id = load_topology(files[0])
    results = []
    for f in files:
        flow = solve_flow(graph, incidence, fracture_lens, lens, inlet, b0, f)
        frac = active_surface_fraction(fracture_lens, lens, frac_id, flow)
        results.append((parse_dissolved_v(f), frac))
    return results


def main():
    data = {}
    for sigma in SIGMAS:
        for g in GS:
            for da_eff in DAEFFS:
                dirname = os.path.join(
                    ROOT_DIR, f'sigma_{sigma:.2f}',
                    f'G{g:.4f}Daeff{da_eff:.4f}', '0')
                if not os.path.isdir(dirname):
                    print(f'[skip] missing directory: {dirname}')
                    continue
                results = process_run(dirname)
                if not results:
                    print(f'[skip] no snapshots in: {dirname}')
                    continue
                data[(sigma, g, da_eff)] = results
                print(f'[done] {dirname}: {len(results)} snapshots')

    fig, axes = plt.subplots(
        len(GS), len(DAEFFS), figsize=(4.2 * len(DAEFFS), 3.6 * len(GS)),
        sharex=True, sharey=True, squeeze=False)

    for row, g in enumerate(GS):
        for col, da_eff in enumerate(DAEFFS):
            ax = axes[row][col]
            for sigma, color in zip(SIGMAS, SIGMA_COLORS):
                results = data.get((sigma, g, da_eff))
                if not results:
                    continue
                t, frac = zip(*results)
                ax.plot(t, 100 * np.array(frac), color=color, linewidth=2,
                    marker='o', markersize=5, label=f'{sigma:.2f}')
            ax.set_title(f'G = {g:g}, Da$_{{eff}}$ = {da_eff:g}', fontsize=11)
            ax.grid(True, color='#e1e0d9', linewidth=0.8)
            ax.set_ylim(0, 100)
            ax.set_xlim(0, 1)
            if row == len(GS) - 1:
                ax.set_xlabel('dissolved volume')
            if col == 0:
                ax.set_ylabel('active surface area (%)')

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, title='sigma', loc='center left',
        bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig('active_surface_fraction.png', dpi=200, bbox_inches='tight')
    print('Saved active_surface_fraction.png')


if __name__ == '__main__':
    main()
