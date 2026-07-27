import re
from pathlib import Path
from collections import defaultdict

import networkx as nx
import numpy as np
from network import Graph
import scipy.sparse as spr
import matplotlib.pyplot as plt
import matplotlib
import os
from matplotlib import gridspec
from utils import solve_equation

font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 50}

matplotlib.rc('font', **font)
matplotlib.rcParams['font.family']      = 'Times New Roman'   # text outside $...$
matplotlib.rcParams['mathtext.fontset']  = 'stix'
#matplotlib.rcParams['mathtext.default']  = 'rm'     # <- no italics  

def load_graph(name):
    graph = Graph.from_json_file_draw(name)
    # copy for saving network exactly same as initial, but with evolving
    # apertures and permeabilities
    # get rid of additional edges with inf permeability from inlet/outlet node
    # (we keep them in graph_real)
    for edge in graph.edges():
        n1, n2 = edge
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

    remove_nodes = []
    for node in graph.nodes():
        if isinstance(node, str):
            remove_nodes.append(node)
    for node in remove_nodes:
        graph.remove_node(node)
    # update parameters in sid based on loaded graph
    lens = nx.get_edge_attributes(graph, 'length').values()
    l0 = 25
    n_edges = len(graph.edges())
    n_nodes = len(graph.nodes())
    graph.in_vec = np.zeros(n_nodes)
    graph.out_vec = np.zeros(n_nodes)
    for node in graph.in_nodes:
        graph.in_vec[node] = 1
    for node in graph.out_nodes:
        graph.out_vec[node] = 1
    # data for standard incidence matrix (ne x nsq)
    data, row, col = [], [], []
    data_in, row_in, col_in = [], [], []
    # vectors of edges parameters (ne)
    apertures, lens, fracture_lens = [], [], []
    # data for matrix keeping connections of only middle nodes (nsq x nsq)
    for i, e in enumerate(graph.edges()):
        n1, n2 = e
        b = graph[n1][n2]['b']
        l = graph[n1][n2]['length']
        #q = graph[n1][n2]['q']
        data.append(-1)
        row.append(i)
        col.append(n1)
        data.append(1)
        row.append(i)
        col.append(n2)
        fracture_lens.append(graph[n1][n2]['area'] / b)
        apertures.append(b)
        lens.append(l)
        if n1 not in graph.in_nodes and n2 in graph.in_nodes:
            data_in.append(1)
            row_in.append(i)
            col_in.append(n1)
            data_in.append(-1)
            row_in.append(i)
            col_in.append(n2)
        elif n1 in graph.in_nodes and n2 not in graph.in_nodes:
            data_in.append(1)
            row_in.append(i)
            col_in.append(n2)
            data_in.append(-1)
            row_in.append(i)
            col_in.append(n1)
    fracture_lens = np.array(fracture_lens)
    fracture_lens /= np.average(fracture_lens)
    lens = np.array(lens) / l0
    apertures = np.array(apertures)
    b0 = sum(apertures) / len(apertures)
    incidence = spr.csr_matrix((data, (row, col)), shape=(n_edges, \
        n_nodes))
    inlet = spr.csr_matrix((data_in, (row_in, col_in)), \
        shape = (n_edges, n_nodes))
    
    return graph, incidence, fracture_lens, b0, l0, lens, inlet

def find_flow(graph, incidence, b0, fracture_lens, lens, inlet, name):
    graph2 = Graph.from_json_file_draw(name)
    apertures = []
    for edge in graph2.edges():
        n1, n2 = edge
        if isinstance(n1, str) or isinstance(n2, str):
            graph2.remove_edge(n1, n2)
    for i, e in enumerate(graph2.edges()):
        n1, n2 = e
        b = graph2[n1][n2]['b']
        apertures.append(b)
    apertures = np.array(apertures) / b0
    p_matrix = incidence.T @ spr.diags(fracture_lens * apertures ** 3 / lens) @ incidence
    p_matrix = p_matrix.multiply((1 - graph.in_vec - graph.out_vec)[:, np.newaxis]) + spr.diags(graph.in_vec + graph.out_vec)
    pressure = solve_equation(p_matrix, graph.in_vec)
    q_in = np.abs(np.sum(fracture_lens * apertures ** 3 \
        / lens * (inlet @ pressure)))
    pressure /= q_in
    flow = apertures ** 3 / lens * (incidence @ pressure)
    return flow

def network_value(path):
    """
    Extract numeric value from names like:
    network_0.00.json, network_1.01.json, network_10.95.json
    """
    m = re.search(r"network_([-+]?\d*\.?\d+)\.json$", Path(path).name)
    if not m:
        raise ValueError(f"Could not parse network value from {path}")
    return float(m.group(1))


def flatten_unique(seq):
    """Flatten a possibly nested sequence of nodes and keep unique order."""
    out = []
    for item in seq:
        if isinstance(item, (list, tuple, np.ndarray)):
            out.extend(item)
        else:
            out.append(item)
    return list(dict.fromkeys(out))


def make_pos2d_from_xyz(graph):
    """
    Build 2D PCA projection from node attributes x,y,z.
    Returns dict: node -> (x2d, y2d)
    """
    xyz = []
    node_ids = []

    for n, attr in graph.nodes.items():
        if all(k in attr for k in ("x", "y", "z")):
            xyz.append([attr["x"], attr["y"], attr["z"]])
            node_ids.append(n)

    xyz = np.asarray(xyz, dtype=float)
    if xyz.size == 0:
        raise ValueError("No nodes with x,y,z attributes found.")

    xyz = xyz - xyz.mean(axis=0)

    # PCA plane from first two right-singular vectors
    _, _, vh = np.linalg.svd(xyz, full_matrices=False)
    plane = vh[:2].T
    proj = xyz @ plane

    #return {node_ids[i]: proj[i] for i in range(len(node_ids))}
    return {node_ids[i]: (xyz[i][0], xyz[i][2]) for i in range(len(node_ids))}


def draw_one_network_with_profile(json_path, profile_path, profile_row_idx, flux_frac=0.0):
    """
    Create a two-panel plot:
      top    - network with edge widths proportional to |flow|
      bottom - flow focusing profile for the selected network row

    profile_row_idx is the row in slices.txt corresponding to this network.
    """
    json_path = Path(json_path)
    profile_path = Path(profile_path)

    # --- load network and flow ---
    graph, incidence, fracture_lens, b0, l0, lens, inlet = load_graph(str(json_path))
    flow = find_flow(graph, incidence, b0, fracture_lens, lens, inlet, str(json_path))

    # attach |flow| to graph edges
    for i, (n1, n2) in enumerate(graph.edges()):
        graph[n1][n2]["q"] = float(np.abs(flow[i]))

    # threshold for drawing
    flux_th = np.max(np.abs(flow)) / flux_frac if flux_frac > 0 else 0.0

    def keep_edge(u, v):
        return graph[u][v]["q"] > flux_th

    edges_to_draw = [(u, v) for u, v in graph.edges() if keep_edge(u, v)]
    q_to_draw = np.array([graph[u][v]["q"] for u, v in edges_to_draw], dtype=float)

    # inlet / outlet nodes
    nodes_to_draw = flatten_unique(getattr(graph, "in_nodes", []) + getattr(graph, "out_nodes", []))
    nodes_to_draw_in = flatten_unique(getattr(graph, "in_nodes", []))
    nodes_to_draw_out = flatten_unique(getattr(graph, "out_nodes", []))

    # --- 2D projection for plotting the network ---
    pos2d = make_pos2d_from_xyz(graph)

    # --- load focusing profile ---
    slices = np.loadtxt(profile_path)
    slices = np.atleast_2d(slices)

    if profile_row_idx >= slices.shape[0]:
        raise IndexError(
            f"{profile_path} has only {slices.shape[0]} rows, "
            f"but row {profile_row_idx} was requested for {json_path.name}"
        )

    edge_number = np.asarray(slices[0], dtype=float)
    initial_channeling = np.asarray(slices[1], dtype=float)
    current_channeling = np.asarray(slices[profile_row_idx], dtype=float)

    # x coordinates for profile
    node_x = np.array([attr["x"] for _, attr in graph.nodes(data=True) if "x" in attr], dtype=float)
    xmin_x = np.min(node_x)
    xmax_x = np.max(node_x)

    n_profile = edge_number.size
    slice_x = np.linspace(xmin_x, xmax_x, n_profile + 2)[1:-1]

    focusing_initial = (edge_number - 2.0 * initial_channeling) / edge_number
    focusing_current = (edge_number - 2.0 * current_channeling) / edge_number

    # --- make figure ---
    fig = plt.figure(figsize=(20, 13))
    #spec = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2.2, 1.0], hspace=0.)

    # top panel: network
    ax1 = fig.add_subplot()
    nx.draw_networkx_edges(
        graph,
        pos2d,
        edgelist=edges_to_draw,
        width=100. * q_to_draw,
        alpha=0.3,
        edge_color="black",
        ax=ax1,
    )
    if nodes_to_draw:
        nx.draw_networkx_nodes(
            graph,
            pos2d,
            nodelist=nodes_to_draw_in,
            node_size=8,
            alpha=1.0,
            node_color="red",
            ax=ax1,
        )
        nx.draw_networkx_nodes(
            graph,
            pos2d,
            nodelist=nodes_to_draw_out,
            node_size=8,
            alpha=1.0,
            node_color="blue",
            ax=ax1,
        )

    pos_arr = np.array(list(pos2d.values()))
    xmin, ymin = pos_arr.min(axis=0)
    xmax, ymax = pos_arr.max(axis=0)
    ax1.set_xlim(xmin - 0.15, xmax + 0.15)
    ax1.set_ylim(ymin, ymax)
    ax1.set_xlabel("x")
    ax1.set_ylabel("z")
    ax1.set_frame_on(True)

    ax1.tick_params(
    bottom=True,
    left=True,
    labelbottom=True,
    labelleft=True
)
    #ax1.set_title(json_path.name, fontsize=14)

    # bottom panel: focusing profile
    # ax2 = fig.add_subplot(spec[1])
    # ax2.plot(slice_x / 25, focusing_initial, linewidth=10, color="black", label="initial")
    # ax2.plot(
    #     slice_x / 25,
    #     focusing_current,
    #     linewidth=10,
    #     color="tab:red",
    #     label=f"current",
    # )
    # ax2.set_ylim(0, 1.05)
    # ax2.set_xlabel(r"$x \, / \, L$")
    # ax2.set_ylabel("flow focusing index")
    # ax2.set_yticks([0, 0.5, 1.0])
    # ax2.legend(frameon=False, loc="lower center", mode = "expand", ncol = 2)

    out_path = json_path.with_suffix(".png")
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Saved: {out_path}")


def plot_all_networks_with_profiles(root_dir, profile_name="slices.txt", flux_frac=10000.0):
    """
    Search recursively for network_*.json under root_dir.
    For each directory:
      - sort network files by the numeric suffix
      - load sibling slices.txt
      - for the i-th network use row 2+i from slices.txt
      - save as network_*.png
    """
    root_dir = Path(root_dir)

    grouped = defaultdict(list)
    for json_path in root_dir.rglob("network_*.json"):
        grouped[json_path.parent].append(json_path)

    if not grouped:
        print(f"No network_*.json files found under {root_dir}")
        return

    for folder, json_files in grouped.items():
        json_files = sorted(json_files, key=network_value)
        profile_path = folder / profile_name

        if not profile_path.exists():
            print(f"Skipping {folder}: missing {profile_path.name}")
            continue

        slices = np.loadtxt(profile_path)
        slices = np.atleast_2d(slices)

        # rows: 0=edge_number, 1=initial, 2... = network profiles
        n_available = max(0, slices.shape[0] - 2)
        if len(json_files) > n_available:
            print(
                f"Warning: {folder} has {len(json_files)} network files but only "
                f"{n_available} profile rows in {profile_path.name}"
            )

        for i, json_path in enumerate(json_files):
            profile_row_idx = 1 + i
            if profile_row_idx >= slices.shape[0]:
                print(f"Skipping {json_path.name}: no corresponding row {profile_row_idx} in {profile_path.name}")
                continue

            draw_one_network_with_profile(
                json_path=json_path,
                profile_path=profile_path,
                profile_row_idx=profile_row_idx,
                flux_frac=flux_frac,
            )

#draw_one_network_with_profile("oman_dfn_v1/G5.0000Daeff0.0020/0/network_0.00.json", "oman_dfn_v1/G5.0000Daeff0.0020/0/slices.txt", 5)

# plot_all_networks_with_profiles(
#     root_dir="samples",
#     profile_name="slices.txt",
#     flux_frac=100.0,
# )

from mpl_toolkits.mplot3d.art3d import Line3DCollection


def make_pos3d_from_xyz(graph):
    """
    Build 3D positions from node attributes x,y,z.
    Returns dict: node -> (x, y, z)
    """
    pos3d = {}

    for n, attr in graph.nodes(data=True):
        if all(k in attr for k in ("x", "y", "z")):
            pos3d[n] = (
                float(attr["x"]),
                float(attr["y"]),
                float(attr["z"]),
            )

    if not pos3d:
        raise ValueError("No nodes with x,y,z attributes found.")

    return pos3d

def draw_3d(json_path, flux_frac):
    json_path = Path(json_path)


    # --- load network and flow ---
    graph, incidence, fracture_lens, b0, l0, lens, inlet = load_graph(str(json_path))
    flow = find_flow(graph, incidence, b0, fracture_lens, lens, inlet, str(json_path))

    # attach |flow| to graph edges
    for i, (n1, n2) in enumerate(graph.edges()):
        graph[n1][n2]["q"] = float(np.abs(flow[i]))

    # threshold for drawing
    flux_th = np.max(np.abs(flow)) / flux_frac if flux_frac > 0 else 0.0

    def keep_edge(u, v):
        return graph[u][v]["q"] > flux_th

    edges_to_draw = [(u, v) for u, v in graph.edges() if keep_edge(u, v)]
    q_to_draw = np.array([graph[u][v]["q"] for u, v in edges_to_draw], dtype=float)

    # inlet / outlet nodes
    nodes_to_draw = flatten_unique(getattr(graph, "in_nodes", []) + getattr(graph, "out_nodes", []))
    nodes_to_draw_in = flatten_unique(getattr(graph, "in_nodes", []))
    nodes_to_draw_out = flatten_unique(getattr(graph, "out_nodes", []))


    # --- 3D positions for plotting the network ---
    pos3d = make_pos3d_from_xyz(graph)

    # --- make figure ---
    fig = plt.figure(figsize=(20, 13))
    ax1 = fig.add_subplot(111, projection="3d")

    # Build 3D edge segments
    segments = []
    for u, v in edges_to_draw:
        if u in pos3d and v in pos3d:
            segments.append([pos3d[u], pos3d[v]])

    # Normalize edge widths so they look reasonable
    if len(q_to_draw) > 0 and np.max(q_to_draw) > 0:
        edge_widths = 0.0005 + 3 * q_to_draw / np.max(q_to_draw)
    else:
        edge_widths = 0.0005

    # Draw edges
    edge_collection = Line3DCollection(
        segments,
        linewidths=edge_widths,
        colors="black",
        alpha=0.3,
    )

    ax1.add_collection3d(edge_collection)

    # Draw inlet nodes
    if nodes_to_draw_in:
        pts_in = np.array([pos3d[n] for n in nodes_to_draw_in if n in pos3d])
        if len(pts_in) > 0:
            ax1.scatter(
                pts_in[:, 0],
                pts_in[:, 1],
                pts_in[:, 2],
                s=20,
                c="red",
                alpha=1.0,
                label="inlet",
            )

    # Draw outlet nodes
    if nodes_to_draw_out:
        pts_out = np.array([pos3d[n] for n in nodes_to_draw_out if n in pos3d])
        if len(pts_out) > 0:
            ax1.scatter(
                pts_out[:, 0],
                pts_out[:, 1],
                pts_out[:, 2],
                s=20,
                c="blue",
                alpha=1.0,
                label="outlet",
            )

    # Set limits
    pos_arr = np.array(list(pos3d.values()))
    xmin, ymin, zmin = pos_arr.min(axis=0)
    xmax, ymax, zmax = pos_arr.max(axis=0)

    pad_x = 0.05 * (xmax - xmin) if xmax > xmin else 1
    pad_y = 0.05 * (ymax - ymin) if ymax > ymin else 1
    pad_z = 0.05 * (zmax - zmin) if zmax > zmin else 1

    ax1.set_xlim(xmin - pad_x, xmax + pad_x)
    ax1.set_ylim(ymin - pad_y, ymax + pad_y)
    ax1.set_zlim(zmin - pad_z, zmax + pad_z)

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")

    # Keeps aspect ratio closer to real geometry
    ax1.set_box_aspect((
        xmax - xmin if xmax > xmin else 1,
        ymax - ymin if ymax > ymin else 1,
        zmax - zmin if zmax > zmin else 1,
    ))

    # Choose viewing angle
    ax1.view_init(elev=20, azim=-60)

    ax1.tick_params(
        axis="both",
        which="major",
        labelsize=10,
    )

    ax1.legend(frameon=False)

    out_path = json_path.with_suffix(".png")
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Saved: {out_path}")

def draw_3d_small(json_path, flux_frac):
    json_path_path = Path(json_path)


    # --- load network and flow ---
    graph, incidence, fracture_lens, b0, l0, lens, inlet = load_graph(str(json_path))
    flow = find_flow(graph, incidence, b0, fracture_lens, lens, inlet, str(json_path))

    # attach |flow| to graph edges
    for i, (n1, n2) in enumerate(graph.edges()):
        graph[n1][n2]["q"] = float(np.abs(flow[i]))

    # threshold for drawing
    flux_th = np.max(np.abs(flow)) / flux_frac if flux_frac > 0 else 0.0

    def keep_edge(u, v):
        return graph[u][v]["q"] > flux_th

    edges_to_draw = [(u, v) for u, v in graph.edges() if keep_edge(u, v)]
    q_to_draw = np.array([graph[u][v]["q"] for u, v in edges_to_draw], dtype=float)

    # inlet / outlet nodes
    nodes_to_draw = flatten_unique(getattr(graph, "in_nodes", []) + getattr(graph, "out_nodes", []))
    nodes_to_draw_in = flatten_unique(getattr(graph, "in_nodes", []))
    nodes_to_draw_out = flatten_unique(getattr(graph, "out_nodes", []))


    # --- 3D positions for plotting the network ---
    pos3d = make_pos3d_from_xyz(graph)

    # --- make figure ---
    fig = plt.figure(figsize=(20, 13))
    ax1 = fig.add_subplot(111, projection="3d")

    # Build 3D edge segments
    segments = []
    for u, v in edges_to_draw:
        if u in pos3d and v in pos3d:
            segments.append([pos3d[u], pos3d[v]])

    # Normalize edge widths so they look reasonable
    if len(q_to_draw) > 0 and np.max(q_to_draw) > 0:
        edge_widths = 0.0005 + 3 * q_to_draw / np.max(q_to_draw)
    else:
        edge_widths = 0.0005

    # Draw edges
    edge_collection = Line3DCollection(
        segments,
        linewidths=edge_widths,
        colors="black",
        alpha=0.3,
    )

    ax1.add_collection3d(edge_collection)

    # Draw inlet nodes
    if nodes_to_draw_in:
        pts_in = np.array([pos3d[n] for n in nodes_to_draw_in if n in pos3d])
        if len(pts_in) > 0:
            ax1.scatter(
                pts_in[:, 0],
                pts_in[:, 1],
                pts_in[:, 2],
                s=20,
                c="red",
                alpha=1.0,
                label="inlet",
            )

    # Draw outlet nodes
    if nodes_to_draw_out:
        pts_out = np.array([pos3d[n] for n in nodes_to_draw_out if n in pos3d])
        if len(pts_out) > 0:
            ax1.scatter(
                pts_out[:, 0],
                pts_out[:, 1],
                pts_out[:, 2],
                s=20,
                c="blue",
                alpha=1.0,
                label="outlet",
            )

    # Set limits
    pos_arr = np.array(list(pos3d.values()))
    xmin, ymin, zmin = pos_arr.min(axis=0)
    xmax, ymax, zmax = pos_arr.max(axis=0)

    pad_x = 0.05 * (xmax - xmin) if xmax > xmin else 1
    pad_y = 0.05 * (ymax - ymin) if ymax > ymin else 1
    pad_z = 0.05 * (zmax - zmin) if zmax > zmin else 1

    ax1.set_xlim(-10, 10)
    ax1.set_ylim(-10, 10)
    ax1.set_zlim(zmin - pad_z, zmax + pad_z)

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")

    # Keeps aspect ratio closer to real geometry
    ax1.set_box_aspect((
        xmax - xmin if xmax > xmin else 1,
        ymax - ymin if ymax > ymin else 1,
        zmax - zmin if zmax > zmin else 1,
    ))

    # Choose viewing angle
    ax1.view_init(elev=20, azim=-60)

    ax1.tick_params(
        axis="both",
        which="major",
        labelsize=10,
    )

    ax1.legend(frameon=False)

    out_path = json_path[:-5]+"_small.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Saved: {out_path}")

networks = []
dirname = "oman_dfn_v3/G5.0000Daeff0.0200/2/"
for name in os.listdir(dirname):
    if name[:3] == 'net':
        networks.append(name[8:12])
for net in networks:
    name = dirname + 'network_' + net + '.json'    
    draw_3d(name, 10000)
    draw_3d_small(name, 10000)

# draw_3d("oman_dfn_v2/G5.0000Daeff0.0020/1/network_1.00.json", 10000)
# draw_3d_small("oman_dfn_v2/G5.0000Daeff0.0020/1/network_1.00.json", 10000)