#%%
#!/usr/bin/env python3
"""
Plot a 3-D DFN in 2-D (PCA projection) with optional flux-threshold filtering.

User settings (edit near the top):
    JSON_FILE      – path to NetworkX-compatible JSON
    FLUX_FILE      – CSV/TSV/space file with `src  tgt  flux`
    FLUX_THRESHOLD – only edges with flux ≥ threshold are drawn
"""

# --------------------------------------------------------------------
# 0.  User settings
# --------------------------------------------------------------------
JSON_FILE       = "network_1.00.json"   # graph with 3-D coords
FLUX_FILE       = "edge_flux.txt"       # leave "" if not available
FLUX_THRESHOLD  = 1e-4                  # set to 0 to draw everything

# --------------------------------------------------------------------
# 1.  Imports and loading
# --------------------------------------------------------------------
import json, pathlib
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

print("Reading graph :", JSON_FILE)
with open(pathlib.Path(JSON_FILE), "r") as f:
    data = json.load(f)

G = nx.Graph()
# --- nodes
for nd in data["nodes"]:
    n = nd["id"]
    G.add_node(n, **{k: v for k, v in nd.items() if k != "id"})
# --- edges
for el in data["links"]:
    u, v = el["source"], el["target"]
    G.add_edge(u, v, **{k: v for k, v in el.items()
                        if k not in ("source", "target")})

# --------------------------------------------------------------------
# 2.  If available, read fluxes  ( format:  src  tgt  flux )
# --------------------------------------------------------------------
if FLUX_FILE:
    print("Reading flux file:", FLUX_FILE)
    for line in pathlib.Path(FLUX_FILE).read_text().splitlines():
        if not line.strip():
            continue
        u, v, flx = line.split()
        # cast to int if numeric node labels
        try:
            u, v = int(u), int(v)
        except ValueError:
            pass
        if G.has_edge(u, v):
            G[u][v]["flux"] = float(flx)

# --------------------------------------------------------------------
# 3.  Build a 2-D PCA projection of node coordinates
# --------------------------------------------------------------------
xyz = []
node_ids = []
# many DFN formats store coordinates under 'x','y','z'
for n, attr in G.nodes.items():
    if all(k in attr for k in ("x", "y", "z")):
        xyz.append([attr["x"], attr["y"], attr["z"]])
        node_ids.append(n)
xyz = np.asarray(xyz, dtype=float)

# centre and project to first two principal directions
xyz -= xyz.mean(axis=0)
u, *_ = np.linalg.svd(xyz, full_matrices=False)
plane = u[:, :2]                       # (3×2) PCA basis
proj = xyz @ plane                     # (N×3)·(3×2) → (N×2)

pos2d = {node_ids[i]: proj[i] for i in range(len(node_ids))}

# give small random jitter to nodes that lacked xyz
missing = set(G.nodes()) - set(pos2d)
if missing:
    rng = np.random.default_rng(42)
    jitter = rng.normal(scale=0.01, size=(len(missing), 2))
    for i, n in enumerate(missing):
        pos2d[n] = jitter[i]

# --------------------------------------------------------------------
# 4.  Filter edges by flux if requested
# --------------------------------------------------------------------
def keep_edge(u, v, attr):
    if FLUX_FILE and "flux" in attr:
        return attr["flux"] >= FLUX_THRESHOLD
    return True                         # keep everything

edges_to_draw = [(u, v) for u, v, a in G.edges(data=True) if keep_edge(u, v, a)]
nodes_to_draw = set(sum(edges_to_draw, ()))            # flatten edge pairs

print(f"Edges shown: {len(edges_to_draw)} / {G.number_of_edges()}")

# --------------------------------------------------------------------
# 5.  Plot
# --------------------------------------------------------------------
plt.figure(figsize=(7, 5))
nx.draw_networkx_edges(G, pos2d, edgelist=edges_to_draw,
                       width=0.4, alpha=0.3, edge_color="grey")
nx.draw_networkx_nodes(G, pos2d, nodelist=nodes_to_draw,
                       node_size=8, node_color="black")
plt.axis("off")
title = f"PCA projection – flux ≥ {FLUX_THRESHOLD:g}" if FLUX_FILE else "PCA projection"
plt.title(title, fontsize=12)
plt.tight_layout()
plt.show()
