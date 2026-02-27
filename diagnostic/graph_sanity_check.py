import torch
import numpy as np

# ===============================
# 1. Edge Density
# ===============================
def check_density(edge_index, num_nodes, name):
    edges = edge_index.shape[1]
    density = edges / (num_nodes * (num_nodes-1))

    print(f"\n[{name}]")
    print(f"Edges: {edges}")
    print(f"Density: {density:.8f}")

    if density < 1e-6:
        print("WARNING: Graph terlalu sparse -> GNN tidak belajar")
    elif density > 1e-2:
        print("WARNING: Graph terlalu dense -> noise dominan")
    else:
        print("OK: Density masuk range belajar")


# ===============================
# 2. Fraud Homophily
# ===============================
def check_homophily(edge_index, y):
    src = edge_index[0]
    dst = edge_index[1]

    same = (y[src] == y[dst]).sum().item()
    total = edge_index.shape[1]

    ratio = same / total

    print("\n[HOMOPHILY CHECK]")
    print(f"Same-label edges ratio: {ratio:.4f}")

    if ratio < 0.55:
        print("BAD: Fraud tidak berkumpul → GNN susah belajar")
    elif ratio < 0.7:
        print("MEDIUM: Masih bisa tapi lemah")
    else:
        print("GOOD: GNN akan efektif")


# ===============================
# 3. Degree Distribution
# ===============================
def check_degree(edge_index, num_nodes):
    deg = torch.zeros(num_nodes)

    for n in edge_index[0]:
        deg[n] += 1

    print("\n[DEGREE DISTRIBUTION]")
    print("Max degree:", deg.max().item())
    print("Mean degree:", deg.mean().item())

    if deg.max() > 1000:
        print("WARNING: Hub dominated graph")


# ===============================
# 4. Feature Quality
# ===============================
def check_features(x, name):
    print(f"\n[FEATURE CHECK: {name}]")

    if torch.isnan(x).any():
        print("ERROR: Ada NaN")
    else:
        print("OK: No NaN")

    if torch.std(x) < 1e-6:
        print("WARNING: Feature hampir konstan")
    else:
        print("OK: Variance normal")


# ===============================
# MAIN RUNNER
# ===============================
def run_all(edge_original, edge_new, sim_edges, temporal_edges,
            x_struct, x_behavior, y):

    num_nodes = x_struct.shape[0]

    print("\n========== GRAPH SANITY CHECK ==========")

    check_density(edge_original, num_nodes, "Original Graph")
    check_density(edge_new, num_nodes, "Combined Graph")
    check_density(sim_edges, num_nodes, "Similarity Graph")
    check_density(temporal_edges, num_nodes, "Temporal Graph")

    check_homophily(edge_new, y)

    check_degree(edge_new, num_nodes)

    check_features(x_struct, "Structural Features")
    check_features(x_behavior, "Behavior Features")

    print("\n========== CHECK COMPLETE ==========")