import random
import numpy as np
import networkx as nx
import scipy.sparse.linalg as spla
from networkx.algorithms.community import modularity, louvain_communities

from .graph_build import lcc_subgraph

def add_dist_attr(G: nx.Graph) -> nx.Graph:
    H = G.copy()
    for _, _, d in H.edges(data=True):
        w = float(d.get("weight", 1.0))
        d["dist"] = 1.0 / w if w > 0 else 1e12
    return H

def approx_weighted_efficiency(G: nx.Graph, sources_k: int, seed: int) -> float:
    N = G.number_of_nodes()
    if N < 2:
        return 0.0

    H = add_dist_attr(G)
    nodes = list(H.nodes())
    rng = random.Random(int(seed))
    k = min(int(sources_k), N)
    sources = nodes if k == N else rng.sample(nodes, k)

    denom = N * (N - 1)
    total = 0.0
    for s in sources:
        dist = nx.single_source_dijkstra_path_length(H, s, weight="dist")
        acc = 0.0
        for t, d in dist.items():
            if t == s:
                continue
            if d > 0 and np.isfinite(d):
                acc += 1.0 / d
        total += acc

    est_full = total * (N / max(1, k))
    return float(est_full / denom)

def spectral_radius_weighted_adjacency(G: nx.Graph) -> float:
    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        return float("nan")
    A = nx.adjacency_matrix(G, weight="weight").astype(float)
    try:
        v = spla.eigsh(A, k=1, which="LM", return_eigenvectors=False)[0]
        return float(v)
    except Exception:
        return float("nan")

def lambda2_connected(G: nx.Graph, eps: float = 1e-10) -> float:
    n = G.number_of_nodes()
    if n < 3 or G.number_of_edges() == 0:
        return float("nan")
    if not nx.is_connected(G):
        return 0.0

    L = nx.laplacian_matrix(G, weight="weight").astype(float)

    if n <= 500:
        vals = np.linalg.eigvalsh(L.toarray())
        vals = np.sort(vals)
        for v in vals:
            if v > eps:
                return float(v)
        return 0.0

    try:
        vals = spla.eigsh(L, k=6, sigma=0.0, which="LM", return_eigenvectors=False)
        vals = np.sort(np.real(vals))
        for v in vals:
            if v > eps:
                return float(v)
        return 0.0
    except Exception:
        try:
            vals = spla.eigsh(L, k=6, which="SM", return_eigenvectors=False)
            vals = np.sort(np.real(vals))
            for v in vals:
                if v > eps:
                    return float(v)
            return 0.0
        except Exception:
            return float("nan")

def lambda2_global(G: nx.Graph) -> float:
    if G.number_of_nodes() < 3 or G.number_of_edges() == 0:
        return float("nan")
    if not nx.is_connected(G):
        return 0.0
    return lambda2_connected(G)

def lambda2_lcc(G: nx.Graph) -> float:
    H = lcc_subgraph(G)
    if H.number_of_nodes() < 3 or H.number_of_edges() == 0:
        return float("nan")
    if not nx.is_connected(H):
        return 0.0
    return lambda2_connected(H)

def modularity_louvain(G: nx.Graph, seed: int) -> float:
    if G.number_of_nodes() < 3 or G.number_of_edges() == 0:
        return float("nan")
    try:
        comm = louvain_communities(G, weight="weight", seed=int(seed))
        return float(modularity(G, comm, weight="weight"))
    except TypeError:
        try:
            comm = louvain_communities(G, weight="weight")
            return float(modularity(G, comm, weight="weight"))
        except Exception:
            return float("nan")
    except Exception:
        return float("nan")

def beta_cycles(G: nx.Graph) -> int:
    return int(G.number_of_edges() - G.number_of_nodes() + nx.number_connected_components(G))

def lcc_fraction(G: nx.Graph, N0: int) -> float:
    if N0 <= 0 or G.number_of_nodes() == 0:
        return 0.0
    lcc = len(max(nx.connected_components(G), key=len))
    return float(lcc) / float(N0)

def calculate_metrics(G: nx.Graph, eff_sources_k: int, seed: int, compute_heavy: bool = True) -> dict:
    N = G.number_of_nodes()
    E = G.number_of_edges()
    C = nx.number_connected_components(G) if N > 0 else 0

    out = {
        "N": int(N),
        "E": int(E),
        "C": int(C),
        "beta": beta_cycles(G),
    }

    if compute_heavy:
        out["l2_lcc"] = lambda2_lcc(G)
        out["l2_global"] = lambda2_global(G)

        out["Q_global"] = modularity_louvain(G, seed=seed)
        H = lcc_subgraph(G)
        out["Q_lcc"] = modularity_louvain(H, seed=seed) if H.number_of_nodes() >= 3 else float("nan")

        out["eff_w"] = approx_weighted_efficiency(G, sources_k=int(eff_sources_k), seed=int(seed))
        out["lmax"] = spectral_radius_weighted_adjacency(G)
    else:
        out["l2_lcc"] = float("nan")
        out["l2_global"] = float("nan")
        out["Q_global"] = float("nan")
        out["Q_lcc"] = float("nan")
        out["eff_w"] = float("nan")
        out["lmax"] = float("nan")

    return out
