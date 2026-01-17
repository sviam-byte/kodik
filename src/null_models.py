import random
import numpy as np
import networkx as nx

def _sample_weight(emp_weights: list[float], rng: random.Random) -> float:
    if not emp_weights:
        return 1.0
    w = float(emp_weights[rng.randrange(0, len(emp_weights))])
    if not np.isfinite(w) or w <= 0:
        return 1e-12
    return w

def make_er_gnm_like(G: nx.Graph, seed: int = 42) -> nx.Graph:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    nodes = list(G.nodes())
    if n == 0:
        return nx.Graph()

    H = nx.gnm_random_graph(n, m, seed=int(seed))
    mapping = {i: nodes[i] for i in range(n)}
    H = nx.relabel_nodes(H, mapping)

    rng = random.Random(int(seed))
    emp_w = [float(d.get("weight", 1.0)) for _, _, d in G.edges(data=True)]

    for u, v in H.edges():
        H[u][v]["weight"] = _sample_weight(emp_w, rng)
        H[u][v]["confidence"] = 0.0
    return H

def make_configuration_like(G: nx.Graph, seed: int = 42, max_tries: int = 30) -> nx.Graph:
    nodes = list(G.nodes())
    if not nodes:
        return nx.Graph()

    deg_seq = [int(G.degree(n)) for n in nodes]
    emp_w = [float(d.get("weight", 1.0)) for _, _, d in G.edges(data=True)]
    rng = random.Random(int(seed))

    best = None
    best_m = -1

    for t in range(max_tries):
        M = nx.configuration_model(deg_seq, seed=int(seed) + t)
        H = nx.Graph(M)
        H.remove_edges_from(nx.selfloop_edges(H))

        mapping = {i: nodes[i] for i in range(len(nodes))}
        H = nx.relabel_nodes(H, mapping)

        if H.number_of_edges() > best_m:
            best = H
            best_m = H.number_of_edges()

        if abs(H.number_of_edges() - G.number_of_edges()) <= max(1, int(0.02 * G.number_of_edges())):
            break

    if best is None:
        best = nx.Graph()

    for u, v in best.edges():
        best[u][v]["weight"] = _sample_weight(emp_w, rng)
        best[u][v]["confidence"] = 0.0

    return best

def degree_preserving_rewire(G: nx.Graph, p: float, seed: int = 42, tries_scale: int = 20) -> nx.Graph:
    p = float(np.clip(p, 0.0, 1.0))
    H = G.copy()
    m = H.number_of_edges()
    if m < 2 or H.number_of_nodes() < 4 or p <= 0:
        return H

    nswap = int(p * m)
    if nswap <= 0:
        return H
    max_tries = max(100, int(tries_scale * nswap))

    try:
        nx.double_edge_swap(H, nswap=nswap, max_tries=max_tries, seed=int(seed))
    except TypeError:
        nx.double_edge_swap(H, nswap=nswap, max_tries=max_tries)
    return H

def build_null(G_emp: nx.Graph, kind: str, seed: int, mix_p: float = 0.0) -> nx.Graph:
    if kind == "empirical":
        return G_emp.copy()
    if kind == "er_gnm":
        return make_er_gnm_like(G_emp, seed=seed)
    if kind == "configuration":
        return make_configuration_like(G_emp, seed=seed)
    if kind == "rewired_degree":
        return degree_preserving_rewire(G_emp, p=1.0, seed=seed)
    if kind == "mix_rewire":
        return degree_preserving_rewire(G_emp, p=mix_p, seed=seed)
    return G_emp.copy()
