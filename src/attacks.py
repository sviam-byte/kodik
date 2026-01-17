import random
import numpy as np
import networkx as nx

from .metrics import add_dist_attr

def strength_dict(G: nx.Graph) -> dict:
    return dict(G.degree(weight="weight"))

def strength_ranking(G: nx.Graph) -> list:
    s = strength_dict(G)
    return sorted(G.nodes(), key=lambda n: s.get(n, 0.0), reverse=True)

def richclub_top_fraction(G: nx.Graph, rc_frac: float) -> list:
    nodes_sorted = strength_ranking(G)
    if not nodes_sorted:
        return []
    k = max(1, int(len(nodes_sorted) * float(rc_frac)))
    return nodes_sorted[:k]

def richclub_by_density_threshold(G: nx.Graph, min_density: float, max_frac: float) -> list:
    nodes_sorted = strength_ranking(G)
    n = len(nodes_sorted)
    if n == 0:
        return []
    if n < 3:
        return nodes_sorted

    maxK = max(3, int(n * float(max_frac)))
    maxK = min(maxK, n)

    best = nodes_sorted[:3]
    for K in range(3, maxK + 1):
        club = nodes_sorted[:K]
        H = G.subgraph(club)
        if nx.density(H) >= float(min_density):
            best = club
    return best

def rc_seams_targets(G: nx.Graph, club: list, step_size: int) -> list:
    club_set = set(club)
    if not club:
        return []

    score = {}
    for v in G.nodes():
        if v in club_set:
            continue
        s = 0.0
        for u in club:
            if G.has_edge(v, u):
                s += float(G[v][u].get("weight", 1.0))
        if s > 0:
            score[v] = s

    if not score:
        return []
    return sorted(score.keys(), key=lambda x: score[x], reverse=True)[:step_size]

def stealth_targets(G: nx.Graph, step_size: int, seed: int, k_samples: int = 200, eps: float = 1e-9) -> list:
    nodes = list(G.nodes())
    if not nodes:
        return []
    H = add_dist_attr(G)
    n = H.number_of_nodes()
    k = min(int(k_samples), n)

    try:
        bc = nx.betweenness_centrality(H, k=k, weight="dist", normalized=True, seed=int(seed))
    except TypeError:
        bc = nx.betweenness_centrality(H, k=k, weight="dist", normalized=True)

    s = strength_dict(G)
    score = {v: float(bc.get(v, 0.0)) / (float(s.get(v, 0.0)) + eps) for v in nodes}
    return sorted(nodes, key=lambda v: score.get(v, 0.0), reverse=True)[:step_size]

def pick_targets_for_attack(
    G: nx.Graph,
    attack_kind: str,
    step_size: int,
    seed: int,
    rc_frac: float,
    rc_min_density: float,
    rc_max_frac: float,
) -> list:
    nodes = list(G.nodes())
    if not nodes:
        return []

    rng = random.Random(int(seed))

    if attack_kind == "random":
        k = min(len(nodes), step_size)
        return rng.sample(nodes, k)

    if attack_kind == "strength":
        s = strength_dict(G)
        return sorted(nodes, key=lambda n: s.get(n, 0.0), reverse=True)[:step_size]

    if attack_kind == "betweenness":
        H = add_dist_attr(G)
        n = H.number_of_nodes()
        k_samples = min(200, n)
        try:
            bc = nx.betweenness_centrality(H, k=k_samples, weight="dist", normalized=True, seed=int(seed))
        except TypeError:
            bc = nx.betweenness_centrality(H, k=k_samples, weight="dist", normalized=True)
        return sorted(nodes, key=lambda n: bc.get(n, 0.0), reverse=True)[:step_size]

    if attack_kind == "kcore":
        try:
            core = nx.core_number(G)
        except Exception:
            core = {n: 0 for n in nodes}
        return sorted(nodes, key=lambda n: core.get(n, 0), reverse=True)[:step_size]

    if attack_kind == "richclub_top":
        club = richclub_top_fraction(G, rc_frac=rc_frac)
        return club[:min(step_size, len(club))]

    if attack_kind == "richclub_density":
        club = richclub_by_density_threshold(G, min_density=rc_min_density, max_frac=rc_max_frac)
        return club[:min(step_size, len(club))]

    if attack_kind == "richclub_seams_top":
        club = richclub_top_fraction(G, rc_frac=rc_frac)
        return rc_seams_targets(G, club, step_size)

    if attack_kind == "richclub_seams_density":
        club = richclub_by_density_threshold(G, min_density=rc_min_density, max_frac=rc_max_frac)
        return rc_seams_targets(G, club, step_size)

    if attack_kind == "stealth":
        return stealth_targets(G, step_size=step_size, seed=int(seed))

    return []
