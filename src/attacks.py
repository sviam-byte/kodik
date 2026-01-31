import math
import random
from typing import Optional

import numpy as np
import pandas as pd
import networkx as nx

from .metrics import calculate_metrics, add_dist_attr
from src.utils import as_simple_undirected


def _strength(G: nx.Graph, n) -> float:
    """Weighted strength for a node (sum of incident edge weights)."""
    strength = 0.0
    for _, _, d in G.edges(n, data=True):
        w = d.get("weight", 1.0)
        try:
            strength += float(w)
        except Exception:
            strength += 1.0
    return float(strength)


def _pick_nodes_adaptive(
    H: nx.Graph,
    attack_kind: str,
    k: int,
    rng: np.random.Generator,
) -> Optional[list]:
    """
    Pick k nodes from the CURRENT graph H according to adaptive strategy.
    Returns None for unsupported strategies to fall back to existing code.
    """
    if k <= 0 or H.number_of_nodes() == 0:
        return []

    nodes = list(H.nodes())

    if attack_kind == "random":
        rng.shuffle(nodes)
        return nodes[:k]

    if attack_kind == "low_degree":
        nodes.sort(key=lambda n: H.degree(n))
        return nodes[:k]

    if attack_kind == "weak_strength":
        nodes.sort(key=lambda n: _strength(H, n))
        return nodes[:k]

    return None

# =========================
# Rich-Club helpers
# =========================
def strength_ranking(G: nx.Graph) -> list:
    """Rank nodes by weighted degree (strength)."""
    strength = dict(G.degree(weight="weight"))
    nodes = list(G.nodes())
    nodes_sorted = sorted(nodes, key=lambda n: strength.get(n, 0.0), reverse=True)
    return nodes_sorted


def richclub_top_fraction(G: nx.Graph, rc_frac: float) -> list:
    """Return top fraction of nodes by strength."""
    nodes_sorted = strength_ranking(G)
    if not nodes_sorted:
        return []
    k = max(1, int(len(nodes_sorted) * float(rc_frac)))
    return nodes_sorted[:k]


def richclub_by_density_threshold(G: nx.Graph, min_density: float, max_frac: float) -> list:
    """Return the largest prefix with induced density above threshold."""
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
        dens = nx.density(H)
        if dens >= float(min_density):
            best = club
    return best


def pick_targets_for_attack(
    G: nx.Graph,
    attack_kind: str,
    step_size: int,
    seed: int,
    rc_frac: float,
    rc_min_density: float,
    rc_max_frac: float,
) -> list:
    """Select nodes to remove per attack strategy."""
    nodes = list(G.nodes())
    if not nodes:
        return []

    rng = random.Random(int(seed))

    if attack_kind == "random":
        k = min(len(nodes), step_size)
        return rng.sample(nodes, k)

    if attack_kind == "degree":
        strength = dict(G.degree(weight="weight"))
        return sorted(nodes, key=lambda n: strength.get(n, 0.0), reverse=True)[:step_size]

    if attack_kind == "betweenness":
        H = add_dist_attr(G)
        n = H.number_of_nodes()
        # Aggressive sampling: k ~= sqrt(n) capped for speed on large graphs.
        k_samples = min(int(math.sqrt(n)) + 1, 100, n)
        bc = nx.betweenness_centrality(H, k=k_samples, weight="dist", normalized=True, seed=int(seed))
        return sorted(nodes, key=lambda n: bc.get(n, 0.0), reverse=True)[:step_size]

    if attack_kind == "kcore":
        try:
            core = nx.core_number(G)
        except Exception:
            core = {n: 0 for n in nodes}
        return sorted(nodes, key=lambda n: core.get(n, 0), reverse=True)[:step_size]

    if attack_kind == "richclub_top":
        club = richclub_top_fraction(G, rc_frac=rc_frac)
        if not club:
            return []
        return club[:min(step_size, len(club))]

    if attack_kind == "richclub_density":
        club = richclub_by_density_threshold(G, min_density=rc_min_density, max_frac=rc_max_frac)
        if not club:
            return []
        return club[:min(step_size, len(club))]

    return []


def lcc_fraction(G: nx.Graph, N0: int) -> float:
    """Compute fraction of nodes in the largest connected component."""
    if G.number_of_nodes() == 0 or N0 <= 0:
        return 0.0
    lcc = len(max(nx.connected_components(G), key=len))
    return float(lcc) / float(N0)

# =========================
# Attack simulation
# =========================
def run_attack(
    G_in: nx.Graph,
    attack_kind: str,
    remove_frac: float,
    steps: int,
    seed: int,
    eff_sources_k: int,
    rc_frac: float = 0.10,
    rc_min_density: float = 0.30,
    rc_max_frac: float = 0.30,
    compute_heavy_every: int = 1,
    keep_states: bool = False,
):
    """
    Возвращает:
      df_hist: stepwise metrics
      aux: dict with 'removed_nodes' list (critical for UI) and optionally 'states'
    """
    attack_kind = str(attack_kind)
    G_in = as_simple_undirected(G_in)
    
    # -----------------------------------------------------
    # BRANCH 1: Adaptive attacks (weak nodes / low degree)
    # -----------------------------------------------------
    if attack_kind in ("low_degree", "weak_strength"):
        H0 = G_in
        N0 = H0.number_of_nodes()
        if N0 < 2:
            return pd.DataFrame(), {"removed_nodes": [], "states": []}

        rng = np.random.default_rng(int(seed))
        H = H0.copy()
        states = []
        removed_nodes = []

        total_remove = int(N0 * float(remove_frac))
        total_remove = max(0, min(total_remove, N0))
        steps_count = max(1, int(steps))
        ks = np.linspace(0, total_remove, steps_count + 1).round().astype(int).tolist()

        history = []
        removed_total = 0

        for step in range(steps_count):
            if H.number_of_nodes() < 2:
                break

            if keep_states:
                states.append(H.copy())

            heavy = (step % max(1, int(compute_heavy_every)) == 0)
            if heavy:
                met = calculate_metrics(
                    H,
                    eff_sources_k=int(eff_sources_k),
                    seed=int(seed),
                    compute_curvature=False,
                )
            else:
                met = {
                    "N": H.number_of_nodes(),
                    "E": H.number_of_edges(),
                    "C": nx.number_connected_components(H) if H.number_of_nodes() else 0,
                    "density": nx.density(H) if H.number_of_nodes() > 1 else 0.0,
                    "avg_degree": (2 * H.number_of_edges() / H.number_of_nodes()) if H.number_of_nodes() else 0.0,
                    "lcc_size": len(max(nx.connected_components(H), key=len)) if H.number_of_nodes() else 0,
                    "lcc_frac": 0.0,
                }

            met["step"] = int(step)
            met["nodes_left"] = int(H.number_of_nodes())
            met["removed_total"] = int(removed_total)
            met["removed_frac"] = float(removed_total / max(1, N0))
            met["lcc_frac"] = lcc_fraction(H, N0)

            history.append(met)

            next_k = ks[step + 1]
            delta = int(next_k - removed_total)
            if delta > 0:
                picked = _pick_nodes_adaptive(H, attack_kind, delta, rng)
                if picked is None:
                    break
                H.remove_nodes_from(picked)
                removed_nodes.extend(picked)
                removed_total += len(picked)

            if removed_total >= total_remove:
                break

        if keep_states and H.number_of_nodes() > 0:
            states.append(H.copy())

        df_hist = pd.DataFrame(history)
        aux = {"removed_nodes": removed_nodes, "mode": "adaptive", "states": states}
        return df_hist, aux

    # -----------------------------------------------------
    # BRANCH 2: Standard attacks (centrality / random)
    # -----------------------------------------------------
    G_curr = G_in.copy()
    N0 = G_curr.number_of_nodes()
    if N0 < 2:
        return pd.DataFrame(), {"removed_nodes": [], "states": []}

    total_remove = int(N0 * float(remove_frac))
    step_size = max(1, total_remove // int(steps))

    rng = random.Random(int(seed))
    history = []
    states = []
    removed_nodes_all = [] # <--- ВАЖНО: накапливаем удаленные узлы

    removed_total = 0

    for step in range(int(steps)):
        if G_curr.number_of_nodes() < 2:
            break

        if keep_states:
            states.append(G_curr.copy())

        heavy = (step % max(1, int(compute_heavy_every)) == 0)

        if heavy:
            met = calculate_metrics(
                G_curr,
                eff_sources_k=int(eff_sources_k),
                seed=int(seed),
                compute_curvature=False,
            )
        else:
            met = {
                "N": G_curr.number_of_nodes(),
                "E": G_curr.number_of_edges(),
                "C": nx.number_connected_components(G_curr) if G_curr.number_of_nodes() else 0,
                "density": nx.density(G_curr) if G_curr.number_of_nodes() > 1 else 0.0,
                "avg_degree": (2 * G_curr.number_of_edges() / G_curr.number_of_nodes()) if G_curr.number_of_nodes() else 0.0,
                "lcc_size": len(max(nx.connected_components(G_curr), key=len)) if G_curr.number_of_nodes() else 0,
                "lcc_frac": 0.0,
            }

        met["step"] = int(step)
        met["nodes_left"] = int(G_curr.number_of_nodes())
        met["removed_total"] = int(removed_total)
        met["removed_frac"] = float(removed_total / max(1, N0))
        met["lcc_frac"] = lcc_fraction(G_curr, N0)

        history.append(met)

        # Pick targets
        targets = pick_targets_for_attack(
            G_curr,
            attack_kind=attack_kind,
            step_size=step_size,
            seed=int(seed) + step,
            rc_frac=float(rc_frac),
            rc_min_density=float(rc_min_density),
            rc_max_frac=float(rc_max_frac),
        )

        if not targets:
            nodes = list(G_curr.nodes())
            k = min(len(nodes), step_size)
            targets = rng.sample(nodes, k) if k > 0 else []

        G_curr.remove_nodes_from(targets)
        removed_nodes_all.extend(targets) # <--- ВАЖНО: сохраняем порядок
        removed_total += len(targets)

        if removed_total >= total_remove:
            break

    if keep_states and G_curr.number_of_nodes() > 0:
        states.append(G_curr.copy())

    df_hist = pd.DataFrame(history)
    
    # Возвращаем словарь с removed_nodes, как того ожидает app.py
    aux = {"removed_nodes": removed_nodes_all, "mode": "standard", "states": states}
    return df_hist, aux
