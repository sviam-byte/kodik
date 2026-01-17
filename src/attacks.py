import random
import numpy as np
import pandas as pd
import networkx as nx

from .metrics import calculate_metrics, add_dist_attr

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
      states: list of graphs (если keep_states)
    """
    G_curr = G_in.copy()
    N0 = G_curr.number_of_nodes()
    if N0 < 2:
        return pd.DataFrame(), []

    total_remove = int(N0 * float(remove_frac))
    step_size = max(1, total_remove // int(steps))

    rng = random.Random(int(seed))
    history = []
    states = []

    removed_total = 0

    for step in range(int(steps)):
        if G_curr.number_of_nodes() < 2:
            break

        if keep_states:
            states.append(G_curr.copy())

        # metrics: heavy every k, but LCC frac always
        heavy = (step % max(1, int(compute_heavy_every)) == 0)

        if heavy:
            met = calculate_metrics(G_curr, eff_sources_k=int(eff_sources_k), seed=int(seed))
        else:
            # cheap metrics
            met = {
                "N": G_curr.number_of_nodes(),
                "E": G_curr.number_of_edges(),
                "C": nx.number_connected_components(G_curr) if G_curr.number_of_nodes() else 0,
                "eff_w": np.nan,
                "l2_lcc": np.nan,
                "tau_lcc": np.nan,
                "lmax": np.nan,
                "thresh": np.nan,
                "mod": np.nan,
                "density": nx.density(G_curr) if G_curr.number_of_nodes() > 1 else 0.0,
                "avg_degree": (2 * G_curr.number_of_edges() / G_curr.number_of_nodes()) if G_curr.number_of_nodes() else 0.0,
                "beta": int(G_curr.number_of_edges() - G_curr.number_of_nodes() + nx.number_connected_components(G_curr)) if G_curr.number_of_nodes() else 0,
                "lcc_size": len(max(nx.connected_components(G_curr), key=len)) if G_curr.number_of_nodes() else 0,
                "lcc_frac": 0.0,  # will be overwritten below
                "entropy_deg": np.nan,
                "assortativity": np.nan,
                "clustering": np.nan,
                "diameter_approx": None,
            }

        met["step"] = int(step)
        met["nodes_left"] = int(G_curr.number_of_nodes())
        met["removed_total"] = int(removed_total)
        met["removed_frac"] = float(removed_total / max(1, N0))
        met["lcc_frac"] = lcc_fraction(G_curr, N0)

        history.append(met)

        # pick targets
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

        # apply removal
        G_curr.remove_nodes_from(targets)
        removed_total += len(targets)

        if removed_total >= total_remove:
            break

    # final state snapshot
    if keep_states and G_curr.number_of_nodes() > 0:
        states.append(G_curr.copy())

    df_hist = pd.DataFrame(history)
    return df_hist, states
