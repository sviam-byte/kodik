import random
import numpy as np
import pandas as pd
import networkx as nx

from .metrics import calculate_metrics, lcc_fraction, add_dist_attr


# -------------------------
# Rich-club pickers
# -------------------------
def strength_ranking(G: nx.Graph) -> list:
    """Rank nodes by weighted degree (strength)."""
    strength = dict(G.degree(weight="weight"))
    nodes = list(G.nodes())
    return sorted(nodes, key=lambda n: strength.get(n, 0.0), reverse=True)


def richclub_top_fraction(G: nx.Graph, rc_frac: float) -> list:
    """Pick top fraction of nodes by strength."""
    nodes_sorted = strength_ranking(G)
    if not nodes_sorted:
        return []
    k = max(1, int(len(nodes_sorted) * float(rc_frac)))
    return nodes_sorted[:k]


def richclub_by_density_threshold(G: nx.Graph, min_density: float, max_frac: float) -> list:
    """Pick the largest prefix where induced density exceeds threshold."""
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


def pick_targets_for_attack(
    G: nx.Graph,
    attack_kind: str,
    step_size: int,
    seed: int,
    rc_frac: float,
    rc_min_density: float,
    rc_max_frac: float,
) -> list:
    """Select nodes to remove for a given attack strategy."""
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
        # approximate betweenness for speed
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
        return club[:min(step_size, len(club))]

    if attack_kind == "richclub_density":
        club = richclub_by_density_threshold(G, min_density=rc_min_density, max_frac=rc_max_frac)
        return club[:min(step_size, len(club))]

    return []


# -------------------------
# Phase transition detector
# -------------------------
def detect_abrupt_collapse(y: np.ndarray, x: np.ndarray) -> dict:
    """
    y = order parameter, e.g. lcc_frac
    x = removed_fraction

    We detect maximal negative slope (largest drop per small delta-x).
    """
    if len(y) < 3:
        return {"is_abrupt": False, "crit_x": None, "max_drop": 0.0}

    dy = np.diff(y)
    dx = np.diff(x)
    slopes = dy / np.maximum(dx, 1e-12)

    j = int(np.argmin(slopes))  # most negative slope
    max_drop = float(-slopes[j])

    # heuristic threshold: "abrupt" if drop rate is huge
    # You can calibrate: e.g. > 2.0 means lcc falls ~2 per 1.0 removal fraction
    is_abrupt = bool(max_drop > 2.0)

    crit_x = float(x[j + 1])
    return {"is_abrupt": is_abrupt, "crit_x": crit_x, "max_drop": max_drop}


# -------------------------
# Attack runner
# -------------------------
def run_attack(
    G: nx.Graph,
    attack_kind: str,
    remove_frac: float,
    steps: int,
    seed: int,
    eff_sources_k: int,
    rc_frac: float = 0.10,
    rc_min_density: float = 0.30,
    rc_max_frac: float = 0.30,
    compute_heavy_every: int = 1,
) -> pd.DataFrame:
    """
    compute_heavy_every:
      1 -> compute all metrics each step (slow)
      k -> compute heavy metrics each k steps (faster)
    """
    if G.number_of_nodes() < 2:
        return pd.DataFrame()

    G_curr = G.copy()
    N0 = G_curr.number_of_nodes()
    total_remove = int(N0 * float(remove_frac))
    step_size = max(1, total_remove // int(steps))

    history = []

    for step in range(int(steps)):
        if G_curr.number_of_nodes() < 2:
            break

        do_heavy = (step % max(1, int(compute_heavy_every)) == 0)

        met = {"step": step, "nodes_left": G_curr.number_of_nodes()}
        met["removed_frac"] = 1.0 - (G_curr.number_of_nodes() / max(1, N0))
        met["lcc_frac"] = lcc_fraction(G_curr, N0)

        if do_heavy:
            m2 = calculate_metrics(G_curr, eff_sources_k=int(eff_sources_k), seed=int(seed))
            met.update(m2)

        history.append(met)

        targets = pick_targets_for_attack(
            G_curr,
            attack_kind=attack_kind,
            step_size=step_size,
            seed=int(seed) + step,
            rc_frac=float(rc_frac),
            rc_min_density=float(rc_min_density),
            rc_max_frac=float(rc_max_frac),
        )

        # fallback
        if not targets:
            nodes = list(G_curr.nodes())
            k = min(len(nodes), step_size)
            rng = random.Random(int(seed) + 9999 + step)
            targets = rng.sample(nodes, k) if k > 0 else []

        G_curr.remove_nodes_from(targets)

    df = pd.DataFrame(history)
    if len(df):
        # fill heavy metrics forward for plotting continuity
        df = df.sort_values("step").reset_index(drop=True)
        df = df.ffill()

        # abrupt collapse stats based on LCC
        info = detect_abrupt_collapse(df["lcc_frac"].values, df["removed_frac"].values)
        df.attrs["phase"] = info

    return df
