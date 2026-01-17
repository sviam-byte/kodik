import random
import pandas as pd
import networkx as nx

from .attacks import pick_targets_for_attack
from .metrics import calculate_metrics, lcc_fraction

def run_attack(
    G0: nx.Graph,
    attack_kind: str,
    remove_frac: float,
    steps: int,
    seed: int,
    eff_k: int,
    rc_frac: float,
    rc_min_density: float,
    rc_max_frac: float,
    compute_heavy_every: int = 1,
) -> pd.DataFrame:
    if G0.number_of_nodes() < 2:
        return pd.DataFrame()

    G = G0.copy()
    N0 = G.number_of_nodes()
    rng = random.Random(int(seed))

    total_remove = int(N0 * float(remove_frac))
    step_size = max(1, total_remove // int(steps))

    rows = []
    for step in range(int(steps)):
        if G.number_of_nodes() < 2:
            break

        heavy = (compute_heavy_every > 0 and (step % int(compute_heavy_every) == 0))
        met = calculate_metrics(G, eff_sources_k=int(eff_k), seed=int(seed), compute_heavy=heavy)

        met["step"] = int(step)
        met["nodes_left"] = int(G.number_of_nodes())
        met["lcc_frac"] = float(lcc_fraction(G, N0))
        rows.append(met)

        targets = pick_targets_for_attack(
            G,
            attack_kind=attack_kind,
            step_size=step_size,
            seed=int(seed) + step,
            rc_frac=float(rc_frac),
            rc_min_density=float(rc_min_density),
            rc_max_frac=float(rc_max_frac),
        )

        if not targets:
            nodes = list(G.nodes())
            k = min(len(nodes), step_size)
            targets = rng.sample(nodes, k) if k > 0 else []

        G.remove_nodes_from(targets)

    return pd.DataFrame(rows)
