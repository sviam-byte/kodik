from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import networkx as nx


def add_dist_attr(G: nx.Graph) -> nx.Graph:
    """Copy a graph and add inverse-weight distance attribute for path algorithms."""
    H = G.copy()
    for _, _, d in H.edges(data=True):
        w = float(d.get("weight", 1.0))
        d["dist"] = 1.0 / w if w > 0 else 1e12
    return H


# ------------------------------------------------------------
# 1) Entropy rate of random walk
# ------------------------------------------------------------
def network_entropy_rate(G: nx.Graph, *, base: float = math.e) -> float:
    """
    Entropy rate:
        H_rw = - Σ_i π_i Σ_j P_ij log P_ij

    where P_ij = w_ij / strength(i) (undirected, weighted random walk),
    π_i = strength(i) / Σ_k strength(k).

    Returns nats if base=e, bits if base=2.
    """
    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        return 0.0

    H = G
    if hasattr(H, "is_directed") and H.is_directed():
        H = H.to_undirected(as_view=False)

    strength = dict(H.degree(weight="weight"))
    total_strength = float(sum(max(0.0, float(s)) for s in strength.values()))
    if total_strength <= 0:
        return 0.0

    log = math.log
    log_base = log(base)

    out = 0.0
    for i in H.nodes():
        s_i = float(strength.get(i, 0.0))
        if s_i <= 0:
            continue
        pi_i = s_i / total_strength

        row = 0.0
        for j, d in H[i].items():
            w = d.get("weight", 1.0)
            try:
                w = float(w)
            except Exception:
                w = 1.0
            if w <= 0:
                continue
            p = w / s_i
            if p > 0:
                row -= p * log(p) / log_base

        out += pi_i * row

    return float(out)


# ------------------------------------------------------------
# 2) Ollivier–Ricci curvature
# ------------------------------------------------------------
def _one_step_measure(H: nx.Graph, x) -> Dict:
    """µ_x(y) for neighbors y, proportional to w_xy."""
    neigh = list(H.neighbors(x))
    if not neigh:
        return {}
    ws = []
    for y in neigh:
        d = H[x][y]
        w = d.get("weight", 1.0)
        try:
            w = float(w)
        except Exception:
            w = 1.0
        ws.append(max(0.0, w))
    s = float(sum(ws))
    if s <= 0:
        p = 1.0 / len(neigh)
        return {y: p for y in neigh}
    return {y: w / s for y, w in zip(neigh, ws)}


def _quantize_probs(probs: Dict, scale: int) -> Tuple[List, List[int]]:
    """Map {node: p} -> (nodes, integer masses) summing exactly to scale."""
    items = [(k, float(v)) for k, v in probs.items() if float(v) > 0]
    if not items:
        return [], []
    nodes = [k for k, _ in items]
    ps = np.array([p for _, p in items], dtype=float)
    s = float(ps.sum())
    if s <= 0:
        return [], []
    ps = ps / s

    masses = np.floor(ps * scale).astype(int)
    rem = int(scale - masses.sum())

    if rem > 0:
        frac = ps * scale - np.floor(ps * scale)
        order = np.argsort(-frac)
        for idx in order[:rem]:
            masses[int(idx)] += 1
    elif rem < 0:
        order = np.argsort(-masses)
        k = 0
        while rem < 0 and k < len(order):
            idx = int(order[k])
            if masses[idx] > 0:
                masses[idx] -= 1
                rem += 1
            else:
                k += 1

    if int(masses.sum()) != int(scale):
        masses[-1] += int(scale - masses.sum())

    return nodes, masses.tolist()


def _emd_w1_transport(
    supply: Dict,
    demand: Dict,
    dist: Dict[Tuple, float],
    *,
    scale: int,
    missing_cost: float,
) -> float:
    """
    Exact W1 using transportation min-cost flow (network_simplex).
    Costs are real; we scale demands to integers, keep costs as floats.
    """
    S_nodes, S_mass = _quantize_probs(supply, scale)
    D_nodes, D_mass = _quantize_probs(demand, scale)
    if not S_nodes or not D_nodes:
        return 0.0

    Gf = nx.DiGraph()
    for u, m in zip(S_nodes, S_mass):
        Gf.add_node(("S", u), demand=-int(m))
    for v, m in zip(D_nodes, D_mass):
        Gf.add_node(("D", v), demand=+int(m))

    for u in S_nodes:
        for v in D_nodes:
            c = dist.get((u, v), dist.get((v, u), None))
            if c is None or not np.isfinite(c):
                c = float(missing_cost)
            Gf.add_edge(("S", u), ("D", v), weight=float(c), capacity=int(scale))

    cost, _ = nx.network_simplex(Gf)
    return float(cost) / float(scale)


def ollivier_ricci_edge(
    G: nx.Graph,
    x,
    y,
    *,
    max_support: int = 60,
    cutoff: float = 8.0,
    scale: int = 200_000,
    missing_cost: float = 1e6,
) -> float | None:
    """
    κ(x,y) = 1 - W1(µ_x, µ_y) / d(x,y)

    Distances use kodik convention: edge dist = 1/weight (stored in attr 'dist'),
    and shortest paths are computed with Dijkstra on that 'dist'.
    """
    if G.number_of_edges() == 0:
        return None

    H = G
    if hasattr(H, "is_directed") and H.is_directed():
        H = H.to_undirected(as_view=False)

    if not H.has_edge(x, y):
        return None

    Hw = add_dist_attr(H)

    mu_x = _one_step_measure(H, x)
    mu_y = _one_step_measure(H, y)
    if not mu_x or not mu_y:
        return None

    sx = list(mu_x.keys())
    sy = list(mu_y.keys())
    if (len(sx) + len(sy)) > int(max_support):
        return None

    # d(x,y): direct edge distance (1/weight)
    dxy = float(Hw[x][y].get("dist", 1.0))
    if not np.isfinite(dxy) or dxy <= 0:
        return None

    # pairwise distances between supports via Dijkstra (bounded)
    dist = {}
    for u in sx:
        dists = nx.single_source_dijkstra_path_length(Hw, u, cutoff=float(cutoff), weight="dist")
        for v in sy:
            if v in dists:
                dist[(u, v)] = float(dists[v])

    W1 = _emd_w1_transport(mu_x, mu_y, dist, scale=int(scale), missing_cost=float(missing_cost))
    kappa = 1.0 - (W1 / dxy)
    return float(kappa)


@dataclass
class CurvatureSummary:
    kappa_mean: float
    kappa_median: float
    kappa_frac_negative: float
    computed_edges: int
    skipped_edges: int


def ollivier_ricci_summary(
    G: nx.Graph,
    *,
    sample_edges: int = 150,
    seed: int = 42,
    max_support: int = 60,
    cutoff: float = 8.0,
    scale: int = 200_000,
) -> CurvatureSummary:
    """Compute summary stats of κ over a random sample of edges."""
    if G.number_of_edges() == 0:
        return CurvatureSummary(0.0, 0.0, 0.0, 0, 0)

    H = G
    if hasattr(H, "is_directed") and H.is_directed():
        H = H.to_undirected(as_view=False)

    edges = list(H.edges())
    rng = random.Random(int(seed))
    if len(edges) > int(sample_edges):
        edges = rng.sample(edges, int(sample_edges))

    kappas: List[float] = []
    skipped = 0

    for x, y in edges:
        k = ollivier_ricci_edge(
            H,
            x,
            y,
            max_support=max_support,
            cutoff=cutoff,
            scale=scale,
        )
        if k is None or not np.isfinite(k):
            skipped += 1
            continue
        kappas.append(float(k))

    if len(kappas) == 0:
        return CurvatureSummary(float("nan"), float("nan"), float("nan"), 0, int(skipped))

    arr = np.array(kappas, dtype=float)
    return CurvatureSummary(
        kappa_mean=float(arr.mean()),
        kappa_median=float(np.median(arr)),
        kappa_frac_negative=float((arr < 0).mean()),
        computed_edges=int(arr.size),
        skipped_edges=int(skipped),
    )


# ------------------------------------------------------------
# 3) Fragility proxies
# ------------------------------------------------------------
def fragility_from_entropy(h_rw: float, eps: float = 1e-9) -> float:
    """Higher entropy rate -> lower fragility."""
    if not np.isfinite(h_rw):
        return float("nan")
    return float(1.0 / max(eps, float(h_rw)))


def fragility_from_curvature(kappa_mean: float, eps: float = 1e-9) -> float:
    """
    κ̄ can be negative; map to positive score using shift:
        frag = 1 / max(eps, 1 + κ̄)
    """
    if not np.isfinite(kappa_mean):
        return float("nan")
    return float(1.0 / max(eps, 1.0 + float(kappa_mean)))


# ------------------------------------------------------------
# 4) Demetrius evolutionary entropy (PF-Markov construction)
# ------------------------------------------------------------
@dataclass
class PFMarkov:
    nodes: List
    lam: float
    u: np.ndarray          # right PF eigenvector (positive)
    v: np.ndarray          # left PF eigenvector (positive)
    P: "np.ndarray"        # dense transition matrix (row-stochastic)


def _largest_pf_eigs(A_csr):
    """
    Return (lam, u_right, v_left) for a nonnegative matrix A (sparse).
    Uses scipy if available; falls back to numpy (dense) for tiny graphs.
    """
    n = A_csr.shape[0]
    # Try sparse eigs first
    try:
        import scipy.sparse.linalg as spla

        # Right eigenvector of A
        vals_r, vecs_r = spla.eigs(A_csr, k=1, which="LR")
        lam = float(np.real(vals_r[0]))
        u = np.real(vecs_r[:, 0])

        # Left eigenvector = right eigenvector of A^T
        vals_l, vecs_l = spla.eigs(A_csr.T, k=1, which="LR")
        v = np.real(vecs_l[:, 0])

        return lam, u, v
    except Exception:
        # Dense fallback (small n)
        A = A_csr.toarray().astype(float)
        vals, vecs = np.linalg.eig(A)
        idx = int(np.argmax(np.real(vals)))
        lam = float(np.real(vals[idx]))
        u = np.real(vecs[:, idx])

        vals2, vecs2 = np.linalg.eig(A.T)
        idx2 = int(np.argmax(np.real(vals2)))
        v = np.real(vecs2[:, idx2])

        return lam, u, v


def demetrius_pf_markov(G: nx.Graph) -> Optional[PFMarkov]:
    """
    Build Demetrius PF-Markov chain from weighted adjacency A (nonnegative).
    For nonnegative irreducible A:
        p_ij = a_ij * u_j / (lam * u_i)
    where u is the right PF eigenvector (A u = lam u).

    Stationary distribution for this P:
        π_i ∝ u_i * v_i
    where v is left PF eigenvector (A^T v = lam v).

    Returns PFMarkov with dense P (since sizes in Kodik are typically moderate).
    """
    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        return None

    H = G
    if isinstance(H, (nx.MultiGraph, nx.MultiDiGraph)):
        # collapse multi-edges by summing weights
        S = nx.DiGraph() if H.is_directed() else nx.Graph()
        S.add_nodes_from(H.nodes(data=True))
        for a, b, d in H.edges(data=True):
            w = d.get("weight", 1.0)
            try:
                w = float(w)
            except Exception:
                w = 1.0
            if w < 0:
                w = 0.0
            if S.has_edge(a, b):
                S[a][b]["weight"] = float(S[a][b].get("weight", 0.0)) + w
            else:
                S.add_edge(a, b, weight=w)
        H = S

    # For undirected graphs: treat as directed both ways (A is symmetric).
    if not H.is_directed():
        H = H.to_directed()

    nodes = list(H.nodes())
    n = len(nodes)
    idx = {nodes[i]: i for i in range(n)}

    # Build nonnegative weighted adjacency
    A = nx.adjacency_matrix(H, nodelist=nodes, weight="weight").astype(float).tocsr()
    if A.nnz == 0:
        return None

    # Ensure nonnegativity
    if A.data.size > 0:
        A.data = np.maximum(A.data, 0.0)

    lam, u, v = _largest_pf_eigs(A)

    if not np.isfinite(lam) or lam <= 0:
        return None

    # Make u, v positive (sign ambiguity)
    u = np.abs(u) + 1e-15
    v = np.abs(v) + 1e-15

    # Build P_ij = a_ij * u_j / (lam * u_i)
    P = np.zeros((n, n), dtype=float)
    A_coo = A.tocoo()
    for i, j, aij in zip(A_coo.row, A_coo.col, A_coo.data):
        if aij <= 0:
            continue
        P[i, j] += float(aij) * float(u[j]) / (float(lam) * float(u[i]))

    # Row-normalize defensively (should already be stochastic if irreducible)
    row_sums = P.sum(axis=1)
    for i in range(n):
        s = float(row_sums[i])
        if s > 0:
            P[i, :] /= s
        else:
            # dead row -> uniform (rare); keeps chain defined
            P[i, :] = 1.0 / n

    return PFMarkov(nodes=nodes, lam=float(lam), u=u, v=v, P=P)


def evolutionary_entropy_demetrius(G: nx.Graph, *, base: float = math.e) -> float:
    """
    Demetrius evolutionary entropy as entropy rate of PF-Markov chain.

    H_evo = - Σ_i π_i Σ_j P_ij log(P_ij)
    with π_i ∝ u_i v_i (PF left/right eigenvectors of A).
    """
    mk = demetrius_pf_markov(G)
    if mk is None:
        return float("nan")

    u = mk.u
    v = mk.v
    P = mk.P

    pi = u * v
    Z = float(pi.sum())
    if not np.isfinite(Z) or Z <= 0:
        return float("nan")
    pi = pi / Z

    log = math.log
    log_base = log(base)

    # entropy rate
    H = 0.0
    for i in range(P.shape[0]):
        row = P[i, :]
        # -Σ p log p
        mask = row > 0
        if not np.any(mask):
            continue
        h_i = -float(np.sum(row[mask] * (np.log(row[mask]) / log_base)))
        H += float(pi[i]) * h_i

    return float(H)
