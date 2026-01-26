import math
import random
import numpy as np
import networkx as nx
import scipy.sparse.linalg as spla
import plotly.graph_objects as go

from networkx.algorithms.community import modularity, louvain_communities

from src.robust_geom import (
    network_entropy_rate,
    ollivier_ricci_summary,
    fragility_from_entropy,
    fragility_from_curvature,
)


# -------------------------
# Helpers
# -------------------------
def add_dist_attr(G: nx.Graph) -> nx.Graph:
    """Copy a graph and add inverse-weight distance attribute for path algorithms."""
    H = G.copy()
    for _, _, d in H.edges(data=True):
        w = float(d.get("weight", 1.0))
        d["dist"] = 1.0 / w if w > 0 else 1e12
    return H


def _shannon_entropy_from_counts(counts: np.ndarray) -> float:
    """Shannon entropy H = -sum p log2 p for a nonnegative count vector."""
    counts = np.asarray(counts, dtype=float)
    counts = counts[np.isfinite(counts)]
    counts = counts[counts > 0]
    if counts.size == 0:
        return float("nan")
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _shannon_entropy_from_values(values, bins: int = 32) -> float:
    """Histogram-based Shannon entropy for a list/array of real values."""
    xs = np.asarray(list(values), dtype=float)
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        return float("nan")
    # If all equal -> entropy 0 (single bin effectively).
    if float(xs.min()) == float(xs.max()):
        return 0.0
    hist, _ = np.histogram(xs, bins=int(max(2, bins)))
    return _shannon_entropy_from_counts(hist)


def approx_weighted_efficiency(G: nx.Graph, sources_k: int, seed: int) -> float:
    """
    E_w = (1/(N(N-1))) * sum_{i!=j} 1/d_ij, dist = 1/weight
    """
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

    est_full_sum = total * (N / max(1, k))
    return float(est_full_sum / denom)


def spectral_radius_weighted_adjacency(G: nx.Graph) -> float:
    """Return largest eigenvalue of weighted adjacency matrix."""
    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        return 0.0
    A = nx.adjacency_matrix(G, weight="weight").astype(float)
    try:
        v = spla.eigsh(A, k=1, which="LM", return_eigenvectors=False)[0]
        return float(v)
    except Exception:
        return 0.0


def lambda2_robust_connected(G: nx.Graph, eps: float = 1e-10) -> float:
    """
    λ2 для связного графа .
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n < 3 or m == 0:
        return 0.0
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
            return 0.0


def lambda2_on_lcc(G: nx.Graph) -> float:
    if G.number_of_nodes() < 3 or G.number_of_edges() == 0:
        return 0.0
    if not nx.is_connected(G):
        return 0.0
    return lambda2_robust_connected(G)


def compute_modularity_louvain(G: nx.Graph, seed: int) -> float:
    if G.number_of_nodes() < 3 or G.number_of_edges() == 0:
        return 0.0
    try:
        comm = louvain_communities(G, weight="weight", seed=int(seed))
        return float(modularity(G, comm, weight="weight"))
    except TypeError:
        comm = louvain_communities(G, weight="weight")
        return float(modularity(G, comm, weight="weight"))
    except Exception:
        return 0.0


def degree_entropy(G: nx.Graph) -> float:
    """
    Энтропия распределения степеней 
    """
    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    degs = np.array([d for _, d in G.degree()], dtype=float)
    if degs.sum() <= 0:
        return 0.0
    p = degs / degs.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def approx_diameter_lcc(G: nx.Graph, seed: int = 42, samples: int = 16) -> int | None:
    """
    BFS от нескольких случайных узлов
    """
    if G.number_of_nodes() < 2:
        return 0
    nodes = list(G.nodes())
    rng = random.Random(int(seed))
    k = min(samples, len(nodes))
    picks = rng.sample(nodes, k)
    best = 0
    for s in picks:
        dist = nx.single_source_shortest_path_length(G, s)
        if dist:
            best = max(best, max(dist.values()))
    return int(best)


def calculate_metrics(G: nx.Graph, eff_sources_k: int, seed: int) -> dict:
    N = G.number_of_nodes()
    E = G.number_of_edges()
    if N > 0:
        try:
            C = (
                nx.number_connected_components(G)
                if not G.is_directed()
                else nx.number_weakly_connected_components(G)
            )
        except Exception:
            C = 1
    else:
        C = 0
    dens = nx.density(G) if N > 1 else 0.0
    avg_deg = (2 * E / N) if N > 0 else 0.0

    lcc_size = len(max(nx.connected_components(G), key=len)) if N > 0 else 0
    lcc_frac = lcc_size / N if N > 0 else 0.0

    lmax = spectral_radius_weighted_adjacency(G)
    thresh = (1.0 / lmax) if lmax > 0 else 0.0

    l2 = lambda2_on_lcc(G)
    tau = (1.0 / l2) if l2 > 0 else float("inf")

    eff_w = approx_weighted_efficiency(G, sources_k=int(eff_sources_k), seed=int(seed))
    Q = compute_modularity_louvain(G, seed=int(seed))

    ent = degree_entropy(G)
    assort = nx.degree_assortativity_coefficient(G) if N > 2 and E > 0 else 0.0
    clust = nx.average_clustering(G) if N > 2 and E > 0 else 0.0
    diam = approx_diameter_lcc(G, seed=int(seed), samples=16)

    beta = int(E - N + C) if N > 0 else 0

    # -----------------------------
    # Entropy Profile
    # -----------------------------
    degs = np.array([d for _, d in G.degree()], dtype=float)
    if degs.size > 0:
        _, deg_counts = np.unique(degs, return_counts=True)
        H_deg = _shannon_entropy_from_counts(deg_counts)
    else:
        H_deg = float("nan")

    ws = []
    cs = []
    for _, _, d in G.edges(data=True):
        w = d.get("weight", np.nan)
        c = d.get("confidence", np.nan)
        try:
            ws.append(float(w))
        except Exception:
            pass
        try:
            cs.append(float(c))
        except Exception:
            pass
    H_w = _shannon_entropy_from_values(ws, bins=32) if ws else float("nan")
    H_conf = _shannon_entropy_from_values(cs, bins=32) if cs else float("nan")

    if E > 0:
        beta_red = (E - (N - C)) / float(E)
    else:
        beta_red = float("nan")

    # Relaxation time τ (diffusion-like proxy).
    tau_relax = (1.0 / l2) if (l2 and np.isfinite(l2) and l2 > 1e-12) else float("nan")
    epi_thr = (1.0 / lmax) if (lmax and np.isfinite(lmax) and lmax > 1e-12) else float("nan")

    # ----------------------------Robust geometry (additive; safe defaults)
    try:
        H_rw = float(network_entropy_rate(G, base=math.e))
    except Exception:
        H_rw = float("nan")

    try:
        curv = ollivier_ricci_summary(
            G,
            sample_edges=150,
            seed=int(seed),
            max_support=60,
            cutoff=8.0,
            scale=200_000,
        )
        kappa_mean = float(curv.kappa_mean)
        kappa_median = float(curv.kappa_median)
        kappa_frac_negative = float(curv.kappa_frac_negative)
        kappa_computed_edges = int(curv.computed_edges)
        kappa_skipped_edges = int(curv.skipped_edges)
    except Exception:
        kappa_mean = float("nan")
        kappa_median = float("nan")
        kappa_frac_negative = float("nan")
        kappa_computed_edges = 0
        kappa_skipped_edges = 0

    frag_H = float(fragility_from_entropy(H_rw)) if np.isfinite(H_rw) else float("nan")
    frag_k = (
        float(fragility_from_curvature(kappa_mean)) if np.isfinite(kappa_mean) else float("nan")
    )

    return {
        "N": N,
        "E": E,
        "C": C,
        "density": float(dens),
        "avg_degree": float(avg_deg),
        "beta": beta,
        "beta_red": float(beta_red) if np.isfinite(beta_red) else float("nan"),
        "lcc_size": int(lcc_size),
        "lcc_frac": float(lcc_frac),
        "eff_w": float(eff_w),
        "l2_lcc": float(l2),
        "tau_lcc": float(tau),
        "tau_relax": float(tau_relax) if np.isfinite(tau_relax) else float("nan"),
        "lmax": float(lmax),
        "thresh": float(thresh),
        "epi_thr": float(epi_thr) if np.isfinite(epi_thr) else float("nan"),
        "mod": float(Q),
        "entropy_deg": float(ent),
        "H_deg": float(H_deg) if np.isfinite(H_deg) else float("nan"),
        "H_w": float(H_w) if np.isfinite(H_w) else float("nan"),
        "H_conf": float(H_conf) if np.isfinite(H_conf) else float("nan"),
        "assortativity": float(assort) if np.isfinite(assort) else 0.0,
        "clustering": float(clust) if np.isfinite(clust) else 0.0,
        "diameter_approx": int(diam) if diam is not None else None,

        # NEW keys (front-ready)
        "H_rw": float(H_rw) if np.isfinite(H_rw) else float("nan"),
        "kappa_mean": float(kappa_mean) if np.isfinite(kappa_mean) else float("nan"),
        "kappa_median": float(kappa_median) if np.isfinite(kappa_median) else float("nan"),
        "kappa_frac_negative": float(kappa_frac_negative) if np.isfinite(kappa_frac_negative) else float("nan"),
        "kappa_computed_edges": int(kappa_computed_edges),
        "kappa_skipped_edges": int(kappa_skipped_edges),
        "fragility_H": float(frag_H) if np.isfinite(frag_H) else float("nan"),
        "fragility_kappa": float(frag_k) if np.isfinite(frag_k) else float("nan"),
    }


# =========================
# 3D layout + traces
# =========================
def compute_3d_layout(G: nx.Graph, seed: int) -> dict:
    """Compute a deterministic 3D spring layout for visualization."""
    if G.number_of_nodes() == 0:
        return {}
    return nx.spring_layout(G, dim=3, weight="weight", seed=int(seed))


def make_3d_traces(G: nx.Graph, pos3d: dict, show_scale: bool = False):
    """Create Plotly traces for 3D node/edge rendering."""
    if G.number_of_nodes() == 0:
        return None, None

    strength = dict(G.degree(weight="weight"))
    nodes = [n for n in G.nodes() if n in pos3d]

    xs = [pos3d[n][0] for n in nodes]
    ys = [pos3d[n][1] for n in nodes]
    zs = [pos3d[n][2] for n in nodes]
    colors = [strength.get(n, 0.0) for n in nodes]
    texts = [f"{n}: strength={strength.get(n, 0.0):.3f}" for n in nodes]

    node_trace = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers",
        marker=dict(size=4, color=colors, colorscale="Inferno", showscale=show_scale),
        text=texts,
        hoverinfo="text",
        name="nodes",
    )

    ex, ey, ez = [], [], []
    for u, v in G.edges():
        if u not in pos3d or v not in pos3d:
            continue
        ex.extend([pos3d[u][0], pos3d[v][0], None])
        ey.extend([pos3d[u][1], pos3d[v][1], None])
        ez.extend([pos3d[u][2], pos3d[v][2], None])

    edge_trace = go.Scatter3d(
        x=ex,
        y=ey,
        z=ez,
        mode="lines",
        line=dict(color="#444", width=1),
        hoverinfo="none",
        name="edges",
    )
    return edge_trace, node_trace
