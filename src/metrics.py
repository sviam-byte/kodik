import math
import random
import numpy as np
import networkx as nx
import scipy.sparse.linalg as spla
import plotly.graph_objects as go

from networkx.algorithms.community import modularity, louvain_communities

from src.robust_geom import (
    network_entropy_rate,
    evolutionary_entropy_demetrius,
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


def spectral_radius_weighted_adjacency(G: nx.Graph) -> float:
    """Largest eigenvalue of weighted adjacency (Perron-Frobenius radius)."""
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return 0.0
    try:
        A = nx.adjacency_matrix(G, weight="weight").astype(float)
        # Use sparse eigs for largest magnitude eigenvalue
        vals = spla.eigs(A, k=1, which="LR", return_eigenvectors=False)
        lmax = float(np.real(vals[0]))
        return max(0.0, lmax)
    except Exception:
        try:
            A = nx.to_numpy_array(G, weight="weight", dtype=float)
            vals = np.linalg.eigvals(A)
            lmax = float(np.max(np.real(vals)))
            return max(0.0, lmax)
        except Exception:
            return 0.0


def lambda2_on_lcc(G: nx.Graph) -> float:
    """Second-smallest eigenvalue of normalized Laplacian on LCC (undirected)."""
    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        return 0.0
    if G.is_directed():
        # For directed, use undirected view for this proxy.
        H = G.to_undirected(as_view=False)
    else:
        H = G

    comps = list(nx.connected_components(H))
    if not comps:
        return 0.0
    lcc = max(comps, key=len)
    if len(lcc) < 2:
        return 0.0

    Hs = H.subgraph(lcc).copy()
    try:
        L = nx.normalized_laplacian_matrix(Hs, weight="weight").astype(float)
        vals = spla.eigs(L, k=min(3, L.shape[0] - 1), which="SR", return_eigenvectors=False)
        vals = np.sort(np.real(vals))
        if vals.size >= 2:
            return float(max(0.0, vals[1]))
        return 0.0
    except Exception:
        try:
            L = nx.normalized_laplacian_matrix(Hs, weight="weight").toarray().astype(float)
            vals = np.sort(np.real(np.linalg.eigvals(L)))
            if vals.size >= 2:
                return float(max(0.0, vals[1]))
            return 0.0
        except Exception:
            return 0.0


def approx_weighted_efficiency(G: nx.Graph, sources_k: int = 32, seed: int = 0) -> float:
    """Approximate global efficiency using k random sources + Dijkstra on dist=1/weight."""
    N = G.number_of_nodes()
    if N < 2 or G.number_of_edges() == 0:
        return 0.0

    H = add_dist_attr(G)
    rng = random.Random(int(seed))
    nodes = list(H.nodes())
    k = min(int(sources_k), len(nodes))
    if k <= 0:
        return 0.0
    sources = rng.sample(nodes, k)

    inv_sum = 0.0
    cnt = 0
    for s in sources:
        dists = nx.single_source_dijkstra_path_length(H, s, weight="dist")
        for t, d in dists.items():
            if s == t:
                continue
            if d and np.isfinite(d) and d > 0:
                inv_sum += 1.0 / float(d)
                cnt += 1
    return float(inv_sum / cnt) if cnt > 0 else 0.0


def compute_modularity_louvain(G: nx.Graph, seed: int = 0) -> float:
    """Louvain modularity (undirected view for directed graphs)."""
    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        return 0.0
    H = G.to_undirected(as_view=False) if G.is_directed() else G
    try:
        parts = louvain_communities(H, weight="weight", seed=int(seed))
        return float(modularity(H, parts, weight="weight"))
    except Exception:
        return 0.0


def degree_entropy(G: nx.Graph) -> float:
    """Entropy of degree distribution (unweighted degrees)."""
    N = G.number_of_nodes()
    if N == 0:
        return 0.0
    degs = np.array([d for _, d in G.degree()], dtype=float)
    if degs.size == 0:
        return 0.0
    vals, counts = np.unique(degs, return_counts=True)
    p = counts.astype(float) / float(counts.sum())
    p = p[p > 0]
    return float(-np.sum(p * np.log(p))) if p.size > 0 else 0.0


def approx_diameter_lcc(G: nx.Graph, seed: int = 0, samples: int = 16):
    """Approximate diameter on LCC using double-sweep BFS from random seeds."""
    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        return None
    H = G.to_undirected(as_view=False) if G.is_directed() else G

    comps = list(nx.connected_components(H))
    if not comps:
        return None
    lcc = max(comps, key=len)
    if len(lcc) < 2:
        return 0
    S = H.subgraph(lcc).copy()

    rng = random.Random(int(seed))
    nodes = list(S.nodes())
    if not nodes:
        return None
    k = min(int(samples), len(nodes))
    starts = rng.sample(nodes, k)

    best = 0
    for s in starts:
        d1 = nx.single_source_shortest_path_length(S, s)
        if not d1:
            continue
        u = max(d1, key=d1.get)
        d2 = nx.single_source_shortest_path_length(S, u)
        if not d2:
            continue
        diam = max(d2.values())
        if diam > best:
            best = diam
    return int(best)


def _shannon_entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    s = float(np.sum(counts))
    if s <= 0:
        return float("nan")
    p = counts / s
    p = p[p > 0]
    if p.size == 0:
        return float("nan")
    return float(-np.sum(p * np.log(p)))


def _shannon_entropy_from_values(values, bins: int = 32) -> float:
    xs = np.asarray(values, dtype=float)
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        return float("nan")
    hist, _ = np.histogram(xs, bins=int(bins))
    return _shannon_entropy_from_counts(hist)


# =========================
# Metrics
# =========================
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

    # Demetrius evolutionary entropy (PF-Markov)
    try:
        H_evo = float(evolutionary_entropy_demetrius(G, base=math.e))
    except Exception:
        H_evo = float("nan")

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
    frag_evo = float(fragility_from_entropy(H_evo)) if np.isfinite(H_evo) else float("nan")
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
        "H_evo": float(H_evo) if np.isfinite(H_evo) else float("nan"),
        "kappa_mean": float(kappa_mean) if np.isfinite(kappa_mean) else float("nan"),
        "kappa_median": float(kappa_median) if np.isfinite(kappa_median) else float("nan"),
        "kappa_frac_negative": float(kappa_frac_negative) if np.isfinite(kappa_frac_negative) else float("nan"),
        "kappa_computed_edges": int(kappa_computed_edges),
        "kappa_skipped_edges": int(kappa_skipped_edges),
        "fragility_H": float(frag_H) if np.isfinite(frag_H) else float("nan"),
        "fragility_evo": float(frag_evo) if np.isfinite(frag_evo) else float("nan"),
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
    texts = [
        f"{str(n)}<br>strength={strength.get(n, 0.0):.3g}"
        for n in nodes
    ]

    node_trace = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers",
        marker=dict(
            size=6,
            color=colors,
            colorscale="Viridis",
            showscale=bool(show_scale),
            colorbar=dict(title="strength") if show_scale else None,
        ),
        text=texts,
        hoverinfo="text",
        name="nodes",
    )

    # edges
    ex = []
    ey = []
    ez = []
    ew = []
    for a, b, d in G.edges(data=True):
        if a not in pos3d or b not in pos3d:
            continue
        ex += [pos3d[a][0], pos3d[b][0], None]
        ey += [pos3d[a][1], pos3d[b][1], None]
        ez += [pos3d[a][2], pos3d[b][2], None]
        ew.append(float(d.get("weight", 1.0)))

    edge_trace = go.Scatter3d(
        x=ex,
        y=ey,
        z=ez,
        mode="lines",
        line=dict(width=2, color="rgba(150,150,150,0.55)"),
        hoverinfo="none",
        name="edges",
    )

    return node_trace, edge_trace
