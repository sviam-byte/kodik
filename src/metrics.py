# src/metrics.py
import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
import scipy.sparse.linalg as spla
import plotly.graph_objects as go
import plotly.colors as pc

from networkx.algorithms.community import modularity, louvain_communities

from src.robust_geom import (
    network_entropy_rate,
    evolutionary_entropy_demetrius,
    ollivier_ricci_summary,
    ollivier_ricci_edge,
    fragility_from_entropy,
    fragility_from_curvature,
)


def add_dist_attr(G: nx.Graph) -> nx.Graph:
    H = G.copy()
    for _, _, d in H.edges(data=True):
        w = d.get("weight", 1.0)
        try:
            w = float(w)
        except Exception:
            w = 1.0
        d["dist"] = 1.0 / w if w > 0 else 1e12
    return H


def spectral_radius_weighted_adjacency(G: nx.Graph) -> float:
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return 0.0
    try:
        A = nx.adjacency_matrix(G, weight="weight").astype(float)
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
    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        return 0.0
    H = G.to_undirected(as_view=False) if G.is_directed() else G

    comps = list(nx.connected_components(H))
    if not comps:
        return 0.0
    lcc = max(comps, key=len)
    if len(lcc) < 2:
        return 0.0

    Hs = H.subgraph(lcc).copy()
    try:
        L = nx.normalized_laplacian_matrix(Hs, weight="weight").astype(float)
        # Shift-invert around sigma=0 to target smallest eigenvalues faster.
        if L.shape[0] <= 2:
            vals = np.sort(np.real(np.linalg.eigvals(L.toarray())))
            return float(max(0.0, vals[1])) if vals.size >= 2 else 0.0
        vals = spla.eigs(L, k=2, which="SM", sigma=0, return_eigenvectors=False)
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


def lcc_fraction(G: nx.Graph, N0: int) -> float:
    """Fraction of original nodes that remain in the largest connected component (undirected view)."""
    try:
        N0 = int(N0)
    except Exception:
        N0 = 0
    if N0 <= 0 or G.number_of_nodes() == 0:
        return 0.0
    H_u = G.to_undirected(as_view=False) if G.is_directed() else G
    try:
        lcc = max(nx.connected_components(H_u), key=len)
        return float(len(lcc) / float(N0))
    except Exception:
        return 0.0


def approx_weighted_efficiency(G: nx.Graph, sources_k: int = 32, seed: int = 0) -> float:
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
    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        return 0.0
    H = G.to_undirected(as_view=False) if G.is_directed() else G
    try:
        parts = louvain_communities(H, weight="weight", seed=int(seed))
        return float(modularity(H, parts, weight="weight"))
    except Exception:
        return 0.0


def degree_entropy(G: nx.Graph) -> float:
    N = G.number_of_nodes()
    if N == 0:
        return 0.0
    degs = np.array([d for _, d in G.degree()], dtype=float)
    if degs.size == 0:
        return 0.0
    _, counts = np.unique(degs, return_counts=True)
    p = counts.astype(float) / float(counts.sum())
    p = p[p > 0]
    return float(-np.sum(p * np.log(p))) if p.size > 0 else 0.0


def approx_diameter_lcc(G: nx.Graph, seed: int = 0, samples: int = 16):
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
        best = max(best, diam)
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


def calculate_metrics(
    G: nx.Graph,
    eff_sources_k: int,
    seed: int,
    compute_curvature: bool = True,
    curvature_sample_edges: int = 150,
    curvature_max_support: int = 60,
    curvature_cutoff: float = 8.0,
    **kwargs,
) -> dict:
    N = G.number_of_nodes()
    E = G.number_of_edges()
    # Back-compat: older callers used compute_heavy=True/False.
    if "compute_heavy" in kwargs:
        try:
            heavy = bool(kwargs.get("compute_heavy"))
        except Exception:
            heavy = True
        if not heavy:
            compute_curvature = False
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

    # LCC fraction (undirected view)
    H_u = G.to_undirected(as_view=False) if G.is_directed() else G
    lcc_size = len(max(nx.connected_components(H_u), key=len)) if N > 0 else 0
    lcc_frac = lcc_size / N if N > 0 else 0.0

    lmax = spectral_radius_weighted_adjacency(G)
    thresh = (1.0 / lmax) if lmax > 0 else 0.0

    l2 = lambda2_on_lcc(G)
    tau = (1.0 / l2) if l2 > 0 else float("inf")

    eff_w = approx_weighted_efficiency(G, sources_k=int(eff_sources_k), seed=int(seed))
    Q = compute_modularity_louvain(G, seed=int(seed))

    ent = degree_entropy(G)
    assort = nx.degree_assortativity_coefficient(G) if N > 2 and E > 0 else 0.0
    clust = nx.average_clustering(H_u) if N > 2 and E > 0 else 0.0
    diam = approx_diameter_lcc(G, seed=int(seed), samples=16)

    beta = int(E - N + C) if N > 0 else 0

    # Entropy profile
    degs = np.array([d for _, d in G.degree()], dtype=float)
    if degs.size > 0:
        _, deg_counts = np.unique(degs, return_counts=True)
        H_deg = _shannon_entropy_from_counts(deg_counts)
    else:
        H_deg = float("nan")

    ws = []
    cs = []
    for _, _, d in G.edges(data=True):
        try:
            ws.append(float(d.get("weight", np.nan)))
        except Exception:
            pass
        try:
            cs.append(float(d.get("confidence", np.nan)))
        except Exception:
            pass

    H_w = _shannon_entropy_from_values(ws, bins=32) if ws else float("nan")
    H_conf = _shannon_entropy_from_values(cs, bins=32) if cs else float("nan")

    beta_red = (E - (N - C)) / float(E) if E > 0 else float("nan")

    tau_relax = (1.0 / l2) if (np.isfinite(l2) and l2 > 1e-12) else float("nan")
    epi_thr = (1.0 / lmax) if (np.isfinite(lmax) and lmax > 1e-12) else float("nan")

    # --- robust geometry (safe)
    try:
        H_rw = float(network_entropy_rate(G, base=math.e))
    except Exception:
        H_rw = float("nan")

    try:
        H_evo = float(evolutionary_entropy_demetrius(G, base=math.e))
    except Exception:
        H_evo = float("nan")

    if compute_curvature and G.number_of_edges() > 0:
        try:
            curv = ollivier_ricci_summary(
                G,
                sample_edges=int(curvature_sample_edges),
                seed=int(seed),
                max_support=int(curvature_max_support),
                cutoff=float(curvature_cutoff),
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
    else:
        # Curvature is the main latency driver (LP/EMD per edge). Keep it opt-in.
        kappa_mean = float("nan")
        kappa_median = float("nan")
        kappa_frac_negative = float("nan")
        kappa_computed_edges = 0
        kappa_skipped_edges = 0

    frag_H = float(fragility_from_entropy(H_rw)) if np.isfinite(H_rw) else float("nan")
    frag_evo = float(fragility_from_entropy(H_evo)) if np.isfinite(H_evo) else float("nan")
    frag_k = float(fragility_from_curvature(kappa_mean)) if np.isfinite(kappa_mean) else float("nan")

    return {
        "N": int(N),
        "E": int(E),
        "C": int(C),
        "density": float(dens),
        "avg_degree": float(avg_deg),
        "beta": int(beta),
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


def compute_3d_layout(G: nx.Graph, seed: int) -> dict:
    if G.number_of_nodes() == 0:
        return {}
    return nx.spring_layout(G, dim=3, weight="weight", seed=int(seed))


def _as_undirected_simple(G: nx.Graph) -> nx.Graph:
    """Normalize to a simple undirected graph with numeric weights."""
    H = G
    if hasattr(H, "is_directed") and H.is_directed():
        H = H.to_undirected(as_view=False)

    if isinstance(H, (nx.MultiGraph, nx.MultiDiGraph)):
        S = nx.Graph()
        S.add_nodes_from(H.nodes(data=True))
        for u, v, d in H.edges(data=True):
            w = d.get("weight", 1.0)
            try:
                w = float(w)
            except Exception:
                w = 1.0
            if S.has_edge(u, v):
                S[u][v]["weight"] = float(S[u][v].get("weight", 0.0)) + w
            else:
                S.add_edge(u, v, weight=w)
        H = S
    else:
        H = nx.Graph(H)

    for _, _, d in H.edges(data=True):
        w = d.get("weight", 1.0)
        try:
            w = float(w)
        except Exception:
            w = 1.0
        if not np.isfinite(w) or w <= 0:
            w = 1.0
        d["weight"] = w
    return H


def _rw_transition_matrix(G: nx.Graph, nodes: List) -> np.ndarray:
    """Row-stochastic transition matrix for a weighted random walk."""
    n = len(nodes)
    idx = {nodes[i]: i for i in range(n)}
    P = np.zeros((n, n), dtype=float)
    for u in nodes:
        i = idx[u]
        nbrs = list(G.neighbors(u))
        if not nbrs:
            P[i, i] = 1.0
            continue
        js = []
        ws = []
        for v in nbrs:
            w = float(G[u][v].get("weight", 1.0))
            js.append(idx[v])
            ws.append(w)
        s = float(np.sum(ws))
        if s <= 0:
            P[i, i] = 1.0
        else:
            for w, j in zip(ws, js):
                P[i, j] = w / s
    return P


def _pf_markov(G: nx.Graph, nodes: List, iters: int = 2000, tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """PF-Markov (Demetrius-style) transition matrix from Perron-Frobenius structure."""
    n = len(nodes)
    idx = {nodes[i]: i for i in range(n)}
    A = np.zeros((n, n), dtype=float)
    for u, v, d in G.edges(data=True):
        i = idx[u]
        j = idx[v]
        w = float(d.get("weight", 1.0))
        A[i, j] += w
        A[j, i] += w

    x = np.ones(n, dtype=float) / max(1, n)
    lam_old = 0.0
    for _ in range(int(max(10, iters))):
        y = A @ x
        norm = float(np.linalg.norm(y))
        if norm == 0:
            break
        x = y / norm
        lam = float((x @ (A @ x)) / max(1e-12, (x @ x)))
        if abs(lam - lam_old) < tol:
            break
        lam_old = lam

    v = np.abs(x) + 1e-15
    lam = float((v @ (A @ v)) / max(1e-12, (v @ v)))
    if not np.isfinite(lam) or lam <= 0:
        P = _rw_transition_matrix(G, nodes)
        pi = np.ones(n, dtype=float) / max(1, n)
        return P, pi

    P = np.zeros((n, n), dtype=float)
    for i in range(n):
        denom = lam * v[i]
        if denom <= 0:
            P[i, i] = 1.0
            continue
        row = (A[i, :] * v) / denom
        rs = float(row.sum())
        if rs <= 0:
            P[i, i] = 1.0
        else:
            P[i, :] = row / rs

    pi_raw = v * v
    pi = pi_raw / max(1e-12, float(pi_raw.sum()))
    return P, pi


def compute_energy_flow(
    G: nx.Graph,
    steps: int = 20,
    flow_mode: str = "rw",
    damping: float = 1.0,
    sources: Optional[List] = None,
) -> Tuple[Dict, Dict[Tuple, float]]:
    """Simulate energy diffusion and return node energies + edge flux values.

    Notes:
        - Uses dense matrices; prefer modest graph sizes for interactive 3D usage.
        - If sources are not provided, we seed energy at the highest-strength node.
    """
    H = _as_undirected_simple(G)
    nodes = list(H.nodes())
    if not nodes:
        return {}, {}

    if sources:
        srcs = [s for s in sources if s in H]
    else:
        srcs = []
    if not srcs:
        strengths = dict(H.degree(weight="weight"))
        srcs = [max(strengths, key=strengths.get)]

    steps = int(max(0, steps))

    if str(flow_mode).lower().startswith("evo"):
        P, _pi = _pf_markov(H, nodes)
    else:
        P = _rw_transition_matrix(H, nodes)

    n = len(nodes)
    idx = {nodes[i]: i for i in range(n)}
    e = np.zeros(n, dtype=float)
    for s in srcs:
        e[idx[s]] += 1.0 / len(srcs)

    damp = float(damping)
    if not np.isfinite(damp):
        damp = 1.0
    damp = max(0.0, min(1.0, damp))

    for _ in range(steps):
        e = e @ P
        if damp != 1.0:
            e *= damp

    node_energy = {nodes[i]: float(e[i]) for i in range(n)}

    edge_flux: Dict[Tuple, float] = {}
    for u, v in H.edges():
        iu = idx[u]
        iv = idx[v]
        f_uv = float(e[iu] * P[iu, iv])
        f_vu = float(e[iv] * P[iv, iu])
        edge_flux[(u, v)] = max(f_uv, f_vu)

    return node_energy, edge_flux


# ============================================================
# Energy flow (animated 3D)
# ============================================================
def simulate_energy_flow(
    G: nx.Graph,
    steps: int = 25,
    flow_mode: str = "rw",
    damping: float = 1.0,
    sources: Optional[List] = None,
) -> Tuple[List[Dict], List[Dict[Tuple, float]]]:
    """Per-step node energies + per-step edge fluxes (for Plotly frames).

    Output lengths: steps+1 for t = 0..steps.
    """
    H = _as_undirected_simple(G)
    nodes = list(H.nodes())
    if not nodes:
        return [], []

    # Sources: user-provided subset, else top-strength node.
    srcs: List = []
    if sources:
        srcs = [s for s in sources if s in H]
    if not srcs:
        strengths = dict(H.degree(weight="weight"))
        srcs = [max(strengths, key=strengths.get)] if strengths else [nodes[0]]

    steps = int(max(0, steps))

    if str(flow_mode).lower().startswith("evo"):
        P, _pi = _pf_markov(H, nodes)
    else:
        P = _rw_transition_matrix(H, nodes)

    n = len(nodes)
    idx = {nodes[i]: i for i in range(n)}

    e = np.zeros(n, dtype=float)
    for s in srcs:
        e[idx[s]] += 1.0 / float(len(srcs))

    damp = float(damping)
    if not np.isfinite(damp):
        damp = 1.0
    damp = max(0.0, min(1.0, damp))

    node_frames: List[Dict] = []
    edge_frames: List[Dict[Tuple, float]] = []

    def _snapshot(evec: np.ndarray) -> Tuple[Dict, Dict[Tuple, float]]:
        node_energy = {nodes[i]: float(evec[i]) for i in range(n)}
        edge_flux: Dict[Tuple, float] = {}
        for u, v in H.edges():
            iu = idx[u]
            iv = idx[v]
            f_uv = float(evec[iu] * P[iu, iv])
            f_vu = float(evec[iv] * P[iv, iu])
            edge_flux[(u, v)] = max(f_uv, f_vu)
        return node_energy, edge_flux

    ne0, ef0 = _snapshot(e)
    node_frames.append(ne0)
    edge_frames.append(ef0)

    for _ in range(steps):
        e = e @ P
        if damp != 1.0:
            e = e * damp
        ne, ef = _snapshot(e)
        node_frames.append(ne)
        edge_frames.append(ef)

    return node_frames, edge_frames


def make_energy_flow_figure_3d(
    G: nx.Graph,
    pos3d: dict,
    *,
    steps: int = 25,
    flow_mode: str = "rw",
    damping: float = 1.0,
    sources: Optional[List] = None,
    node_size: int = 6,
    edge_bins: int = 7,
    height: int = 820,
) -> go.Figure:
    """Animated 3D Plotly figure: node energy + edge flux overlay."""
    if G.number_of_nodes() == 0:
        return go.Figure()

    node_frames, edge_frames = simulate_energy_flow(
        G,
        steps=int(steps),
        flow_mode=str(flow_mode),
        damping=float(damping),
        sources=sources,
    )

    nodes = [n for n in G.nodes() if n in pos3d]
    xs = [pos3d[n][0] for n in nodes]
    ys = [pos3d[n][1] for n in nodes]
    zs = [pos3d[n][2] for n in nodes]

    e0 = node_frames[0] if node_frames else {}
    c0 = [float(e0.get(n, 0.0)) for n in nodes]

    node_trace = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers",
        marker=dict(
            size=int(node_size),
            color=c0,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="energy"),
        ),
        text=[str(n) for n in nodes],
        hoverinfo="text",
        name="nodes",
    )

    # Base edges (faint, always present)
    ex, ey, ez = [], [], []
    for u, v, _ in G.edges(data=True):
        if u not in pos3d or v not in pos3d:
            continue
        ex += [pos3d[u][0], pos3d[v][0], None]
        ey += [pos3d[u][1], pos3d[v][1], None]
        ez += [pos3d[u][2], pos3d[v][2], None]
    base_edges = go.Scatter3d(
        x=ex,
        y=ey,
        z=ez,
        mode="lines",
        line=dict(width=2, color="rgba(150,150,150,0.25)"),
        hoverinfo="none",
        name="edges",
    )

    overlay0 = _build_edge_overlay_traces(
        pos3d,
        edge_frames[0] if edge_frames else {},
        nbins=int(edge_bins),
    )

    frames = []
    for t in range(len(node_frames)):
        et = node_frames[t]
        ct = [float(et.get(n, 0.0)) for n in nodes]

        overlays = _build_edge_overlay_traces(
            pos3d,
            edge_frames[t] if t < len(edge_frames) else {},
            nbins=int(edge_bins),
        )

        fr_data = []
        fr_data.append(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers",
                marker=dict(
                    size=int(node_size),
                    color=ct,
                    colorscale="Viridis",
                    showscale=False,
                ),
                hoverinfo="skip",
                name="nodes",
            )
        )
        fr_data.append(base_edges)
        fr_data.extend(overlays)
        frames.append(go.Frame(data=fr_data, name=str(t)))

    fig = go.Figure(data=[node_trace, base_edges, *overlay0], frames=frames)
    fig.update_layout(
        template="plotly_dark",
        height=int(height),
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False,
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, title=""),
            yaxis=dict(showbackground=False, showticklabels=False, title=""),
            zaxis=dict(showbackground=False, showticklabels=False, title=""),
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=0.02,
                x=0.02,
                xanchor="left",
                yanchor="bottom",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {"frame": {"duration": 120, "redraw": True}, "fromcurrent": True}],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                y=0.02,
                x=0.18,
                len=0.78,
                pad={"b": 10, "t": 0},
                currentvalue={"prefix": "t="},
                steps=[
                    {"args": [[str(t)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}], "label": str(t), "method": "animate"}
                    for t in range(len(node_frames))
                ],
            )
        ],
    )
    return fig


def _build_edge_overlay_traces(
    pos3d: dict,
    edge_values: Dict[Tuple, float],
    nbins: int = 7,
) -> List[go.Scatter3d]:
    """Build colored edge traces by binning edge_values into nbins."""
    if not edge_values:
        return []

    vals = list(edge_values.values())
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = vmin - 1.0, vmax + 1.0

    bins = np.linspace(vmin, vmax, int(max(2, nbins)) + 1)
    colors = pc.sample_colorscale(
        "Viridis",
        [i / max(1, (len(bins) - 2)) for i in range(len(bins) - 1)],
    )

    traces: List[go.Scatter3d] = []
    for bi in range(len(bins) - 1):
        lo, hi = bins[bi], bins[bi + 1]
        xs, ys, zs = [], [], []
        for (u, v), val in edge_values.items():
            if u not in pos3d or v not in pos3d:
                continue
            if (val >= lo and val < hi) or (bi == len(bins) - 2 and val <= hi):
                x0, y0, z0 = pos3d[u]
                x1, y1, z1 = pos3d[v]
                xs.extend([x0, x1, None])
                ys.extend([y0, y1, None])
                zs.extend([z0, z1, None])
        if xs:
            traces.append(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="lines",
                    line=dict(color=colors[bi], width=2),
                    hoverinfo="none",
                    showlegend=False,
                )
            )
    return traces


def make_3d_traces(
    G: nx.Graph,
    pos3d: dict,
    show_scale: bool = False,
    kappa_edges_max: int = 220,
    edge_overlay: str = "ricci",
    flow_mode: str = "rw",
):
    """
    Returns (edge_traces, node_trace).
    edge_traces is a list: [base_grey_edges, neg_kappa_edges, pos_kappa_edges].

    edge_overlay:
      - 'ricci' (default): κ<0 red / κ>0 green
      - 'flux': stationary edge flux overlay
      - 'weight': edge weight overlay (log10)
      - 'confidence': edge confidence overlay
      - 'none': base grey edges only
    """
    if G.number_of_nodes() == 0:
        return [], None

    # nodes
    strength = dict(G.degree(weight="weight"))
    nodes = [n for n in G.nodes() if n in pos3d]
    xs = [pos3d[n][0] for n in nodes]
    ys = [pos3d[n][1] for n in nodes]
    zs = [pos3d[n][2] for n in nodes]
    colors = [strength.get(n, 0.0) for n in nodes]
    texts = [f"{str(n)}<br>strength={strength.get(n, 0.0):.3g}" for n in nodes]

    node_trace = go.Scatter3d(
        x=xs, y=ys, z=zs,
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

    # base grey edges (all)
    ex, ey, ez = [], [], []
    for a, b, _ in G.edges(data=True):
        if a not in pos3d or b not in pos3d:
            continue
        ex += [pos3d[a][0], pos3d[b][0], None]
        ey += [pos3d[a][1], pos3d[b][1], None]
        ez += [pos3d[a][2], pos3d[b][2], None]

    base_edges = go.Scatter3d(
        x=ex, y=ey, z=ez,
        mode="lines",
        line=dict(width=2, color="rgba(150,150,150,0.35)"),
        hoverinfo="none",
        name="edges",
    )

    overlay_mode = str(edge_overlay or "ricci").lower()
    if overlay_mode == "none":
        return [base_edges], node_trace

    if overlay_mode in ("weight", "confidence"):
        edge_values: Dict[Tuple, float] = {}
        for u, v, d in G.edges(data=True):
            if u not in pos3d or v not in pos3d:
                continue
            if overlay_mode == "confidence":
                raw = d.get("confidence", 1.0)
                try:
                    val = float(raw)
                except Exception:
                    val = 1.0
            else:
                raw = d.get("weight", 1.0)
                try:
                    w = float(raw)
                except Exception:
                    w = 1.0
                w = max(w, 1e-12)
                val = float(np.log10(w))
            edge_values[(u, v)] = val

        overlay_traces = _build_edge_overlay_traces(pos3d, edge_values)
        if overlay_traces:
            return [base_edges, *overlay_traces], node_trace

    if overlay_mode == "flux":
        _, edge_flux = compute_energy_flow(G, flow_mode=flow_mode)
        overlay_traces = _build_edge_overlay_traces(pos3d, edge_flux)
        if overlay_traces:
            return [base_edges, *overlay_traces], node_trace

    # Default: compute κ on a sample of edges and overlay colored subsets.
    edges = list(G.edges())
    if int(kappa_edges_max) <= 0:
        return [base_edges], node_trace
    if len(edges) > int(kappa_edges_max):
        rng = random.Random(42)
        edges = rng.sample(edges, int(kappa_edges_max))

    negx, negy, negz = [], [], []
    posx, posy, posz = [], [], []

    for a, b in edges:
        if a not in pos3d or b not in pos3d:
            continue
        try:
            k = ollivier_ricci_edge(G, a, b, max_support=60, cutoff=8.0)
        except Exception:
            k = None
        if k is None or not np.isfinite(k):
            continue
        if k < 0:
            negx += [pos3d[a][0], pos3d[b][0], None]
            negy += [pos3d[a][1], pos3d[b][1], None]
            negz += [pos3d[a][2], pos3d[b][2], None]
        elif k > 0:
            posx += [pos3d[a][0], pos3d[b][0], None]
            posy += [pos3d[a][1], pos3d[b][1], None]
            posz += [pos3d[a][2], pos3d[b][2], None]

    neg_edges = go.Scatter3d(
        x=negx, y=negy, z=negz,
        mode="lines",
        line=dict(width=4, color="rgba(255,80,80,0.85)"),
        hoverinfo="none",
        name="κ<0",
    )

    pos_edges = go.Scatter3d(
        x=posx, y=posy, z=posz,
        mode="lines",
        line=dict(width=4, color="rgba(80,255,140,0.85)"),
        hoverinfo="none",
        name="κ>0",
    )

    return [base_edges, neg_edges, pos_edges], node_trace
