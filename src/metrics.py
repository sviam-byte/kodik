# src/metrics.py
import math
import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import networkx as nx
import scipy.sparse.linalg as spla
import plotly.graph_objects as go

from networkx.algorithms.community import modularity, louvain_communities

from src.config import APPROX_EFFICIENCY_K, RICCI_CUTOFF, RICCI_MAX_SUPPORT
from src.types import GraphMetrics
from src.utils import as_simple_undirected, safe_float

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


def approx_weighted_efficiency(G: nx.Graph, sources_k: int = APPROX_EFFICIENCY_K, seed: int = 0) -> float:
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
    curvature_max_support: int = RICCI_MAX_SUPPORT,
    curvature_cutoff: float = RICCI_CUTOFF,
    **kwargs,
) -> GraphMetrics:
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


def _normalize_edge_weights(G: nx.Graph) -> nx.Graph:
    """Ensure every edge has a positive finite weight to keep metrics stable."""
    for _, _, d in G.edges(data=True):
        w = safe_float(d.get("weight", 1.0), 1.0)
        if not np.isfinite(w) or w <= 0:
            w = 1.0
        d["weight"] = w
    return G


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
    H = _normalize_edge_weights(as_simple_undirected(G))
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
# Energy flow (animated 3D): PHYSICAL pressure/flow
# ============================================================
def _simulate_energy_physical(
    G: nx.Graph,
    steps: int,
    damping: float,
    sources: Optional[List],
    cap_mode: str = "strength",
    injection: float = 0.15,
    leak: float = 0.02,
) -> Tuple[List[Dict], List[Dict[Tuple, float]]]:
    """Pressure/flow simulator with a stable dt for smoother dynamics."""
    H = _normalize_edge_weights(as_simple_undirected(G))
    nodes = list(H.nodes())
    if not nodes:
        return [], []

    # Capacity (degree or weighted strength).
    if cap_mode == "degree":
        cap = {n: float(H.degree(n)) for n in nodes}
    else:
        cap = {n: float(H.degree(n, weight="weight")) for n in nodes}

    # Avoid division by zero in pressure.
    for n in cap:
        if cap[n] <= 0:
            cap[n] = 1.0

    # Sources: user-provided subset, else top-capacity node.
    srcs = []
    if sources:
        srcs = [s for s in sources if s in H]
    if not srcs:
        srcs = [max(cap, key=cap.get)] if cap else [nodes[0]]

    # Initialize energy with a noticeable impulse at sources.
    E = {n: 0.0 for n in nodes}
    for s in srcs:
        E[s] = 10.0

    # === Physical hack: dt ===
    # dt controls flow smoothness; too large makes the system unstable.
    dt = 0.15

    node_frames = []
    edge_frames = []

    for t in range(steps + 1):
        # Snapshot current energies.
        node_frames.append(E.copy())

        # Pressure = Energy / Capacity.
        P = {n: E[n] / cap[n] for n in nodes}

        dE = {n: 0.0 for n in nodes}
        edge_flux = {}

        for u, v, d in H.edges(data=True):
            w = d.get("weight", 1.0)
            # Flow = pressure difference * conductance (weight).
            # dt keeps updates smooth so nodes don't go negative in one step.
            flux = w * (P[u] - P[v]) * dt

            dE[u] -= flux
            dE[v] += flux
            edge_flux[(u, v)] = abs(flux)

        edge_frames.append(edge_flux)

        if t == steps:
            break

        # Apply updates.
        for n in nodes:
            E[n] += dE[n]

            # Injection (constant input at sources).
            if n in srcs:
                E[n] += float(injection) * dt * 10.0

            # Leak & damping.
            E[n] *= damping
            E[n] -= float(leak) * dt
            if E[n] < 0:
                E[n] = 0.0

    return node_frames, edge_frames


# ============================================================
# Energy flow (animated 3D)
# ============================================================
def simulate_energy_flow(
    G: nx.Graph,
    steps: int = 25,
    flow_mode: str = "rw",
    damping: float = 1.0,
    sources: Optional[List] = None,
    phys_injection: float = 0.15,
    phys_leak: float = 0.02,
    phys_cap_mode: str = "strength",
    rw_impulse: bool = True,
) -> Tuple[List[Dict], List[Dict[Tuple, float]]]:
    """Per-step node energies + per-step edge fluxes (for Plotly frames).

    Output lengths: steps+1 for t = 0..steps.
    Physical mode parameters are forwarded to the pressure/flow simulator.
    For RW/Evo modes, phys_injection controls optional reinjection at sources:
      - rw_impulse=True: inject only once at t=0 (wave that dissipates).
      - rw_impulse=False: inject each step (continuous forcing).
    """
    fm = str(flow_mode).lower().strip()
    if fm in ("phys", "pressure", "flow"):
        return _simulate_energy_physical(
            G,
            steps=int(steps),
            damping=float(damping),
            sources=sources,
            cap_mode=str(phys_cap_mode),
            injection=float(phys_injection),
            leak=float(phys_leak),
        )

    H = _normalize_edge_weights(as_simple_undirected(G))
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

    inj = float(phys_injection)
    if not np.isfinite(inj):
        inj = 0.0
    inj = max(0.0, min(1.0, inj))

    if bool(rw_impulse) and inj > 0.0:
        add = inj / float(len(srcs))
        for s in srcs:
            e[idx[s]] += add
        inj = 0.0

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
        if inj > 0.0:
            add = inj / float(len(srcs))
            for s in srcs:
                e[idx[s]] += add
        ne, ef = _snapshot(e)
        node_frames.append(ne)
        edge_frames.append(ef)

    return node_frames, edge_frames


def make_energy_flow_figure_3d(
    G: nx.Graph,
    pos3d: dict,
    *,
    steps: int = 25,
    node_frames: Optional[List[Dict]] = None,
    edge_frames: Optional[List[Dict[Tuple, float]]] = None,
    # Старые аргументы оставлены для совместимости с существующими вызовами.
    flow_mode: str = "phys",
    damping: float = 1.0,
    sources: Optional[List] = None,
    phys_injection: float = 0.15,
    phys_leak: float = 0.02,
    phys_cap_mode: str = "strength",
    edge_bins: int = 7,
    hotspot_q: float = 0.92,
    hotspot_size_mult: float = 4.0,
    base_node_opacity: float = 0.25,
    rw_impulse: bool = True,
    show_labels: bool = False,
    height: int = 820,
    frame_stride: int = 2,
    # Новые параметры для интерактивной и яркой анимации.
    anim_duration: int = 150,
    node_size: int = 6,
    vis_contrast: float = 2.0,
    vis_clip: float = 0.05,
    vis_log: bool = True,
    max_edges_viz: int = 2000,
    edge_subset_mode: str = "top_flux",
    **kwargs,
) -> go.Figure:
    """Build an interactive 3D energy-flow animation optimized for mouse control.

    Мы увеличиваем длительность кадров/перехода и оставляем redraw=True,
    чтобы браузер успевал обрабатывать мышь во время анимации 3D.
    """
    if G.number_of_nodes() == 0:
        return go.Figure()
    # Reuse precomputed frames when provided to avoid resimulating on UI-only changes.
    if node_frames is None or edge_frames is None:
        node_frames, edge_frames = simulate_energy_flow(
            G,
            steps=int(steps),
            flow_mode=str(flow_mode),
            damping=float(damping),
            sources=sources,
            phys_injection=float(phys_injection),
            phys_leak=float(phys_leak),
            phys_cap_mode=str(phys_cap_mode),
            rw_impulse=bool(rw_impulse),
        )

    # --- 1. Подготовка данных ---
    nodes = [n for n in G.nodes() if n in pos3d]
    if not nodes:
        return go.Figure()

    node_frames = node_frames or [{}]
    if edge_frames is None:
        edge_frames = [{} for _ in range(len(node_frames))]
    elif len(edge_frames) < len(node_frames):
        # Гарантируем, что индексы кадров не выйдут за пределы списка.
        edge_frames = list(edge_frames) + [{} for _ in range(len(node_frames) - len(edge_frames))]

    xs = [pos3d[n][0] for n in nodes]
    ys = [pos3d[n][1] for n in nodes]
    zs = [pos3d[n][2] for n in nodes]

    # --- 2. Логика цвета (ОГНЕННАЯ) ---
    def _get_colors(energy_map: Dict) -> np.ndarray:
        """Вернуть нормированные цвета узлов в [0, 1] с гамма-контрастом."""
        vals = np.array([energy_map.get(n, 0.0) for n in nodes], dtype=float)
        if vis_log:
            vals = np.log1p(np.maximum(vals, 0.0))

        # Clip & Normalize: фиксируем 0, режем только верх.
        v_max = np.quantile(vals, 1.0 - vis_clip) if vis_clip < 1.0 else np.max(vals)
        if not np.isfinite(v_max) or v_max <= 0:
            v_max = 1.0
        vals = np.clip(vals / v_max, 0.0, 1.0)

        # Gamma contrast (делаем средние значения ярче).
        vals = np.power(vals, 1.0 / float(vis_contrast))
        return vals

    # --- 3. Базовый слой (Скелет) ---
    edges_all = []
    flux_sum: Dict[Tuple, float] = {}

    # Считаем суммарный поток для выбора топ-ребер.
    for fr in edge_frames:
        for k, v in fr.items():
            flux_sum[k] = flux_sum.get(k, 0.0) + float(v)

    for u, v, d in G.edges(data=True):
        if u in pos3d and v in pos3d:
            w = float(d.get("weight", 1.0) or 1.0)
            f = flux_sum.get((u, v), flux_sum.get((v, u), 0.0))
            edges_all.append((u, v, w, f))

    # Сортировка и обрезка.
    edge_subset_mode = str(edge_subset_mode).lower().strip()
    if edge_subset_mode == "top_flux":
        edges_all.sort(key=lambda x: x[3], reverse=True)
    elif edge_subset_mode == "top_weight":
        edges_all.sort(key=lambda x: x[2], reverse=True)

    max_edges_viz = int(max(0, max_edges_viz))
    edges_viz = edges_all if edge_subset_mode == "all" else edges_all[:max_edges_viz or len(edges_all)]
    if edge_subset_mode == "all" and max_edges_viz:
        edges_viz = edges_viz[:max_edges_viz]

    # Статический фон (серые тонкие линии).
    base_x, base_y, base_z = [], [], []
    for u, v, _, _ in edges_viz:
        base_x.extend([pos3d[u][0], pos3d[v][0], None])
        base_y.extend([pos3d[u][1], pos3d[v][1], None])
        base_z.extend([pos3d[u][2], pos3d[v][2], None])

    trace_edges_base = go.Scatter3d(
        x=base_x,
        y=base_y,
        z=base_z,
        mode="lines",
        line=dict(color="rgba(255,255,255,0.05)", width=1),
        hoverinfo="none",
        name="structure",
    )

    # Начальное состояние узлов.
    c0 = _get_colors(node_frames[0])
    trace_nodes = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers",
        marker=dict(
            size=int(node_size),
            color=c0,
            colorscale="Blackbody",
            cmin=0,
            cmax=1,
            showscale=True,
            colorbar=dict(title="Energy", thickness=15, x=0),
        ),
        text=[f"{n}" for n in nodes],
        name="nodes",
    )

    # --- 4. Кадры анимации ---
    frames = []
    step_stride = max(1, int(frame_stride))

    for i in range(0, len(node_frames), step_stride):
        # Узлы.
        c_i = _get_colors(node_frames[i])

        # Ребра (активный поток).
        flux_map = edge_frames[i] if i < len(edge_frames) else {}

        f_vals = [flux_map.get((u, v), flux_map.get((v, u), 0.0)) for u, v, _, _ in edges_viz]
        f_max = max(f_vals) if f_vals else 1.0
        if f_max <= 0:
            f_max = 1.0

        # В этом кадре собираем координаты для "горячих" ребер.
        hot_x, hot_y, hot_z = [], [], []
        for (u, v, _, _), f_val in zip(edges_viz, f_vals):
            norm_f = float(f_val) / float(f_max)
            if norm_f > 0.05:
                hot_x.extend([pos3d[u][0], pos3d[v][0], None])
                hot_y.extend([pos3d[u][1], pos3d[v][1], None])
                hot_z.extend([pos3d[u][2], pos3d[v][2], None])

        # Трейс 0: Nodes, Трейс 1: Base Edges (не меняется), Трейс 2: Active Flow.
        frames.append(
            go.Frame(
                data=[
                    go.Scatter3d(marker=dict(color=c_i)),
                    go.Scatter3d(),
                    go.Scatter3d(
                        x=hot_x,
                        y=hot_y,
                        z=hot_z,
                        mode="lines",
                        line=dict(color="#ffaa00", width=4),
                        opacity=0.6,
                    ),
                ],
                name=str(i),
            )
        )

    # --- 5. Сборка фигуры ---
    trace_flow = go.Scatter3d(
        x=[],
        y=[],
        z=[],
        mode="lines",
        line=dict(color="#ffaa00", width=4),
        name="flow",
    )

    fig = go.Figure(data=[trace_nodes, trace_edges_base, trace_flow], frames=frames)

    # --- 6. Настройки анимации (Ключ к успеху) ---
    fig.update_layout(
        template="plotly_dark",
        height=int(kwargs.get("height", height)),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=0.05,
                x=0.05,
                buttons=[
                    dict(
                        label="▶ PLAY",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=int(anim_duration), redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=int(anim_duration), easing="linear"),
                            ),
                        ],
                    ),
                    dict(
                        label="⏸ PAUSE",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                currentvalue=dict(prefix="Step: "),
                steps=[
                    dict(
                        args=[[str(k)], dict(frame=dict(duration=0, redraw=True), mode="immediate")],
                        label=str(k),
                        method="animate",
                    )
                    for k in range(len(frames))
                ],
            )
        ],
    )

    return fig


def _build_edge_overlay_traces(
    pos3d: dict,
    edge_values: Dict[Tuple, float],
    nbins: int = 7,
    allowed_edges: Optional[Set[Tuple]] = None,
    vis_contrast: Optional[float] = None,
    vis_clip: Optional[float] = None,
    vis_log: Optional[bool] = None,
) -> List[go.Scatter3d]:
    """Build colored edge traces by binning edge_values into nbins.

    When allowed_edges is provided, only those edges (undirected) are rendered.
    If vis_* settings are provided, apply robust scaling to emphasize differences.
    Colors stay in a warm palette to align with the energy-flow view.
    """
    if not edge_values:
        return []

    if allowed_edges is not None:
        edge_values = {k: v for k, v in edge_values.items() if k in allowed_edges or (k[1], k[0]) in allowed_edges}
        if not edge_values:
            return []

    vals = list(edge_values.values())
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = vmin - 1.0, vmax + 1.0

    def _as_float_array(values: List[float]) -> np.ndarray:
        """Return finite float array; invalid entries become zeros."""
        arr = np.array([float(v) for v in values], dtype=float)
        arr[~np.isfinite(arr)] = 0.0
        return arr

    def _robust_unit(arr: np.ndarray, clip: float) -> np.ndarray:
        """Normalize to [0,1] while keeping 0 as the true minimum."""
        clip = max(0.0, min(0.49, float(clip)))
        hi = np.quantile(arr, 1.0 - clip)
        if not np.isfinite(hi) or hi <= 0.0:
            hi = np.max(arr) + 1e-12
        return np.clip(arr / hi, 0.0, 1.0)

    def _boost01(arr: np.ndarray, contrast: float) -> np.ndarray:
        """Gamma-like contrast on [0,1] values."""
        contrast = float(contrast)
        if not np.isfinite(contrast) or contrast <= 0:
            contrast = 1.0
        return np.power(arr, 1.0 / contrast)

    if vis_contrast is None and vis_clip is None and vis_log is None:
        bins = np.linspace(vmin, vmax, int(max(2, nbins)) + 1)
        # Используем огненную шкалу: от прозрачного красного к ярко-желтому
        colors = [
            f"rgba(255, 50, 50, {0.3 + 0.7 * i/(len(bins)-2)})"
            for i in range(len(bins) - 1)
        ]
        normed = None
    else:
        arr = _as_float_array(vals)
        if bool(vis_log):
            arr = np.log1p(np.maximum(arr, 0.0))
        arr = _robust_unit(arr, float(vis_clip or 0.0))
        arr = _boost01(arr, float(vis_contrast or 1.0))
        normed = arr
        bins = np.linspace(0.0, 1.0, int(max(2, nbins)) + 1)
        # Используем огненную шкалу: от прозрачного красного к ярко-желтому
        colors = [
            f"rgba(255, 50, 50, {0.3 + 0.7 * i/(len(bins)-2)})"
            for i in range(len(bins) - 1)
        ]

    traces: List[go.Scatter3d] = []
    for bi in range(len(bins) - 1):
        lo, hi = bins[bi], bins[bi + 1]
        xs, ys, zs = [], [], []
        for idx, ((u, v), val) in enumerate(edge_values.items()):
            if u not in pos3d or v not in pos3d:
                continue
            val_check = normed[idx] if normed is not None else val
            if (val_check >= lo and val_check < hi) or (bi == len(bins) - 2 and val_check <= hi):
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
                    line=dict(color=colors[bi], width=0.8),
                    opacity=0.25,
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
