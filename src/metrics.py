import random
import numpy as np
import networkx as nx
import scipy.sparse.linalg as spla
import plotly.graph_objects as go

from networkx.algorithms.community import modularity, louvain_communities


def add_dist_attr(G: nx.Graph) -> nx.Graph:
    """Copy a graph and add inverse-weight distance attribute for path algorithms."""
    H = G.copy()
    for _, _, d in H.edges(data=True):
        w = float(d.get("weight", 1.0))
        d["dist"] = 1.0 / w if w > 0 else 1e12
    return H


def approx_weighted_efficiency(G: nx.Graph, sources_k: int, seed: int) -> float:
    """
    E_w = (1/(N(N-1))) * sum_{i!=j} 1/d_ij, dist = 1/weight
    Аппрокс по k источникам: k Dijkstra вместо N Dijkstra.
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
    λ2 для связного графа (взвешенный лапласиан).
    Малые графы -> dense eigvalsh (точно).
    Большие -> sparse eigsh около 0.
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
    """Return algebraic connectivity for a (connected) LCC graph."""
    if G.number_of_nodes() < 3 or G.number_of_edges() == 0:
        return 0.0
    # assume already LCC or connected-ish
    if not nx.is_connected(G):
        return 0.0
    return lambda2_robust_connected(G)


def compute_modularity_louvain(G: nx.Graph, seed: int) -> float:
    """Compute Louvain modularity with version-safe seed handling."""
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
    Энтропия распределения степеней (weighted degree strength тоже можно).
    Здесь: обычная степень.
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
    Дешёвая оценка диаметра: BFS от нескольких случайных узлов.
    Возвращает максимум найденных эксцентриситетов.
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
    """Compute a suite of scalar metrics for analysis dashboards."""
    N = G.number_of_nodes()
    E = G.number_of_edges()
    C = nx.number_connected_components(G) if N > 0 else 0
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

    return {
        "N": N,
        "E": E,
        "C": C,
        "density": float(dens),
        "avg_degree": float(avg_deg),
        "beta": beta,
        "lcc_size": int(lcc_size),
        "lcc_frac": float(lcc_frac),
        "eff_w": float(eff_w),
        "l2_lcc": float(l2),
        "tau_lcc": float(tau),
        "lmax": float(lmax),
        "thresh": float(thresh),
        "mod": float(Q),
        "entropy_deg": float(ent),
        "assortativity": float(assort) if np.isfinite(assort) else 0.0,
        "clustering": float(clust) if np.isfinite(clust) else 0.0,
        "diameter_approx": int(diam) if diam is not None else None,
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
