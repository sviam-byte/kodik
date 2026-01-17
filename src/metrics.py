import math
import random
import numpy as np
import networkx as nx
import scipy.sparse.linalg as spla


# -------------------------
# Helpers
# -------------------------
def add_dist_attr(G: nx.Graph) -> nx.Graph:
    """Add distance attribute as inverse weight for weighted shortest paths."""
    H = G.copy()
    for _, _, d in H.edges(data=True):
        w = float(d.get("weight", 1.0))
        d["dist"] = 1.0 / w if w > 0 else 1e12
    return H


def approx_weighted_efficiency(G: nx.Graph, sources_k: int, seed: int) -> float:
    """
    Weighted efficiency:
      E = (1/(N(N-1))) * sum_{i!=j} 1/d_ij
    where d_ij is Dijkstra distance using dist=1/weight.

    Approximation:
      sample k sources and scale.
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

    est_full = total * (N / max(1, k))
    return float(est_full / denom)


def spectral_radius_weighted_adjacency(G: nx.Graph) -> float:
    """Largest eigenvalue of weighted adjacency matrix."""
    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        return 0.0

    A = nx.adjacency_matrix(G, weight="weight").astype(float)

    # Lanczos
    v = spla.eigsh(A, k=1, which="LM", return_eigenvectors=False)[0]
    return float(np.real(v))


def lambda2_connected(G: nx.Graph, eps: float = 1e-10) -> float:
    """
    Algebraic connectivity (weighted Laplacian):
      L = D - A
      λ2 = 2nd smallest eigenvalue (for connected graph)

    For large graphs:
      compute few eigenvalues near 0 using shift-invert.
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

    vals = spla.eigsh(L, k=6, sigma=0.0, which="LM", return_eigenvectors=False)
    vals = np.sort(np.real(vals))
    for v in vals:
        if v > eps:
            return float(v)
    return 0.0


def lambda2_global(G: nx.Graph) -> float:
    """Lambda-2 for the whole graph if connected."""
    if G.number_of_nodes() < 3 or G.number_of_edges() == 0:
        return 0.0
    if not nx.is_connected(G):
        return 0.0
    return lambda2_connected(G)


def lambda2_on_lcc(G: nx.Graph) -> float:
    """Lambda-2 on the largest connected component."""
    if G.number_of_nodes() == 0:
        return 0.0
    comp = max(nx.connected_components(G), key=len)
    H = G.subgraph(comp).copy()
    if H.number_of_nodes() < 3 or H.number_of_edges() == 0:
        return 0.0
    if not nx.is_connected(H):
        return 0.0
    return lambda2_connected(H)


# -------------------------
# Entropies / Complexity
# -------------------------
def degree_entropy(G: nx.Graph, weighted: bool = True) -> float:
    """
    Shannon entropy of strength distribution (if weighted) or degree distribution.
      p_i = s_i / sum s
      H = -sum p_i log p_i
    """
    if G.number_of_nodes() == 0:
        return 0.0

    if weighted:
        xs = np.array([float(v) for _, v in G.degree(weight="weight")], dtype=float)
    else:
        xs = np.array([float(v) for _, v in G.degree()], dtype=float)

    S = xs.sum()
    if S <= 0:
        return 0.0

    p = xs / S
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def von_neumann_entropy_laplacian(G: nx.Graph, normalized: bool = True, k: int = 64) -> float:
    """
    Von Neumann entropy based on Laplacian spectrum:
      rho = L / Tr(L)
      S = -Tr(rho log rho) = -sum_i mu_i log mu_i

    For large n:
      approximate using k smallest eigenvalues.
      Note: strict VN entropy needs full spectrum; this is an approximation.
    """
    n = G.number_of_nodes()
    if n < 2 or G.number_of_edges() == 0:
        return 0.0

    if normalized:
        L = nx.normalized_laplacian_matrix(G, weight="weight").astype(float)
    else:
        L = nx.laplacian_matrix(G, weight="weight").astype(float)

    tr = float(L.diagonal().sum())
    if tr <= 0:
        return 0.0

    # Small graphs: exact spectrum
    if n <= 400:
        vals = np.linalg.eigvalsh(L.toarray())
        mu = np.real(vals) / tr
        mu = mu[mu > 1e-15]
        return float(-(mu * np.log(mu)).sum())

    # Large graphs: approximate with k eigenvalues near 0 + a crude tail bound
    kk = min(int(k), n - 1)
    vals = spla.eigsh(L, k=kk, which="SM", return_eigenvectors=False)
    mu = np.real(vals) / tr
    mu = mu[mu > 1e-15]
    return float(-(mu * np.log(mu)).sum())


# -------------------------
# Other topology metrics
# -------------------------
def beta_cycles(G: nx.Graph) -> int:
    """Number of independent cycles (E - N + C)."""
    return int(G.number_of_edges() - G.number_of_nodes() + nx.number_connected_components(G))


def lcc_fraction(G: nx.Graph, N0: int) -> float:
    """Largest connected component fraction with respect to N0."""
    if G.number_of_nodes() == 0 or N0 <= 0:
        return 0.0
    lcc = len(max(nx.connected_components(G), key=len))
    return float(lcc) / float(N0)


def clustering_weighted(G: nx.Graph) -> float:
    """
    Average weighted clustering coefficient.
    networkx has clustering(G, weight="weight") for weighted.
    """
    if G.number_of_nodes() < 3:
        return 0.0
    vals = nx.clustering(G, weight="weight").values()
    xs = np.array(list(vals), dtype=float)
    return float(xs.mean()) if len(xs) else 0.0


def assortativity_strength(G: nx.Graph) -> float:
    """
    Degree assortativity on strength is not built-in directly.
    We'll use weighted degree as "degree" proxy:
      assortativity on node attribute = corr(attr(u), attr(v)) over edges
    """
    if G.number_of_edges() == 0:
        return 0.0

    s = dict(G.degree(weight="weight"))
    x = []
    y = []
    for u, v in G.edges():
        x.append(float(s.get(u, 0.0)))
        y.append(float(s.get(v, 0.0)))

    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def calculate_metrics(G: nx.Graph, eff_sources_k: int, seed: int) -> dict:
    """Compute base and heavy metrics for a given graph."""
    N = G.number_of_nodes()
    E = G.number_of_edges()
    C = nx.number_connected_components(G) if N > 0 else 0

    lmax = spectral_radius_weighted_adjacency(G) if E > 0 else 0.0
    thresh = (1.0 / lmax) if lmax > 0 else 0.0

    l2g = lambda2_global(G)
    l2l = lambda2_on_lcc(G)

    eff_w = approx_weighted_efficiency(G, sources_k=int(eff_sources_k), seed=int(seed))
    beta = beta_cycles(G)

    clust = clustering_weighted(G)
    assort = assortativity_strength(G)

    Hdeg = degree_entropy(G, weighted=True)
    Svn = von_neumann_entropy_laplacian(G, normalized=True, k=64)

    return {
        "N": N,
        "E": E,
        "C": C,
        "beta": beta,
        "eff_w": eff_w,
        "l2_global": l2g,
        "tau_global": (1.0 / l2g) if l2g > 0 else float("inf"),
        "l2_lcc": l2l,
        "tau_lcc": (1.0 / l2l) if l2l > 0 else float("inf"),
        "lmax": lmax,
        "thresh_SIS": thresh,
        "clust_w": clust,
        "assort_strength": assort,
        "H_strength": Hdeg,
        "S_vn_laplacian": Svn,
    }
