import time
import uuid
import hashlib
import random
import json
import base64

import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse.linalg as spla
import plotly.graph_objects as go
import streamlit as st

from scipy.integrate import trapz
from networkx.algorithms.community import louvain_communities, modularity as nx_modularity
from networkx.algorithms.centrality import betweenness_centrality
from networkx.algorithms.core import core_number
from networkx.algorithms.assortativity import degree_assortativity_coefficient
from networkx.algorithms.cluster import average_clustering
from networkx.algorithms.distance_measures import diameter
from networkx.algorithms.shortest_paths import average_shortest_path_length
from networkx.generators.stochastic_block_model import stochastic_block_model

# Assuming src modules are available; if not, integrate or stub
# For this rewrite, I'll integrate key functions from src into helpers where needed.

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Прикольчик: Network Resilience Analyzer", layout="wide", page_icon="💀")
st.title("Прикольчик: Network Resilience Analyzer")

# Sticky topbar CSS (enhanced for better UX)
st.markdown(
    """
    <style>
      /* Sticky container */
      div[data-testid="stVerticalBlock"] > div:has(> div.sticky-topbar){
        position: sticky;
        top: 0;
        z-index: 999;
        background: rgba(15, 17, 22, 0.96);
        backdrop-filter: blur(8px);
        border-bottom: 1px solid rgba(255,255,255,0.08);
        padding: 0.5rem;
      }

      .sticky-topbar .stButton>button { height: 2.4rem; min-width: 80px; }
      .sticky-topbar .stTextInput input { height: 2.4rem; }
      .sticky-topbar .stSelectbox div[data-baseweb="select"] { min-height: 2.4rem; }
      .sticky-topbar .stNumberInput input { height: 2.4rem; }
      .sticky-topbar .stSlider div { margin-top: 0.5rem; }

      /* Tooltips for ? icons */
      .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
      }
      .tooltip .tooltiptext {
        visibility: hidden;
        width: 220px;
        background-color: #333;
        color: #fff;
        text-align: left;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -110px;
        opacity: 0;
        transition: opacity 0.3s;
      }
      .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
      }

      /* Reduce gaps */
      .sticky-topbar [data-testid="stVerticalBlock"] { gap: 0.25rem; }
      section.main > div { padding-top: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Session State
# =========================
def _init_state():
    if "graphs" not in st.session_state:
        st.session_state["graphs"] = {}  # gid -> {id, name, source, tags, edges(df), created_at}

    if "experiments" not in st.session_state:
        st.session_state["experiments"] = []  # list of {id, name, graph_id, attack_kind, params, history(df), created_at}

    if "active_graph_id" not in st.session_state:
        st.session_state["active_graph_id"] = None

    if "seed_top" not in st.session_state:
        st.session_state["seed_top"] = 42

    if "last_uploaded_fingerprint" not in st.session_state:
        st.session_state["last_uploaded_fingerprint"] = None


_init_state()

# =========================
# Helpers
# =========================
def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def add_graph_to_workspace(name: str, df_edges: pd.DataFrame, source: str, tags: dict | None = None) -> str:
    gid = new_id("G")
    st.session_state["graphs"][gid] = {
        "id": gid,
        "name": name,
        "source": source,
        "tags": tags or {},
        "edges": df_edges.copy(),
        "created_at": time.time(),
    }
    st.session_state["active_graph_id"] = gid
    return gid


def get_active_graph_entry():
    gid = st.session_state.get("active_graph_id")
    if gid is None:
        return None
    return st.session_state["graphs"].get(gid)


def sorted_graph_ids():
    graphs = st.session_state["graphs"]
    return sorted(list(graphs.keys()), key=lambda k: graphs[k].get("created_at", 0.0))


def build_graph_from_edges(df_edges: pd.DataFrame, src_col: str, dst_col: str, weighted: bool = True, directed: bool = False) -> nx.Graph:
    graph_class = nx.DiGraph if directed else nx.Graph
    edge_attr = ["weight", "confidence"] if weighted else None
    G = nx.from_pandas_edgelist(
        df_edges,
        source=src_col,
        target=dst_col,
        edge_attr=edge_attr,
        create_using=graph_class(),
    )
    # Normalize attributes
    for _, _, d in G.edges(data=True):
        w = float(d.get("weight", 1.0))
        c = float(d.get("confidence", 0.0))
        d["weight"] = max(1e-12, w) if w > 0 else 1e-12
        d["confidence"] = c if np.isfinite(c) else 0.0
    # Remove self-loops and collapse multi-edges if not directed
    if not directed:
        G.remove_edges_from(nx.selfloop_edges(G))
    return G


def filter_edges(df_edges: pd.DataFrame, src_col: str, dst_col: str, min_conf: int, min_weight: float) -> pd.DataFrame:
    df = df_edges.copy()
    df = df[df["confidence"] >= min_conf]
    df = df[df["weight"] >= min_weight]
    df = df.dropna(subset=[src_col, dst_col, "confidence", "weight"])
    df = df[df["weight"] > 0]
    return df


def lcc_subgraph(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return G.copy()
    comp = max(nx.connected_components(G), key=len)
    return G.subgraph(comp).copy()


def graph_summary(G: nx.Graph) -> str:
    N = G.number_of_nodes()
    E = G.number_of_edges()
    C = nx.number_connected_components(G) if N > 0 else 0
    dens = nx.density(G) if N > 1 else 0.0
    selfloops = nx.number_of_selfloops(G)
    isolates = sum(1 for n in G.nodes() if G.degree(n) == 0)
    return f"N={N}\nE={E}\nComponents={C}\nDensity={dens:.6g}\nSelf-loops={selfloops}\nIsolates={isolates} ({isolates/N:.2%} if N>0)"


def fingerprint_upload(uploaded) -> str:
    b = uploaded.getvalue()
    return hashlib.md5(b).hexdigest() + f":{uploaded.name}:{len(b)}"


def calculate_light_metrics(G: nx.Graph) -> dict:
    N = G.number_of_nodes()
    E = G.number_of_edges()
    C = nx.number_connected_components(G) if N > 0 else 0
    dens = nx.density(G) if N > 1 else 0.0
    avg_deg = (2 * E / N) if N > 0 else 0.0
    beta = E - N + C if N > 0 else 0
    lcc_size = len(max(nx.connected_components(G), key=len)) if N > 0 else 0
    lcc_frac = lcc_size / N if N > 0 else 0.0
    assort = degree_assortativity_coefficient(G) if N > 2 and E > 0 else 0.0
    clust = average_clustering(G) if N > 2 and E > 0 else 0.0
    return {
        "N": N,
        "E": E,
        "C": C,
        "density": dens,
        "avg_degree": avg_deg,
        "beta": beta,
        "lcc_size": lcc_size,
        "lcc_frac": lcc_frac,
        "assortativity": assort if np.isfinite(assort) else 0.0,
        "clustering": clust if np.isfinite(clust) else 0.0,
    }


def calculate_heavy_metrics(G: nx.Graph, eff_sources_k: int, seed: int) -> dict:
    lcc = lcc_subgraph(G)
    l2 = lambda2_robust_connected(lcc)
    tau = (1.0 / l2) if l2 > 0 else float("inf")
    lmax = spectral_radius_weighted_adjacency(G)
    thresh = (1.0 / lmax) if lmax > 0 else 0.0
    eff_w = approx_weighted_efficiency(G, sources_k=eff_sources_k, seed=seed)
    Q = compute_modularity_louvain(G, seed=seed)
    ent = degree_entropy(G)
    diam = approx_diameter_lcc(G, seed=seed, samples=16)
    kcore_max = max(core_number(G).values()) if G.number_of_nodes() > 0 else 0
    return {
        "eff_w": eff_w,
        "l2_lcc": l2,
        "tau_lcc": tau,
        "lmax": lmax,
        "thresh": thresh,
        "mod": Q,
        "entropy_deg": ent,
        "diameter_approx": diam,
        "kcore_max": kcore_max,
    }


def lambda2_robust_connected(G: nx.Graph, eps: float = 1e-10) -> float:
    n = G.number_of_nodes()
    if n < 3 or G.number_of_edges() == 0:
        return 0.0
    if not nx.is_connected(G):
        return 0.0
    L = nx.laplacian_matrix(G, weight="weight").astype(float)
    if n <= 500:
        vals = np.linalg.eigvalsh(L.toarray())
    else:
        vals = spla.eigsh(L, k=6, which="LM", return_eigenvectors=False, sigma=0.0)
    vals = np.sort(np.real(vals))
    for v in vals:
        if v > eps:
            return float(v)
    return 0.0


def spectral_radius_weighted_adjacency(G: nx.Graph) -> float:
    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        return 0.0
    A = nx.adjacency_matrix(G, weight="weight").astype(float)
    v = spla.eigsh(A, k=1, which="LM", return_eigenvectors=False)[0]
    return float(v)


def approx_weighted_efficiency(G: nx.Graph, sources_k: int, seed: int) -> float:
    N = G.number_of_nodes()
    if N < 2:
        return 0.0
    H = add_dist_attr(G)
    nodes = list(H.nodes())
    rng = random.Random(seed)
    k = min(sources_k, N)
    sources = rng.sample(nodes, k)
    denom = N * (N - 1)
    total = 0.0
    for s in sources:
        dist = nx.single_source_dijkstra_path_length(H, s, weight="dist")
        acc = sum(1.0 / d for t, d in dist.items() if t != s and d > 0 and np.isfinite(d))
        total += acc
    est_full_sum = total * (N / max(1, k))
    return est_full_sum / denom


def add_dist_attr(G: nx.Graph) -> nx.Graph:
    H = G.copy()
    for _, _, d in H.edges(data=True):
        w = d.get("weight", 1.0)
        d["dist"] = 1.0 / w if w > 0 else 1e12
    return H


def compute_modularity_louvain(G: nx.Graph, seed: int) -> float:
    if G.number_of_nodes() < 3 or G.number_of_edges() == 0:
        return 0.0
    comm = louvain_communities(G, weight="weight", seed=seed)
    return nx_modularity(G, comm, weight="weight")


def degree_entropy(G: nx.Graph) -> float:
    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    degs = np.array([d for _, d in G.degree()], dtype=float)
    if degs.sum() <= 0:
        return 0.0
    p = degs / degs.sum()
    p = p[p > 0]
    return -(p * np.log(p)).sum()


def approx_diameter_lcc(G: nx.Graph, seed: int, samples: int = 16) -> int:
    if G.number_of_nodes() < 2:
        return 0
    nodes = list(G.nodes())
    rng = random.Random(seed)
    k = min(samples, len(nodes))
    picks = rng.sample(nodes, k)
    best = 0
    for s in picks:
        dist = nx.single_source_shortest_path_length(G, s)
        if dist:
            best = max(best, max(dist.values()))
    return best


def strength_ranking(G: nx.Graph) -> list:
    strength = dict(G.degree(weight="weight"))
    nodes = sorted(G.nodes(), key=lambda n: strength.get(n, 0.0), reverse=True)
    return nodes


def richclub_top_fraction(G: nx.Graph, rc_frac: float) -> list:
    nodes_sorted = strength_ranking(G)
    k = max(1, int(len(nodes_sorted) * rc_frac))
    return nodes_sorted[:k]


def richclub_by_density_threshold(G: nx.Graph, min_density: float, max_frac: float) -> list:
    nodes_sorted = strength_ranking(G)
    n = len(nodes_sorted)
    if n < 3:
        return nodes_sorted
    maxK = max(3, int(n * max_frac))
    best = nodes_sorted[:3]
    best_dens = nx.density(G.subgraph(best))
    for K in range(3, maxK + 1):
        club = nodes_sorted[:K]
        H = G.subgraph(club)
        dens = nx.density(H)
        if dens >= min_density and dens >= best_dens:
            best = club
            best_dens = dens
    return best


def pick_targets_for_attack(G: nx.Graph, attack_kind: str, step_size: int, seed: int, rc_frac: float, rc_min_density: float, rc_max_frac: float, adaptive: bool = True) -> list:
    nodes = list(G.nodes())
    rng = random.Random(seed)
    if attack_kind == "random":
        k = min(len(nodes), step_size)
        return rng.sample(nodes, k)
    elif attack_kind == "degree":
        strength = dict(G.degree(weight="weight"))
        return sorted(nodes, key=lambda n: strength.get(n, 0.0), reverse=True)[:step_size]
    elif attack_kind == "betweenness":
        bc = betweenness_centrality(add_dist_attr(G), k=min(200, G.number_of_nodes()), weight="dist", seed=seed)
        return sorted(nodes, key=lambda n: bc.get(n, 0.0), reverse=True)[:step_size]
    elif attack_kind == "kcore":
        core = core_number(G)
        return sorted(nodes, key=lambda n: core.get(n, 0), reverse=True)[:step_size]
    elif attack_kind == "richclub_top":
        club = richclub_top_fraction(G, rc_frac)
        return club[:min(step_size, len(club))]
    elif attack_kind == "richclub_density":
        club = richclub_by_density_threshold(G, rc_min_density, rc_max_frac)
        return club[:min(step_size, len(club))]
    return []


def lcc_fraction(G: nx.Graph, N0: int) -> float:
    if G.number_of_nodes() == 0 or N0 <= 0:
        return 0.0
    lcc = len(max(nx.connected_components(G), key=len))
    return lcc / N0


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
    adaptive: bool = True,
) -> tuple[pd.DataFrame, list]:
    G_curr = G_in.copy()
    N0 = G_curr.number_of_nodes()
    if N0 < 2:
        return pd.DataFrame(), []
    total_remove = int(N0 * remove_frac)
    step_size = max(1, total_remove // steps)
    history = []
    states = []
    removed_total = 0
    for step in range(steps):
        if G_curr.number_of_nodes() < 2:
            break
        if keep_states:
            states.append(G_curr.copy())
        heavy = (step % max(1, compute_heavy_every) == 0)
        light_met = calculate_light_metrics(G_curr)
        heavy_met = calculate_heavy_metrics(G_curr, eff_sources_k, seed) if heavy else {k: np.nan for k in ["eff_w", "l2_lcc", "tau_lcc", "lmax", "thresh", "mod", "entropy_deg", "diameter_approx", "kcore_max"]}
        met = {**light_met, **heavy_met}
        met["step"] = step
        met["removed_total"] = removed_total
        met["removed_frac"] = removed_total / N0
        met["lcc_frac"] = lcc_fraction(G_curr, N0)
        history.append(met)
        targets = pick_targets_for_attack(G_curr, attack_kind, step_size, seed + step, rc_frac, rc_min_density, rc_max_frac)
        G_curr.remove_nodes_from(targets)
        removed_total += len(targets)
        if removed_total >= total_remove:
            break
    df_hist = pd.DataFrame(history)
    df_hist["auc_lcc"] = trapz(df_hist["lcc_frac"], df_hist["removed_frac"])
    return df_hist, states


def classify_phase_transition(df: pd.DataFrame, x_col: str = "removed_frac", y_col: str = "lcc_frac") -> dict:
    if df.empty:
        return {"is_abrupt": False, "critical_x": np.nan, "jump": 0.0, "jump_fraction": 0.0}
    x = df[x_col].values
    y = df[y_col].values
    dy = np.diff(y)
    idx = np.argmin(dy)
    jump = -dy[idx]
    y_span = np.nanmax(y) - np.nanmin(y)
    jump_fraction = jump / max(1e-12, y_span)
    critical_x = x[idx + 1] if idx + 1 < len(x) else x[-1]
    is_abrupt = jump_fraction >= 0.35
    return {"is_abrupt": is_abrupt, "critical_x": critical_x, "jump": jump, "jump_fraction": jump_fraction}


def make_er_gnm(n: int, m: int, seed: int) -> nx.Graph:
    return nx.gnm_random_graph(n, m, seed=seed)


def make_configuration_model(G_base: nx.Graph, seed: int) -> nx.Graph:
    degs = [d for _, d in G_base.degree()]
    M = nx.configuration_model(degs, seed=seed)
    H = nx.Graph(M)
    H.remove_edges_from(nx.selfloop_edges(H))
    return H


def rewire_mix(G_base: nx.Graph, p: float, seed: int) -> nx.Graph:
    p = max(0.0, min(1.0, p))
    H = G_base.copy()
    if H.number_of_edges() < 2 or p <= 0:
        return H
    swaps = int(p * H.number_of_edges() * 5)
    tries = swaps * 10
    nx.double_edge_swap(H, nswap=swaps, max_tries=tries, seed=seed)
    H.remove_edges_from(nx.selfloop_edges(H))
    return H


def make_sbm_fit(G_base: nx.Graph, seed: int) -> nx.Graph:
    comm = louvain_communities(G_base, seed=seed)
    sizes = [len(c) for c in comm]
    p = [[0.0 for _ in comm] for _ in comm]
    for i, ci in enumerate(comm):
        for j, cj in enumerate(comm):
            if i == j:
                sub = G_base.subgraph(ci)
                p[i][j] = nx.density(sub) if len(ci) > 1 else 0.0
            else:
                e_between = sum(1 for u in ci for v in cj if G_base.has_edge(u, v))
                p[i][j] = e_between / (len(ci) * len(cj)) if len(ci) * len(cj) > 0 else 0.0
    return stochastic_block_model(sizes, p, seed=seed)


def push_experiment(name: str, graph_id: str, attack_kind: str, params: dict, df_hist: pd.DataFrame):
    exp_id = new_id("EXP")
    phase = classify_phase_transition(df_hist)
    st.session_state["experiments"].append(
        {
            "id": exp_id,
            "name": name,
            "graph_id": graph_id,
            "attack_kind": attack_kind,
            "params": {**params, "phase": phase},
            "history": df_hist.copy(),
            "created_at": time.time(),
        }
    )
    return exp_id


def compute_3d_layout(G: nx.Graph, seed: int) -> dict:
    if G.number_of_nodes() == 0:
        return {}
    return nx.spring_layout(G, dim=3, weight="weight", seed=seed)


def make_3d_traces(G: nx.Graph, pos3d: dict, show_scale: bool = False):
    if G.number_of_nodes() == 0:
        return None, None
    strength = dict(G.degree(weight="weight"))
    nodes = [n for n in G.nodes() if n in pos3d]
    xs, ys, zs = [pos3d[n][0] for n in nodes], [pos3d[n][1] for n in nodes], [pos3d[n][2] for n in nodes]
    colors = [strength.get(n, 0.0) for n in nodes]
    texts = [f"{n}: strength={strength.get(n, 0.0):.3f}" for n in nodes]
    node_trace = go.Scatter3d(x=xs, y=ys, z=zs, mode="markers", marker=dict(size=4, color=colors, colorscale="Inferno", showscale=show_scale), text=texts, hoverinfo="text", name="nodes")
    ex, ey, ez = [], [], []
    for u, v in G.edges():
        if u in pos3d and v in pos3d:
            ex.extend([pos3d[u][0], pos3d[v][0], None])
            ey.extend([pos3d[u][1], pos3d[v][1], None])
            ez.extend([pos3d[u][2], pos3d[v][2], None])
    edge_trace = go.Scatter3d(x=ex, y=ey, z=ez, mode="lines", line=dict(color="#444", width=1), hoverinfo="none", name="edges")
    return edge_trace, node_trace


def fig_metrics_over_steps(df_hist: pd.DataFrame, title: str = "") -> go.Figure:
    fig = go.Figure()
    if df_hist.empty:
        return fig
    x = df_hist["removed_frac"]
    for col in ["lcc_frac", "mod", "l2_lcc", "eff_w", "C", "kcore_max"]:
        if col in df_hist.columns:
            fig.add_trace(go.Scatter(x=x, y=df_hist[col], name=col))
    fig.update_layout(template="plotly_dark", title=title, xaxis_title="removed_frac", yaxis_title="value")
    return fig


def fig_compare_attacks(curves: list[tuple[str, pd.DataFrame]], x_col: str, y_col: str, title: str) -> go.Figure:
    fig = go.Figure()
    for name, df in curves:
        if not df.empty:
            fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], name=name))
    fig.update_layout(template="plotly_dark", title=title, xaxis_title=x_col, yaxis_title=y_col)
    return fig


def fig_phase_space(df_hist: pd.DataFrame, x_col: str, y_col: str, color_col: str = "removed_frac", title: str = "") -> go.Figure:
    fig = go.Figure()
    if not df_hist.empty:
        fig.add_trace(go.Scatter(x=df_hist[x_col], y=df_hist[y_col], mode="lines+markers", marker=dict(color=df_hist[color_col], colorscale="Viridis")))
    fig.update_layout(template="plotly_dark", title=title)
    return fig


def fig_degree_distribution(G: nx.Graph, log_scale: bool = False) -> go.Figure:
    degs = [d for _, d in G.degree()]
    fig = go.Figure(go.Histogram(x=degs, histnorm="probability" if log_scale else "", xbins=dict(start=min(degs), end=max(degs), size=1)))
    fig.update_layout(template="plotly_dark", title="Degree Distribution", xaxis_type="log" if log_scale else "linear")
    return fig


def _df_to_b64_csv(df: pd.DataFrame) -> str:
    csv = df.to_csv(index=False).encode("utf-8")
    return base64.b64encode(csv).decode("ascii")


def _b64_csv_to_df(s: str) -> pd.DataFrame:
    raw = base64.b64decode(s)
    return pd.read_csv(io.BytesIO(raw))


def export_workspace_json(graphs: dict, experiments: list) -> bytes:
    g_out = {gid: {**g, "edges_b64": _df_to_b64_csv(g["edges"])} for gid, g in graphs.items()}
    e_out = [{**e, "history_b64": _df_to_b64_csv(e["history"])} for e in experiments]
    payload = {"graphs": g_out, "experiments": e_out}
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def import_workspace_json(blob: bytes) -> tuple[dict, list]:
    payload = json.loads(blob)
    graphs = {gid: {**g, "edges": _b64_csv_to_df(g["edges_b64"])} for gid, g in payload.get("graphs", {}).items()}
    exps = [{**e, "history": _b64_csv_to_df(e["history_b64"])} for e in payload.get("experiments", [])]
    return graphs, exps


# =========================
# Sticky Topbar (enhanced)
# =========================
def render_sticky_topbar():
    graphs = st.session_state["graphs"]
    gids = sorted_graph_ids()

    with st.container():
        st.markdown('<div class="sticky-topbar">', unsafe_allow_html=True)

        if not gids:
            r1, r2, r3, r4, r5, r6 = st.columns([2.0, 1.0, 1.0, 1.3, 1.3, 1.4])
            with r1:
                st.warning("Workspace empty.")
            with r2:
                seed = st.number_input("Seed", value=st.session_state["seed_top"], step=1)
                st.session_state["seed_top"] = seed
            with r3:
                n_val = st.number_input("N", min_value=2, value=200)
            with r4:
                m_val = st.number_input("M", min_value=1, value=600)
            with r5:
                name = st.text_input("Name", "Random Graph")
            with r6:
                if st.button("Generate ER"):
                    G_new = make_er_gnm(n_val, m_val, seed)
                    edges = [[u, v, 1.0, 1.0] for u, v in G_new.edges()]
                    df_new = pd.DataFrame(edges, columns=["src", "dst", "weight", "confidence"])
                    add_graph_to_workspace(name or f"ER(n={n_val},m={m_val})", df_new, "null:ER", {"src_col": "src", "dst_col": "dst"})
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
            return None

        active_gid = st.session_state["active_graph_id"]
        if active_gid not in graphs:
            st.session_state["active_graph_id"] = gids[0]
            active_gid = gids[0]

        # Row 1: Graph selection and management
        c1, c2, c3, c4 = st.columns([2.6, 2.2, 1.1, 1.1])
        with c1:
            picked = st.selectbox(
                "Active Graph",
                options=gids,
                index=gids.index(active_gid),
                format_func=lambda gid: graphs[gid]["name"],
            )
            if picked != active_gid:
                st.session_state["active_graph_id"] = picked
                st.rerun()
        with c2:
            new_name = st.text_input("Rename", value=graphs[active_gid]["name"])
        with c3:
            if st.button("Rename"):
                graphs[active_gid]["name"] = new_name.strip()
                st.rerun()
        with c4:
            if st.button("Delete"):
                del graphs[active_gid]
                st.session_state["experiments"] = [e for e in st.session_state["experiments"] if e["graph_id"] != active_gid]
                gids = sorted_graph_ids()
                st.session_state["active_graph_id"] = gids[0] if gids else None
                st.rerun()

        # Row 2: Generator controls
        g1, g2, g3, g4, g5, g6 = st.columns([1.0, 1.6, 1.0, 1.0, 1.6, 1.2])
        with g1:
            seed = st.number_input("Seed", value=st.session_state["seed_top"])
            st.session_state["seed_top"] = seed
        with g2:
            gen_mode = st.selectbox("Mode", ["Based on Active (N,E)", "Manual N/M"])
        active_entry = graphs[active_gid]
        df_base = active_entry["edges"]
        src_col = active_entry["tags"].get("src_col", df_base.columns[0])
        dst_col = active_entry["tags"].get("dst_col", df_base.columns[1])
        G_base = build_graph_from_edges(df_base, src_col, dst_col)
        N0, E0 = G_base.number_of_nodes(), G_base.number_of_edges()
        if gen_mode == "Based on Active (N,E)":
            n_val, m_val = N0, E0
            with g3:
                st.number_input("N", value=n_val, disabled=True)
            with g4:
                st.number_input("M", value=m_val, disabled=True)
        else:
            with g3:
                n_val = st.number_input("N", min_value=2, value=200)
            with g4:
                m_val = st.number_input("M", min_value=1, value=600)
        with g5:
            gen_type = st.selectbox("Type", ["ER G(n,m)", "CFG (from active)", "Mix/Rewire p (from active)", "SBM-fit (from active)"])
        with g6:
            new_name = st.text_input("New Name", "")

        # Row 3: Generate + params
        a1, a2, a3, a4 = st.columns([1.4, 1.0, 1.0, 1.6])
        with a1:
            st.caption(f"Active: N={N0} E={E0}")
        with a2:
            if st.button("Copy Active"):
                add_graph_to_workspace(f"Copy of {active_entry['name']}", df_base, "copy", active_entry["tags"])
                st.rerun()
        with a3:
            if st.button("Clear Exps"):
                st.session_state["experiments"] = [e for e in st.session_state["experiments"] if e["graph_id"] != active_gid]
                st.rerun()
        with a4:
            p = None
            if gen_type == "Mix/Rewire p (from active)":
                p = st.slider("p", 0.0, 1.0, 0.2, 0.05)
            if st.button("Generate"):
                nm = new_name.strip() or gen_type
                if gen_type == "ER G(n,m)":
                    G_new = make_er_gnm(n_val, m_val, seed)
                elif gen_type == "CFG (from active)":
                    G_new = make_configuration_model(G_base, seed)
                elif gen_type == "Mix/Rewire p (from active)":
                    G_new = rewire_mix(G_base, p or 0.2, seed)
                elif gen_type == "SBM-fit (from active)":
                    G_new = make_sbm_fit(G_base, seed)
                edges = [[u, v, 1.0, 1.0] for u, v in G_new.edges()]
                df_new = pd.DataFrame(edges, columns=[src_col, dst_col, "weight", "confidence"])
                add_graph_to_workspace(nm, df_new, f"null:{gen_type}", {"src_col": src_col, "dst_col": dst_col})
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    return st.session_state["active_graph_id"]


# =========================
# Sidebar (enhanced)
# =========================
with st.sidebar:
    st.header("🧠 Workspace")

    with st.expander("💾 Import / Export"):
        c1, c2 = st.columns(2)
        if c1.button("Export Workspace"):
            blob = export_workspace_json(st.session_state["graphs"], st.session_state["experiments"])
            st.download_button("⬇️ workspace.json", blob, "workspace.json", "application/json")
        up_ws = st.file_uploader("Import workspace.json", type=["json"])
        if up_ws:
            graphs_new, exps_new = import_workspace_json(up_ws.getvalue())
            st.session_state["graphs"] = graphs_new
            st.session_state["experiments"] = exps_new
            st.session_state["active_graph_id"] = sorted_graph_ids()[0] if graphs_new else None
            st.rerun()

    st.write("---")
    st.header("📎 Upload Graph")
    uploaded = st.file_uploader("CSV/Excel (fixed format)", type=["csv", "xlsx", "xls"])
    if uploaded:
        fp = fingerprint_upload(uploaded)
        if fp != st.session_state["last_uploaded_fingerprint"]:
            df_any = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
            df_edges = df_any.iloc[:, [0,1,8,9]].copy()
            df_edges.columns = ["src", "dst", "confidence", "weight"]
            add_graph_to_workspace(uploaded.name, df_edges, "upload", {"src_col": "src", "dst_col": "dst"})
            st.session_state["last_uploaded_fingerprint"] = fp
            st.rerun()

    st.write("---")
    st.header("🎛️ Global Params")
    min_conf = st.number_input("Min Confidence", value=0)
    min_weight = st.number_input("Min Weight", value=0.0, step=0.1)
    eff_k = st.slider("Efficiency k", 8, 256, 64)
    seed_analysis = st.number_input("Analysis Seed", value=42)
    directed = st.checkbox("Directed Graph", value=False)
    weighted = st.checkbox("Weighted", value=True)
    analysis_mode = st.radio("Analysis Scope", ["Full Graph", "LCC", "Top-K Nodes"])
    if analysis_mode == "Top-K Nodes":
        top_k = st.number_input("Top K by Degree", min_value=10, value=100)

    st.write("---")
    st.header("🧹 Cleanup")
    if st.button("Clear All"):
        st.session_state["graphs"] = {}
        st.session_state["experiments"] = []
        st.session_state["active_graph_id"] = None
        st.session_state["last_uploaded_fingerprint"] = None
        st.rerun()

# =========================
# Main
# =========================
render_sticky_topbar()

entry = get_active_graph_entry()
if entry is None:
    st.stop()

graph_id = entry["id"]
df_edges = entry["edges"]
src_col = entry["tags"].get("src_col", df_edges.columns[0])
dst_col = entry["tags"].get("dst_col", df_edges.columns[1])
df_f = filter_edges(df_edges, src_col, dst_col, min_conf, min_weight)
G = build_graph_from_edges(df_f, src_col, dst_col, weighted, directed)
if analysis_mode == "LCC":
    G_view = lcc_subgraph(G)
elif analysis_mode == "Top-K Nodes":
    nodes_top = sorted(G.nodes(), key=lambda n: G.degree(n, weight="weight" if weighted else None), reverse=True)[:top_k]
    G_view = G.subgraph(nodes_top)
else:
    G_view = G

light_base = calculate_light_metrics(G_view)
if st.sidebar.button("Compute Heavy Metrics"):
    heavy_base = calculate_heavy_metrics(G_view, eff_k, seed_analysis)
else:
    heavy_base = {k: "Press to compute" for k in ["eff_w", "l2_lcc", "tau_lcc", "lmax", "thresh", "mod", "entropy_deg", "diameter_approx", "kcore_max"]}

# Tabs (structured as per plan)
t1, t2, t3, t4, t5, t6, t7 = st.tabs(["A: Data & Build", "B: Anatomy", "C: Spectrum", "D: Communities", "E: Viz", "F: Attack Lab", "G: Compare"])

with t1:
    st.subheader("Data Info")
    st.write(f"Name: {entry['name']} | Source: {entry['source']} | ID: {graph_id} | Created: {time.ctime(entry['created_at'])}")
    st.write(f"Src Col: {src_col} | Dst Col: {dst_col}")
    st.write(f"Raw Edges: {len(df_edges)} | Filtered: {len(df_f)} ({len(df_edges) - len(df_f)} removed)")
    st.write(f"Graph Type: {'Directed' if directed else 'Undirected'}, {'Weighted' if weighted else 'Unweighted'}")
    st.write(graph_summary(G_view))
    with st.expander("Raw Edges Preview"):
        st.dataframe(df_f.head(50))

with t2:
    st.subheader("Basic Anatomy")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("N", light_base["N"])
        st.metric("E", light_base["E"])
        st.metric("Density", f"{light_base['density']:.6f}")
        st.metric("Avg Degree", f"{light_base['avg_degree']:.3f}")
    with col2:
        st.metric("Components", light_base["C"])
        st.metric("LCC Size", light_base["lcc_size"])
        st.metric("LCC Frac", f"{light_base['lcc_frac']:.4f}")
        st.metric("Beta Cycles", light_base["beta"])
    log_scale = st.checkbox("Log Scale for Dist")
    st.plotly_chart(fig_degree_distribution(G_view, log_scale))

with t3:
    st.subheader("Spectrum & Resilience")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("λ₂ (LCC)", heavy_base["l2_lcc"])
        st.metric("τ = 1/λ₂", heavy_base["tau_lcc"])
        st.metric("λmax(A)", heavy_base["lmax"])
        st.metric("1/λmax", heavy_base["thresh"])
    with col2:
        st.metric("Weighted Efficiency", heavy_base["eff_w"])
        st.metric("Degree Entropy H", heavy_base["entropy_deg"])
        st.metric("Assortativity r", light_base["assortativity"])
        st.metric("Clustering C̄", light_base["clustering"])
    st.metric("Approx Diameter", heavy_base["diameter_approx"])

with t4:
    st.subheader("Communities & Hierarchy")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Modularity Q", heavy_base["mod"])
        st.metric("k-core Max", heavy_base["kcore_max"])
    with col2:
        # Add more as needed, e.g., rich-club coeff
        pass
    # Graphs for communities, core dist, etc.

with t5:
    st.subheader("Visualization")
    pos3d = compute_3d_layout(G_view, seed_analysis)
    e_tr, n_tr = make_3d_traces(G_view, pos3d, show_scale=True)
    fig = go.Figure(data=[e_tr, n_tr])
    fig.update_layout(template="plotly_dark", scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False))
    st.plotly_chart(fig, use_container_width=True)

with t6:
    st.subheader("Attack Lab")
    with st.expander("Attack Params"):
        attack_kind_ui = st.selectbox("Attack Vector", ["random", "degree (Hubs)", "betweenness (Bridges)", "kcore (Core)", "rich-club A (Top-fraction)", "rich-club B (Density-threshold)"])
        remove_frac = st.slider("Remove Fraction", 0.05, 0.95, 0.50)
        steps = st.slider("Steps", 5, 200, 40)
        attack_seed = st.number_input("Attack Seed", value=seed_analysis)
        adaptive = st.checkbox("Adaptive Ranking", value=True)
        compute_heavy_every = st.slider("Heavy Metrics Every", 1, 10, 1)
        keep_states = st.checkbox("Keep States for Replay")
        name_tag = st.text_input("Exp Tag")
        rc_frac = st.slider("RC-A: Top Frac", 0.02, 0.50, 0.10) if "A" in attack_kind_ui else 0.10
        rc_min_density = st.slider("RC-B: Min Density", 0.05, 1.00, 0.30) if "B" in attack_kind_ui else 0.30
        rc_max_frac = st.slider("RC-B: Max Frac", 0.05, 0.80, 0.30) if "B" in attack_kind_ui else 0.30

    attack_kind = {
        "random": "random",
        "degree (Hubs)": "degree",
        "betweenness (Bridges)": "betweenness",
        "kcore (Core)": "kcore",
        "rich-club A (Top-fraction)": "richclub_top",
        "rich-club B (Density-threshold)": "richclub_density",
    }[attack_kind_ui]

    if st.button("Run Single Attack"):
        df_hist, states = run_attack(G_view, attack_kind, remove_frac, steps, attack_seed, eff_k, rc_frac, rc_min_density, rc_max_frac, compute_heavy_every, keep_states, adaptive)
        push_experiment(f"{attack_kind_ui} {name_tag}", graph_id, attack_kind, {"remove_frac": remove_frac, "steps": steps, "seed": attack_seed}, df_hist)
        st.plotly_chart(fig_metrics_over_steps(df_hist, "Attack Curves"))
        phase = classify_phase_transition(df_hist)
        st.write(f"Phase: Abrupt={phase['is_abrupt']} | Critical ~{phase['critical_x']:.3f} | Jump={phase['jump']:.3f}")
        st.plotly_chart(fig_phase_space(df_hist, "l2_lcc", "mod", "removed_frac", "Phase Space: Q vs λ₂"))

    if st.button("Run Head-to-Head (All Attacks)"):
        kinds = ["random", "degree", "betweenness", "kcore", "richclub_top", "richclub_density"]
        curves = []
        for k in kinds:
            df_hist, _ = run_attack(G_view, k, remove_frac, steps, attack_seed, eff_k, rc_frac, rc_min_density, rc_max_frac, compute_heavy_every, False, adaptive)
            push_experiment(f"H2H: {k}", graph_id, k, {"remove_frac": remove_frac, "steps": steps, "seed": attack_seed}, df_hist)
            curves.append((k, df_hist))
        st.plotly_chart(fig_compare_attacks(curves, "removed_frac", "lcc_frac", "Compare LCC Frac"))

with t7:
    st.subheader("Compare")
    selected_gids = st.multiselect("Graphs", options=sorted_graph_ids(), format_func=lambda gid: graphs[gid]["name"])
    scalar = st.selectbox("Scalar Metric", ["N", "E", "density", "l2_lcc", "mod", "eff_w"])
    if selected_gids:
        rows = []
        for gid in selected_gids:
            g = graphs[gid]
            df_f_g = filter_edges(g["edges"], g["tags"]["src_col"], g["tags"]["dst_col"], min_conf, min_weight)
            G_g = build_graph_from_edges(df_f_g, g["tags"]["src_col"], g["tags"]["dst_col"], weighted, directed)
            met = {**calculate_light_metrics(G_g), **calculate_heavy_metrics(G_g, eff_k, seed_analysis)}
            rows.append({"graph": g["name"], scalar: met.get(scalar)})
        df_cmp = pd.DataFrame(rows)
        fig = go.Figure(go.Bar(x=df_cmp["graph"], y=df_cmp[scalar]))
        fig.update_layout(template="plotly_dark", title=f"Compare {scalar}")
        st.plotly_chart(fig)
        st.dataframe(df_cmp)

    st.write("---")
    selected_exps = st.multiselect("Experiments", options=[e["id"] for e in st.session_state["experiments"]], format_func=lambda eid: next(e["name"] for e in st.session_state["experiments"] if e["id"] == eid))
    y_metric = st.selectbox("Y Metric", ["lcc_frac", "mod", "l2_lcc", "eff_w"])
    if selected_exps:
        curves = []
        for eid in selected_exps:
            e = next(e for e in st.session_state["experiments"] if e["id"] == eid)
            curves.append((e["name"], e["history"]))
        st.plotly_chart(fig_compare_attacks(curves, "removed_frac", y_metric, f"Compare {y_metric}"))

# Add tooltips everywhere, e.g., for metrics
st.markdown('<div class="tooltip">λ₂<span class="tooltiptext">Algebraic connectivity: low values indicate easy disconnection.</span></div>', unsafe_allow_html=True)
