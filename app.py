import streamlit as st

# IMPORTANT: st.set_page_config() must be the first Streamlit command in the script.
st.set_page_config(
    page_title="Kodik Lab",
    layout="wide",
    page_icon="🕸️",
    initial_sidebar_state="expanded",
)
# Quick UI heartbeat: keep at the top so it renders immediately if downstream code stalls.
st.write("BOOT OK")
st.title("Graph Lab")
st.write("UI loaded")

import time
import uuid
import hashlib
import textwrap

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px

from compute import compute_layout as compute_layout_cached
from compute import compute_curvature as compute_curvature_cached
from src.io_load import load_uploaded_any
from src.preprocess import coerce_fixed_format, filter_edges
from src.graph_build import build_graph_from_edges, lcc_subgraph
from src.metrics import (
    calculate_metrics,
    compute_3d_layout,
    compute_energy_flow,
    simulate_energy_flow,
    make_3d_traces,
    make_energy_flow_figure_3d,
)
from src.robust_geom import ollivier_ricci_edge
from src.null_models import make_er_gnm, make_configuration_model, rewire_mix
from src.attacks import run_attack 
from src.attacks_mix import run_mix_attack
from src.plotting import fig_metrics_over_steps, fig_compare_attacks
from src.phase import classify_phase_transition
from src.ui_blocks import help_icon, render_dashboard_metrics, render_dashboard_charts
from src.session_io import (
    export_workspace_json,
    import_workspace_json,
    export_experiments_json,
    import_experiments_json,
)
from src.utils import as_simple_undirected

# -----------------------------
# Streamlit caching helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def _filter_edges_cached(
    graph_id: str,
    df_hash: str,
    src_col: str,
    dst_col: str,
    min_conf: float,
    min_weight: float,
) -> pd.DataFrame:
    """Cache-friendly wrapper around filter_edges keyed by graph ID + data hash."""
    entry = st.session_state["graphs"][graph_id]
    return filter_edges(entry["edges"], src_col, dst_col, min_conf, min_weight)


@st.cache_resource(show_spinner=False)
def _build_graph_cached(
    graph_id: str,
    df_hash: str,
    src_col: str,
    dst_col: str,
    min_conf: float,
    min_weight: float,
    analysis_mode: str,
) -> nx.Graph:
    """Build NetworkX graph once per filter + analysis mode settings."""
    df_filtered = _filter_edges_cached(graph_id, df_hash, src_col, dst_col, min_conf, min_weight)
    G = build_graph_from_edges(df_filtered, src_col, dst_col)
    if analysis_mode.startswith("LCC"):
        G = lcc_subgraph(G)
    return G


@st.cache_data(show_spinner=False)
def _metrics_cached(
    graph_id: str,
    df_hash: str,
    src_col: str,
    dst_col: str,
    min_conf: float,
    min_weight: float,
    analysis_mode: str,
    seed: int,
    compute_curvature: bool,
    curvature_sample_edges: int,
) -> dict:
    """Cache heavy metrics separately from graph construction."""
    G = _build_graph_cached(graph_id, df_hash, src_col, dst_col, min_conf, min_weight, analysis_mode)
    return calculate_metrics(
        G,
        eff_sources_k=32,
        seed=int(seed),
        compute_curvature=bool(compute_curvature),
        curvature_sample_edges=int(curvature_sample_edges),
    )


@st.cache_data(show_spinner=False)
def _layout_cached(
    graph_id: str,
    df_hash: str,
    src_col: str,
    dst_col: str,
    min_conf: float,
    min_weight: float,
    analysis_mode: str,
    seed: int,
) -> dict:
    """Cache 3D layouts so layout recomputation does not block UI."""
    G = _build_graph_cached(graph_id, df_hash, src_col, dst_col, min_conf, min_weight, analysis_mode)
    return compute_3d_layout(G, seed=int(seed))


@st.cache_data(show_spinner=False)
def _energy_frames_cached(
    graph_id: str,
    df_hash: str,
    src_col: str,
    dst_col: str,
    min_conf: float,
    min_weight: float,
    analysis_mode: str,
    *,
    steps: int,
    flow_mode: str,
    damping: float,
    sources: tuple,
    phys_injection: float,
    phys_leak: float,
    phys_cap_mode: str,
    rw_impulse: bool,
) -> tuple[list[dict], list[dict]]:
    """Cache heavy energy frames separately to avoid re-simulating on UI tweaks."""
    G = _build_graph_cached(graph_id, df_hash, src_col, dst_col, min_conf, min_weight, analysis_mode)
    src_list = list(sources) if sources else None
    node_frames, edge_frames = simulate_energy_flow(
        G,
        steps=int(steps),
        flow_mode=str(flow_mode),
        damping=float(damping),
        sources=src_list,
        phys_injection=float(phys_injection),
        phys_leak=float(phys_leak),
        phys_cap_mode=str(phys_cap_mode),
        rw_impulse=bool(rw_impulse),
    )
    return node_frames, edge_frames


def _quick_counts(df: pd.DataFrame, src_col: str, dst_col: str) -> tuple[int, int]:
    """Fast node/edge counts without constructing a NetworkX graph."""
    if df is None or df.empty:
        return 0, 0
    nodes = pd.unique(pd.concat([df[src_col], df[dst_col]], ignore_index=True))
    return int(len(nodes)), int(len(df))

st.markdown(
    """
    <style>
    /* --- STICKY HEADER FIXES --- */
    div[data-testid="stVerticalBlock"] > div:has(> div.sticky-header) {
        position: sticky;
        top: 2.8rem; /* Offset for Streamlit's own top bar */
        z-index: 9999;
        background-color: #0e1117;
        border-bottom: 1px solid rgba(250, 250, 250, 0.1);
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    /* Ensure tabs don't overlap with sticky header */
    .stTabs {
        margin-top: 1rem;
        z-index: 1;
    }

    /* --- BUTTON STYLING --- */
    div.stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease-in-out;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border-color: rgba(255, 255, 255, 0.3);
    }
    div.stButton > button:active {
        transform: translateY(0px);
    }
    /* Primary button specific styling */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #ff4b4b 0%, #ff2b2b 100%);
        border: none;
    }

    /* --- METRICS & TEXT --- */
    div[data-testid="stMetricValue"] { font-size: 1.35rem !important; }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.05rem;
        font-weight: 650;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
METRIC_HELP = {
    "lcc_frac": "Доля узлов в гигантской компоненте связности. Параметр порядка для перколяции.",
    "eff_w": "Глобальная эффективность (среднее 1/кратчайшему пути; аппрокс по k источникам).",
    "l2_lcc": "Алгебраическая связность λ₂ Лапласиана на LCC. 0≈распад связности, больше=лучше.",
    "mod": "Модульность Louvain: выше=сильнее выражены сообщества.",
    "H_deg": "Энтропия распределения степеней (шаг 1).",
    "H_w": "Энтропия распределения весов рёбер (шаг 1).",
    "H_conf": "Энтропия распределения confidence (шаг 1).",
    "H_tri": "Энтропия распределения ‘треугольной поддержки’ (шаг 3, тяжёлая).",
}

ATTACK_PRESETS_NODE = {
    "Node core suite (быстро)": [
        {"kind": "random", "seeds": 3},
        {"kind": "degree", "seeds": 3},
        {"kind": "betweenness", "seeds": 2},
        {"kind": "kcore", "seeds": 2},
        {"kind": "richclub_top", "seeds": 2},
    ],
    "Node weak suite (слабые узлы)": [
        {"kind": "low_degree", "seeds": 5},      
        {"kind": "weak_strength", "seeds": 5},   
    ],
    "Node stress suite (жёстко)": [
        {"kind": "degree", "seeds": 5},
        {"kind": "betweenness", "seeds": 5},
        {"kind": "kcore", "seeds": 5},
        {"kind": "richclub_top", "seeds": 5},
    ],
}

ATTACK_PRESETS_EDGE = {
    "Edge weak suite (слабые связи)": [
        {"kind": "weak_edges_by_weight", "seeds": 1},
        {"kind": "weak_edges_by_confidence", "seeds": 1},
    ],
    "Edge strong-first (контрпример)": [
        {"kind": "strong_edges_by_weight", "seeds": 1},
        {"kind": "strong_edges_by_confidence", "seeds": 1},
    ],
}

AUC_TRAP = getattr(np, "trapezoid", None) or getattr(np, "trapz")

def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:6]}"

def _auto_y_range(series: pd.Series, pad_frac: float = 0.08):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    y0, y1 = float(s.min()), float(s.max())
    if not (np.isfinite(y0) and np.isfinite(y1)):
        return None
    if y0 == y1:
        eps = 1e-6 if y0 == 0 else abs(y0) * 0.05
        return [y0 - eps, y1 + eps]
    pad = (y1 - y0) * pad_frac
    return [y0 - pad, y1 + pad]

def _apply_plot_defaults(fig, height=780, y_range=None):
    fig.update_layout(height=height)
    fig.update_xaxes(showgrid=True, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=True)
    if y_range is not None:
        fig.update_yaxes(range=y_range)
    return fig

def _forward_fill_heavy(df_hist: pd.DataFrame) -> pd.DataFrame:
    df = df_hist.copy()
    for col in ["l2_lcc", "mod", "H_tri"]:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).ffill()
    return df

def _strength(G: nx.Graph, n):
    s = 0.0
    for _, _, d in G.edges(n, data=True):
        w = d.get("weight", 1.0)
        try:
            s += float(w)
        except Exception:
            s += 1.0
    return s

def _extract_removed_order(aux):
    if isinstance(aux, dict):
        for k in ["removed_nodes", "removed_order", "order", "removal_order", "removed"]:
            v = aux.get(k)
            if isinstance(v, (list, tuple)) and v:
                return list(v)
    if isinstance(aux, (list, tuple)) and aux:
        if not isinstance(aux[0], (pd.DataFrame, np.ndarray, dict, list, tuple)):
            return list(aux)
    return None

def _fallback_removal_order(G: nx.Graph, kind: str, seed: int):
    """
    Fallback для 3D-декомпозиции, если src.attacks не вернул порядок удаления.
    ВАЖНО: это не адаптивная атака, только визуальный fallback.
    """
    if G.number_of_nodes() == 0:
        return []

    rng = np.random.default_rng(int(seed))
    H = as_simple_undirected(G)
    nodes = list(H.nodes())

    if kind in ("random",):
        rng.shuffle(nodes)
        return nodes

    if kind in ("degree",):
        nodes.sort(key=lambda n: H.degree(n), reverse=True)
        return nodes

    if kind in ("low_degree",):  
        nodes.sort(key=lambda n: H.degree(n))
        return nodes

    if kind in ("weak_strength",): 
        nodes.sort(key=lambda n: _strength(H, n))
        return nodes

    if kind in ("betweenness",):
        if H.number_of_nodes() > 5000:
            nodes.sort(key=lambda n: H.degree(n), reverse=True)
            return nodes
        b = nx.betweenness_centrality(H, normalized=True)
        nodes.sort(key=lambda n: b.get(n, 0.0), reverse=True)
        return nodes

    if kind in ("kcore",):
        try:
            core = nx.core_number(H)
            nodes.sort(key=lambda n: core.get(n, 0), reverse=True)
            return nodes
        except Exception:
            nodes.sort(key=lambda n: H.degree(n), reverse=True)
            return nodes

    if kind in ("richclub_top",):
        nodes.sort(key=lambda n: _strength(H, n), reverse=True)
        return nodes

    rng.shuffle(nodes)
    return nodes

def _compute_metrics_snapshot(
    G: nx.Graph,
    eff_k: int,
    seed: int,
    heavy: bool,
    compute_curvature: bool,
    curvature_sample_edges: int,
):
    """
    Safe wrapper around calculate_metrics.
    If heavy=False: we still call calculate_metrics, but pass smaller eff_k upstream (already controlled by caller).
    Heavy gating is handled by caller by skipping/ffill some columns.
    """
    m = calculate_metrics(
        G,
        eff_sources_k=int(eff_k),
        seed=int(seed),
        compute_curvature=bool(compute_curvature and heavy),
        curvature_sample_edges=int(curvature_sample_edges),
    )
    return m

def run_edge_attack(
    G: nx.Graph,
    kind: str,
    frac: float,
    steps: int,
    seed: int,
    eff_k: int,
    compute_heavy_every: int = 2,
):
    """
    Edge-removal attack:
    - kind: weak/strong by weight/confidence OR Ricci/flux-based rankings
    - returns df_hist, aux
    aux contains removed_edges_order (list of (u,v)) used for 3D decomposition.
    """
    if G.number_of_edges() == 0:
        df = pd.DataFrame([{"step": 0, "removed_frac": 0.0, "N": G.number_of_nodes(), "E": 0, "lcc_frac": 0.0}])
        return df, {"removed_edges_order": []}

    H0 = as_simple_undirected(G)
    edges = list(H0.edges(data=True))
    kind = str(kind)

    def _sf(x, default: float = 0.0) -> float:
        """Safe float conversion with finite fallback."""
        try:
            v = float(x)
            if not np.isfinite(v):
                return float(default)
            return v
        except Exception:
            return float(default)

    # --------------------------
    # Cheap rankings by attributes
    # --------------------------
    if kind in (
        "weak_edges_by_weight",
        "weak_edges_by_confidence",
        "strong_edges_by_weight",
        "strong_edges_by_confidence",
    ):
        if "confidence" in kind:
            key = lambda e: _sf(e[2].get("confidence", 1.0), 1.0)
        else:
            key = lambda e: _sf(e[2].get("weight", 1.0), 1.0)

        reverse = kind.startswith("strong_")
        edges.sort(key=key, reverse=reverse)

    else:
        # --------------------------
        # Expensive rankings: Ricci / Flux
        # --------------------------
        rng = np.random.default_rng(int(seed))
        max_eval = 600  # Cap edge curvature evaluations for speed.
        edge_list = [(u, v) for (u, v, _d) in edges]
        if len(edge_list) > max_eval:
            sample_idx = rng.choice(len(edge_list), size=max_eval, replace=False)
            sampled = [edge_list[i] for i in sample_idx]
        else:
            sampled = edge_list

        kappa = {}
        flux = {}

        # Flux precompute (RW / Evo).
        if kind in ("flux_high_rw", "flux_high_evo", "flux_high_rw_x_neg_ricci"):
            fm = "evo" if kind.endswith("_evo") else "rw"
            try:
                _ne, ef = compute_energy_flow(H0, steps=20, flow_mode=fm, damping=1.0)
                flux = dict(ef)
            except Exception:
                flux = {}

        # Curvature on sampled edges.
        if kind.startswith("ricci_") or kind == "flux_high_rw_x_neg_ricci":
            for (u, v) in sampled:
                try:
                    val = ollivier_ricci_edge(H0, u, v, max_support=60, cutoff=8.0)
                except Exception:
                    val = None
                if val is None or not np.isfinite(val):
                    continue
                kappa[(u, v)] = float(val)

        def _flux_uv(u, v) -> float:
            if (u, v) in flux:
                return _sf(flux[(u, v)], 0.0)
            if (v, u) in flux:
                return _sf(flux[(v, u)], 0.0)
            return 0.0

        def _kappa_uv(u, v) -> float:
            if (u, v) in kappa:
                return _sf(kappa[(u, v)], 0.0)
            if (v, u) in kappa:
                return _sf(kappa[(v, u)], 0.0)
            return 0.0

        def score(u, v, d) -> float:
            if kind == "flux_high_rw":
                return _flux_uv(u, v)
            if kind == "flux_high_evo":
                return _flux_uv(u, v)
            if kind == "ricci_most_negative":
                return -_kappa_uv(u, v)
            if kind == "ricci_most_positive":
                return _kappa_uv(u, v)
            if kind == "ricci_abs_max":
                return abs(_kappa_uv(u, v))
            if kind == "flux_high_rw_x_neg_ricci":
                return _flux_uv(u, v) * max(0.0, -_kappa_uv(u, v))
            return _sf(d.get("weight", 1.0), 1.0)

        edges.sort(key=lambda e: score(e[0], e[1], e[2]), reverse=True)

    total_e = len(edges)
    remove_total = int(round(float(frac) * total_e))
    remove_total = max(0, min(remove_total, total_e))

    steps = int(steps)
    steps = max(1, steps)
    ks = np.linspace(0, remove_total, steps + 1).round().astype(int).tolist()

    removed_order = [(u, v) for (u, v, _) in edges[:remove_total]]

    H = H0.copy()

    rows = []
    last_heavy = None
    for i, k in enumerate(ks):
        if i == 0:
            pass
        else:
            prev = ks[i - 1]
            for (u, v) in removed_order[prev:k]:
                if H.has_edge(u, v):
                    H.remove_edge(u, v)

        removed_frac = (k / total_e) if total_e else 0.0

        heavy = (i % int(max(1, compute_heavy_every)) == 0) or (i == steps)
        m = _compute_metrics_snapshot(
            H,
            eff_k=eff_k,
            seed=seed,
            heavy=heavy,
            compute_curvature=bool(st.session_state.get("__compute_curvature", False)),
            curvature_sample_edges=int(st.session_state.get("__curvature_sample_edges", 80)),
        )

        row = {
            "step": i,
            "removed_frac": float(removed_frac),
            "removed_k": int(k),
            "N": int(m.get("N", H.number_of_nodes())),
            "E": int(m.get("E", H.number_of_edges())),
            "C": int(m.get("C", np.nan)) if "C" in m else np.nan,
            "lcc_size": int(m.get("lcc_size", np.nan)) if "lcc_size" in m else np.nan,
            "lcc_frac": float(m.get("lcc_frac", np.nan)) if "lcc_frac" in m else np.nan,
            "density": float(m.get("density", np.nan)) if "density" in m else np.nan,
            "avg_degree": float(m.get("avg_degree", np.nan)) if "avg_degree" in m else np.nan,
            "clustering": float(m.get("clustering", np.nan)) if "clustering" in m else np.nan,
            "assortativity": float(m.get("assortativity", np.nan)) if "assortativity" in m else np.nan,
            "eff_w": float(m.get("eff_w", np.nan)) if "eff_w" in m else np.nan,
        }

        if heavy:
            row["mod"] = float(m.get("mod", np.nan)) if "mod" in m else np.nan
            row["l2_lcc"] = float(m.get("l2_lcc", np.nan)) if "l2_lcc" in m else np.nan
            last_heavy = {"mod": row["mod"], "l2_lcc": row["l2_lcc"]}
        else:
            row["mod"] = np.nan
            row["l2_lcc"] = np.nan

        rows.append(row)

    df_hist = pd.DataFrame(rows)
    df_hist = _forward_fill_heavy(df_hist)
    aux = {
        "removed_edges_order": removed_order,
        "total_edges": total_e,
        "kind": kind,
    }
    return df_hist, aux

# ============================================================
# 4) STATE
# ============================================================
def _init_state():
    defaults = {
        "graphs": {},                 
        "experiments": [],            
        "active_graph_id": None,
        "seed": 42,
        "last_upload_hash": None,
        "layout_seed_bump": 0,
        "last_suite_curves": None,
        "last_multi_curves": None,
        "last_exp_id": None,
        "__decomp_step": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

def add_graph(name: str, df_edges: pd.DataFrame, source: str, tags=None) -> str:
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

def save_experiment(name: str, graph_id: str, kind: str, params: dict, df_hist: pd.DataFrame):
    eid = new_id("EXP")
    st.session_state["experiments"].append({
        "id": eid,
        "name": name,
        "graph_id": graph_id,
        "attack_kind": kind,
        "params": params,
        "history": df_hist.copy(),
        "created_at": time.time(),
    })
    st.session_state["last_exp_id"] = eid
    return eid

def run_node_attack_suite(
    G: nx.Graph,
    graph_entry: dict,
    preset_spec: list,
    frac: float,
    steps: int,
    base_seed: int,
    eff_k: int,
    heavy_freq: int,
    rc_frac: float = 0.1,
    tag: str = ""
):
    """
    Node-attack batch runner.
    NOTE: src.attacks.run_attack now supports adaptive weak-node strategies.
    """
    curves = []

    for block in preset_spec:
        kind = block["kind"]
        nseeds = int(block.get("seeds", 1))

        for i in range(nseeds):
            seed_i = int(base_seed) + 1000 * (abs(hash(kind)) % 97) + i

            df_hist, aux = run_attack(
                G, kind, float(frac), int(steps), int(seed_i), int(eff_k),
                rc_frac=float(rc_frac), compute_heavy_every=int(heavy_freq)
            )
            df_hist = _forward_fill_heavy(df_hist)
            removed_order = _extract_removed_order(aux) or _fallback_removal_order(G, kind, seed_i)
            aux_payload = {"removed_order": removed_order, "mode": "src_run_attack_or_fallback"}

            phase_info = classify_phase_transition(df_hist)

            label = f"{graph_entry['name']} | {kind} | seed={seed_i}"
            if tag:
                label += f" [{tag}]"

            save_experiment(
                name=label,
                graph_id=graph_entry["id"],
                kind=kind,
                params={
                    "attack_family": "node",
                    "frac": float(frac),
                    "steps": int(steps),
                    "seed": int(seed_i),
                    "phase": phase_info,
                    "compute_heavy_every": int(heavy_freq),
                    "eff_k": int(eff_k),
                    "rc_frac": float(rc_frac),
                    **aux_payload,
                },
                df_hist=df_hist,
            )
            curves.append((label, df_hist))

    return curves

def emulate_node_attack_from_order(
    G: nx.Graph,
    removed_order: list,
    frac: float,
    steps: int,
    seed: int,
    eff_k: int,
    compute_heavy_every: int = 2,
):
    """
    Static-order node removal (for weak attacks when src.run_attack doesn't support them).
    Returns df_hist like run_attack.
    """
    H0 = as_simple_undirected(G)
    N0 = H0.number_of_nodes()
    if N0 == 0:
        return pd.DataFrame([{"step": 0, "removed_frac": 0.0, "N": 0, "E": 0, "lcc_frac": 0.0}])

    remove_total = int(round(float(frac) * N0))
    remove_total = max(0, min(remove_total, len(removed_order)))

    ks = np.linspace(0, remove_total, int(steps) + 1).round().astype(int).tolist()
    removed_order = [n for n in removed_order if n in H0]
    removed_order = removed_order[:remove_total]

    H = H0.copy()
    rows = []
    for i, k in enumerate(ks):
        if i > 0:
            prev = ks[i - 1]
            for n in removed_order[prev:k]:
                if H.has_node(n):
                    H.remove_node(n)

        removed_frac = (k / N0) if N0 else 0.0
        heavy = (i % int(max(1, compute_heavy_every)) == 0) or (i == int(steps))
        m = _compute_metrics_snapshot(
            H,
            eff_k=eff_k,
            seed=seed,
            heavy=heavy,
            compute_curvature=bool(st.session_state.get("__compute_curvature", False)),
            curvature_sample_edges=int(st.session_state.get("__curvature_sample_edges", 80)),
        )

        row = {
            "step": i,
            "removed_frac": float(removed_frac),
            "removed_k": int(k),
            "N": int(m.get("N", H.number_of_nodes())),
            "E": int(m.get("E", H.number_of_edges())),
            "C": int(m.get("C", np.nan)) if "C" in m else np.nan,
            "lcc_size": int(m.get("lcc_size", np.nan)) if "lcc_size" in m else np.nan,
            "lcc_frac": float(m.get("lcc_frac", np.nan)) if "lcc_frac" in m else np.nan,
            "density": float(m.get("density", np.nan)) if "density" in m else np.nan,
            "avg_degree": float(m.get("avg_degree", np.nan)) if "avg_degree" in m else np.nan,
            "clustering": float(m.get("clustering", np.nan)) if "clustering" in m else np.nan,
            "assortativity": float(m.get("assortativity", np.nan)) if "assortativity" in m else np.nan,
            "eff_w": float(m.get("eff_w", np.nan)) if "eff_w" in m else np.nan,
            "mod": float(m.get("mod", np.nan)) if heavy else np.nan,
            "l2_lcc": float(m.get("l2_lcc", np.nan)) if heavy else np.nan,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = _forward_fill_heavy(df)
    return df

def run_edge_attack_suite(
    G: nx.Graph,
    graph_entry: dict,
    preset_spec: list,
    frac: float,
    steps: int,
    base_seed: int,
    eff_k: int,
    heavy_freq: int,
    tag: str = ""
):
    curves = []
    for block in preset_spec:
        kind = block["kind"]
        nseeds = int(block.get("seeds", 1))
        for i in range(nseeds):
            seed_i = int(base_seed) + 1000 * (abs(hash(kind)) % 97) + i
            df_hist, aux = run_edge_attack(
                G, kind, float(frac), int(steps), int(seed_i), int(eff_k),
                compute_heavy_every=int(heavy_freq)
            )
            df_hist = _forward_fill_heavy(df_hist)
            phase_info = classify_phase_transition(df_hist)

            label = f"{graph_entry['name']} | {kind} | seed={seed_i}"
            if tag:
                label += f" [{tag}]"

            save_experiment(
                name=label,
                graph_id=graph_entry["id"],
                kind=kind,
                params={
                    "attack_family": "edge",
                    "frac": float(frac),
                    "steps": int(steps),
                    "seed": int(seed_i),
                    "phase": phase_info,
                    "compute_heavy_every": int(heavy_freq),
                    "eff_k": int(eff_k),
                    "removed_edges_order": aux.get("removed_edges_order", []),
                    "total_edges": aux.get("total_edges", None),
                },
                df_hist=df_hist,
            )
            curves.append((label, df_hist))
    return curves

# ============================================================
# 5) SIDEBAR (IO, UPLOAD, FILTERS)
# ============================================================
with st.sidebar:
    st.title("🎛️ Kodik Lab")

    with st.expander("📥 Импорт / Экспорт", expanded=False):
        tab_io1, tab_io2 = st.tabs(["Workspace", "Experiments"])

        with tab_io1:
            if st.button("Export Workspace (JSON)"):
                b = export_workspace_json(st.session_state["graphs"], st.session_state["experiments"])
                st.download_button("Скачать workspace.json", b, "workspace.json", "application/json")

            up_ws = st.file_uploader("Загрузить workspace", type=["json"], key="up_ws")
            if up_ws:
                try:
                    gs, ex = import_workspace_json(up_ws.getvalue())
                    st.session_state["graphs"] = gs
                    st.session_state["experiments"] = ex
                    if gs:
                        st.session_state["active_graph_id"] = list(gs.keys())[0]
                    st.success("Workspace загружен!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

        with tab_io2:
            if st.button("Export Exps Only"):
                b = export_experiments_json(st.session_state["experiments"])
                st.download_button("Скачать experiments.json", b, "experiments.json", "application/json")

            up_exps = st.file_uploader("Импорт experiments.json", type=["json"], key="up_exps")
            if up_exps:
                try:
                    ex = import_experiments_json(up_exps.getvalue())
                    if isinstance(ex, list):
                        st.session_state["experiments"].extend(ex)
                    st.success("Experiments импортированы!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")
    st.subheader("📂 Загрузка данных")
    uploaded_file = st.file_uploader("CSV / Excel (Fixed Format)", type=["csv", "xlsx", "xls"])

    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        if file_hash != st.session_state["last_upload_hash"]:
            try:
                df_raw = load_uploaded_any(file_bytes, uploaded_file.name)
                df_edges, meta = coerce_fixed_format(df_raw)
                add_graph(
                    name=uploaded_file.name,
                    df_edges=df_edges,
                    source="upload",
                    tags=meta
                )
                st.session_state["last_upload_hash"] = file_hash
                st.toast(f"Граф {uploaded_file.name} успешно добавлен!", icon="✅")
                st.rerun()
            except Exception as e:
                st.error(f"Ошибка парсинга: {e}")

    st.markdown("---")
    st.subheader("⚙️ Глобальные фильтры")
    min_conf = st.number_input("Min Confidence", 0, 100, 0, help="Отсечь ребра с низкой уверенностью")
    min_weight = st.number_input("Min Weight", 0.0, 1000.0, 0.0, step=0.1, help="Отсечь ребра с малым весом")

    st.markdown("---")
    st.subheader("📈 Визуализация")
    if "plot_height" not in st.session_state:
        st.session_state["plot_height"] = 900
    if "norm_mode" not in st.session_state:
        st.session_state["norm_mode"] = "none"

    st.session_state["plot_height"] = st.slider(
        "Высота графиков",
        600, 1400, int(st.session_state["plot_height"]),
        step=50,
    )
    st.session_state["norm_mode"] = st.selectbox(
        "Нормировка кривых",
        ["none", "rel0", "delta0", "minmax", "zscore"],
        index=["none", "rel0", "delta0", "minmax", "zscore"].index(st.session_state["norm_mode"]),
        help="rel0: y/y0, delta0: y-y0, minmax: [0..1], zscore: (y-mean)/std",
    )

    st.markdown("---")
    if st.button("🗑️ Сбросить всё", type="primary"):
        st.session_state["graphs"] = {}
        st.session_state["experiments"] = []
        st.session_state["active_graph_id"] = None
        st.session_state["last_suite_curves"] = None
        st.session_state["last_multi_curves"] = None
        st.session_state["last_exp_id"] = None
        st.session_state["last_upload_hash"] = None
        st.session_state["__decomp_step"] = 0
        st.rerun()

# ============================================================
# 6) TOP BAR (STICKY)
# ============================================================
def render_top_bar():
    graphs = st.session_state["graphs"]
    active_gid = st.session_state["active_graph_id"]

    st.markdown('<div class="sticky-header">', unsafe_allow_html=True)

    if not graphs:
        st.warning("⚠️ Workspace пуст. Загрузите файл слева или создайте демо-граф.")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if st.button("🎲 Создать демо-граф (ER)", use_container_width=True):
                G_demo = make_er_gnm(200, 800, 42)
                edges = [[u, v, 1.0, 1.0] for u, v in G_demo.edges()]
                df_demo = pd.DataFrame(edges, columns=["src", "dst", "weight", "confidence"])
                add_graph("Demo ER Graph", df_demo, "demo:ER", {"src_col": "src", "dst_col": "dst"})
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        return None

    options = list(graphs.keys())
    options.sort(key=lambda k: graphs[k]["created_at"])
    if active_gid not in options:
        active_gid = options[0]
        st.session_state["active_graph_id"] = active_gid

    col1, col2, col3, col4 = st.columns([3, 2, 1, 1])

    with col1:
        selected = st.selectbox(
            "Активный граф",
            options,
            index=options.index(active_gid),
            format_func=lambda x: f"{graphs[x]['name']} ({graphs[x]['source']})",
            label_visibility="collapsed"
        )
        if selected != active_gid:
            st.session_state["active_graph_id"] = selected
            st.rerun()

    entry = graphs[selected]

    with col2:
        new_name = st.text_input(
            "Rename",
            value=entry["name"],
            label_visibility="collapsed",
            placeholder="Имя графа"
        )

    with col3:
        if st.button("💾 Rename", use_container_width=True):
            st.session_state["graphs"][selected]["name"] = new_name
            st.rerun()

    with col4:
        if st.button("❌ Delete", type="primary", use_container_width=True):
            del st.session_state["graphs"][selected]
            st.session_state["experiments"] = [e for e in st.session_state["experiments"] if e.get("graph_id") != selected]
            remaining = list(st.session_state["graphs"].keys())
            st.session_state["active_graph_id"] = remaining[0] if remaining else None
            st.session_state["last_suite_curves"] = None
            st.session_state["last_multi_curves"] = None
            st.session_state["last_exp_id"] = None
            st.session_state["__decomp_step"] = 0
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    return entry

active_entry = render_top_bar()
if not active_entry:
    # Важно: не останавливаем приложение ДО создания табов.
    tab_main, tab_energy, tab_struct, tab_null, tab_attack, tab_compare = st.tabs([
        "📊 Дэшборд",
        "⚡ Energy & Dynamics",
        "🕸️ Структура и 3D",
        "🧪 Нулевые модели",
        "💥 Attack Lab",
        "🆚 Сравнение",
    ])
    with tab_main:
        st.warning("Workspace пуст. Слева загрузи файл или создай демо-граф.")
    with tab_energy:
        st.info("Сначала нужен граф в Workspace (вкладка Energy & Dynamics).")
    with tab_struct:
        st.info("Сначала нужен граф в Workspace.")
    with tab_null:
        st.info("Сначала нужен граф в Workspace.")
    with tab_attack:
        st.info("Сначала нужен граф в Workspace.")
    with tab_compare:
        st.info("Сначала нужны эксперименты/атаки.")
    st.stop()

# ============================================================
# 7) BUILD ACTIVE GRAPH
# ============================================================
df_edges = active_entry["edges"]
src_col = active_entry["tags"].get("src_col", df_edges.columns[0])
dst_col = active_entry["tags"].get("dst_col", df_edges.columns[1])

# Cache key should avoid hashing the full DataFrame repeatedly.
df_hash = hashlib.md5(pd.util.hash_pandas_object(df_edges).values).hexdigest()

# Fast filtering (cached) and cheap counts. Full NetworkX graph is built lazily after user action.
df_filtered = _filter_edges_cached(
    active_entry["id"],
    df_hash,
    src_col,
    dst_col,
    float(min_conf),
    float(min_weight),
)
est_nodes, est_edges = _quick_counts(df_filtered, src_col, dst_col)

if "__analysis_mode" not in st.session_state:
    st.session_state["__analysis_mode"] = "Global (Весь граф)"

with st.sidebar:
    st.markdown("### 📊 Текущий граф")
    st.caption(f"ID: {active_entry['id']}")
    c1, c2 = st.columns(2)
    c1.metric("Nodes (быстро)", est_nodes)
    c2.metric("Edges (после фильтров)", est_edges)

    st.markdown("---")
    st.markdown("**🔍 Область анализа**")
    analysis_mode = st.radio(
        "Режим",
        ["Global (Весь граф)", "LCC (Гигантская комп.)"],
        index=0 if st.session_state["__analysis_mode"].startswith("Global") else 1,
    )
    st.session_state["__analysis_mode"] = analysis_mode

    seed_val = st.number_input("Random Seed", value=int(st.session_state["seed"]), step=1)
    st.session_state["seed"] = int(seed_val)

    st.markdown("---")
    st.markdown("**🐢 Тяжёлые метрики**")
    if "__curvature_sample_edges" not in st.session_state:
        st.session_state["__curvature_sample_edges"] = 80
    if "__compute_curvature_now" not in st.session_state:
        st.session_state["__compute_curvature_now"] = False

    curv_edges = st.slider(
        "κ: сколько рёбер сэмплировать",
        min_value=20,
        max_value=300,
        value=int(st.session_state["__curvature_sample_edges"]),
        step=10,
    )
    st.session_state["__curvature_sample_edges"] = int(curv_edges)

    if st.button("Compute Ricci (slow)", use_container_width=True):
        # Signal to compute curvature later when the graph is available.
        st.session_state["__compute_curvature_now"] = True

    st.markdown("---")
    # Stop-crane: prevent automatic heavy recomputation on every UI change.
    graph_key = (
        f"{active_entry['id']}|{df_hash}|{src_col}|{dst_col}|"
        f"{float(min_conf)}|{float(min_weight)}|{analysis_mode}"
    )
    load_graph = st.button("Load graph", type="primary", use_container_width=True)
    if load_graph:
        st.session_state["layout_seed_bump"] = int(st.session_state.get("layout_seed_bump", 0)) + 1
        st.session_state["__last_graph_key"] = graph_key

# Lazily build graph + metrics only after explicit user action.
metrics_cache_key = f"metrics_{graph_key}"
G_full = None
G_view = None
met = None

if load_graph:
    with st.spinner("Строю граф…"):
        G_full = _build_graph_cached(
            active_entry["id"],
            df_hash,
            src_col,
            dst_col,
            float(min_conf),
            float(min_weight),
            "Global (Весь граф)",
        )
        G_view = _build_graph_cached(
            active_entry["id"],
            df_hash,
            src_col,
            dst_col,
            float(min_conf),
            float(min_weight),
            analysis_mode,
        )
    with st.spinner("Считаю метрики…"):
        met = _metrics_cached(
            active_entry["id"],
            df_hash,
            src_col,
            dst_col,
            float(min_conf),
            float(min_weight),
            analysis_mode,
            int(seed_val),
            False,
            int(st.session_state.get("__curvature_sample_edges", 80)),
        )
    with st.spinner("Готовлю layout…"):
        # Cache a quick 2D layout explicitly on demand.
        st.session_state[f"layout2d_{graph_key}"] = compute_layout_cached(G_view)
    st.success("Graph ready")
    st.session_state[metrics_cache_key] = met
elif metrics_cache_key in st.session_state:
    G_full = _build_graph_cached(
        active_entry["id"],
        df_hash,
        src_col,
        dst_col,
        float(min_conf),
        float(min_weight),
        "Global (Весь граф)",
    )
    G_view = _build_graph_cached(
        active_entry["id"],
        df_hash,
        src_col,
        dst_col,
        float(min_conf),
        float(min_weight),
        analysis_mode,
    )
    met = st.session_state.get(metrics_cache_key)
else:
    # При пустом состоянии показываем общий prompt, чтобы не дублировать его в табах.
    st.info("👋 Выберите параметры и нажмите **'Load graph'** в сайдбаре для начала анализа.")
    # Не стопаем — пусть отрисуются табы и UI.
    G_full = None
    G_view = None
    met = None

# Trigger curvature computation only after the user explicitly requests it.
curvature_cache_key = (
    f"curvature_{graph_key}|{int(st.session_state.get('__curvature_sample_edges', 80))}|{int(seed_val)}"
)
if (G_view is not None) and st.session_state.get("__compute_curvature_now"):
    st.session_state["__compute_curvature_now"] = False
    with st.spinner("Считаю Ricci (это может занять время)…"):
        curvature_result = compute_curvature_cached(
            G_view,
            sample_edges=int(st.session_state.get("__curvature_sample_edges", 80)),
            seed=int(seed_val),
        )
    st.session_state[curvature_cache_key] = curvature_result
    st.success("Ricci computed")

if met is not None:
    cached_curvature = st.session_state.get(curvature_cache_key)
    if cached_curvature:
        # Merge curvature metrics into the main metrics payload for UI rendering.
        met = {**met, **cached_curvature}

# ============================================================
# 8) MAIN TABS (Attack/Compare are in PART 2)
# ============================================================
if G_view is None:
    st.markdown("### Загрузка графа")
    st.markdown("Нажмите **Load graph** ниже, чтобы начать расчёты и визуализацию.")
    center_left, center_mid, center_right = st.columns([1, 2, 1])
    with center_mid:
        load_graph_main = st.button(
            "Load graph",
            type="primary",
            use_container_width=True,
            key="load_graph_main",
        )
    # Обновляем флаг запуска, чтобы кнопка в центре вела к тем же вычислениям.
    if load_graph_main:
        st.session_state["layout_seed_bump"] = int(st.session_state.get("layout_seed_bump", 0)) + 1
        st.session_state["__last_graph_key"] = graph_key
        load_graph = True

tab_main, tab_energy, tab_struct, tab_null, tab_attack, tab_compare = st.tabs([
    "📊 Дэшборд",
    "⚡ Energy & Dynamics",
    "🕸️ Структура и 3D",
    "🧪 Нулевые модели",
    "💥 Attack Lab",
    "🆚 Сравнение",
])

# ------------------------------
# TAB: DASHBOARD
# ------------------------------
with tab_main:
    if G_view is None:
        pass
    else:
        st.header(f"Обзор: {active_entry['name']}")
        if G_view.number_of_nodes() > 1500:
            st.warning("⚠️ Граф большой. Тяжелые метрики (Ricci, Efficiency) считаются в фоновом режиме.")

        render_dashboard_metrics(G_view, met)

        st.markdown("---")

        render_dashboard_charts(G_view, _apply_plot_defaults)

# ------------------------------
# TAB: ENERGY & DYNAMICS
# ------------------------------
with tab_energy:
    st.header("⚡ Динамика и распространение (Energy Flow)")

    if G_view is None:
        st.info("Сначала загрузите граф в сайдбаре (Load graph).")
    else:
        # --- БЛОК 1: МОДЕЛЬ И ИСТОЧНИКИ ---
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("1. Физика процесса")
            flow_mode_ui = st.selectbox(
                "Тип распространения",
                ["phys", "rw", "evo"],
                help="Phys: давление/поток (как вода). RW: диффузия (как газ).",
            )
            rw_impulse = st.toggle("Импульсный режим (всплеск)", value=True)

            # Логика источников с пояснением.
            if "energy_sources" not in st.session_state:
                st.session_state["energy_sources"] = []

            sources_ui = st.multiselect(
                "Источники (откуда течет)",
                options=list(G_view.nodes()),
                default=st.session_state.get("energy_sources", []),
                key="src_select",
            )
            st.session_state["energy_sources"] = sources_ui

            # Вычисляем и показываем авто-источник, если список пуст.
            final_sources = list(sources_ui)
            if not final_sources:
                # Быстрый расчет "сильного" узла для UI.
                deg = dict(G_view.degree(weight="weight"))
                auto_src = max(deg, key=deg.get)
                st.info(f"🤖 Авто-выбор источника: узел **{auto_src}** (max strength)")

        with c2:
            st.subheader("2. Параметры потока")
            if flow_mode_ui == "phys":
                phys_inj = st.slider("Сила впрыска (Injection)", 0.1, 5.0, 1.0, 0.1)
                phys_leak = st.slider("Утечка (Leak)", 0.0, 0.1, 0.005, 0.001)
                phys_cap = st.selectbox("Емкость узлов", ["strength", "degree"])
                st.session_state["__phys_injection"] = phys_inj
                st.session_state["__phys_leak"] = phys_leak
                st.session_state["__phys_cap"] = phys_cap
            else:
                st.info("Для RW/Evo параметров меньше.")

            flow_steps = st.slider("Длительность (шаги)", 10, 200, 50)

        st.markdown("---")

        # --- БЛОК 2: ВИЗУАЛИЗАЦИЯ ---
        st.subheader("🎨 Настройка Вида (Сделай красиво)")

        vc1, vc2, vc3 = st.columns(3)
        with vc1:
            # Важный слайдер для "замедления".
            anim_duration = st.slider(
                "Скорость анимации (мс/кадр)",
                50,
                1000,
                150,
                50,
                help="Больше = медленнее. Позволяет вращать граф во время полета.",
            )
            vis_contrast = st.slider("Яркость (Gamma)", 1.0, 10.0, 4.5)
        with vc2:
            node_size_energy = st.slider("Размер узлов", 2, 20, 7)
            vis_clip = st.slider("Срез пиков (Clip)", 0.0, 0.5, 0.05)
        with vc3:
            edge_subset_mode = st.selectbox("Отрисовка связей", ["top_flux", "top_weight", "all"], index=0)
            max_edges_viz = st.slider("Макс. кол-во ребер", 100, 5000, 1500)

        # КНОПКА ЗАПУСКА
        if st.button("🔥 ЗАПУСТИТЬ СИМУЛЯЦИЮ", type="primary", use_container_width=True):
            with st.spinner("Моделирование физики..."):
                # Layout.
                base_seed = int(seed_val) + int(st.session_state.get("layout_seed_bump", 0))
                pos3d_local = _layout_cached(
                    active_entry["id"],
                    df_hash,
                    src_col,
                    dst_col,
                    float(min_conf),
                    float(min_weight),
                    analysis_mode,
                    base_seed,
                )

                # Simulation.
                src_key = tuple(final_sources) if final_sources else tuple()

                # Параметры физики берем из стейта или дефолтов.
                inj_val = float(st.session_state.get("__phys_injection", 1.0))
                leak_val = float(st.session_state.get("__phys_leak", 0.005))
                cap_val = str(st.session_state.get("__phys_cap", "strength"))

                node_frames, edge_frames = _energy_frames_cached(
                    active_entry["id"],
                    df_hash,
                    src_col,
                    dst_col,
                    float(min_conf),
                    float(min_weight),
                    analysis_mode,
                    steps=int(flow_steps),
                    flow_mode=str(flow_mode_ui),
                    damping=0.98,  # Дефолт.
                    sources=src_key,
                    phys_injection=inj_val,
                    phys_leak=leak_val,
                    phys_cap_mode=cap_val,
                    rw_impulse=bool(rw_impulse),
                )

                # Rendering.
                fig_flow = make_energy_flow_figure_3d(
                    G_view,
                    pos3d_local,
                    steps=int(flow_steps),
                    node_frames=node_frames,
                    edge_frames=edge_frames,
                    # Передаем параметры визуализации.
                    node_size=int(node_size_energy),
                    vis_contrast=float(vis_contrast),
                    vis_clip=float(vis_clip),
                    # Скорость анимации.
                    anim_duration=int(anim_duration),
                    # Фильтрация.
                    max_edges_viz=int(max_edges_viz),
                    edge_subset_mode=str(edge_subset_mode),
                    # Цвета.
                    vis_log=True,
                )

            st.plotly_chart(fig_flow, use_container_width=True, key="plot_energy_flow")

# ------------------------------
# TAB: STRUCTURE & 3D (static)
# ------------------------------
with tab_struct:
    if G_view is None:
        pass
    else:
        if G_view.number_of_nodes() > 1500:
            st.warning("⚠️ Граф большой. Тяжелые метрики (Ricci, Efficiency) считаются в фоновом режиме.")
        col_vis_ctrl, col_vis_main = st.columns([1, 4])

        with col_vis_ctrl:
            st.subheader("Настройки 3D")
            show_labels = st.checkbox("Показать ID узлов", False)
            node_size = st.slider("Размер узлов", 1, 20, 4)
            layout_mode = st.selectbox("Layout", ["Fixed (по исходному графу)", "Recompute (по текущему виду)"], index=0)

            st.info("3D-визуализация: фиксированный layout лучше для сравнения по шагам (не прыгает).")

            if st.button("🔄 Обновить layout seed (анти-кэш)"):
                st.session_state["layout_seed_bump"] = int(st.session_state.get("layout_seed_bump", 0)) + 1

            # Edge overlay options for 3D (coloring by edge-specific metrics).
            edge_overlay_ui = st.selectbox(
                "Разметка рёбер",
                [
                    "Ricci sign (κ<0/κ>0)",
                    "Energy flux (RW)",
                    "Energy flux (Demetrius)",
                    "Weight (log10)",
                    "Confidence",
                    "None",
                ],
                index=0,
            )

        with col_vis_main:
            if G_view.number_of_nodes() > 2000:
                st.warning(f"Граф большой ({G_view.number_of_nodes()} узлов). 3D может тормозить.")

            # Seed учитывает "анти-кэш" и делает layout детерминированным между перерисовками.
            base_seed = int(seed_val) + int(st.session_state.get("layout_seed_bump", 0))

            # 1) Получаем pos3d (режимы остаются детерминированными через seed).
            if layout_mode.startswith("Fixed"):
                pos3d = _layout_cached(
                    active_entry["id"],
                    df_hash,
                    src_col,
                    dst_col,
                    float(min_conf),
                    float(min_weight),
                    analysis_mode,
                    base_seed,
                )
            else:
                pos3d = _layout_cached(
                    active_entry["id"],
                    df_hash,
                    src_col,
                    dst_col,
                    float(min_conf),
                    float(min_weight),
                    analysis_mode,
                    base_seed,
                )

            edge_overlay = "ricci"
            flow_mode = "rw"
            if edge_overlay_ui.startswith("Energy flux"):
                edge_overlay = "flux"
                flow_mode = "evo" if "Demetrius" in edge_overlay_ui else "rw"
            elif edge_overlay_ui.startswith("Weight"):
                edge_overlay = "weight"
            elif edge_overlay_ui.startswith("Confidence"):
                edge_overlay = "confidence"
            elif edge_overlay_ui.startswith("None"):
                edge_overlay = "none"

            # 2) Всегда строим трэйсы, чтобы 3D работал и для Fixed, и для Recompute.
            edge_traces, node_trace = make_3d_traces(
                G_view,
                pos3d,
                show_scale=True,
                edge_overlay=edge_overlay,
                flow_mode=flow_mode,
            )

            # 3) Рисуем внутри col_vis_main, чтобы не ломать сетку.
            if node_trace is not None:
                node_trace.marker.size = node_size
                if show_labels:
                    node_trace.mode = "markers+text"

                fig_3d = go.Figure(data=[*edge_traces, node_trace])
                fig_3d.update_layout(
                    title=f"3D Structure: {active_entry['name']}",
                    template="plotly_dark",
                    showlegend=False,
                    height=820,
                    margin=dict(l=0, r=0, t=30, b=0),
                    scene=dict(
                        xaxis=dict(showbackground=False, showticklabels=False, title=""),
                        yaxis=dict(showbackground=False, showticklabels=False, title=""),
                        zaxis=dict(showbackground=False, showticklabels=False, title=""),
                    ),
                )
                st.plotly_chart(fig_3d, use_container_width=True, key="plot_struct_3d")
            else:
                st.write("Граф пуст.")

        st.markdown("---")
        st.subheader("Матрица смежности (heatmap)")
        if G_view.number_of_nodes() < 1000 and G_view.number_of_nodes() > 0:
            adj = nx.adjacency_matrix(as_simple_undirected(G_view), weight="weight").todense()
            fig_hm = px.imshow(adj, title="Adjacency Heatmap", color_continuous_scale="Viridis")
            fig_hm.update_layout(template="plotly_dark", height=760, width=760)
            st.plotly_chart(fig_hm, use_container_width=False, key="plot_adj_heatmap")
        else:
            st.info("Матрица слишком большая для отображения (N >= 1000) или граф пуст.")

# ------------------------------
# TAB: NULL MODELS
# ------------------------------
with tab_null:
    if G_view is None:
        pass
    else:
        st.header("🧪 Нулевые модели и синтетика")

        nm_col1, nm_col2 = st.columns([1, 2])

        with nm_col1:
            st.subheader("Параметры")
            null_kind = st.selectbox("Тип модели", ["ER G(n,m)", "Configuration Model", "Mix/Rewire (p)"])

            mix_p = 0.0
            if null_kind == "Mix/Rewire (p)":
                mix_p = st.slider("p (rewiring probability)", 0.0, 1.0, 0.2, 0.05, help=help_icon("Mix/Rewire"))

            nm_seed = st.number_input("Seed генерации", value=int(seed_val), step=1)
            new_name_suffix = st.text_input("Суффикс имени", value="_null")

            if st.button("⚙️ Создать и добавить", type="primary"):
                with st.spinner("Генерация..."):
                    if null_kind == "ER G(n,m)":
                        G_new = make_er_gnm(G_full.number_of_nodes(), G_full.number_of_edges(), seed=int(nm_seed))
                        src_tag = "ER"
                    elif null_kind == "Configuration Model":
                        G_new = make_configuration_model(G_full, seed=int(nm_seed))
                        src_tag = "CFG"
                    else:
                        G_new = rewire_mix(G_full, p=float(mix_p), seed=int(nm_seed))
                        src_tag = f"MIX(p={mix_p})"

                    edges = [[u, v, 1.0, 1.0] for u, v in as_simple_undirected(G_new).edges()]
                    df_new = pd.DataFrame(edges, columns=["src", "dst", "weight", "confidence"])

                    add_graph(
                        name=f"{active_entry['name']}{new_name_suffix}",
                        df_edges=df_new,
                        source=f"null:{src_tag}",
                        tags={"src_col": "src", "dst_col": "dst"}
                    )
                    st.success("Граф создан. Переключаюсь на него...")
                    st.rerun()

        with nm_col2:
            st.info("Быстрая проверка против ER-ожиданий (очень грубо):")
            N = G_view.number_of_nodes()
            M = G_view.number_of_edges()
            er_density = 2 * M / (N * (N - 1)) if N > 1 else 0.0
            er_clustering = er_density

            met_light = met  
            cmp_df = pd.DataFrame({
                "Metric": ["Avg Degree", "Density", "Clustering (C)", "Modularity (примерно)"],
                "Active Graph": [met_light.get("avg_degree", np.nan), met_light.get("density", np.nan), met_light.get("clustering", np.nan), met_light.get("mod", np.nan)],
                "ER Expected": [met_light.get("avg_degree", np.nan), er_density, er_clustering, "~0.0"],
            })
            st.dataframe(cmp_df, use_container_width=True)

        # ============================================================
        # 9) ATTACK LAB (Node + Edge, presets, multi-graph, AUC, phase)
        # ============================================================
with tab_attack:
    if G_view is None:
        pass
    else:
        st.header("💥 Attack Lab (node + edge + weak)")

        # --------------------------
        # SINGLE RUN
        # --------------------------
        st.subheader("Single run")
        family = st.radio(
            "Тип атаки",
            ["Node (узлы)", "Edge (рёбра: слабые/сильные)", "Mix/Entropy (Hrish)"],
            horizontal=True,
        )

        col_setup, _ = st.columns([1, 2])

        with col_setup:
            with st.container(border=True):
                st.markdown("### Параметры")

                frac = st.slider("Доля удаления", 0.05, 0.95, 0.5, 0.05)
                steps = st.slider("Шаги", 5, 150, 30)
                seed_run = st.number_input("Seed", value=int(seed_val), step=1)

                with st.expander("Дополнительно"):
                    eff_k = st.slider("Efficiency samples (k)", 8, 256, 32)
                    heavy_freq = st.slider("Тяжёлые метрики каждые N шагов", 1, 10, 2)
                    tag = st.text_input("Тег", "")

                if family.startswith("Node"):
                    attack_ui = st.selectbox(
                        "Стратегия (узлы)",
                        [
                            "random",
                            "degree (Hubs)",
                            "betweenness (Bridges)",
                            "kcore (Deep Core)",
                            "richclub_top (Top Strength)",
                            "low_degree (Weak nodes)",       
                            "weak_strength (Weak strength)",
                        ],
                    )
                    kind_map = {
                        "random": "random",
                        "degree (Hubs)": "degree",
                        "betweenness (Bridges)": "betweenness",
                        "kcore (Deep Core)": "kcore",
                        "richclub_top (Top Strength)": "richclub_top",
                        "low_degree (Weak nodes)": "low_degree",
                        "weak_strength (Weak strength)": "weak_strength",
                    }
                    kind = kind_map.get(attack_ui, "random")

                elif family.startswith("Edge"):
                    attack_ui = st.selectbox(
                        "Стратегия (рёбра)",
                        [
                            "weak_edges_by_weight",
                            "weak_edges_by_confidence",
                            "strong_edges_by_weight",
                            "strong_edges_by_confidence",
                            "ricci_most_negative (κ min)",
                            "ricci_most_positive (κ max)",
                            "ricci_abs_max (|κ| max)",
                            "flux_high_rw",
                            "flux_high_evo",
                            "flux_high_rw_x_neg_ricci",
                        ],
                        help=help_icon("Weak edges")
                    )
                    kind = str(attack_ui).split(" ")[0]

                else:
                    kind = st.selectbox(
                        "Режим Hrish",
                        [
                            "hrish_mix",
                            "mix_degree_preserving",
                            "mix_weightconf_preserving",
                        ],
                        help="hrish_mix = rewire (degree-preserving) + replace из нулевой модели.",
                    )
                    replace_from = st.selectbox("Replace source", ["ER", "CFG"], index=0)
                    alpha_rewire = st.slider("alpha (rewire)", 0.0, 1.0, 0.6, 0.05)
                    beta_replace = st.slider("beta (replace)", 0.0, 1.0, 0.4, 0.05)
                    swaps_per_edge = st.slider("swaps_per_edge", 0.0, 3.0, 0.5, 0.1)
                    st.caption("Ось X здесь: mix_frac (0..1), а не removed_frac.")

                if st.button("🚀 RUN", type="primary", use_container_width=True):
                    if family.startswith("Mix/Entropy"):
                        with st.spinner(f"Mix attack: {kind}"):
                            df_hist, aux = run_mix_attack(
                                G_view,
                                kind=str(kind),
                                steps=int(steps),
                                seed=int(seed_run),
                                eff_sources_k=int(eff_k),
                                heavy_every=int(heavy_freq),
                                alpha_rewire=float(alpha_rewire),
                                beta_replace=float(beta_replace),
                                swaps_per_edge=float(swaps_per_edge),
                                replace_from=str(replace_from),
                            )
                            df_hist = _forward_fill_heavy(df_hist)
                            phase_info = classify_phase_transition(
                                df_hist.rename(columns={"mix_frac": "removed_frac"})
                            )

                            label = f"{active_entry['name']} | mix:{kind} | seed={seed_run}"
                            if tag:
                                label += f" [{tag}]"

                            save_experiment(
                                name=label,
                                graph_id=active_entry["id"],
                                kind=str(kind),
                                params={
                                    "attack_family": "mix",
                                    "steps": int(steps),
                                    "seed": int(seed_run),
                                    "phase": phase_info,
                                    "eff_k": int(eff_k),
                                    "heavy_every": int(heavy_freq),
                                    **aux,
                                },
                                df_hist=df_hist,
                            )
                        st.success("Готово.")
                        st.rerun()

                    if family.startswith("Node"):
                        with st.spinner(f"Node attack: {kind}"):
                            df_hist, aux = run_attack(
                                G_view, kind, float(frac), int(steps), int(seed_run), int(eff_k),
                                rc_frac=0.1, compute_heavy_every=int(heavy_freq)
                            )
                            df_hist = _forward_fill_heavy(df_hist)
                            removed_order = _extract_removed_order(aux) or _fallback_removal_order(G_view, kind, int(seed_run))
                            phase_info = classify_phase_transition(df_hist)

                            label = f"{active_entry['name']} | node:{kind} | seed={seed_run}"
                            if tag:
                                label += f" [{tag}]"

                            save_experiment(
                                name=label,
                                graph_id=active_entry["id"],
                                kind=kind,
                                params={
                                    "attack_family": "node",
                                    "frac": float(frac),
                                    "steps": int(steps),
                                    "seed": int(seed_run),
                                    "phase": phase_info,
                                    "compute_heavy_every": int(heavy_freq),
                                    "eff_k": int(eff_k),
                                    "removed_order": removed_order,
                                    "mode": "src_run_attack_or_fallback",
                                },
                                df_hist=df_hist
                            )
                        st.success("Готово.")
                        st.rerun()

                    else:
                        with st.spinner(f"Edge attack: {kind}"):
                            df_hist, aux = run_edge_attack(
                                G_view, kind, float(frac), int(steps), int(seed_run), int(eff_k),
                                compute_heavy_every=int(heavy_freq)
                            )
                            df_hist = _forward_fill_heavy(df_hist)
                            phase_info = classify_phase_transition(df_hist)

                            label = f"{active_entry['name']} | edge:{kind} | seed={seed_run}"
                            if tag:
                                label += f" [{tag}]"

                            save_experiment(
                                name=label,
                                graph_id=active_entry["id"],
                                kind=kind,
                                params={
                                    "attack_family": "edge",
                                    "frac": float(frac),
                                    "steps": int(steps),
                                    "seed": int(seed_run),
                                    "phase": phase_info,
                                    "compute_heavy_every": int(heavy_freq),
                                    "eff_k": int(eff_k),
                                    "removed_edges_order": aux.get("removed_edges_order", []),
                                    "total_edges": aux.get("total_edges", None),
                                },
                                df_hist=df_hist
                            )
                        st.success("Готово.")
                        st.rerun()

        st.markdown("---")
        st.markdown("## Последний результат (для текущего графа)")

        exps_here = [e for e in st.session_state["experiments"] if e.get("graph_id") == active_entry["id"]]
        if not exps_here:
            st.info("Нет экспериментов. Запусти сверху.")
        else:
            exps_here.sort(key=lambda x: x["created_at"], reverse=True)
            last_exp = exps_here[0]
            df_res = _forward_fill_heavy(last_exp["history"].copy())
            params = last_exp.get("params") or {}
            fam = params.get("attack_family", "node")
            xcol = "mix_frac" if fam == "mix" and "mix_frac" in df_res.columns else "removed_frac"

            ph = (last_exp.get("params") or {}).get("phase", {})
            if ph:
                st.caption(
                    f"Phase: {'🔥 Abrupt' if ph.get('is_abrupt') else '🌊 Continuous'}"
                    f" | critical_x ≈ {float(ph.get('critical_x', 0.0)):.3f}"
                )

            tabA, tabB, tabC = st.tabs(["📉 Curves", "🌀 Phase views", "🧊 3D step-by-step"])

            with tabA:
                with st.expander("❔ Что означают метрики на графиках", expanded=False):
                    st.markdown(
                        "- **lcc_frac**: доля узлов в гигантской компоненте (порядковый параметр перколяции)\n"
                        "- **eff_w**: глобальная эффективность (в среднем насколько короткие пути; выше = сеть “связнее”)\n"
                        "- **l2_lcc**: λ₂ (алгебраическая связность) для LCC; близко к 0 = “на грани распада”\n"
                        "- **mod**: модульность сообществ; рост часто означает фрагментацию на кластеры\n"
                        "- **H_***: энтропии распределений (рост “случайности” структуры)\n"
                    )
                fig = fig_metrics_over_steps(
                    df_res,
                    title="Метрики по шагам",
                    normalize_mode=st.session_state["norm_mode"],
                    height=st.session_state["plot_height"],
                )
                fig.update_layout(template="plotly_dark")
                fig.update_traces(mode="lines+markers")
                fig.update_traces(line_width=3)
                fig = _apply_plot_defaults(fig, height=st.session_state["plot_height"])
                st.plotly_chart(fig, use_container_width=True, key="plot_attack_metrics")

                st.markdown("#### AUC (robustness) по выбранной метрике")
                y_axis = st.selectbox(
                    "Метрика для AUC",
                    [c for c in ["lcc_frac", "eff_w", "l2_lcc", "mod", "H_deg", "H_w", "H_conf", "H_tri"] if c in df_res.columns],
                    index=0,
                    key="auc_y_single",
                )
                st.caption(METRIC_HELP.get(y_axis, ""))

                if y_axis in df_res.columns and xcol in df_res.columns:
                    xs = pd.to_numeric(df_res[xcol], errors="coerce")
                    ys = pd.to_numeric(df_res[y_axis], errors="coerce")
                    mask = xs.notna() & ys.notna()
                    if mask.sum() >= 2:
                        auc_val = float(AUC_TRAP(ys[mask].to_numpy(), xs[mask].to_numpy()))
                        st.metric("AUC", f"{auc_val:.6f}")
                    else:
                        st.info("Недостаточно точек для AUC.")

                with st.expander("❓ Что на этих графиках", expanded=False):
                    txt = """
                    Ось X:
                      - removed_frac: доля удалённых узлов/рёбер (атаки).
                      - mix_frac: уровень энтропизации (Hrish mix), 0..1.

                    Ось Y:
                      - lcc_frac: доля LCC (перколяция).
                      - eff_w: эффективность (качество глобальной связности путей).
                      - l2_lcc: λ₂ (спектральная связность LCC).
                      - mod: модульность (структура сообществ).
                      - H_*: энтропии распределений (рост “случайности”).
                    """
                    st.text(textwrap.dedent(txt).strip())

                with tabB:
                    if xcol in df_res.columns and "lcc_frac" in df_res.columns:
                        fig_lcc = px.line(df_res, x=xcol, y="lcc_frac", title="Order parameter: LCC fraction vs removed fraction")
                        fig_lcc.update_layout(template="plotly_dark")
                        fig_lcc = _apply_plot_defaults(fig_lcc, height=780, y_range=_auto_y_range(df_res["lcc_frac"]))
                        st.plotly_chart(fig_lcc, use_container_width=True, key="plot_phase_lcc")

                    if xcol in df_res.columns and "lcc_frac" in df_res.columns:
                        dfp = df_res.sort_values(xcol).copy()
                        dx = pd.to_numeric(dfp[xcol], errors="coerce").diff()
                        dy = pd.to_numeric(dfp["lcc_frac"], errors="coerce").diff()
                        dfp["suscep"] = (dy / dx).replace([np.inf, -np.inf], np.nan)
                        fig_s = px.line(dfp, x=xcol, y="suscep", title="Susceptibility proxy: d(LCC)/dx")
                        fig_s.update_layout(template="plotly_dark")
                        fig_s = _apply_plot_defaults(fig_s, height=780, y_range=_auto_y_range(dfp["suscep"]))
                        st.plotly_chart(fig_s, use_container_width=True, key="plot_phase_suscep")

                    if "mod" in df_res.columns and "l2_lcc" in df_res.columns:
                        dfp2 = df_res.copy()
                        dfp2["mod"] = pd.to_numeric(dfp2["mod"], errors="coerce")
                        dfp2["l2_lcc"] = pd.to_numeric(dfp2["l2_lcc"], errors="coerce")
                        dfp2 = dfp2.dropna(subset=["mod", "l2_lcc"])
                        if not dfp2.empty:
                            fig_phase = px.line(dfp2, x="l2_lcc", y="mod", title="Phase portrait (trajectory): Q vs λ₂")
                            fig_phase.update_layout(template="plotly_dark")
                            fig_phase = _apply_plot_defaults(fig_phase, height=780)
                            st.plotly_chart(fig_phase, use_container_width=True, key="plot_phase_portrait")

                with tabC:
                    edge_overlay_ui = st.selectbox(
                        "Разметка рёбер (3D step-by-step)",
                        [
                            "Ricci sign (κ<0/κ>0)",
                            "Energy flux (RW)",
                            "Energy flux (Demetrius)",
                            "Weight (log10)",
                            "Confidence",
                            "None",
                        ],
                        index=0,
                        key="edge_overlay_tabc",
                    )
                    edge_overlay = "ricci"
                    flow_mode = "rw"
                    if edge_overlay_ui.startswith("Energy flux"):
                        edge_overlay = "flux"
                        flow_mode = "evo" if "Demetrius" in edge_overlay_ui else "rw"
                    elif edge_overlay_ui.startswith("Weight"):
                        edge_overlay = "weight"
                    elif edge_overlay_ui.startswith("Confidence"):
                        edge_overlay = "confidence"
                    elif edge_overlay_ui.startswith("None"):
                        edge_overlay = "none"

                    base_seed = int(seed_val) + int(st.session_state.get("layout_seed_bump", 0))
                    pos_base = _layout_cached(
                        active_entry["id"],
                        df_hash,
                        src_col,
                        dst_col,
                        float(min_conf),
                        float(min_weight),
                        analysis_mode,
                        base_seed,
                    )

                    if fam == "mix":
                        st.info("Для Mix/Entropy 3D-декомпозиция не поддерживается (нет порядка удаления).")
                    elif fam == "node":
                        removed_order = params.get("removed_order") or []
                        if not removed_order:
                            st.warning("Нет removed_order для 3D. (src.run_attack не дал, а fallback не сохранился.)")
                        else:
                            max_steps = max(1, len(df_res) - 1)
                            step_val = st.slider("Шаг (3D)", 0, max_steps, int(st.session_state.get("__decomp_step", 0)), key="__decomp_step_slider")
                            st.session_state["__decomp_step"] = int(step_val)

                            play = st.toggle("▶ Play", value=False, key="play3d")
                            fps = st.slider("FPS", 1, 10, 3, key="fps3d")

                            frac_here = float(df_res.iloc[int(step_val)]["removed_frac"]) if "removed_frac" in df_res.columns else (step_val / max_steps)
                            k_remove = int(round(frac_here * G_view.number_of_nodes()))
                            k_remove = max(0, min(k_remove, len(removed_order)))

                            removed_set = set(removed_order[:k_remove])
                            H = as_simple_undirected(G_view).copy()
                            H.remove_nodes_from([n for n in removed_set if H.has_node(n)])

                            pos_k = {n: pos_base[n] for n in H.nodes() if n in pos_base}
                            edge_traces, node_trace = make_3d_traces(
                                H,
                                pos_k,
                                show_scale=True,
                                edge_overlay=edge_overlay,
                                flow_mode=flow_mode,
                            )

                            if node_trace is not None:
                                fig = go.Figure(data=[*edge_traces, node_trace])
                                fig.update_layout(template="plotly_dark", height=860, showlegend=False)
                                fig.update_layout(title=f"Node removal | step={step_val}/{max_steps} | removed~{k_remove} | frac={frac_here:.3f}")
                                st.plotly_chart(fig, use_container_width=True, key="plot_attack_3d_node_step")
                            else:
                                st.info("На этом шаге граф пуст.")

                            if play:
                                time.sleep(1.0 / float(fps))
                                nxt = int(step_val) + 1
                                if nxt > max_steps:
                                    nxt = 0
                                st.session_state["__decomp_step"] = nxt
                                st.rerun()

                    else:
                        removed_edges_order = params.get("removed_edges_order") or []
                        total_edges = params.get("total_edges") or len(as_simple_undirected(G_view).edges())
                        if not removed_edges_order:
                            st.warning("Нет removed_edges_order для 3D.")
                        else:
                            max_steps = max(1, len(df_res) - 1)
                            step_val = st.slider("Шаг (3D)", 0, max_steps, int(st.session_state.get("__decomp_step", 0)), key="__decomp_step_slider_edge")
                            st.session_state["__decomp_step"] = int(step_val)

                            play = st.toggle("▶ Play", value=False, key="play3d_edge")
                            fps = st.slider("FPS", 1, 10, 3, key="fps3d_edge")

                            frac_here = float(df_res.iloc[int(step_val)]["removed_frac"]) if "removed_frac" in df_res.columns else (step_val / max_steps)
                            k_remove = int(round(frac_here * float(total_edges)))
                            k_remove = max(0, min(k_remove, len(removed_edges_order)))

                            H = as_simple_undirected(G_view).copy()
                            for (u, v) in removed_edges_order[:k_remove]:
                                if H.has_edge(u, v):
                                    H.remove_edge(u, v)

                            pos_k = {n: pos_base[n] for n in H.nodes() if n in pos_base}
                            edge_traces, node_trace = make_3d_traces(
                                H,
                                pos_k,
                                show_scale=True,
                                edge_overlay=edge_overlay,
                                flow_mode=flow_mode,
                            )

                            if node_trace is not None:
                                fig = go.Figure(data=[*edge_traces, node_trace])
                                fig.update_layout(template="plotly_dark", height=860, showlegend=False)
                                fig.update_layout(title=f"Edge removal | step={step_val}/{max_steps} | removed~{k_remove} edges | frac={frac_here:.3f}")
                                st.plotly_chart(fig, use_container_width=True, key="plot_attack_3d_edge_step")
                            else:
                                st.info("На этом шаге граф пуст.")

                            if play:
                                time.sleep(1.0 / float(fps))
                                nxt = int(step_val) + 1
                                if nxt > max_steps:
                                    nxt = 0
                                st.session_state["__decomp_step"] = nxt
                                st.rerun()

        st.markdown("---")

        # --------------------------
        # PRESET BATCH (same graph)
        # --------------------------
        st.subheader("Preset batch (на одном графе)")
        bcol1, bcol2 = st.columns([1, 2])

        with bcol1:
            batch_family = st.radio("Batch тип", ["Node presets", "Edge presets"], horizontal=True, key="batch_family")

            if batch_family.startswith("Node"):
                preset_name = st.selectbox("Preset", list(ATTACK_PRESETS_NODE.keys()), key="preset_node")
                preset = ATTACK_PRESETS_NODE[preset_name]
            else:
                preset_name = st.selectbox("Preset", list(ATTACK_PRESETS_EDGE.keys()), key="preset_edge")
                preset = ATTACK_PRESETS_EDGE[preset_name]

            frac_b = st.slider("Доля удаления (batch)", 0.05, 0.95, 0.5, 0.05, key="batch_frac")
            steps_b = st.slider("Шаги (batch)", 5, 150, 30, key="batch_steps")
            seed_b = st.number_input("Base seed (batch)", value=123, step=1, key="batch_seed")

            with st.expander("Batch advanced"):
                eff_k_b = st.slider("Efficiency k", 8, 256, 32, key="batch_effk")
                heavy_b = st.slider("Heavy every N", 1, 10, 2, key="batch_heavy")
                tag_b = st.text_input("Тег batch", "", key="batch_tag")

            if st.button("🚀 RUN PRESET SUITE", type="primary", use_container_width=True, key="run_suite"):
                with st.spinner(f"Running preset: {preset_name}"):
                    if batch_family.startswith("Node"):
                        curves = run_node_attack_suite(
                            G_view, active_entry, preset,
                            frac=float(frac_b), steps=int(steps_b), base_seed=int(seed_b),
                            eff_k=int(eff_k_b), heavy_freq=int(heavy_b),
                            rc_frac=0.1, tag=tag_b
                        )
                    else:
                        curves = run_edge_attack_suite(
                            G_view, active_entry, preset,
                            frac=float(frac_b), steps=int(steps_b), base_seed=int(seed_b),
                            eff_k=int(eff_k_b), heavy_freq=int(heavy_b),
                            tag=tag_b
                        )

                st.session_state["last_suite_curves"] = curves
                st.success(f"Готово: {len(curves)} прогонов сохранено.")
                st.rerun()

        with bcol2:
            curves = st.session_state.get("last_suite_curves")
            if curves:
                st.markdown("### Сравнение suite")
                y_axis = st.selectbox("Y", ["lcc_frac", "eff_w", "l2_lcc", "mod"], index=0, key="suite_y")
                fig = fig_compare_attacks(
                    curves,
                    "removed_frac",
                    y_axis,
                    f"Suite compare: {y_axis}",
                    normalize_mode=st.session_state["norm_mode"],
                    height=st.session_state["plot_height"],
                )
                fig.update_layout(template="plotly_dark")
                all_y = pd.concat([pd.to_numeric(df[y_axis], errors="coerce") for _, df in curves if y_axis in df.columns], ignore_index=True)
                fig = _apply_plot_defaults(fig, height=st.session_state["plot_height"], y_range=_auto_y_range(all_y))
                st.plotly_chart(fig, use_container_width=True, key="plot_suite_compare")

                st.markdown("#### AUC ranking")
                rows = []
                for name, df in curves:
                    if "removed_frac" in df.columns and y_axis in df.columns:
                        xs = pd.to_numeric(df["removed_frac"], errors="coerce")
                        ys = pd.to_numeric(df[y_axis], errors="coerce")
                        mask = xs.notna() & ys.notna()
                        if mask.sum() >= 2:
                            rows.append({"run": name, "AUC": float(AUC_TRAP(ys[mask].to_numpy(), xs[mask].to_numpy()))})
                if rows:
                    df_auc = pd.DataFrame(rows).sort_values("AUC", ascending=False)
                    st.dataframe(df_auc, use_container_width=True)
            else:
                st.info("Запусти suite слева, чтобы увидеть сравнение.")

        st.markdown("---")

        # --------------------------
        # MULTI-GRAPH BATCH
        # --------------------------
        st.subheader("Multi-graph batch (на нескольких графах)")
        graphs = st.session_state["graphs"]
        gid_list = list(graphs.keys())

        mg_col1, mg_col2 = st.columns([1, 2])

        with mg_col1:
            mg_family = st.radio("Multi тип", ["Node presets", "Edge presets"], horizontal=True, key="mg_family")

            sel_gids = st.selectbox(
                "Графы (multi) — выбери несколько в списке ниже",
                options=["(выбрать ниже)"],
                index=0,
                help="Основной выбор — в multiselect ниже"
            )

            sel_gids = st.multiselect(
                "Выбери графы",
                gid_list,
                default=[st.session_state["active_graph_id"]] if st.session_state["active_graph_id"] else [],
                format_func=lambda gid: f"{graphs[gid]['name']} ({graphs[gid]['source']})",
                key="mg_gids"
            )

            if mg_family.startswith("Node"):
                preset_name_mg = st.selectbox("Preset (multi)", list(ATTACK_PRESETS_NODE.keys()), key="mg_preset_node")
                preset_mg = ATTACK_PRESETS_NODE[preset_name_mg]
            else:
                preset_name_mg = st.selectbox("Preset (multi)", list(ATTACK_PRESETS_EDGE.keys()), key="mg_preset_edge")
                preset_mg = ATTACK_PRESETS_EDGE[preset_name_mg]

            mg_frac = st.slider("Доля удаления", 0.05, 0.95, 0.5, 0.05, key="mg_frac")
            mg_steps = st.slider("Шаги", 5, 150, 30, key="mg_steps")
            mg_seed = st.number_input("Base seed", value=321, step=1, key="mg_seed")

            with st.expander("Multi advanced"):
                mg_effk = st.slider("Efficiency k", 8, 256, 32, key="mg_effk")
                mg_heavy = st.slider("Heavy every N", 1, 10, 2, key="mg_heavy")
                mg_tag = st.text_input("Тег multi", "", key="mg_tag")

            if st.button("🚀 RUN MULTI-GRAPH SUITE", type="primary", use_container_width=True, key="run_mg"):
                if not sel_gids:
                    st.error("Выбери хотя бы один граф.")
                else:
                    all_curves = []
                    with st.spinner("Running multi-graph suite..."):
                        for gid in sel_gids:
                            entry = graphs[gid]
                            _df = filter_edges(
                                entry["edges"],
                                entry["tags"].get("src_col", "src"),
                                entry["tags"].get("dst_col", "dst"),
                                min_conf, min_weight
                            )
                            _G = build_graph_from_edges(_df, entry["tags"].get("src_col", "src"), entry["tags"].get("dst_col", "dst"))
                            if analysis_mode.startswith("LCC"):
                                _G = lcc_subgraph(_G)

                            if mg_family.startswith("Node"):
                                curves = run_node_attack_suite(
                                    _G, entry, preset_mg,
                                    frac=float(mg_frac), steps=int(mg_steps),
                                    base_seed=int(mg_seed), eff_k=int(mg_effk),
                                    heavy_freq=int(mg_heavy),
                                    rc_frac=0.1,
                                    tag=f"MG:{mg_tag}"
                                )
                            else:
                                curves = run_edge_attack_suite(
                                    _G, entry, preset_mg,
                                    frac=float(mg_frac), steps=int(mg_steps),
                                    base_seed=int(mg_seed), eff_k=int(mg_effk),
                                    heavy_freq=int(mg_heavy),
                                    tag=f"MG:{mg_tag}"
                                )

                            all_curves.extend(curves)

                    st.session_state["last_multi_curves"] = all_curves
                    st.success(f"Готово: {len(all_curves)} прогонов.")
                    st.rerun()

        with mg_col2:
            multi_curves = st.session_state.get("last_multi_curves")
            if multi_curves:
                st.markdown("### Multi сравнение")
                y = st.selectbox("Y (multi)", ["lcc_frac", "eff_w", "l2_lcc", "mod"], index=0, key="mg_y")
                fig = fig_compare_attacks(
                    multi_curves,
                    "removed_frac",
                    y,
                    f"Multi compare: {y}",
                    normalize_mode=st.session_state["norm_mode"],
                    height=st.session_state["plot_height"],
                )
                fig.update_layout(template="plotly_dark")
                all_y = pd.concat([pd.to_numeric(df[y], errors="coerce") for _, df in multi_curves if y in df.columns], ignore_index=True)
                fig = _apply_plot_defaults(fig, height=st.session_state["plot_height"], y_range=_auto_y_range(all_y))
                st.plotly_chart(fig, use_container_width=True, key="plot_multi_compare")
            else:
                st.info("Запусти multi suite слева, чтобы увидеть сравнение.")

        # ============================================================
        # 10) COMPARE TAB (saved graphs + saved experiments)
        # ============================================================
with tab_compare:
    if G_view is None:
        pass
    else:
        st.header("🆚 Сравнение")

        mode_cmp = st.radio("Что сравниваем?", ["Графы (скаляры)", "Эксперименты (траектории)"], horizontal=True)

        graphs = st.session_state["graphs"]
        all_gids = list(graphs.keys())

        if mode_cmp.startswith("Графы"):
            st.subheader("Сравнение скаляров по графам")
            selected_gids = st.multiselect(
                "Выберите графы",
                all_gids,
                default=[active_entry["id"]] if active_entry["id"] in all_gids else [],
                format_func=lambda gid: f"{graphs[gid]['name']} ({graphs[gid]['source']})",
            )

            scalar_metric = st.selectbox(
                "Метрика",
                ["density", "l2_lcc", "mod", "eff_w", "avg_degree", "clustering", "assortativity", "lcc_frac"],
                index=1
            )

            if selected_gids:
                rows = []
                for gid in selected_gids:
                    entry = graphs[gid]
                    _df = filter_edges(
                        entry["edges"],
                        entry["tags"].get("src_col", "src"),
                        entry["tags"].get("dst_col", "dst"),
                        min_conf, min_weight
                    )
                    _G = build_graph_from_edges(_df, entry["tags"].get("src_col", "src"), entry["tags"].get("dst_col", "dst"))
                    if analysis_mode.startswith("LCC"):
                        _G = lcc_subgraph(_G)

                    _m = calculate_metrics(_G, eff_sources_k=16, seed=42)
                    rows.append({"Name": entry["name"], scalar_metric: _m.get(scalar_metric, np.nan)})

                df_cmp = pd.DataFrame(rows)
                fig_bar = px.bar(df_cmp, x="Name", y=scalar_metric, title=f"Comparison: {scalar_metric}", color="Name")
                fig_bar.update_layout(template="plotly_dark", height=780)
                st.plotly_chart(fig_bar, use_container_width=True, key="plot_compare_bar")
                st.dataframe(df_cmp, use_container_width=True)
            else:
                st.info("Выбери графы.")

        else:
            st.subheader("Сравнение экспериментов (кривые)")
            exps = st.session_state["experiments"]
            if not exps:
                st.warning("Нет сохраненных экспериментов.")
            else:
                exp_opts = {e["id"]: e["name"] for e in exps}
                sel_exps = st.multiselect("Выберите эксперименты", list(exp_opts.keys()), format_func=lambda x: exp_opts[x])

                y_axis = st.selectbox("Y Axis", ["lcc_frac", "eff_w", "mod", "l2_lcc"], index=0)
                if sel_exps:
                    curves = []
                    x_candidates = []
                    for eid in sel_exps:
                        e = next(x for x in exps if x["id"] == eid)
                        df_hist = _forward_fill_heavy(e["history"])
                        curves.append((e["name"], df_hist))
                        if "mix_frac" in df_hist.columns:
                            x_candidates.append("mix_frac")
                        else:
                            x_candidates.append("removed_frac")

                    x_col = "mix_frac" if x_candidates and all(x == "mix_frac" for x in x_candidates) else "removed_frac"

                    fig_lines = fig_compare_attacks(
                        curves,
                        x_col,
                        y_axis,
                        f"Comparison: {y_axis}",
                        normalize_mode=st.session_state["norm_mode"],
                        height=st.session_state["plot_height"],
                    )
                    fig_lines.update_layout(template="plotly_dark")
                    all_y = pd.concat([pd.to_numeric(df[y_axis], errors="coerce") for _, df in curves if y_axis in df.columns], ignore_index=True)
                    fig_lines = _apply_plot_defaults(fig_lines, height=st.session_state["plot_height"], y_range=_auto_y_range(all_y))
                    st.plotly_chart(fig_lines, use_container_width=True, key="plot_compare_lines")

                    st.markdown("#### Robustness (AUC)")
                    auc_rows = []
                    for name, df in curves:
                        if y_axis in df.columns and x_col in df.columns:
                            xs = pd.to_numeric(df[x_col], errors="coerce")
                            ys = pd.to_numeric(df[y_axis], errors="coerce")
                            mask = xs.notna() & ys.notna()
                            if mask.sum() >= 2:
                                auc = float(AUC_TRAP(ys[mask].to_numpy(), xs[mask].to_numpy()))
                                auc_rows.append({"Experiment": name, "AUC": auc})

                    if auc_rows:
                        st.dataframe(pd.DataFrame(auc_rows).sort_values("AUC", ascending=False), use_container_width=True)
                else:
                    st.info("Выбери эксперименты.")

        # ============================================================
        # 11) FOOTER
        # ============================================================
st.markdown("---")
st.caption("Kodik Lab | Streamlit + NetworkX | node/edge attacks + weak percolation")
