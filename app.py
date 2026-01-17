# app.py

import time
import uuid
import hashlib
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import streamlit as st

# Импорты (предполагаем структуру файлов из дампа)
from src.io_load import load_uploaded_any
from src.preprocess import coerce_fixed_format, filter_edges
from src.graph_build import build_graph_from_edges, lcc_subgraph, graph_summary
from src.metrics import calculate_metrics, compute_3d_layout, make_3d_traces
from src.null_models import make_er_gnm, make_configuration_model, rewire_mix
from src.attacks import run_attack
from src.plotting import (
    fig_metrics_over_steps,
    fig_compare_attacks,
    fig_compare_graphs_scalar,
)
from src.phase import classify_phase_transition
from src.session_io import (
    export_workspace_json,
    import_workspace_json,
    export_experiments_json,
    import_experiments_json,
)

# -------------------------
# Page Config & Styles
# -------------------------
st.set_page_config(page_title="Network Collapse Lab", layout="wide", page_icon="💀")

st.markdown("""
<style>
    .sticky-topbar {
        position: sticky;
        top: 0;
        z-index: 1000;
        background-color: #0E1117;
        padding: 1rem 0;
        border-bottom: 1px solid #333;
        margin-bottom: 1rem;
    }
    div[data-testid="metric-container"] {
        background-color: #262730;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #444;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Session State Init
# -------------------------
if "graphs" not in st.session_state:
    st.session_state["graphs"] = {}
if "experiments" not in st.session_state:
    st.session_state["experiments"] = []
if "active_graph_id" not in st.session_state:
    st.session_state["active_graph_id"] = None
if "seed_top" not in st.session_state:
    st.session_state["seed_top"] = 42

# -------------------------
# Helper Functions
# -------------------------
def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def get_active_graph():
    gid = st.session_state.get("active_graph_id")
    return st.session_state["graphs"].get(gid) if gid else None

def add_graph(name, df, source, tags=None):
    gid = new_id("G")
    st.session_state["graphs"][gid] = {
        "id": gid, "name": name, "source": source, 
        "tags": tags or {}, "edges": df.copy(), "created_at": time.time()
    }
    st.session_state["active_graph_id"] = gid
    return gid

def run_attack_wrapper(G, kind, params, name_tag):
    """Wrapper to run attack, classify phase, and save experiment."""
    # 1. Run Attack
    # ВАЖНО: параметры передаются точно как в src/attacks.py
    df_hist, states = run_attack(
        G_in=G, 
        attack_kind=kind,
        remove_frac=params['remove_frac'], 
        steps=params['steps'],
        seed=params['seed'], 
        eff_sources_k=params['eff_k'],
        rc_frac=params.get('rc_frac', 0.1),
        rc_min_density=params.get('rc_min_density', 0.3),
        rc_max_frac=params.get('rc_max_frac', 0.3),
        compute_heavy_every=params['heavy_every'],
        keep_states=params['keep_states']
    )
    
    # 2. Classify Phase Transition
    phase = classify_phase_transition(df_hist)
    
    # 3. Save
    exp_id = new_id("EXP")
    st.session_state["experiments"].append({
        "id": exp_id,
        "name": f"{kind} {name_tag}",
        "graph_id": st.session_state["active_graph_id"],
        "attack_kind": kind,
        "params": params,
        "history": df_hist,
        "phase": phase,
        "created_at": time.time()
    })
    return exp_id, df_hist, phase

# -------------------------
# TOPBAR
# -------------------------
def render_topbar():
    with st.container():
        st.markdown('<div class="sticky-topbar">', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([2, 1.5, 1.5, 1])
        
        gids = sorted(st.session_state["graphs"].keys(), key=lambda k: st.session_state["graphs"][k]['created_at'])
        options = ["-- Select --"] + gids
        
        with c1:
            active_id = st.session_state.get("active_graph_id")
            idx = gids.index(active_id) + 1 if active_id in gids else 0
            selected = st.selectbox(
                "📂 Active Graph", 
                options, 
                index=idx, 
                format_func=lambda x: st.session_state["graphs"][x]['name'] if x in st.session_state["graphs"] else x,
                key="top_graph_select",
                label_visibility="collapsed"
            )
            if selected != "-- Select --" and selected != active_id:
                st.session_state["active_graph_id"] = selected
                st.rerun()

        with c2:
            if st.button("🎲 New Random (ER)", use_container_width=True):
                G_er = make_er_gnm(200, 600, seed=42)
                edges = [[u, v, 1.0, 1.0] for u, v in G_er.edges()]
                df_new = pd.DataFrame(edges, columns=["src", "dst", "weight", "confidence"])
                add_graph("Random ER (N=200, M=600)", df_new, "generated")
                st.rerun()
                
        with c3:
            if active_id:
                if st.button("🔀 Mix Active", use_container_width=True):
                    entry = st.session_state["graphs"][active_id]
                    df_base = entry["edges"]
                    # Quick reconstruct for rewire
                    src_col = entry["tags"].get("src_col", df_base.columns[0])
                    dst_col = entry["tags"].get("dst_col", df_base.columns[1])
                    G_base = build_graph_from_edges(df_base, src_col, dst_col)
                    
                    G_mix = rewire_mix(G_base, p=0.2, seed=42)
                    edges = [[u, v, 1.0, 1.0] for u, v in G_mix.edges()]
                    df_mix = pd.DataFrame(edges, columns=[src_col, dst_col, "weight", "confidence"])
                    add_graph(f"Mix (p=0.2) of {entry['name']}", df_mix, "mix")
                    st.rerun()

        with c4:
             if st.button("🗑️ Clear All", type="primary"):
                 st.session_state["graphs"] = {}
                 st.session_state["experiments"] = []
                 st.session_state["active_graph_id"] = None
                 st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

render_topbar()

# -------------------------
# SIDEBAR
# -------------------------
with st.sidebar:
    st.header("🛠️ Data & Filters")
    uploaded = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx"])
    if uploaded:
        try:
            df_any = load_uploaded_any(uploaded.getvalue(), uploaded.name)
            df_edges, meta = coerce_fixed_format(df_any)
            add_graph(f"Upload: {uploaded.name}", df_edges, "upload", meta)
            st.success(f"Loaded {len(df_edges)} edges.")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

    st.write("---")
    min_conf = st.number_input("Min Confidence", 0, 100, 0)
    min_weight = st.number_input("Min Weight", 0.0, 100.0, 0.0, 0.1)
    st.write("---")
    eff_k = st.slider("Efficiency k", 10, 200, 50)
    seed = st.number_input("Global Seed", value=42)

# -------------------------
# MAIN
# -------------------------
graph_entry = get_active_graph()
if not graph_entry:
    st.info("👈 Please upload or generate a graph.")
    st.stop()

# Build Graph
df = graph_entry["edges"]
src = graph_entry["tags"].get("src_col", df.columns[0])
dst = graph_entry["tags"].get("dst_col", df.columns[1])
df_filtered = filter_edges(df, src, dst, min_conf, min_weight)
G_full = build_graph_from_edges(df_filtered, src, dst)

use_lcc = st.checkbox("Analyze LCC Only", value=True)
G_view = lcc_subgraph(G_full) if use_lcc else G_full

# Metrics
metrics = calculate_metrics(G_view, eff_sources_k=eff_k, seed=seed)

st.write(f"### 📊 {graph_entry['name']}")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Nodes", metrics["N"])
m2.metric("Edges", metrics["E"])
m3.metric("λ₂ (Robustness)", f"{metrics['l2_lcc']:.4f}")
m4.metric("Modularity (Q)", f"{metrics['mod']:.4f}")
m5.metric("Efficiency", f"{metrics['eff_w']:.4f}")

# Tabs
tab_viz, tab_attack, tab_compare = st.tabs(["👁️ Vis", "💥 Attack Lab", "🆚 Compare"])

with tab_viz:
    if st.button("Generate 3D View"):
        pos = compute_3d_layout(G_view, seed=seed)
        e_tr, n_tr = make_3d_traces(G_view, pos, show_scale=True)
        fig = go.Figure(data=[e_tr, n_tr])
        fig.update_layout(template="plotly_dark", height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

with tab_attack:
    c_set, c_act = st.columns([1, 2])
    with c_set:
        st.subheader("Settings")
        attack_type = st.selectbox("Strategy", ["random", "degree", "betweenness", "kcore", "richclub_top"])
        frac = st.slider("Remove %", 0.1, 0.9, 0.5, 0.05)
        steps = st.slider("Steps", 5, 50, 20)
        compare_random = st.checkbox("Compare with Null Model", value=True)
        
    with c_act:
        st.subheader("Simulation")
        if st.button("START ATTACK", type="primary", use_container_width=True):
            with st.spinner("Simulating..."):
                # Real Attack
                # FIX: Используем name_tag, а не tag
                exp_id, df_real, phase_real = run_attack_wrapper(
                    G_view, attack_type, 
                    {"remove_frac": frac, "steps": steps, "seed": seed, "eff_k": eff_k, "heavy_every": 1, "keep_states": False},
                    name_tag="Main" 
                )
                
                # Null Model Attack
                df_null = None
                if compare_random:
                    G_rand = make_configuration_model(G_view, seed=seed)
                    _, df_null, _ = run_attack_wrapper(
                        G_rand, attack_type,
                        {"remove_frac": frac, "steps": steps, "seed": seed, "eff_k": eff_k, "heavy_every": 1, "keep_states": False},
                        name_tag="Null Model"
                    )

                # Plots
                st.success("Done.")
                
                # LCC Decay
                fig_lcc = go.Figure()
                fig_lcc.add_trace(go.Scatter(x=df_real['removed_frac'], y=df_real['lcc_frac'], name="Original", line=dict(width=3)))
                if df_null is not None:
                    fig_lcc.add_trace(go.Scatter(x=df_null['removed_frac'], y=df_null['lcc_frac'], name="Null (Randomized)", line=dict(dash='dash')))
                fig_lcc.update_layout(title="LCC Collapse", template="plotly_dark", xaxis_title="Removed %", yaxis_title="LCC Size")
                st.plotly_chart(fig_lcc, use_container_width=True)
                
                # Phase Space
                if 'mod' in df_real.columns and 'l2_lcc' in df_real.columns:
                    fig_phase = go.Figure()
                    fig_phase.add_trace(go.Scatter(
                        x=df_real['mod'], y=df_real['l2_lcc'],
                        mode='markers+lines',
                        marker=dict(color=df_real['removed_frac'], colorscale='Viridis', showscale=True, size=8),
                        name="Trajectory"
                    ))
                    fig_phase.update_layout(title="Phase Space (Q vs λ₂)", xaxis_title="Modularity", yaxis_title="Connectivity", template="plotly_dark")
                    st.plotly_chart(fig_phase, use_container_width=True)
                
                st.info(f"Phase Transition: {'Explosive' if phase_real['is_abrupt'] else 'Continuous'}")

with tab_compare:
    st.write("### Experiments")
    if not st.session_state["experiments"]:
        st.warning("No experiments yet.")
    else:
        exps = st.session_state["experiments"]
        names = [f"{i}: {e['name']}" for i, e in enumerate(exps)]
        sel = st.multiselect("Select", range(len(names)), default=list(range(len(names))))
