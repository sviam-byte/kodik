# app.py

import time
import uuid
import hashlib
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import streamlit as st

# Модули (предполагаем, что они есть в src/)
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
st.set_page_config(page_title="Network Collapse Lab", layout="wide", page_icon="🕸️")

st.markdown("""
<style>
    /* Sticky Topbar */
    .sticky-topbar {
        position: sticky;
        top: 0;
        z-index: 1000;
        background-color: #0E1117; /* Streamlit dark bg */
        padding: 1rem 0;
        border-bottom: 1px solid #333;
        margin-bottom: 1rem;
    }
    
    /* Metrics Styling */
    div[data-testid="metric-container"] {
        background-color: #262730;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #444;
    }
    
    /* Help Tooltips */
    .help-icon {
        font-size: 0.8em;
        color: #888;
        margin-left: 5px;
        cursor: help;
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
    df_hist, states = run_attack(
        G, attack_kind=kind,
        remove_frac=params['remove_frac'], steps=params['steps'],
        seed=params['seed'], eff_sources_k=params['eff_k'],
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
# TOPBAR (Graph Selection & Quick Gen)
# -------------------------
def render_topbar():
    with st.container():
        st.markdown('<div class="sticky-topbar">', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([2, 1.5, 1.5, 1])
        
        # 1. Graph Selector
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

        # 2. Generator Quick Actions
        with c2:
            if st.button("🎲 New Random (ER)", use_container_width=True, help="Create Erdős-Rényi graph"):
                G_er = make_er_gnm(200, 600, seed=42) # Default params
                edges = [[u, v, 1.0, 1.0] for u, v in G_er.edges()]
                df_new = pd.DataFrame(edges, columns=["src", "dst", "weight", "confidence"])
                add_graph("Random ER (N=200, M=600)", df_new, "generated")
                st.rerun()
                
        with c3:
            if active_id:
                if st.button("🔀 Mix Active (Rewire)", use_container_width=True, help="Rewire edges of active graph (Watts-Strogatz)"):
                    # Get active graph first (simplified logic for UI brevity)
                    entry = st.session_state["graphs"][active_id]
                    df_base = entry["edges"]
                    G_base = build_graph_from_edges(df_base, df_base.columns[0], df_base.columns[1])
                    G_mix = rewire_mix(G_base, p=0.2, seed=42)
                    edges = [[u, v, 1.0, 1.0] for u, v in G_mix.edges()]
                    df_mix = pd.DataFrame(edges, columns=["src", "dst", "weight", "confidence"])
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
# SIDEBAR (Data Loading & Filters)
# -------------------------
with st.sidebar:
    st.header("🛠️ Data & Filters")
    
    # Upload
    uploaded = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx"])
    if uploaded:
        # (Checksum logic omitted for brevity, assuming simple load)
        try:
            df_any = load_uploaded_any(uploaded.getvalue(), uploaded.name)
            df_edges, meta = coerce_fixed_format(df_any)
            gid = add_graph(f"Upload: {uploaded.name}", df_edges, "upload", meta)
            st.success(f"Loaded {len(df_edges)} edges.")
        except Exception as e:
            st.error(f"Error: {e}")

    st.write("---")
    
    # Filters for Active Graph
    st.subheader("Filters")
    min_conf = st.number_input("Min Confidence", 0, 100, 0, help="Filter edges by confidence score column")
    min_weight = st.number_input("Min Weight", 0.0, 100.0, 0.0, 0.1, help="Filter edges by weight column")
    
    st.write("---")
    st.subheader("Analysis Params")
    eff_k = st.slider("Efficiency Sources (k)", 10, 200, 50, help="Approximation speed vs accuracy for Global Efficiency")
    seed = st.number_input("Global Seed", value=42)

# -------------------------
# MAIN CONTENT
# -------------------------

# 1. Get Active Graph & Apply Filters
graph_entry = get_active_graph()
if not graph_entry:
    st.info("👈 Please upload a file or generate a random graph to start.")
    st.stop()

# Build Graph object
df = graph_entry["edges"]
src = graph_entry["tags"].get("src_col", df.columns[0])
dst = graph_entry["tags"].get("dst_col", df.columns[1])
df_filtered = filter_edges(df, src, dst, min_conf, min_weight)
G_full = build_graph_from_edges(df_filtered, src, dst)

# LCC Toggle
use_lcc = st.checkbox("Analyze Largest Component Only (LCC)", value=True, help="Focus on the giant component, ignore isolated nodes.")
G_view = lcc_subgraph(G_full) if use_lcc else G_full

# 2. Quick Metrics (Top Dashboard)
metrics = calculate_metrics(G_view, eff_sources_k=eff_k, seed=seed)

st.write(f"### 📊 Analysis: {graph_entry['name']}")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Nodes", metrics["N"], help="Total nodes in view")
m2.metric("Edges", metrics["E"], help="Total edges in view")
m3.metric("λ₂ (Connect)", f"{metrics['l2_lcc']:.4f}", help="Algebraic Connectivity. 0 = disconnected. Higher = harder to cut.")
m4.metric("Modularity (Q)", f"{metrics['mod']:.4f}", help="Community structure strength (Louvain). >0.3 is strong.")
m5.metric("Efficiency", f"{metrics['eff_w']:.4f}", help="Global information flow efficiency.")

# 3. Main Tabs
tab_viz, tab_attack, tab_compare = st.tabs(["👁️ Visualization", "💥 Attack Laboratory", "🆚 Comparison"])

# --- TAB 1: 3D VIZ ---
with tab_viz:
    st.write("**3D Force Layout**")
    if metrics["N"] > 2000:
        st.warning("⚠️ Graph is large (>2k nodes). 3D rendering might be slow.")
    
    if st.button("Generate 3D View"):
        pos = compute_3d_layout(G_view, seed=seed)
        edge_trace, node_trace = make_3d_traces(G_view, pos, show_scale=True)
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(template="plotly_dark", height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: ATTACK LAB ---
with tab_attack:
    c_settings, c_action = st.columns([1, 2])
    
    with c_settings:
        st.subheader("⚙️ Simulation Settings")
        
        attack_type = st.selectbox(
            "Attack Strategy", 
            ["random", "degree", "betweenness", "kcore", "richclub_top"],
            help="Criterion for removing nodes."
        )
        
        frac = st.slider("Remove %", 0.1, 0.9, 0.5, 0.05)
        steps = st.slider("Steps", 5, 50, 20)
        
        # Mix/Randomize on fly
        st.markdown("---")
        st.write("**On-the-fly comparisons:**")
        compare_random = st.checkbox("Compare with Null Model (Random)", value=True, help="Run same attack on a randomized version of this graph")
        
    with c_action:
        st.subheader("🚀 Run Simulation")
        if st.button("START ATTACK", type="primary", use_container_width=True):
            with st.spinner("Simulating collapse..."):
                # 1. Attack Real Graph
                exp_id, df_real, phase_real = run_attack_wrapper(
                    G_view, attack_type, 
                    {"remove_frac": frac, "steps": steps, "seed": seed, "eff_k": eff_k, "heavy_every": 1, "keep_states": False},
                    tag="Main"
                )
                
                # 2. Attack Null Model (if checked)
                df_null = None
                if compare_random:
                    G_rand = make_configuration_model(G_view, seed=seed) # Configuration model preserves degrees
                    _, df_null, _ = run_attack_wrapper(
                        G_rand, attack_type,
                        {"remove_frac": frac, "steps": steps, "seed": seed, "eff_k": eff_k, "heavy_every": 1, "keep_states": False},
                        tag="Null Model"
                    )

                # 3. Plotting
                st.success("Simulation Complete.")
                
                # Plot LCC Collapse
                fig_lcc = go.Figure()
                fig_lcc.add_trace(go.Scatter(x=df_real['removed_frac'], y=df_real['lcc_frac'], name="Original", line=dict(width=3)))
                if df_null is not None:
                    fig_lcc.add_trace(go.Scatter(x=df_null['removed_frac'], y=df_null['lcc_frac'], name="Null Model (Randomized)", line=dict(dash='dash')))
                
                fig_lcc.update_layout(title="LCC Fraction Decay", template="plotly_dark", xaxis_title="Fraction Removed", yaxis_title="LCC Size")
                st.plotly_chart(fig_lcc, use_container_width=True)
                
                # Plot Phase Space (Q vs Lambda2)
                # Check if metrics exist (heavy computation might skip them)
                if 'mod' in df_real.columns and 'l2_lcc' in df_real.columns:
                    fig_phase = go.Figure()
                    fig_phase.add_trace(go.Scatter(
                        x=df_real['mod'], y=df_real['l2_lcc'],
                        mode='markers+lines',
                        marker=dict(color=df_real['removed_frac'], colorscale='Viridis', showscale=True, size=8),
                        name="Trajectory"
                    ))
                    fig_phase.update_layout(
                        title="Phase Space Trajectory (Modularity vs Connectivity)", 
                        xaxis_title="Modularity (Q)", 
                        yaxis_title="Algebraic Connectivity (λ₂)",
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig_phase, use_container_width=True)
                
                # Phase Transition Stats
                c1, c2 = st.columns(2)
                c1.info(f"Phase Transition (Original): {'Abrupt (Explosive)' if phase_real['is_abrupt'] else 'Continuous'}")
                c1.metric("Critical Threshold (fc)", f"{phase_real['critical_x']:.3f}")

# --- TAB 3: COMPARISON ---
with tab_compare:
    st.write("### ⚔️ Experiment Comparison")
    
    if len(st.session_state["experiments"]) == 0:
        st.warning("No experiments run yet. Go to 'Attack Laboratory'.")
    else:
        # Multiselect experiments
        exp_names = [f"{i}: {e['name']}" for i, e in enumerate(st.session_state["experiments"])]
        selected_indices = st.multiselect("Select experiments to compare", range(len(exp_names)), default=list(range(len(exp_names))))
        
        if selected_indices:
            metric_to_plot = st.selectbox("Metric to compare", ["lcc_frac", "eff_w", "mod", "l2_lcc"])
            
            fig_cmp = go.Figure()
            for idx in selected_indices:
                exp = st.session_state["experiments"][idx]
                df = exp["history"]
                if metric_to_plot in df.columns:
                    fig_cmp.add_trace(go.Scatter(
                        x=df["removed_frac"], 
                        y=df[metric_to_plot], 
                        name=exp["name"]
                    ))
            
            fig_cmp.update_layout(title=f"Comparison: {metric_to_plot}", template="plotly_dark", xaxis_title="Removed Fraction")
            st.plotly_chart(fig_cmp, use_container_width=True)
