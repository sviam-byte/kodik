# ===== app.py =====

import time
import uuid
import hashlib
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import streamlit as st

# --- Imports from SRC ---
from src.io_load import load_uploaded_any
from src.preprocess import coerce_fixed_format, filter_edges
from src.graph_build import build_graph_from_edges, lcc_subgraph
from src.metrics import calculate_metrics, compute_3d_layout, make_3d_traces
from src.null_models import make_er_gnm, make_configuration_model, rewire_mix
from src.attacks import run_attack
from src.plotting import fig_metrics_over_steps, fig_compare_attacks
from src.phase import classify_phase_transition
from src.session_io import (
    export_workspace_json,
    import_workspace_json,
)

# -----------------------------------------------------------------------------
# 1. Config & CSS
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Kanonar Lab", layout="wide", page_icon="🧬")

st.markdown("""
<style>
    /* Modern Dashboard Look */
    .block-container { padding-top: 2rem; }
    
    /* Metrics styling */
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    
    /* Sidebar tightness */
    section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
    
    /* Highlight the active graph */
    .active-graph-box {
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 10px;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.05);
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. State & Helpers
# -----------------------------------------------------------------------------
def _init_state():
    defaults = {
        "graphs": {},          # gid -> {id, name, source, edges, created_at}
        "experiments": [],     # list of dicts
        "active_graph_id": None,
        "last_uploaded_fingerprint": None,
        "seed_top": 42
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:6]}"

def get_active_graph():
    gid = st.session_state["active_graph_id"]
    return st.session_state["graphs"].get(gid)

def add_graph(name, df, source, tags=None):
    gid = new_id("G")
    st.session_state["graphs"][gid] = {
        "id": gid, "name": name, "source": source,
        "edges": df.copy(), "created_at": time.time(),
        "tags": tags or {}
    }
    st.session_state["active_graph_id"] = gid
    return gid

# Helper for Phase Space Plotting
def plot_phase_space(df_hist, title="Фазовый портрет (Q vs λ₂)"):
    fig = go.Figure()
    
    # Check data availability
    if "mod" not in df_hist.columns or "l2_lcc" not in df_hist.columns:
        return fig
        
    # Trajectory
    fig.add_trace(go.Scatter(
        x=df_hist["mod"],
        y=df_hist["l2_lcc"],
        mode="lines+markers",
        marker=dict(
            size=6,
            color=df_hist["removed_frac"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="% Removed")
        ),
        text=[f"Step: {s}<br>Nodes: {n}" for s, n in zip(df_hist["step"], df_hist["nodes_left"])],
        name="Траектория распада"
    ))

    # Annotations
    fig.update_layout(
        title=title,
        template="plotly_dark",
        xaxis_title="Modularity (Q) — Структура",
        yaxis_title="Algebraic Connectivity (λ₂) — Связность",
        height=500
    )
    # Add quadrants logic visually (optional)
    fig.add_annotation(x=0.1, y=0.01, text="Хаос / Деменция", showarrow=False, font=dict(color="gray"))
    fig.add_annotation(x=0.8, y=0.01, text="Диссоциация", showarrow=False, font=dict(color="gray"))
    
    return fig

# -----------------------------------------------------------------------------
# 3. Sidebar (Controller)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🎛️ Kanonar")

    # --- Graph Selector ---
    gids = sorted(st.session_state["graphs"].keys(), key=lambda k: st.session_state["graphs"][k]["created_at"])
    
    if not gids:
        st.warning("Нет графов. Загрузи или создай.")
    else:
        graph_names = [st.session_state["graphs"][g]["name"] for g in gids]
        curr_id = st.session_state["active_graph_id"]
        try:
            curr_idx = gids.index(curr_id) if curr_id in gids else 0
        except: curr_idx = 0
            
        selected_name = st.selectbox(
            "Активный граф", 
            options=graph_names, 
            index=curr_idx,
            key="sb_graph_select"
        )
        # Update active ID based on name (simple mapping)
        sel_gid = [g for g in gids if st.session_state["graphs"][g]["name"] == selected_name][0]
        if sel_gid != st.session_state["active_graph_id"]:
            st.session_state["active_graph_id"] = sel_gid
            st.rerun()

        # Rename / Delete controls
        c1, c2 = st.columns(2)
        if c1.button("🗑️ Удалить"):
            del st.session_state["graphs"][sel_gid]
            st.session_state["active_graph_id"] = None
            st.rerun()
            
    st.write("---")
    
    # --- Add New Graph ---
    with st.expander("➕ Добавить граф", expanded=not gids):
        tab_up, tab_gen = st.tabs(["Upload", "Generate"])
        
        with tab_up:
            up_file = st.file_uploader("CSV / XLSX", type=["csv", "xlsx"])
            if up_file:
                fp = hashlib.md5(up_file.getvalue()).hexdigest()
                if fp != st.session_state["last_uploaded_fingerprint"]:
                    try:
                        df_any = load_uploaded_any(up_file.getvalue(), up_file.name)
                        df_edges, meta = coerce_fixed_format(df_any)
                        add_graph(f"Up: {up_file.name}", df_edges, "upload", meta)
                        st.session_state["last_uploaded_fingerprint"] = fp
                        st.toast("Граф загружен!")
                        time.sleep(0.5)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Ошибка: {e}")

        with tab_gen:
            gen_type = st.selectbox("Тип", ["ER (Random)", "CFG (Scale-free-ish)"])
            n_node = st.number_input("N", 10, 5000, 200)
            m_edge = st.number_input("M", 10, 20000, 600)
            seed_gen = st.number_input("Seed", 42)
            
            if st.button("Создать"):
                if gen_type.startswith("ER"):
                    G_new = make_er_gnm(n_node, m_edge, seed_gen)
                    src_tag = "null:ER"
                else:
                    # Dummy base for CFG if no active graph, else use active degree seq
                    if get_active_graph():
                        df_b = get_active_graph()["edges"]
                        meta = get_active_graph()["tags"]
                        G_base = build_graph_from_edges(df_b, meta.get("src_col", df_b.columns[0]), meta.get("dst_col", df_b.columns[1]))
                        G_new = make_configuration_model(G_base, seed_gen)
                        src_tag = f"null:CFG({get_active_graph()['name']})"
                    else:
                        G_new = make_er_gnm(n_node, m_edge, seed_gen) # fallback
                        src_tag = "null:CFG(fallback)"

                # Convert nx to df
                edges = [[u, v, 1.0, 1.0] for u, v in G_new.edges()]
                df_new = pd.DataFrame(edges, columns=["src", "dst", "weight", "confidence"])
                add_graph(f"{gen_type} N={len(G_new)}", df_new, src_tag, {"src_col":"src", "dst_col":"dst"})
                st.rerun()

    st.write("---")
    
    # --- Filters ---
    st.caption("Фильтры данных")
    f_conf = st.slider("Min Confidence", 0, 100, 0)
    f_weight = st.slider("Min Weight", 0.0, 50.0, 0.0, 0.5)

# -----------------------------------------------------------------------------
# 4. Main Workspace
# -----------------------------------------------------------------------------
entry = get_active_graph()

if not entry:
    st.info("👈 Выберите или создайте граф в меню слева.")
    st.stop()

# Build Graph Object
df_raw = entry["edges"]
tags = entry["tags"]
df_filt = filter_edges(df_raw, tags.get("src_col", df_raw.columns[0]), tags.get("dst_col", df_raw.columns[1]), f_conf, f_weight)
G_full = build_graph_from_edges(df_filt, tags.get("src_col", df_raw.columns[0]), tags.get("dst_col", df_raw.columns[1]))
G = lcc_subgraph(G_full) # Always work with LCC for physics stability

# --- Header ---
c_title, c_kpi1, c_kpi2, c_kpi3 = st.columns([2, 1, 1, 1])
with c_title:
    st.subheader(f"{entry['name']}")
    st.caption(f"{entry['source']} | LCC Nodes: {len(G)} | Edges: {G.number_of_edges()}")

# Pre-calc base metrics (lightweight)
if "base_metrics" not in st.session_state or st.session_state.get("last_gid") != entry["id"]:
    # Recalc only on graph switch
    with st.spinner("Считаю топологию..."):
        m = calculate_metrics(G, eff_sources_k=32, seed=42)
        st.session_state["base_metrics"] = m
        st.session_state["last_gid"] = entry["id"]

metrics = st.session_state["base_metrics"]

with c_kpi1:
    st.metric("λ₂ (Связность)", f"{metrics['l2_lcc']:.4f}")
with c_kpi2:
    st.metric("Modularity (Q)", f"{metrics['mod']:.3f}")
with c_kpi3:
    st.metric("Efficiency", f"{metrics['eff_w']:.3f}")

# -----------------------------------------------------------------------------
# 5. Tabs Logic
# -----------------------------------------------------------------------------
tab_viz, tab_sim, tab_cmp, tab_mix = st.tabs([
    "👁️ Анатомия", 
    "🔥 Краш-тест (Simulation)", 
    "⚔️ Head-to-Head",
    "🧪 Лаборатория (Mixing)"
])

# --- TAB 1: VISUALIZATION ---
with tab_viz:
    col_3d, col_info = st.columns([3, 1])
    
    with col_3d:
        if len(G) > 2000:
            st.warning("Граф большой, 3D может тормозить.")
            if st.button("Показать 3D всё равно"):
                pos = compute_3d_layout(G, 42)
                e_tr, n_tr = make_3d_traces(G, pos, True)
                fig = go.Figure([e_tr, n_tr])
                fig.update_layout(template="plotly_dark", margin=dict(t=0,b=0,l=0,r=0), height=500)
                st.plotly_chart(fig, use_container_width=True)
        else:
            pos = compute_3d_layout(G, 42)
            e_tr, n_tr = make_3d_traces(G, pos, True)
            fig = go.Figure([e_tr, n_tr])
            fig.update_layout(template="plotly_dark", margin=dict(t=0,b=0,l=0,r=0), height=500)
            st.plotly_chart(fig, use_container_width=True)
            
    with col_info:
        st.markdown("### Свойства")
        st.write(f"**Assortativity:** {metrics['assortativity']:.3f}")
        st.write(f"**Clustering:** {metrics['clustering']:.3f}")
        st.write(f"**Diameter (~):** {metrics['diameter_approx']}")
        st.write(f"**Density:** {metrics['density']:.5f}")
        
        st.info("""
        **λ₂ (Fiedler Value)**: Скорость диффузии. 
        Если низкая — граф легко распадается на куски.
        
        **Modularity (Q)**: Наличие кланов.
        Если высокая — граф состоит из плотных групп.
        """)

# --- TAB 2: CRASH TEST ---
with tab_sim:
    c_set, c_res = st.columns([1, 2])
    
    with c_set:
        st.markdown("### Сценарий Атаки")
        
        mode = st.radio("Режим", ["Быстрый", "Кастомный"], horizontal=True)
        
        if mode == "Быстрый":
            attack_type = st.selectbox("Стратегия", ["random", "degree", "betweenness", "richclub_top"])
            steps = 20
            frac = 0.5
            eff_k = 16
        else:
            attack_type = st.selectbox("Стратегия", ["random", "degree", "betweenness", "kcore", "richclub_top", "richclub_density"])
            frac = st.slider("% Уничтожения", 0.1, 0.9, 0.5)
            steps = st.slider("Шаги", 5, 100, 30)
            eff_k = st.slider("Точность Efficiency", 8, 128, 32)
        
        btn_run = st.button("💀 ЗАПУСТИТЬ", type="primary", use_container_width=True)
        
    with c_res:
        if btn_run:
            with st.spinner("Ломаем систему..."):
                # Run Logic
                df_hist, _ = run_attack(
                    G, attack_kind=attack_type, 
                    remove_frac=frac, steps=steps, 
                    seed=42, eff_sources_k=eff_k,
                    # defaults for RC
                    rc_frac=0.1, rc_min_density=0.3, rc_max_frac=0.3,
                    compute_heavy_every=1
                )
                
                # Analyze Phase
                phase = classify_phase_transition(df_hist)
                
                # Save to session for persistency
                st.session_state["last_run"] = {
                    "df": df_hist,
                    "phase": phase,
                    "name": f"{attack_type} on {entry['name']}"
                }
        
        # Display Results
        if "last_run" in st.session_state:
            res = st.session_state["last_run"]
            df_h = res["df"]
            ph = res["phase"]
            
            st.success(f"Результат: {res['name']}")
            
            # Phase Transition Metrics
            c_p1, c_p2 = st.columns(2)
            c_p1.metric("Тип перехода", "Взрывной (1-го рода)" if ph['is_abrupt'] else "Плавный (2-го рода)")
            c_p2.metric("Критическая точка", f"{ph['critical_x']:.2%} removed")

            # Visuals
            t_dyn, t_phase = st.tabs(["📉 Динамика", "🌀 Фазовый Портрет"])
            
            with t_dyn:
                fig = fig_metrics_over_steps(df_h, title="Деградация метрик")
                st.plotly_chart(fig, use_container_width=True)
            
            with t_phase:
                fig_p = plot_phase_space(df_h)
                st.plotly_chart(fig_p, use_container_width=True)
                st.caption("Траектория смещается из зоны 'Интеграции' (верхний правый) в зону 'Распада' (нижний левый). Форма кривой показывает тип патологии.")

# --- TAB 3: HEAD TO HEAD ---
with tab_sim: # Wait, logic better in separate tab
    pass 

with tab_cmp:
    st.markdown("### Сравнение Стратегий")
    st.caption("Запускает 4 базовых сценария атаки параллельно и сравнивает устойчивость.")
    
    if st.button("⚔️ FIGHT: Random vs Hubs vs Bridges"):
        with st.spinner("Прогон сценариев..."):
            scenarios = [
                ("random", "Random Failure"),
                ("degree", "Hubs Attack (Targeted)"),
                ("betweenness", "Bridges Cut (Connector)"),
                ("richclub_top", "Rich-Club Decapitation")
            ]
            
            results = []
            for kind, label in scenarios:
                df, _ = run_attack(
                    G, attack_kind=kind, remove_frac=0.5, steps=25,
                    seed=42, eff_sources_k=16, compute_heavy_every=2
                )
                results.append((label, df))
            
            st.session_state["cmp_results"] = results
            
    if "cmp_results" in st.session_state:
        res_list = st.session_state["cmp_results"]
        
        # Plot LCC
        fig_lcc = fig_compare_attacks(res_list, "removed_frac", "lcc_frac", "Живучесть сети (LCC)")
        st.plotly_chart(fig_lcc, use_container_width=True)
        
        # Plot Efficiency
        fig_eff = fig_compare_attacks(res_list, "removed_frac", "eff_w", "Падение эффективности")
        st.plotly_chart(fig_eff, use_container_width=True)

# --- TAB 4: MIXING LAB ---
with tab_mix:
    st.markdown("### 🧪 Смешивание (Rewiring)")
    st.caption("Создать гибрид между текущим графом и рандомом, сохраняя степени узлов.")
    
    c_p, c_btn = st.columns([3, 1])
    with c_p:
        p_val = st.slider("Коэффициент хаоса (p)", 0.0, 1.0, 0.1, 0.05, 
                          help="0 = Оригинал, 1 = Полный рандом")
    
    with c_btn:
        st.write("") # spacing
        if st.button("Создать мутанта"):
            with st.spinner("Rewiring..."):
                G_mut = rewire_mix(G, p=p_val, seed=int(time.time()))
                
                # To DF
                edges = [[u, v, 1.0, 1.0] for u, v in G_mut.edges()]
                df_mut = pd.DataFrame(edges, columns=["src", "dst", "weight", "confidence"])
                
                new_name = f"Mix(p={p_val}) of {entry['name']}"
                add_graph(new_name, df_mut, "mix", tags=tags)
                st.success(f"Создан граф: {new_name}")
                time.sleep(1)
                st.rerun()

    st.info("💡 Используй это, чтобы проверить: насколько структура твоего графа (мозга/города) уникальна по сравнению со случайной сетью с такими же хабами?")
