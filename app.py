# app.py
import time
import uuid
import hashlib
import json

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Local imports from your structure
from src.io_load import load_uploaded_any
from src.preprocess import coerce_fixed_format, filter_edges
from src.graph_build import build_graph_from_edges, lcc_subgraph
from src.metrics import calculate_metrics, compute_3d_layout, make_3d_traces
from src.null_models import make_er_gnm, make_configuration_model, rewire_mix
from src.attacks import run_attack
from src.plotting import (
    fig_metrics_over_steps,
    fig_compare_attacks,
)
from src.phase import classify_phase_transition
from src.session_io import (
    export_workspace_json,
    import_workspace_json,
    export_experiments_json,
    import_experiments_json,
)

# ==========================================
# 0. CONFIG & STYLES
# ==========================================
st.set_page_config(
    page_title="Kodik Lab",
    layout="wide",
    page_icon="🕸️",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Sticky header container */
    div[data-testid="stVerticalBlock"] > div:has(> div.sticky-header) {
        position: sticky;
        top: 0;
        z-index: 9999;
        background: #0e1117;
        border-bottom: 1px solid rgba(250, 250, 250, 0.1);
        padding-top: 1rem;
        padding-bottom: 0.5rem;
    }
    .sticky-header { margin-bottom: 0.5rem; }

    /* Metrics nice styling */
    div[data-testid="stMetricValue"] { font-size: 1.4rem !important; }

    /* Compact tabs */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 1. TOOLTIPS / DICTIONARY
# ==========================================
HELP_TEXT = {
    "N": "Количество узлов (Nodes) в графе.",
    "E": "Количество рёбер (Edges) в графе.",
    "Density": "Плотность графа: отношение существующих рёбер к максимально возможному числу рёбер.",
    "LCC": "Largest Connected Component (Гигантская компонента). Самый большой связный кусок графа.",
    "LCC frac": "Доля узлов, находящихся в гигантской компоненте. Если < 1.0, граф развален.",
    "Efficiency (Global)": "Средняя обратная длина кратчайшего пути (аппрокс).",
    "Modularity Q": "Мера выраженности сообществ (кластеров).",
    "Lambda2 (Algebraic Connectivity)": "Второе собственное число Лапласиана: близко к 0 => легко разорвать.",
    "LambdaMax (Spectral Radius)": "Максимальное собственное число матрицы смежности.",
    "Assortativity": "Корреляция степеней соединенных узлов.",
    "Entropy (Degree)": "Энтропия Шеннона распределения степеней.",
    "Clustering": "Средний коэффициент кластеризации.",
    "Mix/Rewire": "p=0 оригинал, p=1 полная рандомизация с сохранением числа рёбер.",
    "Phase Space": "Фазпортреты. Control: removed_frac. Order: lcc_frac / λ2 / efficiency.",
}
def help_icon(key: str) -> str:
    return HELP_TEXT.get(key, "")

# ==========================================
# 1.5 PRESETS (BATCH)
# ==========================================
ATTACK_PRESETS = {
    "Core suite (быстро)": [
        {"kind": "random", "seeds": 3},
        {"kind": "degree", "seeds": 3},
        {"kind": "betweenness", "seeds": 2},
        {"kind": "kcore", "seeds": 2},
    ],
    "Stress suite (жёстко)": [
        {"kind": "degree", "seeds": 5},
        {"kind": "betweenness", "seeds": 5},
        {"kind": "kcore", "seeds": 5},
        {"kind": "richclub_top", "seeds": 5},
    ],
    "Only random (статистика)": [
        {"kind": "random", "seeds": 20},
    ],
}

# ==========================================
# 2. STATE MANAGEMENT
# ==========================================
def _init_state():
    defaults = {
        "graphs": {},        # gid -> dict
        "experiments": [],   # list of dict
        "active_graph_id": None,
        "seed": 42,
        "last_upload_hash": None,
        "layout_seed_bump": 0,
        "last_suite_curves": None,
        "last_multi_curves": None,
        "last_exp_id": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:6]}"

def get_active_graph():
    gid = st.session_state["active_graph_id"]
    if gid and gid in st.session_state["graphs"]:
        return st.session_state["graphs"][gid]
    return None

def set_active_graph(gid: str):
    if gid in st.session_state["graphs"]:
        st.session_state["active_graph_id"] = gid

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
        "created_at": time.time()
    })

def _forward_fill_heavy(df_hist: pd.DataFrame) -> pd.DataFrame:
    df = df_hist.copy()
    for col in ["l2_lcc", "mod"]:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).ffill()
    return df

def run_attack_suite(
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
    curves = []
    for block in preset_spec:
        kind = block["kind"]
        nseeds = int(block.get("seeds", 1))
        for i in range(nseeds):
            # стабильное, но различающееся по kind
            seed_i = int(base_seed) + 1000 * (abs(hash(kind)) % 97) + i
            df_hist, _ = run_attack(
                G, kind, frac, steps, seed_i, eff_k,
                rc_frac=rc_frac, compute_heavy_every=heavy_freq
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
                    "frac": frac,
                    "steps": steps,
                    "seed": seed_i,
                    "phase": phase_info,
                    "preset": True,
                    "compute_heavy_every": heavy_freq,
                    "eff_k": eff_k,
                    "rc_frac": rc_frac,
                },
                df_hist=df_hist
            )
            curves.append((label, df_hist))
    return curves

# ==========================================
# 3. SIDEBAR & DATA LOADING
# ==========================================
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
                    # merge (простая конкатенация)
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
    if st.button("🗑️ Сбросить всё", type="primary"):
        st.session_state["graphs"] = {}
        st.session_state["experiments"] = []
        st.session_state["active_graph_id"] = None
        st.session_state["last_suite_curves"] = None
        st.session_state["last_multi_curves"] = None
        st.session_state["last_exp_id"] = None
        st.rerun()

# ==========================================
# 4. TOP BAR (STICKY)
# ==========================================
def render_top_bar():
    graphs = st.session_state["graphs"]
    active_gid = st.session_state["active_graph_id"]

    st.markdown('<div class="sticky-header">', unsafe_allow_html=True)

    if not graphs:
        st.warning("⚠️ Workspace пуст. Загрузите файл слева или создайте случайный граф.")
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

    col1, col2, col3, col4 = st.columns([3, 2, 1, 1])

    with col1:
        options = list(graphs.keys())
        options.sort(key=lambda k: graphs[k]["created_at"])

        selected = st.selectbox(
            "Активный граф",
            options,
            index=options.index(active_gid) if active_gid in options else 0,
            format_func=lambda x: f"{graphs[x]['name']} ({graphs[x]['source']})",
            label_visibility="collapsed"
        )
        if selected != active_gid:
            st.session_state["active_graph_id"] = selected
            st.rerun()

    current_entry = graphs[selected]

    with col2:
        new_name = st.text_input(
            "Rename",
            value=current_entry["name"],
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
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    return current_entry

active_entry = render_top_bar()
if not active_entry:
    st.stop()

# ==========================================
# 5. BUILD ACTIVE GRAPH
# ==========================================
df_edges = active_entry["edges"]
src_col = active_entry["tags"].get("src_col", df_edges.columns[0])
dst_col = active_entry["tags"].get("dst_col", df_edges.columns[1])

df_filtered = filter_edges(df_edges, src_col, dst_col, min_conf, min_weight)
G_full = build_graph_from_edges(df_filtered, src_col, dst_col)

with st.sidebar:
    st.markdown("### 📊 Текущий граф")
    st.caption(f"ID: {active_entry['id']}")
    col_s1, col_s2 = st.columns(2)
    col_s1.metric("Nodes", G_full.number_of_nodes())
    col_s2.metric("Edges", G_full.number_of_edges())

    st.markdown("---")
    st.markdown("**🔍 Область анализа**")
    analysis_mode = st.radio("Режим", ["Global (Весь граф)", "LCC (Гигантская комп.)"])

    if analysis_mode.startswith("LCC"):
        G_view = lcc_subgraph(G_full)
    else:
        G_view = G_full

    seed_val = st.number_input("Random Seed", value=int(st.session_state["seed"]), step=1)
    st.session_state["seed"] = int(seed_val)

# ==========================================
# 6. MAIN TABS
# ==========================================
tab_main, tab_struct, tab_null, tab_attack, tab_compare = st.tabs([
    "📊 Дэшборд",
    "🕸️ Структура и 3D",
    "🧪 Нулевые модели",
    "💥 Attack Lab",
    "🆚 Сравнение"
])

# ---------------------------------------------------------------------
# TAB 1: DASHBOARD
# ---------------------------------------------------------------------
with tab_main:
    st.header(f"Обзор: {active_entry['name']}")

    met = calculate_metrics(G_view, eff_sources_k=32, seed=seed_val)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("N (Nodes)", met["N"], help=help_icon("N"))
    k2.metric("E (Edges)", met["E"], help=help_icon("E"))
    k3.metric("Density", f"{met['density']:.5f}", help=help_icon("Density"))
    k4.metric("Avg Degree", f"{met['avg_degree']:.2f}")

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Components", met["C"])
    c2.metric("LCC Size", met["lcc_size"], f"{met['lcc_frac']*100:.1f}%", help=help_icon("LCC frac"))
    c3.metric("Diameter (approx)", met["diameter_approx"] if met.get("diameter_approx") else "N/A")
    c4.metric("Efficiency", f"{met['eff_w']:.4f}", help=help_icon("Efficiency (Global)"))

    st.markdown("---")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Modularity Q", f"{met['mod']:.4f}", help=help_icon("Modularity Q"))
    m2.metric("Lambda2 (LCC)", f"{met['l2_lcc']:.5f}", help=help_icon("Lambda2 (Algebraic Connectivity)"))
    m3.metric("Assortativity", f"{met['assortativity']:.4f}", help=help_icon("Assortativity"))
    m4.metric("Clustering", f"{met['clustering']:.4f}", help=help_icon("Clustering"))

    st.markdown("### 📈 Распределения")
    d1, d2 = st.columns(2)

    with d1:
        degrees = [d for _, d in G_view.degree()]
        if degrees:
            fig_deg = px.histogram(
                x=degrees, nbins=30, title="Degree Distribution",
                labels={'x': 'Degree', 'y': 'Count'}
            )
            fig_deg.update_layout(template="plotly_dark", showlegend=False)
            st.plotly_chart(fig_deg, use_container_width=True)
        else:
            st.info("Пустой граф")

    with d2:
        weights = [d.get('weight', 0) for _, _, d in G_view.edges(data=True)]
        if weights:
            fig_w = px.histogram(
                x=weights, nbins=30, title="Weight Distribution",
                labels={'x': 'Weight', 'y': 'Count'}
            )
            fig_w.update_layout(template="plotly_dark", showlegend=False)
            st.plotly_chart(fig_w, use_container_width=True)
        else:
            st.info("Нет весов")

# ---------------------------------------------------------------------
# TAB 2: STRUCTURE & 3D
# ---------------------------------------------------------------------
with tab_struct:
    col_vis_ctrl, col_vis_main = st.columns([1, 4])

    with col_vis_ctrl:
        st.subheader("Настройки 3D")
        show_labels = st.checkbox("Показать ID узлов", False)
        node_size = st.slider("Размер узлов", 1, 20, 4)
        st.info("3D: force-directed. Для больших графов может тормозить.")
        if st.button("🔄 Пересчитать Layout"):
            st.session_state["layout_seed_bump"] += 1
            st.rerun()

    with col_vis_main:
        if G_view.number_of_nodes() > 2000:
            st.warning(f"Граф большой ({G_view.number_of_nodes()} узлов). Рендеринг может тормозить.")

        layout_seed = int(seed_val) + int(st.session_state["layout_seed_bump"])
        pos3d = compute_3d_layout(G_view, seed=layout_seed)
        edge_trace, node_trace = make_3d_traces(G_view, pos3d, show_scale=True)

        if node_trace:
            node_trace.marker.size = node_size
            if show_labels:
                node_trace.mode = "markers+text"

            fig_3d = go.Figure(data=[edge_trace, node_trace])
            fig_3d.update_layout(
                title=f"3D Structure: {active_entry['name']}",
                template="plotly_dark",
                showlegend=False,
                margin=dict(l=0, r=0, t=30, b=0),
                scene=dict(
                    xaxis=dict(showbackground=False, showticklabels=False, title=''),
                    yaxis=dict(showbackground=False, showticklabels=False, title=''),
                    zaxis=dict(showbackground=False, showticklabels=False, title=''),
                )
            )
            st.plotly_chart(fig_3d, use_container_width=True, height=600)
        else:
            st.write("Граф пуст.")

    st.markdown("---")
    st.subheader("Матрица смежности")
    if G_view.number_of_nodes() < 1000 and G_view.number_of_nodes() > 0:
        adj = nx.adjacency_matrix(G_view, weight="weight").todense()
        fig_hm = px.imshow(adj, title="Adjacency Heatmap", color_continuous_scale="Viridis")
        fig_hm.update_layout(template="plotly_dark", width=650, height=650)
        st.plotly_chart(fig_hm, use_container_width=False)
    else:
        st.warning("Матрица слишком большая (N >= 1000) или граф пуст.")

# ---------------------------------------------------------------------
# TAB 3: NULL MODELS
# ---------------------------------------------------------------------
with tab_null:
    st.header("🧬 Генератор и Нулевые модели")
    st.markdown(
        """
        1) **ER G(n,m)**: сохраняет N и M, убивает структуру.  
        2) **Configuration model**: сохраняет распределение степеней.  
        3) **Mix/Rewire(p)**: постепенная рандомизация.
        """
    )

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

                edges = [[u, v, 1.0, 1.0] for u, v in G_new.edges()]
                df_new = pd.DataFrame(edges, columns=["src", "dst", "weight", "confidence"])

                add_graph(
                    name=f"{active_entry['name']}{new_name_suffix}",
                    df_edges=df_new,
                    source=f"null:{src_tag}",
                    tags={"src_col": "src", "dst_col": "dst"}
                )
                st.success("Граф создан. Переключаюсь...")
                st.rerun()

    with nm_col2:
        st.info("Сравнение активного графа с ожиданиями ER (грубая аппрокс):")
        met_here = calculate_metrics(G_view, eff_sources_k=16, seed=seed_val)

        N, M = G_view.number_of_nodes(), G_view.number_of_edges()
        er_density = 2 * M / (N * (N - 1)) if N > 1 else 0.0
        er_clustering = er_density

        cmp_df = pd.DataFrame({
            "Metric": ["Avg Degree", "Density", "Clustering (C)", "Modularity (rough)"],
            "Active Graph": [met_here["avg_degree"], met_here["density"], met_here["clustering"], met_here["mod"]],
            "ER Expected": [met_here["avg_degree"], er_density, er_clustering, 0.0],
        })
        st.dataframe(cmp_df, use_container_width=True)

# ---------------------------------------------------------------------
# TAB 4: ATTACK LAB (+ batch, + multi-graph)
# ---------------------------------------------------------------------
with tab_attack:
    st.header("💥 Лаборатория разрушения (Attack Lab)")
    st.caption("Control parameter: removed_frac. Order parameters: lcc_frac / λ₂ / efficiency. Это и есть язык фазового перехода.")

    subtab_single, subtab_batch, subtab_multigraph = st.tabs(["Single", "Preset batch", "Multi-graph batch"])

    # ---- Single ----
    with subtab_single:
        col_atk_setup, col_atk_res = st.columns([1, 2])

        with col_atk_setup:
            with st.container(border=True):
                st.subheader("Параметры атаки (single)")

                attack_type = st.selectbox(
                    "Стратегия",
                    ["random", "degree (Hubs)", "betweenness (Bridges)", "kcore (Deep Core)", "rich-club (Top Strength)"],
                )
                kind_map = {
                    "random": "random",
                    "degree (Hubs)": "degree",
                    "betweenness (Bridges)": "betweenness",
                    "kcore (Deep Core)": "kcore",
                    "rich-club (Top Strength)": "richclub_top",
                }
                real_kind = kind_map.get(attack_type, "random")

                frac = st.slider("Удалить долю узлов", 0.05, 0.95, 0.5, 0.05, key="single_frac")
                steps = st.slider("Количество шагов", 5, 150, 30, key="single_steps")
                atk_seed = st.number_input("Seed симуляции", value=int(seed_val), key="single_seed")

                with st.expander("Дополнительно"):
                    eff_k = st.slider("Efficiency Samples (k)", 16, 256, 32, key="single_effk")
                    heavy_freq = st.slider("Тяжёлые метрики каждые N шагов", 1, 10, 2, key="single_heavy")
                    exp_tag = st.text_input("Тег эксперимента", "", key="single_tag")

                if st.button("🚀 ЗАПУСТИТЬ SINGLE", type="primary", use_container_width=True):
                    if G_view.number_of_nodes() < 5:
                        st.error("Граф слишком мал.")
                    else:
                        with st.spinner(f"Выполняется атака {real_kind}..."):
                            df_hist, _ = run_attack(
                                G_view, real_kind, frac, steps, int(atk_seed), int(eff_k),
                                rc_frac=0.1, compute_heavy_every=int(heavy_freq)
                            )
                            df_hist = _forward_fill_heavy(df_hist)
                            phase_info = classify_phase_transition(df_hist)

                            label = f"{active_entry['name']} | {real_kind}"
                            if exp_tag:
                                label += f" [{exp_tag}]"

                            save_experiment(
                                name=label,
                                graph_id=active_entry["id"],
                                kind=real_kind,
                                params={
                                    "frac": frac,
                                    "steps": steps,
                                    "seed": int(atk_seed),
                                    "phase": phase_info,
                                    "compute_heavy_every": int(heavy_freq),
                                    "eff_k": int(eff_k),
                                },
                                df_hist=df_hist
                            )
                            st.session_state["last_exp_id"] = st.session_state["experiments"][-1]["id"]
                        st.success("Готово. Сохранено.")
                        st.rerun()

        with col_atk_res:
            exps = [e for e in st.session_state["experiments"] if e["graph_id"] == active_entry["id"]]
            if not exps:
                st.info("Нет экспериментов для активного графа.")
            else:
                exps.sort(key=lambda x: x["created_at"], reverse=True)
                last_exp = exps[0]

                st.subheader(f"Последний результат: {last_exp['name']}")
                ph = last_exp.get("params", {}).get("phase", {})
                if ph:
                    st.caption(
                        f"Phase: {'Abrupt' if ph.get('is_abrupt') else 'Continuous'} "
                        f"| Critical ~ {ph.get('critical_x', 0):.3f}"
                    )

                df_res = _forward_fill_heavy(last_exp["history"])

                tab_plot1, tab_plot2, tab_plot3 = st.tabs(["📉 Кривые", "🌀 Фазпортреты", "📌 Susceptibility"])
                with tab_plot1:
                    fig = fig_metrics_over_steps(df_res, title="Динамика метрик")
                    st.plotly_chart(fig, use_container_width=True)

                with tab_plot2:
                    if "l2_lcc" in df_res.columns and "lcc_frac" in df_res.columns:
                        fig1 = px.line(df_res, x="removed_frac", y="lcc_frac", title="Order: LCC fraction vs control (removed_frac)")
                        fig1.update_layout(template="plotly_dark")
                        st.plotly_chart(fig1, use_container_width=True)

                        if "eff_w" in df_res.columns:
                            fig2 = px.line(df_res, x="removed_frac", y="eff_w", title="Order: Efficiency vs control (removed_frac)")
                            fig2.update_layout(template="plotly_dark")
                            st.plotly_chart(fig2, use_container_width=True)

                        if "mod" in df_res.columns:
                            fig3 = px.line(df_res, x="l2_lcc", y="mod", title="Phase portrait: Q vs λ₂ (trajectory)")
                            fig3.update_layout(template="plotly_dark")
                            st.plotly_chart(fig3, use_container_width=True)
                    else:
                        st.warning("Нет нужных колонок для фазового анализа.")

                with tab_plot3:
                    if "lcc_frac" in df_res.columns and "removed_frac" in df_res.columns:
                        dfp = df_res.sort_values("removed_frac").copy()
                        dx = dfp["removed_frac"].diff()
                        dfp["d_lcc_dx"] = dfp["lcc_frac"].diff() / dx.replace(0, np.nan)
                        figS = px.line(dfp, x="removed_frac", y="d_lcc_dx", title="Susceptibility proxy: d(LCC)/dx")
                        figS.update_layout(template="plotly_dark")
                        st.plotly_chart(figS, use_container_width=True)
                    else:
                        st.warning("Нет removed_frac/lcc_frac.")

    # ---- Preset batch (single graph, many runs) ----
    with subtab_batch:
        col_b1, col_b2 = st.columns([1, 2])

        with col_b1:
            with st.container(border=True):
                st.subheader("Preset batch на активном графе")

                preset_name = st.selectbox("Preset", list(ATTACK_PRESETS.keys()), key="batch_preset")
                preset = ATTACK_PRESETS[preset_name]

                frac = st.slider("Удалить долю узлов", 0.05, 0.95, 0.5, 0.05, key="batch_frac")
                steps = st.slider("Шаги", 5, 200, 40, key="batch_steps")
                base_seed = st.number_input("Base seed", value=int(seed_val), step=1, key="batch_seed")

                with st.expander("Batch advanced"):
                    eff_k = st.slider("Efficiency k", 16, 256, 32, key="batch_effk")
                    heavy_freq = st.slider("Heavy every N", 1, 10, 2, key="batch_heavy")
                    rc_frac = st.slider("Rich-club frac", 0.01, 0.30, 0.10, 0.01, key="batch_rc")
                    exp_tag = st.text_input("Тег", "", key="batch_tag")

                if st.button("🚀 RUN PRESET SUITE", type="primary", use_container_width=True):
                    if G_view.number_of_nodes() < 5:
                        st.error("Граф слишком мал.")
                    else:
                        with st.spinner(f"Running preset: {preset_name}"):
                            curves = run_attack_suite(
                                G_view, active_entry, preset,
                                frac=float(frac), steps=int(steps), base_seed=int(base_seed),
                                eff_k=int(eff_k), heavy_freq=int(heavy_freq),
                                rc_frac=float(rc_frac), tag=exp_tag
                            )
                        st.session_state["last_suite_curves"] = curves
                        st.success(f"Готово: {len(curves)} прогонов сохранено.")
                        st.rerun()

        with col_b2:
            curves = st.session_state.get("last_suite_curves")
            if not curves:
                st.info("Запусти preset batch слева — здесь появится сравнение.")
            else:
                st.subheader("Сравнение (preset suite)")
                y_axis = st.selectbox("Y", ["lcc_frac", "eff_w", "l2_lcc", "mod"], index=0, key="suite_y")
                fig = fig_compare_attacks(curves, "removed_frac", y_axis, f"Preset compare: {y_axis}")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")
                st.subheader("AUC (robustness) по выбранной метрике")
                rows = []
                for name, df in curves:
                    if y_axis in df.columns and "removed_frac" in df.columns:
                        y = pd.to_numeric(df[y_axis], errors="coerce").fillna(method="ffill").fillna(0.0).values
                        x = pd.to_numeric(df["removed_frac"], errors="coerce").fillna(0.0).values
                        rows.append({"run": name, "AUC": float(np.trapz(y, x))})
                if rows:
                    st.dataframe(pd.DataFrame(rows).sort_values("AUC", ascending=False), use_container_width=True)

    # ---- Multi-graph batch ----
    with subtab_multigraph:
        st.subheader("Multi-graph batch: один preset на нескольких графах")
        graphs = st.session_state["graphs"]
        gid_list = list(graphs.keys())

        sel_gids = st.multiselect(
            "Графы для прогона",
            gid_list,
            default=[st.session_state["active_graph_id"]] if st.session_state["active_graph_id"] else [],
            format_func=lambda gid: f"{graphs[gid]['name']} ({graphs[gid]['source']})",
            key="mg_gids"
        )

        cA, cB = st.columns([1, 1])
        with cA:
            preset_name_mg = st.selectbox("Preset", list(ATTACK_PRESETS.keys()), key="mg_preset")
        with cB:
            mg_tag = st.text_input("Тег", value="MG", key="mg_tag")

        mg_frac = st.slider("Удалить долю узлов", 0.05, 0.95, 0.5, 0.05, key="mg_frac")
        mg_steps = st.slider("Шаги", 5, 200, 40, key="mg_steps")

        cC, cD, cE = st.columns(3)
        with cC:
            mg_seed = st.number_input("Base seed", value=123, step=1, key="mg_seed")
        with cD:
            mg_effk = st.slider("Efficiency k", 16, 256, 32, key="mg_effk")
        with cE:
            mg_heavy = st.slider("Heavy every N", 1, 10, 2, key="mg_heavy")

        if st.button("🚀 RUN MULTI-GRAPH SUITE", type="primary", use_container_width=True):
            if not sel_gids:
                st.error("Выбери хотя бы один граф.")
            else:
                preset_mg = ATTACK_PRESETS[preset_name_mg]
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

                        curves = run_attack_suite(
                            _G, entry, preset_mg,
                            frac=float(mg_frac), steps=int(mg_steps),
                            base_seed=int(mg_seed),
                            eff_k=int(mg_effk), heavy_freq=int(mg_heavy),
                            rc_frac=0.1, tag=f"{mg_tag}:{preset_name_mg}"
                        )
                        all_curves.extend(curves)

                st.session_state["last_multi_curves"] = all_curves
                st.success(f"Готово: {len(all_curves)} прогонов.")
                st.rerun()

        multi_curves = st.session_state.get("last_multi_curves")
        if multi_curves:
            st.markdown("---")
            y = st.selectbox("Y (multi compare)", ["lcc_frac", "eff_w", "l2_lcc", "mod"], key="mg_y")
            fig = fig_compare_attacks(multi_curves, "removed_frac", y, f"Multi-graph compare: {y}")
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------
# TAB 5: COMPARISON
# ---------------------------------------------------------------------
with tab_compare:
    st.header("🆚 Сравнение")
    mode_cmp = st.radio("Что сравниваем?", ["Графы (Скаляры)", "Атаки (Траектории)"], horizontal=True)

    graphs = st.session_state["graphs"]
    all_gids = list(graphs.keys())

    if mode_cmp.startswith("Графы"):
        st.subheader("Сравнение структурных свойств")
        selected_gids = st.multiselect(
            "Выберите графы",
            all_gids,
            default=[active_entry["id"]] if active_entry["id"] in all_gids else [],
            format_func=lambda gid: f"{graphs[gid]['name']} ({graphs[gid]['source']})"
        )

        scalar_metric = st.selectbox(
            "Метрика",
            ["density", "l2_lcc", "mod", "eff_w", "avg_degree", "clustering", "assortativity"],
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
            fig_bar.update_layout(template="plotly_dark", showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
            st.dataframe(df_cmp, use_container_width=True)
        else:
            st.info("Выбери хотя бы один граф.")

    else:
        st.subheader("Сравнение устойчивости (кривые)")
        exps = st.session_state["experiments"]
        if not exps:
            st.warning("Нет сохраненных экспериментов.")
        else:
            exp_opts = {e["id"]: f"{e['name']} ({time.strftime('%H:%M', time.localtime(e['created_at']))})" for e in exps}
            sel_exps = st.multiselect(
                "Выберите эксперименты",
                list(exp_opts.keys()),
                format_func=lambda x: exp_opts[x]
            )

            y_axis = st.selectbox("Y Axis", ["lcc_frac", "eff_w", "mod", "l2_lcc"], index=0)

            if sel_exps:
                curves = []
                for eid in sel_exps:
                    e = next(x for x in exps if x["id"] == eid)
                    curves.append((e["name"], _forward_fill_heavy(e["history"])))

                fig_lines = fig_compare_attacks(curves, "removed_frac", y_axis, f"Comparison: {y_axis}")
                st.plotly_chart(fig_lines, use_container_width=True)

                st.markdown("**Robustness (AUC)**")
                auc_rows = []
                for name, df in curves:
                    if y_axis in df.columns and "removed_frac" in df.columns:
                        y = pd.to_numeric(df[y_axis], errors="coerce").fillna(method="ffill").fillna(0.0).values
                        x = pd.to_numeric(df["removed_frac"], errors="coerce").fillna(0.0).values
                        auc_rows.append({"Experiment": name, "AUC": float(np.trapz(y, x))})
                if auc_rows:
                    st.dataframe(pd.DataFrame(auc_rows).sort_values("AUC", ascending=False), use_container_width=True)
            else:
                st.info("Выбери эксперименты для сравнения.")

# Footer
st.markdown("---")
st.caption("💀 Kodik Lab v2.1 | Streamlit + NetworkX")
