import time
import uuid
import hashlib

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import streamlit as st

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
# Page
# -------------------------
st.set_page_config(page_title="прикольчик", layout="wide", page_icon="💀")
st.title("прикольчик")

# Sticky topbar CSS
st.markdown(
    """
    <style>
      /* Make the container that holds .sticky-topbar sticky */
      div[data-testid="stVerticalBlock"] > div:has(> div.sticky-topbar){
        position: sticky;
        top: 0;
        z-index: 999;
        background: rgba(15, 17, 22, 0.96);
        backdrop-filter: blur(8px);
        border-bottom: 1px solid rgba(255,255,255,0.08);
        padding-top: 0.25rem;
        padding-bottom: 0.25rem;
      }

      .sticky-topbar .stButton>button { height: 2.4rem; }
      .sticky-topbar .stTextInput input { height: 2.4rem; }
      .sticky-topbar .stSelectbox div[data-baseweb="select"] { min-height: 2.4rem; }
      .sticky-topbar .stNumberInput input { height: 2.4rem; }

      /* Reduce vertical gaps inside topbar */
      .sticky-topbar [data-testid="stVerticalBlock"] { gap: 0.25rem; }

      /* Slightly tighter labels */
      .sticky-topbar label { font-size: 0.85rem; opacity: 0.9; }

      /* Avoid huge top padding under sticky bar */
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
        # gid -> {id,name,source,tags,edges,created_at}
        st.session_state["graphs"] = {}

    if "experiments" not in st.session_state:
        st.session_state["experiments"] = []

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
        "tags": tags or {},  # expects {"src_col":..., "dst_col":...}
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


def build_active_graph(min_conf: int, min_weight: float):
    entry = get_active_graph_entry()
    if entry is None:
        return None, None

    df_edges = entry["edges"]
    src_col = entry["tags"].get("src_col", df_edges.columns[0])
    dst_col = entry["tags"].get("dst_col", df_edges.columns[1])

    df_f = filter_edges(
        df_edges,
        src_col=src_col,
        dst_col=dst_col,
        min_conf=int(min_conf),
        min_weight=float(min_weight),
    )
    G = build_graph_from_edges(df_f, src_col=src_col, dst_col=dst_col)
    return G, {"src_col": src_col, "dst_col": dst_col, "df_filtered": df_f}


def push_experiment(name: str, graph_id: str, attack_kind: str, params: dict, df_hist: pd.DataFrame):
    exp_id = new_id("EXP")
    st.session_state["experiments"].append(
        {
            "id": exp_id,
            "name": name,
            "graph_id": graph_id,
            "attack_kind": attack_kind,
            "params": params,
            "history": df_hist.copy(),
            "created_at": time.time(),
        }
    )
    return exp_id


def fingerprint_upload(uploaded) -> str:
    b = uploaded.getvalue()
    return hashlib.md5(b).hexdigest() + f":{uploaded.name}:{len(b)}"


# =========================
# Sticky Topbar
# =========================
def render_sticky_topbar():
    graphs = st.session_state["graphs"]
    gids = sorted_graph_ids()

    with st.container():
        st.markdown('<div class="sticky-topbar">', unsafe_allow_html=True)

        # If empty, show only generator (manual ER possible), but we still need src/dst column names.
        # So: if no graphs exist yet, allow creating a "fresh ER" with synthetic columns.
        if not gids:
            r1, r2, r3, r4, r5, r6 = st.columns([2.0, 1.0, 1.0, 1.3, 1.3, 1.4])
            with r1:
                st.warning("Workspace пустой.")
            with r2:
                seed = st.number_input("Seed", value=int(st.session_state["seed_top"]), step=1, key="top_seed_empty")
                st.session_state["seed_top"] = int(seed)
            with r3:
                n_val = st.number_input("N", min_value=2, value=200, step=1, key="top_n_empty")
            with r4:
                m_val = st.number_input("M", min_value=1, value=600, step=1, key="top_m_empty")
            with r5:
                name = st.text_input("Имя", value="случайный граф", key="top_name_empty", label_visibility="collapsed")
            with r6:
                if st.button("Generate ER", key="top_gen_er_empty", use_container_width=True):
                    G_new = make_er_gnm(int(n_val), int(m_val), seed=int(seed))
                    edges = [[u, v, 1.0, 1.0] for u, v in G_new.edges()]
                    df_new = pd.DataFrame(edges, columns=["src", "dst", "weight", "confidence"])
                    add_graph_to_workspace(
                        name=(name or "").strip() or f"ER(n={int(n_val)},m={int(m_val)},seed={seed})",
                        df_edges=df_new,
                        source="null:ER",
                        tags={"src_col": "src", "dst_col": "dst"},
                    )
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)
            return None

        # Ensure active exists
        if st.session_state["active_graph_id"] not in graphs:
            st.session_state["active_graph_id"] = gids[0]

        active_gid = st.session_state["active_graph_id"]
        active_entry = graphs[active_gid]

        # Row 1: switch / rename / delete
        c1, c2, c3, c4 = st.columns([2.6, 2.2, 1.1, 1.1])

        with c1:
            picked = st.selectbox(
                "Активный граф",
                options=gids,
                index=gids.index(active_gid),
                format_func=lambda gid: graphs[gid]["name"],
                key="top_active_graph",
                label_visibility="collapsed",
            )
            if picked != active_gid:
                st.session_state["active_graph_id"] = picked
                st.rerun()

        with c2:
            new_name = st.text_input(
                "Имя вкладки",
                value=active_entry["name"],
                key="top_rename",
                label_visibility="collapsed",
                placeholder="переименовать",
            )

        with c3:
            if st.button("Rename", key="top_rename_btn", use_container_width=True):
                nn = (new_name or "").strip()
                if nn:
                    st.session_state["graphs"][active_gid]["name"] = nn
                    st.rerun()

        with c4:
            if st.button("Delete", key="top_delete_btn", use_container_width=True):
                st.session_state["graphs"].pop(active_gid, None)
                st.session_state["experiments"] = [e for e in st.session_state["experiments"] if e.get("graph_id") != active_gid]
                gids2 = sorted_graph_ids()
                st.session_state["active_graph_id"] = gids2[0] if gids2 else None
                st.rerun()

        # Compute base graph stats for "based on active" mode
        active_entry = st.session_state["graphs"][st.session_state["active_graph_id"]]
        df_base = active_entry["edges"]
        src_col = active_entry["tags"].get("src_col", df_base.columns[0])
        dst_col = active_entry["tags"].get("dst_col", df_base.columns[1])
        G_base_full = build_graph_from_edges(df_base, src_col=src_col, dst_col=dst_col)
        N0, E0 = G_base_full.number_of_nodes(), G_base_full.number_of_edges()

        # Row 2: generator controls
        g1, g2, g3, g4, g5, g6 = st.columns([1.0, 1.6, 1.0, 1.0, 1.6, 1.2])

        with g1:
            seed = st.number_input(
                "Seed",
                value=int(st.session_state.get("seed_top", 42)),
                step=1,
                key="top_seed",
                label_visibility="collapsed",
            )
            st.session_state["seed_top"] = int(seed)

        with g2:
            gen_mode = st.selectbox(
                "Режим",
                ["На основе активного (N,E)", "Задать N и M вручную"],
                key="top_gen_mode",
                label_visibility="collapsed",
            )

        if gen_mode.startswith("На основе"):
            n_val, m_val = N0, E0
            with g3:
                st.number_input("N", value=int(n_val), disabled=True, key="top_n_disabled", label_visibility="collapsed")
            with g4:
                st.number_input("M", value=int(m_val), disabled=True, key="top_m_disabled", label_visibility="collapsed")
        else:
            with g3:
                n_val = st.number_input("N", min_value=2, value=max(2, int(N0) if N0 else 200), step=1, key="top_n")
            with g4:
                m_val = st.number_input("M", min_value=1, value=max(1, int(E0) if E0 else 600), step=1, key="top_m")

        with g5:
            gen_type = st.selectbox(
                "Тип",
                ["ER G(n,m)", "CFG (от активного)", "Mix/Rewire p (от активного)"],
                key="top_gen_type",
                label_visibility="collapsed",
            )

        with g6:
            new_graph_name = st.text_input(
                "Имя нового",
                value="",
                key="top_new_name",
                label_visibility="collapsed",
                placeholder="например: случайный граф",
            )

        # Row 3: action buttons + optional p
        a1, a2, a3, a4 = st.columns([1.4, 1.0, 1.0, 1.6])

        with a1:
            st.caption(f"Активный: N={N0} E={E0}")

        with a2:
            if st.button("Copy", key="top_copy_btn", use_container_width=True):
                add_graph_to_workspace(
                    name=f"copy:{active_entry['name']}",
                    df_edges=active_entry["edges"],
                    source="copy",
                    tags=active_entry["tags"],
                )
                st.rerun()

        with a3:
            # Delete all experiments for active graph quickly
            if st.button("Clear exp", key="top_clear_exp_btn", use_container_width=True):
                gid = st.session_state["active_graph_id"]
                st.session_state["experiments"] = [e for e in st.session_state["experiments"] if e.get("graph_id") != gid]
                st.rerun()

        with a4:
            if gen_type == "Mix/Rewire p (от активного)":
                p = st.slider("p", 0.0, 1.0, 0.2, 0.05, key="top_mix_p")
            else:
                p = None

            if st.button("Generate", key="top_generate_btn", use_container_width=True):
                nm = (new_graph_name or "").strip()

                if gen_type == "ER G(n,m)":
                    G_new = make_er_gnm(int(n_val), int(m_val), seed=int(seed))
                    edges = [[u, v, 1.0, 1.0] for u, v in G_new.edges()]
                    df_new = pd.DataFrame(edges, columns=[src_col, dst_col, "weight", "confidence"])
                    add_graph_to_workspace(
                        name=nm or f"ER(n={int(n_val)},m={int(m_val)},seed={seed})",
                        df_edges=df_new,
                        source="null:ER",
                        tags={"src_col": src_col, "dst_col": dst_col},
                    )
                    st.rerun()

                elif gen_type == "CFG (от активного)":
                    G_new = make_configuration_model(G_base_full, seed=int(seed))
                    edges = [[u, v, 1.0, 1.0] for u, v in G_new.edges()]
                    df_new = pd.DataFrame(edges, columns=[src_col, dst_col, "weight", "confidence"])
                    add_graph_to_workspace(
                        name=nm or f"CFG(deg,seed={seed}) from {active_entry['name']}",
                        df_edges=df_new,
                        source="null:CFG",
                        tags={"src_col": src_col, "dst_col": dst_col},
                    )
                    st.rerun()

                else:
                    p_val = float(p if p is not None else 0.2)
                    G_new = rewire_mix(G_base_full, p=p_val, seed=int(seed))
                    edges = [[u, v, 1.0, 1.0] for u, v in G_new.edges()]
                    df_new = pd.DataFrame(edges, columns=[src_col, dst_col, "weight", "confidence"])
                    add_graph_to_workspace(
                        name=nm or f"MIX(p={p_val:.2f},seed={seed}) from {active_entry['name']}",
                        df_edges=df_new,
                        source="mix:rewire",
                        tags={"src_col": src_col, "dst_col": dst_col},
                    )
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    return st.session_state["active_graph_id"]


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("🧠 Workspace")

    with st.expander("💾 Import / Export", expanded=False):
        c1, c2 = st.columns(2)

        if c1.button("Export workspace"):
            blob = export_workspace_json(st.session_state["graphs"], st.session_state["experiments"])
            st.download_button(
                "⬇️ workspace.json",
                data=blob,
                file_name="workspace.json",
                mime="application/json",
                use_container_width=True,
            )

        if c2.button("Export experiments"):
            blob = export_experiments_json(st.session_state["experiments"])
            st.download_button(
                "⬇️ experiments.json",
                data=blob,
                file_name="experiments.json",
                mime="application/json",
                use_container_width=True,
            )

        up_ws = st.file_uploader("Import workspace.json", type=["json"], key="ws_import")
        if up_ws is not None:
            try:
                graphs_new, exps_new = import_workspace_json(up_ws.getvalue())
                st.session_state["graphs"] = graphs_new
                st.session_state["experiments"] = exps_new
                gids = sorted_graph_ids()
                st.session_state["active_graph_id"] = gids[0] if gids else None
                st.success("Workspace импортирован.")
                st.rerun()
            except Exception as e:
                st.error(f"Ошибка импорта workspace: {e}")

        up_exps = st.file_uploader("Import experiments.json", type=["json"], key="exps_import")
        if up_exps is not None:
            try:
                exps_add = import_experiments_json(up_exps.getvalue())
                st.session_state["experiments"].extend(exps_add)
                st.success("Experiments импортированы (добавлены).")
                st.rerun()
            except Exception as e:
                st.error(f"Ошибка импорта experiments: {e}")

    st.write("---")
    st.header("📎 Upload графа")

    uploaded = st.file_uploader(
        "CSV/Excel (фикс. формат)",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=False,
        key="file_uploader_main",
    )
    st.caption("Ожидается: 1-я=source id, 2-я=target id, 9-я=confidence, 10-я=weight.")

    # Guard against duplicates on rerun:
    if uploaded is not None:
        fp = fingerprint_upload(uploaded)
        if fp != st.session_state["last_uploaded_fingerprint"]:
            try:
                df_any = load_uploaded_any(uploaded.getvalue(), uploaded.name)
                df_edges, meta = coerce_fixed_format(df_any)  # meta={"src_col":..., "dst_col":...}
                add_graph_to_workspace(
                    name=f"uploaded:{uploaded.name}",
                    df_edges=df_edges,
                    source="upload",
                    tags=meta,
                )
                st.session_state["last_uploaded_fingerprint"] = fp
                st.success(f"Добавлен: uploaded:{uploaded.name}")
                st.rerun()
            except Exception as e:
                st.error(f"Ошибка загрузки/парсинга: {e}")
        else:
            st.caption("Этот файл уже добавлен (не дублирую).")

    st.write("---")
    st.header("🎛️ Параметры анализа")

    min_conf = st.number_input("confidence ≥", value=0, step=1)
    min_weight = st.number_input("weight ≥", value=0.0, step=0.1)
    eff_sources_k = st.slider("Efficiency k", 8, 256, 64, 8)
    seed_analysis = st.number_input("Seed (анализ/3D)", value=42, step=1)

    st.write("---")
    st.header("🧹 Очистка")
    cc1, cc2 = st.columns(2)
    with cc1:
        if st.button("Clear experiments", use_container_width=True):
            st.session_state["experiments"] = []
            st.rerun()
    with cc2:
        if st.button("Clear ALL", use_container_width=True):
            st.session_state["graphs"] = {}
            st.session_state["experiments"] = []
            st.session_state["active_graph_id"] = None
            st.session_state["last_uploaded_fingerprint"] = None
            st.rerun()

# =========================
# Sticky Topbar (always renders)
# =========================
render_sticky_topbar()

# =========================
# Main: need an active graph to proceed
# =========================
entry = get_active_graph_entry()
if entry is None:
    st.stop()

graph_id = entry["id"]

# Build filtered graph
G, ctx = build_active_graph(min_conf=int(min_conf), min_weight=float(min_weight))
if G is None:
    st.stop()

mode = st.radio("Масштаб", ["Глобальный (весь граф)", "LCC (гигантская компонента)"], horizontal=True)
G_view = lcc_subgraph(G) if mode.startswith("LCC") else G

# Quick summary
s1, s2, s3, s4 = st.columns([2.6, 1.1, 1.1, 1.2])
with s1:
    st.subheader(f"Активный граф: {entry['name']}")
    st.caption(f"source={entry['source']} · id={graph_id} · фильтры: conf≥{min_conf}, w≥{min_weight}")
with s2:
    st.metric("Nodes", int(G_view.number_of_nodes()))
with s3:
    st.metric("Edges", int(G_view.number_of_edges()))
with s4:
    st.metric("Components", int(nx.number_connected_components(G_view)) if G_view.number_of_nodes() else 0)

base = calculate_metrics(G_view, eff_sources_k=int(eff_sources_k), seed=int(seed_analysis))

# =========================
# Tabs
# =========================
t1, t2, t3, t4, t5 = st.tabs(
    ["📊 База", "🧬 Спектр+сложность", "👁️ 3D", "💥 ATTACK LAB", "🆚 Сравнение"]
)

with t1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("N", base["N"])
    c2.metric("E", base["E"])
    c3.metric("Components C", base["C"])
    c4.metric("Beta cycles", base["beta"])

    c1.metric("LCC size", base["lcc_size"])
    c2.metric("LCC frac", f"{base['lcc_frac']:.4f}")
    c3.metric("Density", f"{base['density']:.6f}")
    c4.metric("Avg degree", f"{base['avg_degree']:.3f}")

    st.write("---")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Weighted Efficiency", f"{base['eff_w']:.6f}")
    c6.metric("λ₂ (LCC)", f"{base['l2_lcc']:.10f}")
    c7.metric("τ = 1/λ₂", f"{base['tau_lcc']:.4g}")
    c8.metric("Modularity Q", f"{base['mod']:.6f}")

    st.write("---")

    c9, c10, c11, c12 = st.columns(4)
    c9.metric("λmax(A)", f"{base['lmax']:.6f}")
    c10.metric("1/λmax", f"{base['thresh']:.6f}")
    c11.metric("Entropy(H_deg)", f"{base['entropy_deg']:.6f}")
    c12.metric("Assortativity(r)", f"{base['assortativity']:.6f}")

    st.write("---")
    st.subheader("Сводка")
    st.code(graph_summary(G_view), language="text")

with t2:
    left, right = st.columns([1.2, 1.0])
    with left:
        st.subheader("λ₂ / Q")
        st.write(f"- λ₂ (LCC) = {base['l2_lcc']:.6g}")
        st.write(f"- Q (Louvain) = {base['mod']:.6g}")
        st.markdown(
            "- λ₂ низкая → сеть легко рассоединяется\n"
            "- Q высокая → сильная модульность\n"
            "- Q низкая → размытая структура\n"
        )
    with right:
        st.subheader("Структура")
        st.write(f"- Degree-entropy H = {base['entropy_deg']:.6g}")
        st.write(f"- Assortativity r = {base['assortativity']:.6g}")
        st.write(f"- Clustering C̄ = {base['clustering']:.6g}")
        st.write(f"- Diameter (approx) = {base['diameter_approx']}")

    st.write("---")
    with st.expander("Диагностика (если λ₂=0/NaN)", expanded=False):
        st.write(f"N={base['N']} E={base['E']} C={base['C']}")
        H = lcc_subgraph(G_view)
        st.write(
            f"LCC: n={H.number_of_nodes()} e={H.number_of_edges()} "
            f"connected={nx.is_connected(H) if H.number_of_nodes() else False}"
        )

with t3:
    st.subheader("3D Projection")
    if G_view.number_of_nodes() == 0:
        st.info("Пустой граф после фильтров.")
    else:
        pos3d = compute_3d_layout(G_view, seed=int(seed_analysis))
        e_tr, n_tr = make_3d_traces(G_view, pos3d, show_scale=True)
        fig = go.Figure(data=[e_tr, n_tr])
        fig.update_layout(
            template="plotly_dark",
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
        )
        st.plotly_chart(fig, use_container_width=True)

with t4:
    st.subheader("💥 Лаборатория разрушения")
    st.caption("Результаты сохраняются в Workspace → вкладка 🆚.")

    with st.expander("Настройки симуляции", expanded=True):
        a1, a2, a3, a4 = st.columns(4)
        attack_kind_ui = a1.selectbox(
            "Вектор атаки",
            [
                "random",
                "degree (Hubs)",
                "betweenness (Bridges)",
                "kcore (Core)",
                "rich-club A (Top-fraction)",
                "rich-club B (Density-threshold)",
            ],
        )
        remove_frac = a2.slider("Удалить долю узлов", 0.05, 0.95, 0.50, 0.05)
        steps = a3.slider("Шагов", 5, 200, 40)
        attack_seed = a4.number_input("Seed (атака)", value=int(seed_analysis), step=1)

        st.write("—")
        b1, b2, b3, b4 = st.columns(4)
        eff_k_sim = b1.slider("Efficiency k (sim)", 8, 256, int(eff_sources_k), 8)
        compute_heavy_every = b2.slider("Heavy metrics every k steps", 1, 10, 1, 1)
        keep_states_3d = b3.checkbox("Хранить states (replay)", value=False)
        name_tag = b4.text_input("Тег эксперимента", value="")

        rc_frac = 0.10
        rc_min_density = 0.30
        rc_max_frac = 0.30

        if attack_kind_ui.startswith("rich-club A"):
            rc_frac = st.slider("RC-A: доля топ-узлов", 0.02, 0.50, 0.10, 0.02)
        if attack_kind_ui.startswith("rich-club B"):
            rc_min_density = st.slider("RC-B: min density клуба", 0.05, 1.00, 0.30, 0.05)
            rc_max_frac = st.slider("RC-B: max frac для поиска", 0.05, 0.80, 0.30, 0.05)

    if attack_kind_ui.startswith("degree"):
        attack_kind = "degree"
    elif attack_kind_ui.startswith("betweenness"):
        attack_kind = "betweenness"
    elif attack_kind_ui.startswith("kcore"):
        attack_kind = "kcore"
    elif attack_kind_ui.startswith("rich-club A"):
        attack_kind = "richclub_top"
    elif attack_kind_ui.startswith("rich-club B"):
        attack_kind = "richclub_density"
    else:
        attack_kind = "random"

    run_col1, run_col2 = st.columns(2)
    run_one = run_col1.button("▶️ Запустить (1)", use_container_width=True)
    run_all = run_col2.button("⚔️ Head-to-head (4)", use_container_width=True)

    def _run_and_render(kind: str, title_prefix: str):
        if G_view.number_of_nodes() < 2:
            st.warning("Граф слишком маленький.")
            return None

        df_hist, _states = run_attack(
            G_view,
            attack_kind=kind,
            remove_frac=float(remove_frac),
            steps=int(steps),
            seed=int(attack_seed),
            eff_sources_k=int(eff_k_sim),
            rc_frac=float(rc_frac),
            rc_min_density=float(rc_min_density),
            rc_max_frac=float(rc_max_frac),
            compute_heavy_every=int(compute_heavy_every),
            keep_states=bool(keep_states_3d),
        )

        phase = classify_phase_transition(df_hist, x_col="removed_frac", y_col="lcc_frac")

        label = title_prefix
        if (name_tag or "").strip():
            label += f" · {name_tag.strip()}"

        push_experiment(
            name=label,
            graph_id=graph_id,
            attack_kind=kind,
            params={
                "remove_frac": float(remove_frac),
                "steps": int(steps),
                "seed": int(attack_seed),
                "eff_k": int(eff_k_sim),
                "compute_heavy_every": int(compute_heavy_every),
                "rc_frac": float(rc_frac),
                "rc_min_density": float(rc_min_density),
                "rc_max_frac": float(rc_max_frac),
                "mode": mode,
                "active_graph_name": entry["name"],
                "phase": phase,
            },
            df_hist=df_hist,
        )

        st.success(f"Готово: {label}")
        st.caption(f"Phase: abrupt={phase['is_abrupt']} · critical~{phase['critical_x']:.3f} · jump={phase['jump']:.3f}")

        fig = fig_metrics_over_steps(df_hist, title=f"{label} — кривые")
        st.plotly_chart(fig, use_container_width=True)
        return df_hist

    if run_one:
        _run_and_render(attack_kind, attack_kind_ui)

    if run_all:
        st.write("### Head-to-head")
        kinds = [
            ("random", "random"),
            ("degree", "degree (Hubs)"),
            ("betweenness", "betweenness (Bridges)"),
            ("richclub_top", "rich-club A"),
        ]
        results = []
        for k, label in kinds:
            st.write(f"**Running:** {label}")
            res = _run_and_render(k, f"H2H:{label}")
            if res is not None and not res.empty:
                results.append((label, res))

        if results:
            st.write("---")
            fig_cmp = fig_compare_attacks(results, x_col="removed_frac", y_col="lcc_frac", title="Head-to-head: LCC")
            st.plotly_chart(fig_cmp, use_container_width=True)

with t5:
    st.subheader("🆚 Сравнение")

    graphs = st.session_state["graphs"]
    exps = st.session_state["experiments"]

    st.write("### Сравнить графы (скаляры)")
    graph_ids = sorted_graph_ids()
    selected_graphs = st.multiselect(
        "Графы",
        options=graph_ids,
        default=[graph_id] if graph_id in graph_ids else [],
        format_func=lambda gid: f"{graphs[gid]['name']} ({graphs[gid]['source']})",
    )

    scalar = st.selectbox(
        "Скаляр",
        ["N", "E", "density", "eff_w", "l2_lcc", "mod", "lmax", "entropy_deg", "assortativity", "clustering"],
    )

    if selected_graphs:
        rows = []
        for gid in selected_graphs:
            gentry = graphs[gid]
            df_edges = gentry["edges"]
            src_col = gentry["tags"].get("src_col", df_edges.columns[0])
            dst_col = gentry["tags"].get("dst_col", df_edges.columns[1])
            df_f = filter_edges(df_edges, src_col, dst_col, min_conf=int(min_conf), min_weight=float(min_weight))
            Gi = build_graph_from_edges(df_f, src_col, dst_col)
            Gi_view = lcc_subgraph(Gi) if mode.startswith("LCC") else Gi
            met = calculate_metrics(Gi_view, eff_sources_k=int(eff_sources_k), seed=int(seed_analysis))
            rows.append({"graph": gentry["name"], "source": gentry["source"], scalar: met.get(scalar)})

        df_cmp = pd.DataFrame(rows)
        fig = fig_compare_graphs_scalar(df_cmp, x="graph", y=scalar, title=f"Graphs compare: {scalar}")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_cmp, use_container_width=True)

    st.write("---")
    st.write("### Сравнить эксперименты (траектории)")

    if not exps:
        st.info("Нет экспериментов. Запусти атаку в 💥.")
    else:
        exp_by_id = {e["id"]: e for e in exps}
        exp_ids = [e["id"] for e in exps]
        selected_exps = st.multiselect(
            "Эксперименты",
            options=exp_ids,
            default=exp_ids[-3:] if len(exp_ids) >= 3 else exp_ids,
            format_func=lambda eid: f"{exp_by_id[eid]['name']} (graph={graphs.get(exp_by_id[eid]['graph_id'], {}).get('name','?')})",
        )
        y_metric = st.selectbox("Y", ["lcc_frac", "mod", "l2_lcc", "eff_w", "lmax"], index=0)

        curves = []
        for eid in selected_exps:
            e = exp_by_id[eid]
            dfh = e["history"]
            if isinstance(dfh, pd.DataFrame) and not dfh.empty:
                curves.append((e["name"], dfh))

        if curves:
            fig = fig_compare_attacks(curves, x_col="removed_frac", y_col=y_metric, title=f"Compare: {y_metric}")
            st.plotly_chart(fig, use_container_width=True)
