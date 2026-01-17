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

# -----------------------------------------------------------------------------
# 1) Config & CSS (dashboard look + sticky topbar)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Kanonar Lab", layout="wide", page_icon="🧬")

st.markdown(
    """
<style>
  /* Modern Dashboard Look */
  .block-container { padding-top: 0.75rem; }

  /* Metrics styling */
  div[data-testid="stMetricValue"] { font-size: 1.6rem; }

  /* Sidebar tightness */
  section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

  /* Sticky topbar container */
  div[data-testid="stVerticalBlock"] > div:has(> div.sticky-topbar){
    position: sticky;
    top: 0;
    z-index: 999;
    background: rgba(15, 17, 22, 0.96);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid rgba(255,255,255,0.08);
    padding-top: 0.35rem;
    padding-bottom: 0.25rem;
  }
  .sticky-topbar .stButton>button { height: 2.35rem; }
  .sticky-topbar .stTextInput input { height: 2.35rem; }
  .sticky-topbar .stSelectbox div[data-baseweb="select"] { min-height: 2.35rem; }
  .sticky-topbar .stNumberInput input { height: 2.35rem; }
  .sticky-topbar label { font-size: 0.82rem; opacity: 0.9; }
  .sticky-topbar [data-testid="stVerticalBlock"] { gap: 0.25rem; }

  /* Small helper box */
  .active-graph-box {
    border: 1px solid rgba(255, 255, 255, 0.16);
    padding: 10px;
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.04);
    margin-bottom: 10px;
  }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2) State & helpers
# -----------------------------------------------------------------------------
def _init_state():
    defaults = {
        "graphs": {},  # gid -> {id,name,source,tags,edges,created_at}
        "experiments": [],  # list of {id,name,graph_id,attack_kind,params,history,created_at}
        "active_graph_id": None,
        "last_uploaded_fingerprint": None,
        "seed_top": 42,
        "mix_p_by_gid": {},  # base_gid -> p_now
        "base_metrics_cache": {},  # key=(gid, conf, w, lcc_only) -> metrics
        "last_run": None,
        "cmp_results": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def sorted_graph_ids():
    g = st.session_state["graphs"]
    return sorted(list(g.keys()), key=lambda k: g[k].get("created_at", 0.0))


def get_active_graph_entry():
    gid = st.session_state["active_graph_id"]
    if gid is None:
        return None
    return st.session_state["graphs"].get(gid)


def add_graph(name: str, df_edges: pd.DataFrame, source: str, tags: dict | None = None) -> str:
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


def delete_graph(gid: str):
    st.session_state["graphs"].pop(gid, None)
    st.session_state["experiments"] = [e for e in st.session_state["experiments"] if e.get("graph_id") != gid]
    gids = sorted_graph_ids()
    st.session_state["active_graph_id"] = gids[0] if gids else None


def fingerprint_upload(uploaded) -> str:
    b = uploaded.getvalue()
    return hashlib.md5(b).hexdigest() + f":{uploaded.name}:{len(b)}"


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


def build_graph_from_entry(entry, min_conf: int, min_weight: float):
    df_raw = entry["edges"]
    tags = entry["tags"]
    src_col = tags.get("src_col", df_raw.columns[0])
    dst_col = tags.get("dst_col", df_raw.columns[1])
    df_filt = filter_edges(df_raw, src_col, dst_col, min_conf, float(min_weight))
    G_full = build_graph_from_edges(df_filt, src_col, dst_col)
    return G_full, {"src_col": src_col, "dst_col": dst_col, "df_filtered": df_filt}


def cached_metrics_for(G: nx.Graph, cache_key: str, eff_k: int, seed: int):
    cache = st.session_state["base_metrics_cache"]
    if cache_key in cache:
        return cache[cache_key]
    m = calculate_metrics(G, eff_sources_k=int(eff_k), seed=int(seed))
    cache[cache_key] = m
    return m


# Helper for Phase Space Plotting
def plot_phase_space(df_hist: pd.DataFrame, title="Фазовый портрет (Q vs λ₂)"):
    fig = go.Figure()
    if df_hist is None or df_hist.empty:
        return fig
    if ("mod" not in df_hist.columns) or ("l2_lcc" not in df_hist.columns):
        return fig

    fig.add_trace(
        go.Scatter(
            x=df_hist["mod"],
            y=df_hist["l2_lcc"],
            mode="lines+markers",
            marker=dict(
                size=6,
                color=df_hist["removed_frac"] if "removed_frac" in df_hist.columns else None,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="% Removed"),
            ),
            text=[
                f"Step: {s}<br>Nodes: {n}"
                for s, n in zip(
                    df_hist["step"] if "step" in df_hist.columns else range(len(df_hist)),
                    df_hist["nodes_left"] if "nodes_left" in df_hist.columns else ["?"] * len(df_hist),
                )
            ],
            name="Траектория распада",
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        xaxis_title="Modularity (Q) — Структура",
        yaxis_title="Algebraic Connectivity (λ₂) — Связность",
        height=520,
        margin=dict(l=10, r=10, t=55, b=10),
    )
    fig.add_annotation(x=0.08, y=0.01, text="Хаос / Деменция", showarrow=False, font=dict(color="gray"))
    fig.add_annotation(x=0.80, y=0.01, text="Диссоциация", showarrow=False, font=dict(color="gray"))
    return fig


# -----------------------------------------------------------------------------
# 3) Sticky topbar: graph switch + rename + generator + gradual mixing
# -----------------------------------------------------------------------------
def render_sticky_topbar():
    graphs = st.session_state["graphs"]
    gids = sorted_graph_ids()

    with st.container():
        st.markdown('<div class="sticky-topbar">', unsafe_allow_html=True)

        # Empty workspace: allow ER creation from scratch
        if not gids:
            r1, r2, r3, r4, r5 = st.columns([1.7, 0.9, 0.9, 1.1, 1.4])
            with r1:
                st.markdown("**Workspace пустой**")
                st.caption("Сгенерь ER или загрузи файл в sidebar.")
            with r2:
                seed = st.number_input("Seed", value=int(st.session_state["seed_top"]), step=1, key="top_seed_empty")
                st.session_state["seed_top"] = int(seed)
            with r3:
                n_val = st.number_input("N", min_value=2, value=200, step=1, key="top_n_empty")
            with r4:
                m_val = st.number_input("M", min_value=1, value=600, step=1, key="top_m_empty")
            with r5:
                name = st.text_input("Имя", value="случайный граф", key="top_name_empty", label_visibility="collapsed")
                if st.button("Generate ER", key="top_gen_er_empty", use_container_width=True):
                    G_new = make_er_gnm(int(n_val), int(m_val), int(seed))
                    edges = [[u, v, 1.0, 1.0] for u, v in G_new.edges()]
                    df_new = pd.DataFrame(edges, columns=["src", "dst", "weight", "confidence"])
                    add_graph((name or "").strip() or f"ER(n={int(n_val)},m={int(m_val)},seed={seed})", df_new, "null:ER", {"src_col": "src", "dst_col": "dst"})
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)
            return

        # Ensure active valid
        if st.session_state["active_graph_id"] not in graphs:
            st.session_state["active_graph_id"] = gids[0]

        active_gid = st.session_state["active_graph_id"]
        active_entry = graphs[active_gid]

        # Row 1: switch / rename / delete / seed
        a1, a2, a3, a4 = st.columns([2.3, 2.1, 1.0, 0.9])

        with a1:
            picked = st.selectbox(
                "Активный граф",
                options=gids,
                index=gids.index(active_gid),
                format_func=lambda gid: graphs[gid]["name"],
                key="top_active_select",
                label_visibility="collapsed",
            )
            if picked != active_gid:
                st.session_state["active_graph_id"] = picked
                st.rerun()

        with a2:
            nm = st.text_input(
                "Имя вкладки",
                value=active_entry["name"],
                key="top_rename_input",
                label_visibility="collapsed",
                placeholder="переименовать вкладку",
            )

        with a3:
            if st.button("Rename", key="top_rename_btn", use_container_width=True):
                nn = (nm or "").strip()
                if nn:
                    st.session_state["graphs"][active_gid]["name"] = nn
                    st.rerun()

        with a4:
            seed = st.number_input("Seed", value=int(st.session_state["seed_top"]), step=1, key="top_seed", label_visibility="collapsed")
            st.session_state["seed_top"] = int(seed)

        # Build base for generation based on active
        df_base = active_entry["edges"]
        tags = active_entry["tags"]
        src_col = tags.get("src_col", df_base.columns[0])
        dst_col = tags.get("dst_col", df_base.columns[1])
        G_base = build_graph_from_edges(df_base, src_col, dst_col)
        N0, E0 = G_base.number_of_nodes(), G_base.number_of_edges()

        # Row 2: generator controls
        g1, g2, g3, g4, g5 = st.columns([1.4, 1.8, 1.0, 1.0, 1.8])
        with g1:
            st.caption(f"N={N0} · E={E0}")

        with g2:
            gen_mode = st.selectbox(
                "Режим генерации",
                ["На основе активного (N,E)", "Задать N и M вручную"],
                key="top_gen_mode",
                label_visibility="collapsed",
            )

        if gen_mode.startswith("На основе"):
            n_val, m_val = int(N0), int(E0)
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
                ["ER (G(n,m))", "CFG (сохр. степени активного)", "Mix/Rewire p (от активного)"],
                key="top_gen_type",
                label_visibility="collapsed",
            )

        # Row 3: name + actions
        b1, b2, b3, b4 = st.columns([2.1, 1.1, 1.1, 1.7])
        with b1:
            new_name = st.text_input(
                "Имя нового графа",
                value="",
                key="top_new_name",
                label_visibility="collapsed",
                placeholder="например: случайный граф / мутант / контроль",
            )

        with b2:
            if st.button("Copy", key="top_copy_btn", use_container_width=True):
                add_graph(f"copy:{active_entry['name']}", active_entry["edges"], "copy", tags)
                st.rerun()

        with b3:
            if st.button("Delete", key="top_delete_btn", use_container_width=True):
                delete_graph(active_gid)
                st.rerun()

        with b4:
            p_for_mix = None
            if gen_type.startswith("Mix/"):
                p_for_mix = st.slider("p", 0.0, 1.0, 0.2, 0.05, key="top_mix_p")

            if st.button("Generate", key="top_generate_btn", use_container_width=True):
                nm2 = (new_name or "").strip()

                if gen_type.startswith("ER"):
                    G_new = make_er_gnm(int(n_val), int(m_val), int(seed))
                    edges = [[u, v, 1.0, 1.0] for u, v in G_new.edges()]
                    df_new = pd.DataFrame(edges, columns=[src_col, dst_col, "weight", "confidence"])
                    add_graph(nm2 or f"ER(n={int(n_val)},m={int(m_val)},seed={seed})", df_new, "null:ER", {"src_col": src_col, "dst_col": dst_col})
                    st.rerun()

                elif gen_type.startswith("CFG"):
                    G_new = make_configuration_model(G_base, int(seed))
                    edges = [[u, v, 1.0, 1.0] for u, v in G_new.edges()]
                    df_new = pd.DataFrame(edges, columns=[src_col, dst_col, "weight", "confidence"])
                    add_graph(nm2 or f"CFG(deg,seed={seed}) from {active_entry['name']}", df_new, "null:CFG", {"src_col": src_col, "dst_col": dst_col})
                    st.rerun()

                else:
                    p_val = float(p_for_mix if p_for_mix is not None else 0.2)
                    G_new = rewire_mix(G_base, p=p_val, seed=int(seed))
                    edges = [[u, v, 1.0, 1.0] for u, v in G_new.edges()]
                    df_new = pd.DataFrame(edges, columns=[src_col, dst_col, "weight", "confidence"])
                    add_graph(nm2 or f"MIX(p={p_val:.2f},seed={seed}) from {active_entry['name']}", df_new, "mix:rewire", {"src_col": src_col, "dst_col": dst_col})
                    st.rerun()

        # Row 4: gradual mixing (one-click adds a new graph each step)
        c1, c2, c3 = st.columns([2.1, 1.1, 1.8])
        with c1:
            st.caption("Постепенная рандомизация: каждый шаг создаёт новый граф (и его можно анализировать).")
        with c2:
            step = st.slider("Δp", 0.01, 0.50, 0.05, 0.01, key="top_mix_step")
        with c3:
            p_now = float(st.session_state["mix_p_by_gid"].get(active_gid, 0.0))
            st.caption(f"p сейчас: {p_now:.2f}")
            x1, x2 = st.columns(2)
            with x1:
                if st.button("Step mix", key="top_stepmix_btn", use_container_width=True):
                    p_new = min(1.0, p_now + float(step))
                    st.session_state["mix_p_by_gid"][active_gid] = p_new
                    G_new = rewire_mix(G_base, p=float(p_new), seed=int(seed))
                    edges = [[u, v, 1.0, 1.0] for u, v in G_new.edges()]
                    df_new = pd.DataFrame(edges, columns=[src_col, dst_col, "weight", "confidence"])
                    add_graph(f"MIX-step(p={p_new:.2f},seed={seed}) from {active_entry['name']}", df_new, "mix:step", {"src_col": src_col, "dst_col": dst_col})
                    st.rerun()
            with x2:
                if st.button("Reset p", key="top_resetp_btn", use_container_width=True):
                    st.session_state["mix_p_by_gid"][active_gid] = 0.0
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 4) Sidebar: upload/import/export + filters
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🎛️ Kanonar")

    st.markdown("### 💾 Import / Export")
    c1, c2 = st.columns(2)
    if c1.button("Export workspace", use_container_width=True):
        blob = export_workspace_json(st.session_state["graphs"], st.session_state["experiments"])
        st.download_button("⬇️ workspace.json", data=blob, file_name="workspace.json", mime="application/json", use_container_width=True)

    if c2.button("Export experiments", use_container_width=True):
        blob = export_experiments_json(st.session_state["experiments"])
        st.download_button("⬇️ experiments.json", data=blob, file_name="experiments.json", mime="application/json", use_container_width=True)

    up_ws = st.file_uploader("Import workspace.json", type=["json"], key="sb_ws_import")
    if up_ws is not None:
        try:
            graphs_new, exps_new = import_workspace_json(up_ws.getvalue())
            st.session_state["graphs"] = graphs_new
            st.session_state["experiments"] = exps_new
            gids = sorted_graph_ids()
            st.session_state["active_graph_id"] = gids[0] if gids else None
            st.session_state["base_metrics_cache"] = {}
            st.success("Workspace импортирован.")
            st.rerun()
        except Exception as e:
            st.error(f"Ошибка импорта workspace: {e}")

    up_exps = st.file_uploader("Import experiments.json", type=["json"], key="sb_exps_import")
    if up_exps is not None:
        try:
            exps_add = import_experiments_json(up_exps.getvalue())
            st.session_state["experiments"].extend(exps_add)
            st.success("Experiments импортированы (добавлены).")
            st.rerun()
        except Exception as e:
            st.error(f"Ошибка импорта experiments: {e}")

    st.write("---")
    st.markdown("### ➕ Добавить граф")
    up_file = st.file_uploader("CSV / XLSX", type=["csv", "xlsx", "xls"], key="sb_upload")
    st.caption("Фикс-формат: (1)src (2)dst … (9)confidence (10)weight")

    # Critical: guard against rerun duplicates
    if up_file is not None:
        fp = fingerprint_upload(up_file)
        if fp != st.session_state["last_uploaded_fingerprint"]:
            try:
                df_any = load_uploaded_any(up_file.getvalue(), up_file.name)
                df_edges, meta = coerce_fixed_format(df_any)  # meta should include {"src_col":..., "dst_col":...}
                add_graph(f"Up: {up_file.name}", df_edges, "upload", meta)
                st.session_state["last_uploaded_fingerprint"] = fp
                st.session_state["base_metrics_cache"] = {}
                st.success("Граф загружен.")
                st.rerun()
            except Exception as e:
                st.error(f"Ошибка: {e}")
        else:
            st.caption("Этот файл уже добавлен (не дублирую).")

    st.write("---")
    st.markdown("### 🔎 Фильтры")
    f_conf = st.slider("Min Confidence", 0, 100, 0, 1, key="sb_f_conf")
    f_weight = st.slider("Min Weight", 0.0, 50.0, 0.0, 0.5, key="sb_f_weight")
    eff_k_base = st.slider("k для Efficiency (быстро)", 8, 128, 32, 8, key="sb_eff_k_base")
    seed_analysis = st.number_input("Seed (анализ)", value=42, step=1, key="sb_seed_analysis")

    st.write("---")
    st.markdown("### 🧹 Очистка")
    d1, d2 = st.columns(2)
    if d1.button("Clear exps", use_container_width=True):
        st.session_state["experiments"] = []
        st.rerun()
    if d2.button("Clear ALL", use_container_width=True):
        st.session_state["graphs"] = {}
        st.session_state["experiments"] = []
        st.session_state["active_graph_id"] = None
        st.session_state["last_uploaded_fingerprint"] = None
        st.session_state["mix_p_by_gid"] = {}
        st.session_state["base_metrics_cache"] = {}
        st.session_state["last_run"] = None
        st.session_state["cmp_results"] = None
        st.rerun()


# -----------------------------------------------------------------------------
# 5) Render sticky topbar and build active graph
# -----------------------------------------------------------------------------
render_sticky_topbar()

entry = get_active_graph_entry()
if not entry:
    st.info("👈 Загрузи граф или сгенерь случайный. Переключение и генерация — сверху.")
    st.stop()

graph_id = entry["id"]

G_full, ctx = build_graph_from_entry(entry, int(f_conf), float(f_weight))
G_lcc = lcc_subgraph(G_full)

# Work mode selector (physics stability default LCC, but allow whole graph)
mode = st.radio("Режим графа для анализа", ["LCC (рекомендовано)", "Весь граф"], horizontal=True, key="main_mode")
G = G_lcc if mode.startswith("LCC") else G_full

# -----------------------------------------------------------------------------
# 6) Header + base metrics (cached)
# -----------------------------------------------------------------------------
cache_key = f"{graph_id}|conf={int(f_conf)}|w={float(f_weight)}|mode={'lcc' if mode.startswith('LCC') else 'full'}"
metrics = cached_metrics_for(G, cache_key=cache_key, eff_k=int(eff_k_base), seed=int(seed_analysis))

c_title, c_kpi1, c_kpi2, c_kpi3 = st.columns([2.3, 1.1, 1.1, 1.1])
with c_title:
    st.subheader(entry["name"])
    st.caption(f"{entry['source']} | id={graph_id} | N={G.number_of_nodes()} E={G.number_of_edges()} | filters: conf≥{f_conf}, w≥{f_weight}")

with c_kpi1:
    st.metric("λ₂ (Связность)", f"{metrics.get('l2_lcc', float('nan')):.4f}")
with c_kpi2:
    st.metric("Modularity (Q)", f"{metrics.get('mod', float('nan')):.3f}")
with c_kpi3:
    st.metric("Efficiency", f"{metrics.get('eff_w', float('nan')):.3f}")

# -----------------------------------------------------------------------------
# 7) Tabs
# -----------------------------------------------------------------------------
tab_viz, tab_sim, tab_cmp, tab_mix, tab_workspace = st.tabs(
    ["👁️ Анатомия", "🔥 Краш-тест", "⚔️ Head-to-Head", "🧪 Mixing Lab", "📦 Workspace"]
)

# --- TAB: VISUALIZATION ---
with tab_viz:
    col_3d, col_info = st.columns([3, 1])

    with col_3d:
        st.markdown("### 3D")
        if G.number_of_nodes() == 0:
            st.info("Пустой граф после фильтров.")
        elif G.number_of_nodes() > 2500:
            st.warning("Граф большой, 3D может тормозить.")
            if st.button("Показать 3D всё равно", key="viz_force_3d"):
                pos = compute_3d_layout(G, int(seed_analysis))
                e_tr, n_tr = make_3d_traces(G, pos, True)
                fig = go.Figure([e_tr, n_tr])
                fig.update_layout(template="plotly_dark", margin=dict(t=0, b=0, l=0, r=0), height=540)
                st.plotly_chart(fig, use_container_width=True)
        else:
            pos = compute_3d_layout(G, int(seed_analysis))
            e_tr, n_tr = make_3d_traces(G, pos, True)
            fig = go.Figure([e_tr, n_tr])
            fig.update_layout(template="plotly_dark", margin=dict(t=0, b=0, l=0, r=0), height=540)
            st.plotly_chart(fig, use_container_width=True)

    with col_info:
        st.markdown("### Свойства")
        st.write(f"**Assortativity:** {metrics.get('assortativity', float('nan')):.3f}")
        st.write(f"**Clustering:** {metrics.get('clustering', float('nan')):.3f}")
        st.write(f"**Diameter (~):** {metrics.get('diameter_approx', '?')}")
        st.write(f"**Density:** {metrics.get('density', float('nan')):.5f}")
        st.write(f"**Components:** {metrics.get('C', '?')}")

        st.info(
            "**λ₂ (Fiedler value)**: скорость диффузии/связность. Низкая — граф легко распадается.\n\n"
            "**Q (modularity)**: клановость/сообщества. Высокая — сильные группы."
        )

        with st.expander("Сводка графа (text)", expanded=False):
            try:
                st.code(graph_summary(G), language="text")
            except Exception:
                st.code(f"N={G.number_of_nodes()} E={G.number_of_edges()}", language="text")


# --- TAB: CRASH TEST ---
with tab_sim:
    c_set, c_res = st.columns([1.1, 2.0])

    with c_set:
        st.markdown("### Сценарий атаки")

        mode_sim = st.radio("Режим", ["Быстрый", "Кастомный"], horizontal=True, key="sim_mode")
        if mode_sim == "Быстрый":
            attack_type = st.selectbox("Стратегия", ["random", "degree", "betweenness", "richclub_top"], key="sim_attack_fast")
            steps = 20
            frac = 0.5
            eff_k = 16
            heavy_every = 2
        else:
            attack_type = st.selectbox(
                "Стратегия",
                ["random", "degree", "betweenness", "kcore", "richclub_top", "richclub_density"],
                key="sim_attack_custom",
            )
            frac = st.slider("% Уничтожения", 0.1, 0.9, 0.5, 0.05, key="sim_frac")
            steps = st.slider("Шаги", 5, 120, 30, 1, key="sim_steps")
            eff_k = st.slider("Точность Efficiency (k)", 8, 128, 32, 8, key="sim_eff_k")
            heavy_every = st.slider("Тяжёлые метрики каждые k шагов", 1, 10, 1, 1, key="sim_heavy_every")

        exp_tag = st.text_input("Тег эксперимента", value="", key="sim_tag")
        btn_run = st.button("💀 ЗАПУСТИТЬ", type="primary", use_container_width=True, key="sim_run")

    with c_res:
        if btn_run:
            if G.number_of_nodes() < 2:
                st.warning("Граф слишком маленький.")
            else:
                with st.spinner("Ломаем систему..."):
                    df_hist, _ = run_attack(
                        G,
                        attack_kind=attack_type,
                        remove_frac=float(frac),
                        steps=int(steps),
                        seed=int(seed_analysis),
                        eff_sources_k=int(eff_k),
                        rc_frac=0.1,
                        rc_min_density=0.3,
                        rc_max_frac=0.3,
                        compute_heavy_every=int(heavy_every),
                        keep_states=False,
                    )

                    phase = classify_phase_transition(df_hist)

                    exp_name = f"{attack_type} on {entry['name']}"
                    if exp_tag.strip():
                        exp_name += f" · {exp_tag.strip()}"

                    push_experiment(
                        name=exp_name,
                        graph_id=graph_id,
                        attack_kind=attack_type,
                        params={
                            "remove_frac": float(frac),
                            "steps": int(steps),
                            "seed": int(seed_analysis),
                            "eff_k": int(eff_k),
                            "heavy_every": int(heavy_every),
                            "mode": mode,
                            "filters": {"min_conf": int(f_conf), "min_weight": float(f_weight)},
                            "phase": phase,
                        },
                        df_hist=df_hist,
                    )

                    st.session_state["last_run"] = {"df": df_hist, "phase": phase, "name": exp_name}

        if st.session_state.get("last_run"):
            res = st.session_state["last_run"]
            df_h = res["df"]
            ph = res["phase"]

            st.success(f"Результат: {res['name']}")
            p1, p2, p3 = st.columns([1.2, 1.0, 1.0])
            p1.metric("Тип перехода", "Взрывной (1-го рода)" if ph.get("is_abrupt") else "Плавный (2-го рода)")
            p2.metric("Критическая точка", f"{ph.get('critical_x', 0.0):.2%} removed")
            p3.metric("Скачок", f"{ph.get('jump', 0.0):.3f}")

            t_dyn, t_phase = st.tabs(["📉 Динамика", "🌀 Фазовый портрет"])
            with t_dyn:
                fig = fig_metrics_over_steps(df_h, title="Деградация метрик")
                st.plotly_chart(fig, use_container_width=True)
            with t_phase:
                fig_p = plot_phase_space(df_h)
                st.plotly_chart(fig_p, use_container_width=True)
                st.caption("Форма траектории и наличие скачка — быстрый маркер типа распада.")


# --- TAB: HEAD TO HEAD ---
with tab_cmp:
    st.markdown("### Сравнение стратегий (Head-to-Head)")
    st.caption("Запускает 4 сценария и сравнивает траектории. Результаты сохраняются как experiments.")

    colA, colB = st.columns([1.1, 1.9])
    with colA:
        h_frac = st.slider("remove_frac", 0.1, 0.9, 0.5, 0.05, key="h_frac")
        h_steps = st.slider("steps", 5, 120, 25, 1, key="h_steps")
        h_eff = st.slider("eff_k", 8, 128, 16, 8, key="h_eff")
        h_heavy = st.slider("heavy_every", 1, 10, 2, 1, key="h_heavy")
        h_tag = st.text_input("Тег", value="H2H", key="h_tag")

        run_h2h = st.button("⚔️ FIGHT", use_container_width=True, key="h_run")

    with colB:
        if run_h2h:
            if G.number_of_nodes() < 2:
                st.warning("Граф слишком маленький.")
            else:
                with st.spinner("Прогон сценариев..."):
                    scenarios = [
                        ("random", "Random Failure"),
                        ("degree", "Hubs Attack"),
                        ("betweenness", "Bridges Cut"),
                        ("richclub_top", "Rich-Club"),
                    ]

                    results = []
                    for kind, label in scenarios:
                        df, _ = run_attack(
                            G,
                            attack_kind=kind,
                            remove_frac=float(h_frac),
                            steps=int(h_steps),
                            seed=int(seed_analysis),
                            eff_sources_k=int(h_eff),
                            rc_frac=0.1,
                            rc_min_density=0.3,
                            rc_max_frac=0.3,
                            compute_heavy_every=int(h_heavy),
                            keep_states=False,
                        )
                        exp_name = f"{h_tag.strip() or 'H2H'}:{label} on {entry['name']}"
                        push_experiment(
                            name=exp_name,
                            graph_id=graph_id,
                            attack_kind=kind,
                            params={
                                "remove_frac": float(h_frac),
                                "steps": int(h_steps),
                                "seed": int(seed_analysis),
                                "eff_k": int(h_eff),
                                "heavy_every": int(h_heavy),
                                "mode": mode,
                                "filters": {"min_conf": int(f_conf), "min_weight": float(f_weight)},
                            },
                            df_hist=df,
                        )
                        results.append((label, df))

                    st.session_state["cmp_results"] = results

        if st.session_state.get("cmp_results"):
            res_list = st.session_state["cmp_results"]
            fig_lcc = fig_compare_attacks(res_list, "removed_frac", "lcc_frac", "Живучесть сети (LCC)")
            st.plotly_chart(fig_lcc, use_container_width=True)

            if all(("eff_w" in df.columns) for _, df in res_list if df is not None and not df.empty):
                fig_eff = fig_compare_attacks(res_list, "removed_frac", "eff_w", "Падение эффективности")
                st.plotly_chart(fig_eff, use_container_width=True)

    st.write("---")
    st.markdown("### Сравнить графы (скаляры)")
    graphs = st.session_state["graphs"]
    gids = sorted_graph_ids()
    pick = st.multiselect(
        "Графы",
        options=gids,
        default=[graph_id],
        format_func=lambda gid: graphs[gid]["name"],
        key="cmp_graphs_pick",
    )
    scalar = st.selectbox(
        "Скаляр",
        ["N", "E", "density", "eff_w", "l2_lcc", "mod", "lmax", "entropy_deg", "assortativity", "clustering"],
        key="cmp_scalar_pick",
    )
    if pick:
        rows = []
        for gid in pick:
            e2 = graphs[gid]
            G2_full, _ = build_graph_from_entry(e2, int(f_conf), float(f_weight))
            G2 = lcc_subgraph(G2_full) if mode.startswith("LCC") else G2_full
            m2 = calculate_metrics(G2, eff_sources_k=int(eff_k_base), seed=int(seed_analysis))
            rows.append({"graph": e2["name"], "source": e2["source"], scalar: m2.get(scalar)})
        df_cmp = pd.DataFrame(rows)
        fig = fig_compare_graphs_scalar(df_cmp, x="graph", y=scalar, title=f"Graphs compare: {scalar}")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_cmp, use_container_width=True)

    st.write("---")
    st.markdown("### Сравнить эксперименты (траектории)")
    exps = st.session_state["experiments"]
    if not exps:
        st.info("Нет experiments. Запусти краш-тест или head-to-head.")
    else:
        exp_by_id = {e["id"]: e for e in exps}
        exp_ids = [e["id"] for e in exps]
        default_ids = exp_ids[-4:] if len(exp_ids) >= 4 else exp_ids

        chosen = st.multiselect(
            "Experiments",
            options=exp_ids,
            default=default_ids,
            format_func=lambda eid: exp_by_id[eid]["name"],
            key="cmp_exp_pick",
        )
        y_metric = st.selectbox("Y", ["lcc_frac", "eff_w", "mod", "l2_lcc", "lmax"], index=0, key="cmp_exp_y")

        curves = []
        for eid in chosen:
            e = exp_by_id[eid]
            dfh = e["history"]
            if isinstance(dfh, pd.DataFrame) and not dfh.empty and y_metric in dfh.columns:
                curves.append((e["name"], dfh))

        if curves:
            fig = fig_compare_attacks(curves, "removed_frac", y_metric, f"Compare experiments: {y_metric}")
            st.plotly_chart(fig, use_container_width=True)


# --- TAB: MIXING LAB ---
with tab_mix:
    st.markdown("### 🧪 Смешивание (Rewiring)")
    st.caption("Создаёт новый граф в workspace (анализируется как обычный).")

    c_p, c_btn = st.columns([3, 1])
    with c_p:
        p_val = st.slider(
            "Коэффициент хаоса (p)",
            0.0,
            1.0,
            0.1,
            0.05,
            help="0 = оригинал, 1 = сильная рандомизация (rewire)",
            key="mix_p",
        )

    with c_btn:
        st.write("")
        if st.button("Создать мутанта", key="mix_make", use_container_width=True):
            with st.spinner("Rewiring..."):
                # mix from CURRENT analysis graph G (LCC or full), but keep naming based on entry
                G_mut = rewire_mix(G, p=float(p_val), seed=int(time.time()))

                tags = entry["tags"]
                src_col = tags.get("src_col", "src")
                dst_col = tags.get("dst_col", "dst")

                edges = [[u, v, 1.0, 1.0] for u, v in G_mut.edges()]
                df_mut = pd.DataFrame(edges, columns=[src_col, dst_col, "weight", "confidence"])

                new_name = f"Mix(p={p_val:.2f}) of {entry['name']}"
                add_graph(new_name, df_mut, "mix", tags=tags)
                st.success(f"Создан граф: {new_name}")
                st.rerun()

    st.info(
        "Идея: проверить, насколько твой граф уникален по сравнению со случайной сетью, "
        "когда «смысловые» связи постепенно заменяются случайными."
    )


# --- TAB: WORKSPACE ---
with tab_workspace:
    st.markdown("### 📦 Workspace: графы и эксперименты")
    graphs = st.session_state["graphs"]
    exps = st.session_state["experiments"]

    st.markdown("#### Графы")
    gids = sorted_graph_ids()
    if not gids:
        st.info("Пусто.")
    else:
        for gid in gids[-12:][::-1]:
            e = graphs[gid]
            st.markdown(
                f"""
<div class="active-graph-box">
  <div style="display:flex;justify-content:space-between;gap:12px;">
    <div>
      <div style="font-weight:700;">{e["name"]}</div>
      <div style="opacity:0.75;font-size:0.85rem;">id={gid} · source={e["source"]}</div>
    </div>
    <div style="opacity:0.75;font-size:0.85rem;">edges rows={len(e["edges"])}</div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

    st.markdown("#### Experiments")
    if not exps:
        st.info("Нет experiments.")
    else:
        for e in exps[-12:][::-1]:
            st.markdown(
                f"- **{e['name']}**  \n"
                f"  graph_id={e['graph_id']} · kind={e['attack_kind']} · created={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(e['created_at']))}"
            )
