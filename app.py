import io
import time
import uuid
import random
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

# =========================
# Page
# =========================
st.set_page_config(page_title="прикольчик", layout="wide", page_icon="💀")
st.title("прикольчик")

# =========================
# Session State
# =========================
def _init_state():
    """Initialize Streamlit session state keys used by the app."""
    if "graphs" not in st.session_state:
        # graphs: dict[graph_id] -> dict(meta + edges_df cached + node/edge counts + seed tags)
        st.session_state["graphs"] = {}

    if "experiments" not in st.session_state:
        # list of dict: {id, name, graph_id, attack_kind, params, history_df(json), ts}
        st.session_state["experiments"] = []

    if "active_graph_id" not in st.session_state:
        st.session_state["active_graph_id"] = None

    if "last_upload_name" not in st.session_state:
        st.session_state["last_upload_name"] = None


_init_state()

# =========================
# Helpers
# =========================
def new_id(prefix: str) -> str:
    """Generate a short unique id with a prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def add_graph_to_workspace(name: str, df_edges: pd.DataFrame, source: str, tags: dict | None = None) -> str:
    """Persist a cleaned edge table into workspace storage and set it active."""
    gid = new_id("G")
    tags = tags or {}
    # store edges (already cleaned & with columns: SRC_COL, DST_COL, weight, confidence)
    st.session_state["graphs"][gid] = {
        "id": gid,
        "name": name,
        "source": source,
        "tags": tags,
        "edges": df_edges.copy(),
        "created_at": time.time(),
    }
    st.session_state["active_graph_id"] = gid
    return gid


def get_active_graph_entry():
    """Return the active graph entry from session state."""
    gid = st.session_state.get("active_graph_id")
    if gid is None:
        return None
    return st.session_state["graphs"].get(gid)


def build_active_graph(filtered_conf: int, filtered_weight: float):
    """Filter and build a NetworkX graph for the active graph entry."""
    entry = get_active_graph_entry()
    if entry is None:
        return None, None

    df_edges = entry["edges"]
    # detect src/dst cols by fixed-format expectation in our pipeline
    src_col = entry["tags"].get("src_col", None)
    dst_col = entry["tags"].get("dst_col", None)

    if src_col is None or dst_col is None:
        # fallback: first two cols
        src_col = df_edges.columns[0]
        dst_col = df_edges.columns[1]

    df_f = filter_edges(
        df_edges,
        src_col=src_col,
        dst_col=dst_col,
        min_conf=filtered_conf,
        min_weight=filtered_weight,
    )

    G = build_graph_from_edges(df_f, src_col=src_col, dst_col=dst_col)
    return G, {"src_col": src_col, "dst_col": dst_col, "df_filtered": df_f}


def list_graphs_ui():
    """Render active graph selector in the sidebar."""
    graphs = st.session_state["graphs"]
    if not graphs:
        st.info("Workspace пустой. Добавь граф загрузкой или генерацией.")
        return None

    items = []
    for gid, ge in graphs.items():
        items.append((gid, f"{ge['name']}  ·  ({ge['source']})"))

    # keep stable ordering by created_at
    items = sorted(items, key=lambda x: graphs[x[0]].get("created_at", 0.0))

    gid = st.selectbox(
        "Активный граф",
        [x[0] for x in items],
        index=(
            [x[0] for x in items].index(st.session_state["active_graph_id"])
            if st.session_state["active_graph_id"] in [x[0] for x in items]
            else 0
        ),
        format_func=lambda g: dict(items).get(g, g),
    )
    st.session_state["active_graph_id"] = gid
    return gid


def push_experiment(name: str, graph_id: str, attack_kind: str, params: dict, df_hist: pd.DataFrame):
    """Store an experiment history in the workspace."""
    exp_id = new_id("EXP")
    obj = {
        "id": exp_id,
        "name": name,
        "graph_id": graph_id,
        "attack_kind": attack_kind,
        "params": params,
        "history": df_hist.copy(),
        "created_at": time.time(),
    }
    st.session_state["experiments"].append(obj)
    return exp_id


def experiments_for_graph(graph_id: str):
    """Filter experiments belonging to a particular graph id."""
    return [e for e in st.session_state["experiments"] if e.get("graph_id") == graph_id]


# =========================
# Sidebar: Workspace I/O + Add graphs
# =========================
with st.sidebar:
    st.header("🧠 Workspace")

    # ---- Workspace import/export
    with st.expander("💾 Import / Export", expanded=False):
        c1, c2 = st.columns(2)

        # Export workspace JSON
        if c1.button("Export workspace"):
            blob = export_workspace_json(st.session_state["graphs"], st.session_state["experiments"])
            st.download_button(
                "⬇️ скачать workspace.json",
                data=blob,
                file_name="workspace.json",
                mime="application/json",
                use_container_width=True,
            )

        # Export experiments only
        if c2.button("Export experiments"):
            blob = export_experiments_json(st.session_state["experiments"])
            st.download_button(
                "⬇️ скачать experiments.json",
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
                # set active to first if any
                if graphs_new:
                    first = sorted(list(graphs_new.keys()), key=lambda k: graphs_new[k].get("created_at", 0.0))[0]
                    st.session_state["active_graph_id"] = first
                else:
                    st.session_state["active_graph_id"] = None
                st.success("Workspace импортирован.")
            except Exception as e:
                st.error(f"Ошибка импорта workspace: {e}")

        up_exps = st.file_uploader("Import experiments.json", type=["json"], key="exps_import")
        if up_exps is not None:
            try:
                exps_add = import_experiments_json(up_exps.getvalue())
                # merge (keep existing)
                st.session_state["experiments"].extend(exps_add)
                st.success("Experiments импортированы (добавлены).")
            except Exception as e:
                st.error(f"Ошибка импорта experiments: {e}")

    st.write("---")

    # ---- Add graph: upload
    st.header("📎 Добавить граф")
    uploaded = st.file_uploader(
        "Загрузите CSV/Excel (фикс. формат)",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=False,
        key="file_uploader_main",
    )
    st.caption(
        "Ожидается формат: 1-я колонка=source id, 2-я=target id, 9-я=confidence, 10-я=weight."
    )

    if uploaded is not None:
        try:
            df_any = load_uploaded_any(uploaded.getvalue(), uploaded.name)
            df_edges, meta = coerce_fixed_format(df_any)
            gname = f"uploaded:{uploaded.name}"
            add_graph_to_workspace(
                name=gname,
                df_edges=df_edges,
                source="upload",
                tags=meta,
            )
            st.session_state["last_upload_name"] = uploaded.name
            st.success(f"Добавлен граф: {gname}")
        except Exception as e:
            st.error(f"Ошибка загрузки/парсинга: {e}")

    st.write("---")

    # ---- Null models / mixing
    st.header("🧪 Null Models & Mixing")

    gid_active = list_graphs_ui()
    entry_active = get_active_graph_entry()

    if entry_active is None:
        st.stop()

    # build base graph for generation sizing
    # We use full edges without extra filters here; filters are analysis-level.
    df_edges_base = entry_active["edges"]
    src_col = entry_active["tags"].get("src_col", df_edges_base.columns[0])
    dst_col = entry_active["tags"].get("dst_col", df_edges_base.columns[1])
    G_base_full = build_graph_from_edges(df_edges_base, src_col=src_col, dst_col=dst_col)
    N0 = G_base_full.number_of_nodes()
    E0 = G_base_full.number_of_edges()

    st.caption(f"База для генерации: N={N0}, E={E0}")

    gen_kind = st.selectbox("Что создать", ["ER G(n,m)", "Configuration (degree-preserving-ish)", "Mix/Rewire (p)"])

    gen_seed = st.number_input("Seed (генерация)", value=42, step=1, key="seed_gen")
    if gen_kind == "Mix/Rewire (p)":
        p = st.slider("Randomness p (0=оригинал, 1=хаос)", 0.0, 1.0, 0.2, 0.05)
    else:
        p = 0.0

    if st.button("➕ Создать и добавить в Workspace", use_container_width=True):
        try:
            if gen_kind == "ER G(n,m)":
                G_null = make_er_gnm(N0, E0, seed=int(gen_seed))
                name = f"ER(n={N0},m={E0},seed={gen_seed})"
                source = "null:ER"

            elif gen_kind == "Configuration (degree-preserving-ish)":
                G_null = make_configuration_model(G_base_full, seed=int(gen_seed))
                name = f"CFG(deg,seed={gen_seed})"
                source = "null:CFG"

            else:
                G_null = rewire_mix(G_base_full, p=float(p), seed=int(gen_seed))
                name = f"MIX(p={p:.2f},seed={gen_seed})"
                source = "mix:rewire"

            # convert to edges df with weight=1 confidence=1 (or preserve if possible)
            edges = []
            for u, v in G_null.edges():
                edges.append([u, v, 1.0, 1.0])
            df_new = pd.DataFrame(edges, columns=[src_col, dst_col, "weight", "confidence"])

            add_graph_to_workspace(
                name=name,
                df_edges=df_new,
                source=source,
                tags={"src_col": src_col, "dst_col": dst_col},
            )
            st.success(f"Добавлен граф: {name}")
        except Exception as e:
            st.error(f"Не смогла создать граф: {e}")

    st.write("---")

    # ---- Global analysis controls
    st.header("🎛️ Анализ активного графа")

    # sliders depend on edges weights/conf, handle empty gracefully
    df_e = entry_active["edges"]
    max_conf = int(pd.to_numeric(df_e.get("confidence", pd.Series([0])), errors="coerce").max() or 0)
    max_w = float(pd.to_numeric(df_e.get("weight", pd.Series([0.0])), errors="coerce").max() or 0.0)

    min_conf = st.slider(
        "Порог confidence",
        0,
        max_conf if max_conf > 0 else 0,
        min(100, max_conf) if max_conf > 0 else 0,
    )
    min_weight = st.number_input(
        "Мин. вес weight",
        0.0,
        max_w if max_w > 0 else 0.0,
        0.0,
        step=0.1,
    )

    eff_sources_k = st.slider("Efficiency k (аппрокс)", 8, 256, 64, 8)
    seed_analysis = st.number_input("Seed (анализ/3D/метрики)", value=42, step=1)

# =========================
# Build Active Graph
# =========================
G, ctx = build_active_graph(filtered_conf=int(min_conf), filtered_weight=float(min_weight))
if G is None:
    st.stop()

graph_id = st.session_state["active_graph_id"]
graph_entry = get_active_graph_entry()

# optional: mode global vs LCC
mode = st.radio("Масштаб", ["Глобальный (весь граф)", "LCC (гигантская компонента)"], horizontal=True)
if mode.startswith("LCC"):
    G_view = lcc_subgraph(G)
else:
    G_view = G

# =========================
# Top bar: Graph list and quick summary
# =========================
top1, top2, top3, top4 = st.columns([2.3, 1.2, 1.2, 1.3])
with top1:
    st.subheader(f"Активный граф: {graph_entry['name']}")
    st.caption(f"source={graph_entry['source']} · graph_id={graph_id}")

with top2:
    st.metric("Nodes", int(G_view.number_of_nodes()))
with top3:
    st.metric("Edges", int(G_view.number_of_edges()))
with top4:
    st.metric("Components", int(nx.number_connected_components(G_view)) if G_view.number_of_nodes() else 0)

# =========================
# Base metrics
# =========================
base = calculate_metrics(G_view, eff_sources_k=int(eff_sources_k), seed=int(seed_analysis))

# =========================
# Tabs
# =========================
t1, t2, t3, t4, t5 = st.tabs(
    [
        "📊 База (метрики)",
        "🧬 Спектр + сложность",
        "👁️ 3D",
        "💥 ATTACK LAB",
        "🆚 Сравнение графов/экспериментов",
    ]
)

# -------------------------
# Tab 1: metrics dashboard
# -------------------------
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

# -------------------------
# Tab 2: Spectrum + complexity + interpretation
# -------------------------
with t2:
    left, right = st.columns([1.2, 1.0])

    with left:
        st.subheader("λ₂ / Q интерпретация (текущая точка)")
        l2 = base["l2_lcc"]
        Q = base["mod"]

        st.write(
            f"- λ₂ (LCC) = {l2:.6g}\n"
            f"- Q (Louvain) = {Q:.6g}\n"
        )

        st.markdown(
            "**Грубая эвристика:**\n"
            "- λ₂ низкая → сеть легко рассоединяется (глобальная связность слабая)\n"
            "- Q высокая → сеть распадается на модули (сегрегация)\n"
            "- Q низкая → структура деградирует без выраженных кластеров\n"
        )

    with right:
        st.subheader("Сложность / структура")
        st.write(f"- Degree-entropy H = {base['entropy_deg']:.6g}")
        st.write(f"- Assortativity r = {base['assortativity']:.6g}")
        st.write(f"- Clustering C̄ = {base['clustering']:.6g}")
        st.write(f"- Diameter (LCC, approx) = {base['diameter_approx']}")

    st.write("---")
    with st.expander("Диагностика (почему λ₂ может быть 0)", expanded=False):
        st.write(f"N={base['N']} E={base['E']} C={base['C']}")
        H = lcc_subgraph(G_view)
        st.write(
            "LCC: "
            f"n={H.number_of_nodes()} e={H.number_of_edges()} "
            f"connected={nx.is_connected(H) if H.number_of_nodes() else False}"
        )

# -------------------------
# Tab 3: 3D
# -------------------------
with t3:
    st.subheader("3D Projection (color = strength)")

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

# -------------------------
# Tab 4: Attack Lab
# -------------------------
with t4:
    st.subheader("💥 Лаборатория разрушения")
    st.caption("Результаты сохраняются в Workspace → можно сравнивать между собой и экспортировать.")

    with st.expander("Настройки симуляции", expanded=True):
        c1, c2, c3, c4 = st.columns(4)

        attack_kind_ui = c1.selectbox(
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
        remove_frac = c2.slider("Удалить долю узлов", 0.05, 0.95, 0.50, 0.05)
        steps = c3.slider("Шагов", 5, 200, 40)
        attack_seed = c4.number_input("Seed (атака)", value=int(seed_analysis), step=1)

        st.write("—")
        cc1, cc2, cc3, cc4 = st.columns(4)
        eff_k_sim = cc1.slider("Efficiency k (sim)", 8, 256, int(eff_sources_k), 8)
        compute_heavy_every = cc2.slider("Heavy metrics every k steps", 1, 10, 1, 1)
        keep_states_3d = cc3.checkbox("Хранить states для 3D replay", value=False)
        name_tag = cc4.text_input("Название эксперимента (tag)", value="")

        # RC params
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

    run_col1, run_col2 = st.columns([1.0, 1.0])
    with run_col1:
        run_one = st.button("▶️ Запустить (1 сценарий)", use_container_width=True)

    with run_col2:
        run_all = st.button("⚔️ Run head-to-head (4 атаки)", use_container_width=True)

    def _run_and_render(attack_kind_local: str, title_prefix: str):
        """Run a simulation and render plots, storing results in session."""
        if G_view.number_of_nodes() < 2:
            st.warning("Граф слишком маленький.")
            return None

        df_hist, states = run_attack(
            G_view,
            attack_kind=attack_kind_local,
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

        # phase classifier on LCC fraction
        phase = classify_phase_transition(df_hist, x_col="removed_frac", y_col="lcc_frac")

        label = title_prefix
        if name_tag.strip():
            label += f" · {name_tag.strip()}"

        push_experiment(
            name=label,
            graph_id=graph_id,
            attack_kind=attack_kind_local,
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
                "active_graph_name": graph_entry["name"],
                "phase": phase,
            },
            df_hist=df_hist,
        )

        st.success(f"Готово: {label}")
        st.caption(
            f"Phase check: abrupt={phase['is_abrupt']} · critical~{phase['critical_x']:.3f} "
            f"· jump={phase['jump']:.3f}"
        )

        # render quick plots
        fig = fig_metrics_over_steps(df_hist, title=f"{label} — базовые кривые")
        st.plotly_chart(fig, use_container_width=True)

        return df_hist

    if run_one:
        _run_and_render(attack_kind, f"{attack_kind_ui}")

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
            st.subheader("Сводное сравнение (LCC fraction)")
            fig_cmp = fig_compare_attacks(
                results,
                x_col="removed_frac",
                y_col="lcc_frac",
                title="Head-to-head: LCC fraction",
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

# -------------------------
# Tab 5: Comparison UI
# -------------------------
with t5:
    st.subheader("🆚 Сравнение")

    graphs = st.session_state["graphs"]
    exps = st.session_state["experiments"]

    if not graphs:
        st.info("Нет графов в Workspace.")
        st.stop()

    st.write("### Сравнить графы (скаляры)")
    graph_ids = list(graphs.keys())
    # stable ordering by created_at
    graph_ids = sorted(graph_ids, key=lambda k: graphs[k].get("created_at", 0.0))

    selected_graphs = st.multiselect(
        "Выбери графы",
        options=graph_ids,
        default=[st.session_state["active_graph_id"]] if st.session_state["active_graph_id"] in graph_ids else [],
        format_func=lambda gid: f"{graphs[gid]['name']} ({graphs[gid]['source']})",
    )

    scalar = st.selectbox(
        "Скаляр для сравнения",
        [
            "N",
            "E",
            "density",
            "eff_w",
            "l2_lcc",
            "mod",
            "lmax",
            "entropy_deg",
            "assortativity",
            "clustering",
        ],
    )

    if selected_graphs:
        # compute metrics per selected graph (with current filters!)
        rows = []
        for gid in selected_graphs:
            gentry = graphs[gid]
            # build graph with current filters but per-graph
            df_edges = gentry["edges"]
            src_col = gentry["tags"].get("src_col", df_edges.columns[0])
            dst_col = gentry["tags"].get("dst_col", df_edges.columns[1])
            df_f = filter_edges(df_edges, src_col, dst_col, min_conf=int(min_conf), min_weight=float(min_weight))
            Gi = build_graph_from_edges(df_f, src_col, dst_col)
            Gi_view = lcc_subgraph(Gi) if mode.startswith("LCC") else Gi
            met = calculate_metrics(Gi_view, eff_sources_k=int(eff_sources_k), seed=int(seed_analysis))
            rows.append(
                {
                    "graph": gentry["name"],
                    "source": gentry["source"],
                    scalar: met.get(scalar),
                }
            )

        df_cmp = pd.DataFrame(rows)
        fig = fig_compare_graphs_scalar(df_cmp, x="graph", y=scalar, title=f"Graphs compare: {scalar}")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_cmp, use_container_width=True)

    st.write("---")
    st.write("### Сравнить эксперименты (траектории)")
    if not exps:
        st.info("Нет экспериментов. Запусти атаку в ATTACK LAB.")
        st.stop()

    # choose subset
    exp_ids = [e["id"] for e in exps]
    exp_by_id = {e["id"]: e for e in exps}
    selected_exps = st.multiselect(
        "Эксперименты",
        options=exp_ids,
        default=exp_ids[-3:] if len(exp_ids) >= 3 else exp_ids,
        format_func=lambda eid: (
            f"{exp_by_id[eid]['name']} (graph={graphs.get(exp_by_id[eid]['graph_id'], {}).get('name','?')})"
        ),
    )

    y_metric = st.selectbox("Y", ["lcc_frac", "mod", "l2_lcc", "eff_w", "lmax"], index=0)

    curves = []
    for eid in selected_exps:
        e = exp_by_id[eid]
        dfh = e["history"]
        if isinstance(dfh, pd.DataFrame) and not dfh.empty:
            curves.append((e["name"], dfh))

    if curves:
        fig = fig_compare_attacks(curves, x_col="removed_frac", y_col=y_metric, title=f"Compare experiments: {y_metric}")
        st.plotly_chart(fig, use_container_width=True)

    st.write("---")
    st.write("### Управление Workspace")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        if st.button("🧹 Очистить experiments", use_container_width=True):
            st.session_state["experiments"] = []
            st.success("Experiments очищены.")
    with cc2:
        if st.button("🗑️ Удалить активный граф", use_container_width=True):
            gid = st.session_state["active_graph_id"]
            if gid in st.session_state["graphs"]:
                st.session_state["graphs"].pop(gid)
                # also delete experiments referencing it
                st.session_state["experiments"] = [
                    e for e in st.session_state["experiments"] if e.get("graph_id") != gid
                ]
                # set new active
                if st.session_state["graphs"]:
                    new_active = sorted(
                        list(st.session_state["graphs"].keys()),
                        key=lambda k: st.session_state["graphs"][k].get("created_at", 0.0),
                    )[0]
                    st.session_state["active_graph_id"] = new_active
                else:
                    st.session_state["active_graph_id"] = None
                st.success("Активный граф удалён.")
            else:
                st.warning("Нет активного графа.")
    with cc3:
        if st.button("🧨 Очистить всё", use_container_width=True):
            st.session_state["graphs"] = {}
            st.session_state["experiments"] = []
            st.session_state["active_graph_id"] = None
            st.success("Workspace очищен.")
