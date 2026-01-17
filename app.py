import time
import uuid
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

# =========================
# Session State
# =========================
def _init_state():
    if "graphs" not in st.session_state:
        st.session_state["graphs"] = {}  # gid -> {id,name,source,tags,edges,created_at}
    if "experiments" not in st.session_state:
        st.session_state["experiments"] = []  # list[{id,name,graph_id,...}]
    if "active_graph_id" not in st.session_state:
        st.session_state["active_graph_id"] = None
    if "last_upload_name" not in st.session_state:
        st.session_state["last_upload_name"] = None
    if "mix_p_by_gid" not in st.session_state:
        st.session_state["mix_p_by_gid"] = {}  # base_gid -> current p
    if "seed_top" not in st.session_state:
        st.session_state["seed_top"] = 42


_init_state()

# =========================
# Helpers
# =========================
def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def add_graph_to_workspace(name: str, df_edges: pd.DataFrame, source: str, tags: dict | None = None) -> str:
    gid = new_id("G")
    tags = tags or {}
    st.session_state["graphs"][gid] = {
        "id": gid,
        "name": name,
        "source": source,
        "tags": tags,  # expects {"src_col":..., "dst_col":...}
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


# IMPORTANT:
# Do NOT access a global `meta` at import time.
# File parsing produces `meta` only AFTER upload.
# We store src/dst col names in graph["tags"] (lowercase keys).


# =========================
# Top Graph Panel (tabs-like) + generator + mix
# =========================
def _sorted_graph_ids():
    graphs = st.session_state["graphs"]
    return sorted(list(graphs.keys()), key=lambda k: graphs[k].get("created_at", 0.0))


def render_top_graph_panel():
    graphs = st.session_state["graphs"]
    if not graphs:
        st.info("Workspace пустой: загрузите CSV/Excel в sidebar или создайте null-graph ниже.")
        return None

    gids = _sorted_graph_ids()
    if st.session_state["active_graph_id"] not in graphs:
        st.session_state["active_graph_id"] = gids[0]

    active_gid = st.session_state["active_graph_id"]

    # --- "Tabs" row (radio horizontal)
    # If too many graphs, fallback to selectbox to avoid UI overflow.
    MAX_TABS = 10
    row1, row2 = st.columns([3.2, 2.8])

    with row1:
        if len(gids) <= MAX_TABS:
            labels = [graphs[gid]["name"] for gid in gids]
            idx = gids.index(active_gid)
            picked_label = st.radio(
                "Графы",
                options=list(range(len(gids))),
                index=idx,
                horizontal=True,
                format_func=lambda i: labels[i],
                key="top_graph_tabs_radio",
            )
            picked_gid = gids[int(picked_label)]
        else:
            picked_gid = st.selectbox(
                "Граф",
                options=gids,
                index=gids.index(active_gid),
                format_func=lambda gid: f"{graphs[gid]['name']} · ({graphs[gid]['source']})",
                key="top_graph_selectbox",
            )

        if picked_gid != active_gid:
            st.session_state["active_graph_id"] = picked_gid
            st.rerun()

    entry = graphs[st.session_state["active_graph_id"]]
    df_edges_base = entry["edges"]
    src_col = entry["tags"].get("src_col", df_edges_base.columns[0])
    dst_col = entry["tags"].get("dst_col", df_edges_base.columns[1])
    G_base_full = build_graph_from_edges(df_edges_base, src_col=src_col, dst_col=dst_col)
    N0 = G_base_full.number_of_nodes()
    E0 = G_base_full.number_of_edges()

    with row2:
        st.caption(f"Активный: {entry['name']} · source={entry['source']} · N={N0} E={E0}")

        # Rename (tab label = graph name)
        r1, r2 = st.columns([2.2, 1.0])
        with r1:
            new_name = st.text_input(
                "Переименовать вкладку",
                value=entry["name"],
                key="top_rename_input",
                label_visibility="collapsed",
                placeholder="Имя вкладки/графа",
            )
        with r2:
            if st.button("Rename", use_container_width=True):
                new_name = (new_name or "").strip()
                if new_name:
                    st.session_state["graphs"][entry["id"]]["name"] = new_name
                    st.rerun()

    # --- Generator + mix row
    gen1, gen2, gen3, gen4 = st.columns([1.2, 1.2, 1.6, 2.0])

    with gen1:
        seed = st.number_input("Seed", value=int(st.session_state["seed_top"]), step=1, key="seed_top_input")
        st.session_state["seed_top"] = int(seed)

    with gen2:
        if st.button("➕ ER(n,m)", use_container_width=True):
            G_null = make_er_gnm(N0, E0, seed=int(seed))
            edges = [[u, v, 1.0, 1.0] for u, v in G_null.edges()]
            df_new = pd.DataFrame(edges, columns=[src_col, dst_col, "weight", "confidence"])
            add_graph_to_workspace(
                name=f"ER(n={N0},m={E0},seed={seed})",
                df_edges=df_new,
                source="null:ER",
                tags={"src_col": src_col, "dst_col": dst_col},
            )
            st.rerun()

    with gen3:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("➕ CFG(deg)", use_container_width=True):
                G_null = make_configuration_model(G_base_full, seed=int(seed))
                edges = [[u, v, 1.0, 1.0] for u, v in G_null.edges()]
                df_new = pd.DataFrame(edges, columns=[src_col, dst_col, "weight", "confidence"])
                add_graph_to_workspace(
                    name=f"CFG(deg,seed={seed})",
                    df_edges=df_new,
                    source="null:CFG",
                    tags={"src_col": src_col, "dst_col": dst_col},
                )
                st.rerun()
        with c2:
            if st.button("📎 Copy", use_container_width=True):
                add_graph_to_workspace(
                    name=f"copy:{entry['name']}",
                    df_edges=entry["edges"],
                    source="copy",
                    tags=entry["tags"],
                )
                st.rerun()

    with gen4:
        base_gid = entry["id"]
        p_now = float(st.session_state["mix_p_by_gid"].get(base_gid, 0.0))
        p_step = st.slider("Постепенная рандомизация Δp", 0.01, 0.50, 0.05, 0.01, key="mix_p_step_top")
        st.caption(f"Текущий p для этой базы: {p_now:.2f}")

        b1, b2 = st.columns(2)
        with b1:
            if st.button("Step mix", use_container_width=True):
                p_new = min(1.0, p_now + float(p_step))
                st.session_state["mix_p_by_gid"][base_gid] = p_new
                G_mix = rewire_mix(G_base_full, p=float(p_new), seed=int(seed))
                edges = [[u, v, 1.0, 1.0] for u, v in G_mix.edges()]
                df_new = pd.DataFrame(edges, columns=[src_col, dst_col, "weight", "confidence"])
                add_graph_to_workspace(
                    name=f"MIX(p={p_new:.2f},seed={seed}) from {entry['name']}",
                    df_edges=df_new,
                    source="mix:rewire",
                    tags={"src_col": src_col, "dst_col": dst_col},
                )
                st.rerun()
        with b2:
            if st.button("Reset p", use_container_width=True):
                st.session_state["mix_p_by_gid"][base_gid] = 0.0
                st.rerun()

    st.write("---")
    return st.session_state["active_graph_id"]


# =========================
# Sidebar: Workspace I/O + Upload + Analysis controls
# =========================
with st.sidebar:
    st.header("🧠 Workspace")

    with st.expander("💾 Import / Export", expanded=False):
        c1, c2 = st.columns(2)

        if c1.button("Export workspace"):
            blob = export_workspace_json(st.session_state["graphs"], st.session_state["experiments"])
            st.download_button(
                "⬇️ скачать workspace.json",
                data=blob,
                file_name="workspace.json",
                mime="application/json",
                use_container_width=True,
            )

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
                gids = _sorted_graph_ids()
                st.session_state["active_graph_id"] = gids[0] if gids else None
                st.success("Workspace импортирован.")
            except Exception as e:
                st.error(f"Ошибка импорта workspace: {e}")

        up_exps = st.file_uploader("Import experiments.json", type=["json"], key="exps_import")
        if up_exps is not None:
            try:
                exps_add = import_experiments_json(up_exps.getvalue())
                st.session_state["experiments"].extend(exps_add)
                st.success("Experiments импортированы (добавлены).")
            except Exception as e:
                st.error(f"Ошибка импорта experiments: {e}")

    st.write("---")
    st.header("📎 Добавить граф")

    uploaded = st.file_uploader(
        "Загрузите CSV/Excel (фикс. формат)",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=False,
        key="file_uploader_main",
    )
    st.caption("Ожидается: 1-я колонка=source id, 2-я=target id, 9-я=confidence, 10-я=weight.")

    if uploaded is not None:
        try:
            df_any = load_uploaded_any(uploaded.getvalue(), uploaded.name)
            df_edges, meta = coerce_fixed_format(df_any)  # meta={"src_col":..., "dst_col":...}
            add_graph_to_workspace(
                name=f"uploaded:{uploaded.name}",
                df_edges=df_edges,
                source="upload",
                tags=meta,
            )
            st.session_state["last_upload_name"] = uploaded.name
            st.success(f"Добавлен граф: uploaded:{uploaded.name}")
        except Exception as e:
            st.error(f"Ошибка загрузки/парсинга: {e}")

    st.write("---")
    st.header("🎛️ Анализ")

    # Controls can be set even if no graphs; the main area will stop gracefully.
    min_conf = st.number_input("Порог confidence (>=)", value=0, step=1)
    min_weight = st.number_input("Мин. вес weight (>=)", value=0.0, step=0.1)
    eff_sources_k = st.slider("Efficiency k (аппрокс)", 8, 256, 64, 8)
    seed_analysis = st.number_input("Seed (анализ/3D/метрики)", value=42, step=1)

    st.write("---")
    st.header("🧹 Очистка")
    cc1, cc2 = st.columns(2)
    with cc1:
        if st.button("Очистить experiments", use_container_width=True):
            st.session_state["experiments"] = []
            st.success("Experiments очищены.")
    with cc2:
        if st.button("Очистить всё", use_container_width=True):
            st.session_state["graphs"] = {}
            st.session_state["experiments"] = []
            st.session_state["active_graph_id"] = None
            st.session_state["mix_p_by_gid"] = {}
            st.success("Workspace очищен.")


# =========================
# Top panel: graph switching + generation
# =========================
active_gid = render_top_graph_panel()
if active_gid is None:
    st.stop()

graph_entry = get_active_graph_entry()
graph_id = graph_entry["id"]

# =========================
# Build Active Graph (filtered) + optional LCC view
# =========================
G, ctx = build_active_graph(min_conf=int(min_conf), min_weight=float(min_weight))
if G is None:
    st.stop()

mode = st.radio("Масштаб", ["Глобальный (весь граф)", "LCC (гигантская компонента)"], horizontal=True)
G_view = lcc_subgraph(G) if mode.startswith("LCC") else G

# =========================
# Quick summary row
# =========================
top1, top2, top3, top4 = st.columns([2.3, 1.2, 1.2, 1.3])
with top1:
    st.subheader(f"Активный граф: {graph_entry['name']}")
    st.caption(f"source={graph_entry['source']} · graph_id={graph_id} · фильтры: conf>={min_conf}, w>={min_weight}")
with top2:
    st.metric("Nodes", int(G_view.number_of_nodes()))
with top3:
    st.metric("Edges", int(G_view.number_of_edges()))
with top4:
    st.metric("Components", int(nx.number_connected_components(G_view)) if G_view.number_of_nodes() else 0)

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
        "🆚 Сравнение",
    ]
)

# -------------------------
# Tab 1: metrics
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
# Tab 2: spectrum/complexity text
# -------------------------
with t2:
    left, right = st.columns([1.2, 1.0])

    with left:
        st.subheader("λ₂ / Q интерпретация")
        st.write(f"- λ₂ (LCC) = {base['l2_lcc']:.6g}")
        st.write(f"- Q (Louvain) = {base['mod']:.6g}")
        st.markdown(
            "**Эвристика:**\n"
            "- λ₂ низкая → сеть легко рассоединяется\n"
            "- Q высокая → сильная модульность\n"
            "- Q низкая → нет выраженных сообществ\n"
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

# -------------------------
# Tab 4: Attack Lab
# -------------------------
with t4:
    st.subheader("💥 Лаборатория разрушения")
    st.caption("Эксперименты сохраняются в Workspace → можно сравнивать на вкладке 🆚.")

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
        keep_states_3d = b3.checkbox("Хранить states для 3D replay", value=False)
        name_tag = b4.text_input("Название эксперимента (tag)", value="")

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
    run_one = run_col1.button("▶️ Запустить (1 сценарий)", use_container_width=True)
    run_all = run_col2.button("⚔️ Head-to-head (4 атаки)", use_container_width=True)

    def _run_and_render(attack_kind_local: str, title_prefix: str):
        if G_view.number_of_nodes() < 2:
            st.warning("Граф слишком маленький.")
            return None

        df_hist, _states = run_attack(
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
            f"Phase: abrupt={phase['is_abrupt']} · critical~{phase['critical_x']:.3f} · jump={phase['jump']:.3f}"
        )

        fig = fig_metrics_over_steps(df_hist, title=f"{label} — кривые")
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
            fig_cmp = fig_compare_attacks(results, x_col="removed_frac", y_col="lcc_frac", title="Head-to-head: LCC")
            st.plotly_chart(fig_cmp, use_container_width=True)

# -------------------------
# Tab 5: Comparison
# -------------------------
with t5:
    st.subheader("🆚 Сравнение")

    graphs = st.session_state["graphs"]
    exps = st.session_state["experiments"]

    st.write("### Сравнить графы (скаляры)")
    graph_ids = _sorted_graph_ids()
    selected_graphs = st.multiselect(
        "Выбери графы",
        options=graph_ids,
        default=[graph_id],
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
        st.info("Нет экспериментов. Запусти атаку в 💥 ATTACK LAB.")
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
