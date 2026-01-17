import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st
from concurrent.futures import ProcessPoolExecutor, as_completed

# Make sure local imports work on Streamlit Cloud
sys.path.append(os.path.dirname(__file__))

from src.io_load import load_uploaded_bytes
from src.preprocess import preprocess_fixed_format
from src.graph_build import build_graph, pick_component
from src.null_models import build_null
from src.cache_keys import graph_to_edge_payload, payload_to_graph
from src.sim import run_attack
from src.phase import phase_indicators
from src.plots import plot_compare_runs, plot_lambda2_Q_phase


# =========================
# Worker for parallel head-to-head
# (must be defined BEFORE UI code)
# =========================
def _run_attack_worker(edge_payload, attack_kind, remove_frac, steps, seed, eff_k,
                       rc_frac, rc_min_density, rc_max_frac, heavy_every):
    import pandas as pd
    from src.cache_keys import payload_to_graph
    from src.sim import run_attack
    from src.phase import phase_indicators

    G = payload_to_graph(edge_payload)
    df_hist = run_attack(
        G0=G,
        attack_kind=attack_kind,
        remove_frac=remove_frac,
        steps=steps,
        seed=seed,
        eff_k=eff_k,
        rc_frac=rc_frac,
        rc_min_density=rc_min_density,
        rc_max_frac=rc_max_frac,
        compute_heavy_every=heavy_every,
    )
    phi = phase_indicators(df_hist)
    name = f"head2head|{attack_kind}|seed={seed}"
    meta = {"attack": attack_kind, **phi}
    return name, df_hist, meta


@st.cache_data(show_spinner=False)
def load_and_preprocess(file_bytes: bytes, filename: str):
    df_any = load_uploaded_bytes(file_bytes, filename)
    df_raw, SRC_COL, DST_COL = preprocess_fixed_format(df_any)
    return df_raw, SRC_COL, DST_COL


@st.cache_data(show_spinner=False)
def cached_attack(edge_payload, attack_kind, remove_frac, steps, seed, eff_k,
                  rc_frac, rc_min_density, rc_max_frac, heavy_every):
    G = payload_to_graph(edge_payload)
    return run_attack(
        G0=G,
        attack_kind=attack_kind,
        remove_frac=remove_frac,
        steps=steps,
        seed=seed,
        eff_k=eff_k,
        rc_frac=rc_frac,
        rc_min_density=rc_min_density,
        rc_max_frac=rc_max_frac,
        compute_heavy_every=heavy_every,
    )


# =========================
# Page
# =========================
st.set_page_config(page_title="прикольчик", layout="wide", page_icon="💀")
st.title("прикольчик")

# =========================
# Session state
# =========================
if "experiments" not in st.session_state:
    st.session_state["experiments"] = []  # list[{"name": str, "df": DataFrame, "meta": dict}]


# =========================
# Upload
# =========================
with st.sidebar:
    st.header("📎 Данные")
    uploaded = st.file_uploader(
        "Загрузите файл (CSV или Excel) в текущем формате",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=False,
    )
    st.caption("Формат: 1=src id, 2=dst id, 9=confidence, 10=weight")

if uploaded is None:
    st.info("Загрузи CSV/XLSX/XLS, чтобы начать.")
    st.stop()

df_raw, SRC_COL, DST_COL = load_and_preprocess(uploaded.getvalue(), uploaded.name)

if df_raw.empty:
    st.error("После очистки данные пустые.")
    st.stop()


# =========================
# Sidebar controls
# =========================
with st.sidebar:
    st.header("🎛️ Фильтры")

    max_conf = int(df_raw["confidence"].max()) if len(df_raw) else 0
    min_conf = st.slider("Порог confidence", 0, max_conf, min(100, max_conf))

    max_w = float(df_raw["weight"].max()) if len(df_raw) else 0.0
    min_weight = st.number_input("Мин. weight", 0.0, max_w, 0.0, step=0.1)

    st.write("---")
    st.header("🔍 Масштаб")
    mode = st.radio("Режим", ["Глобальный", "Компонента"])

    st.write("---")
    st.header("🧪 Null models")
    graph_kind = st.selectbox(
        "Граф для эксперимента",
        ["empirical", "mix_rewire", "rewired_degree", "configuration", "er_gnm"],
        format_func=lambda x: {
            "empirical": "Эмпирический",
            "mix_rewire": "Смешивание (rewire p)",
            "rewired_degree": "Полный rewire (степени сохранены)",
            "configuration": "Configuration model",
            "er_gnm": "ER G(n,m)",
        }[x],
    )

    mix_p = 0.0
    if graph_kind == "mix_rewire":
        mix_p = st.slider("rewire p", 0.0, 1.0, 0.0, 0.05)

    st.write("---")
    st.header("⚙️ Настройки")
    eff_sources_k = st.slider("Efficiency sources k", 8, 256, 64, 8)
    seed = st.number_input("Seed", value=42, step=1)

    st.write("---")
    st.header("⚡ Скорость")
    compare_mode = st.radio("Режим атак", ["Single run", "Compare head-to-head"])
    heavy_every = st.slider("Heavy metrics каждые N шагов", 1, 10, 1, 1)
    parallel = st.checkbox("Параллельно (только head-to-head)", value=True)


# =========================
# Graph build (filtered empirical -> component -> null model)
# =========================
mask = (df_raw["confidence"] >= min_conf) & (df_raw["weight"] >= min_weight)
df_filtered = df_raw.loc[mask].copy()

G_full = build_graph(df_filtered, SRC_COL, DST_COL)

comp_id = None
if mode == "Компонента":
    comps = sorted(list(nx.connected_components(G_full)), key=len, reverse=True)
    if comps:
        with st.sidebar:
            comp_id = st.selectbox(
                "Компонента",
                range(len(comps)),
                format_func=lambda i: f"Cluster {i} (n={len(comps[i])})",
            )

G_emp = pick_component(G_full, comp_id=comp_id if mode == "Компонента" else None)

if G_emp.number_of_nodes() == 0:
    st.warning("Граф пустой после фильтров/выбора компоненты.")
    st.stop()

G_exp = build_null(G_emp, kind=graph_kind, seed=int(seed), mix_p=float(mix_p))


# =========================
# Tabs
# =========================
t1, t2 = st.tabs(["💥 ATTACK LAB", "📈 Сравнение/Фазы"])


with t1:
    st.subheader("Атаки")

    attack_items = [
        ("random", "Random"),
        ("strength", "Hubs (strength)"),
        ("betweenness", "Bridges (betweenness)"),
        ("kcore", "k-core"),
        ("richclub_top", "Rich-club (top strength)"),
        ("richclub_density", "Rich-club (dense)"),
        ("richclub_seams_top", "Rich-club seams (top)"),
        ("richclub_seams_density", "Rich-club seams (dense)"),
        ("stealth", "Stealth (betweenness/strength)"),
    ]

    attack_kind = st.selectbox(
        "Стратегия",
        [k for k, _ in attack_items],
        format_func=lambda k: dict(attack_items)[k],
    )

    remove_frac = st.slider("Удалить долю узлов", 0.05, 0.95, 0.50, 0.05)
    steps = st.slider("Шагов", 5, 120, 30)

    st.write("### Rich-club параметры (используются где надо)")
    rc_frac = st.slider("RC-A доля", 0.02, 0.50, 0.10, 0.02)
    rc_min_density = st.slider("RC-B min density", 0.05, 1.00, 0.30, 0.05)
    rc_max_frac = st.slider("RC-B max frac", 0.05, 0.80, 0.30, 0.05)

    default_name = f"{graph_kind}|{attack_kind}|seed={seed}|p={mix_p:.2f}|conf>={min_conf}|w>={min_weight}"
    run_name = st.text_input("Имя прогона", value=default_name)

    c1, c2, c3 = st.columns(3)
    add_run = c1.button("ADD RUN")
    clear_runs = c2.button("CLEAR RUNS")

    if compare_mode == "Compare head-to-head":
        run_all = c3.button("RUN ALL (head-to-head)")
    else:
        run_all = None

    if clear_runs:
        st.session_state["experiments"] = []
        st.success("Runs очищены.")

    # Single add run
    if add_run:
        edge_payload = graph_to_edge_payload(G_exp)
        df_hist = cached_attack(
            edge_payload=edge_payload,
            attack_kind=attack_kind,
            remove_frac=float(remove_frac),
            steps=int(steps),
            seed=int(seed),
            eff_k=int(eff_sources_k),
            rc_frac=float(rc_frac),
            rc_min_density=float(rc_min_density),
            rc_max_frac=float(rc_max_frac),
            heavy_every=int(heavy_every),
        )

        phi = phase_indicators(df_hist)
        meta = {
            "graph_kind": graph_kind,
            "mix_p": float(mix_p),
            "attack": attack_kind,
            "min_conf": int(min_conf),
            "min_weight": float(min_weight),
            **phi,
        }

        st.session_state["experiments"].append({"name": run_name, "df": df_hist, "meta": meta})
        st.success(f"Добавлен прогон: {run_name} | jump={meta['jump']:.3f} | fc_removed≈{meta['fc_removed_frac']:.3f}")

    # Head-to-head
    if run_all:
        scenarios = ["random", "strength", "betweenness", "kcore", "richclub_top", "richclub_seams_top", "stealth"]
        edge_payload = graph_to_edge_payload(G_exp)

        if parallel:
            jobs = []
            with ProcessPoolExecutor(max_workers=min(7, os.cpu_count() or 4)) as ex:
                for ak in scenarios:
                    jobs.append(ex.submit(
                        _run_attack_worker,
                        edge_payload,
                        ak,
                        float(remove_frac),
                        int(steps),
                        int(seed),
                        int(eff_sources_k),
                        float(rc_frac),
                        float(rc_min_density),
                        float(rc_max_frac),
                        int(heavy_every),
                    ))

                for fut in as_completed(jobs):
                    name, df_hist, meta = fut.result()
                    full_name = f"{graph_kind}|{name}|p={mix_p:.2f}|conf>={min_conf}|w>={min_weight}"
                    meta = {
                        "graph_kind": graph_kind,
                        "mix_p": float(mix_p),
                        "min_conf": int(min_conf),
                        "min_weight": float(min_weight),
                        **meta,
                    }
                    st.session_state["experiments"].append({"name": full_name, "df": df_hist, "meta": meta})

            st.success("Head-to-head готов.")
        else:
            for ak in scenarios:
                df_hist = cached_attack(
                    edge_payload=edge_payload,
                    attack_kind=ak,
                    remove_frac=float(remove_frac),
                    steps=int(steps),
                    seed=int(seed),
                    eff_k=int(eff_sources_k),
                    rc_frac=float(rc_frac),
                    rc_min_density=float(rc_min_density),
                    rc_max_frac=float(rc_max_frac),
                    heavy_every=int(heavy_every),
                )
                phi = phase_indicators(df_hist)
                full_name = f"{graph_kind}|{ak}|seed={seed}|p={mix_p:.2f}|conf>={min_conf}|w>={min_weight}"
                meta = {
                    "graph_kind": graph_kind,
                    "mix_p": float(mix_p),
                    "attack": ak,
                    "min_conf": int(min_conf),
                    "min_weight": float(min_weight),
                    **phi,
                }
                st.session_state["experiments"].append({"name": full_name, "df": df_hist, "meta": meta})

            st.success("Head-to-head готов.")


with t2:
    runs = st.session_state["experiments"]

    if not runs:
        st.info("Нет сохранённых прогонов. Добавь run в ATTACK LAB.")
        st.stop()

    st.subheader("Order parameter: S(f)=|LCC|/N0")
    st.plotly_chart(plot_compare_runs(runs, "lcc_frac", "S(f) под атакой"), use_container_width=True)

    st.subheader("λ₂(LCC)")
    st.plotly_chart(plot_compare_runs(runs, "l2_lcc", "λ₂(LCC) под атакой"), use_container_width=True)

    st.subheader("Q(LCC)")
    st.plotly_chart(plot_compare_runs(runs, "Q_lcc", "Q(LCC) под атакой"), use_container_width=True)

    st.subheader("Phase portrait: λ₂ vs Q (LCC)")
    st.plotly_chart(plot_lambda2_Q_phase(runs), use_container_width=True)

    st.subheader("Взрывность (jump) и критическая область")
    for r in runs[-12:]:
        meta = r.get("meta", {})
        st.write(
            f"- **{r['name']}** | "
            f"jump={meta.get('jump', np.nan):.3f} | "
            f"fc_removed≈{meta.get('fc_removed_frac', np.nan):.3f}"
        )
