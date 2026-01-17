import networkx as nx
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.io_load import load_uploaded_as_raw, clean_fixed_format
from src.graph_build import build_graph, filter_edges, connected_components_sorted, lcc_subgraph
from src.metrics import calculate_metrics
from src.attacks import run_attack
from src.null_models import gnm_null_model, configuration_null_model, rewire_mix, copy_weights_from_original
from src.viz import compute_3d_layout, make_3d_traces, plot_attack_curves


# -------------------------
# Page
# -------------------------
st.set_page_config(page_title="прикольчик", layout="wide", page_icon="💀")
st.title("прикольчик")


# -------------------------
# Session state init
# -------------------------
if "experiments" not in st.session_state:
    st.session_state["experiments"] = []


# -------------------------
# Upload
# -------------------------
with st.sidebar:
    st.header("📎 Данные")
    uploaded = st.file_uploader(
        "Загрузите CSV/XLSX/XLS (текущий фикс-формат)",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=False,
    )
    st.caption("Формат: 1=src id, 2=dst id, 9=confidence, 10=weight")

if uploaded is None:
    st.info("Загрузи файл, чтобы начать.")
    st.stop()


@st.cache_data(show_spinner=False)
def cached_load(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Cache file load to speed up Streamlit reruns."""
    return load_uploaded_as_raw(file_bytes, filename)


df_any = cached_load(uploaded.getvalue(), uploaded.name)

try:
    df_raw, meta = clean_fixed_format(df_any)
except Exception as e:
    st.error(f"Ошибка формата: {e}")
    st.stop()

SRC_COL = meta["SRC_COL"]
DST_COL = meta["DST_COL"]


# -------------------------
# Sidebar controls
# -------------------------
with st.sidebar:
    st.header("🎛️ Калибровка")

    max_conf = int(df_raw["confidence"].max()) if len(df_raw) else 0
    min_conf = st.slider("Min confidence", 0, max_conf, min(100, max_conf))

    max_w = float(df_raw["weight"].max()) if len(df_raw) else 0.0
    min_weight = st.number_input("Min weight", 0.0, max_w, 0.0, step=0.1)

    st.write("---")
    mode = st.radio("Масштаб", ["Глобальный", "Локальный (Компонент)"])

    st.write("---")
    st.header("⚙️ Вычисления")
    seed = st.number_input("Seed", value=42, step=1)
    eff_sources_k = st.slider("Efficiency sources k", 8, 256, 64, 8)

    st.write("---")
    st.header("🧪 Нулевые модели / Mix")
    null_kind = st.selectbox("Null model", ["off", "ER (G(n,m))", "Configuration (deg-preserving)"])
    mix_p = st.slider("Mix randomness p (rewire)", 0.0, 1.0, 0.0, 0.05)
    keep_weight_dist = st.checkbox("Assign weights to null-model edges", value=True)


# -------------------------
# Build graph
# -------------------------
@st.cache_data(show_spinner=False)
def cached_build_graph(df_raw: pd.DataFrame, min_conf: float, min_weight: float, src_col: str, dst_col: str):
    """Cache filtered graph building to avoid recomputation."""
    df_f = filter_edges(df_raw, min_conf=min_conf, min_weight=min_weight)
    G_full = build_graph(df_f, src_col=src_col, dst_col=dst_col)
    comps = connected_components_sorted(G_full)
    return G_full, comps


G_full, comps = cached_build_graph(df_raw, min_conf, min_weight, SRC_COL, DST_COL)

if mode == "Локальный (Компонент)":
    if len(comps) == 0:
        G0 = G_full.copy()
    else:
        with st.sidebar:
            comp_id = st.selectbox(
                "Компонента",
                range(len(comps)),
                format_func=lambda i: f"#{i} n={len(comps[i])}",
            )
        G0 = G_full.subgraph(comps[comp_id]).copy()
else:
    G0 = G_full.copy()

G = G0.copy()

if null_kind != "off":
    if null_kind.startswith("ER"):
        H = gnm_null_model(G0, seed=int(seed))
    else:
        H = configuration_null_model(G0, seed=int(seed))

    if keep_weight_dist:
        H = copy_weights_from_original(G0, H, seed=int(seed))

    G = H

if mix_p > 0:
    # mix is applied on the current chosen graph (original or null)
    G = rewire_mix(G, p=float(mix_p), seed=int(seed), swaps_per_edge=1.0)


# -------------------------
# Base metrics
# -------------------------
base = calculate_metrics(G, eff_sources_k=int(eff_sources_k), seed=int(seed))


# -------------------------
# Tabs
# -------------------------
t1, t2, t3, t4, t5 = st.tabs([
    "📊 Метрики графа",
    "🧩 Компоненты",
    "👁️ 3D",
    "💥 Attack Lab",
    "🧪 Compare / Experiments",
])

with t1:
    st.subheader("Базовые метрики (выбранный граф)")

    c1, c2, c3 = st.columns(3)
    c1.metric("N", base["N"])
    c1.metric("E", base["E"])
    c2.metric("Components", base["C"])
    c2.metric("β = E-N+C", int(base["beta"]))
    c3.metric("Efficiency (weighted)", f"{base['eff_w']:.6f}")
    c3.metric("λmax(Aw)", f"{base['lmax']:.6f}")

    st.write("---")

    c4, c5, c6 = st.columns(3)
    c4.metric("λ₂ (LCC)", f"{base['l2_lcc']:.10f}")
    c4.metric("τ(LCC)=1/λ₂", f"{base['tau_lcc']:.3f}" if base["tau_lcc"] < 1e18 else "inf")

    c5.metric("λ₂ (global)", f"{base['l2_global']:.10f}")
    c5.metric("SIS threshold ~ 1/λmax", f"{base['thresh_SIS']:.6f}")

    c6.metric("Weighted clustering", f"{base['clust_w']:.6f}")
    c6.metric("Strength assortativity", f"{base['assort_strength']:.6f}")

    st.write("---")
    c7, c8 = st.columns(2)
    c7.metric("Entropy H(strength)", f"{base['H_strength']:.6f}")
    c8.metric("Von Neumann entropy (Laplacian)", f"{base['S_vn_laplacian']:.6f}")

    with st.expander("Что всё это значит (коротко)"):
        st.markdown(
            """
- **λ₂ (algebraic connectivity)** — насколько “жёстко склеен” граф: падает → сеть легче разваливается на куски.
- **Modularity Q** (будет считаться в атаках) — насколько сеть распадается на модули/сообщества.
- **β = E-N+C** — число независимых циклов (приближённо “резервных путей”).
- **Efficiency** — интеграция: насколько короткие маршруты по сети.
- **Entropy** — грубо “насколько неравномерно распределена нагрузка/связность”.
- **Von Neumann entropy** — спектральная “сложность” через лапласиан.
"""
        )

with t2:
    st.subheader("Компоненты")
    if G.number_of_nodes() == 0:
        st.info("Пустой граф.")
    else:
        comps2 = connected_components_sorted(G)
        sizes = [len(c) for c in comps2]
        st.write(f"Всего компонент: {len(comps2)}")
        st.write(f"Топ-10 размеров: {sizes[:10]}")

        H = lcc_subgraph(G)
        st.write(f"LCC: n={H.number_of_nodes()} e={H.number_of_edges()} connected={nx.is_connected(H)}")

with t3:
    st.subheader("3D проекция (цвет = strength)")
    if G.number_of_nodes() == 0:
        st.info("Пустой граф.")
    else:
        pos3d = compute_3d_layout(G, seed=int(seed))
        e, n = make_3d_traces(G, pos3d, show_scale=True)
        fig = go.Figure(data=[e, n])
        fig.update_layout(
            template="plotly_dark",
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
        )
        st.plotly_chart(fig, use_container_width=True)

with t4:
    st.subheader("Attack Lab (single run)")

    c1, c2, c3 = st.columns(3)
    attack_kind_ui = c1.selectbox(
        "Attack kind",
        [
            "random",
            "degree (hubs)",
            "betweenness (bridges)",
            "kcore",
            "richclub_top",
            "richclub_density",
        ],
    )
    remove_frac = c2.slider("Remove fraction", 0.05, 0.95, 0.50, 0.05)
    steps = c3.slider("Steps", 5, 120, 30)

    eff_k_sim = st.slider("Efficiency k (sim)", 8, 256, int(eff_sources_k), 8)
    compute_heavy_every = st.slider("Heavy metrics every k steps", 1, 10, 1)

    rc_frac = st.slider("RC top frac", 0.02, 0.50, 0.10, 0.02)
    rc_min_density = st.slider("RC min density", 0.05, 1.00, 0.30, 0.05)
    rc_max_frac = st.slider("RC max frac", 0.05, 0.80, 0.30, 0.05)

    if st.button("RUN ATTACK"):
        attack_kind = attack_kind_ui.split()[0]
        df = run_attack(
            G,
            attack_kind=attack_kind,
            remove_frac=float(remove_frac),
            steps=int(steps),
            seed=int(seed),
            eff_sources_k=int(eff_k_sim),
            rc_frac=float(rc_frac),
            rc_min_density=float(rc_min_density),
            rc_max_frac=float(rc_max_frac),
            compute_heavy_every=int(compute_heavy_every),
        )

        if df.empty:
            st.warning("Слишком маленький граф / ничего не посчиталось.")
        else:
            phase = df.attrs.get("phase", {})
            st.success(
                f"Done. Phase: abrupt={phase.get('is_abrupt')} "
                f"crit={phase.get('crit_x')} drop={phase.get('max_drop')}"
            )

            exp_name = (
                f"{attack_kind} | rem={remove_frac:.2f} | steps={steps} "
                f"| null={null_kind} | mix={mix_p:.2f}"
            )
            st.session_state["experiments"].append({"name": exp_name, "df": df})

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["removed_frac"], y=df["lcc_frac"], name="LCC fraction"))
            if "l2_lcc" in df.columns:
                fig.add_trace(go.Scatter(x=df["removed_frac"], y=df["l2_lcc"], name="λ₂(LCC)"))
            if "eff_w" in df.columns:
                fig.add_trace(go.Scatter(x=df["removed_frac"], y=df["eff_w"], name="Efficiency"))
            fig.update_layout(template="plotly_dark", title="Single run dynamics", xaxis_title="Removed fraction")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(df)

with t5:
    st.subheader("Experiments сравнение")
    exps = st.session_state["experiments"]

    if not exps:
        st.info("Пока нет экспериментов. Запусти атаки — они появятся тут.")
    else:
        y_key = st.selectbox(
            "Y metric",
            ["lcc_frac", "l2_lcc", "eff_w", "lmax", "clust_w", "H_strength", "S_vn_laplacian"],
        )
        fig = plot_attack_curves(exps, y_key=y_key, title=f"Compare: {y_key}")
        st.plotly_chart(fig, use_container_width=True)

        if st.button("CLEAR experiments"):
            st.session_state["experiments"] = []
            st.success("Cleared.")
