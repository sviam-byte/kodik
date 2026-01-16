import io
import os
import random
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse.linalg as spla
import plotly.graph_objects as go
import streamlit as st

from networkx.algorithms.community import modularity, louvain_communities

# =========================
# Page
# =========================
st.set_page_config(page_title="прикольчик", layout="wide", page_icon="💀")
st.title("прикольчик")

# =========================
# Upload (expects the current fixed format)
# =========================
with st.sidebar:
    st.header("📎 Данные")
    uploaded = st.file_uploader(
        "Загрузите файл (CSV или Excel) в текущем формате",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=False,
    )
    st.caption(
        "Ожидается текущий формат:\n"
        "1-я колонка = source id, 2-я = target id,\n"
        "9-я = confidence, 10-я = weight."
    )

if uploaded is None:
    st.info("Загрузи CSV/XLSX/XLS, чтобы начать.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_uploaded_as_raw(file_bytes: bytes, filename: str) -> pd.DataFrame:
    name = (filename or "").lower()
    bio = io.BytesIO(file_bytes)

    if name.endswith(".csv"):
        df = pd.read_csv(bio, sep=None, engine="python")
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(bio)
    else:
        raise ValueError("Неподдерживаемый формат файла")

    df.columns = [str(c).strip() for c in df.columns]
    return df

df_any = load_uploaded_as_raw(uploaded.getvalue(), uploaded.name)

if df_any.shape[1] < 10:
    st.error("Файл должен содержать минимум 10 колонок (как в текущем формате).")
    st.stop()

# Fixed format columns by position
SRC_COL = df_any.columns[0]
DST_COL = df_any.columns[1]
CONF_COL = df_any.columns[8]
WEIGHT_COL = df_any.columns[9]

# Cast / clean
df_raw = df_any.copy()
df_raw[SRC_COL] = pd.to_numeric(df_raw[SRC_COL], errors="coerce").astype("Int64")
df_raw[DST_COL] = pd.to_numeric(df_raw[DST_COL], errors="coerce").astype("Int64")

df_raw[CONF_COL] = pd.to_numeric(df_raw[CONF_COL], errors="coerce")
df_raw[WEIGHT_COL] = pd.to_numeric(
    df_raw[WEIGHT_COL].astype(str).str.replace(",", ".", regex=False),
    errors="coerce",
)

df_raw = df_raw.rename(columns={CONF_COL: "confidence", WEIGHT_COL: "weight"})
df_raw = df_raw.dropna(subset=[SRC_COL, DST_COL, "confidence", "weight"])
df_raw = df_raw[df_raw["weight"] > 0]  # remove non-positive weights

if df_raw.empty:
    st.error("После очистки данные пустые (проверь numeric-колонки confidence/weight и id).")
    st.stop()

# =========================
# Sidebar controls
# =========================
with st.sidebar:
    st.header("🎛️ Калибровка мира")

    max_conf = int(df_raw["confidence"].max()) if len(df_raw) else 0
    min_conf = st.slider("Порог поддержки (Confidence)", 0, max_conf, min(100, max_conf))

    max_w = float(df_raw["weight"].max()) if len(df_raw) else 0.0
    min_weight = st.number_input("Мин. вес (Fibers)", 0.0, max_w, 0.0, step=0.1)

    st.write("---")
    st.header("🔍 Режим")
    mode = st.radio("Масштаб", ["Глобальный (Весь граф)", "Локальный (Компонент)"])

    st.write("---")
    st.header("⚙️ Настройки")
    eff_sources_k = st.slider("Efficiency: источники (точнее/медленнее)", 8, 256, 64, 8)
    seed = st.number_input("Seed", value=42, step=1)
    st.caption(
        "Seed фиксирует случайность: одинаковый seed → повторяемые выборы/3D-раскладка.\n"
        "Меняешь seed → меняется random-атака, выбор источников для efficiency и 3D."
    )

# =========================
# Graph build
# =========================
def build_graph(df_edges: pd.DataFrame) -> nx.Graph:
    G = nx.from_pandas_edgelist(
        df_edges,
        source=SRC_COL,
        target=DST_COL,
        edge_attr=["weight", "confidence"],
        create_using=nx.Graph(),
    )
    for _, _, d in G.edges(data=True):
        w = float(d.get("weight", 1.0))
        c = float(d.get("confidence", 0.0))
        if not np.isfinite(w) or w <= 0:
            w = 1e-12
        if not np.isfinite(c):
            c = 0.0
        d["weight"] = w
        d["confidence"] = c
    return G

mask = (df_raw["confidence"] >= min_conf) & (df_raw["weight"] >= min_weight)
df_filtered = df_raw.loc[mask].copy()
G_full = build_graph(df_filtered)

components = sorted(nx.connected_components(G_full), key=len, reverse=True)
num_comp = len(components)

if mode == "Локальный (Компонент)":
    if num_comp == 0:
        G = nx.Graph()
    else:
        comp_id = st.sidebar.selectbox(
            "Выбрать кластер",
            range(num_comp),
            format_func=lambda i: f"Cluster {i} (n={len(components[i])})",
        )
        G = G_full.subgraph(components[comp_id]).copy()
else:
    G = G_full.copy()

# =========================
# Utilities (metrics)
# =========================
def lcc_subgraph(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return G
    comp = max(nx.connected_components(G), key=len)
    return G.subgraph(comp).copy()

def add_dist_attr(G: nx.Graph) -> nx.Graph:
    H = G.copy()
    for _, _, d in H.edges(data=True):
        w = float(d.get("weight", 1.0))
        d["dist"] = 1.0 / w if w > 0 else 1e12
    return H

def approx_weighted_efficiency(G: nx.Graph, sources_k: int, seed: int) -> float:
    N = G.number_of_nodes()
    if N < 2:
        return 0.0
    H = add_dist_attr(G)
    nodes = list(H.nodes())
    rng = random.Random(int(seed))
    k = min(int(sources_k), N)
    sources = nodes if k == N else rng.sample(nodes, k)

    denom = N * (N - 1)
    total = 0.0
    for s in sources:
        dist = nx.single_source_dijkstra_path_length(H, s, weight="dist")
        acc = 0.0
        for t, d in dist.items():
            if t == s:
                continue
            if d > 0 and np.isfinite(d):
                acc += 1.0 / d
        total += acc

    est_full = total * (N / max(1, k))
    return float(est_full / denom)

def spectral_radius_weighted_adjacency(G: nx.Graph) -> float:
    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        return 0.0
    A = nx.adjacency_matrix(G, weight="weight").astype(float)
    try:
        v = spla.eigsh(A, k=1, which="LM", return_eigenvectors=False)[0]
        return float(v)
    except Exception:
        return 0.0

def lambda2_robust_connected(G: nx.Graph, eps: float = 1e-10) -> float:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n < 3 or m == 0:
        return 0.0
    if not nx.is_connected(G):
        return 0.0

    L = nx.laplacian_matrix(G, weight="weight").astype(float)

    if n <= 500:
        vals = np.linalg.eigvalsh(L.toarray())
        vals = np.sort(vals)
        for v in vals:
            if v > eps:
                return float(v)
        return 0.0

    try:
        vals = spla.eigsh(L, k=6, sigma=0.0, which="LM", return_eigenvectors=False)
        vals = np.sort(np.real(vals))
        for v in vals:
            if v > eps:
                return float(v)
        return 0.0
    except Exception:
        try:
            vals = spla.eigsh(L, k=6, which="SM", return_eigenvectors=False)
            vals = np.sort(np.real(vals))
            for v in vals:
                if v > eps:
                    return float(v)
            return 0.0
        except Exception:
            return 0.0

def lambda2_global(G: nx.Graph) -> float:
    if G.number_of_nodes() < 3 or G.number_of_edges() == 0:
        return 0.0
    if not nx.is_connected(G):
        return 0.0
    return lambda2_robust_connected(G)

def lambda2_on_lcc(G: nx.Graph) -> float:
    H = lcc_subgraph(G)
    if H.number_of_nodes() < 3 or H.number_of_edges() == 0:
        return 0.0
    if not nx.is_connected(H):
        return 0.0
    return lambda2_robust_connected(H)

def compute_modularity_louvain(G: nx.Graph, seed: int) -> float:
    if G.number_of_nodes() < 3 or G.number_of_edges() == 0:
        return 0.0
    try:
        comm = louvain_communities(G, weight="weight", seed=int(seed))
        return float(modularity(G, comm, weight="weight"))
    except TypeError:
        try:
            comm = louvain_communities(G, weight="weight")
            return float(modularity(G, comm, weight="weight"))
        except Exception:
            return 0.0
    except Exception:
        return 0.0

def beta_cycles(G: nx.Graph) -> int:
    return int(G.number_of_edges() - G.number_of_nodes() + nx.number_connected_components(G))

def lcc_fraction(G: nx.Graph, N0: int) -> float:
    if G.number_of_nodes() == 0 or N0 <= 0:
        return 0.0
    lcc = len(max(nx.connected_components(G), key=len))
    return float(lcc) / float(N0)

def calculate_metrics(G: nx.Graph, eff_sources_k: int, seed: int) -> dict:
    N = G.number_of_nodes()
    E = G.number_of_edges()
    C = nx.number_connected_components(G) if N > 0 else 0

    lmax = spectral_radius_weighted_adjacency(G)
    thresh = (1.0 / lmax) if lmax > 0 else 0.0

    l2g = lambda2_global(G)
    l2l = lambda2_on_lcc(G)

    eff_w = approx_weighted_efficiency(G, sources_k=int(eff_sources_k), seed=int(seed))
    mod = compute_modularity_louvain(G, seed=int(seed))

    return {
        "N": N,
        "E": E,
        "C": C,
        "beta": beta_cycles(G),
        "eff_w": eff_w,
        "l2_global": l2g,
        "tau_global": (1.0 / l2g) if l2g > 0 else float("inf"),
        "l2_lcc": l2l,
        "tau_lcc": (1.0 / l2l) if l2l > 0 else float("inf"),
        "lmax": lmax,
        "thresh": thresh,
        "mod": mod,
    }

base = calculate_metrics(G, eff_sources_k=int(eff_sources_k), seed=int(seed))

# =========================
# 3D helpers + animation
# =========================
def compute_3d_layout(G: nx.Graph, seed: int) -> dict:
    if G.number_of_nodes() == 0:
        return {}
    return nx.spring_layout(G, dim=3, weight="weight", seed=int(seed))

def make_3d_traces(G: nx.Graph, pos3d: dict, show_scale: bool = False):
    if G.number_of_nodes() == 0:
        return None, None

    strength = dict(G.degree(weight="weight"))
    nodes = [n for n in G.nodes() if n in pos3d]

    xs = [pos3d[n][0] for n in nodes]
    ys = [pos3d[n][1] for n in nodes]
    zs = [pos3d[n][2] for n in nodes]
    colors = [strength.get(n, 0.0) for n in nodes]
    texts = [f"{n}: strength={strength.get(n, 0.0):.3f}" for n in nodes]

    node_trace = go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="markers",
        marker=dict(size=4, color=colors, colorscale="Inferno", showscale=show_scale),
        text=texts, hoverinfo="text",
        name="nodes",
    )

    ex, ey, ez = [], [], []
    for u, v in G.edges():
        if u not in pos3d or v not in pos3d:
            continue
        ex.extend([pos3d[u][0], pos3d[v][0], None])
        ey.extend([pos3d[u][1], pos3d[v][1], None])
        ez.extend([pos3d[u][2], pos3d[v][2], None])

    edge_trace = go.Scatter3d(
        x=ex, y=ey, z=ez,
        mode="lines",
        line=dict(color="#444", width=1),
        hoverinfo="none",
        name="edges",
    )
    return edge_trace, node_trace

def make_attack_3d_animation(states: list[nx.Graph], pos3d: dict) -> go.Figure:
    if not states:
        return go.Figure()

    e0, n0 = make_3d_traces(states[0], pos3d, show_scale=True)
    fig = go.Figure(data=[e0, n0])

    frames = []
    for i, Gi in enumerate(states):
        ei, ni = make_3d_traces(Gi, pos3d, show_scale=False)
        frames.append(go.Frame(data=[ei, ni], name=str(i)))
    fig.frames = frames

    slider_steps = [{
        "method": "animate",
        "args": [[str(i)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
        "label": str(i),
    } for i in range(len(states))]

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, b=0, t=35),
        scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
        title="3D Attack Replay (frames = steps)",
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.0, "y": 1.15,
            "buttons": [
                {"label": "Play", "method": "animate",
                 "args": [None, {"fromcurrent": True, "frame": {"duration": 250, "redraw": True}, "transition": {"duration": 0}}]},
                {"label": "Pause", "method": "animate",
                 "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]},
            ],
        }],
        sliders=[{
            "active": 0,
            "x": 0.0, "y": 1.05,
            "len": 1.0,
            "pad": {"t": 0, "b": 0},
            "steps": slider_steps,
        }],
    )
    return fig

# =========================
# Rich-Club helpers + attack
# =========================
def strength_ranking(G: nx.Graph) -> list:
    strength = dict(G.degree(weight="weight"))
    nodes = list(G.nodes())
    return sorted(nodes, key=lambda n: strength.get(n, 0.0), reverse=True)

def richclub_top_fraction(G: nx.Graph, rc_frac: float) -> list:
    nodes_sorted = strength_ranking(G)
    if not nodes_sorted:
        return []
    k = max(1, int(len(nodes_sorted) * float(rc_frac)))
    return nodes_sorted[:k]

def richclub_by_density_threshold(G: nx.Graph, min_density: float, max_frac: float) -> list:
    nodes_sorted = strength_ranking(G)
    n = len(nodes_sorted)
    if n == 0:
        return []
    if n < 3:
        return nodes_sorted

    maxK = max(3, int(n * float(max_frac)))
    maxK = min(maxK, n)

    best = nodes_sorted[:3]
    for K in range(3, maxK + 1):
        club = nodes_sorted[:K]
        H = G.subgraph(club)
        if nx.density(H) >= float(min_density):
            best = club
    return best

def pick_targets_for_attack(
    G: nx.Graph,
    attack_kind: str,
    step_size: int,
    seed: int,
    rc_frac: float,
    rc_min_density: float,
    rc_max_frac: float,
) -> list:
    nodes = list(G.nodes())
    if not nodes:
        return []

    rng = random.Random(int(seed))

    if attack_kind == "random":
        k = min(len(nodes), step_size)
        return rng.sample(nodes, k)

    if attack_kind == "degree":
        strength = dict(G.degree(weight="weight"))
        return sorted(nodes, key=lambda n: strength.get(n, 0.0), reverse=True)[:step_size]

    if attack_kind == "betweenness":
        H = add_dist_attr(G)
        n = H.number_of_nodes()
        k_samples = min(200, n)
        try:
            bc = nx.betweenness_centrality(H, k=k_samples, weight="dist", normalized=True, seed=int(seed))
        except TypeError:
            bc = nx.betweenness_centrality(H, k=k_samples, weight="dist", normalized=True)
        return sorted(nodes, key=lambda n: bc.get(n, 0.0), reverse=True)[:step_size]

    if attack_kind == "kcore":
        try:
            core = nx.core_number(G)
        except Exception:
            core = {n: 0 for n in nodes}
        return sorted(nodes, key=lambda n: core.get(n, 0), reverse=True)[:step_size]

    if attack_kind == "richclub_top":
        club = richclub_top_fraction(G, rc_frac=rc_frac)
        return club[:min(step_size, len(club))]

    if attack_kind == "richclub_density":
        club = richclub_by_density_threshold(G, min_density=rc_min_density, max_frac=rc_max_frac)
        return club[:min(step_size, len(club))]

    return []

# =========================
# Diagnostics expander (lambda2)
# =========================
with st.expander("Диагностика (почему λ₂ может быть 0)", expanded=True):
    st.write(f"- Nodes N = {base['N']}, Edges E = {base['E']}, Components C = {base['C']}")
    if G.number_of_nodes() > 0:
        H = lcc_subgraph(G)
        st.write(f"- LCC: n={H.number_of_nodes()}, e={H.number_of_edges()}, connected={nx.is_connected(H) if H.number_of_nodes()>0 else False}")


# =========================
# Tabs UI
# =========================
t1, t2, t3, t4 = st.tabs(["📊 Скелет", "⚡ Нервы (Спектр)", "👁️ 3D", "💥 ATTACK LAB"])

with t1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Nodes", base["N"])
    c1.metric("Edges", base["E"])

    c2.metric("Components", base["C"])
    lcc_size = len(max(nx.connected_components(G), key=len)) if G.number_of_nodes() > 0 else 0
    c2.metric("LCC Size", lcc_size)

    c3.metric("Beta (Cycles)", int(base["beta"]))
    c3.metric("Weighted Efficiency", f"{base['eff_w']:.6f}")

with t2:
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("λ₂")
        st.metric("λ₂ on LCC", f"{base['l2_lcc']:.10f}")
        st.caption(f"tau(LCC) = {base['tau_lcc']:.2f}")

        st.write("")
        st.metric("λ₂ global", f"{base['l2_global']:.10f}")
        if base["C"] > 1:
            st.caption("λ₂ global = 0 при несвязности (корректно).")
        else:
            st.caption(f"tau(global) = {base['tau_global']:.2f}")

    with col_r:
        st.subheader("λmax (weighted adjacency)")
        st.metric("Spectral Radius", f"{base['lmax']:.6f}")
        st.caption(f"1/λmax ≈ {base['thresh']:.6f}")

        st.write("")
        st.subheader("Модульность")
        st.metric("Louvain Modularity", f"{base['mod']:.6f}")

with t3:
    st.header("3D Projection (color = weighted degree strength)")
    if G.number_of_nodes() == 0:
        st.info("Пустой граф после фильтров.")
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
    st.header("💥 Лаборатория Разрушения")

    with st.expander("Настройки симуляции", expanded=True):
        c1, c2, c3 = st.columns(3)

        attack_kind_ui = c1.selectbox(
            "Вектор атаки",
            [
                "random",
                "degree (Hubs)",
                "betweenness (Bridges)",
                "kcore (Core)",
                "rich-club A (Top-fraction by strength)",
                "rich-club B (Density-threshold club)",
            ],
        )
        remove_frac = c2.slider("Уничтожить % узлов", 0.05, 0.95, 0.50, 0.05)
        steps = c3.slider("Шагов", 5, 80, 20)

        attack_seed = c1.number_input("Seed атаки", value=int(seed), step=1)
        eff_k_sim = c2.slider("Efficiency k (sim)", 8, 256, int(eff_sources_k), 8)

        # RC params shown only when relevant (defaults otherwise)
        rc_frac = 0.10
        rc_min_density = 0.30
        rc_max_frac = 0.30

        if attack_kind_ui.startswith("rich-club A"):
            st.write("### Rich-Club A params")
            rc_frac = st.slider("RC-A: доля топ-узлов", 0.02, 0.50, 0.10, 0.02)

        if attack_kind_ui.startswith("rich-club B"):
            st.write("### Rich-Club B params")
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

    if st.button("ЗАПУСТИТЬ"):
        if G.number_of_nodes() < 2:
            st.warning("Граф слишком маленький.")
        else:
            progress = st.progress(0)
            history = []
            states_3d = []

            G_curr = G.copy()
            N0 = G_curr.number_of_nodes()
            rng = random.Random(int(attack_seed))

            # stable 3D layout for replay
            pos3d_attack = compute_3d_layout(G_curr, seed=int(attack_seed))

            total_remove = int(N0 * float(remove_frac))
            step_size = max(1, total_remove // int(steps))

            for step in range(int(steps)):
                if G_curr.number_of_nodes() < 2:
                    break

                states_3d.append(G_curr.copy())

                met = calculate_metrics(G_curr, eff_sources_k=int(eff_k_sim), seed=int(attack_seed))
                met["step"] = step
                met["nodes_left"] = G_curr.number_of_nodes()
                met["lcc_frac"] = lcc_fraction(G_curr, N0)
                history.append(met)

                targets = pick_targets_for_attack(
                    G_curr,
                    attack_kind=attack_kind,
                    step_size=step_size,
                    seed=int(attack_seed) + step,
                    rc_frac=float(rc_frac),
                    rc_min_density=float(rc_min_density),
                    rc_max_frac=float(rc_max_frac),
                )

                if not targets:
                    nodes = list(G_curr.nodes())
                    k = min(len(nodes), step_size)
                    targets = rng.sample(nodes, k) if k > 0 else []

                G_curr.remove_nodes_from(targets)
                progress.progress((step + 1) / int(steps))

            if G_curr.number_of_nodes() > 0:
                states_3d.append(G_curr.copy())

            st.success("Готово.")

            df_hist = pd.DataFrame(history)
            if df_hist.empty:
                st.warning("Слишком быстро развалилось (почти ничего не успели измерить).")
            else:
                st.subheader("👁️ 3D сразу + динамика 3D")
                fig_anim = make_attack_3d_animation(states_3d, pos3d_attack)
                st.plotly_chart(fig_anim, use_container_width=True)

                fig_top = go.Figure()
                fig_top.add_trace(go.Scatter(x=df_hist["step"], y=df_hist["lcc_frac"], name="LCC fraction"))
                fig_top.add_trace(go.Scatter(x=df_hist["step"], y=df_hist["mod"], name="Modularity"))
                fig_top.update_layout(title="Топология", template="plotly_dark")
                st.plotly_chart(fig_top, use_container_width=True)

                fig_spec = go.Figure()
                fig_spec.add_trace(go.Scatter(x=df_hist["step"], y=df_hist["l2_lcc"], name="λ₂ on LCC"))
                fig_spec.add_trace(go.Scatter(x=df_hist["step"], y=df_hist["l2_global"], name="λ₂ global", line=dict(dash="dot")))
                fig_spec.add_trace(go.Scatter(x=df_hist["step"], y=df_hist["lmax"], name="λmax"))
                fig_spec.update_layout(title="Спектр", template="plotly_dark")
                st.plotly_chart(fig_spec, use_container_width=True)

                fig_eff = go.Figure()
                fig_eff.add_trace(go.Scatter(x=df_hist["step"], y=df_hist["eff_w"], name="Weighted Efficiency"))
                fig_eff.update_layout(title="Взвешенная эффективность", template="plotly_dark")
                st.plotly_chart(fig_eff, use_container_width=True)
