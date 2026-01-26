import plotly.graph_objects as go
import networkx as nx


def compute_3d_layout(G: nx.Graph, seed: int) -> dict:
    """Compute a deterministic 3D spring layout for visualization."""
    if G.number_of_nodes() == 0:
        return {}
    return nx.spring_layout(G, dim=3, weight="weight", seed=int(seed))


def make_3d_traces(G: nx.Graph, pos3d: dict, show_scale: bool = False):
    """Create Plotly 3D traces for nodes and edges.

    Returns:
        (edge_traces, node_trace)
        edge_traces is always a list (possibly empty) so callers can safely do [*edge_traces, node_trace].
    """
    if G.number_of_nodes() == 0:
        return [], None

    strength = dict(G.degree(weight="weight"))
    nodes = [n for n in G.nodes() if n in pos3d]

    if not nodes:
        return [], None

    xs = [pos3d[n][0] for n in nodes]
    ys = [pos3d[n][1] for n in nodes]
    zs = [pos3d[n][2] for n in nodes]
    colors = [strength.get(n, 0.0) for n in nodes]
    texts = [f"{n} | strength={strength.get(n, 0.0):.3f}" for n in nodes]

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

    edge_traces = []
    if ex:
        edge_traces.append(
            go.Scatter3d(
                x=ex, y=ey, z=ez,
                mode="lines",
                line=dict(width=1),
                hoverinfo="none",
                name="edges",
            )
        )

    return edge_traces, node_trace



def plot_attack_curves(experiments: list[dict], y_key: str, title: str) -> go.Figure:
    """Plot comparison curves for stored attack experiments."""
    fig = go.Figure()
    for exp in experiments:
        df = exp["df"]
        label = exp["name"]
        if y_key not in df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df["removed_frac"],
            y=df[y_key],
            mode="lines",
            name=label,
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Removed fraction",
        template="plotly_dark",
    )
    return fig
