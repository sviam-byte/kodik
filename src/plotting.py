import pandas as pd
import plotly.graph_objects as go


def fig_metrics_over_steps(df_hist: pd.DataFrame, title: str = "") -> go.Figure:
    """Plot key metrics over removal steps for a single experiment."""
    fig = go.Figure()
    if df_hist is None or df_hist.empty:
        fig.update_layout(title="empty")
        return fig

    x = df_hist["removed_frac"] if "removed_frac" in df_hist.columns else df_hist["step"]

    fig.add_trace(go.Scatter(x=x, y=df_hist["lcc_frac"], name="LCC fraction"))
    if "mod" in df_hist.columns:
        fig.add_trace(go.Scatter(x=x, y=df_hist["mod"], name="Modularity Q"))
    if "l2_lcc" in df_hist.columns:
        fig.add_trace(go.Scatter(x=x, y=df_hist["l2_lcc"], name="λ₂ (LCC)"))
    if "eff_w" in df_hist.columns:
        fig.add_trace(go.Scatter(x=x, y=df_hist["eff_w"], name="Efficiency (w)"))

    fig.update_layout(template="plotly_dark", title=title, xaxis_title="removed_frac", yaxis_title="value")
    return fig


def fig_compare_attacks(
    curves: list[tuple[str, pd.DataFrame]],
    x_col: str,
    y_col: str,
    title: str,
) -> go.Figure:
    """Compare multiple experiment curves on a shared axis."""
    fig = go.Figure()
    for name, df in curves:
        if df is None or df.empty or x_col not in df.columns or y_col not in df.columns:
            continue
        fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], name=name))
    fig.update_layout(template="plotly_dark", title=title, xaxis_title=x_col, yaxis_title=y_col)
    return fig


def fig_compare_graphs_scalar(df_cmp: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    """Compare graph-level scalar metrics as a bar chart."""
    fig = go.Figure()
    if df_cmp is None or df_cmp.empty:
        fig.update_layout(title="empty")
        return fig
    fig.add_trace(go.Bar(x=df_cmp[x], y=df_cmp[y], name=y))
    fig.update_layout(template="plotly_dark", title=title, xaxis_title=x, yaxis_title=y)
    return fig
