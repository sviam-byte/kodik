import numpy as np
import plotly.graph_objects as go

def plot_compare_runs(runs: list[dict], y_key: str, title: str):
    fig = go.Figure()
    for r in runs:
        df = r.get("df")
        if df is None or df.empty:
            continue
        if "step" not in df.columns or y_key not in df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df["step"],
            y=df[y_key],
            name=r.get("name", "run"),
        ))
    fig.update_layout(title=title, template="plotly_dark", height=360)
    return fig

def plot_lambda2_Q_phase(runs: list[dict], title: str = "Phase portrait (λ₂ vs Q)"):
    fig = go.Figure()
    for r in runs:
        df = r.get("df")
        if df is None or df.empty:
            continue
        if "l2_lcc" not in df.columns or "Q_lcc" not in df.columns:
            continue
        x = df["l2_lcc"].to_numpy(dtype=float)
        y = df["Q_lcc"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            continue
        fig.add_trace(go.Scatter(
            x=x[mask],
            y=y[mask],
            mode="lines+markers",
            name=r.get("name", "run"),
        ))
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=360,
        xaxis_title="λ₂(LCC)",
        yaxis_title="Q(LCC)",
    )
    return fig
