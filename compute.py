import streamlit as st

import networkx as nx
import numpy as np
import pandas as pd

from src.robust_geom import fragility_from_curvature, ollivier_ricci_summary


@st.cache_resource(show_spinner=False)
def build_graph(df: pd.DataFrame) -> nx.Graph:
    """Build a NetworkX graph from a pandas edge list.

    This is intentionally cached because turning large dataframes into graphs
    is one of the most expensive UI steps.
    """
    return nx.from_pandas_edgelist(df, "src", "dst", edge_attr=True)


@st.cache_data(show_spinner=False)
def compute_layout(G: nx.Graph) -> dict:
    """Compute and cache a deterministic 2D layout for quick preview plots."""
    return nx.spring_layout(G, seed=42)


@st.cache_data(show_spinner=False)
def compute_curvature(
    G: nx.Graph,
    sample_edges: int = 150,
    seed: int = 42,
    max_support: int = 60,
    cutoff: float = 8.0,
) -> dict:
    """Compute Ollivier–Ricci curvature summary metrics for a graph.

    The return shape mirrors the metrics dict used in the app so results can be
    merged into the cached metrics payload when the user explicitly requests it.
    """
    if G.number_of_edges() == 0:
        return {
            "kappa_mean": float("nan"),
            "kappa_median": float("nan"),
            "kappa_frac_negative": float("nan"),
            "kappa_computed_edges": 0,
            "kappa_skipped_edges": 0,
            "fragility_kappa": float("nan"),
        }

    try:
        curv = ollivier_ricci_summary(
            G,
            sample_edges=int(sample_edges),
            seed=int(seed),
            max_support=int(max_support),
            cutoff=float(cutoff),
        )
    except Exception:
        return {
            "kappa_mean": float("nan"),
            "kappa_median": float("nan"),
            "kappa_frac_negative": float("nan"),
            "kappa_computed_edges": 0,
            "kappa_skipped_edges": 0,
            "fragility_kappa": float("nan"),
        }

    kappa_mean = float(curv.kappa_mean) if np.isfinite(curv.kappa_mean) else float("nan")
    fragility_kappa = (
        float(fragility_from_curvature(kappa_mean)) if np.isfinite(kappa_mean) else float("nan")
    )

    return {
        "kappa_mean": kappa_mean,
        "kappa_median": float(curv.kappa_median) if np.isfinite(curv.kappa_median) else float("nan"),
        "kappa_frac_negative": float(curv.kappa_frac_negative)
        if np.isfinite(curv.kappa_frac_negative)
        else float("nan"),
        "kappa_computed_edges": int(curv.computed_edges),
        "kappa_skipped_edges": int(curv.skipped_edges),
        "fragility_kappa": fragility_kappa,
    }
