import streamlit as st

import hashlib

import networkx as nx
import numpy as np
import pandas as pd

from src.robust_geom import fragility_from_curvature, ollivier_ricci_summary


def _hash_nx_graph(G: nx.Graph) -> str:
    """Return a stable hash for a NetworkX graph.

    Streamlit's cache needs a hashable representation of function arguments.
    NetworkX graphs contain mutable dicts, so Streamlit's default hasher may
    fail with UnhashableParamError.

    We hash a deterministic serialization of nodes and edges including edge
    attributes.
    """
    h = hashlib.sha256()

    h.update(b"directed=")
    h.update(b"1" if G.is_directed() else b"0")
    h.update(b"\n")

    # Nodes
    for n in sorted(G.nodes(), key=lambda x: str(x)):
        h.update(b"N:")
        h.update(str(n).encode("utf-8", errors="replace"))
        h.update(b"\n")

    # Edges + attributes (sorted for determinism)
    def _edge_key(e):
        u, v, _d = e
        return (str(u), str(v))

    for u, v, d in sorted(G.edges(data=True), key=_edge_key):
        h.update(b"E:")
        h.update(str(u).encode("utf-8", errors="replace"))
        h.update(b"->")
        h.update(str(v).encode("utf-8", errors="replace"))
        h.update(b"|")
        if d:
            for k in sorted(d.keys(), key=lambda x: str(x)):
                h.update(str(k).encode("utf-8", errors="replace"))
                h.update(b"=")
                h.update(repr(d.get(k)).encode("utf-8", errors="replace"))
                h.update(b";")
        h.update(b"\n")

    return h.hexdigest()


@st.cache_resource(show_spinner=False)
def build_graph(df: pd.DataFrame) -> nx.Graph:
    """Build a NetworkX graph from a pandas edge list.

    This is intentionally cached because turning large dataframes into graphs
    is one of the most expensive UI steps.
    """
    return nx.from_pandas_edgelist(df, "src", "dst", edge_attr=True)


@st.cache_data(show_spinner=False, hash_funcs={nx.Graph: _hash_nx_graph})
def compute_layout(G: nx.Graph) -> dict:
    """Compute and cache a deterministic 2D layout for quick preview plots."""
    return nx.spring_layout(G, seed=42)


@st.cache_data(show_spinner=False, hash_funcs={nx.Graph: _hash_nx_graph})
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
