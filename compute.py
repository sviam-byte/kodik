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

    We hash a deterministic serialization of graph, node, and edge attributes
    so the result is stable across runs and process boundaries.
    """
    h = hashlib.sha256()

    h.update(b"directed=")
    h.update(b"1" if G.is_directed() else b"0")
    h.update(b"\n")

    def _hash_value(value) -> None:
        """Update the hash with a stable serialization of common types."""
        if isinstance(value, dict):
            h.update(b"{")
            for key in sorted(value.keys(), key=lambda x: str(x)):
                h.update(str(key).encode("utf-8", errors="replace"))
                h.update(b":")
                _hash_value(value[key])
                h.update(b",")
            h.update(b"}")
            return
        if isinstance(value, (list, tuple)):
            h.update(b"[")
            for item in value:
                _hash_value(item)
                h.update(b",")
            h.update(b"]")
            return
        if isinstance(value, (set, frozenset)):
            h.update(b"set(")
            for item in sorted(value, key=lambda x: str(x)):
                _hash_value(item)
                h.update(b",")
            h.update(b")")
            return
        if isinstance(value, np.ndarray):
            h.update(b"ndarray:")
            h.update(str(value.dtype).encode("utf-8", errors="replace"))
            h.update(b":")
            h.update(str(value.shape).encode("utf-8", errors="replace"))
            h.update(b":")
            h.update(value.tobytes())
            return

        h.update(repr(value).encode("utf-8", errors="replace"))

    # Graph-level attributes are part of the cache key.
    if G.graph:
        h.update(b"graph:")
        _hash_value(G.graph)
        h.update(b"\n")

    # Nodes are hashed in a stable order by stringified label.
    for n, attrs in sorted(G.nodes(data=True), key=lambda x: str(x[0])):
        h.update(b"N:")
        h.update(str(n).encode("utf-8", errors="replace"))
        if attrs:
            h.update(b"|")
            _hash_value(attrs)
        h.update(b"\n")

    # Edges + attributes (sorted for determinism across graph types).
    def _edge_pair(u, v):
        if G.is_directed():
            return (str(u), str(v))
        ordered = sorted((str(u), str(v)))
        return (ordered[0], ordered[1])

    if G.is_multigraph():
        def _edge_key(e):
            u, v, k, _d = e
            return (*_edge_pair(u, v), str(k))

        edges = G.edges(keys=True, data=True)
    else:
        def _edge_key(e):
            u, v, _d = e
            return _edge_pair(u, v)

        edges = G.edges(data=True)

    for edge in sorted(edges, key=_edge_key):
        if G.is_multigraph():
            u, v, k, d = edge
        else:
            u, v, d = edge
            k = None

        h.update(b"E:")
        h.update(str(u).encode("utf-8", errors="replace"))
        h.update(b"->")
        h.update(str(v).encode("utf-8", errors="replace"))
        if k is not None:
            h.update(b"#")
            h.update(str(k).encode("utf-8", errors="replace"))
        h.update(b"|")
        if d:
            _hash_value(d)
        h.update(b"\n")

    return h.hexdigest()


@st.cache_resource(show_spinner=False)
def build_graph(df: pd.DataFrame) -> nx.Graph:
    """Build a NetworkX graph from a pandas edge list.

    This is intentionally cached because turning large dataframes into graphs
    is one of the most expensive UI steps.
    """
    return nx.from_pandas_edgelist(df, "src", "dst", edge_attr=True)


_NX_HASH_FUNCS = {
    nx.Graph: _hash_nx_graph,
    nx.DiGraph: _hash_nx_graph,
    nx.MultiGraph: _hash_nx_graph,
    nx.MultiDiGraph: _hash_nx_graph,
}


@st.cache_data(show_spinner=False, hash_funcs=_NX_HASH_FUNCS)
def compute_layout(G: nx.Graph) -> dict:
    """Compute and cache a deterministic 2D layout for quick preview plots."""
    return nx.spring_layout(G, seed=42)


@st.cache_data(show_spinner=False, hash_funcs=_NX_HASH_FUNCS)
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
