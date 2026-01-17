"""Entropy utilities for graph attribute distributions."""

from __future__ import annotations

import numpy as np
import networkx as nx


def _entropy_from_probs(p: np.ndarray) -> float:
    """Compute Shannon entropy for a probability vector."""
    p = np.asarray(p, dtype=float)
    p = p[np.isfinite(p)]
    p = p[p > 0]
    if p.size == 0:
        return float("nan")
    return float(-(p * np.log(p)).sum())


def entropy_histogram(x, bins="fd") -> float:
    """Estimate entropy from a histogram (default: Freedman–Diaconis bins)."""
    x = np.asarray(list(x), dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    hist, _ = np.histogram(x, bins=bins)
    s = hist.sum()
    if s <= 0:
        return float("nan")
    p = hist.astype(float) / float(s)
    return _entropy_from_probs(p)


def entropy_degree(G: nx.Graph) -> float:
    """Entropy of the degree distribution."""
    return entropy_histogram([d for _, d in G.degree()], bins="fd")


def entropy_weights(G: nx.Graph) -> float:
    """Entropy of edge weight distribution."""
    ws = []
    for _, _, d in G.edges(data=True):
        w = d.get("weight", 1.0)
        try:
            ws.append(float(w))
        except Exception:
            ws.append(1.0)
    return entropy_histogram(ws, bins="fd")


def entropy_confidence(G: nx.Graph) -> float:
    """Entropy of edge confidence distribution."""
    cs = []
    for _, _, d in G.edges(data=True):
        c = d.get("confidence", 1.0)
        try:
            cs.append(float(c))
        except Exception:
            cs.append(1.0)
    return entropy_histogram(cs, bins="fd")


def triangle_support_edge(G: nx.Graph):
    """Count how many triangles each edge participates in (heavy for large graphs)."""
    tri = nx.triangles(G)
    out = []
    for u, v in G.edges():
        out.append(min(tri.get(u, 0), tri.get(v, 0)))
    return out


def entropy_triangle_support(G: nx.Graph) -> float:
    """Entropy of triangle-support distribution (edge triangle counts)."""
    try:
        ts = triangle_support_edge(G)
    except Exception:
        return float("nan")
    return entropy_histogram(ts, bins="fd")
