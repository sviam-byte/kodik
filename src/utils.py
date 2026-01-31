"""Shared utility helpers for graph normalization and safe casting."""

import networkx as nx
import numpy as np


def as_simple_undirected(G: nx.Graph) -> nx.Graph:
    """
    Приводит граф к простому неориентированному виду.
    Для MultiGraph суммирует веса параллельных ребер и сохраняет атрибуты первого ребра.
    """
    H = G
    if hasattr(H, "is_directed") and H.is_directed():
        H = H.to_undirected(as_view=False)

    if isinstance(H, (nx.MultiGraph, nx.MultiDiGraph)):
        simple = nx.Graph()
        simple.add_nodes_from(H.nodes(data=True))
        for u, v, d in H.edges(data=True):
            w = d.get("weight", 1.0)
            try:
                w = float(w)
            except (ValueError, TypeError):
                w = 1.0

            if simple.has_edge(u, v):
                simple[u][v]["weight"] = float(simple[u][v].get("weight", 0.0)) + w
            else:
                edge_attrs = dict(d)
                edge_attrs["weight"] = w
                simple.add_edge(u, v, **edge_attrs)
        return simple

    return nx.Graph(H)


def safe_float(x, default: float = 0.0) -> float:
    """Безопасное приведение к float."""
    try:
        val = float(x)
        return val if np.isfinite(val) else default
    except (ValueError, TypeError):
        return default
