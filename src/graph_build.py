import numpy as np
import networkx as nx
import pandas as pd


def build_graph(df_edges: pd.DataFrame, src_col: str, dst_col: str) -> nx.Graph:
    """Build a simple undirected weighted graph from an edge table."""
    G = nx.from_pandas_edgelist(
        df_edges,
        source=src_col,
        target=dst_col,
        edge_attr=["weight", "confidence"],
        create_using=nx.Graph(),
    )

    # Ensure weights/confidence are finite and positive.
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


def filter_edges(df_raw: pd.DataFrame, min_conf: float, min_weight: float) -> pd.DataFrame:
    """Filter edges by confidence and weight thresholds."""
    mask = (df_raw["confidence"] >= float(min_conf)) & (df_raw["weight"] >= float(min_weight))
    return df_raw.loc[mask].copy()


def connected_components_sorted(G: nx.Graph) -> list[set]:
    """Return connected components sorted by size (descending)."""
    return sorted(nx.connected_components(G), key=len, reverse=True)


def lcc_subgraph(G: nx.Graph) -> nx.Graph:
    """Extract the largest connected component as a subgraph copy."""
    if G.number_of_nodes() == 0:
        return G
    comp = max(nx.connected_components(G), key=len)
    return G.subgraph(comp).copy()
