import numpy as np
import pandas as pd
import networkx as nx

def build_graph(df_edges: pd.DataFrame, src_col: str, dst_col: str) -> nx.Graph:
    G = nx.from_pandas_edgelist(
        df_edges,
        source=src_col,
        target=dst_col,
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

def pick_component(G_full: nx.Graph, comp_id: int | None) -> nx.Graph:
    if comp_id is None:
        return G_full.copy()

    comps = sorted(nx.connected_components(G_full), key=len, reverse=True)
    if not comps:
        return nx.Graph()

    comp_id = max(0, min(int(comp_id), len(comps) - 1))
    return G_full.subgraph(comps[comp_id]).copy()

def lcc_subgraph(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return G
    comp = max(nx.connected_components(G), key=len)
    return G.subgraph(comp).copy()
