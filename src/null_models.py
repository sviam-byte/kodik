import random
import networkx as nx


def gnm_null_model(G: nx.Graph, seed: int) -> nx.Graph:
    """
    Erdos-Renyi G(n, m) preserving N and E.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    H = nx.gnm_random_graph(n, m, seed=int(seed))
    # relabel to original node IDs (keep UI consistent)
    nodes = list(G.nodes())
    mapping = {i: nodes[i] for i in range(len(nodes))}
    H = nx.relabel_nodes(H, mapping)
    return H


def configuration_null_model(G: nx.Graph, seed: int) -> nx.Graph:
    """
    Configuration model preserving degree sequence (topology only).
    Weighted degree distribution is NOT preserved, only degrees.
    """
    degs = [d for _, d in G.degree()]
    # networkx uses numpy RNG internally; seed is ok
    M = nx.configuration_model(degs, seed=int(seed))
    H = nx.Graph(M)  # remove multi-edges by collapsing
    H.remove_edges_from(nx.selfloop_edges(H))

    # relabel to original IDs
    nodes = list(G.nodes())
    mapping = {i: nodes[i] for i in range(len(nodes))}
    H = nx.relabel_nodes(H, mapping)
    return H


def rewire_mix(G: nx.Graph, p: float, seed: int, swaps_per_edge: float = 0.5) -> nx.Graph:
    """
    "Mix original with random" via double-edge swaps.
    p=0 -> original
    p=1 -> lots of rewiring

    We do:
      ns = int(p * swaps_per_edge * E)
      double_edge_swap preserves degree sequence.
    """
    H = G.copy()
    E = H.number_of_edges()
    if E < 2:
        return H

    ns = int(float(p) * float(swaps_per_edge) * float(E))
    ns = max(0, ns)

    if ns == 0:
        return H

    # double_edge_swap may fail if graph too constrained -> we allow max_tries
    nx.double_edge_swap(H, nswap=ns, max_tries=ns * 20, seed=int(seed))
    return H


def copy_weights_from_original(G_orig: nx.Graph, H: nx.Graph, seed: int) -> nx.Graph:
    """
    Optional: assign weights to null-model edges by sampling weights from original edge weights.
    Not "scientifically unique", but keeps weight distribution comparable.
    """
    import random

    rng = random.Random(int(seed))
    ws = [float(d.get("weight", 1.0)) for _, _, d in G_orig.edges(data=True)]
    if not ws:
        return H

    for u, v in H.edges():
        H[u][v]["weight"] = float(rng.choice(ws))
        H[u][v]["confidence"] = 0.0

    return H
