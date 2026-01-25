import numpy as np
import networkx as nx


def dSepAdji(AdjMat, x, y, condSet, PathMatrix=None, PathMatrix2=None):
    """
    d-separation test (DAG): checks whether X and Y are d-separated given condSet.

    Parameters
    ----------
    AdjMat : (p,p) array-like
        Adjacency matrix of a DAG (1 means i -> j)
    x, y : int
        Node indices (0-based)
    condSet : array-like
        Conditioning set (0-based indices)
    PathMatrix, PathMatrix2 : optional
        Ignored here (kept for API-compatibility with R/SID)

    Returns
    -------
    bool
        True  => d-separated (independent given condSet)
        False => d-connected (dependent given condSet)
    """
    AdjMat = np.asarray(AdjMat, dtype=int)
    p = AdjMat.shape[0]
    condSet = set([] if condSet is None else list(condSet))

    # Build DAG
    G = nx.DiGraph()
    G.add_nodes_from(range(p))
    edges = np.argwhere(AdjMat == 1)
    G.add_edges_from((int(i), int(j)) for i, j in edges)

    # Moralize the ancestral graph of {x,y} ∪ condSet
    nodes_of_interest = {x, y} | condSet

    # ancestors of nodes_of_interest (including themselves)
    anc = set()
    for n in nodes_of_interest:
        anc |= nx.ancestors(G, n)
        anc.add(n)

    H = G.subgraph(anc).copy()

    # Moralization: connect parents of each node
    UG = nx.Graph()
    UG.add_nodes_from(H.nodes())
    UG.add_edges_from(H.to_undirected().edges())

    for child in H.nodes():
        parents = list(H.predecessors(child))
        for i in range(len(parents)):
            for j in range(i + 1, len(parents)):
                UG.add_edge(parents[i], parents[j])

    # Remove conditioned nodes
    UG.remove_nodes_from(condSet)

    # If x and y are disconnected => d-separated
    return not nx.has_path(UG, x, y)
