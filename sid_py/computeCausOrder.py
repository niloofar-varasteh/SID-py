import numpy as np


def computeCausOrder(G: np.ndarray) -> np.ndarray:
    """
    R-equivalent of computeCausOrder(G).

    Finds a causal/topological order by repeatedly picking a "root"
    (node with zero indegree), choosing the smallest index in the
    current reduced graph (matches R: min(which(colSums(G)==0))).

    Parameters
    ----------
    G : (p,p) ndarray (0/1), adjacency where G[i,j]=1 means i -> j

    Returns
    -------
    order : (p,) ndarray of node indices (0-based)
    """
    G = np.asarray(G, dtype=int)
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError("G must be a square adjacency matrix")

    p = G.shape[0]
    remaining = list(range(p))
    caus_order = [None] * p

    G_work = G.copy()

    for i in range(p - 1):
        indeg = G_work.sum(axis=0)
        roots = np.where(indeg == 0)[0]
        if roots.size == 0:
            raise ValueError("Graph seems to contain a cycle (no root found).")

        root_pos = int(roots.min())  # matches R's min(which(...))
        caus_order[i] = remaining[root_pos]

        # remove that node from remaining + shrink matrix
        remaining.pop(root_pos)
        G_work = np.delete(G_work, root_pos, axis=0)
        G_work = np.delete(G_work, root_pos, axis=1)

    caus_order[p - 1] = remaining[0]
    return np.array(caus_order, dtype=int)
