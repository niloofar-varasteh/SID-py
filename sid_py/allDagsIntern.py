import numpy as np


def allDagsIntern(gm: np.ndarray, a: np.ndarray, row_names: np.ndarray, tmp=None):
    """
    R-equivalent of allDagsIntern(gm, a, row.names, tmp)

    Parameters
    ----------
    gm : full adjacency matrix (p,p)
    a : submatrix of an UNDIRECTED component (k,k) (must be symmetric)
    row_names : indices (0-based) of nodes of that component in gm
    tmp : internal recursion accumulator (list of vectors)

    Returns
    -------
    tmp_arr : ndarray shape (n_dags, p*p) containing column-major vectorized gm
    """
    gm = np.asarray(gm, dtype=int)
    a = np.asarray(a, dtype=int)
    row_names = np.asarray(row_names, dtype=int)

    if tmp is None:
        tmp = []

    # sanity: must be entirely undirected
    if np.any((a + a.T) == 1):
        raise ValueError("The matrix is not entirely undirected. This should not happen!")

    if a.sum() == 0:
        vec = gm.flatten(order="F")  # R's c(gm) is column-major
        # avoid duplicates
        if not any(np.array_equal(vec, old) for old in tmp):
            tmp.append(vec)
        return np.vstack(tmp) if len(tmp) > 0 else np.empty((0, gm.size), dtype=int)

    # sinks are nodes with neighbors
    sinks = np.where(a.sum(axis=0) > 0)[0]

    for x in sinks:
        Adj = (a == 1)
        Adjx = Adj[x, :]

        # check clique condition on neighbors
        if Adjx.any():
            un = np.where(Adjx)[0]
            Adj2 = Adj[np.ix_(un, un)].copy()
            np.fill_diagonal(Adj2, True)
        else:
            Adj2 = np.array([[True]])

        if Adj2.all():
            gm2 = gm.copy()

            if Adjx.any():
                un_global = row_names[np.where(Adjx)[0]]
                sink_global = row_names[x]

                # orient all undirected neighbors -> sink
                gm2[un_global, sink_global] = 1
                gm2[sink_global, un_global] = 0

            # remove x from component
            mask = np.ones(len(row_names), dtype=bool)
            mask[x] = False
            a2 = a[np.ix_(mask, mask)]
            row_names2 = row_names[mask]

            tmp_arr = allDagsIntern(gm2, a2, row_names2, tmp)
            tmp = [row for row in tmp_arr]  # keep accumulating as list
            tmp = [np.asarray(r, dtype=int) for r in tmp]

    return np.vstack(tmp) if len(tmp) > 0 else np.empty((0, gm.size), dtype=int)
