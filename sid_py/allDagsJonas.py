import numpy as np
from .allDagsIntern import allDagsIntern


def allDagsJonas(adj: np.ndarray, row_names):
    """
    R-equivalent of allDagsJonas(adj, row.names)

    Parameters
    ----------
    adj : (p,p) adjacency matrix containing an UNDIRECTED component inside row_names
    row_names : indices (0-based) specifying the undirected component to extend

    Returns
    -------
    ndarray:
      - if invalid (component not fully undirected): returns -1
      - else returns (n_dags, p*p) matrix of vectorized DAGs
    """
    adj = np.asarray(adj, dtype=int)
    row_names = np.asarray(row_names, dtype=int)

    a = adj[np.ix_(row_names, row_names)]

    # If any directed edge inside supposed-undirected component => return -1
    if np.any((a + a.T) == 1):
        return -1

    return allDagsIntern(adj, a, row_names, tmp=None)
