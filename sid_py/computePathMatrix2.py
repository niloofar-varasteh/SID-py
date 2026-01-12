import numpy as np
import math
from scipy import sparse


def computePathMatrix2(G, condSet, PathMatrix1, spars=False):
    """
    Python implementation of computePathMatrix2.R.
    This function computes a path matrix after removing all edges
    that leave the nodes in the condition set (condSet).
    """
    p = G.shape[1]

    # If the condition set is not empty, remove outgoing edges from those nodes
    if len(condSet) > 0:
        G_modified = np.array(G, copy=True)

        # Note: Indices in Python start from 0.
        # Ensure condSet contains 0-based indices.
        G_modified[condSet, :] = 0

        if spars:
            # Using scipy.sparse for efficient computation
            path_matrix2 = sparse.eye(p, format='csr') + sparse.csr_matrix(G_modified)
        else:
            # Using dense numpy arrays
            path_matrix2 = np.eye(p) + G_modified

        # Matrix squaring to find all paths (Transitive Closure)
        k = math.ceil(math.log(p) / math.log(2))
        for _ in range(k):
            path_matrix2 = path_matrix2 @ path_matrix2
            if spars:
                path_matrix2.data = np.ones_like(path_matrix2.data)
            else:
                path_matrix2 = (path_matrix2 > 0).astype(float)

        return (path_matrix2 > 0).astype(int)
    else:
        # If condSet is empty, return the original PathMatrix1 as per R implementation
        return PathMatrix1