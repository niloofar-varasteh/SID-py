import numpy as np
import math
from scipy import sparse


def computePathMatrix(G, spars=False):
    """
    Python implementation of computePathMatrix.R
    """
    p = G.shape[1]

    if p > 3000 and not spars:
        print("Warning: Maybe you should use the sparse version by using spars=TRUE")

    if spars:
        # Equivalent to the 'Matrix' library in R
        path_matrix = sparse.eye(p, format='csr') + sparse.csr_matrix(G)
    else:
        path_matrix = np.eye(p) + np.array(G)

    k = math.ceil(math.log(p) / math.log(2))
    for _ in range(k):
        path_matrix = path_matrix @ path_matrix
        if spars:
            path_matrix.data = np.ones_like(path_matrix.data)
        else:
            path_matrix = (path_matrix > 0).astype(float)

            # At the end of the computePathMatrix function
            if spars and sparse.issparse(path_matrix):
                return (path_matrix > 0).astype(int)
            else:
                return (path_matrix > 0).astype(int)

    return (path_matrix > 0).astype(int)