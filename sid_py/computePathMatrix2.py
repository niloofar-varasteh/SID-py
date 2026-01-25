import numpy as np
import math
from scipy import sparse

def computePathMatrix2(G, condSet, PathMatrix1, spars=False):
    G = np.asarray(G, dtype=int)
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError("G must be a square adjacency matrix (p x p).")
    p = G.shape[0]

    if condSet is None or len(condSet) == 0:
        return np.asarray(PathMatrix1, dtype=int)

    G_modified = G.copy()
    # condSet باید 0-based باشد
    G_modified[list(condSet), :] = 0

    if spars:
        P = sparse.eye(p, format="csr", dtype=int) + sparse.csr_matrix(G_modified, dtype=int)
    else:
        P = np.eye(p, dtype=int) + G_modified

    k = int(math.ceil(math.log(p) / math.log(2))) if p > 1 else 0

    for _ in range(k):
        P = P @ P
        # Optional: جلوگیری از رشد عددها
        if spars:
            P.data = np.ones_like(P.data)
        else:
            P = (P > 0).astype(int)

    return (P > 0).astype(int)
