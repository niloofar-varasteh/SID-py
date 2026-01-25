import numpy as np
import math
from scipy import sparse

def computePathMatrix(G, spars=False):
    """
    Behavioral match to SID/R computePathMatrix:
      PathMatrix = I + G
      repeat k=ceil(log(p)/log(2)) times: PathMatrix = PathMatrix %*% PathMatrix
      return (PathMatrix > 0) as 0/1 matrix
    """
    G = np.asarray(G)
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError("G must be a square adjacency matrix (p x p).")

    p = G.shape[0]

    if p > 3000 and not spars:
        print("Warning: Maybe you should use the sparse version by using spars=TRUE")

    if spars:
        P = sparse.eye(p, format="csr", dtype=int) + sparse.csr_matrix(G, dtype=int)
    else:
        P = (np.eye(p, dtype=int) + G.astype(int))

    # R: k <- ceiling(log(p)/log(2))
    k = int(math.ceil(math.log(p) / math.log(2))) if p > 1 else 0

    for _ in range(k):
        P = P @ P

        # Optional (safe) booleanization each step to prevent blow-up.
        # It doesn't change the final (P>0) result.
        if spars:
            P.data = np.ones_like(P.data)
        else:
            P = (P > 0).astype(int)

    # Final threshold (R does this at the end)
    return (P > 0).astype(int)
