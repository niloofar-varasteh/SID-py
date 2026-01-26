import numpy as np


def dag2cpdagAdj(Adj: np.ndarray) -> np.ndarray:
    """
    Approx-equivalent of R dag2cpdagAdj() for DAG inputs.

    Output is CPDAG adjacency matrix:
      - directed i->j : M[i,j]=1, M[j,i]=0
      - undirected i-j : M[i,j]=M[j,i]=1
    """
    Adj = np.asarray(Adj, dtype=int)
    if Adj.ndim != 2 or Adj.shape[0] != Adj.shape[1]:
        raise ValueError("Adj must be a square adjacency matrix")

    p = Adj.shape[0]
    if Adj.sum() == 0:
        return Adj.copy()

    # skeleton
    skel = ((Adj + Adj.T) > 0).astype(int)

    # start with all edges undirected
    cp = skel.copy()

    def is_adj(a, b):
        return cp[a, b] == 1 or cp[b, a] == 1

    def is_dir(a, b):
        return cp[a, b] == 1 and cp[b, a] == 0

    def is_undir(a, b):
        return cp[a, b] == 1 and cp[b, a] == 1

    def orient(a, b):
        # orient a -> b
        cp[a, b] = 1
        cp[b, a] = 0

    # 1) orient unshielded colliders from original DAG
    for k in range(p):
        parents = np.where(Adj[:, k] == 1)[0]
        if len(parents) < 2:
            continue
        for i in range(len(parents)):
            for j in range(i + 1, len(parents)):
                a = parents[i]
                b = parents[j]
                if skel[a, b] == 0:  # unshielded
                    orient(a, k)
                    orient(b, k)

    # 2) Meek rules (subset R1-R3 usually sufficient)
    changed = True
    while changed:
        changed = False

        # R1: a->b and b-c undirected and a not adjacent c => b->c
        for a in range(p):
            for b in range(p):
                if not is_dir(a, b):
                    continue
                for c in range(p):
                    if c == a or c == b:
                        continue
                    if is_undir(b, c) and not is_adj(a, c):
                        orient(b, c)
                        changed = True

        # R2: a->b and b->c and a-c undirected => a->c
        for a in range(p):
            for b in range(p):
                if not is_dir(a, b):
                    continue
                for c in range(p):
                    if c == a or c == b:
                        continue
                    if is_dir(b, c) and is_undir(a, c):
                        orient(a, c)
                        changed = True

        # R3: a-b undirected, and there exist c,d with c->b, d->b, c and d not adjacent,
        # and a-c undirected, a-d undirected => a->b
        for a in range(p):
            for b in range(p):
                if a == b or not is_undir(a, b):
                    continue
                preds = [c for c in range(p) if is_dir(c, b)]
                if len(preds) < 2:
                    continue
                found = False
                for i in range(len(preds)):
                    for j in range(i + 1, len(preds)):
                        c = preds[i]
                        d = preds[j]
                        if is_adj(c, d):
                            continue
                        if is_undir(a, c) and is_undir(a, d):
                            found = True
                            break
                    if found:
                        break
                if found:
                    orient(a, b)
                    changed = True

    return cp.astype(int)
