import numpy as np

def randomDAG(p, probConnect, causalOrder=None):
    """
    R-equivalent randomDAG:
      - choose causalOrder if None: a random permutation of 0..p-1
      - for i = 0..p-3:
          node = causalOrder[i]
          possibleParents = causalOrder[i+1:]
          numberParents ~ Binomial(n=len(possibleParents), p=probConnect)
          parents = sample(possibleParents, numberParents, replace=False)
          add edges parents -> node
      - special case for last pair:
          node = causalOrder[p-2]
          add edge causalOrder[p-1] -> node with probability probConnect
    """
    if causalOrder is None:
        causalOrder = np.random.permutation(p)
    else:
        causalOrder = np.array(causalOrder, dtype=int)

    DAG = np.zeros((p, p), dtype=int)

    # main loop: i = 0..p-3 (matches R: 1..p-2)
    for i in range(p - 2):
        node = causalOrder[i]
        possibleParents = causalOrder[i + 1:]
        numberParents = np.random.binomial(n=len(possibleParents), p=probConnect)

        if numberParents > 0:
            parents = np.random.choice(possibleParents, size=numberParents, replace=False)
            DAG[parents, node] = 1

    # special case: last pair (matches R)
    if p >= 2:
        node = causalOrder[p - 2]
        DAG[causalOrder[p - 1], node] = np.random.binomial(1, probConnect)

    return DAG
