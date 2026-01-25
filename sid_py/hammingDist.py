import numpy as np

def hammingDist(G1, G2, allMistakesOne=True):
    G1 = np.asarray(G1)
    G2 = np.asarray(G2)

    if G1.shape != G2.shape or G1.ndim != 2 or G1.shape[0] != G1.shape[1]:
        raise ValueError("G1 and G2 must be square matrices of the same shape.")

    # enforce binary adjacency (optional but recommended)
    G1 = (G1 != 0).astype(int)
    G2 = (G2 != 0).astype(int)

    if allMistakesOne:
        Gtmp = (G1 + G2) % 2
        Gtmp = Gtmp + Gtmp.T
        nrReversals = np.sum(Gtmp == 2) / 2
        nrInclDel = np.sum(Gtmp == 1) / 2
        hammingDis = nrReversals + nrInclDel
    else:
        hammingDis = np.sum(np.abs(G1 - G2))
        term1 = G1 * G1.T * (1 - G2) * (1 - G2.T)
        term2 = G2 * G2.T * (1 - G1) * (1 - G1.T)
        hammingDis = hammingDis - 0.5 * np.sum(term1 + term2)

    return int(round(hammingDis))
