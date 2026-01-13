import numpy as np


def hammingDist(G1, G2, allMistakesOne=True):
    """
    Computes the Hamming distance between two graph objects.
    """
    G1 = np.array(G1)
    G2 = np.array(G2)

    if allMistakesOne:
        # If an edge is reversed (X->Y vs X<-Y), it counts as one mistake [cite: 72]
        Gtmp = (G1 + G2) % 2
        Gtmp = Gtmp + Gtmp.T
        nrReversals = np.sum(Gtmp == 2) / 2
        nrInclDel = np.sum(Gtmp == 1) / 2
        hammingDis = nrReversals + nrInclDel
    else:
        # Standard absolute difference with correction for undirected edges
        hammingDis = np.sum(np.abs(G1 - G2))
        # correction: dist(-,.) = 1, not 2
        term1 = G1 * G1.T * (1 - G2) * (1 - G2.T)
        term2 = G2 * G2.T * (1 - G1) * (1 - G1.T)
        hammingDis = hammingDis - 0.5 * np.sum(term1 + term2)

    return int(hammingDis)