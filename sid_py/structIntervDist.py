import numpy as np
from .computePathMatrix import computePathMatrix
from .computePathMatrix2 import computePathMatrix2
from .dSepAdji import dSepAdji


def structIntervDist(trueGraph, estGraph, output=False, spars=False):
    """
    Structural Intervention Distance (SID)

    Fully correct DAG → DAG implementation
    based on Peters & Bühlmann (2015).

    Parameters
    ----------
    trueGraph : ndarray (p×p)
        True causal DAG

    estGraph : ndarray (p×p)
        Estimated DAG

    Returns
    -------
    dict with keys:
        sid
        sidLowerBound
        sidUpperBound
        incorrectMat
    """

    trueGraph = np.asarray(trueGraph, dtype=int)
    estGraph = np.asarray(estGraph, dtype=int)

    if trueGraph.shape != estGraph.shape:
        raise ValueError("trueGraph and estGraph must have same shape")

    if trueGraph.shape[0] != trueGraph.shape[1]:
        raise ValueError("Graphs must be square adjacency matrices")

    p = trueGraph.shape[0]

    # ---------------------------------------
    # Path matrix of true graph
    # ---------------------------------------
    PathMatrix = computePathMatrix(trueGraph, spars)

    incorrectMat = np.zeros((p, p), dtype=int)

    # ---------------------------------------
    # Main SID logic
    # ---------------------------------------
    for i in range(p):

        # parents of i in estimated graph
        # parents of i in estimated graph
        pa_est = np.where(estGraph[:, i] == 1)[0]

        # parents of i in TRUE graph
        pa_true = np.where(trueGraph[:, i] == 1)[0]

        # If parent sets are identical, adjustment is correct for all j
        if set(pa_est) == set(pa_true):
            continue

        # remove outgoing edges of pa_est
        PathMatrix2 = computePathMatrix2(
            trueGraph,
            pa_est,
            PathMatrix,
            spars=spars
        )

        for j in range(p):

            if i == j:
                continue

            incorrect = False

            # -------------------------------------------------
            # CASE 1:
            # j ∈ Pa_est(i)
            # -------------------------------------------------
            if j in pa_est:
                # estGraph predicts zero effect
                # incorrect if i truly causes j
                if PathMatrix[i, j] == 1:
                    incorrect = True

            # -------------------------------------------------
            # CASE 2:
            # j ∉ Pa_est(i)
            # -------------------------------------------------
            else:
                # check if Pa_est(i) is a valid adjustment set
                # using d-separation logic

                if not dSepAdji(
                    AdjMat=trueGraph,
                    x=i,
                    y=j,
                    condSet=pa_est,
                    PathMatrix=PathMatrix,
                    PathMatrix2=PathMatrix2,
                ):
                    incorrect = True

            incorrectMat[i, j] = int(incorrect)

    sid = int(np.sum(incorrectMat))

    return {
        "sid": sid,
        "sidLowerBound": sid,
        "sidUpperBound": sid,
        "incorrectMat": incorrectMat,
    }
