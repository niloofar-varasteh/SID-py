import numpy as np
from .computePathMatrix import computePathMatrix
from .computePathMatrix2 import computePathMatrix2


def structIntervDist(trueGraph, estGraph, output=False, spars=False):
    """
    Computes the Structural Intervention Distance (SID) between two graphs.
    Implementation based on structIntervDist.R
    """
    # Initialize matrices
    trueGraph = np.array(trueGraph)
    estGraph = np.array(estGraph)
    p = trueGraph.shape[1]

    incorrectInt = np.zeros((p, p))
    correctInt = np.zeros((p, p))

    # Step 1: Compute path matrix for the true graph
    # entry (i,j) is 1 if there is a directed path from i to j
    path_matrix = computePathMatrix(trueGraph, spars)

    # Initial result structure
    results = {
        "sid": 0,
        "sidUpperBound": 0,
        "sidLowerBound": 0,
        "incorrectMat": None
    }

    # Logical core: Iterate through nodes to check parent adjustment sets
    for i in range(p):
        # Parents of i in the estimated graph
        # Note: In Python, we use 0-based indexing
        pa_est = np.where(estGraph[:, i] == 1)[0]

        # Compute PathMatrix2 for this specific intervention
        path_matrix2 = computePathMatrix2(trueGraph, pa_est, path_matrix, spars)

        # Simplified Logic for DAG comparison (further helper functions needed for full CPDAG support)
        for j in range(p):
            if i == j:
                continue

            # If true graph has no path, but estimated graph predicts one incorrectly, etc.
            # This follows the specific causal checks in structIntervDist.R
            is_incorrect = False

            # Check if j is reachable from i in trueGraph
            has_path = path_matrix[i, j] > 0

            # Basic SID rule: check if parent adjustment is valid
            # (This part requires the dSepAdji logic from the R source)
            # For now, we implement the primary path-based check:
            if not has_path and (j in pa_est):
                is_incorrect = True

            if is_incorrect:
                incorrectInt[i, j] = 1
            else:
                correctInt[i, j] = 1

    results["sid"] = int(np.sum(incorrectInt))
    results["incorrectMat"] = incorrectInt
    results["sidLowerBound"] = results["sid"]  # For DAGs, lower=upper=sid
    results["sidUpperBound"] = results["sid"]

    return results