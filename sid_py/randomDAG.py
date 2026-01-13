import numpy as np
from scipy import sparse


def randomDAG(p, probConnect, causalOrder=None):
    """
    Generates the adjacency matrix of a randomly generated directed acyclic graph (DAG). [cite: 4]

    Args:
        p (int): Number of nodes. [cite: 6]
        probConnect (float): Probability of connecting two nodes.
        causalOrder (list, optional): Causal or topological order of the nodes.
                                      If None, it's chosen randomly. [cite: 9]

    Returns:
        numpy.ndarray or scipy.sparse.csr_matrix: Adjacency matrix where entry (i,j)
                                                  is 1 if there is an edge from i to j. [cite: 10, 11]
    """
    if causalOrder is None:
        # Generate a random topological order
        causalOrder = np.random.permutation(p)
    else:
        # Ensure it's a numpy array for indexing
        causalOrder = np.array(causalOrder)

    # Initialize a sparse matrix as in the R source code
    DAG = np.zeros((p, p))

    # Iterate through nodes to add edges based on the causal order
    for i in range(p - 1):
        node = causalOrder[i]
        # Possible parents are nodes later in the causal order
        possibleParents = causalOrder[(i + 1):p]
        num_possible = len(possibleParents)

        # Determine the number of parents using a binomial distribution
        numberParents = np.random.binomial(n=num_possible, p=probConnect)

        if numberParents > 0:
            # Randomly select which nodes will be parents
            parents = np.random.choice(possibleParents, size=numberParents, replace=False)
            # In R code: DAG[Parents, node] <- 1
            DAG[parents, node] = 1

    return DAG