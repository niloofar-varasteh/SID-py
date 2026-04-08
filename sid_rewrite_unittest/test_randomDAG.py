import numpy as np
import unittest

def randomDAG(p, probConnect, causalOrder=None):
    """
    Generates a random Directed Acyclic Graph based on a causal order.
    Edges are sampled using a binomial distribution for each node pair.
    """
    if causalOrder is None:
        causalOrder = np.random.permutation(p)

    DAG = np.zeros((p, p), dtype=int)
    for i in range(p - 1):
        node = causalOrder[i]
        possibleParents = causalOrder[(i + 1):p]
        num_possible = len(possibleParents)

        if num_possible > 0:
            # Randomly choose number of parents using binomial distribution
            numParents = np.random.binomial(n=num_possible, p=probConnect)
            if numParents > 0:
                parents = np.random.choice(possibleParents, size=numParents, replace=False)
                DAG[parents, node] = 1
    return DAG

class TestRandomDAG(unittest.TestCase):
    def test_default_causal_order(self):
        # Coverage: Triggers 'if causalOrder is None'
        res = randomDAG(5, 0.3)
        self.assertEqual(res.shape, (5, 5))

    def test_zero_parents(self):
        # Coverage: Ensures the logic handles cases with no connections correctly
        res = randomDAG(3, 0.0)
        self.assertEqual(np.sum(res), 0)