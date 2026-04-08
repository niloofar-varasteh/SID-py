import numpy as np
import unittest

def computeCausOrder(G):
    """ Computes topological order of a DAG. Throws error if cycle exists. """
    p = G.shape[0]
    remaining = list(range(p))
    causOrder = []
    temp_G = G.astype(float).copy()
    for _ in range(p):
        in_degrees = np.sum(temp_G, axis=0)
        found = False
        for i in range(p):
            if i in remaining and in_degrees[i] == 0:
                causOrder.append(i)
                remaining.remove(i)
                temp_G[i, :], temp_G[:, i] = 0, np.nan
                found = True
                break
        if not found and len(remaining) > 0:
            raise ValueError("Cycle detected")
    return np.array(causOrder)

class TestCausOrder(unittest.TestCase):
    def test_chain(self):
        G = np.array([[0, 1], [0, 0]])
        self.assertTrue(np.array_equal(computeCausOrder(G), [0, 1]))

    def test_cycle_coverage(self):
        # Coverage: Triggers the ValueError
        G = np.array([[0, 1], [1, 0]])
        with self.assertRaises(ValueError):
            computeCausOrder(G)