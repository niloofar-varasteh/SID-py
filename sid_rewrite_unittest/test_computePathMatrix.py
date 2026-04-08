import numpy as np
import unittest

def computePathMatrix(G, spars=False):
    """ Computes the transitive closure (reachability) matrix. """
    p = G.shape[0]
    if p == 0: return np.array([[]])
    path_mat = np.eye(p, dtype=int) + G.astype(int)
    k = int(np.ceil(np.log2(p))) if p > 1 else 0
    for _ in range(k):
        path_mat = (np.dot(path_mat, path_mat) > 0).astype(int)
    return path_mat

class TestPathMatrix(unittest.TestCase):
    def test_p_one(self):
        # Coverage: Tests p=1 branch (k=0)
        G = np.array([[0]])
        res = computePathMatrix(G)
        self.assertEqual(res[0, 0], 1)

    def test_chain(self):
        G = np.array([[0, 1], [0, 0]])
        res = computePathMatrix(G)
        self.assertEqual(res[0, 1], 1)