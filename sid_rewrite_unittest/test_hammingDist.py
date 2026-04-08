import numpy as np
import unittest


def computePathMatrix(G):
    """ Helper to compute reachability via matrix powers. """
    p = G.shape[0]
    path_mat = np.eye(p, dtype=int) + G.astype(int)
    k = int(np.ceil(np.log2(p))) if p > 1 else 0
    for _ in range(k):
        path_mat = (np.dot(path_mat, path_mat) > 0).astype(int)
    return path_mat


def computePathMatrix2(G, condSet, PathMatrix1, spars=False):
    """
    Computes path matrix after removing outgoing edges from conditioning set.
    If condSet is empty, returns the pre-computed PathMatrix1.
    """
    if len(condSet) == 0:
        return PathMatrix1

    p = G.shape[0]
    G_mod = G.copy()
    G_mod[condSet, :] = 0
    return computePathMatrix(G_mod)


class TestComputePathMatrix2(unittest.TestCase):
    def test_empty_condSet(self):
        # Coverage: Triggers the 'if len(condSet) == 0' branch
        G = np.array([[0, 1], [0, 0]])
        P1 = np.array([[1, 1], [0, 1]])
        res = computePathMatrix2(G, [], P1)
        self.assertTrue(np.array_equal(res, P1))

    def test_path_blocking(self):
        # Coverage: Triggers the edge removal logic
        # 0 -> 1 -> 2. Condition on 1 should remove 1->2.
        G = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        P1 = computePathMatrix(G)
        res = computePathMatrix2(G, [1], P1)
        # 0 can still reach 1, but 1 (and 0) cannot reach 2 anymore
        self.assertEqual(res[0, 2], 0)
        self.assertEqual(res[0, 1], 1)