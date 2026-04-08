import numpy as np
import unittest

from sid_rewrite_unittest.test_allDagsIntern import allDagsIntern


def allDagsJonas(adj, nodes_to_extend):
    """
    Validation wrapper for allDagsIntern.
    Returns -1 if directed edges exist in the provided subset.
    """
    a = adj[np.ix_(nodes_to_extend, nodes_to_extend)]
    if np.any((a + a.T) == 1):
        return -1
    return allDagsIntern(adj, a, np.array(nodes_to_extend))

class TestAllDagsJonas(unittest.TestCase):
    def test_invalid_input(self):
        # Coverage: Triggers the -1 return branch
        adj = np.array([[0, 1], [0, 0]])
        res = allDagsJonas(adj, [0, 1])
        self.assertEqual(res, -1)

    def test_valid_input(self):
        adj = np.array([[0, 1], [1, 0]])
        res = allDagsJonas(adj, [0, 1])
        self.assertNotEqual(res, -1)