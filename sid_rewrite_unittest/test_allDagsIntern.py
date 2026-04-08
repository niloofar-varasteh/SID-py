import numpy as np
import unittest

def allDagsIntern(gm, a, nodes_idx, tmp=None):
    """
    Generates all possible DAGs by orienting undirected edges.
    Ensures no new v-structures or cycles are created.
    """
    if tmp is None: tmp = []
    if np.any((a + a.T) == 1):
        raise ValueError("The matrix is not entirely undirected in component 'a'.")

    if np.sum(a) == 0:
        if not any(np.array_equal(gm, x) for x in tmp):
            tmp.append(gm.copy())
        return tmp
    else:
        sinks = np.where(np.sum(a, axis=0) > 0)[0]
        for x in sinks:
            adj_x = (a[x, :] == 1)
            if np.any(adj_x):
                un = np.where(adj_x)[0]
                pp = len(un)
                adj_sub = a[np.ix_(un, un)]
                if np.all(adj_sub + np.eye(pp)):
                    gm2, a2 = gm.copy(), np.delete(a, x, axis=0)
                    a2 = np.delete(a2, x, axis=1)
                    nodes_idx2 = np.delete(nodes_idx, x)
                    # Orient edges towards sink x
                    gm2[nodes_idx[un], nodes_idx[x]] = 1
                    gm2[nodes_idx[x], nodes_idx[un]] = 0
                    allDagsIntern(gm2, a2, nodes_idx2, tmp)
        return tmp

class TestAllDagsIntern(unittest.TestCase):
    def test_basic_extension(self):
        adj = np.array([[0, 1], [1, 0]])
        res = allDagsIntern(adj.copy(), adj, np.array([0, 1]))
        self.assertEqual(len(res), 2)

    def test_not_undirected_error(self):
        # Coverage: Triggers the ValueError branch
        adj = np.array([[0, 1], [0, 0]])
        with self.assertRaises(ValueError):
            allDagsIntern(adj, adj, np.array([0, 1]))