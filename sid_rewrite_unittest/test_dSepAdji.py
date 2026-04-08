import numpy as np
import unittest

def dSepAdji(AdjMat, i, condSet):
    """ Simple reachability check considering conditioning set. """
    p = AdjMat.shape[0]
    reachable = np.zeros(p, dtype=bool)
    reachable[i] = True
    # Basic path search
    for _ in range(p):
        for u in range(p):
            if reachable[u]:
                for v in range(p):
                    if AdjMat[u, v] == 1 and v not in condSet:
                        reachable[v] = True
    return reachable

class TestDSep(unittest.TestCase):
    def test_reach(self):
        G = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        # Path open
        self.assertTrue(dSepAdji(G, 0, [])[2])
        # Path blocked
        self.assertFalse(dSepAdji(G, 0, [1])[2])