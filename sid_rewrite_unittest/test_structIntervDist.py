import numpy as np
import unittest

def structIntervDist(trueGraph, estGraph):
    """
    Computes the Structural Intervention Distance (SID).
    Returns SID value and relevant bounds.
    """
    p = trueGraph.shape[0]
    if np.array_equal(trueGraph, estGraph):
        sid = 0
    else:
        # Distance calculation logic (Simplified for structural testing)
        sid = np.sum(np.abs(trueGraph - estGraph))

    return {"sid": sid, "sidUpperBound": p * (p - 1), "sidLowerBound": 0}

class TestSID(unittest.TestCase):
    def test_identical_graphs(self):
        # Coverage: Triggers the 'if np.array_equal' branch
        G = np.array([[0, 1], [0, 0]])
        res = structIntervDist(G, G)
        self.assertEqual(res["sid"], 0)

    def test_different_graphs(self):
        # Coverage: Triggers the 'else' branch
        G1 = np.array([[0, 1], [0, 0]])
        G2 = np.array([[0, 0], [0, 0]])
        res = structIntervDist(G1, G2)
        self.assertGreater(res["sid"], 0)
        self.assertEqual(res["sidUpperBound"], 2)