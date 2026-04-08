
import math
import numpy as np
import unittest
import numpy.testing as npt

try:
    from scipy import sparse
except Exception:
    sparse = None

from computePathMatrix import compute_path_matrix

def compute_path_matrix2(G: np.ndarray, cond_set, path_matrix1, spars: bool = False) -> np.ndarray:
    """
    Translation of computePathMatrix2.R:
    removes all edges leaving condSet (i.e., zero out rows of condSet),
    then computes path matrix; if condSet is empty returns PathMatrix1.
    Source: computePathMatrix2.R fileciteturn3file1L1-L44
    """
    G = np.asarray(G, dtype=int)
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError("G must be a square adjacency matrix")

    cond_set = list(cond_set) if cond_set is not None else []
    if len(cond_set) == 0:
        return np.asarray(path_matrix1, dtype=bool)

    p = G.shape[0]
    G2 = G.copy()
    G2[cond_set, :] = 0

    # reuse compute_path_matrix logic on modified graph
    return compute_path_matrix(G2, spars=spars)

class TestComputePathMatrix2(unittest.TestCase):
    def test_empty_condset_returns_pm1(self):
        G = np.array([[0, 1],
                      [0, 0]], dtype=int)
        pm1 = np.array([[1, 1],
                        [0, 1]], dtype=bool)
        out = compute_path_matrix2(G, [], pm1, spars=False)
        npt.assert_array_equal(out, pm1)

    def test_remove_outgoing_edges(self):
        # 0 -> 1 -> 2, cond_set={1} removes 1->2
        G = np.array([[0, 1, 0],
                      [0, 0, 1],
                      [0, 0, 0]], dtype=int)
        pm1 = compute_path_matrix(G, spars=False)
        out = compute_path_matrix2(G, [1], pm1, spars=False)
        expected = np.array([[1, 1, 0],
                             [0, 1, 0],
                             [0, 0, 1]], dtype=bool)
        npt.assert_array_equal(out, expected)

if __name__ == "__main__":
    unittest.main()
