import pytest
import numpy as np
from sid_py.computePathMatrix import computePathMatrix
from sid_py.computePathMatrix2 import computePathMatrix2

def test_computePathMatrix2_logic():
    # Graph structure: 0 -> 1 -> 2
    G = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])

    # First, compute the standard path matrix
    pm1 = computePathMatrix(G)

    # Now, if node 1 is placed in the condSet, the edge 1 -> 2 should be removed.
    # Consequently, node 0 will no longer have a path to node 2 (since it passed through 1).
    pm2 = computePathMatrix2(G, condSet=[1], PathMatrix1=pm1)

    # Expected result: node 0 only has paths to node 1 and itself.
    expected = np.array([
        [1, 1, 0],
        [0, 1, 0],  # Outgoing edge from node 1 has been removed
        [0, 0, 1]
    ])

    np.testing.assert_array_equal(pm2, expected)