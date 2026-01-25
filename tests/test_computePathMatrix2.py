import numpy as np
import pytest

from sid_py.computePathMatrix import computePathMatrix
from sid_py.computePathMatrix2 import computePathMatrix2


def test_computePathMatrix2_breaks_reachability_simple_chain():
    # 0 -> 1 -> 2
    G = np.array([[0, 1, 0],
                  [0, 0, 1],
                  [0, 0, 0]], dtype=int)

    pm1 = computePathMatrix(G)
    pm2 = computePathMatrix2(G, condSet=[1], PathMatrix1=pm1)  # remove outgoing from node 1

    expected = np.array([[1, 1, 0],
                         [0, 1, 0],
                         [0, 0, 1]], dtype=int)

    np.testing.assert_array_equal(pm2, expected)


def test_computePathMatrix2_empty_condset_returns_pm1():
    G = np.array([[0, 1, 0],
                  [0, 0, 1],
                  [0, 0, 0]], dtype=int)

    pm1 = computePathMatrix(G)
    pm2 = computePathMatrix2(G, condSet=[], PathMatrix1=pm1)

    np.testing.assert_array_equal(pm2, pm1)


def test_computePathMatrix2_long_chain_breaks_reachability():
    # 0 -> 1 -> 2 -> 3 -> 4
    G = np.zeros((5, 5), dtype=int)
    for i in range(4):
        G[i, i + 1] = 1

    pm1 = computePathMatrix(G)
    pm2 = computePathMatrix2(G, condSet=[2], PathMatrix1=pm1)  # remove outgoing from node 2

    # 0 can still reach 2, but can't reach 4 anymore
    assert pm2[0, 2] == 1
    assert pm2[0, 4] == 0
