import numpy as np
from scipy import sparse
from sid_py.computePathMatrix import computePathMatrix


def test_long_chain_dense_needs_multiple_squarings():
    # 0 -> 1 -> 2 -> 3 -> 4
    G = np.zeros((5, 5), dtype=int)
    for i in range(4):
        G[i, i+1] = 1

    actual = computePathMatrix(G, spars=False)
    expected = np.array([
        [1,1,1,1,1],
        [0,1,1,1,1],
        [0,0,1,1,1],
        [0,0,0,1,1],
        [0,0,0,0,1],
    ], dtype=int)

    np.testing.assert_array_equal(actual, expected)

def test_long_chain_sparse_matches_dense():
    G = np.zeros((8, 8), dtype=int)
    for i in range(7):
        G[i, i+1] = 1

    dense = computePathMatrix(G, spars=False)
    sp = computePathMatrix(G, spars=True)
    if sparse.issparse(sp):
        sp = sp.toarray()

    np.testing.assert_array_equal(dense, sp)
