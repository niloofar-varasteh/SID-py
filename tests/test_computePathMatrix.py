import pytest
import numpy as np
from sid_py.computePathMatrix import computePathMatrix
from scipy import sparse

# Test 1: Check a standard scenario (Chain Graph)
def test_standard_chain():
    # Input: 0 -> 1 -> 2
    G = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]
    ])

    actual = computePathMatrix(G)

    # Expected output (including the diagonal which must be 1)
    expected = np.array([
        [1, 1, 1],  # Node 0 has paths to itself, 1, and 2
        [0, 1, 1],  # Node 1 has paths to itself and 2
        [0, 0, 1]   # Node 2 has a path only to itself
    ])

    # Compare actual output with expected output
    np.testing.assert_array_equal(actual, expected)


# Test 2: Check graph with no edges (Empty Graph)
def test_no_edges():
    G = np.zeros((2, 2))
    actual = computePathMatrix(G)
    # According to R documentation, even in an empty graph, the diagonal must be 1
    expected = np.eye(2)
    np.testing.assert_array_equal(actual, expected)


# Test 3: Check Sparse mode
def test_sparse_mode():
    G = np.array([[0, 1], [0, 0]])
    actual_dense = computePathMatrix(G, spars=False)
    actual_sparse = computePathMatrix(G, spars=True)

    # If the output is Sparse, convert it to a dense array for comparison
    if sparse.issparse(actual_sparse):
        actual_sparse = actual_sparse.toarray()

    # Perform the assertion
    np.testing.assert_array_equal(actual_dense, actual_sparse)