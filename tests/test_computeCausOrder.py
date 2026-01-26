import numpy as np
from sid_py.computeCausOrder import computeCausOrder


def test_computeCausOrder_simple_chain():
    # 0 -> 1 -> 2
    G = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ])
    order = computeCausOrder(G)
    assert order.tolist() == [0, 1, 2]


def test_computeCausOrder_two_roots_picks_smallest_position():
    # Two roots: 0 and 1, and 1->2
    G = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 0],
    ])
    order = computeCausOrder(G)
    # root chosen first is node 0 (smallest position), then 1, then 2
    assert order.tolist() == [0, 1, 2]


def test_computeCausOrder_cycle_raises():
    # 0 -> 1 -> 0
    G = np.array([
        [0, 1],
        [1, 0],
    ])
    import pytest
    with pytest.raises(ValueError):
        computeCausOrder(G)
