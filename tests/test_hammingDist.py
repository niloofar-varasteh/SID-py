import numpy as np
import pytest

from sid_py.hammingDist import hammingDist


def test_identical_graphs():
    """Two identical graphs -> distance 0"""
    G1 = np.array([[0, 1, 0],
                   [0, 0, 1],
                   [0, 0, 0]], dtype=int)
    G2 = np.array([[0, 1, 0],
                   [0, 0, 1],
                   [0, 0, 0]], dtype=int)
    assert hammingDist(G1, G2) == 0


def test_edge_reversal_all_mistakes_one():
    """Reversal counts as 1 when allMistakesOne=True"""
    G1 = np.array([[0, 1],
                   [0, 0]], dtype=int)  # 0->1
    G2 = np.array([[0, 0],
                   [1, 0]], dtype=int)  # 1->0
    assert hammingDist(G1, G2, allMistakesOne=True) == 1


def test_edge_reversal_not_all_mistakes_one():
    """Reversal counts as 2 when allMistakesOne=False"""
    G1 = np.array([[0, 1],
                   [0, 0]], dtype=int)  # 0->1
    G2 = np.array([[0, 0],
                   [1, 0]], dtype=int)  # 1->0
    assert hammingDist(G1, G2, allMistakesOne=False) == 2


def test_inclusion_deletion():
    """Adding or deleting a single directed edge counts as 1 in both modes"""
    G1 = np.array([[0, 1],
                   [0, 0]], dtype=int)  # 0->1
    G2 = np.array([[0, 0],
                   [0, 0]], dtype=int)  # no edge
    assert hammingDist(G1, G2, allMistakesOne=True) == 1
    assert hammingDist(G1, G2, allMistakesOne=False) == 1


def test_symmetry():
    """Distance must be symmetric"""
    G1 = np.array([[0, 1, 1],
                   [0, 0, 0],
                   [0, 1, 0]], dtype=int)
    G2 = np.array([[0, 0, 1],
                   [1, 0, 0],
                   [0, 0, 0]], dtype=int)
    assert hammingDist(G1, G2) == hammingDist(G2, G1)


def test_large_graph_logic():
    """One reversal + one deletion"""
    # G1 edges: 0->1, 1->2, 2->3
    G1 = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
    ], dtype=int)

    # G2 edges: 1->0 (reversal of 0->1), 2->3 (kept), and 1->2 deleted
    G2 = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
    ], dtype=int)

    # allMistakesOne=True: reversal(1) + deletion(1) = 2
    assert hammingDist(G1, G2, allMistakesOne=True) == 2
    # allMistakesOne=False: reversal(2) + deletion(1) = 3
    assert hammingDist(G1, G2, allMistakesOne=False) == 3


def test_multiple_reversals_and_add_del():
    """Mix multiple reversals + add/del in one go (stronger coverage)"""
    # G1 edges: 0->1, 1->2, 3->2
    G1 = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
    ], dtype=int)

    # Make:
    # - reverse 0->1 to 1->0
    # - delete 1->2
    # - add 2->3
    # - keep 3->2
    G2 = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=int)

    # allMistakesOne=True: reversal(1) + deletion(1) + addition(1) = 3
    assert hammingDist(G1, G2, allMistakesOne=True) == 3
    # allMistakesOne=False: reversal(2) + deletion(1) + addition(1) = 4
    assert hammingDist(G1, G2, allMistakesOne=False) == 4


def test_undirected_edge_correction():
    """
    CPDAG-style undirected edge represented as symmetric adjacency.
    Removing an undirected edge should count as 1 (not 2) when allMistakesOne=False,
    thanks to the correction term.
    """
    G1 = np.array([[0, 1],
                   [1, 0]], dtype=int)  # undirected 0--1
    G2 = np.array([[0, 0],
                   [0, 0]], dtype=int)  # none
    assert hammingDist(G1, G2, allMistakesOne=False) == 1
