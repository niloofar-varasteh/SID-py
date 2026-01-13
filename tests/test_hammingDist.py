import pytest
import numpy as np
from sid_py.hammingDist import hammingDist

def test_identical_graphs():
    """Test two completely identical graphs (distance should be zero)"""
    G1 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    G2 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    assert hammingDist(G1, G2) == 0

def test_edge_reversal_all_mistakes_one():
    """Test edge reversal (allMistakesOne=True)"""
    # In this case, X->Y and X<-Y are considered one difference
    G1 = np.array([[0, 1], [0, 0]])
    G2 = np.array([[0, 0], [1, 0]])
    assert hammingDist(G1, G2, allMistakesOne=True) == 1

def test_edge_reversal_not_all_mistakes_one():
    """Test edge reversal (allMistakesOne=False)"""
    # In this case, X->Y and X<-Y are considered two differences
    G1 = np.array([[0, 1], [0, 0]])
    G2 = np.array([[0, 0], [1, 0]])
    assert hammingDist(G1, G2, allMistakesOne=False) == 2

def test_inclusion_deletion():
    """Test adding or deleting an edge"""
    G1 = np.array([[0, 1], [0, 0]])
    G2 = np.array([[0, 0], [0, 0]])
    assert hammingDist(G1, G2, allMistakesOne=True) == 1
    assert hammingDist(G1, G2, allMistakesOne=False) == 1

def test_symmetry():
    """Test distance symmetry (G1,G2 == G2,G1)"""
    G1 = np.array([[0, 1, 1], [0, 0, 0], [0, 1, 0]])
    G2 = np.array([[0, 0, 1], [1, 0, 0], [0, 0, 0]])
    assert hammingDist(G1, G2) == hammingDist(G2, G1)

def test_large_graph_logic():
    """Test on a larger graph"""
    G1 = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    G2 = np.array([
        [0, 0, 0, 0], # Delete first edge
        [1, 0, 0, 0], # Reverse second edge
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    # One reversal (1 unit) + One deletion (1 unit) = 2 units
    assert hammingDist(G1, G2, allMistakesOne=True) == 2