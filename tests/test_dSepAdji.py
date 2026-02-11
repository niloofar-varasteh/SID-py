import numpy as np
from sid_py.dSepAdji import dSepAdji


def test_chain_blocks_when_conditioning_middle():
    # 0 -> 1 -> 2
    G = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]
    ])
    # Conditioning on 1 blocks 0 and 2
    assert dSepAdji(G, 0, 2, condSet=[1]) is True


def test_chain_connects_without_conditioning():
    # 0 -> 1 -> 2
    G = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]
    ])
    assert dSepAdji(G, 0, 2, condSet=[]) is False


def test_collider_opens_when_conditioning_collider():
    # 0 -> 2 <- 1  (collider)
    G = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 0]
    ])
    # Without conditioning: 0 and 1 are d-separated
    assert dSepAdji(G, 0, 1, condSet=[]) is True
    # Conditioning on collider opens path
    assert dSepAdji(G, 0, 1, condSet=[2]) is False
