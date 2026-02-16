import numpy as np
from sid_py.structIntervDist import structIntervDist


def test_sid_zero_for_identical_graph():
    G = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]
    ])

    result = structIntervDist(G, G)

    assert result["sid"] == 0
    assert np.all(result["incorrectMat"] == 0)


def test_simple_chain_reverse_edge():
    # true: 0 → 1 → 2
    trueG = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]
    ])

    # estimated: 1 → 0 → 2   (wrong)
    estG = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 0, 0]
    ])

    result = structIntervDist(trueG, estG)

    assert result["sid"] > 0


def test_rd_example_from_package():
    # Example from structIntervDist.Rd

    G = np.array([
        [0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])

    H1 = np.array([
        [0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])

    result = structIntervDist(G, H1)

    assert "sid" in result
    assert result["sid"] >= 0
    assert result["incorrectMat"].shape == (5, 5)


def test_sid_detects_wrong_adjustment():
    # true: 0 → 1 → 2
    trueG = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]
    ])

    # estimated graph makes 1 parent of 0
    estG = np.array([
        [0, 0, 0],
        [1, 0, 1],
        [0, 0, 0]
    ])

    result = structIntervDist(trueG, estG)

    # adjusting for descendant causes SID error
    assert result["sid"] > 0
