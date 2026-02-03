import numpy as np
import networkx as nx
from sid_py.allDagsIntern import allDagsIntern


def _vec_to_mat(vec, p):
    # R-style vectorization is column-major; we used order="F"
    return np.array(vec, dtype=int).reshape((p, p), order="F")


def test_allDagsIntern_empty_component_returns_single_graph():
    # gm is already fully directed (no undirected component left)
    p = 3
    gm = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ], dtype=int)

    # component submatrix a has no undirected edges
    row_names = np.array([0, 1, 2], dtype=int)
    a = np.zeros((3, 3), dtype=int)

    res = allDagsIntern(gm, a, row_names)
    assert res.shape == (1, p * p)

    g_back = _vec_to_mat(res[0], p)
    np.testing.assert_array_equal(g_back, gm)


def test_allDagsIntern_two_node_undirected_edge_produces_two_orientations():
    # gm contains an undirected edge 0--1 represented as symmetric ones
    p = 2
    gm = np.array([
        [0, 1],
        [1, 0],
    ], dtype=int)

    row_names = np.array([0, 1], dtype=int)
    a = gm.copy()  # the undirected component is the whole graph

    res = allDagsIntern(gm, a, row_names)
    assert res.shape[0] == 2  # two DAGs

    dags = [_vec_to_mat(res[i], p) for i in range(res.shape[0])]

    # Each result must be a DAG and must be one of the two orientations
    allowed1 = np.array([[0, 1], [0, 0]], dtype=int)  # 0->1
    allowed2 = np.array([[0, 0], [1, 0]], dtype=int)  # 1->0

    ok = 0
    for d in dags:
        assert nx.is_directed_acyclic_graph(nx.DiGraph(d))
        if np.array_equal(d, allowed1) or np.array_equal(d, allowed2):
            ok += 1
    assert ok == 2


def test_allDagsIntern_raises_if_not_entirely_undirected_component():
    # a must be entirely undirected; if it contains a directed-only edge pattern
    # (a + a.T == 1), it should raise.
    gm = np.array([
        [0, 1],
        [0, 0],
    ], dtype=int)

    row_names = np.array([0, 1], dtype=int)

    # a contains a directed edge only (not symmetric)
    a = np.array([
        [0, 1],
        [0, 0],
    ], dtype=int)

    import pytest
    with pytest.raises(ValueError):
        allDagsIntern(gm, a, row_names)
