import numpy as np
import networkx as nx
from sid_py.allDagsJonas import allDagsJonas


def _vec_to_mat(vec, p):
    return np.array(vec, dtype=int).reshape((p, p), order="F")


def test_allDagsJonas_path_of_3_undirected_edges_gives_3_dags():
    # 0--1--2 undirected inside component -> 3 unique DAGs (R removes duplicates)
    p = 3
    adj = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ], dtype=int)

    res = allDagsJonas(adj, [0, 1, 2])
    assert res.shape[0] == 3

    dags = [_vec_to_mat(res[i], p) for i in range(res.shape[0])]

    # expected 3 DAGs:
    expected = [
        np.array([[0, 1, 0],
                  [0, 0, 1],
                  [0, 0, 0]], dtype=int),  # 0 -> 1 -> 2

        np.array([[0, 0, 0],
                  [1, 0, 0],
                  [0, 1, 0]], dtype=int),  # 2 -> 1 -> 0

        np.array([[0, 0, 0],
                  [1, 0, 1],
                  [0, 0, 0]], dtype=int),  # 1 -> 0 and 1 -> 2 (fork)
    ]

    # ensure all results are DAGs and match one of the expected
    for d in dags:
        assert nx.is_directed_acyclic_graph(nx.DiGraph(d))
        assert any(np.array_equal(d, e) for e in expected)
