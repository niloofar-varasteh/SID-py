import numpy as np
import pytest

"""
NAME: randomDAG
DESCRIPTION: Outputs the adjacency matrix of a randomly generated DAG.
ARGUMENTS: p (nodes), probConnect (edge probability), causalOrder (optional)
"""


def random_dag(p, prob_connect, causal_order=None):
    if causal_order is None:
        causal_order = np.random.permutation(p)

    DAG = np.zeros((p, p), dtype=int)
    # در پکیج R، والدین از نودهایی با ایندکس بالاتر در ترتیب علیتی انتخاب می‌شوند
    for i in range(p - 1):
        node = causal_order[i]
        possible_parents = causal_order[i + 1:]
        for p_node in possible_parents:
            if np.random.rand() < prob_connect:
                DAG[p_node, node] = 1
    return DAG


# --- Pytest In-line Tests ---
def test_random_dag_structure():
    p = 5
    g = random_dag(p, 0.5)
    assert g.shape == (p, p)
    # چک کردن اینکه روی قطر اصلی مقداری نیست (No self-loops)
    assert np.trace(g) == 0