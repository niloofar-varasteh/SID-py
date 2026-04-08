import numpy as np
import pytest

"""
NAME: computeCausOrder
DESCRIPTION: Computes the causal or topological order of the nodes in a DAG.
"""


def compute_caus_order(G):
    p = G.shape[0]
    remaining = list(range(p))
    caus_order = []
    temp_G = G.astype(float).copy()

    for _ in range(p):
        # پیدا کردن نودی که مجموع ستون‌هایش صفر است (بدون یال ورودی)
        in_degrees = np.nansum(temp_G, axis=0)
        found = False
        for i in range(p):
            if i in remaining and in_degrees[i] == 0:
                caus_order.append(i)
                remaining.remove(i)
                temp_G[i, :] = 0
                temp_G[:, i] = np.nan
                found = True
                break
        if not found and len(remaining) > 0:
            raise ValueError("The graph contains a cycle!")
    return np.array(caus_order)


# --- Pytest In-line Tests ---
def test_causal_order_chain():
    # 0 -> 1 -> 2
    g = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    assert np.array_equal(compute_caus_order(g), [0, 1, 2])