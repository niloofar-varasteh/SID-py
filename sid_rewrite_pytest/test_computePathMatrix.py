import numpy as np
import math
import pytest

"""
NAME: computePathMatrix & computePathMatrix2
DESCRIPTION: Computes a path matrix where entry(i,j)=1 means a directed path from i to j exists.
"""

def compute_path_matrix(G):
    p = G.shape[0]
    path_mat = np.eye(p, dtype=int) + G
    k = math.ceil(math.log2(p)) if p > 1 else 0
    for _ in range(k):
        path_mat = (path_mat @ path_mat) > 0
        path_mat = path_mat.astype(int)
    return path_mat

def compute_path_matrix_2(G, cond_set, path_matrix_1):
    if len(cond_set) == 0:
        return path_matrix_1
    G_mod = G.copy()
    # حذف تمام یال‌هایی که از مجموعه شرطی خارج می‌شوند (طبق سورس R)
    G_mod[cond_set, :] = 0
    return compute_path_matrix(G_mod)

# --- Pytest In-line Tests ---
def test_path_blocking():
    # 0 -> 1 -> 2
    g = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    pm1 = compute_path_matrix(g)
    # با شرطی کردن روی نود 1، مسیر 0 به 2 باید قطع شود
    pm2 = compute_path_matrix_2(g, [1], pm1)
    assert pm2[0, 2] == 0