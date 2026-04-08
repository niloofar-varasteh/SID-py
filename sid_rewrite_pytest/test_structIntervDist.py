import numpy as np
import math
import pytest

"""
NAME: computePathMatrix2
DESCRIPTION: ماتریس مسیر را محاسبه می‌کند، با این تفاوت که تمام یال‌های خروجی از یک مجموعه گره خاص (condSet) را حذف می‌کند.
ARGUMENTS: 
    - G: ماتریس مجاورت
    - cond_set: مجموعه‌ای از گره‌ها که یال‌های خروجی آن‌ها باید قطع شود.
    - path_matrix_1: ماتریس مسیر پیش‌فرض (در صورت خالی بودن مجموعه شرطی)
"""


def compute_path_matrix(G):
    p = G.shape[0]
    P = np.eye(p, dtype=int) + G
    k = math.ceil(math.log2(p)) if p > 1 else 0
    for _ in range(k):
        P = (P @ P) > 0
        P = P.astype(int)
    return P


def compute_path_matrix_2(G, cond_set, path_matrix_1):
    if len(cond_set) == 0:
        return path_matrix_1

    p = G.shape[0]
    G_mod = np.array(G, copy=True)
    # حذف یال‌های خروجی از مجموعه شرطی (مطابق computePathMatrix2.R)
    G_mod[cond_set, :] = 0

    return compute_path_matrix(G_mod)


# --- Pytest In-line Tests ---
@pytest.mark.parametrize("cond, expected_reach", [
    ([], 1),  # بدون شرط: مسیر 0 به 2 باز است
    ([1], 0),  # شرط روی 1: مسیر 0 به 2 قطع می‌شود
])
def test_path_matrix_blocking(cond, expected_reach):
    # 0 -> 1 -> 2
    g = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    pm1 = compute_path_matrix(g)
    pm2 = compute_path_matrix_2(g, cond, pm1)
    assert pm2[0, 2] == expected_reach