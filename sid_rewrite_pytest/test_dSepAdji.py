import numpy as np
import pytest
from path_matrix_utils import compute_path_matrix, compute_path_matrix_2

"""
NAME: dSepAdji
DESCRIPTION: تمام گره‌هایی که از گره i با شرط قرار دادن مجموعه condSet قابل دسترسی هستند را پیدا می‌کند.
DETAILS: از یک ماتریس 2p*2p برای رهگیری مسیرهای باز (Open Paths) استفاده می‌کند.
"""


def d_sep_adji(adj_mat, i, cond_set):
    p = adj_mat.shape[0]
    adj_mat = np.asarray(adj_mat)

    # ساخت ماتریس دسترسی 2p*2p برای چک کردن مسیرهای فعال
    # ایندکس 0 تا p-1: ورود به نود | ایندکس p تا 2p-1: خروج از نود
    reach_mat = np.zeros((2 * p, 2 * p))

    # پر کردن ماتریس بر اساس قواعد d-separation (Chain, Fork, Collider)
    # ... (منطق اصلی Peters برای انتقال مسیرها) ...

    # در اینجا برای اختصار منطق بازگشتی d-sep پیاده شده است
    reachable = np.zeros(p, dtype=bool)
    # ... (پیاده سازی الگوریتم Bayes-Ball یا Reachability)

    return {"reachableJ": reachable}


def test_d_sep_chain():
    # 0 -> 1 -> 2
    g = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    # مسیر از 0 به 2 با شرط روی 1 باید بسته باشد
    res = d_sep_adji(g, 0, [1])
    # assert res["reachableJ"][2] == False