import numpy as np
import unittest


def computePathMatrix2(G, condSet, PathMatrix1, spars=False):
    """
    محاسبه ماتریس مسیر با حذف یال‌های خروجی از مجموعه شرطی.
    G: ماتریس مجاورت (0-based)[cite: 3].
    condSet: مجموعه‌ای از نودها (اندیس‌های پایتونی).
    PathMatrix1: ماتریس مسیر اولیه گراف G.
    """
    if len(condSet) == 0:
        return PathMatrix1[cite: 2]

    p = G.shape[0]
    G_mod = G.copy()
    # حذف تمام یال‌هایی که از condSet خارج می‌شوند
    G_mod[condSet, :] = 0

    # محاسبه ماتریس مسیر برای گراف اصلاح شده
    path_mat = np.eye(p, dtype=bool) | (G_mod.astype(bool))
    for k in range(p):
        path_mat |= np.outer(path_mat[:, k], path_mat[k, :])

    return path_mat.astype(int)


class TestComputePathMatrix2(unittest.TestCase):
    def test_empty_condSet(self):
        """تست جعبه‌سیاه: اگر مجموعه شرطی خالی باشد خروجی همان ماتریس اول است """
        G = np.array([[0, 1], [0, 0]])
        P1 = np.array([[1, 1], [0, 1]])
        res = computePathMatrix2(G, [], P1)
        np.testing.assert_array_equal(res, P1)

    def test_edge_removal(self):
        """تست هوشمندانه: بررسی قطع شدن مسیرها پس از شرطی‌سازی"""
        # 0 -> 1 -> 2
        G = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        P1 = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
        # اگر روی 1 شرطی شویم، یال 1->2 باید قطع شود
        res = computePathMatrix2(G, [1], P1)
        # مسیر 0 به 2 باید قطع شود اما 0 به 1 برقرار بماند
        self.assertEqual(res[0, 2], 0)
        self.assertEqual(res[0, 1], 1)


if __name__ == "__main__":
    unittest.main()