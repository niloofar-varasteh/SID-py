import numpy as np
import unittest


def computeCausOrder(G):
    """
    استخراج ترتیب علیتی (Topological Order).
    ورودی باید یک DAG باشد[cite: 15].
    """
    p = G.shape[0]
    order = []
    temp_G = G.copy()

    for _ in range(p):
        # پیدا کردن نودهایی که ورودی ندارند (Parentless)
        in_degrees = temp_G.sum(axis=0)
        candidates = np.where(in_degrees == 0)[0]

        # نودهایی که قبلاً در لیست نبودند
        candidates = [c for c in candidates if c not in order]

        if not candidates:
            raise ValueError("Graph contains a cycle; cannot compute causal order.")

        next_node = candidates[0]
        order.append(next_node)
        # حذف مجازی نود برای یافتن نود بعدی
        temp_G[next_node, :] = -1  # غیرفعال کردن یال‌های خروجی
        temp_G[:, next_node] = -1  # جلوگیری از شمارش مجدد

    return np.array(order)


class TestCausOrder(unittest.TestCase):
    def test_simple_order(self):
        """تست زنجیره ساده: 2 -> 0 -> 1"""
        G = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]])
        order = computeCausOrder(G)
        # ترتیب باید [2, 0, 1] باشد
        np.testing.assert_array_equal(order, [2, 0, 1])

    def test_cycle_error(self):
        """تست هوشمندانه: تشخیص دور (Cycle)"""
        G = np.array([[0, 1], [1, 0]])  # دور بین 0 و 1
        with self.assertRaises(ValueError):
            computeCausOrder(G)


if __name__ == "__main__":
    unittest.main()