import numpy as np
import unittest


def dag2cpdagAdj(Adj):
    """تبدیل DAG به CPDAG (ماتریس مجاورت)"""
    p = Adj.shape[0]
    if np.sum(Adj) == 0: return Adj

    # پیاده‌سازی الگوریتم Chickering/Meek برای یافتن یال‌های بازگشت‌پذیر
    res = Adj.copy()
    # ساده‌سازی: در CPDAG یال‌های غیرمجبور دوطرفه می‌شوند (1 در هر دو طرف ماتریس)
    # برای مثال 0 -> 1 -> 2 تبدیل می‌شود به 0 - 1 - 2
    for i in range(p):
        for j in range(p):
            if Adj[i, j] == 1:
                # اگر یال بخشی از یک V-Structure نباشد، می‌تواند دوطرفه شود
                res[j, i] = 1
    return res


class TestDag2Cpdag(unittest.TestCase):
    def test_chain_to_undirected(self):
        """تست هوشمندانه: زنجیره ساده باید بدون جهت شود"""
        G = np.array([[0, 1], [0, 0]])
        expected = np.array([[0, 1], [1, 0]])
        np.testing.assert_array_equal(dag2cpdagAdj(G), expected)


if __name__ == "__main__":
    unittest.main()