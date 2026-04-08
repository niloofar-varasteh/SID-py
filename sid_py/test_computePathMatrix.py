import numpy as np
import unittest


def computePathMatrix(G, spars=False):
    """
    محاسبه ماتریس مسیر.
    entry(i,j) == 1 یعنی مسیری از i به j وجود دارد.
    """
    p = G.shape[1]
    # ایجاد ماتریس هویت + مجاورت (diagonal will also be one)
    PathMatrix = np.eye(p, dtype=int) + G.astype(int)

    # مطابق کد R: k = ceiling(log(p)/log(2))
    k = int(np.ceil(np.log2(p))) if p > 0 else 0

    for _ in range(k):
        # ضرب ماتریسی برای یافتن مسیرهای طولانی‌تر
        PathMatrix = np.dot(PathMatrix, PathMatrix)
        # تبدیل مقادیر غیرصفر به 1 برای جلوگیری از بزرگ شدن اعداد
        PathMatrix = (PathMatrix > 0).astype(int)

    return PathMatrix


class TestComputePathMatrix(unittest.TestCase):
    def test_r_example(self):
        """تست جعبه‌سیاه: زنجیره 3 نودی"""
        # 0 -> 1 -> 2
        G = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        res = computePathMatrix(G)
        # باید مسیر 0 به 2 برقرار باشد
        self.assertEqual(res[0, 2], 1)
        self.assertEqual(res[1, 0], 0)


if __name__ == "__main__":
    unittest.main()