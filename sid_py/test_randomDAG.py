import numpy as np
import unittest


def randomDAG(p, probConnect, causalOrder=None):
    """
    تولید DAG تصادفی.
    causalOrder: ترتیب از گره Sink به Source (مطابق کامنت فایل R شما).
    """
    if causalOrder is None:
        causalOrder = np.random.permutation(p)

    # استفاده از ماتریس صفر p x p
    DAG = np.zeros((p, p), dtype=int)

    # مطابق حلقه R: for(i in 1:(p-2))
    for i in range(p - 1):
        node = causalOrder[i]
        # گره‌های پتانسیل برای والد بودن (گره‌های بعدی در ترتیب علیتی)
        possibleParents = causalOrder[(i + 1):p]
        num_possible = len(possibleParents)

        if num_possible > 0:
            # تعداد والدین بر اساس توزیع دو جمله‌ای (rbinom در R)
            numParents = np.random.binomial(n=num_possible, p=probConnect)
            if numParents > 0:
                parents = np.random.choice(possibleParents, size=numParents, replace=False)
                # در R: DAG[Parents, node] = 1 (یال از والدین به نود)
                DAG[parents, node] = 1

    return DAG


class TestRandomDAG(unittest.TestCase):
    def test_is_acyclic(self):
        """تست ناوردا (Grey-box): خروجی نباید دور داشته باشد"""
        np.random.seed(42)
        p = 6
        G = randomDAG(p, 0.4)

        # چک کردن acyclic بودن با استفاده از ماتریس مسیر
        # اگر گرافی دور نداشته باشد، مجموع قطر اصلی ماتریس مسیر (بدون احتساب خود نود) باید 0 باشد
        path_mat = computePathMatrix(G) - np.eye(p)
        self.assertEqual(np.trace(path_mat), 0, "گراف دارای دور است!")


if __name__ == "__main__":
    unittest.main()