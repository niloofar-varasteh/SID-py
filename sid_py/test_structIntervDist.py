import numpy as np
import unittest


def structIntervDist(trueGraph, estGraph):
    """
    محاسبه فاصله مداخله‌ای ساختاری (SID).
    trueGraph: حتما باید DAG باشد.
    estGraph: می‌تواند DAG یا CPDAG باشد.
    """
    p = trueGraph.shape[0]
    # 1. محاسبه ماتریس مسیر برای گراف واقعی
    # 2. برای هر نود، بررسی d-separation با توجه به والدین در گراف تخمینی
    # 3. شمارش تعداد اشتباهات (Incorrect interventional distributions)

    incorrect_mat = np.zeros((p, p))
    # ... منطق اصلی بر اساس dSepAdji ...

    return {
        "sid": int(np.sum(incorrect_mat)),
        "sidLowerBound": 0,
        "sidUpperBound": p * (p - 1)
    }


class TestSID(unittest.TestCase):
    def test_identity_property(self):
        """تست هوشمندانه: فاصله یک گراف با خودش باید صفر باشد"""
        G = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
        result = structIntervDist(G, G)
        self.assertEqual(result["sid"], 0)

    def test_maximum_distance(self):
        """تست مقدار مرزی: فاصله گراف خالی و گراف کامل"""
        G_empty = np.zeros((3, 3))
        G_full = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
        result = structIntervDist(G_full, G_empty)
        # در این حالت SID باید بزرگتر از صفر باشد
        self.assertGreater(result["sid"], 0)


if __name__ == "__main__":
    unittest.main()