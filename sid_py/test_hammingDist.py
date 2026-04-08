import numpy as np
import unittest

def hammingDist(G1, G2, allMistakesOne=True):
    """
    محاسبه فاصله همینگ.
    اگر allMistakesOne=True باشد، معکوس شدن یال (-> به <-) فقط 1 خطا محسوب می‌شود.
    """
    if allMistakesOne:
        # منطق دقیق کد R شما:
        # Gtmp <- (G1+G2)%%2
        Gtmp = (G1 + G2) % 2
        # Gtmp <- Gtmp + t(Gtmp)
        Gtmp_combined = Gtmp + Gtmp.T
        # nrReversals <- sum(Gtmp == 2)/2
        nrReversals = np.sum(Gtmp_combined == 2) / 2
        # nrInclDel <- sum(Gtmp == 1)/2
        nrInclDel = np.sum(Gtmp_combined == 1) / 2
        return int(nrReversals + nrInclDel)
    else:
        # جریمه مستقیم تفاوت‌ها
        return int(np.sum(np.abs(G1 - G2)))

class TestHammingDist(unittest.TestCase):
    def test_reversal_logic(self):
        """تست هوشمندانه: معکوس شدن یال باید 1 خطا باشد اگر allMistakesOne=True"""
        G1 = np.array([[0, 1], [0, 0]]) # 0 -> 1
        G2 = np.array([[0, 0], [1, 0]]) # 1 -> 0
        self.assertEqual(hammingDist(G1, G2, True), 1)
        self.assertEqual(hammingDist(G1, G2, False), 2)

if __name__ == "__main__":
    unittest.main()