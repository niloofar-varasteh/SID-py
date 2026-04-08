import numpy as np
import unittest


# از فایل قبلی ایمپورت می‌شود یا در همین فایل قرار می‌گیرد
# from allDagsIntern import allDagsIntern

def allDagsJonas(adj, nodes_to_extend):
    """
    adj: ماتریس مجاورت
    nodes_to_extend: نام یا اندیس نودهایی که بخش بدون جهت را تشکیل می‌دهند
    """
    # استخراج زیرماتریس بخش بدون جهت
    # در پایتون از np.ix_ برای استخراج همزمان سطر و ستون استفاده می‌کنیم
    a = adj[np.ix_(nodes_to_extend, nodes_to_extend)]

    # چک کردن اینکه یال نیمه‌جهت‌دار در این بخش نباشد
    if np.any((a + a.T) == 1):
        # مطابق کد R که return(-1) می‌داد
        return -1

    return allDagsIntern(adj, a, np.array(nodes_to_extend))


class TestAllDagsJonas(unittest.TestCase):
    def test_invalid_input(self):
        """تست شناسایی ورودی اشتباه (یال جهت‌دار در بخش بدون جهت)"""
        # گراف 0 -> 1 در حالی که ادعا شده بدون جهت است
        adj = np.array([[0, 1], [0, 0]])
        res = allDagsJonas(adj, [0, 1])
        self.assertEqual(res, -1)


if __name__ == "__main__":
    unittest.main()