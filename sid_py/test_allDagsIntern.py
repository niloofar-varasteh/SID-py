import numpy as np
import unittest


def allDagsIntern(gm, a, nodes_idx, tmp=None):
    """
    نسخه پایتون allDagsIntern
    gm: ماتریس مجاورت گراف اصلی
    a: زیرماتریس مولفه بدون جهت
    nodes_idx: اندیس نودهای درگیر در مولفه a
    tmp: لیست نتایج (برای حالت بازگشتی)
    """
    if tmp is None:
        tmp = []

    # بررسی اینکه آیا ماتریس کاملاً بدون جهت است (در بخش a)
    # در R: any((a + t(a)) == 1)
    if np.any((a + a.T) == 1):
        raise ValueError("The matrix is not entirely undirected in component 'a'.")

    # اگر تمام یال‌ها جهت‌دهی شدند (ماتریس a خالی شد)
    if np.sum(a) == 0:
        # تبدیل ماتریس به فرمت تخت برای چک کردن تکراری نبودن
        gm_flat = gm.flatten()
        if not any(np.array_equal(gm, x) for x in tmp):
            tmp.append(gm.copy())
        return tmp
    else:
        # پیدا کردن Sinks (نودهایی که می‌توانند خروجی باشند)
        # در R: sinks <- which(colSums(a) > 0)
        sinks = np.where(np.sum(a, axis=0) > 0)[0]

        for x in sinks:
            gm2 = gm.copy()
            # پیدا کردن همسایگان نود x در گراف a
            adj_x = (a[x, :] == 1)

            if np.any(adj_x):
                un = np.where(adj_x)[0]
                pp = len(un)
                # بررسی شرط کلیک (Clique): آیا تمام همسایگان به هم وصل هستند؟
                # اگر وصل نباشند، ایجاد سینک باعث ایجاد v-structure می‌شود.
                adj_sub = a[np.ix_(un, un)]
                # افزودن قطر اصلی برای چک کردن کلیک کامل
                if np.all(adj_sub + np.eye(pp)):
                    # جهت‌دهی یال‌ها به سمت x (x تبدیل به sink می‌شود)
                    real_un = nodes_idx[un]
                    real_x = nodes_idx[x]
                    gm2[real_un, real_x] = 1
                    gm2[real_x, real_un] = 0

                    # حذف نود x از ماتریس a و ادامه به صورت بازگشتی
                    a2 = np.delete(a, x, axis=0)
                    a2 = np.delete(a2, x, axis=1)
                    nodes_idx2 = np.delete(nodes_idx, x)

                    allDagsIntern(gm2, a2, nodes_idx2, tmp)
        return tmp


class TestAllDagsIntern(unittest.TestCase):
    def test_basic_extension(self):
        """تست جعبه‌خاکستری: توسعه یک یال بدون جهت"""
        # گراف: 0 - 1 (بدون جهت)
        gm = np.array([[0, 1], [1, 0]])
        a = gm.copy()
        nodes = np.array([0, 1])
        result = allDagsIntern(gm, a, nodes)
        # خروجی باید 2 DAG باشد: 0->1 و 1->0
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()