import pytest
import numpy as np
from sid_py.computePathMatrix import computePathMatrix
from sid_py.computePathMatrix2 import computePathMatrix2


def test_computePathMatrix2_logic():
    # گراف: 0 -> 1 -> 2
    G = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])

    # ابتدا ماتریس مسیر معمولی را حساب می‌کنیم
    pm1 = computePathMatrix(G)

    # حالا اگر گره 1 را در condSet قرار دهیم، یال 1->2 باید حذف شود
    # در نتیجه 0 دیگر به 2 راه نخواهد داشت (چون از 1 می‌گذشت)
    pm2 = computePathMatrix2(G, condSet=[1], PathMatrix1=pm1)

    # انتظار داریم: 0 فقط به 1 راه داشته باشد (و خودش)
    expected = np.array([
        [1, 1, 0],
        [0, 1, 0],  # یال خروجی از 1 حذف شده
        [0, 0, 1]
    ])

    np.testing.assert_array_equal(pm2, expected)