import pytest
import numpy as np
from sid_py.computePathMatrix import computePathMatrix
from scipy import sparse

# تست ۱: بررسی یک سناریوی استاندارد (گراف زنجیره‌ای)
def test_standard_chain():
    # ورودی: 0 -> 1 -> 2
    G = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]
    ])

    actual = computePathMatrix(G)

    # خروجی مورد انتظار (با احتساب اینکه قطر اصلی باید 1 باشد)
    expected = np.array([
        [1, 1, 1],  # گره 0 به خودش، 1 و 2 راه دارد
        [0, 1, 1],  # گره 1 به خودش و 2 راه دارد
        [0, 0, 1]  # گره 2 فقط به خودش راه دارد
    ])

    # مقایسه خروجی واقعی با مورد انتظار
    np.testing.assert_array_equal(actual, expected)


# تست ۲: بررسی گراف بدون یال (Empty Graph)
def test_no_edges():
    G = np.zeros((2, 2))
    actual = computePathMatrix(G)
    # طبق مستندات R، حتی در گراف خالی قطر اصلی باید 1 باشد [cite: 2]
    expected = np.eye(2)
    np.testing.assert_array_equal(actual, expected)


# تست ۳: بررسی حالت Sparse (ماتریس خلوت)
    def test_sparse_mode():
        G = np.array([[0, 1], [0, 0]])
        actual_dense = computePathMatrix(G, spars=False)
        actual_sparse = computePathMatrix(G, spars=True)

        # اگر خروجی Sparse بود، آن را به آرایه معمولی تبدیل کن تا تست بتواند مقایسه کند
        if sparse.issparse(actual_sparse):
            actual_sparse = actual_sparse.toarray()

        # حالا مقایسه انجام می‌شود
        np.testing.assert_array_equal(actual_dense, actual_sparse)