import numpy as np
import math
from scipy import sparse


def computePathMatrix2(G, condSet, PathMatrix1, spars=False):
    """
    نسخه دوم محاسبه ماتریس مسیر (معادل computePathMatrix2.R).
    این تابع یال‌های خروجی از condSet را حذف می‌کند.
    """
    p = G.shape[1]

    # اگر مجموعه شرطی خالی نباشد، یال‌های خروجی از آن را صفر می‌کنیم
    if len(condSet) > 0:
        G_modified = np.array(G, copy=True)
        # در پایتون ایندکس‌ها از 0 شروع می‌شوند
        # اگر condSet از R می‌آید، باید منهای 1 شود (در اینجا فرض بر ایندکس پایتونی است)
        G_modified[condSet, :] = 0

        if spars:
            path_matrix2 = sparse.eye(p, format='csr') + sparse.csr_matrix(G_modified)
        else:
            path_matrix2 = np.eye(p) + G_modified

        k = math.ceil(math.log(p) / math.log(2))
        for _ in range(k):
            path_matrix2 = path_matrix2 @ path_matrix2
            if spars:
                path_matrix2.data = np.ones_like(path_matrix2.data)
            else:
                path_matrix2 = (path_matrix2 > 0).astype(float)

        return (path_matrix2 > 0).astype(int)
    else:
        # اگر condSet خالی باشد، همان ماتریس مسیر اول بازگردانده می‌شود
        return PathMatrix1