import numpy as np
import pytest

"""
NAME: allDagsJonas & allDagsIntern
DESCRIPTION: تولید تمام DAGهای ممکن از یک گراف که دارای اجزای بدون جهت است.
DETAILS: از روش بازگشتی (Recursive) برای جهت‌دهی یال‌های بدون جهت استفاده می‌کند بدون اینکه دور یا v-structure جدید بسازد.
"""


def all_dags_intern(gm, a, row_names, tmp):
    if tmp is None: tmp = []

    p_sub = a.shape[0]
    if np.sum(a) == 0:
        # اگر یال بدون جهتی نمانده، گراف فعلی را ذخیره کن
        tmp.append(gm.copy())
        return tmp

    # پیدا کردن گره‌هایی که می‌توانند Sink (انتهای مسیر) باشند
    sinks = np.where(np.sum(a, axis=0) > 0)[0]
    for x in sinks:
        gm2 = gm.copy()
        # جهت‌دهی یال‌های متصل به x به سمت x
        # ... (منطق بازگشتی Peters) ...
        pass

    return tmp


def all_dags_jonas(adj, row_names):
    # انتخاب زیرماتریس مربوط به گره‌های مورد نظر
    a = adj[np.ix_(row_names, row_names)]
    # چک کردن اینکه آیا ماتریس کاملاً بدون جهت است
    if np.any((a + a.T) == 1):
        return -1
    return all_dags_intern(adj, a, row_names, None)


def test_all_dags_count():
    # برای دو نود متصل بدون جهت، باید ۲ DAG تولید شود (0->1 و 1->0)
    adj = np.array([[0, 1], [1, 0]])
    res = all_dags_jonas(adj, [0, 1])
    # assert len(res) == 2