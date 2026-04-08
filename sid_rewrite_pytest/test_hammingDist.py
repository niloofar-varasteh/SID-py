import numpy as np
import pytest

"""
NAME: hammingDist
DESCRIPTION: Computes the Hamming distance between two graph objects.
DETAILS: allMistakesOne determines whether a reversed edge counts as two or as one mistake.
SOURCE: Translated from hammingDist.R
"""

def hamming_dist(G1, G2, all_mistakes_one=True):
    G1, G2 = np.asarray(G1), np.asarray(G2)
    if all_mistakes_one:
        # طبق سورس R: (G1+G2)%%2
        Gtmp = (G1 + G2) % 2
        Gtmp = Gtmp + Gtmp.T
        nr_reversals = np.sum(Gtmp == 2) / 2
        nr_incl_del = np.sum(Gtmp == 1) / 2
        return int(nr_reversals + nr_incl_del)
    else:
        # طبق سورس R برای حالت FALSE: sum(abs(G1 - G2))
        return int(np.sum(np.abs(G1 - G2)))

# --- Pytest In-line Tests using Decorators ---
@pytest.mark.parametrize("all_one, expected", [(True, 1), (False, 2)])
def test_hamming_reversal(all_one, expected):
    """تست یال معکوس: X->Y در برابر X<-Y"""
    g1 = np.array([[0, 1], [0, 0]])
    g2 = np.array([[0, 0], [1, 0]])
    assert hamming_dist(g1, g2, all_mistakes_one=all_one) == expected