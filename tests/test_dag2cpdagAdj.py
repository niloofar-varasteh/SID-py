import numpy as np
from sid_py.dag2cpdagAdj import dag2cpdagAdj


def test_chain_becomes_undirected_cpdag():
    # 0->1->2 : no v-structure, edges reversible -> undirected in CPDAG
    G = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ], dtype=int)
    cp = dag2cpdagAdj(G)

    # skeleton edges are undirected: (0-1) and (1-2)
    assert cp[0, 1] == 1 and cp[1, 0] == 1
    assert cp[1, 2] == 1 and cp[2, 1] == 1
    assert cp[0, 2] == 0 and cp[2, 0] == 0


def test_collider_is_compelled():
    # 0->2<-1 : collider compelled
    G = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 0],
    ], dtype=int)
    cp = dag2cpdagAdj(G)

    # must keep arrows into 2
    assert cp[0, 2] == 1 and cp[2, 0] == 0
    assert cp[1, 2] == 1 and cp[2, 1] == 0
    # 0 and 1 not adjacent
    assert cp[0, 1] == 0 and cp[1, 0] == 0
