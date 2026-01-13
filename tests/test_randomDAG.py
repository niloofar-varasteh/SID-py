import pytest
import numpy as np
from sid_py.randomDAG import randomDAG
from sid_py.computePathMatrix import computePathMatrix


def test_randomDAG_shape():
    p = 10
    dag = randomDAG(p, probConnect=0.2)
    assert dag.shape == (p, p)


def test_randomDAG_is_acyclic():
    p = 5
    dag = randomDAG(p, probConnect=0.5)

    # A graph is a DAG if its transitive closure (path matrix)
    # has no 1s on the diagonal except for the self-paths we added.
    # However, computePathMatrix adds 1 to the diagonal by design. [cite: 2]
    # To check for cycles: PathMatrix - Identity should have no 1s on diagonal.
    pm = computePathMatrix(dag)
    # If we subtract the identity, the diagonal must be all zeros.
    assert np.all(np.diag(pm - np.eye(p)) == 0)