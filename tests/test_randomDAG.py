import numpy as np
import pytest
import networkx as nx
from sid_py.randomDAG import randomDAG

def test_randomDAG_shape():
    p = 10
    dag = randomDAG(p, probConnect=0.2)
    assert dag.shape == (p, p)

def test_randomDAG_is_acyclic():
    p = 20
    dag = randomDAG(p, probConnect=0.3)

    G = nx.DiGraph(dag)
    assert nx.is_directed_acyclic_graph(G)

def test_randomDAG_many_runs_acyclic():
    p = 30
    for _ in range(50):
        dag = randomDAG(p, probConnect=0.2)
        assert nx.is_directed_acyclic_graph(nx.DiGraph(dag))
