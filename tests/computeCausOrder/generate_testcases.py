import os
import random
import numpy as np
import networkx as nx
from pathlib import Path
from cdt.data import AcyclicGraphGenerator

#llm
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent
TESTCASE_DIR = PROJECT_DIR / "tests" / "computeCausOrder" / "Testcase"

NUM_TESTCASES = 100
TESTCASE_DIR.mkdir(parents=True, exist_ok=True)

def save_matrix(G, index):
    np.savetxt(TESTCASE_DIR / f"{index}.txt", G, fmt="%d")


def permute_matrix(G):
    n = G.shape[0]
    order = np.random.permutation(n)
    return G[order][:, order]


def make_chain_graph(n):
    G = np.zeros((n, n), dtype=int)
    for i in range(n - 1):
        G[i, i + 1] = 1
    return G


def make_star_graph(n):
    G = np.zeros((n, n), dtype=int)
    for j in range(1, n):
        G[0, j] = 1
    return G


def make_dense_dag(n, prob=0.8):
    G = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < prob:
                G[i, j] = 1
    return G


def make_multiple_root_graph(n):
    G = np.zeros((n, n), dtype=int)
    if n >= 5:
        G[0, 2] = 1
        G[1, 2] = 1
        G[3, 4] = 1
    return G


# emptygraph
n = random.randint(3, 8)
G = np.zeros((n, n), dtype=int)
G = permute_matrix(G)
save_matrix(G, 1)

# chaingraph
n = random.randint(3, 8)
G = make_chain_graph(n)
G = permute_matrix(G)
save_matrix(G, 2)

#stargraph
n = random.randint(4, 8)
G = make_star_graph(n)
G = permute_matrix(G)
save_matrix(G, 3)

# multipleroot graph
n = random.randint(5, 8)
G = make_multiple_root_graph(n)
G = permute_matrix(G)
save_matrix(G, 4)

#dense DAG
n = random.randint(4, 7)
G = make_dense_dag(n, prob=0.8)
G = permute_matrix(G)
save_matrix(G, 5)

#random DAGs from CDT
for i in range(6, NUM_TESTCASES + 1):
    nodes = random.randint(3, 15)
    parents_max = random.randint(1, nodes - 1)

    generator = AcyclicGraphGenerator(
        causal_mechanism="linear",
        npoints=1,
        nodes=nodes,
        parents_max=parents_max,
        expected_degree=2,
        dag_type="default"
    )

    data, graph = generator.generate()
    G = nx.to_numpy_array(graph, dtype=int)

    G = permute_matrix(G)
    save_matrix(G, i)