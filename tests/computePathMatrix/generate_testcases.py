import random
import numpy as np
import networkx as nx
from pathlib import Path
from cdt.data import AcyclicGraphGenerator

#llm
BASE_DIR = Path(__file__).resolve().parent
TESTCASE_DIR = BASE_DIR / "Testcase"

NUM_TESTCASES = 10000
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


def make_disconnected_graph(n):
    G = np.zeros((n, n), dtype=int)
    mid = n // 2

    for i in range(mid - 1):
        G[i, i + 1] = 1

    for j in range(mid + 1, n):
        G[mid, j] = 1

    return G


def make_tree_graph(n):
    G = np.zeros((n, n), dtype=int)
    for child in range(1, n):
        parent = (child - 1) // 2
        G[parent, child] = 1
    return G


#single node
G = np.array([[0]], dtype=int)
G = permute_matrix(G)
save_matrix(G, 1)

#two nodes and no edge
G = np.array([
    [0, 0],
    [0, 0]
], dtype=int)
G = permute_matrix(G)
save_matrix(G, 2)

#two nodes and one edge
G = np.array([
    [0, 1],
    [0, 0]
], dtype=int)
G = permute_matrix(G)
save_matrix(G, 3)

#empty graph
n = random.randint(3, 8)
G = np.zeros((n, n), dtype=int)
G = permute_matrix(G)
save_matrix(G, 4)

#chain graph
n = random.randint(4, 8)
G = make_chain_graph(n)
G = permute_matrix(G)
save_matrix(G, 5)

#star graph
n = random.randint(4, 8)
G = make_star_graph(n)
G = permute_matrix(G)
save_matrix(G, 6)

#tree graph
n = random.randint(5, 10)
G = make_tree_graph(n)
G = permute_matrix(G)
save_matrix(G, 7)

#disconnected graph
n = random.randint(6, 10)
G = make_disconnected_graph(n)
G = permute_matrix(G)
save_matrix(G, 8)

#dense DAG
n = random.randint(4, 8)
G = make_dense_dag(n, prob=0.8)
G = permute_matrix(G)
save_matrix(G, 9)

#DAGs from CDT
for i in range(10, NUM_TESTCASES + 1):
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