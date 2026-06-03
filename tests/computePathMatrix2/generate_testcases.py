import os
import random
import numpy as np

NUM_TESTCASES = 100
os.makedirs("Testcase", exist_ok=True)


def compute_path_matrix(G):
    G = np.array(G, dtype=int)
    p = G.shape[0]

    path_matrix = np.eye(p, dtype=int) + G

    k = int(np.ceil(np.log2(p))) if p > 1 else 0
    for _ in range(k):
        path_matrix = path_matrix @ path_matrix

    path_matrix = (path_matrix > 0).astype(int)
    return path_matrix


def save_testcase(index, G, cond_set):
    path_matrix1 = compute_path_matrix(G)

    with open(f"Testcase/{index}.txt", "w", encoding="utf-8") as f:
        f.write("Matrix:\n")
        for row in G:
            f.write(" ".join(map(str, row)) + "\n")

        f.write("\n")
        f.write("condSet:\n")
        if len(cond_set) == 0:
            f.write("empty\n")
        else:
            f.write(" ".join(map(str, cond_set)) + "\n")

        f.write("\n")
        f.write("PathMatrix1:\n")
        for row in path_matrix1:
            f.write(" ".join(map(str, row)) + "\n")


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


def make_tree_graph(n):
    G = np.zeros((n, n), dtype=int)
    for child in range(1, n):
        parent = (child - 1) // 2
        G[parent, child] = 1
    return G


def make_disconnected_graph(n):
    G = np.zeros((n, n), dtype=int)
    mid = n // 2

    for i in range(mid - 1):
        G[i, i + 1] = 1

    for j in range(mid + 1, n):
        G[mid, j] = 1

    return G


def make_dense_dag(n, prob=0.7):
    G = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < prob:
                G[i, j] = 1
    return G


def make_random_dag(n):
    G = np.zeros((n, n), dtype=int)
    order = list(range(n))
    random.shuffle(order)

    prob = random.choice([0.2, 0.4, 0.6, 0.8])

    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < prob:
                G[order[i], order[j]] = 1

    return G


def random_cond_set(n, allow_empty=True):
    nodes = list(range(1, n + 1))

    if allow_empty and random.random() < 0.3:
        return []

    size = random.randint(1, min(3, n))
    return sorted(random.sample(nodes, size))


case_index = 1


#small and edge cases
G = np.array([[0]], dtype=int)
save_testcase(case_index, G, [])
case_index += 1

G = np.array([[0]], dtype=int)
save_testcase(case_index, G, [1])
case_index += 1

G = np.array([
    [0, 0],
    [0, 0]
], dtype=int)
save_testcase(case_index, G, [])
case_index += 1

G = np.array([
    [0, 1],
    [0, 0]
], dtype=int)
save_testcase(case_index, G, [])
case_index += 1

G = np.array([
    [0, 1],
    [0, 0]
], dtype=int)
save_testcase(case_index, G, [1])
case_index += 1

G = np.array([
    [0, 1],
    [0, 0]
], dtype=int)
save_testcase(case_index, G, [2])
case_index += 1


#simple structured graphs
G = make_chain_graph(3)
save_testcase(case_index, G, [])
case_index += 1

G = make_chain_graph(3)
save_testcase(case_index, G, [1])
case_index += 1

G = make_chain_graph(3)
save_testcase(case_index, G, [2])
case_index += 1

G = make_chain_graph(3)
save_testcase(case_index, G, [3])
case_index += 1

G = make_chain_graph(5)
save_testcase(case_index, G, [3])
case_index += 1

G = make_star_graph(5)
save_testcase(case_index, G, [])
case_index += 1

G = make_star_graph(5)
save_testcase(case_index, G, [1])
case_index += 1

G = make_star_graph(5)
save_testcase(case_index, G, [2])
case_index += 1

G = make_tree_graph(7)
save_testcase(case_index, G, [])
case_index += 1

G = make_tree_graph(7)
save_testcase(case_index, G, [2])
case_index += 1

G = make_tree_graph(7)
save_testcase(case_index, G, [1, 2])
case_index += 1

G = make_disconnected_graph(8)
save_testcase(case_index, G, [])
case_index += 1

G = make_disconnected_graph(8)
save_testcase(case_index, G, [1])
case_index += 1

G = make_disconnected_graph(8)
save_testcase(case_index, G, [4])
case_index += 1


#dense and larger fixed cases #llm

G = make_dense_dag(6, prob=0.8)
save_testcase(case_index, G, [])
case_index += 1

G = make_dense_dag(6, prob=0.8)
save_testcase(case_index, G, [1])
case_index += 1

G = make_dense_dag(6, prob=0.8)
save_testcase(case_index, G, [2, 4])
case_index += 1

G = make_dense_dag(8, prob=0.6)
save_testcase(case_index, G, [3])
case_index += 1

G = make_dense_dag(8, prob=0.6)
save_testcase(case_index, G, [2, 5])
case_index += 1

#structured random sizes
while case_index <= 40:
    choice = random.choice(["chain", "star", "tree", "disconnected", "dense"])

    if choice == "chain":
        G = make_chain_graph(random.randint(4, 8))
    elif choice == "star":
        G = make_star_graph(random.randint(4, 8))
    elif choice == "tree":
        G = make_tree_graph(random.randint(5, 10))
    elif choice == "disconnected":
        G = make_disconnected_graph(random.randint(6, 10))
    else:
        G = make_dense_dag(random.randint(4, 8), prob=random.choice([0.5, 0.7, 0.8]))

    cond_set = random_cond_set(G.shape[0])
    save_testcase(case_index, G, cond_set)
    case_index += 1

#fully random DAGs

while case_index <= NUM_TESTCASES:
    n = random.randint(3, 15)
    G = make_random_dag(n)
    cond_set = random_cond_set(n)

    save_testcase(case_index, G, cond_set)
    case_index += 1