import os
import random
import numpy as np

NUM_TESTCASES = 100
os.makedirs("Testcase", exist_ok=True)


def save_matrix(G, index):
    np.savetxt(f"Testcase/{index}.txt", G, fmt="%d")


def permute_matrix(G):
    n = G.shape[0]
    order = np.random.permutation(n)
    return G[order][:, order]


def make_chain_graph(n):
    G = np.zeros((n, n), dtype=int)
    for i in range(n - 1):
        G[i, i + 1] = 1
    return G


def make_fork_graph():
    G = np.zeros((3, 3), dtype=int)
    G[1, 0] = 1
    G[1, 2] = 1
    return G


def make_collider_graph():
    G = np.zeros((3, 3), dtype=int)
    G[0, 1] = 1
    G[2, 1] = 1
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


case_index = 1


#simple testcases without permutation

G = np.array([[0]], dtype=int)
save_matrix(G, case_index)
case_index += 1

G = np.array([
    [0, 0],
    [0, 0]
], dtype=int)
save_matrix(G, case_index)
case_index += 1

G = np.array([
    [0, 1],
    [0, 0]
], dtype=int)
save_matrix(G, case_index)
case_index += 1

n = random.randint(3, 6)
G = np.zeros((n, n), dtype=int)
save_matrix(G, case_index)
case_index += 1

n = random.randint(4, 8)
G = make_chain_graph(n)
save_matrix(G, case_index)
case_index += 1

G = make_fork_graph()
save_matrix(G, case_index)
case_index += 1

G = make_collider_graph()
save_matrix(G, case_index)
case_index += 1

n = random.randint(4, 8)
G = make_star_graph(n)
save_matrix(G, case_index)
case_index += 1

n = random.randint(5, 10)
G = make_tree_graph(n)
save_matrix(G, case_index)
case_index += 1

n = random.randint(6, 10)
G = make_disconnected_graph(n)
save_matrix(G, case_index)
case_index += 1

n = random.randint(4, 8)
G = make_dense_dag(n, prob=0.8)
save_matrix(G, case_index)
case_index += 1


#structured testcases with permutation

while case_index <= 30:
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
        G = make_dense_dag(random.randint(4, 8), prob=0.7)

    G = permute_matrix(G)
    save_matrix(G, case_index)
    case_index += 1


#random DAGs with permutation for the rest

while case_index <= NUM_TESTCASES:
    n = random.randint(3, 15)
    G = make_random_dag(n)
    G = permute_matrix(G)

    save_matrix(G, case_index)
    case_index += 1