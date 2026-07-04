import os
import random
import numpy as np

NUM_TESTCASES = 10000
os.makedirs("Testcase", exist_ok=True)


def save_testcase(index, G1, G2):
    with open(f"Testcase/{index}.txt", "w", encoding="utf-8") as f:
        f.write("Matrix1:\n")
        for row in G1:
            f.write(" ".join(map(str, row)) + "\n")

        f.write("\n")
        f.write("Matrix2:\n")
        for row in G2:
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


def remove_random_edge(G):
    G2 = G.copy()
    edges = np.argwhere(G2 == 1)

    if len(edges) > 0:
        i, j = random.choice(edges)
        G2[i, j] = 0

    return G2


def add_random_edge(G):
    G2 = G.copy()
    n = G2.shape[0]

    possible = []
    for i in range(n):
        for j in range(n):
            if i != j and G2[i, j] == 0:
                possible.append((i, j))

    if possible:
        i, j = random.choice(possible)
        G2[i, j] = 1

    return G2


def reverse_random_edge(G):
    G2 = G.copy()
    edges = np.argwhere(G2 == 1)

    if len(edges) > 0:
        i, j = random.choice(edges)
        G2[i, j] = 0
        G2[j, i] = 1

    return G2


def make_mixed_change(G):
    G2 = G.copy()

    change_types = random.sample(["add", "remove", "reverse"], k=random.randint(1, 3))

    for change in change_types:
        if change == "add":
            G2 = add_random_edge(G2)
        elif change == "remove":
            G2 = remove_random_edge(G2)
        elif change == "reverse":
            G2 = reverse_random_edge(G2)

    return G2


case_index = 1

#small and simple cases

G1 = np.array([[0]], dtype=int)
G2 = np.array([[0]], dtype=int)
save_testcase(case_index, G1, G2)
case_index += 1

G1 = np.array([
    [0, 0],
    [0, 0]
], dtype=int)
G2 = np.array([
    [0, 0],
    [0, 0]
], dtype=int)
save_testcase(case_index, G1, G2)
case_index += 1

G1 = np.array([
    [0, 1],
    [0, 0]
], dtype=int)
G2 = np.array([
    [0, 1],
    [0, 0]
], dtype=int)
save_testcase(case_index, G1, G2)
case_index += 1

G1 = np.array([
    [0, 1],
    [0, 0]
], dtype=int)
G2 = np.array([
    [0, 0],
    [0, 0]
], dtype=int)
save_testcase(case_index, G1, G2)
case_index += 1

G1 = np.array([
    [0, 0],
    [0, 0]
], dtype=int)
G2 = np.array([
    [0, 1],
    [0, 0]
], dtype=int)
save_testcase(case_index, G1, G2)
case_index += 1

G1 = np.array([
    [0, 1],
    [0, 0]
], dtype=int)
G2 = np.array([
    [0, 0],
    [1, 0]
], dtype=int)
save_testcase(case_index, G1, G2)
case_index += 1

#fixed structure cases
G1 = make_chain_graph(3)
G2 = make_chain_graph(3)
save_testcase(case_index, G1, G2)
case_index += 1

G1 = make_chain_graph(3)
G2 = remove_random_edge(G1)
save_testcase(case_index, G1, G2)
case_index += 1

G1 = make_chain_graph(3)
G2 = reverse_random_edge(G1)
save_testcase(case_index, G1, G2)
case_index += 1

G1 = make_star_graph(5)
G2 = make_star_graph(5)
save_testcase(case_index, G1, G2)
case_index += 1

G1 = make_star_graph(5)
G2 = remove_random_edge(G1)
save_testcase(case_index, G1, G2)
case_index += 1

G1 = make_star_graph(5)
G2 = reverse_random_edge(G1)
save_testcase(case_index, G1, G2)
case_index += 1

G1 = make_tree_graph(7)
G2 = make_tree_graph(7)
save_testcase(case_index, G1, G2)
case_index += 1

G1 = make_tree_graph(7)
G2 = make_mixed_change(G1)
save_testcase(case_index, G1, G2)
case_index += 1

G1 = make_disconnected_graph(8)
G2 = make_disconnected_graph(8)
save_testcase(case_index, G1, G2)
case_index += 1

G1 = make_disconnected_graph(8)
G2 = make_mixed_change(G1)
save_testcase(case_index, G1, G2)
case_index += 1

#dense fixed cases
G1 = make_dense_dag(6, prob=0.8)
G2 = G1.copy()
save_testcase(case_index, G1, G2)
case_index += 1

G1 = make_dense_dag(6, prob=0.8)
G2 = remove_random_edge(G1)
save_testcase(case_index, G1, G2)
case_index += 1

G1 = make_dense_dag(6, prob=0.8)
G2 = add_random_edge(G1)
save_testcase(case_index, G1, G2)
case_index += 1

G1 = make_dense_dag(6, prob=0.8)
G2 = reverse_random_edge(G1)
save_testcase(case_index, G1, G2)
case_index += 1

G1 = make_dense_dag(8, prob=0.6)
G2 = make_mixed_change(G1)
save_testcase(case_index, G1, G2)
case_index += 1


#structured random cases


while case_index <= 40:
    choice = random.choice(["chain", "star", "tree", "disconnected", "dense"])

    if choice == "chain":
        G1 = make_chain_graph(random.randint(4, 8))
    elif choice == "star":
        G1 = make_star_graph(random.randint(4, 8))
    elif choice == "tree":
        G1 = make_tree_graph(random.randint(5, 10))
    elif choice == "disconnected":
        G1 = make_disconnected_graph(random.randint(6, 10))
    else:
        G1 = make_dense_dag(random.randint(4, 8), prob=random.choice([0.5, 0.7, 0.8]))

    G2 = make_mixed_change(G1)
    save_testcase(case_index, G1, G2)
    case_index += 1

#fully random graph pairs


while case_index <= NUM_TESTCASES:
    n = random.randint(3, 15)
    G1 = make_random_dag(n)

    change_type = random.choice(["same", "add", "remove", "reverse", "mixed"])

    if change_type == "same":
        G2 = G1.copy()
    elif change_type == "add":
        G2 = add_random_edge(G1)
    elif change_type == "remove":
        G2 = remove_random_edge(G1)
    elif change_type == "reverse":
        G2 = reverse_random_edge(G1)
    else:
        G2 = make_mixed_change(G1)

    save_testcase(case_index, G1, G2)
    case_index += 1