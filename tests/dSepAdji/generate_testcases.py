import os
import random
import numpy as np

NUM_TESTCASES = 100
os.makedirs("Testcase", exist_ok=True)


def save_testcase(index, G, i, cond_set):
    with open(f"Testcase/{index}.txt", "w", encoding="utf-8") as f:
        f.write("Matrix:\n")
        for row in G:
            f.write(" ".join(map(str, row)) + "\n")

        f.write("\n")
        f.write("i:\n")
        f.write(str(i) + "\n")

        f.write("\n")
        f.write("condSet:\n")
        if len(cond_set) == 0:
            f.write("empty\n")
        else:
            f.write(" ".join(map(str, cond_set)) + "\n")


def make_chain_graph(n):
    G = np.zeros((n, n), dtype=int)
    for k in range(n - 1):
        G[k, k + 1] = 1
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

    for k in range(mid - 1):
        G[k, k + 1] = 1

    for j in range(mid + 1, n):
        G[mid, j] = 1

    return G


def make_dense_dag(n, prob=0.7):
    G = np.zeros((n, n), dtype=int)
    for a in range(n):
        for b in range(a + 1, n):
            if random.random() < prob:
                G[a, b] = 1
    return G


def choose_simple_cond_set(n, i):
    candidates = [x for x in range(1, n + 1) if x != i]

    if len(candidates) == 0:
        return []

    mode = random.choice(["empty", "one", "two"])

    if mode == "empty":
        return []
    elif mode == "one":
        return [random.choice(candidates)]
    else:
        if len(candidates) == 1:
            return [candidates[0]]
        return sorted(random.sample(candidates, 2))


case_index = 1


#small and edge cases
G = np.array([[0]], dtype=int)
save_testcase(case_index, G, 1, [])
case_index += 1

G = np.array([[0]], dtype=int)
save_testcase(case_index, G, 1, [1])
case_index += 1

G = np.array([
    [0, 0],
    [0, 0]
], dtype=int)
save_testcase(case_index, G, 1, [])
case_index += 1

G = np.array([
    [0, 1],
    [0, 0]
], dtype=int)
save_testcase(case_index, G, 1, [])
case_index += 1

G = np.array([
    [0, 1],
    [0, 0]
], dtype=int)
save_testcase(case_index, G, 1, [2])
case_index += 1

G = np.array([
    [0, 1],
    [0, 0]
], dtype=int)
save_testcase(case_index, G, 2, [])
case_index += 1


#important basic structures
#chain 1 -> 2 -> 3
G = make_chain_graph(3)
save_testcase(case_index, G, 1, [])
case_index += 1

G = make_chain_graph(3)
save_testcase(case_index, G, 1, [2])
case_index += 1

G = make_chain_graph(3)
save_testcase(case_index, G, 2, [])
case_index += 1

#fork 1 <- 2 -> 3
G = make_fork_graph()
save_testcase(case_index, G, 1, [])
case_index += 1

G = make_fork_graph()
save_testcase(case_index, G, 1, [2])
case_index += 1

G = make_fork_graph()
save_testcase(case_index, G, 2, [])
case_index += 1

#collider 1 -> 2 <- 3 #Vstructure is different with Collider , if it is Vstructure it is a collider but not inverse it ture !
G = make_collider_graph()
save_testcase(case_index, G, 1, [])
case_index += 1

G = make_collider_graph()
save_testcase(case_index, G, 1, [2])
case_index += 1

G = make_collider_graph()
save_testcase(case_index, G, 3, [])
case_index += 1

G = make_collider_graph()
save_testcase(case_index, G, 3, [2])
case_index += 1


#larger structure graphs
G = make_chain_graph(5)
save_testcase(case_index, G, 1, [3])
case_index += 1

G = make_chain_graph(5)
save_testcase(case_index, G, 2, [4])
case_index += 1

G = make_star_graph(5)
save_testcase(case_index, G, 1, [])
case_index += 1

G = make_star_graph(5)
save_testcase(case_index, G, 1, [3])
case_index += 1

G = make_tree_graph(7)
save_testcase(case_index, G, 1, [])
case_index += 1

G = make_tree_graph(7)
save_testcase(case_index, G, 1, [2])
case_index += 1

G = make_tree_graph(7)
save_testcase(case_index, G, 2, [4, 5])
case_index += 1

G = make_disconnected_graph(8)
save_testcase(case_index, G, 1, [])
case_index += 1

G = make_disconnected_graph(8)
save_testcase(case_index, G, 4, [])
case_index += 1

G = make_disconnected_graph(8)
save_testcase(case_index, G, 4, [5])
case_index += 1

G = make_dense_dag(6, prob=0.8)
save_testcase(case_index, G, 1, [])
case_index += 1

G = make_dense_dag(6, prob=0.8)
save_testcase(case_index, G, 2, [4])
case_index += 1


#structure random cases only
while case_index <= NUM_TESTCASES:
    choice = random.choice(["chain", "fork", "collider", "star", "tree", "disconnected", "dense"])

    if choice == "chain":
        G = make_chain_graph(random.randint(4, 8))
    elif choice == "fork":
        G = make_fork_graph()
    elif choice == "collider":
        G = make_collider_graph()
    elif choice == "star":
        G = make_star_graph(random.randint(4, 8))
    elif choice == "tree":
        G = make_tree_graph(random.randint(5, 10))
    elif choice == "disconnected":
        G = make_disconnected_graph(random.randint(6, 10))
    else:
        G = make_dense_dag(random.randint(4, 8), prob=random.choice([0.5, 0.7, 0.8]))

    n = G.shape[0]
    i = random.randint(1, n)
    cond_set = choose_simple_cond_set(n, i)

    save_testcase(case_index, G, i, cond_set)
    case_index += 1