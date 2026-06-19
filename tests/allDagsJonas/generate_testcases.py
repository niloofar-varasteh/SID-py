import os
import random
import numpy as np

NUM_TESTCASES = 100
os.makedirs("Testcase", exist_ok=True)


def save_testcase(index, adj, row_names):
    with open(f"Testcase/{index}.txt", "w", encoding="utf-8") as f:
        f.write("adj:\n")
        for row in adj:
            f.write(" ".join(map(str, row)) + "\n")

        f.write("\n")
        f.write("row.names:\n")
        f.write(" ".join(map(str, row_names)) + "\n")


def make_empty_component(n):
    return np.zeros((n, n), dtype=int)


def make_single_edge_component():
    a = np.zeros((2, 2), dtype=int)
    a[0, 1] = 1
    a[1, 0] = 1
    return a


def make_chain_component(n):
    a = np.zeros((n, n), dtype=int)
    for i in range(n - 1):
        a[i, i + 1] = 1
        a[i + 1, i] = 1
    return a


def make_star_component(n):
    a = np.zeros((n, n), dtype=int)
    for j in range(1, n):
        a[0, j] = 1
        a[j, 0] = 1
    return a


def make_clique_component(n):
    a = np.ones((n, n), dtype=int)
    np.fill_diagonal(a, 0)
    return a


def make_cycle_component(n):
    a = np.zeros((n, n), dtype=int)
    for i in range(n):
        j = (i + 1) % n
        a[i, j] = 1
        a[j, i] = 1
    return a


def make_tree_component(n):
    a = np.zeros((n, n), dtype=int)
    for child in range(1, n):
        parent = (child - 1) // 2
        a[parent, child] = 1
        a[child, parent] = 1
    return a


def embed_component_in_adj(a, total_nodes, row_names):
    adj = np.zeros((total_nodes, total_nodes), dtype=int)
    idx = [x - 1 for x in row_names]

    for i in range(len(idx)):
        for j in range(len(idx)):
            adj[idx[i], idx[j]] = a[i, j]

    outside = [x for x in range(total_nodes) if x not in idx]

    if len(outside) >= 2:
        for k in range(len(outside) - 1):
            adj[outside[k], outside[k + 1]] = 1

    return adj


def make_invalid_directed_subgraph(n):
    a = np.zeros((n, n), dtype=int)
    for i in range(n - 1):
        a[i, i + 1] = 1
    return a


case_index = 1



#small cases
adj = make_empty_component(1)
save_testcase(case_index, adj, [1])
case_index += 1

adj = make_single_edge_component()
save_testcase(case_index, adj, [1, 2])
case_index += 1

adj = make_chain_component(3)
save_testcase(case_index, adj, [1, 2, 3])
case_index += 1

adj = make_star_component(4)
save_testcase(case_index, adj, [1, 2, 3, 4])
case_index += 1

adj = make_clique_component(3)
save_testcase(case_index, adj, [1, 2, 3])
case_index += 1

adj = make_cycle_component(4)
save_testcase(case_index, adj, [1, 2, 3, 4])
case_index += 1

adj = make_tree_component(5)
save_testcase(case_index, adj, [1, 2, 3, 4, 5])
case_index += 1



#embedded cases #llm
a = make_chain_component(3)
adj = embed_component_in_adj(a, 5, [2, 3, 4])
save_testcase(case_index, adj, [2, 3, 4])
case_index += 1

a = make_star_component(4)
adj = embed_component_in_adj(a, 6, [2, 3, 4, 5])
save_testcase(case_index, adj, [2, 3, 4, 5])
case_index += 1

a = make_clique_component(4)
adj = embed_component_in_adj(a, 6, [1, 3, 4, 6])
save_testcase(case_index, adj, [1, 3, 4, 6])
case_index += 1

a = make_cycle_component(5)
adj = embed_component_in_adj(a, 8, [2, 3, 4, 5, 6])
save_testcase(case_index, adj, [2, 3, 4, 5, 6])
case_index += 1

a = make_tree_component(6)
adj = embed_component_in_adj(a, 9, [2, 3, 4, 5, 6, 7])
save_testcase(case_index, adj, [2, 3, 4, 5, 6, 7])
case_index += 1



#invalid casesc #llm
# These should return -1 in allDagsJonas

adj = make_invalid_directed_subgraph(2)
save_testcase(case_index, adj, [1, 2])
case_index += 1

adj = make_invalid_directed_subgraph(3)
save_testcase(case_index, adj, [1, 2, 3])
case_index += 1

adj = np.array([
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
], dtype=int)
save_testcase(case_index, adj, [1, 2, 3])
case_index += 1

adj = np.array([
    [0, 1, 0, 0, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0]
], dtype=int)
save_testcase(case_index, adj, [2, 3, 4])
case_index += 1



#random cases

while case_index <= 70:
    choice = random.choice(["edge", "chain", "star", "clique", "cycle", "tree"])

    if choice == "edge":
        a = make_single_edge_component()
    elif choice == "chain":
        a = make_chain_component(random.randint(3, 6))
    elif choice == "star":
        a = make_star_component(random.randint(4, 7))
    elif choice == "clique":
        a = make_clique_component(random.randint(3, 5))
    elif choice == "cycle":
        a = make_cycle_component(random.randint(4, 6))
    else:
        a = make_tree_component(random.randint(4, 7))

    comp_size = a.shape[0]
    total_nodes = comp_size + random.randint(0, 3)

    if total_nodes == comp_size:
        adj = a.copy()
        row_names = list(range(1, comp_size + 1))
    else:
        row_names = sorted(random.sample(range(1, total_nodes + 1), comp_size))
        adj = embed_component_in_adj(a, total_nodes, row_names)

    save_testcase(case_index, adj, row_names)
    case_index += 1



#random invalid cases #llm

while case_index <= NUM_TESTCASES:
    comp_size = random.randint(2, 6)
    total_nodes = comp_size + random.randint(0, 3)

    if total_nodes == comp_size:
        row_names = list(range(1, comp_size + 1))
    else:
        row_names = sorted(random.sample(range(1, total_nodes + 1), comp_size))

    a_bad = make_invalid_directed_subgraph(comp_size)
    adj = embed_component_in_adj(a_bad, total_nodes, row_names)

    save_testcase(case_index, adj, row_names)
    case_index += 1