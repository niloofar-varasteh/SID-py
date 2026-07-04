import os
import random
import numpy as np

NUM_TESTCASES = 1500
os.makedirs("Testcase", exist_ok=True)


def save_testcase(index, gm, a, row_names):
    with open(f"Testcase/{index}.txt", "w", encoding="utf-8") as f:
        f.write("gm:\n")
        for row in gm:
            f.write(" ".join(map(str, row)) + "\n")

        f.write("\n")
        f.write("a:\n")
        for row in a:
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


def embed_component_in_gm(a, total_nodes, row_names):
    gm = np.zeros((total_nodes, total_nodes), dtype=int)

    idx = [x - 1 for x in row_names]

    for i in range(len(idx)):
        for j in range(len(idx)):
            gm[idx[i], idx[j]] = a[i, j]

    outside = [x for x in range(total_nodes) if x not in idx]

    # add a few simple directed edges among outside nodes only
    if len(outside) >= 2:
        for k in range(len(outside) - 1):
            gm[outside[k], outside[k + 1]] = 1

    return gm


def make_component_by_type(name, n):
    if name == "empty":
        return make_empty_component(n)
    elif name == "edge":
        return make_single_edge_component()
    elif name == "chain":
        return make_chain_component(n)
    elif name == "star":
        return make_star_component(n)
    elif name == "clique":
        return make_clique_component(n)
    elif name == "cycle":
        return make_cycle_component(n)
    else:
        return make_tree_component(n)


case_index = 1


#small cases
a = make_empty_component(1)
gm = a.copy()
save_testcase(case_index, gm, a, [1])
case_index += 1

a = make_single_edge_component()
gm = a.copy()
save_testcase(case_index, gm, a, [1, 2])
case_index += 1

a = make_chain_component(3)
gm = a.copy()
save_testcase(case_index, gm, a, [1, 2, 3])
case_index += 1

a = make_star_component(4)
gm = a.copy()
save_testcase(case_index, gm, a, [1, 2, 3, 4])
case_index += 1

a = make_clique_component(3)
gm = a.copy()
save_testcase(case_index, gm, a, [1, 2, 3])
case_index += 1

a = make_cycle_component(4)
gm = a.copy()
save_testcase(case_index, gm, a, [1, 2, 3, 4])
case_index += 1

a = make_tree_component(5)
gm = a.copy()
save_testcase(case_index, gm, a, [1, 2, 3, 4, 5])
case_index += 1


#embedded cases #llm
a = make_chain_component(3)
gm = embed_component_in_gm(a, 5, [2, 3, 4])
save_testcase(case_index, gm, a, [2, 3, 4])
case_index += 1

a = make_star_component(4)
gm = embed_component_in_gm(a, 6, [2, 3, 4, 5])
save_testcase(case_index, gm, a, [2, 3, 4, 5])
case_index += 1

a = make_clique_component(4)
gm = embed_component_in_gm(a, 6, [1, 3, 4, 6])
save_testcase(case_index, gm, a, [1, 3, 4, 6])
case_index += 1

a = make_cycle_component(5)
gm = embed_component_in_gm(a, 8, [2, 3, 4, 5, 6])
save_testcase(case_index, gm, a, [2, 3, 4, 5, 6])
case_index += 1

a = make_tree_component(6)
gm = embed_component_in_gm(a, 9, [2, 3, 4, 5, 6, 7])
save_testcase(case_index, gm, a, [2, 3, 4, 5, 6, 7])
case_index += 1


#random cases
while case_index <= 40:
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
        gm = a.copy()
        row_names = list(range(1, comp_size + 1))
    else:
        row_names = sorted(random.sample(range(1, total_nodes + 1), comp_size))
        gm = embed_component_in_gm(a, total_nodes, row_names)

    save_testcase(case_index, gm, a, row_names)
    case_index += 1



#larger cases
while case_index <= NUM_TESTCASES:
    choice = random.choice(["chain", "star", "clique", "cycle", "tree"])

    if choice == "chain":
        a = make_chain_component(random.randint(5, 8))
    elif choice == "star":
        a = make_star_component(random.randint(5, 8))
    elif choice == "clique":
        a = make_clique_component(random.randint(4, 6))
    elif choice == "cycle":
        a = make_cycle_component(random.randint(5, 7))
    else:
        a = make_tree_component(random.randint(5, 8))

    comp_size = a.shape[0]
    total_nodes = comp_size + random.randint(1, 4)
    row_names = sorted(random.sample(range(1, total_nodes + 1), comp_size))
    gm = embed_component_in_gm(a, total_nodes, row_names)

    save_testcase(case_index, gm, a, row_names)
    case_index += 1