
import os
import random

NUM_TESTCASES = 10000
random.seed(7)

os.makedirs("Testcase", exist_ok=True)


def save_testcase(index, true_graph, est_graph, output=False, spars=False):
    with open(f"Testcase/{index}.txt", "w", encoding="utf-8") as f:
        f.write("trueGraph:\n")
        for row in true_graph:
            f.write(" ".join(map(str, row)) + "\n")

        f.write("\n")
        f.write("estGraph:\n")
        for row in est_graph:
            f.write(" ".join(map(str, row)) + "\n")

        f.write("\n")
        f.write("output:\n")
        f.write(str(output) + "\n")

        f.write("\n")
        f.write("spars:\n")
        f.write(str(spars) + "\n")


def copy_graph(g):
    return [row[:] for row in g]


def make_empty_graph(n):
    return [[0 for _ in range(n)] for _ in range(n)]


def make_single_edge_graph():
    g = make_empty_graph(2)
    g[0][1] = 1
    return g


def make_chain_graph(n):
    g = make_empty_graph(n)
    for i in range(n - 1):
        g[i][i + 1] = 1
    return g


def make_star_graph(n):
    g = make_empty_graph(n)
    for j in range(1, n):
        g[0][j] = 1
    return g


def make_fork_graph():
    # 1 <- 2 -> 3
    g = make_empty_graph(3)
    g[1][0] = 1
    g[1][2] = 1
    return g


def make_collider_graph():
    # 1 -> 3 <- 2
    g = make_empty_graph(3)
    g[0][2] = 1
    g[1][2] = 1
    return g


def make_disconnected_graph():
    # 1 -> 2 and 3 -> 4
    g = make_empty_graph(4)
    g[0][1] = 1
    g[2][3] = 1
    return g


def make_tree_graph(n):
    g = make_empty_graph(n)
    for child in range(1, n):
        parent = (child - 1) // 2
        g[parent][child] = 1
    return g


def make_random_dag(n, edge_prob):
    g = make_empty_graph(n)
    order = list(range(n))
    random.shuffle(order)

    for a in range(n):
        for b in range(a + 1, n):
            if random.random() < edge_prob:
                g[order[a]][order[b]] = 1

    # make sure it is not totally empty for n > 1
    if n > 1 and len(list_edges(g)) == 0:
        g[0][1] = 1

    return g

#llm
def list_edges(g):
    edges = []
    n = len(g)
    for i in range(n):
        for j in range(n):
            if g[i][j] == 1:
                edges.append((i, j))
    return edges


def list_non_edges(g):
    non_edges = []
    n = len(g)
    for i in range(n):
        for j in range(n):
            if i != j and g[i][j] == 0:
                non_edges.append((i, j))
    return non_edges


def make_estimated_graph(true_graph, mode):
    est = copy_graph(true_graph)
    edges = list_edges(est)

    if mode == "same":
        return est

    if mode == "remove":
        if len(edges) == 0:
            return est
        i, j = random.choice(edges)
        est[i][j] = 0
        return est

    if mode == "reverse":
        if len(edges) == 0:
            return est
        i, j = random.choice(edges)
        est[i][j] = 0
        est[j][i] = 1
        return est

    if mode == "extra":
        candidates = list_non_edges(est)
        if len(candidates) == 0:
            return est
        i, j = random.choice(candidates)
        est[i][j] = 1
        return est

    if mode == "undirected":
        if len(edges) == 0:
            return est
        i, j = random.choice(edges)
        est[i][j] = 1
        est[j][i] = 1
        return est

    return est


case_index = 1


# --------------------------------
# A) small and fixed cases
# --------------------------------

true_graph = make_empty_graph(1)
est_graph = copy_graph(true_graph)
save_testcase(case_index, true_graph, est_graph)
case_index += 1

true_graph = make_single_edge_graph()
est_graph = copy_graph(true_graph)
save_testcase(case_index, true_graph, est_graph)
case_index += 1

true_graph = make_single_edge_graph()
est_graph = make_estimated_graph(true_graph, "reverse")
save_testcase(case_index, true_graph, est_graph)
case_index += 1

true_graph = make_chain_graph(3)
est_graph = copy_graph(true_graph)
save_testcase(case_index, true_graph, est_graph)
case_index += 1

true_graph = make_chain_graph(3)
est_graph = make_estimated_graph(true_graph, "remove")
save_testcase(case_index, true_graph, est_graph)
case_index += 1

true_graph = make_chain_graph(3)
est_graph = make_estimated_graph(true_graph, "extra")
save_testcase(case_index, true_graph, est_graph)
case_index += 1

true_graph = make_chain_graph(4)
est_graph = make_estimated_graph(true_graph, "undirected")
save_testcase(case_index, true_graph, est_graph)
case_index += 1

true_graph = make_star_graph(4)
est_graph = copy_graph(true_graph)
save_testcase(case_index, true_graph, est_graph)
case_index += 1

true_graph = make_star_graph(4)
est_graph = make_estimated_graph(true_graph, "reverse")
save_testcase(case_index, true_graph, est_graph)
case_index += 1

true_graph = make_fork_graph()
est_graph = copy_graph(true_graph)
save_testcase(case_index, true_graph, est_graph)
case_index += 1

true_graph = make_collider_graph()
est_graph = copy_graph(true_graph)
save_testcase(case_index, true_graph, est_graph)
case_index += 1

true_graph = make_disconnected_graph()
est_graph = copy_graph(true_graph)
save_testcase(case_index, true_graph, est_graph)
case_index += 1

true_graph = make_tree_graph(5)
est_graph = copy_graph(true_graph)
save_testcase(case_index, true_graph, est_graph)
case_index += 1

true_graph = make_tree_graph(5)
est_graph = make_estimated_graph(true_graph, "undirected")
save_testcase(case_index, true_graph, est_graph)
case_index += 1

true_graph = make_tree_graph(6)
est_graph = make_estimated_graph(true_graph, "remove")
save_testcase(case_index, true_graph, est_graph)
case_index += 1


# --------------------------------
# B) some structured fixed random-like cases
# --------------------------------

fixed_cases = [
    (3, 0.2, "same"),
    (3, 0.5, "reverse"),
    (4, 0.2, "same"),
    (4, 0.5, "remove"),
    (4, 0.7, "extra"),
    (5, 0.2, "same"),
    (5, 0.5, "undirected"),
    (5, 0.8, "reverse"),
    (6, 0.2, "same"),
    (6, 0.5, "remove"),
    (6, 0.8, "extra"),
    (6, 0.8, "undirected"),
]

for n, edge_prob, mode in fixed_cases:
    true_graph = make_random_dag(n, edge_prob)
    est_graph = make_estimated_graph(true_graph, mode)
    save_testcase(case_index, true_graph, est_graph)
    case_index += 1


# --------------------------------
# C) random cases for the rest
# --------------------------------

while case_index <= NUM_TESTCASES:
    choice = random.choice(["chain", "star", "tree", "random"])

    if choice == "chain":
        true_graph = make_chain_graph(random.randint(3, 7))
    elif choice == "star":
        true_graph = make_star_graph(random.randint(4, 7))
    elif choice == "tree":
        true_graph = make_tree_graph(random.randint(4, 7))
    else:
        true_graph = make_random_dag(
            random.randint(3, 7),
            random.choice([0.2, 0.4, 0.6])
        )

    mode = random.choice(["same", "remove", "reverse", "extra", "undirected"])
    est_graph = make_estimated_graph(true_graph, mode)

    save_testcase(case_index, true_graph, est_graph)
    case_index += 1

