import os
import numpy as np


def compute_caus_order(G):
    G = np.array(G, dtype=int)
    p = G.shape[1]
    remaining = list(range(1, p + 1))
    caus_order = []

    for _ in range(p - 1):
        #I write other form :)
        root = min(np.where(G.sum(axis=0) == 0)[0])
        caus_order.append(remaining[root])
        remaining.pop(root)
        G = np.delete(G, root, axis=0)
        G = np.delete(G, root, axis=1)

    caus_order.append(remaining[0])
    return caus_order


def dag2cpdag_adj(Adj):
    Adj = np.array(Adj, dtype=int)

    if Adj.sum() == 0:
        return Adj

    causal_order = compute_caus_order(Adj)
    order_idx = [x - 1 for x in causal_order]
    ordered_adj = Adj[np.ix_(order_idx, order_idx)]

    try:
        import causaldag as cd
    except ImportError:
        raise ImportError(
            "The Python package 'causaldag' is required for dag2cpdagAdj. "
            "Install it first, for example with: pip install causaldag"
        )

    nodes = list(range(ordered_adj.shape[0]))
    arcs = set()

    for i in range(ordered_adj.shape[0]):
        for j in range(ordered_adj.shape[1]):
            if ordered_adj[i, j] == 1:
                arcs.add((i, j))

    dag = cd.DAG(nodes=nodes, arcs=arcs)
    cpdag = dag.cpdag()

    cpdag_matrix_ordered = np.zeros_like(ordered_adj, dtype=int)

    #directed edges
    for u, v in cpdag.arcs:
        cpdag_matrix_ordered[u, v] = 1

    #undirected edges
    for u, v in cpdag.edges:
        cpdag_matrix_ordered[u, v] = 1
        cpdag_matrix_ordered[v, u] = 1

    result = np.zeros_like(Adj, dtype=int)
    result[np.ix_(order_idx, order_idx)] = cpdag_matrix_ordered

    return result



#llm

script_dir = os.getcwd()
project_dir = os.path.dirname(script_dir)

testcase_dir = os.path.join(project_dir, "tests", "dag2cpdagAdj", "Testcase")
output_dir = os.path.join(project_dir, "tests", "dag2cpdagAdj", "Py_outputs")

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

output_file = os.path.join(output_dir, "all_results.txt")

print("Reading testcases from:", testcase_dir)
print("Saving output to:", output_file)

with open(output_file, "w", encoding="utf-8") as f:
    for i in range(1, 101):
        filepath = os.path.join(testcase_dir, f"{i}.txt")
        G = np.loadtxt(filepath, dtype=int, ndmin=2)

        result = dag2cpdag_adj(G)

        f.write(f"Testcase {i}\n")
        f.write("Matrix:\n")
        for row in G:
            f.write(" ".join(map(str, row)) + "\n")

        f.write("Result Matrix:\n")
        for row in result:
            f.write(" ".join(map(str, row)) + "\n")

        f.write("\n")

print("Done. Output saved to:", output_file)