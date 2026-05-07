import os
import numpy as np
from pathlib import Path

def compute_caus_order(G):

    G = np.array(G, dtype=int)
    p = G.shape[1]
    remaining = list(range(1, p + 1))
    caus_order = [None] * p

    for i in range(p - 1):

        root = np.where(G.sum(axis=0) == 0)[0][0] #LLM
        caus_order[i] = remaining[root]
        del remaining[root]
        G = np.delete(G, root, axis=0)
        G = np.delete(G, root, axis=1)
    caus_order[p - 1] = remaining[0]


    return caus_order

#LLM

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
testcase_dir = PROJECT_DIR / "tests" / "computeCausOrder" / "Testcase"
output_dir = PROJECT_DIR / "tests" / "computeCausOrder" / "Py_outputs"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "all_results.txt"


with open(output_file, "w", encoding="utf-8") as f:


    for i in range(1, 101):

        filepath = testcase_dir / f"{i}.txt"
        # Builds the path for the current test case file.

        G = np.loadtxt(filepath, dtype=int)
        # Loads the adjacency matrix from the text file into a NumPy array.

        result = compute_caus_order(G.copy())


        f.write(f"Testcase {i}\n")


        f.write("Matrix:\n")


        for row in G:


            f.write(" ".join(map(str, row.tolist())) + "\n")

        f.write("Result:\n")


        f.write("[" + ", ".join(map(str, result)) + "]\n")
        # Writes the causal order as a list-like string [1, 2, 3, 4]

        f.write("\n")
