import os
import numpy as np


def compute_path_matrix(G, spars=False):
    G = np.array(G, dtype=int)
    p = G.shape[1]

    if p > 3000 and spars == False:
        print("Warning: maybe use a sparse version for speed")

    path_matrix = np.eye(p, dtype=int) + G
    k = int(np.ceil(np.log2(p)))

    for i in range(k):
        path_matrix = path_matrix @ path_matrix

    path_matrix = (path_matrix > 0).astype(int)
    return path_matrix


#llm

script_dir = os.getcwd()
project_dir = os.path.dirname(script_dir)

testcase_dir = os.path.join(project_dir, "tests", "computePathMatrix", "Testcase")
output_dir = os.path.join(project_dir, "tests", "computePathMatrix", "Py_outputs")

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

output_file = os.path.join(output_dir, "all_results.txt")

print("Reading testcases from:", testcase_dir)
print("Saving output to:", output_file)

with open(output_file, "w", encoding="utf-8") as f:
    for i in range(1, 10001):
        filepath = os.path.join(testcase_dir, f"{i}.txt")
        G = np.loadtxt(filepath, dtype=int, ndmin=2)

        result = compute_path_matrix(G)

        f.write(f"Testcase {i}\n")
        f.write("Matrix:\n")
        for row in G:
            f.write(" ".join(map(str, row)) + "\n")

        f.write("Result Matrix:\n")
        for row in result:
            f.write(" ".join(map(str, row)) + "\n")

        f.write("\n")

print("Done. Output saved to:", output_file)