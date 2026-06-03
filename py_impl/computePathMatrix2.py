import os
import numpy as np


def compute_path_matrix2(G, cond_set, path_matrix1, spars=False):
    G = np.array(G, dtype=int)
    path_matrix1 = np.array(path_matrix1, dtype=int)
    p = G.shape[1]

    if len(cond_set) > 0:
        G = G.copy()
        for node in cond_set:
            G[node - 1, :] = 0

        path_matrix2 = np.eye(p, dtype=int) + G
        k = int(np.ceil(np.log2(p))) if p > 1 else 0

        for i in range(k):
            path_matrix2 = path_matrix2 @ path_matrix2

        path_matrix2 = (path_matrix2 > 0).astype(int)
    else:
        path_matrix2 = path_matrix1

    return path_matrix2


def read_testcase(filepath):
    G = []
    path_matrix1 = []
    cond_set = []
    mode = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line == "":
                continue

            if line == "Matrix:":
                mode = "matrix"
                continue

            if line == "condSet:":
                mode = "condSet"
                continue

            if line == "PathMatrix1:":
                mode = "path1"
                continue

            if mode == "matrix":
                G.append([int(x) for x in line.split()])
            elif mode == "condSet":
                if line != "empty":
                    cond_set = [int(x) for x in line.split()]
            elif mode == "path1":
                path_matrix1.append([int(x) for x in line.split()])

    return np.array(G, dtype=int), cond_set, np.array(path_matrix1, dtype=int)


#llm
script_dir = os.getcwd()
project_dir = os.path.dirname(script_dir)

testcase_dir = os.path.join(project_dir, "tests", "computePathMatrix2", "Testcase")
output_dir = os.path.join(project_dir, "tests", "computePathMatrix2", "Py_outputs")

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

output_file = os.path.join(output_dir, "all_results.txt")

print("Reading testcases from:", testcase_dir)
print("Saving output to:", output_file)

with open(output_file, "w", encoding="utf-8") as f:
    for i in range(1, 101):
        filepath = os.path.join(testcase_dir, f"{i}.txt")
        G, cond_set, path_matrix1 = read_testcase(filepath)

        result = compute_path_matrix2(G, cond_set, path_matrix1)

        f.write(f"Testcase {i}\n")
        f.write("Matrix:\n")
        for row in G:
            f.write(" ".join(map(str, row)) + "\n")

        f.write("condSet:\n")
        if len(cond_set) == 0:
            f.write("empty\n")
        else:
            f.write(" ".join(map(str, cond_set)) + "\n")

        f.write("PathMatrix1:\n")
        for row in path_matrix1:
            f.write(" ".join(map(str, row)) + "\n")

        f.write("Result Matrix:\n")
        for row in result:
            f.write(" ".join(map(str, row)) + "\n")

        f.write("\n")

print("Done. Output saved to:", output_file)