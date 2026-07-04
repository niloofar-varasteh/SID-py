import os
import numpy as np
from allDagsIntern import all_dags_intern


def all_dags_jonas(adj, row_names):
    adj = np.array(adj, dtype=int)
    row_names = np.array(row_names, dtype=int)

    a = adj[np.ix_(row_names - 1, row_names - 1)]

    if np.any((a + a.T) == 1):
        return -1

    tmp_init = np.empty((0, adj.size), dtype=int)
    return all_dags_intern(adj, a, row_names, tmp_init)


def read_testcase(filepath):
    adj = []
    row_names = []
    mode = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line == "":
                continue

            if line == "adj:":
                mode = "adj"
                continue

            if line == "row.names:":
                mode = "row_names"
                continue

            if mode == "adj":
                adj.append([int(x) for x in line.split()])
            elif mode == "row_names":
                row_names = [int(x) for x in line.split()]

    return np.array(adj, dtype=int), np.array(row_names, dtype=int)


#llm

script_dir = os.getcwd()
project_dir = os.path.dirname(script_dir)

testcase_dir = os.path.join(project_dir, "tests", "allDagsJonas", "Testcase")
output_dir = os.path.join(project_dir, "tests", "allDagsJonas", "Py_outputs")

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

output_file = os.path.join(output_dir, "all_results.txt")

print("Reading testcases from:", testcase_dir)
print("Saving output to:", output_file)

with open(output_file, "w", encoding="utf-8") as f:
    for i in range(1, 10001):
        filepath = os.path.join(testcase_dir, f"{i}.txt")
        adj, row_names = read_testcase(filepath)

        result = all_dags_jonas(adj, row_names)

        f.write(f"Testcase {i}\n")

        f.write("adj:\n")
        for row in adj:
            f.write(" ".join(map(str, row)) + "\n")

        f.write("row.names:\n")
        f.write(" ".join(map(str, row_names)) + "\n")

        f.write("Result:\n")
        if isinstance(result, int) and result == -1:
            f.write("-1\n")
        elif result.shape[0] == 0:
            f.write("empty\n")
        else:
            for row in result:
                f.write(" ".join(map(str, row)) + "\n")

        f.write("\n")

print("Done. Output saved to:", output_file)