import os
import numpy as np


def all_dags_intern(gm, a, row_names, tmp):
    gm = np.array(gm, dtype=int)
    a = np.array(a, dtype=int)
    row_names = np.array(row_names, dtype=int)

    if np.any((a + a.T) == 1):
        raise ValueError("The matrix is not entirely undirected. This should not happen!")

    if np.sum(a) == 0:
        gm_flat = gm.flatten(order="F")

        if tmp.size == 0:
            tmp2 = np.array([gm_flat], dtype=int)
        else:
            tmp2 = np.vstack([tmp, gm_flat])

        if len(np.unique(tmp2, axis=0)) == len(tmp2):
            tmp = tmp2
    else:
        sinks = np.where(np.sum(a, axis=0) > 0)[0]

        for x in sinks:
            gm2 = gm.copy()

            Adj = (a == 1)
            Adjx = Adj[x, :]

            if np.any(Adjx):
                un = np.where(Adjx)[0]
                #Adjx = Adj[x, :]
                pp = len(un)
                Adj2 = Adj[np.ix_(un, un)].copy()
                np.fill_diagonal(Adj2, True)
            else:
                Adj2 = np.array([[True]])

            if np.all(Adj2):
                if np.any(Adjx):
                    un_global = row_names[np.where(Adjx)[0]]
                    x_global = row_names[x]

                    gm2[un_global - 1, x_global - 1] = 1
                    gm2[x_global - 1, un_global - 1] = 0

                if a.shape[0] == 1:
                    a2 = np.zeros((0, 0), dtype=int)
                    row_names2 = np.array([], dtype=int)
                else:
                    mask = np.ones(a.shape[0], dtype=bool)
                    mask[x] = False
                    a2 = a[np.ix_(mask, mask)]
                    row_names2 = row_names[mask]

                tmp = all_dags_intern(gm2, a2, row_names2, tmp)

    return tmp


def read_testcase(filepath):
    gm = []
    a = []
    row_names = []
    mode = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line == "":
                continue

            if line == "gm:":
                mode = "gm"
                continue

            if line == "a:":
                mode = "a"
                continue

            if line == "row.names:":
                mode = "row_names"
                continue

            if mode == "gm":
                gm.append([int(x) for x in line.split()])
            elif mode == "a":
                a.append([int(x) for x in line.split()])
            elif mode == "row_names":
                row_names = [int(x) for x in line.split()]

    return np.array(gm, dtype=int), np.array(a, dtype=int), np.array(row_names, dtype=int)


#llm

script_dir = os.getcwd()
project_dir = os.path.dirname(script_dir)

testcase_dir = os.path.join(project_dir, "tests", "allDagsIntern", "Testcase")
output_dir = os.path.join(project_dir, "tests", "allDagsIntern", "Py_outputs")

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

output_file = os.path.join(output_dir, "all_results.txt")

print("Reading testcases from:", testcase_dir)
print("Saving output to:", output_file)

with open(output_file, "w", encoding="utf-8") as f:
    for i in range(1, 1501):
        filepath = os.path.join(testcase_dir, f"{i}.txt")
        gm, a, row_names = read_testcase(filepath)

        tmp_init = np.empty((0, gm.size), dtype=int)
        result = all_dags_intern(gm, a, row_names, tmp_init)

        f.write(f"Testcase {i}\n")

        f.write("gm:\n")
        for row in gm:
            f.write(" ".join(map(str, row)) + "\n")

        f.write("a:\n")
        for row in a:
            f.write(" ".join(map(str, row)) + "\n")

        f.write("row.names:\n")
        f.write(" ".join(map(str, row_names)) + "\n")

        f.write("Result:\n")
        if result.shape[0] == 0:
            f.write("empty\n")
        else:
            for row in result:
                f.write(" ".join(map(str, row)) + "\n")

        f.write("\n")

print("Done. Output saved to:", output_file)