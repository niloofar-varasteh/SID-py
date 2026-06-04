import os
import numpy as np


def hamming_dist(G1, G2, all_mistakes_one=True):
    G1 = np.array(G1, dtype=int)
    G2 = np.array(G2, dtype=int)

    if all_mistakes_one:
        Gtmp = (G1 + G2) % 2
        Gtmp = Gtmp + Gtmp.T
        nr_reversals = np.sum(Gtmp == 2) / 2
        nr_incl_del = np.sum(Gtmp == 1) / 2
        hamming_dis = nr_reversals + nr_incl_del
    else:
        hamming_dis = np.sum(np.abs(G1 - G2))
        hamming_dis = hamming_dis - 0.5 * np.sum(
            G1 * G1.T * (1 - G2) * (1 - G2).T +
            G2 * G2.T * (1 - G1) * (1 - G1).T
        )

    return hamming_dis


def read_testcase(filepath):
    G1 = []
    G2 = []
    mode = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line == "":
                continue

            if line == "Matrix1:":
                mode = "matrix1"
                continue

            if line == "Matrix2:":
                mode = "matrix2"
                continue

            if mode == "matrix1":
                G1.append([int(x) for x in line.split()])
            elif mode == "matrix2":
                G2.append([int(x) for x in line.split()])

    return np.array(G1, dtype=int), np.array(G2, dtype=int)


#llm

script_dir = os.getcwd()
project_dir = os.path.dirname(script_dir)

testcase_dir = os.path.join(project_dir, "tests", "hammingDist", "Testcase")
output_dir = os.path.join(project_dir, "tests", "hammingDist", "Py_outputs")

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

output_file = os.path.join(output_dir, "all_results.txt")

print("Reading testcases from:", testcase_dir)
print("Saving output to:", output_file)

with open(output_file, "w", encoding="utf-8") as f:
    for i in range(1, 101):
        filepath = os.path.join(testcase_dir, f"{i}.txt")
        G1, G2 = read_testcase(filepath)

        result = hamming_dist(G1, G2)

        f.write(f"Testcase {i}\n")

        f.write("Matrix1:\n")
        for row in G1:
            f.write(" ".join(map(str, row)) + "\n")

        f.write("Matrix2:\n")
        for row in G2:
            f.write(" ".join(map(str, row)) + "\n")

        f.write("Result:\n")
        f.write(str(result) + "\n")

        f.write("\n")

print("Done. Output saved to:", output_file)