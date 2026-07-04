import os
import random
import numpy as np


def random_dag(p, prob_connect, causal_order=None):
    if causal_order is None:
        causal_order = random.sample(range(1, p + 1), p)

    dag = np.zeros((p, p), dtype=int)

    for i in range(p - 2):
        node = causal_order[i]
        possible_parents = causal_order[i + 1:p]

        number_parents = np.random.binomial(p - i - 1, prob_connect)

        if number_parents > 0:
            parents = random.sample(possible_parents, number_parents)
            for parent in parents:
                dag[parent - 1, node - 1] = 1

    if p >= 2:
        node = causal_order[p - 2]
        parent_yes_no = np.random.binomial(1, prob_connect)
        dag[causal_order[p - 1] - 1, node - 1] = parent_yes_no

    return dag


def read_testcase(filepath):
    values = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line == "":
                continue

            parts = line.split("=", 1)
            key = parts[0].strip()
            value = parts[1].strip()

            if key == "p":
                values["p"] = int(value)
            elif key == "probConnect":
                values["probConnect"] = float(value)
            elif key == "causalOrder":
                values["causalOrder"] = [int(x) for x in value.split()]

    return values


#llm

script_dir = os.getcwd()
project_dir = os.path.dirname(script_dir)

testcase_dir = os.path.join(project_dir, "tests", "randomDAG", "Testcase")
output_dir = os.path.join(project_dir, "tests", "randomDAG", "Py_outputs")

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

output_file = os.path.join(output_dir, "all_results.txt")

print("Reading testcases from:", testcase_dir)
print("Saving output to:", output_file)

with open(output_file, "w", encoding="utf-8") as f:
    for i in range(1, 10001):
        filepath = os.path.join(testcase_dir, f"{i}.txt")
        params = read_testcase(filepath)

        p = params["p"]
        prob_connect = params["probConnect"]
        causal_order = params["causalOrder"]

        result = random_dag(p, prob_connect, causal_order)

        f.write(f"Testcase {i}\n")
        f.write(f"p={p}\n")
        f.write(f"probConnect={prob_connect}\n")
        f.write("causalOrder=" + " ".join(map(str, causal_order)) + "\n")

        f.write("Result Matrix:\n")
        for row in result:
            f.write(" ".join(map(str, row)) + "\n")

        f.write("\n")

print("Done. Output saved to:", output_file)