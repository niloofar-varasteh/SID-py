import re
import numpy as np

#llm
def read_results(filepath):
    results = {}

    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.rstrip() for line in f]

    i = 0

    while i < len(lines):

        if lines[i].startswith("Testcase"):

            testcase = int(lines[i].split()[1])

            result = {}

            i += 1

            while i < len(lines) and not lines[i].startswith("Testcase"):

                line = lines[i].strip()

                if line == "trueGraph:":
                    graph = []
                    i += 1
                    while i < len(lines) and re.fullmatch(r"[01 ]+", lines[i]):
                        graph.append([int(x) for x in lines[i].split()])
                        i += 1
                    result["trueGraph"] = graph
                    continue

                if line == "estGraph:":
                    graph = []
                    i += 1
                    while i < len(lines) and re.fullmatch(r"[01 ]+", lines[i]):
                        graph.append([int(x) for x in lines[i].split()])
                        i += 1
                    result["estGraph"] = graph
                    continue

                if line == "output:":
                    result["output"] = lines[i + 1].strip()
                    i += 2
                    continue

                if line == "spars:":
                    result["spars"] = lines[i + 1].strip()
                    i += 2
                    continue

                if line == "sid:":
                    result["sid"] = int(lines[i + 1])
                    i += 2
                    continue

                if line == "sidUpperBound:":
                    result["sidUpperBound"] = int(lines[i + 1])
                    i += 2
                    continue

                if line == "sidLowerBound:":
                    result["sidLowerBound"] = int(lines[i + 1])
                    i += 2
                    continue

                if line == "incorrectMat:":
                    matrix = []
                    i += 1

                    while (
                        i < len(lines)
                        and lines[i] != ""
                        and not lines[i].startswith("Testcase")
                    ):
                        if re.fullmatch(r"[01 ]+", lines[i]):
                            matrix.append([int(x) for x in lines[i].split()])
                            i += 1
                        else:
                            break

                    result["incorrectMat"] = np.array(matrix, dtype=int)
                    continue

                i += 1

            results[testcase] = result

        else:
            i += 1

    return results


r_results = read_results("R_outputs/all_results.txt")
py_results = read_results("Py_outputs/all_results.txt")

all_cases = sorted(set(r_results.keys()) | set(py_results.keys()))

passed = 0

for tc in all_cases:

    if tc not in r_results:
        print(f"Testcase {tc}: missing in R")
        continue

    if tc not in py_results:
        print(f"Testcase {tc}: missing in Python")
        continue

    r = r_results[tc]
    p = py_results[tc]

    ok = True

    if r["sid"] != p["sid"]:
        print(f"Testcase {tc}: sid mismatch")
        ok = False

    if r["sidUpperBound"] != p["sidUpperBound"]:
        print(f"Testcase {tc}: sidUpperBound mismatch")
        ok = False

    if r["sidLowerBound"] != p["sidLowerBound"]:
        print(f"Testcase {tc}: sidLowerBound mismatch")
        ok = False

    if not np.array_equal(r["incorrectMat"], p["incorrectMat"]):
        print(f"Testcase {tc}: incorrectMat mismatch")
        ok = False

    if ok:
        passed += 1

print()
print("=" * 50)
print(f"Passed {passed}/{len(all_cases)} testcases")
print("=" * 50)