import os


base_dir = os.path.dirname(os.path.abspath(__file__))
r_file = os.path.join(base_dir, "R_outputs", "all_results.txt")
py_file = os.path.join(base_dir, "Py_outputs", "all_results.txt")


def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    results = {}
    case_num = None
    p = None
    prob_connect = None
    causal_order = None
    result_matrix = []
    mode = None

    for line in lines:
        line = line.strip()

        if line.startswith("Testcase "):
            if case_num is not None:
                results[case_num] = {
                    "p": p,
                    "probConnect": prob_connect,
                    "causalOrder": causal_order,
                    "result_matrix": result_matrix
                }

            case_num = int(line.replace("Testcase ", ""))
            p = None
            prob_connect = None
            causal_order = None
            result_matrix = []
            mode = None

        elif line.startswith("p="):
            p = int(line.replace("p=", "").strip())

        elif line.startswith("probConnect="):
            prob_connect = float(line.replace("probConnect=", "").strip())

        elif line.startswith("causalOrder="):
            value = line.replace("causalOrder=", "").strip()
            causal_order = [int(x) for x in value.split()]

        elif line == "Result Matrix:":
            mode = "result"

        elif line == "":
            continue

        else:
            if mode == "result":
                row = [int(x) for x in line.split()]
                result_matrix.append(row)

    if case_num is not None:
        results[case_num] = {
            "p": p,
            "probConnect": prob_connect,
            "causalOrder": causal_order,
            "result_matrix": result_matrix
        }

    return results


def is_binary_matrix(matrix):
    for row in matrix:
        for x in row:
            if x not in [0, 1]:
                return False
    return True


def has_zero_diagonal(matrix):
    n = len(matrix)
    for i in range(n):
        if matrix[i][i] != 0:
            return False
    return True


def has_correct_shape(matrix, p):
    if len(matrix) != p:
        return False
    for row in matrix:
        if len(row) != p:
            return False
    return True


def respects_causal_order(matrix, causal_order):
    pos = {}
    for i, node in enumerate(causal_order):
        pos[node] = i

    n = len(matrix)

    for parent in range(n):
        for child in range(n):
            if matrix[parent][child] == 1:
                parent_node = parent + 1
                child_node = child + 1

                if pos[parent_node] <= pos[child_node]:
                    return False

    return True


def is_acyclic(matrix):
    n = len(matrix)
    indegree = [0] * n

    for i in range(n):
        for j in range(n):
            if matrix[i][j] == 1:
                indegree[j] += 1

    queue = []
    for i in range(n):
        if indegree[i] == 0:
            queue.append(i)

    visited = 0

    while queue:
        node = queue.pop(0)
        visited += 1

        for j in range(n):
            if matrix[node][j] == 1:
                indegree[j] -= 1
                if indegree[j] == 0:
                    queue.append(j)

    return visited == n


def check_properties(case_data):
    problems = []

    p = case_data["p"]
    causal_order = case_data["causalOrder"]
    matrix = case_data["result_matrix"]

    if not has_correct_shape(matrix, p):
        problems.append("wrong matrix shape")

    if not is_binary_matrix(matrix):
        problems.append("matrix is not binary")

    if has_correct_shape(matrix, p):
        if not has_zero_diagonal(matrix):
            problems.append("diagonal is not zero")

        if not is_acyclic(matrix):
            problems.append("graph has a cycle")

        if not respects_causal_order(matrix, causal_order):
            problems.append("graph does not respect causalOrder")

    return problems


if not os.path.exists(r_file):
    print("R file not found:", r_file)
    raise SystemExit

if not os.path.exists(py_file):
    print("Python file not found:", py_file)
    raise SystemExit


r_results = read_file(r_file)
py_results = read_file(py_file)

all_cases = sorted(set(r_results.keys()) | set(py_results.keys()))
differences = []
passed_cases = []

for case in all_cases:
    if case not in r_results:
        differences.append(f"Testcase {case}: missing in R file")
        continue

    if case not in py_results:
        differences.append(f"Testcase {case}: missing in Python file")
        continue

    r_case = r_results[case]
    py_case = py_results[case]

    case_ok = True

    if r_case["p"] != py_case["p"]:
        differences.append(f"Testcase {case}: p mismatch")
        case_ok = False

    if r_case["probConnect"] != py_case["probConnect"]:
        differences.append(f"Testcase {case}: probConnect mismatch")
        case_ok = False

    if r_case["causalOrder"] != py_case["causalOrder"]:
        differences.append(f"Testcase {case}: causalOrder mismatch")
        case_ok = False

    r_problems = check_properties(r_case)
    py_problems = check_properties(py_case)

    if r_problems:
        differences.append(f"Testcase {case}: R output problem -> {', '.join(r_problems)}")
        case_ok = False

    if py_problems:
        differences.append(f"Testcase {case}: Python output problem -> {', '.join(py_problems)}")
        case_ok = False

    if case_ok:
        passed_cases.append(case)


print("===== COMPARISON REPORT =====")
print("R testcases found:", len(r_results))
print("Python testcases found:", len(py_results))
print("Passed testcases:", len(passed_cases))
print("Failed testcases:", len(differences))

if differences:
    print("\nProblems found:")
    for item in differences:
        print("-", item)
else:
    print("\nAll testcases passed the property-based checks.")

report_file = os.path.join(base_dir, "comparison_report.txt")

with open(report_file, "w", encoding="utf-8") as f:
    f.write("===== COMPARISON REPORT =====\n")
    f.write(f"R testcases found: {len(r_results)}\n")
    f.write(f"Python testcases found: {len(py_results)}\n")
    f.write(f"Passed testcases: {len(passed_cases)}\n")
    f.write(f"Failed testcases: {len(differences)}\n\n")

    if differences:
        f.write("Problems found:\n")
        for item in differences:
            f.write(item + "\n")
    else:
        f.write("All testcases passed the property-based checks.\n")

print("\nReport saved to", report_file)