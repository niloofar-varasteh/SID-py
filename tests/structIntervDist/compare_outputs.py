import re
import numpy as np


def normalize_bool(value):
    return value.strip().lower()


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

                if line == "":
                    i += 1
                    continue

                if line == "trueGraph:":
                    graph = []
                    i += 1
                    while i < len(lines) and re.fullmatch(r"[01 ]+", lines[i].strip()):
                        graph.append([int(x) for x in lines[i].split()])
                        i += 1
                    result["trueGraph"] = np.array(graph, dtype=int)
                    continue

                if line == "estGraph:":
                    graph = []
                    i += 1
                    while i < len(lines) and re.fullmatch(r"[01 ]+", lines[i].strip()):
                        graph.append([int(x) for x in lines[i].split()])
                        i += 1
                    result["estGraph"] = np.array(graph, dtype=int)
                    continue

                if line == "output:":
                    result["output"] = normalize_bool(lines[i + 1])
                    i += 2
                    continue

                if line == "spars:":
                    result["spars"] = normalize_bool(lines[i + 1])
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
                        and lines[i].strip() != ""
                        and not lines[i].startswith("Testcase")
                    ):
                        if re.fullmatch(r"[01 ]+", lines[i].strip()):
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


def matrices_equal(a, b):
    if a.shape != b.shape:
        return False
    return np.array_equal(a, b)


def compare_results(r_results, py_results):
    all_cases = sorted(set(r_results.keys()) | set(py_results.keys()))

    matched = 0
    differences = []

    for tc in all_cases:
        if tc not in r_results:
            differences.append(f"- Testcase {tc}: missing in R file")
            continue

        if tc not in py_results:
            differences.append(f"- Testcase {tc}: missing in Python file")
            continue

        r = r_results[tc]
        p = py_results[tc]

        ok = True

        if not matrices_equal(r["trueGraph"], p["trueGraph"]):
            differences.append(f"- Testcase {tc}: trueGraph mismatch")
            ok = False

        if not matrices_equal(r["estGraph"], p["estGraph"]):
            differences.append(f"- Testcase {tc}: estGraph mismatch")
            ok = False

        if r["output"] != p["output"]:
            differences.append(f"- Testcase {tc}: output flag mismatch")
            ok = False

        if r["spars"] != p["spars"]:
            differences.append(f"- Testcase {tc}: spars flag mismatch")
            ok = False

        if r["sid"] != p["sid"]:
            differences.append(f"- Testcase {tc}: sid mismatch")
            ok = False

        if r["sidUpperBound"] != p["sidUpperBound"]:
            differences.append(f"- Testcase {tc}: sidUpperBound mismatch")
            ok = False

        if r["sidLowerBound"] != p["sidLowerBound"]:
            differences.append(f"- Testcase {tc}: sidLowerBound mismatch")
            ok = False

        if not matrices_equal(r["incorrectMat"], p["incorrectMat"]):
            differences.append(f"- Testcase {tc}: incorrectMat mismatch")
            ok = False

        if ok:
            matched += 1

    return all_cases, matched, differences


r_results = read_results("R_outputs/all_results.txt")
py_results = read_results("Py_outputs/all_results.txt")

all_cases, matched, differences = compare_results(r_results, py_results)

report_lines = []
report_lines.append("===== COMPARISON REPORT structIntervDist =====")
report_lines.append(f"R testcases found: {len(r_results)}")
report_lines.append(f"Python testcases found: {len(py_results)}")
report_lines.append(f"Matched testcases: {matched}")
report_lines.append(f"Differences found: {len(differences)}")
report_lines.append("")

if len(differences) == 0:
    report_lines.append("All testcase inputs and SID results are identical.")
else:
    report_lines.append("Differences:")
    report_lines.extend(differences)

report_text = "\n".join(report_lines)

print(report_text)

with open("comparison_report.txt", "w", encoding="utf-8") as f:
    f.write(report_text + "\n")