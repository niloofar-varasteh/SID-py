import re
from pathlib import Path

#LLM

base_dir = Path(__file__).resolve().parent
r_file = base_dir / "R_outputs" / "all_results.txt"
py_file = base_dir / "Py_outputs" / "all_results.txt"


def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    results = {}
    case_num = None
    matrix = []
    result = None
    reading_matrix = False

    for line in lines:
        line = line.strip()

        if line.startswith("Testcase "):
            if case_num is not None:
                results[case_num] = {
                    "matrix": matrix,
                    "result": result
                }

            case_num = int(line.replace("Testcase ", ""))
            matrix = []
            result = None
            reading_matrix = False

        elif line == "Matrix:":
            reading_matrix = True

        elif line == "Result:":
            reading_matrix = False

        elif line == "":
            continue

        else:
            if reading_matrix:
                row = [int(x) for x in line.split()]
                matrix.append(row)
            else:
                nums = re.findall(r"-?\d+", line)
                result = [int(x) for x in nums]

    if case_num is not None:
        results[case_num] = {
            "matrix": matrix,
            "result": result
        }

    return results

#LLM

if not r_file.exists():
    print("R file not found:", r_file)
    raise SystemExit

if not py_file.exists():
    print("Python file not found:", py_file)
    raise SystemExit

r_results = read_file(r_file)
py_results = read_file(py_file)

all_cases = sorted(set(r_results.keys()) | set(py_results.keys()))
differences = []

for case in all_cases:
    if case not in r_results:
        differences.append(f"Testcase {case}: missing in R file")
        continue

    if case not in py_results:
        differences.append(f"Testcase {case}: missing in Python file")
        continue

    if r_results[case]["matrix"] != py_results[case]["matrix"]:
        differences.append(f"Testcase {case}: matrix mismatch")

    if r_results[case]["result"] != py_results[case]["result"]:
        differences.append(
            f"Testcase {case}: result mismatch | R={r_results[case]['result']} | PY={py_results[case]['result']}"
        )

#LLM

print("===== COMPARISON REPORT =====")
print("R testcases found:", len(r_results))
print("Python testcases found:", len(py_results))
print("Differences found:", len(differences))

if differences:
    print("\nDifferences:")
    for item in differences:
        print("-", item)
else:
    print("\nAll testcase matrices and results are identical.")

report_file = base_dir / "comparison_report.txt"

with open(report_file, "w", encoding="utf-8") as f:
    f.write("===== COMPARISON REPORT =====\n")
    f.write(f"R testcases found: {len(r_results)}\n")
    f.write(f"Python testcases found: {len(py_results)}\n")
    f.write(f"Differences found: {len(differences)}\n\n")

    if differences:
        f.write("Differences:\n")
        for item in differences:
            f.write(item + "\n")
    else:
        f.write("All testcase matrices and results are identical.\n")

print("\nReport saved to", report_file)