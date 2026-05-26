from pathlib import Path

#llm
base_dir = Path(__file__).resolve().parent
r_file = base_dir / "R_outputs" / "all_results.txt"
py_file = base_dir / "Py_outputs" / "all_results.txt"


def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    results = {}
    case_num = None
    matrix = []
    result_matrix = []
    mode = None

    for line in lines:
        line = line.strip()

        if line.startswith("Testcase "):
            if case_num is not None:
                results[case_num] = {
                    "matrix": matrix,
                    "result_matrix": result_matrix
                }

            case_num = int(line.replace("Testcase ", ""))
            matrix = []
            result_matrix = []
            mode = None

        elif line == "Matrix:":
            mode = "matrix"

        elif line == "Result Matrix:":
            mode = "result"

        elif line == "":
            continue

        else:
            row = [int(x) for x in line.split()]

            if mode == "matrix":
                matrix.append(row)
            elif mode == "result":
                result_matrix.append(row)

    if case_num is not None:
        results[case_num] = {
            "matrix": matrix,
            "result_matrix": result_matrix
        }

    return results


if not r_file.exists():
    print("R file not found:", r_file)
    raise SystemExit

if not py_file.exists():
    print("Python file not found:", py_file)
    raise SystemExit

#llm
r_results = read_file(r_file)
py_results = read_file(py_file)

all_cases = sorted(set(r_results.keys()) | set(py_results.keys()))
differences = []
matched_cases = []

for case in all_cases:
    if case not in r_results:
        differences.append(f"Testcase {case}: missing in R file")
        continue

    if case not in py_results:
        differences.append(f"Testcase {case}: missing in Python file")
        continue

    case_ok = True

    if r_results[case]["matrix"] != py_results[case]["matrix"]:
        differences.append(f"Testcase {case}: input matrix mismatch")
        case_ok = False

    if r_results[case]["result_matrix"] != py_results[case]["result_matrix"]:
        differences.append(f"Testcase {case}: result matrix mismatch")
        case_ok = False

    if case_ok:
        matched_cases.append(case)


print("===== COMPARISON REPORT =====")
print("R testcases found:", len(r_results))
print("Python testcases found:", len(py_results))
print("Matched testcases:", len(matched_cases))
print("Differences found:", len(differences))

if differences:
    print("\nDifferences:")
    for item in differences:
        print("-", item)
else:
    print("\nAll testcase matrices and result matrices are identical.")

report_file = base_dir / "comparison_report.txt"

with open(report_file, "w", encoding="utf-8") as f:
    f.write("===== COMPARISON REPORT =====\n")
    f.write(f"R testcases found: {len(r_results)}\n")
    f.write(f"Python testcases found: {len(py_results)}\n")
    f.write(f"Matched testcases: {len(matched_cases)}\n")
    f.write(f"Differences found: {len(differences)}\n\n")

    if differences:
        f.write("Differences:\n")
        for item in differences:
            f.write(item + "\n")
    else:
        f.write("All testcase matrices and result matrices are identical.\n")

print("\nReport saved to", report_file)