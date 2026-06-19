from pathlib import Path

base_dir = Path(__file__).resolve().parent
r_file = base_dir / "R_outputs" / "all_results.txt"
py_file = base_dir / "Py_outputs" / "all_results.txt"


def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    results = {}
    case_num = None
    adj = []
    row_names = []
    result = []
    mode = None

    for line in lines:
        line = line.strip()

        if line.startswith("Testcase "):
            if case_num is not None:
                results[case_num] = {
                    "adj": adj,
                    "row_names": row_names,
                    "result": result
                }

            case_num = int(line.replace("Testcase ", ""))
            adj = []
            row_names = []
            result = []
            mode = None

        elif line == "adj:":
            mode = "adj"

        elif line == "row.names:":
            mode = "row_names"

        elif line == "Result:":
            mode = "result"

        elif line == "":
            continue

        else:
            if mode == "adj":
                adj.append([int(x) for x in line.split()])
            elif mode == "row_names":
                row_names = [int(x) for x in line.split()]
            elif mode == "result":
                if line == "-1":
                    result = -1
                elif line == "empty":
                    result = []
                else:
                    if result == -1:
                        result = []
                    result.append([int(x) for x in line.split()])

    if case_num is not None:
        results[case_num] = {
            "adj": adj,
            "row_names": row_names,
            "result": result
        }

    return results


def normalize_result(value):
    if value == -1:
        return -1
    if value == []:
        return []
    return sorted(value)


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
matched_cases = []

for case in all_cases:
    if case not in r_results:
        differences.append(f"Testcase {case}: missing in R file")
        continue

    if case not in py_results:
        differences.append(f"Testcase {case}: missing in Python file")
        continue

    case_ok = True

    if r_results[case]["adj"] != py_results[case]["adj"]:
        differences.append(f"Testcase {case}: adj mismatch")
        case_ok = False

    if r_results[case]["row_names"] != py_results[case]["row_names"]:
        differences.append(f"Testcase {case}: row.names mismatch")
        case_ok = False

    r_result = normalize_result(r_results[case]["result"])
    py_result = normalize_result(py_results[case]["result"])

    if r_result != py_result:
        differences.append(f"Testcase {case}: result mismatch")
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
    print("\nAll testcase inputs and results are identical.")

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
        f.write("All testcase inputs and results are identical.\n")

print("\nReport saved to", report_file)