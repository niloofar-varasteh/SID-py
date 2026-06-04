from pathlib import Path

base_dir = Path(__file__).resolve().parent
r_file = base_dir / "R_outputs" / "all_results.txt"
py_file = base_dir / "Py_outputs" / "all_results.txt"

def normalize_result(value):
    try:
        num = float(value)
        if num.is_integer():
            return int(num)
        return num
    except ValueError:
        return value


def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    results = {}
    case_num = None
    matrix1 = []
    matrix2 = []
    result = None
    mode = None

    for line in lines:
        line = line.strip()

        if line.startswith("Testcase "):
            if case_num is not None:
                results[case_num] = {
                    "matrix1": matrix1,
                    "matrix2": matrix2,
                    "result": result
                }

            case_num = int(line.replace("Testcase ", ""))
            matrix1 = []
            matrix2 = []
            result = None
            mode = None

        elif line == "Matrix1:":
            mode = "matrix1"

        elif line == "Matrix2:":
            mode = "matrix2"

        elif line == "Result:":
            mode = "result"

        elif line == "":
            continue

        else:
            if mode == "matrix1":
                matrix1.append([int(x) for x in line.split()])
            elif mode == "matrix2":
                matrix2.append([int(x) for x in line.split()])
            elif mode == "result":
                result = normalize_result(line)

    if case_num is not None:
        results[case_num] = {
            "matrix1": matrix1,
            "matrix2": matrix2,
            "result": result
        }

    return results


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

    if r_results[case]["matrix1"] != py_results[case]["matrix1"]:
        differences.append(f"Testcase {case}: Matrix1 mismatch")
        case_ok = False

    if r_results[case]["matrix2"] != py_results[case]["matrix2"]:
        differences.append(f"Testcase {case}: Matrix2 mismatch")
        case_ok = False

    if r_results[case]["result"] != py_results[case]["result"]:
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