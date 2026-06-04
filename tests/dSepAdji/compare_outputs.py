from pathlib import Path

base_dir = Path(__file__).resolve().parent
r_file = base_dir / "R_outputs" / "all_results.txt"
py_file = base_dir / "Py_outputs" / "all_results.txt"


def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    results = {}
    case_num = None
    matrix = []
    i_value = None
    cond_set = []
    reachable_j = []
    reachable_noncausal = []
    mode = None

    for line in lines:
        line = line.strip()

        if line.startswith("Testcase "):
            if case_num is not None:
                results[case_num] = {
                    "matrix": matrix,
                    "i": i_value,
                    "cond_set": cond_set,
                    "reachable_j": reachable_j,
                    "reachable_noncausal": reachable_noncausal,
                }

            case_num = int(line.replace("Testcase ", ""))
            matrix = []
            i_value = None
            cond_set = []
            reachable_j = []
            reachable_noncausal = []
            mode = None

        elif line == "Matrix:":
            mode = "matrix"

        elif line == "i:":
            mode = "i"

        elif line == "condSet:":
            mode = "condSet"

        elif line == "reachableJ:":
            mode = "reachableJ"

        elif line == "reachableOnNonCausalPath:":
            mode = "reachableOnNonCausalPath"

        elif line == "timeComputePM:" or line == "timeComputePM2:":
            mode = "time"
            continue

        elif line == "":
            continue

        else:
            if mode == "matrix":
                matrix.append([int(x) for x in line.split()])
            elif mode == "i":
                i_value = int(line)
            elif mode == "condSet":
                if line == "empty":
                    cond_set = []
                else:
                    cond_set = [int(x) for x in line.split()]
            elif mode == "reachableJ":
                reachable_j = [int(x) for x in line.split()]
            elif mode == "reachableOnNonCausalPath":
                reachable_noncausal = [int(x) for x in line.split()]
            elif mode == "time":
                continue

    if case_num is not None:
        results[case_num] = {
            "matrix": matrix,
            "i": i_value,
            "cond_set": cond_set,
            "reachable_j": reachable_j,
            "reachable_noncausal": reachable_noncausal,
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

    if r_results[case]["matrix"] != py_results[case]["matrix"]:
        differences.append(f"Testcase {case}: input matrix mismatch")
        case_ok = False

    if r_results[case]["i"] != py_results[case]["i"]:
        differences.append(f"Testcase {case}: i mismatch")
        case_ok = False

    if r_results[case]["cond_set"] != py_results[case]["cond_set"]:
        differences.append(f"Testcase {case}: condSet mismatch")
        case_ok = False

    if r_results[case]["reachable_j"] != py_results[case]["reachable_j"]:
        differences.append(f"Testcase {case}: reachableJ mismatch")
        case_ok = False

    if r_results[case]["reachable_noncausal"] != py_results[case]["reachable_noncausal"]:
        differences.append(f"Testcase {case}: reachableOnNonCausalPath mismatch")
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
    print("\nAll testcase inputs and main results are identical.")

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
        f.write("All testcase inputs and main results are identical.\n")

print("\nReport saved to", report_file)