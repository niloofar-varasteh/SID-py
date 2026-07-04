import os
import time
import numpy as np
from computePathMatrix import compute_path_matrix
from computePathMatrix2 import compute_path_matrix2


def d_sep_adji(adj_mat, i, cond_set, path_matrix=None, path_matrix2=None, spars=None):
    adj_mat = np.array(adj_mat, dtype=int).copy()
    p = adj_mat.shape[1]

    if spars is None:
        spars = p > 99

    # llm
    if path_matrix is None:
        path_matrix = compute_path_matrix(adj_mat, spars=spars)

    if path_matrix2 is None:
        path_matrix2 = np.full((p, p), np.nan)

    time_compute_pm2 = 0.0
    time_compute_pm = 0.0

    if np.isnan(path_matrix2).any():
        start = time.time()
        path_matrix2 = compute_path_matrix2(adj_mat.copy(), cond_set, path_matrix, spars=spars)
        time_compute_pm2 += time.time() - start

    if len(cond_set) == 0:
        anc_of_cond_set = np.array([], dtype=int)
    elif len(cond_set) == 1:
        anc_of_cond_set = np.where(path_matrix[:, cond_set[0] - 1] > 0)[0] + 1
    else:
        cols = [x - 1 for x in cond_set]
        anc_of_cond_set = np.where(np.sum(path_matrix[:, cols], axis=1) > 0)[0] + 1

    reachability_matrix = np.zeros((2 * p, 2 * p), dtype=int)
    reachable_on_noncausal_path_later = np.zeros((2, 2), dtype=int)

    reachable_nodes = np.zeros(2 * p, dtype=int)
    reachable_on_noncausal_path = np.zeros(2 * p, dtype=int)
    already_checked = np.zeros(p, dtype=int)

    k = 1
    to_check = [0, 0]

    i_idx = i - 1

    reachable_ch = np.where(adj_mat[i_idx, :] == 1)[0] + 1
    if len(reachable_ch) > 0:
        to_check.extend(reachable_ch.tolist())
        reachable_nodes[reachable_ch - 1] = 1
        adj_mat[i_idx, reachable_ch - 1] = 0

    reachable_pa = np.where(adj_mat[:, i_idx] == 1)[0] + 1
    if len(reachable_pa) > 0:
        to_check.extend(reachable_pa.tolist())
        reachable_nodes[reachable_pa - 1 + p] = 1
        reachable_on_noncausal_path[reachable_pa - 1 + p] = 1
        adj_mat[reachable_pa - 1, i_idx] = 0

    while k < len(to_check) - 1:
        k += 1
        a1 = to_check[k]

        if already_checked[a1 - 1] == 0:
            current_node = a1
            current_idx = current_node - 1
            already_checked[a1 - 1] = 1

            #parents of current node #llm
            pa = np.where(adj_mat[:, current_idx] == 1)[0] + 1

            pa1 = np.array([x for x in pa if x not in cond_set], dtype=int)
            if len(pa1) > 0:
                reachability_matrix[pa1 - 1, current_idx] = 1
                reachability_matrix[pa1 - 1 + p, current_idx] = 1

            if np.any(anc_of_cond_set == current_node):
                if len(pa) > 0:
                    reachability_matrix[current_idx, pa - 1 + p] = 1

                if path_matrix2[i_idx, current_idx] > 0 and len(pa) > 0:
                    add_rows = np.column_stack(
                        (np.full(len(pa), current_node, dtype=int), pa)
                    )
                    reachable_on_noncausal_path_later = np.vstack(
                        (reachable_on_noncausal_path_later, add_rows)
                    )

                new_to_check = [x for x in pa if already_checked[x - 1] == 0]
                to_check.extend(new_to_check)

            if current_node not in cond_set and len(pa) > 0:
                reachability_matrix[current_idx + p, pa - 1 + p] = 1
                new_to_check = [x for x in pa if already_checked[x - 1] == 0]
                to_check.extend(new_to_check)

            #children of current node
            ch = np.where(adj_mat[current_idx, :] == 1)[0] + 1

            ch1 = np.array([x for x in ch if x not in cond_set], dtype=int)
            if len(ch1) > 0:
                reachability_matrix[ch1 - 1 + p, current_idx + p] = 1

            ch2 = np.array([x for x in ch if x in anc_of_cond_set], dtype=int)
            if len(ch2) > 0:
                reachability_matrix[ch2 - 1, current_idx + p] = 1

            reachable_from_i = np.where(path_matrix2[i_idx, :] > 0)[0] + 1
            ch2b = np.array([x for x in ch2 if x in reachable_from_i], dtype=int)
            if len(ch2b) > 0:
                add_rows = np.column_stack(
                    (ch2b, np.full(len(ch2b), current_node, dtype=int))
                )
                reachable_on_noncausal_path_later = np.vstack(
                    (reachable_on_noncausal_path_later, add_rows)
                )

            if current_node not in cond_set and len(ch) > 0:
                reachability_matrix[current_idx, ch - 1] = 1
                reachability_matrix[current_idx + p, ch - 1] = 1
                new_to_check = [x for x in ch if already_checked[x - 1] == 0]
                to_check.extend(new_to_check)

    start = time.time()
    reachability_matrix = compute_path_matrix(reachability_matrix, spars=spars)
    time_compute_pm += time.time() - start
    reachability_matrix = np.array(reachability_matrix, dtype=int)

    ttt2 = np.where(reachable_nodes == 1)[0]
    if len(ttt2) == 1:
        tt2 = np.where(reachability_matrix[ttt2[0], :] > 0)[0]
    elif len(ttt2) > 1:
        tt2 = np.where(np.sum(reachability_matrix[ttt2, :], axis=0) > 0)[0]
    else:
        tt2 = np.array([], dtype=int)
    reachable_nodes[tt2] = 1

    ttt = np.where(reachable_on_noncausal_path == 1)[0]
    if len(ttt) == 1:
        tt = np.where(reachability_matrix[ttt[0], :] > 0)[0]
    elif len(ttt) > 1:
        tt = np.where(np.sum(reachability_matrix[ttt, :], axis=0) > 0)[0]
    else:
        tt = np.array([], dtype=int)
    reachable_on_noncausal_path[tt] = 1

    if reachable_on_noncausal_path_later.shape[0] > 2:
        for kk in range(2, reachable_on_noncausal_path_later.shape[0]):
            reachable_through = reachable_on_noncausal_path_later[kk, 0]
            new_reachable = reachable_on_noncausal_path_later[kk, 1]

            reachable_on_noncausal_path[new_reachable - 1 + p] = 1

            rt = reachable_through - 1
            nr = new_reachable - 1

            reachability_matrix[nr, rt] = 0
            reachability_matrix[nr, rt + p] = 0
            reachability_matrix[nr + p, rt] = 0
            reachability_matrix[nr + p, rt + p] = 0

        ttt = np.where(reachable_on_noncausal_path == 1)[0]
        if len(ttt) == 1:
            tt = np.where(reachability_matrix[ttt[0], :] > 0)[0]
        elif len(ttt) > 1:
            tt = np.where(np.sum(reachability_matrix[ttt, :], axis=0) > 0)[0]
        else:
            tt = np.array([], dtype=int)
        reachable_on_noncausal_path[tt] = 1

    reachable_j = (
        np.column_stack((reachable_nodes[:p], reachable_nodes[p:2 * p])).sum(axis=1) > 0
    )
    reachable_on_noncausal = (
        np.column_stack((reachable_on_noncausal_path[:p], reachable_on_noncausal_path[p:2 * p])).sum(axis=1) > 0
    )

    result = {
        "timeComputePM": time_compute_pm,
        "timeComputePM2": time_compute_pm2,
        "reachableJ": reachable_j.astype(int),
        "reachableOnNonCausalPath": reachable_on_noncausal.astype(int),
    }

    return result


def read_testcase(filepath):
    adj_mat = []
    cond_set = []
    i = None
    mode = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line == "":
                continue

            if line == "Matrix:":
                mode = "matrix"
                continue

            if line == "i:":
                mode = "i"
                continue

            if line == "condSet:":
                mode = "condSet"
                continue

            if mode == "matrix":
                adj_mat.append([int(x) for x in line.split()])
            elif mode == "i":
                i = int(line)
            elif mode == "condSet":
                if line != "empty":
                    cond_set = [int(x) for x in line.split()]

    return np.array(adj_mat, dtype=int), i, cond_set


#llm

script_dir = os.getcwd()
project_dir = os.path.dirname(script_dir)

testcase_dir = os.path.join(project_dir, "tests", "dSepAdji", "Testcase")
output_dir = os.path.join(project_dir, "tests", "dSepAdji", "Py_outputs")

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

output_file = os.path.join(output_dir, "all_results.txt")

print("Reading testcases from:", testcase_dir)
print("Saving output to:", output_file)

with open(output_file, "w", encoding="utf-8") as f:
    for case_index in range(1, 10001):
        filepath = os.path.join(testcase_dir, f"{case_index}.txt")
        adj_mat, i, cond_set = read_testcase(filepath)

        result = d_sep_adji(adj_mat, i, cond_set)

        f.write(f"Testcase {case_index}\n")

        f.write("Matrix:\n")
        for row in adj_mat:
            f.write(" ".join(map(str, row)) + "\n")

        f.write("i:\n")
        f.write(str(i) + "\n")

        f.write("condSet:\n")
        if len(cond_set) == 0:
            f.write("empty\n")
        else:
            f.write(" ".join(map(str, cond_set)) + "\n")

        f.write("reachableJ:\n")
        f.write(" ".join(map(str, result["reachableJ"])) + "\n")

        f.write("reachableOnNonCausalPath:\n")
        f.write(" ".join(map(str, result["reachableOnNonCausalPath"])) + "\n")

        f.write("timeComputePM:\n")
        f.write(str(result["timeComputePM"]) + "\n")

        f.write("timeComputePM2:\n")
        f.write(str(result["timeComputePM2"]) + "\n")

        f.write("\n")

print("Done. Output saved to:", output_file)