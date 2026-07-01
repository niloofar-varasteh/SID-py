
import os
import time
import itertools
import numpy as np
import networkx as nx

from computePathMatrix import compute_path_matrix
from computePathMatrix2 import compute_path_matrix2
from allDagsJonas import all_dags_jonas
from dSepAdji import d_sep_adji
#llm

def _same_parent_columns(i, p):
    return (i - 1) + np.arange(p) * p


def _add_incorrect_to_duplicate_parent_sets(incorrect_sum, mmm, current_row_idx, unique_rows, parent_cols):
    all_rows = np.arange(mmm.shape[0])
    unique_row_set = set(unique_rows)
    current_parent_pattern = mmm[current_row_idx, parent_cols]

    for row_idx in all_rows:
        if row_idx in unique_row_set:
            continue
        if np.array_equal(current_parent_pattern, mmm[row_idx, parent_cols]):
            incorrect_sum[row_idx] += 1


def struct_interv_dist(true_graph, est_graph, output=False, spars=False):
    est_graph = np.array(est_graph, dtype=int)
    true_graph = np.array(true_graph, dtype=int)

    p = true_graph.shape[1]

    incorrect_int = np.zeros((p, p), dtype=int)
    correct_int = np.zeros((p, p), dtype=int)

    minimum_total = 0
    maximum_total = 0

    time_path_matrix2 = 0.0
    time_all_compute_pm2 = 0.0
    time_all_compute_pm = 0.0
    time_all_d_sep = 0.0
    time_exp_graph = 0.0

    num_checks = 0
    gtmp = np.eye(p, dtype=int)
    ptm_total = time.time()

    path_matrix = np.array(compute_path_matrix(true_graph, spars=spars), dtype=int)

    gp_undir = est_graph * est_graph.T
    undir_graph = nx.from_numpy_array((gp_undir > 0).astype(int))

    conn_comp = [
        np.array(sorted([node + 1 for node in comp]), dtype=int)
        for comp in nx.connected_components(undir_graph)
    ]
    num_conn_comp = len(conn_comp)

    gp_is_essential_graph = True

    for ll in range(num_conn_comp):
        comp = conn_comp[ll]

        if len(comp) > 1:
            sub = gp_undir[np.ix_(comp - 1, comp - 1)]
            sub_graph = nx.from_numpy_array((sub > 0).astype(int))
            chordal = nx.is_chordal(sub_graph)

            if not chordal:
                print(
                    "The estimated graph is not chordal, i.e. it is not a CPDAG! "
                    "We thus consider local expansions of the graph "
                    "(some combinations of which may lead to cycles)."
                )
                gp_is_essential_graph = False

            if len(comp) > 8:
                print(
                    "The connected component is too large (>8 nodes) in order to be "
                    "extended to all DAGs in a reasonable amount of time. "
                    "We thus consider local expansions of the graph "
                    "(some combinations of which may lead to cycles)."
                )
                gp_is_essential_graph = False

    for ll in range(num_conn_comp):
        ptm = time.time()
        comp = conn_comp[ll]

        if len(comp) > 0:
            if gp_is_essential_graph:
                if len(comp) > 1:
                    mmm = all_dags_jonas(est_graph, comp)
                    mmm = np.array(mmm, dtype=int)
                    if mmm.ndim == 1:
                        mmm = mmm.reshape(1, -1)
                else:
                    mmm = est_graph.flatten(order="F").reshape(1, p**2)

                if np.sum(mmm == -1) == 1:
                    gp_is_essential_graph = False
                    mmm = est_graph.flatten(order="F").reshape(1, p**2)

                # Convert each DAG row to the same row-wise layout used later in the checks
                mmm = np.array(
                    [row.reshape((p, p), order="F").flatten(order="C") for row in mmm],
                    dtype=int
                )

                if mmm is None:
                    print(
                        "Something is wrong. Maybe the estimated graph is not a CPDAG? "
                        "We expand the undirected components locally."
                    )
                    gp_is_essential_graph = False
                else:
                    incorrect_sum = np.zeros(mmm.shape[0], dtype=int)

        time_exp_graph += time.time() - ptm

        for i in comp:
            pa_g = np.where(true_graph[:, i - 1] == 1)[0] + 1
            certain_pa_gp = np.where(
                (est_graph[:, i - 1] * (np.ones(p, dtype=int) - est_graph[i - 1, :])) == 1
            )[0] + 1
            possible_pa_gp = np.where(
                (est_graph[:, i - 1] * est_graph[i - 1, :]) == 1
            )[0] + 1

            all_parents_of_i = _same_parent_columns(i, p)

            if not gp_is_essential_graph:
                maxcount = 2 ** len(possible_pa_gp)
                unique_rows = np.arange(maxcount, dtype=int)

                base_row = est_graph.flatten(order="C")
                mmm = np.tile(base_row, (maxcount, 1))

                if len(possible_pa_gp) > 0:
                    combinations = np.array(
                        list(itertools.product([0, 1], repeat=len(possible_pa_gp))),
                        dtype=int
                    )
                    target_cols = (i - 1) + (possible_pa_gp - 1) * p
                    mmm[:, target_cols] = combinations

                incorrect_sum = np.zeros(maxcount, dtype=int)
            else:
                if mmm.shape[0] > 1:
                    seen = set()
                    unique_rows = []
                    for row_idx in range(mmm.shape[0]):
                        key = tuple(mmm[row_idx, all_parents_of_i].tolist())
                        if key not in seen:
                            seen.add(key)
                            unique_rows.append(row_idx)
                    unique_rows = np.array(unique_rows, dtype=int)
                    maxcount = len(unique_rows)
                else:
                    maxcount = 1
                    unique_rows = np.array([0], dtype=int)

            count = 0
            while count < maxcount:
                current_row_idx = unique_rows[count]

                if maxcount == 1:
                    pa_gp = certain_pa_gp.copy()
                else:
                    gp_new = mmm[current_row_idx, :].reshape((p, p), order="C")
                    pa_gp = np.where(gp_new[:, i - 1] == 1)[0] + 1

                    if output:
                        print(
                            i,
                            " has ",
                            len(pa_gp),
                            " parents in expansion nr. ",
                            current_row_idx + 1,
                            " of Gp:",
                            sep=""
                        )
                        print(pa_gp)

                ptm = time.time()
                path_matrix2 = compute_path_matrix2(
                    true_graph,
                    pa_gp.tolist(),
                    path_matrix,
                    spars=spars
                )
                time_path_matrix2 += time.time() - ptm

                ptm = time.time()
                check_all_d_sep = d_sep_adji(
                    true_graph,
                    i,
                    pa_gp.tolist(),
                    path_matrix=path_matrix,
                    path_matrix2=path_matrix2,
                    spars=spars
                )
                num_checks += 1
                time_all_d_sep += time.time() - ptm
                time_all_compute_pm2 += check_all_d_sep["timeComputePM2"]
                time_all_compute_pm += check_all_d_sep["timeComputePM"]

                reachable_without_causal_path = np.array(
                    check_all_d_sep["reachableOnNonCausalPath"],
                    dtype=int
                )

                for j in range(1, p + 1):
                    if i != j:
                        finished = False
                        ij_g_null = False
                        ij_gp_null = False

                        if path_matrix[i - 1, j - 1] == 0:
                            ij_g_null = True

                        if np.sum(pa_gp == j) == 1:
                            ij_gp_null = True

                        if ij_gp_null and ij_g_null:
                            finished = True
                            correct_int[i - 1, j - 1] = 1

                        if ij_gp_null and not ij_g_null:
                            incorrect_int[i - 1, j - 1] = 1
                            incorrect_sum[current_row_idx] += 1
                            _add_incorrect_to_duplicate_parent_sets(
                                incorrect_sum, mmm, current_row_idx, unique_rows, all_parents_of_i
                            )
                            finished = True

                        if (not finished) and set(pa_g.tolist()) == set(pa_gp.tolist()):
                            finished = True
                            correct_int[i - 1, j - 1] = 1

                        if not finished:
                            if path_matrix[i - 1, j - 1] > 0:
                                chi_caus_path = np.where(
                                    (true_graph[i - 1, :] > 0) & (path_matrix[:, j - 1] > 0)
                                )[0] + 1

                                if len(chi_caus_path) > 0 and len(pa_gp) > 0:
                                    if np.sum(path_matrix[np.ix_(chi_caus_path - 1, pa_gp - 1)]) > 0:
                                        incorrect_int[i - 1, j - 1] = 1
                                        incorrect_sum[current_row_idx] += 1
                                        _add_incorrect_to_duplicate_parent_sets(
                                            incorrect_sum, mmm, current_row_idx, unique_rows, all_parents_of_i
                                        )
                                        finished = True

                            if not finished:
                                if reachable_without_causal_path[j - 1] == 1:
                                    incorrect_int[i - 1, j - 1] = 1
                                    incorrect_sum[current_row_idx] += 1
                                    _add_incorrect_to_duplicate_parent_sets(
                                        incorrect_sum, mmm, current_row_idx, unique_rows, all_parents_of_i
                                    )
                                else:
                                    correct_int[i - 1, j - 1] = 1

                count += 1

            if not gp_is_essential_graph:
                minimum_total += int(np.min(incorrect_sum))
                maximum_total += int(np.max(incorrect_sum))
                incorrect_sum = np.array([0], dtype=int)

            if len(incorrect_sum) > 1 and output:
                print(f"For variable {i} we have more than one possible set of parents.")
                print("The following vector is adding the number of incorrect interventions for the possible parent sets.")
                print(
                    f"(the computation is done in a clever way since some of the components "
                    f"may correspond to the same parent set of node {i})."
                )
                print(incorrect_sum)
                print("Only for a new connected component this vector is set to zero again.\n")

        minimum_total += int(np.min(incorrect_sum))
        maximum_total += int(np.max(incorrect_sum))
        incorrect_sum = np.array([0], dtype=int)

    time_total = time.time() - ptm_total

    if output and p < 11:
        print("These all are incorrectly predicted interventions:")
        print(incorrect_int)
        print("And these are all correctly predicted interventions:")
        print(correct_int)

    result = {
        "sid": int(np.sum(incorrect_int)),
        "sidUpperBound": int(maximum_total),
        "sidLowerBound": int(minimum_total),
        "incorrectMat": incorrect_int
    }

    if output:
        print("Time needed for ...")
        print("... expanding the graph:", time_exp_graph)
        print("... computing path matrices (used for checking d-seps):", time_path_matrix2)
        print("... checking d-separations:", time_all_d_sep)
        print("... ... hereof: computePathMatrix2:", time_all_compute_pm2)
        print("... ... hereof: computePathMatrix:", time_all_compute_pm)
        print("... in total:", time_total)
        print("number of times we ran *check all d-seps*:", num_checks)

    return result


def read_testcase(filepath):
    true_graph = []
    est_graph = []
    output = False
    spars = False
    mode = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line == "":
                continue

            if line == "trueGraph:":
                mode = "trueGraph"
                continue

            if line == "estGraph:":
                mode = "estGraph"
                continue

            if line == "output:":
                mode = "output"
                continue

            if line == "spars:":
                mode = "spars"
                continue

            if mode == "trueGraph":
                true_graph.append([int(x) for x in line.split()])
            elif mode == "estGraph":
                est_graph.append([int(x) for x in line.split()])
            elif mode == "output":
                output = line.lower() == "true"
            elif mode == "spars":
                spars = line.lower() == "true"

    return (
        np.array(true_graph, dtype=int),
        np.array(est_graph, dtype=int),
        output,
        spars
    )


script_dir = os.getcwd()
project_dir = os.path.dirname(script_dir)

testcase_dir = os.path.join(project_dir, "tests", "structIntervDist", "Testcase")
output_dir = os.path.join(project_dir, "tests", "structIntervDist", "Py_outputs")

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

output_file = os.path.join(output_dir, "all_results.txt")

print("Reading testcases from:", testcase_dir)
print("Saving output to:", output_file)

with open(output_file, "w", encoding="utf-8") as f:
    for i in range(1, 101):
        filepath = os.path.join(testcase_dir, f"{i}.txt")
        true_graph, est_graph, output_flag, spars_flag = read_testcase(filepath)

        result = struct_interv_dist(
            true_graph,
            est_graph,
            output=output_flag,
            spars=spars_flag
        )

        f.write(f"Testcase {i}\n")

        f.write("trueGraph:\n")
        for row in true_graph:
            f.write(" ".join(map(str, row)) + "\n")

        f.write("estGraph:\n")
        for row in est_graph:
            f.write(" ".join(map(str, row)) + "\n")

        f.write("output:\n")
        f.write(str(output_flag) + "\n")

        f.write("spars:\n")
        f.write(str(spars_flag) + "\n")

        f.write("sid:\n")
        f.write(str(result["sid"]) + "\n")

        f.write("sidUpperBound:\n")
        f.write(str(result["sidUpperBound"]) + "\n")

        f.write("sidLowerBound:\n")
        f.write(str(result["sidLowerBound"]) + "\n")

        f.write("incorrectMat:\n")
        for row in result["incorrectMat"]:
            f.write(" ".join(map(str, row)) + "\n")

        f.write("\n")

print("Done. Output saved to:", output_file)

