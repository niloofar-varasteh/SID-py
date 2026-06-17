import time
import itertools
from typing import Optional

import numpy as np
import networkx as nx


#todo: sanity check for later
from allDagsJonas import all_dags_jonas
from computePathMatrix import compute_path_matrix
from computePathMatrix2 import compute_path_matrix2
from dSepAdji import d_sep_adji



class StructIntervDistResult:
    """Container for mirroring the ress list variable in R code."""

    def __init__(self, sid, sid_upper_bound, sid_lower_bound, incorrect_mat):
        self.sid = sid
        self.sid_upper_bound = sid_upper_bound
        self.sid_lower_bound = sid_lower_bound
        self.incorrect_mat = incorrect_mat

    def __repr__(self):
        return (
            f"StructIntervDistResult(sid={self.sid}, "
            f"sid_upper_bound={self.sid_upper_bound}, "
            f"sid_lower_bound={self.sid_lower_bound})"
        )

#The actual function for SID
def SID(true_graph, est_graph, output: bool = False, spars: bool = False):
    """
    Compute the Structural Intervention Distance (SID) between a true DAG
    and an estimated graph (DAG, CPDAG, or PDAG).

    Parameters
    ----------
    true_graph : array-like (p x p)
        Adjacency matrix of the true DAG. true_graph[i, j] == 1 means i -> j.
    est_graph : array-like (p x p)
        Adjacency matrix of the estimated graph. Same convention.
    output : bool
        If True, print diagnostic information (mirrors R's `output` flag).
    spars : bool
        If True, use sparse-matrix-aware path computations in the helper
        functions (not entirely sure if there is a need for that in Python).

    Returns
    -------
    StructIntervDistResult
        .sid            : exact SID (only meaningful if the estimated graph
                           could be treated as a single DAG, we should use the upper resp. lower bounds otherwise).
        .sid_upper_bound : upper bound on the SID over all valid DAG
                            extensions of est_graph.
        .sid_lower_bound : lower bound on the SID over all valid DAG
                            extensions of est_graph.
        .incorrect_mat   : (p x p) 0/1 matrix flagging incorrectly predicted
                            interventions (i, j).
    """
    est_graph = np.asarray(est_graph, dtype=float)
    true_graph = np.asarray(true_graph, dtype=float)
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
    ptm_total = time.process_time()

    # Compute the path matrix whose entry (i, j) is True if there is a
    # directed path from i to j. The diagonal is True too.
    path_matrix = compute_path_matrix(true_graph, spars)
    path_matrix = np.asarray(path_matrix)

    # We now compute the undirected graph and all its connected components.
    # The graph does not contain undirected components if it is a DAG.
    gp_undir = est_graph * est_graph.T
    g_undir_nx = nx.from_numpy_array(gp_undir, create_using=nx.Graph)
    # connected components, as 0-indexed node lists (R version is 1-indexed)
    conn_comp = [sorted(c) for c in nx.connected_components(g_undir_nx)]
    num_conn_comp = len(conn_comp)  # == p if (but not only if) est_graph is a DAG

    gp_is_essential_graph = True
    for ll in range(num_conn_comp):
        comp = conn_comp[ll]
        if len(comp) > 1:
            sub = gp_undir[np.ix_(comp, comp)]
            sub_graph = nx.from_numpy_array(sub, create_using=nx.Graph)
            chordal = nx.is_chordal(sub_graph)
            if not chordal:
                print(
                    "The estimated graph is not chordal, i.e. it is not a "
                    "CPDAG! We thus consider local expansions of the graph "
                    "(some combinations of which may lead to cycles)."
                )
                gp_is_essential_graph = False
            if len(comp) > 8:
                print(
                    "The connected component is too large (>8 nodes) in "
                    "order to be extended to all DAGs in a reasonable "
                    "amount of time. We thus consider local expansions of "
                    "the graph (some combinations of which may lead to "
                    "cycles)."
                )
                gp_is_essential_graph = False

    for ll in range(num_conn_comp):
        comp = conn_comp[ll]
        ptm = time.process_time()
        mmm = None
        incorrect_sum = None
        unique_rows = None
        if len(comp) > 0:
            if gp_is_essential_graph:
                # expand the connected component into DAGs
                if len(comp) > 1:
                    mmm = all_dags_jonas(est_graph, [c + 1 for c in comp])  # 1-indexed for helper, if it expects R convention
                else:
                    mmm = est_graph.reshape(1, p ** 2)

                if mmm is not None and np.sum(mmm == -1) == 1:
                    gp_is_essential_graph = False
                    mmm = est_graph.reshape(1, p ** 2)

                if mmm is None:
                    print(
                        "Something is wrong. Maybe the estimated graph is "
                        "not a CPDAG? We expand the undirected components "
                        "locally."
                    )
                    gp_is_essential_graph = False
                else:
                    # each row in mmm contains one DAG, flattened in
                    # row-major order with parents-of-node convention
                    # already matching the column-major "children of node"
                    # convention used below (caller's responsibility, as in
                    # the R original's newInd re-indexing step).
                    incorrect_sum = np.zeros(mmm.shape[0])
        time_exp_graph += time.process_time() - ptm

        for i in comp:  # nodes are 0-indexed Python node ids
            pa_g = np.where(true_graph[:, i] == 1)[0]  # parents of i in true_graph
            certain_pa_gp = np.where((est_graph[:, i] * (1 - est_graph[i, :])) == 1)[0]
            possible_pa_gp = np.where((est_graph[:, i] * est_graph[i, :]) == 1)[0]

            all_parents_of_i = None  # set below when gp_is_essential_graph

            if not gp_is_essential_graph:
                # go through all local combinations of parents; do not care
                # whether this is consistent with a graph structure.
                maxcount = 2 ** len(possible_pa_gp)
                unique_rows = list(range(maxcount))
                flat_est = est_graph.flatten(order="C")
                mmm = np.tile(flat_est, (maxcount, 1))
                if len(possible_pa_gp) > 0:
                    combos = np.array(
                        list(itertools.product([0, 1], repeat=len(possible_pa_gp)))
                    )
                    cols = i * p + possible_pa_gp  # row-major flat index for est_graph[possible_pa_gp, i] entries... see note below
                    mmm[:, cols] = combos
                incorrect_sum = np.zeros(maxcount)
            else:
                if mmm.shape[0] > 1:
                    # each row in mmm contains a different DAG expansion of
                    # the ll-th connected component. However, the parent
                    # sets of node i might be the same for many DAGs.
                    all_parents_of_i = np.array([i + k * p for k in range(p)])
                    cols = mmm[:, all_parents_of_i]
                    _, unique_idx = np.unique(cols, axis=0, return_index=True)
                    unique_rows = sorted(unique_idx.tolist())
                    maxcount = len(unique_rows)
                else:
                    maxcount = 1
                    unique_rows = [0]

            count = 0
            while count < maxcount:
                if maxcount == 1:
                    pa_gp = certain_pa_gp
                else:
                    row = mmm[unique_rows[count], :]
                    gp_new = row.reshape(p, p).T  # undo the row-major flatten
                    pa_gp = np.where(gp_new[:, i] == 1)[0]
                    if output:
                        print(
                            f"{i} has {len(pa_gp)} parents in expansion nr. "
                            f"{unique_rows[count]} of Gp: {pa_gp}"
                        )

                # the following computations are the same for all j (i is fixed)
                ptm = time.process_time()
                path_matrix2 = compute_path_matrix2(true_graph, pa_gp, path_matrix, spars)
                time_path_matrix2 += time.process_time() - ptm

                ptm = time.process_time()
                check_all_d_sep = d_sep_adji(
                    true_graph, i, pa_gp, path_matrix, path_matrix2, spars=spars
                )
                num_checks += 1
                time_all_d_sep += time.process_time() - ptm
                time_all_compute_pm2 += check_all_d_sep["time_compute_pm2"]
                time_all_compute_pm += check_all_d_sep["time_compute_pm"]
                reachable_w_out_causal_path = check_all_d_sep["reachable_on_non_causal_path"]

                def _propagate_incorrect(uniq_rows, count_idx, all_parents_of_i_local):
                    """Mirror the R block that also increments incorrectSum
                    for all other rows sharing the same parent set of i."""
                    incorrect_sum[uniq_rows[count_idx]] += 1
                    all_others = [
                        r for r in range(mmm.shape[0]) if r not in uniq_rows
                    ]
                    if len(all_others) >= 1:
                        target = mmm[uniq_rows[count_idx], all_parents_of_i_local]
                        others = mmm[np.ix_(all_others, all_parents_of_i_local)]
                        matches = np.sum(others == target, axis=1) == p
                        ind_in_all_others = np.where(matches)[0]
                        for idx in ind_in_all_others:
                            incorrect_sum[all_others[idx]] += 1

                for j in range(p):
                    if i == j:
                        continue  # test the intervention effect from i to j

                    # The order of the following checks and the `finished`
                    # flag are made such that as few tests as possible are
                    # performed.
                    finished = False
                    ij_g_null = False
                    ij_gp_null = False

                    # ijGNull means the causal effect from i to j is zero in
                    # G; more precisely, p(x_j | do(x_i=a)) = p(x_j)
                    if path_matrix[i, j] == 0:
                        ij_g_null = True

                    # if j -> i exists in Gp (j is a parent of i)
                    if j in pa_gp:
                        ij_gp_null = True

                    # if both are zero
                    if ij_gp_null and ij_g_null:
                        finished = True
                        correct_int[i, j] = 1

                    # if Gp predicts zero but G says it is not
                    if ij_gp_null and not ij_g_null:
                        incorrect_int[i, j] = 1
                        if gp_is_essential_graph:
                            _propagate_incorrect(unique_rows, count, all_parents_of_i)
                        else:
                            incorrect_sum[unique_rows[count]] += 1
                        finished = True

                    # if the parent sets are the same
                    if not finished and set(pa_g.tolist()) == set(pa_gp.tolist()):
                        finished = True
                        correct_int[i, j] = 1

                    # this part contains the difficult computations
                    if not finished:
                        if path_matrix[i, j] > 0:
                            # which children are part of a causal path?
                            chi_caus_path = np.where(
                                (true_graph[i, :] == 1) & (path_matrix[:, j] > 0)
                            )[0]
                            # check whether pa_gp contains a descendant of a
                            # "proper" child of i
                            if len(chi_caus_path) > 0 and len(pa_gp) > 0:
                                reach = np.sum(path_matrix[np.ix_(chi_caus_path, pa_gp)])
                            else:
                                reach = 0
                            if reach > 0:
                                incorrect_int[i, j] = 1
                                if gp_is_essential_graph:
                                    _propagate_incorrect(unique_rows, count, all_parents_of_i)
                                else:
                                    incorrect_sum[unique_rows[count]] += 1
                                finished = True

                        if not finished:
                            # check whether all non-causal paths are blocked
                            if reachable_w_out_causal_path[j] == 1:
                                incorrect_int[i, j] = 1
                                if gp_is_essential_graph:
                                    _propagate_incorrect(unique_rows, count, all_parents_of_i)
                                else:
                                    incorrect_sum[unique_rows[count]] += 1
                            else:
                                correct_int[i, j] = 1

                count += 1  # while-loop over count <= maxcount

            if not gp_is_essential_graph:
                minimum_total += np.min(incorrect_sum)
                maximum_total += np.max(incorrect_sum)
                incorrect_sum = np.array([0])

            if incorrect_sum is not None and len(incorrect_sum) > 1 and output:
                print(f"For variable {i} we have more than one possible set of parents.")
                print(
                    "The following vector adds the number of incorrect "
                    "interventions for the possible parent sets."
                )
                print(
                    "(the computation is done in a clever way since some "
                    f"of the components may correspond to the same parent "
                    f"set of node {i})."
                )
                print(incorrect_sum)
                print("Only for a new connected component is this vector reset to zero.\n")

        if incorrect_sum is not None:
            minimum_total += np.min(incorrect_sum)
            maximum_total += np.max(incorrect_sum)
        incorrect_sum = np.array([0])

    time_total = time.process_time() - ptm_total

    # The rest is output and return.
    if output and p < 11:
        print("These all are incorrectly predicted interventions:")
        print(incorrect_int)
        print("And these are all correctly predicted interventions:")
        print(correct_int)
      
    result = StructIntervDistResult(
        sid=int(np.sum(incorrect_int)),
        sid_upper_bound=maximum_total,
        sid_lower_bound=minimum_total,
        incorrect_mat=incorrect_int,
    )

    if output:
        print("Time needed for ... ")
        print(f"... expanding the graph: {time_exp_graph}")
        print(f"... computing path matrices (used for checking d-seps): {time_path_matrix2}")
        print(f"... checking d-separations: {time_all_d_sep}")
        print(f"... ... thereof: compute_path_matrix2: {time_all_compute_pm2}")
        print(f"... ... thereof: compute_path_matrix: {time_all_compute_pm}")
        print(f"... in total: {time_total}")
        print(f"number of times we ran *check all d-seps*: {num_checks}")

    return result
