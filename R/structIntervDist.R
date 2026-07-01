#llm
library(methods)
library(igraph)

source("computePathMatrix.R")
source("computePathMatrix2.R")
source("allDagsJonas.R")
source("dSepAdji.R")

structIntervDist <- function(trueGraph, estGraph, output = FALSE, spars = FALSE)
    # Copyright (c) 2013 - 2013  Jonas Peters  [peters@stat.math.ethz.ch]
    # All rights reserved.  See the file COPYING for license terms.
{
    # These are all initializations
    estGraph <- as(estGraph, "matrix")
    trueGraph <- as(trueGraph, "matrix")
    p <- dim(trueGraph)[2]
    rownames(estGraph) <- 1:p
    colnames(estGraph) <- 1:p

    incorrectInt <- matrix(0, p, p)
    correctInt <- matrix(0, p, p)
    minimumTotal <- 0
    maximumTotal <- 0
    timePathMatrix2 <- 0
    timeAllComputePM2 <- 0
    timeAllComputePM <- 0
    timeAlldSep <- 0
    timeExpGraph <- 0
    numChecks <- 0
    Gtmp <- diag(1, p)
    ptmtotal <- proc.time()

    # Compute the path matrix whose entry (i,j) is TRUE if there is a directed path
    # from i to j. The diagonal is TRUE, too.
    PathMatrix <- computePathMatrix(trueGraph, spars)
    PathMatrix <- as(PathMatrix, "matrix")

    # Undirected part of estimated graph
    Gp.undir <- estGraph * t(estGraph)

    # Build undirected graph directly from adjacency matrix
    g.undir <- igraph::graph_from_adjacency_matrix(
        (Gp.undir > 0) * 1,
        mode = "undirected",
        diag = FALSE
    )

    comp.obj <- igraph::components(g.undir)
    conn.comp <- split(seq_len(p), comp.obj$membership)
    numConnComp <- length(conn.comp)

    GpIsEssentialGraph <- TRUE

    for (ll in 1:numConnComp)
    {
        conn.comp[[ll]] <- as.numeric(conn.comp[[ll]])

        if (length(conn.comp[[ll]]) > 1)
        {
            sub_undir <- (Gp.undir[conn.comp[[ll]], conn.comp[[ll]]] > 0) * 1

            sub_graph <- igraph::graph_from_adjacency_matrix(
                sub_undir,
                mode = "undirected",
                diag = FALSE
            )

            chordal <- igraph::is.chordal(sub_graph, fillin = TRUE)$chordal

            if (!chordal)
            {
                show("The estimated graph is not chordal, i.e. it is not a CPDAG! We thus consider local expansions of the graph (some combinations of which may lead to cycles).")
                GpIsEssentialGraph <- FALSE
            }

            if (length(conn.comp[[ll]]) > 8)
            {
                show("The connected component is too large (>8 nodes) in order to be extended to all DAGs in a reasonable amount of time. We thus consider local expansions of the graph (some combinations of which may lead to cycles).")
                GpIsEssentialGraph <- FALSE
            }
        }
    }

    for (ll in 1:numConnComp)
    {
        ptm <- proc.time()

        if (length(conn.comp[[ll]]) > 0)
        {
            if (GpIsEssentialGraph)
            {
                # expand the connected component into DAGs
                if (length(conn.comp[[ll]]) > 1)
                {
                    mmm <- allDagsJonas(estGraph, conn.comp[[ll]])
                }
                else
                {
                    mmm <- matrix(estGraph, 1, p^2)
                }

                if (sum(mmm == -1) == 1)
                {
                    GpIsEssentialGraph <- FALSE
                    mmm <- matrix(estGraph, 1, p^2)
                }

                # each row in mmm contains one DAG.
                # the first p entries of estGraph are the parents of node 1
                # in the following lines we change this to: the first p entries of estGraph are the children of node 1
                newInd <- seq(1, p^3, by = p) -
                    rep(seq(0, (p - 1) * (p^2 - 1), by = (p^2 - 1)), each = p)

                dimM <- dim(mmm)
                mmm <- matrix(mmm[, newInd], dimM)

                if (is.null(mmm))
                {
                    show("Something is wrong. Maybe the estimated graph is not a CPDAG? We expand the undirected components locally.")
                    GpIsEssentialGraph <- FALSE
                }
                else
                {
                    incorrectSum <- rep(0, dim(mmm)[1])
                }
            }
        }

        timeExpGraph <- timeExpGraph + proc.time() - ptm

        for (i in conn.comp[[ll]])
        {
            paG <- which(trueGraph[, i] == 1)
            certainpaGp <- which((estGraph[, i] * (rep(1, p) - estGraph[i, ])) == 1)
            possiblepaGp <- which((estGraph[, i] * estGraph[i, ]) == 1)

            if (!GpIsEssentialGraph)
            {
                maxcount <- 2^length(possiblepaGp)
                uniqueRows <- 1:maxcount
                mmm <- t(matrix(rep(t(estGraph)[1:length(estGraph)], maxcount), length(estGraph), maxcount))
                mmm[, i + (possiblepaGp - 1) * p] <- as.matrix(
                    expand.grid(rep(list(0:1), length(possiblepaGp)))
                )
                incorrectSum <- rep(0, maxcount)
            }
            else
            {
                if (dim(mmm)[1] > 1)
                {
                    allParentsOfI <- seq(i, (p - 1) * p + i, by = p)
                    uniqueRows <- which(!duplicated(mmm[, allParentsOfI]))
                    maxcount <- length(uniqueRows)
                }
                else
                {
                    maxcount <- 1
                    uniqueRows <- 1
                }
            }

            count <- 1

            while (count <= maxcount)
            {
                if (maxcount == 1)
                {
                    paGp <- certainpaGp
                }
                else
                {
                    Gpnew <- t(matrix(mmm[uniqueRows[count], ], p, p))
                    paGp <- which(Gpnew[, i] == 1)

                    if (output)
                    {
                        cat(i, " has ", length(paGp), " parents in expansion nr. ", uniqueRows[count], " of Gp:")
                        show(paGp)
                    }
                }

                # the following computations are the same for all j (i is fixed)
                ptm <- proc.time()
                PathMatrix2 <- computePathMatrix2(trueGraph, paGp, PathMatrix, spars)
                timePathMatrix2 <- timePathMatrix2 + proc.time() - ptm

                ptm <- proc.time()
                checkAlldSep <- dSepAdji(trueGraph, i, paGp, PathMatrix, PathMatrix2, spars = spars)
                numChecks <- numChecks + 1
                timeAlldSep <- timeAlldSep + proc.time() - ptm
                timeAllComputePM2 <- timeAllComputePM2 + checkAlldSep$timeComputePM2
                timeAllComputePM <- timeAllComputePM + checkAlldSep$timeComputePM
                reachableWOutCausalPath <- checkAlldSep$reachableOnNonCausalPath

                for (j in 1:p)
                {
                    if (i != j)
                    {
                        # The order of the following checks and the flag finished are
                        # made such that as few tests are performed as possible.

                        finished <- FALSE
                        ijGNull <- FALSE
                        ijGpNull <- FALSE

                        # ijGNull means that the causal effect from i to j is zero in G
                        # more precisely, p(x_j | do (x_i=a)) = p(x_j)
                        if (PathMatrix[i, j] == 0)
                        {
                            ijGNull <- TRUE
                        }

                        # if j->i exists in Gp
                        if ((sum(paGp == j) == 1))
                        {
                            ijGpNull <- TRUE
                        }

                        # if both are zero
                        if (ijGpNull & ijGNull)
                        {
                            finished <- TRUE
                            correctInt[i, j] <- 1
                        }

                        # if Gp predicts zero but G says it is not
                        if (ijGpNull & !ijGNull)
                        {
                            incorrectInt[i, j] <- 1
                            incorrectSum[uniqueRows[count]] <- incorrectSum[uniqueRows[count]] + 1

                            ###############
                            # also add one to all entries of incorrectSum that have the same set of parents of i - find them
                            allOthers <- setdiff(1:(dim(mmm)[1]), uniqueRows)

                            if (length(allOthers) > 1)
                            {
                                indInAllOthers <- which(
                                    colSums(!xor(mmm[uniqueRows[count], allParentsOfI], t(mmm[allOthers, allParentsOfI]))) == p
                                )
                                if (length(indInAllOthers) > 0)
                                {
                                    incorrectSum[allOthers[indInAllOthers]] <- incorrectSum[allOthers[indInAllOthers]] + rep(1, length(indInAllOthers))
                                }
                            }

                            if (length(allOthers) == 1)
                            {
                                indInAllOthers <- which(
                                    sum(!xor(mmm[uniqueRows[count], allParentsOfI], t(mmm[allOthers, allParentsOfI]))) == p
                                )
                                if (length(indInAllOthers) > 0)
                                {
                                    incorrectSum[allOthers[indInAllOthers]] <- incorrectSum[allOthers[indInAllOthers]] + rep(1, length(indInAllOthers))
                                }
                            }
                            ###############

                            finished <- TRUE
                        }

                        # if the set of parents are the same
                        if (!finished && setequal(paG, paGp))
                        {
                            finished <- TRUE
                            correctInt[i, j] <- 1
                        }

                        # this part contains the difficult computations
                        if (!finished)
                        {
                            if (PathMatrix[i, j] > 0)
                            {
                                # which children are part of a causal path?
                                chiCausPath <- which(trueGraph[i, ] & PathMatrix[, j])

                                # check whether in paGp there is a descendant of a "proper" child of i
                                if (sum(PathMatrix[chiCausPath, paGp]) > 0)
                                {
                                    incorrectInt[i, j] <- 1
                                    incorrectSum[uniqueRows[count]] <- incorrectSum[uniqueRows[count]] + 1

                                    ###############
                                    allOthers <- setdiff(1:(dim(mmm)[1]), uniqueRows)

                                    if (length(allOthers) > 1)
                                    {
                                        indInAllOthers <- which(
                                            colSums(!xor(mmm[uniqueRows[count], allParentsOfI], t(mmm[allOthers, allParentsOfI]))) == p
                                        )
                                        if (length(indInAllOthers) > 0)
                                        {
                                            incorrectSum[allOthers[indInAllOthers]] <- incorrectSum[allOthers[indInAllOthers]] + rep(1, length(indInAllOthers))
                                        }
                                    }

                                    if (length(allOthers) == 1)
                                    {
                                        indInAllOthers <- which(
                                            sum(!xor(mmm[uniqueRows[count], allParentsOfI], t(mmm[allOthers, allParentsOfI]))) == p
                                        )
                                        if (length(indInAllOthers) > 0)
                                        {
                                            incorrectSum[allOthers[indInAllOthers]] <- incorrectSum[allOthers[indInAllOthers]] + rep(1, length(indInAllOthers))
                                        }
                                    }
                                    ###############

                                    finished <- TRUE
                                }
                            }

                            if (!finished)
                            {
                                # check whether all non-causal paths are blocked
                                if (reachableWOutCausalPath[j] == 1)
                                {
                                    incorrectInt[i, j] <- 1
                                    incorrectSum[uniqueRows[count]] <- incorrectSum[uniqueRows[count]] + 1

                                    ###############
                                    allOthers <- setdiff(1:(dim(mmm)[1]), uniqueRows)

                                    if (length(allOthers) > 1)
                                    {
                                        indInAllOthers <- which(
                                            colSums(!xor(mmm[uniqueRows[count], allParentsOfI], t(mmm[allOthers, allParentsOfI]))) == p
                                        )
                                        if (length(indInAllOthers) > 0)
                                        {
                                            incorrectSum[allOthers[indInAllOthers]] <- incorrectSum[allOthers[indInAllOthers]] + rep(1, length(indInAllOthers))
                                        }
                                    }

                                    if (length(allOthers) == 1)
                                    {
                                        indInAllOthers <- which(
                                            sum(!xor(mmm[uniqueRows[count], allParentsOfI], t(mmm[allOthers, allParentsOfI]))) == p
                                        )
                                        if (length(indInAllOthers) > 0)
                                        {
                                            incorrectSum[allOthers[indInAllOthers]] <- incorrectSum[allOthers[indInAllOthers]] + rep(1, length(indInAllOthers))
                                        }
                                    }
                                    ###############
                                }
                                else
                                {
                                    correctInt[i, j] <- 1
                                }
                            }
                        }
                    }
                } # for-loop over j

                count <- count + 1
            } # while-loop count<=maxcount

            if (!GpIsEssentialGraph)
            {
                minimumTotal <- minimumTotal + min(incorrectSum)
                maximumTotal <- maximumTotal + max(incorrectSum)
                incorrectSum <- 0
            }

            if (length(incorrectSum) > 1 & output)
            {
                cat("For variable ", i, "we have more than one possible set of parents.\n")
                cat("The following vector is adding the number of incorrect interventions for the possible parent sets.\n")
                cat("(the computation is done in a clever way since some of the components may correspond to the same parent set of node ", i, ").\n")
                show(incorrectSum)
                cat("Only for a new connected component this vector is set to zero again.\n\n")
            }
        } # i in conn.comp

        minimumTotal <- minimumTotal + min(incorrectSum)
        maximumTotal <- maximumTotal + max(incorrectSum)
        incorrectSum <- 0
    }

    timeTotal <- proc.time() - ptmtotal

    # The rest is output and return.
    if (output && p < 11)
    {
        show("These all are incorrectly predicted interventions:")
        show(incorrectInt)
        show("And these are all correctly predicted interventions:")
        show(correctInt)
    }

    ress <- list()
    ress$sid <- sum(incorrectInt)
    ress$sidUpperBound <- maximumTotal
    ress$sidLowerBound <- minimumTotal
    ress$incorrectMat <- incorrectInt

    if (output)
    {
        cat("Time needed for ... \n")
        cat("... expanding the graph: ", timeExpGraph[3], "\n")
        cat("... computing path matrices (used for checking d-seps): ", timePathMatrix2[3], "\n")
        cat("... checking d-separations: ", timeAlldSep[3], "\n")
        cat("... ... herof: computePathMatrix2: ", timeAllComputePM2, "\n")
        cat("... ... herof: computePathMatrix: ", timeAllComputePM, "\n")
        cat("... in total: ", timeTotal[3], "\n")
        cat("number of times we ran *check all d-seps*:", numChecks, "\n")
    }

    return(ress)
}


read_testcase <- function(filepath)
{
    lines <- readLines(filepath, warn = FALSE)

    trueGraph <- list()
    estGraph <- list()
    output <- FALSE
    spars <- FALSE
    mode <- NULL

    for (line in lines)
    {
        line <- trimws(line)

        if (line == "")
        {
            next
        }

        if (line == "trueGraph:")
        {
            mode <- "trueGraph"
            next
        }

        if (line == "estGraph:")
        {
            mode <- "estGraph"
            next
        }

        if (line == "output:")
        {
            mode <- "output"
            next
        }

        if (line == "spars:")
        {
            mode <- "spars"
            next
        }

        if (mode == "trueGraph")
        {
            trueGraph[[length(trueGraph) + 1]] <- as.integer(strsplit(line, " +")[[1]])
        }
        else if (mode == "estGraph")
        {
            estGraph[[length(estGraph) + 1]] <- as.integer(strsplit(line, " +")[[1]])
        }
        else if (mode == "output")
        {
            output <- tolower(line) == "true"
        }
        else if (mode == "spars")
        {
            spars <- tolower(line) == "true"
        }
    }

    trueGraph <- do.call(rbind, trueGraph)
    estGraph <- do.call(rbind, estGraph)

    return(list(
        trueGraph = trueGraph,
        estGraph = estGraph,
        output = output,
        spars = spars
    ))
}


script_dir <- getwd()
project_dir <- dirname(script_dir)

testcase_dir <- file.path(project_dir, "tests", "structIntervDist", "Testcase")
output_dir <- file.path(project_dir, "tests", "structIntervDist", "R_outputs")

if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
}

output_file <- file.path(output_dir, "all_results.txt")
out <- file(output_file, open = "w")

for (i in 1:100)
{
    filepath <- file.path(testcase_dir, paste0(i, ".txt"))
    values <- read_testcase(filepath)

    trueGraph <- values$trueGraph
    estGraph <- values$estGraph
    output_flag <- values$output
    spars_flag <- values$spars

    result <- structIntervDist(
        trueGraph,
        estGraph,
        output = output_flag,
        spars = spars_flag
    )

    writeLines(paste0("Testcase ", i), out)

    writeLines("trueGraph:", out)
    for (r in 1:nrow(trueGraph)) {
        writeLines(paste(trueGraph[r, ], collapse = " "), out)
    }

    writeLines("estGraph:", out)
    for (r in 1:nrow(estGraph)) {
        writeLines(paste(estGraph[r, ], collapse = " "), out)
    }

    writeLines("output:", out)
    writeLines(as.character(output_flag), out)

    writeLines("spars:", out)
    writeLines(as.character(spars_flag), out)

    writeLines("sid:", out)
    writeLines(as.character(result$sid), out)

    writeLines("sidUpperBound:", out)
    writeLines(as.character(result$sidUpperBound), out)

    writeLines("sidLowerBound:", out)
    writeLines(as.character(result$sidLowerBound), out)

    writeLines("incorrectMat:", out)
    for (r in 1:nrow(result$incorrectMat)) {
        writeLines(paste(result$incorrectMat[r, ], collapse = " "), out)
    }

    writeLines("", out)
}

close(out)

