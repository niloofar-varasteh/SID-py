# If you have clean function-only versions of these files, you can source them here.
source("computePathMatrix.R")
source("computePathMatrix2.R")

dSepAdji <- function(AdjMat, i, condSet, PathMatrix = NULL, PathMatrix2 = NULL, spars = NULL)
    # Copyright (c) 2013 - 2014  Jonas Peters  [peters@stat.math.ethz.ch]
    # All rights reserved.  See the file COPYING for license terms.
    # This function looks for all j, such that i is d-sep to j given the condSet
    # The PathMatrix contains ancestor-relations.
    #
    # REACHABLE:
    #      means that there is a path from i that is not blocked by condSet
    # REACHABLEONNONCAUSALPATH = REACHABLEONNONDIRECTEDPATH (both notations exist):
    #      means that there is a path from i that is not blocked by condSet and that is NOT directed
    #
    # PathMatrix2 contains ancestor-relations when removing arrows condSet->
    # reachableOnNonCausalPath indicates whether in AdjMat there is a nonDirected path from i to j that is not blocked by condSet.
    # More details: For all combinations of nodes and edge direction (incoming or outgoing), we check, to which
    # neighbouring node-orientation pair an open path can be propagated further.
{
    p <- dim(AdjMat)[2]

    if (is.null(spars)) {
        spars <- (p > 99)
    }

    if (is.null(PathMatrix)) {
        PathMatrix <- computePathMatrix(AdjMat)
    }

    if (is.null(PathMatrix2)) {
        PathMatrix2 <- matrix(NA, p, p)
    }

    timeComputePM2 <- 0
    timeComputePM <- 0

    if(is.na(sum(PathMatrix2)))
    {
        ptm <- proc.time()
        PathMatrix2 <- computePathMatrix2(AdjMat, condSet, PathMatrix)
        timeComputePM2 <- timeComputePM2 + (proc.time() - ptm)[3]
    }

    if(length(condSet) == 0)
    {
        AncOfCondSet <- c()
    }

    if(length(condSet) == 1)
    {
        AncOfCondSet <- which(PathMatrix[, condSet] > 0)
    }

    if(length(condSet) > 1)
    {
        AncOfCondSet <- which(rowSums(PathMatrix[, condSet]) > 0)
    }

    reachabilityMatrix <- matrix(0, 2 * p, 2 * p)
    reachableOnNonCausalPathLater <- matrix(0, 2, 2)

    reachableNodes <- rep(0, 2 * p)
    reachableOnNonCausalPath <- rep(0, 2 * p)
    alreadyChecked <- rep(0, p)
    k <- 2
    toCheck <- c(0, 0)

    reachableCh <- which(AdjMat[i, ] == 1)
    if(length(reachableCh) > 0)
    {
        toCheck <- c(toCheck, reachableCh)
        reachableNodes[reachableCh] <- rep(1, length(reachableCh))
        AdjMat[i, reachableCh] <- rep(0, length(reachableCh))
    }

    reachablePa <- which(AdjMat[, i] == 1)
    if(length(reachablePa) > 0)
    {
        toCheck <- c(toCheck, reachablePa)
        reachableNodes[reachablePa + p] <- rep(1, length(reachablePa))
        reachableOnNonCausalPath[reachablePa + p] <- rep(1, length(reachablePa))
        AdjMat[reachablePa, i] <- rep(0, length(reachablePa))
    }

    while(k < length(toCheck))
    {
        k <- k + 1
        a1 <- toCheck[k]

        if(alreadyChecked[a1] == 0)
        {
            currentNode <- a1
            alreadyChecked[a1] <- 1

            Pa <- which(AdjMat[, currentNode] == 1)

            Pa1 <- setdiff(Pa, condSet)
            reachabilityMatrix[Pa1, currentNode] <- rep(1, length(Pa1))
            reachabilityMatrix[Pa1 + p, currentNode] <- rep(1, length(Pa1))

            if(sum(AncOfCondSet == currentNode) > 0)
            {
                reachabilityMatrix[currentNode, Pa + p] <- rep(1, length(Pa))

                if(PathMatrix2[i, currentNode] > 0)
                {
                    reachableOnNonCausalPathLater <- rbind(
                        reachableOnNonCausalPathLater,
                        cbind(rep(currentNode, length(Pa)), Pa)
                    )
                }

                newtoCheck <- Pa
                newtoCheck <- newtoCheck[which(alreadyChecked[newtoCheck] == 0)]
                toCheck <- c(toCheck, newtoCheck)
            }

            if(sum(condSet == currentNode) == 0)
            {
                reachabilityMatrix[currentNode + p, Pa + p] <- rep(1, length(Pa))
                newtoCheck <- Pa
                newtoCheck <- newtoCheck[which(alreadyChecked[newtoCheck] == 0)]
                toCheck <- c(toCheck, newtoCheck)
            }

            Ch <- which(AdjMat[currentNode, ] == 1)

            Ch1 <- setdiff(Ch, condSet)
            reachabilityMatrix[Ch1 + p, currentNode + p] <- rep(1, length(Ch1))

            Ch2 <- intersect(Ch, AncOfCondSet)
            reachabilityMatrix[Ch2, currentNode + p] <- rep(1, length(Ch2))
            Ch2b <- intersect(Ch2, which(PathMatrix2[i, ] > 0))
            reachableOnNonCausalPathLater <- rbind(
                reachableOnNonCausalPathLater,
                cbind(Ch2b, rep(currentNode, length(Ch2b)))
            )

            if(sum(condSet == currentNode) == 0)
            {
                reachabilityMatrix[currentNode, Ch] <- rep(1, length(Ch))
                reachabilityMatrix[currentNode + p, Ch] <- rep(1, length(Ch))
                newtoCheck <- Ch
                newtoCheck <- newtoCheck[which(alreadyChecked[newtoCheck] == 0)]
                toCheck <- c(toCheck, newtoCheck)
            }
        }
    }

    ptm <- proc.time()
    reachabilityMatrix <- computePathMatrix(reachabilityMatrix, spars = spars)
    timeComputePM <- timeComputePM + (proc.time() - ptm)[3]
    reachabilityMatrix <- as(reachabilityMatrix, "matrix")

    ttt2 <- which(reachableNodes == 1)
    if(length(ttt2) == 1)
    {
        tt2 <- which(reachabilityMatrix[ttt2, ] > 0)
    } else
    {
        tt2 <- which(colSums(reachabilityMatrix[ttt2, ]) > 0)
    }
    reachableNodes[tt2] <- rep(1, length(tt2))

    ttt <- which(reachableOnNonCausalPath == 1)
    if(length(ttt) == 1)
    {
        tt <- which(reachabilityMatrix[ttt, ] > 0)
    } else
    {
        tt <- which(colSums(reachabilityMatrix[ttt, ]) > 0)
    }
    reachableOnNonCausalPath[tt] <- rep(1, length(tt))

    if(dim(reachableOnNonCausalPathLater)[1] > 2)
    {
        for(kk in 3:(dim(reachableOnNonCausalPathLater)[1]))
        {
            ReachableThrough <- reachableOnNonCausalPathLater[kk, 1]
            newReachable <- reachableOnNonCausalPathLater[kk, 2]
            reachableOnNonCausalPath[newReachable + p] <- 1

            reachabilityMatrix[newReachable, ReachableThrough] <- 0
            reachabilityMatrix[newReachable, ReachableThrough + p] <- 0
            reachabilityMatrix[newReachable + p, ReachableThrough] <- 0
            reachabilityMatrix[newReachable + p, ReachableThrough + p] <- 0
        }

        ttt <- which(reachableOnNonCausalPath == 1)
        if(length(ttt) == 1)
        {
            tt <- which(reachabilityMatrix[ttt, ] > 0)
        } else
        {
            tt <- which(colSums(reachabilityMatrix[ttt, ]) > 0)
        }
        reachableOnNonCausalPath[tt] <- rep(1, length(tt))
    }

    result <- list()
    result$timeComputePM <- timeComputePM
    result$timeComputePM2 <- timeComputePM2
    result$reachableJ <- rowSums(cbind(reachableNodes[1:p], reachableNodes[(p + 1):(2 * p)])) > 0
    result$reachableOnNonCausalPath <- rowSums(cbind(reachableOnNonCausalPath[1:p], reachableOnNonCausalPath[(p + 1):(2 * p)])) > 0

    return(result)
}

#llm
read_testcase <- function(filepath)
{
    lines <- readLines(filepath, warn = FALSE)

    AdjMat <- list()
    condSet <- integer(0)
    i <- NULL
    mode <- NULL

    for(line in lines)
    {
        line <- trimws(line)

        if(line == "")
        {
            next
        }

        if(line == "Matrix:")
        {
            mode <- "matrix"
            next
        }

        if(line == "i:")
        {
            mode <- "i"
            next
        }

        if(line == "condSet:")
        {
            mode <- "condSet"
            next
        }

        if(mode == "matrix")
        {
            AdjMat[[length(AdjMat) + 1]] <- as.integer(strsplit(line, " +")[[1]])
        } else if(mode == "i")
        {
            i <- as.integer(line)
        } else if(mode == "condSet")
        {
            if(line != "empty")
            {
                condSet <- as.integer(strsplit(line, " +")[[1]])
            }
        }
    }

    AdjMat <- do.call(rbind, AdjMat)

    return(list(
        AdjMat = AdjMat,
        i = i,
        condSet = condSet
    ))
}


script_dir <- getwd()
project_dir <- dirname(script_dir)

testcase_dir <- file.path(project_dir, "tests", "dSepAdji", "Testcase")
output_dir <- file.path(project_dir, "tests", "dSepAdji", "R_outputs")

if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
}

output_file <- file.path(output_dir, "all_results.txt")
out <- file(output_file, open = "w")

for(i_case in 1:10000)
{
    filepath <- file.path(testcase_dir, paste0(i_case, ".txt"))
    values <- read_testcase(filepath)

    AdjMat <- values$AdjMat
    i <- values$i
    condSet <- values$condSet

    result <- dSepAdji(AdjMat, i, condSet)

    writeLines(paste0("Testcase ", i_case), out)

    writeLines("Matrix:", out)
    for(r in 1:nrow(AdjMat)) {
        writeLines(paste(AdjMat[r, ], collapse = " "), out)
    }

    writeLines("i:", out)
    writeLines(as.character(i), out)

    writeLines("condSet:", out)
    if(length(condSet) == 0) {
        writeLines("empty", out)
    } else {
        writeLines(paste(condSet, collapse = " "), out)
    }

    writeLines("reachableJ:", out)
    writeLines(paste(as.integer(result$reachableJ), collapse = " "), out)

    writeLines("reachableOnNonCausalPath:", out)
    writeLines(paste(as.integer(result$reachableOnNonCausalPath), collapse = " "), out)

    writeLines("timeComputePM:", out)
    writeLines(as.character(result$timeComputePM), out)

    writeLines("timeComputePM2:", out)
    writeLines(as.character(result$timeComputePM2), out)

    writeLines("", out)
}

close(out)