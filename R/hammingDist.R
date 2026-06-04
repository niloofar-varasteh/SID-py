hammingDist <- function(G1, G2, allMistakesOne = TRUE)
    # hammingDist(G1,G2)
    #
    # Computes Hamming Distance between DAGs G1 and G2 with SHD(->,<-) = 1 if allMistakesOne == TRUE
    #
    # INPUT:  G1, G2     adjacency graph containing only zeros and ones: (i,j)=1 means edge from X_i to X_j.
    #
    # OUTPUT: hammingDis Hamming Distance between G1 and G2
    #
    # Copyright (c) 2012-2013  Jonas Peters [peters@stat.math.ethz.ch]
    # All rights reserved.  See the file COPYING for license terms.
{
    if(allMistakesOne)
    {
        Gtmp <- (G1 + G2) %% 2
        Gtmp <- Gtmp + t(Gtmp)
        nrReversals <- sum(Gtmp == 2) / 2
        nrInclDel <- sum(Gtmp == 1) / 2
        hammingDis <- nrReversals + nrInclDel
    } else
    {
        hammingDis <- sum(abs(G1 - G2))
        # correction: dist(-,.) = 1, not 2
        hammingDis <- hammingDis - 0.5 * sum(G1 * t(G1) * (1 - G2) * t(1 - G2) + G2 * t(G2) * (1 - G1) * t(1 - G1))
    }

    return(hammingDis)
}


read_testcase <- function(filepath)
{
    lines <- readLines(filepath, warn = FALSE)

    G1 <- list()
    G2 <- list()
    mode <- NULL

    for(line in lines)
    {
        line <- trimws(line)

        if(line == "")
        {
            next
        }

        if(line == "Matrix1:")
        {
            mode <- "matrix1"
            next
        }

        if(line == "Matrix2:")
        {
            mode <- "matrix2"
            next
        }

        if(mode == "matrix1")
        {
            G1[[length(G1) + 1]] <- as.integer(strsplit(line, " +")[[1]])
        } else if(mode == "matrix2")
        {
            G2[[length(G2) + 1]] <- as.integer(strsplit(line, " +")[[1]])
        }
    }

    G1 <- do.call(rbind, G1)
    G2 <- do.call(rbind, G2)

    return(list(G1 = G1, G2 = G2))
}

#llm
script_dir <- getwd()
project_dir <- dirname(script_dir)

testcase_dir <- file.path(project_dir, "tests", "hammingDist", "Testcase")
output_dir <- file.path(project_dir, "tests", "hammingDist", "R_outputs")

if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
}

output_file <- file.path(output_dir, "all_results.txt")
out <- file(output_file, open = "w")

for(i in 1:100)
{
    filepath <- file.path(testcase_dir, paste0(i, ".txt"))
    values <- read_testcase(filepath)

    G1 <- values$G1
    G2 <- values$G2

    result <- hammingDist(G1, G2)

    writeLines(paste0("Testcase ", i), out)

    writeLines("Matrix1:", out)
    for(r in 1:nrow(G1)) {
        writeLines(paste(G1[r, ], collapse = " "), out)
    }

    writeLines("Matrix2:", out)
    for(r in 1:nrow(G2)) {
        writeLines(paste(G2[r, ], collapse = " "), out)
    }

    writeLines("Result:", out)
    writeLines(as.character(result), out)

    writeLines("", out)
}

close(out)