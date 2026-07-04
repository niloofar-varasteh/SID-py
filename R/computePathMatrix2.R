computePathMatrix2 <- function(G, condSet, PathMatrix1, spars = FALSE)
# Copyright (c) 2013 - 2013  Jonas Peters  [peters@stat.math.ethz.ch]
# All rights reserved.  See the file COPYING for license terms.
{
    # The only difference to the function computePathMatrix is that this function changes
    # the graph by removing all edges that leave condSet.
    # If condSet is empty, it just returns PathMatrix1.

    p <- dim(G)[2]

    if(length(condSet) > 0)
    {
        G[condSet, ] <- matrix(0, length(condSet), p)

        if(spars)
        {
            G <- Matrix(G)
            PathMatrix2 <- Diagonal(p) + G
        } else
        {
            PathMatrix2 <- diag(1, p) + G
        }

        k <- ceiling(log(p) / log(2))
        for(i in 1:k)
        {
            PathMatrix2 <- PathMatrix2 %*% PathMatrix2
        }

        PathMatrix2 <- PathMatrix2 > 0
    } else
    {
        PathMatrix2 <- PathMatrix1
    }

    return(PathMatrix2)
}

#llm
read_testcase <- function(filepath)
{
    lines <- readLines(filepath, warn = FALSE)

    G <- list()
    PathMatrix1 <- list()
    condSet <- integer(0)

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

        if(line == "condSet:")
        {
            mode <- "condSet"
            next
        }

        if(line == "PathMatrix1:")
        {
            mode <- "path1"
            next
        }

        if(mode == "matrix")
        {
            G[[length(G) + 1]] <- as.integer(strsplit(line, " +")[[1]])
        } else if(mode == "condSet")
        {
            if(line != "empty")
            {
                condSet <- as.integer(strsplit(line, " +")[[1]])
            }
        } else if(mode == "path1")
        {
            PathMatrix1[[length(PathMatrix1) + 1]] <- as.integer(strsplit(line, " +")[[1]])
        }
    }

    G <- do.call(rbind, G)
    PathMatrix1 <- do.call(rbind, PathMatrix1)

    return(list(
        G = G,
        condSet = condSet,
        PathMatrix1 = PathMatrix1
    ))
}

#llm
script_dir <- getwd()
project_dir <- dirname(script_dir)

testcase_dir <- file.path(project_dir, "tests", "computePathMatrix2", "Testcase")
output_dir <- file.path(project_dir, "tests", "computePathMatrix2", "R_outputs")

if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
}

output_file <- file.path(output_dir, "all_results.txt")
out <- file(output_file, open = "w")

for(i in 1:10000)
{
    filepath <- file.path(testcase_dir, paste0(i, ".txt"))
    values <- read_testcase(filepath)

    G <- values$G
    condSet <- values$condSet
    PathMatrix1 <- values$PathMatrix1

    result <- computePathMatrix2(G, condSet, PathMatrix1)

    writeLines(paste0("Testcase ", i), out)

    writeLines("Matrix:", out)
    for(r in 1:nrow(G)) {
        writeLines(paste(G[r, ], collapse = " "), out)
    }

    writeLines("condSet:", out)
    if(length(condSet) == 0) {
        writeLines("empty", out)
    } else {
        writeLines(paste(condSet, collapse = " "), out)
    }

    writeLines("PathMatrix1:", out)
    for(r in 1:nrow(PathMatrix1)) {
        writeLines(paste(PathMatrix1[r, ], collapse = " "), out)
    }

    writeLines("Result Matrix:", out)
    for(r in 1:nrow(result)) {
        writeLines(paste(as.integer(result[r, ]), collapse = " "), out)
    }

    writeLines("", out)
}

close(out)