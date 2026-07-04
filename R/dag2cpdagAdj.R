# Note:
# This file uses graphNEL and pcalg::dag2cpdag.
# Make sure the required packages are installed and available.


#llm
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
library(methods)
library(graph)
library(pcalg)

source("computeCausOrder.R")

dag2cpdagAdj <- function(Adj)
# Copyright (c) 2010 - 2012  Jonas Peters  [peters@stat.math.ethz.ch]
# All rights reserved.  See the file COPYING for license terms.
{
    if(sum(Adj) == 0)
    {
        return(Adj)
    }

    cO <- computeCausOrder(Adj)
    d <- as(Adj[cO, cO], "graphNEL")
    cpd <- pcalg::dag2cpdag(d)
    res <- matrix(NA, dim(Adj)[1], dim(Adj)[1])
    res[cO, cO] <- as(cpd, "matrix")
    result <- res

    return(result)
}

#llm
script_dir <- getwd()
project_dir <- dirname(script_dir)

testcase_dir <- file.path(project_dir, "tests", "dag2cpdagAdj", "Testcase")
output_dir <- file.path(project_dir, "tests", "dag2cpdagAdj", "R_outputs")

if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
}

output_file <- file.path(output_dir, "all_results.txt")
out <- file(output_file, open = "w")

for(i in 1:10000)
{
    filepath <- file.path(testcase_dir, paste0(i, ".txt"))
    G <- as.matrix(read.table(filepath))

    result <- dag2cpdagAdj(G)

    writeLines(paste0("Testcase ", i), out)
    writeLines("Matrix:", out)
    for(r in 1:nrow(G)) {
        writeLines(paste(G[r, ], collapse = " "), out)
    }

    writeLines("Result Matrix:", out)
    for(r in 1:nrow(result)) {
        writeLines(paste(result[r, ], collapse = " "), out)
    }

    writeLines("", out)
}

close(out)