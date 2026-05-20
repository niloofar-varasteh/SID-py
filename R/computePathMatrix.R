computePathMatrix <- function(G, spars=FALSE)
    # Copyright (c) 2013 - 2013  Jonas Peters  [peters@stat.math.ethz.ch]
    # All rights reserved.  See the file COPYING for license terms. 
{
    #this function takes an adjacency matrix G from a DAG and computes a path matrix for which 
    # entry(i,j) being one means that there is a directed path from i to j
    # the diagonal will also be one
    p <- dim(G)[2]
    
    if((p > 3000) && (spars == FALSE))
    {
        warning("Maybe you should use the sparse version by using spars=TRUE to increase speed")
    }
    
    if(spars)
    {
        G <- Matrix(G)
        PathMatrix <- Diagonal(p) + G
    } else
    {
        PathMatrix <- diag(1,p) + G
        # PathMatrix <- as(diag(1,p) + G, "sparseMatrix")
        #   sparseMatrix does not seem to lead to a big improvement in speed (in fact, it seems slightly slower)
    }

    k <- ceiling(log(p)/log(2))
    for(i in 1:k)
    {
        PathMatrix <- PathMatrix %*% PathMatrix            
    }
    PathMatrix <- PathMatrix > 0
    
    return(PathMatrix)
}

# llm

script_dir <- getwd()
project_dir <- dirname(script_dir)

testcase_dir <- file.path(project_dir, "tests", "computePathMatrix", "Testcase")
output_dir <- file.path(project_dir, "tests", "computePathMatrix", "R_outputs")

if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
}

output_file <- file.path(output_dir, "all_results.txt")
out <- file(output_file, open = "w")

for(i in 1:100)
{
    filepath <- file.path(testcase_dir, paste0(i, ".txt"))
    G <- as.matrix(read.table(filepath))
    result <- computePathMatrix(G)

    writeLines(paste0("Testcase ", i), out)
    writeLines("Matrix:", out)
    for(r in 1:nrow(G)) {
        writeLines(paste(G[r, ], collapse = " "), out)
    }

    writeLines("Result Matrix:", out)
    for(r in 1:nrow(result)) {
        writeLines(paste(as.integer(result[r, ]), collapse = " "), out)
    }

    writeLines("", out)
}

close(out)