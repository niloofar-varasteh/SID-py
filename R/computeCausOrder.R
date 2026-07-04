computeCausOrder <- function(G)
{
    p <- dim(G)[2]
    remaining <- 1:p
    causOrder <- rep(NA, p)

    for(i in 1:(p-1))
    {
        root <- min(which(colSums(G) == 0))
        causOrder[i] <- remaining[root]
        remaining <- remaining[-root]
        G <- G[-root, -root]
    }

    causOrder[p] <- remaining[1]
    return(causOrder)
}


script_dir <- getwd()
project_dir <- dirname(script_dir)

testcase_dir <- file.path(project_dir, "tests", "computeCausOrder", "Testcase")
output_dir <- file.path(project_dir, "tests", "computeCausOrder", "R_outputs")

if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
}

output_file <- file.path(output_dir, "all_results.txt")
out <- file(output_file, open = "w")

for(i in 1:10000)
{
    filepath <- file.path(testcase_dir, paste0(i, ".txt"))
    G <- as.matrix(read.table(filepath))
    result <- computeCausOrder(G)

    writeLines(paste0("Testcase ", i), out)
    writeLines("Matrix:", out)
    for(r in 1:nrow(G)) {
        writeLines(paste(G[r, ], collapse = " "), out)
    }

    writeLines("Result:", out)
    writeLines(paste0("[", paste(result, collapse = ", "), "]"), out)
    writeLines("", out)
}

close(out)
