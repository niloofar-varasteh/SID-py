source("allDagsIntern.R")

allDagsJonas <- function(adj, row.names)
{
    # Input: adj. mat of a DAG with row.names, containing the undirected component that
    # should be extended
    # !!!! the function can probably be faster if we use partial orderings

    a <- adj[row.names, row.names]

    if(any((a + t(a)) == 1))
    {
        #warning("The matrix is not entirely undirected.")
        return(-1)
    }

    return(allDagsIntern(adj, a, row.names, NULL))
}


read_testcase <- function(filepath)
{
    lines <- readLines(filepath, warn = FALSE)

    adj <- list()
    row_names <- integer(0)
    mode <- NULL

    for(line in lines)
    {
        line <- trimws(line)

        if(line == "")
        {
            next
        }

        if(line == "adj:")
        {
            mode <- "adj"
            next
        }

        if(line == "row.names:")
        {
            mode <- "row_names"
            next
        }

        if(mode == "adj")
        {
            adj[[length(adj) + 1]] <- as.integer(strsplit(line, " +")[[1]])
        }
        else if(mode == "row_names")
        {
            row_names <- as.integer(strsplit(line, " +")[[1]])
        }
    }

    adj <- do.call(rbind, adj)

    return(list(
        adj = adj,
        row_names = row_names
    ))
}


script_dir <- getwd()
project_dir <- dirname(script_dir)

testcase_dir <- file.path(project_dir, "tests", "allDagsJonas", "Testcase")
output_dir <- file.path(project_dir, "tests", "allDagsJonas", "R_outputs")

if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
}

output_file <- file.path(output_dir, "all_results.txt")
out <- file(output_file, open = "w")

for(i in 1:10000)
{
    filepath <- file.path(testcase_dir, paste0(i, ".txt"))
    values <- read_testcase(filepath)

    adj <- values$adj
    row_names <- values$row_names

    result <- allDagsJonas(adj, row_names)

    writeLines(paste0("Testcase ", i), out)

    writeLines("adj:", out)
    for(r in 1:nrow(adj)) {
        writeLines(paste(adj[r, ], collapse = " "), out)
    }

    writeLines("row.names:", out)
    writeLines(paste(row_names, collapse = " "), out)

    writeLines("Result:", out)
    if(length(result) == 1 && result[1] == -1)
    {
        writeLines("-1", out)
    }
    else if(is.null(result))
    {
        writeLines("empty", out)
    }
    else if(nrow(result) == 0)
    {
        writeLines("empty", out)
    }
    else
    {
        for(r in 1:nrow(result)) {
            writeLines(paste(result[r, ], collapse = " "), out)
        }
    }

    writeLines("", out)
}

close(out)