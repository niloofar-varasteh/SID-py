randomDAG <- function(p, probConnect, causalOrder = sample(p, p, replace = FALSE))
{
    DAG <- matrix(0, p, p)

    if (p >= 3) {
        for (i in 1:(p - 2)) {
            node <- causalOrder[i]
            possibleParents <- causalOrder[(i + 1):p]
            numberParents <- rbinom(n = 1, size = (p - i), prob = probConnect)

            if (numberParents > 0) {
                Parents <- sample(x = possibleParents, size = numberParents, replace = FALSE)
                DAG[Parents, node] <- rep(1, numberParents)
            }
        }
    }

    if (p >= 2) {
        node <- causalOrder[p - 1]
        ParentYesNo <- rbinom(n = 1, size = 1, prob = probConnect)
        DAG[causalOrder[p], node] <- ParentYesNo
    }

    return(DAG)
}

# llm
read_testcase <- function(filepath)
{
    lines <- readLines(filepath, warn = FALSE)
    values <- list()

    for (line in lines) {
        line <- trimws(line)

        if (line == "") {
            next
        }

        parts <- strsplit(line, "=", fixed = TRUE)[[1]]
        key <- trimws(parts[1])
        value <- trimws(parts[2])

        if (key == "p") {
            values$p <- as.integer(value)
        } else if (key == "probConnect") {
            values$probConnect <- as.numeric(value)
        } else if (key == "causalOrder") {
            values$causalOrder <- as.integer(strsplit(value, " +")[[1]])
        }
    }

    return(values)
}


script_dir <- getwd()
project_dir <- dirname(script_dir)

testcase_dir <- file.path(project_dir, "tests", "randomDAG", "Testcase")
output_dir <- file.path(project_dir, "tests", "randomDAG", "R_outputs")

if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
}

output_file <- file.path(output_dir, "all_results.txt")
out <- file(output_file, open = "w")

for (i in 1:10000)
{
    filepath <- file.path(testcase_dir, paste0(i, ".txt"))
    values <- read_testcase(filepath)

    p <- values$p
    probConnect <- values$probConnect
    causalOrder <- values$causalOrder

    result <- randomDAG(p, probConnect, causalOrder)

    writeLines(paste0("Testcase ", i), out)
    writeLines(paste0("p=", p), out)
    writeLines(paste0("probConnect=", probConnect), out)
    writeLines(paste0("causalOrder=", paste(causalOrder, collapse = " ")), out)

    writeLines("Result Matrix:", out)
    for (r in 1:nrow(result)) {
        writeLines(paste(result[r, ], collapse = " "), out)
    }

    writeLines("", out)
}

close(out)