allDagsIntern <- 
function (gm, a, row.names, tmp) 
    # Input: adj. mat gm, submatrix a of an UNDIRECTED component and tmp that is set to NULL 
    # (due to the recursive nature of the function),     
{
    if(any((a + t(a))==1))
    #Transpose of a = t(a)
    #Because:if both sides exist, the sum becomes 2 and if only one side exists, the sum becomes 1
    #“If there is at least one place where the edge exists only in one direction, then this submatrix is not fully undirected.

    {
        stop('The matrix is not entirely undirected. This should not happen!')
    }
    
    if (sum(a) == 0)
    # if there is no edge in the undirected component, we are done and can add the DAG to the list of DAGs
    {
        tmp2 <- rbind(tmp, c(gm))
        #c(gm) : converts the matrix gm into a vector by stacking its columns. This is done to create a single row vector that represents the DAG, which can then be added to the list of DAGs in tmp.
        #rbind : This function combines the existing list of DAGs in tmp with the new DAG represented by c(gm).
        if (all(!duplicated(tmp2))) # if tmp2 contains no element twice
            tmp <- tmp2
    }
    else 
    {
        # all nodes can be sinks, but we consider only those who have neighbors.
        sinks <- which(colSums(a) > 0)
        for (x in sinks) 
        {
            gm2 <- gm
            a2 <- NULL
            row.names2 <- NULL
            
            Adj <- (a == 1)
            #a == 1 means “there is an edge here
            Adjx <- Adj[x,]
            #gives the neighbors of node x inside a
            if(any(Adjx))
            {
                un <- which(Adjx)
                #un=undirected
                pp <- length(un)
                #pp = number of neighbors of x
                Adj2 <- matrix(Adj[un,un],pp,pp)
                #Take only the neighbors of x, and look at the connections among those neighbors themselves.So this builds a smaller matrix for the neighbor set.
                diag(Adj2) <- rep(TRUE,pp)
            } else # x does not have any neighbors
            {
                Adj2 <- TRUE
            }
            # Are all (undirected) neighbors of x connected? (O/wise there will be a v-structure if 
            # x becomes a sink node)
            if(all(Adj2)) #if not, don't do anything
            {
                if(any(Adjx))
                {
                    un <- row.names[which(Adjx)]
                    pp <- length(un)
                    #Orient all edges between the neighbors and x into x.
                    gm2[un,row.names[x]] <- rep(1,pp)
                    gm2[row.names[x],un] <- rep(0,pp)
                }
                #Remove node x from the undirected submatrix a.So now you get a smaller subproblem.
                a2 <- a[-x, -x]
                #Also remove x from the node-label list, so the mapping stays correct.
                row.names2 <- row.names[-x]
                tmp <- allDagsIntern(gm2, a2, row.names2, tmp)
            }
        }
    }
    return(tmp)
}



read_testcase <- function(filepath)
{
    lines <- readLines(filepath, warn = FALSE)

    gm <- list()
    a <- list()
    row_names <- integer(0)
    mode <- NULL

    for(line in lines)
    {
        line <- trimws(line)

        if(line == "")
        {
            next
        }

        if(line == "gm:")
        {
            mode <- "gm"
            next
        }

        if(line == "a:")
        {
            mode <- "a"
            next
        }

        if(line == "row.names:")
        {
            mode <- "row_names"
            next
        }

        if(mode == "gm")
        {
            gm[[length(gm) + 1]] <- as.integer(strsplit(line, " +")[[1]])
        }
        else if(mode == "a")
        {
            a[[length(a) + 1]] <- as.integer(strsplit(line, " +")[[1]])
        }
        else if(mode == "row_names")
        {
            row_names <- as.integer(strsplit(line, " +")[[1]])
        }
    }

    gm <- do.call(rbind, gm)
    a <- do.call(rbind, a)

    return(list(
        gm = gm,
        a = a,
        row_names = row_names
    ))
}


script_dir <- getwd()
project_dir <- dirname(script_dir)

testcase_dir <- file.path(project_dir, "tests", "allDagsIntern", "Testcase")
output_dir <- file.path(project_dir, "tests", "allDagsIntern", "R_outputs")

if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
}

output_file <- file.path(output_dir, "all_results.txt")
out <- file(output_file, open = "w")

for(i in 1:1500)
{
    filepath <- file.path(testcase_dir, paste0(i, ".txt"))
    values <- read_testcase(filepath)

    gm <- values$gm
    a <- values$a
    row_names <- values$row_names

    tmp_init <- matrix(numeric(0), ncol = length(c(gm)))
    result <- allDagsIntern(gm, a, row_names, tmp_init)

    writeLines(paste0("Testcase ", i), out)

    writeLines("gm:", out)
    for(r in 1:nrow(gm)) {
        writeLines(paste(gm[r, ], collapse = " "), out)
    }

    writeLines("a:", out)
    for(r in 1:nrow(a)) {
        writeLines(paste(a[r, ], collapse = " "), out)
    }

    writeLines("row.names:", out)
    writeLines(paste(row_names, collapse = " "), out)

    writeLines("Result:", out)
    if(nrow(result) == 0)
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