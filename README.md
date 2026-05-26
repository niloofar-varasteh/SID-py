$env:Path += ";C:\Program Files\R\R-4.5.3\bin"  
Rscript -e ".libPaths(c(Sys.getenv('R_LIBS_USER'), .libPaths())); source('R/dag2cpdagAdj.R')"