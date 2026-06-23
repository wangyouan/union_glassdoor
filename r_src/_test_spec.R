source("/data/disk4/workspace/projects/union_glassdoor/outputs/20260622/current_sweep/cur_helpers.R")
cur <- prep(read_parquet(paste0(OUT, "current_base.parquet")))
message("Data ready: ", nrow(cur), " rows")

# Test: N=5, wlb only
ids <- elig(cur, "each", 5)
d <- filter(cur, election_id %in% ids)
message("N=5 filter: ", nrow(d), " reviews, ", n_distinct(d$election_id), " elections")

t0 <- Sys.time()
f <- feols(v7c("wlb"), d, cluster = CL, warn = FALSE, notes = FALSE)
message("Done in ", round(difftime(Sys.time(), t0, units = "secs"), 1), " seconds")
print(coeftable(f)["win_post", ])
