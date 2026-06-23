OUT <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260622/current_sweep/"
source(paste0(OUT,"cur_helpers.R"))
fp <- paste0(OUT,"sweep_T2_baseline.csv"); if(file.exists(fp)) quit(save="no")

cur <- prep(read_parquet(paste0(OUT,"current_base.parquet")))
message("prepped: ", nrow(cur), " rows")

rows <- list()
for(t in THR){
  d <- get_sub(cur,t); ne <- n_distinct(d$election_id);
  message("  ", thr_label(t), ": ", nrow(d), " reviews, ", ne, " elections")
  if(ne < 20) next
  for(y in OUTS){
    f <- tryCatch(feols(v7c(y), d, cluster=CL, warn=FALSE, notes=FALSE), error=function(e)NULL)
    v <- win_post_row(f)
    rows[[length(rows)+1]] <- tibble(table="T2", filter=thr_label(t), outcome=y,
      estimate=v[1], se=v[2], pvalue=v[3], n_obs=nrow(d), n_events=ne)
  }
}
out <- bind_rows(rows)
write_csv(out, fp)
message("saved ", fp, " (", nrow(out), " rows)")
