OUT <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260622/current_sweep/"
source(paste0(OUT,"cur_helpers.R"))
fp <- paste0(OUT,"sweep_T5_bandwidth.csv"); if(file.exists(fp)) quit(save="no")

cur <- prep(read_parquet(paste0(OUT,"current_base.parquet")))
message("prepped: ", nrow(cur), " rows")

bw_cut <- function(d, bw){
  if(bw == "m20") filter(d, abs(margin) <= .20)
  else if(bw == "m10") filter(d, abs(margin) <= .10)
  else if(bw == "m05") filter(d, abs(margin) <= .05)
  else d
}

rows <- list()
for(t in THR){
  d_full <- get_sub(cur, t)
  for(bw in c("global", "m20", "m10", "m05")){
    d <- bw_cut(d_full, bw)
    ne <- n_distinct(d$election_id)
    message("  ", thr_label(t), " / ", bw, ": ", nrow(d), " reviews, ", ne, " elections")
    if(ne < 20) next
    f <- tryCatch(feols(v7c("wlb"), d, cluster=CL, warn=FALSE, notes=FALSE), error=function(e)NULL)
    v <- win_post_row(f)
    rows[[length(rows)+1]] <- tibble(table="T5", filter=thr_label(t), bandwidth=bw, outcome="wlb",
      estimate=v[1], se=v[2], pvalue=v[3], n_events=ne)
  }
}
out <- bind_rows(rows)
write_csv(out, fp)
message("saved ", fp, " (", nrow(out), " rows)")
