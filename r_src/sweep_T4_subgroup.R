OUT <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260622/current_sweep/"
source(paste0(OUT,"cur_helpers.R"))
fp <- paste0(OUT,"sweep_T4_subgroup.csv"); if(file.exists(fp)) quit(save="no")

cur <- prep(read_parquet(paste0(OUT,"current_base.parquet")))
message("prepped: ", nrow(cur), " rows")
message("union_classification values: ", paste(unique(cur$union_classification), collapse=", "))

rows <- list()
for(t in THR){
  d_full <- get_sub(cur, t)
  for(g in c("likely_excluded", "likely_unionizable")){
    d <- d_full |> filter(union_classification == g)
    ne <- n_distinct(d$election_id)
    message("  ", thr_label(t), " / ", g, ": ", nrow(d), " reviews, ", ne, " elections")
    if(ne < 20) next
    for(y in c("wlb", "comp_benefit")){
      f <- tryCatch(feols(v7c(y), d, cluster=CL, warn=FALSE, notes=FALSE), error=function(e)NULL)
      v <- win_post_row(f)
      rows[[length(rows)+1]] <- tibble(table="T4", filter=thr_label(t), group=g, outcome=y,
        estimate=v[1], se=v[2], pvalue=v[3], n_events=ne)
    }
  }
}
out <- bind_rows(rows)
write_csv(out, fp)
message("saved ", fp, " (", nrow(out), " rows)")
