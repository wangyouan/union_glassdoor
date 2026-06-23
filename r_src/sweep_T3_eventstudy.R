OUT <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260622/current_sweep/"
source(paste0(OUT,"cur_helpers.R"))
fp <- paste0(OUT,"sweep_T3_eventstudy.csv"); if(file.exists(fp)) quit(save="no")

cur <- prep(read_parquet(paste0(OUT,"current_base.parquet")))
message("prepped: ", nrow(cur), " rows")

es_fml <- function(y) as.formula(paste0(y," ~ i(event_q,win,ref='-1')+win+post:margin+emp_status+seniority_f+state_clean+role_clean | gvkey+review_year+event_q"))

rows <- list()
for(t in THR){
  d0 <- get_sub(cur,t)
  d <- d0 |> filter(event_q >= -3, event_q <= 3) |> mutate(event_q = factor(event_q, levels = as.character(-3:3)))
  ne <- n_distinct(d$election_id)
  message("  ", thr_label(t), ": ", nrow(d), " reviews, ", ne, " elections")
  if(ne < 20) next
  for(y in KEY){
    f <- tryCatch(feols(es_fml(y), d, cluster=CL, warn=FALSE, notes=FALSE), error=function(e)NULL)
    pre <- tryCatch(fixest::wald(f, "event_q::(-3|-2):win", print=FALSE)$p, error=function(e)NA)
    t0 <- NA
    if(!is.null(f)){
      ct <- coeftable(f)
      rn <- grep("event_q::0:win", rownames(ct), value=TRUE)
      if(length(rn)) t0 <- ct[rn[1], "Pr(>|t|)"]
    }
    # pooled post
    fp2 <- tryCatch(feols(v7c(y), d0, cluster=CL, warn=FALSE, notes=FALSE), error=function(e)NULL)
    v <- win_post_row(fp2)
    rows[[length(rows)+1]] <- tibble(table="T3", filter=thr_label(t), outcome=y,
      pretrend_p=pre, t0_p=t0, pooled_est=v[1], pooled_p=v[3], n_events=ne)
  }
}
out <- bind_rows(rows)
write_csv(out, fp)
message("saved ", fp, " (", nrow(out), " rows)")
