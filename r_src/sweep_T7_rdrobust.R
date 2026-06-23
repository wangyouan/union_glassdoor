OUT <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260622/current_sweep/"
source(paste0(OUT,"cur_helpers.R"))
fp <- paste0(OUT,"sweep_T7_rdrobust.csv"); if(file.exists(fp)) quit(save="no")

if(!requireNamespace("rdrobust", quietly=TRUE)) install.packages("rdrobust", repos="https://cloud.r-project.org")
library(rdrobust)

cur <- read_parquet(paste0(OUT,"current_base.parquet"))  # no prep FE needed for aggregate
message("loaded: ", nrow(cur), " rows")

rows <- list()
for(t in THR){
  ids <- elig(cur, t$type, t$N)
  d <- cur |> filter(election_id %in% ids)
  ne <- n_distinct(d$election_id)
  message("  ", thr_label(t), ": ", nrow(d), " reviews, ", ne, " elections")
  if(ne < 20) next

  for(y in OUTS){
    agg <- d |> group_by(election_id) |> summarise(
      pre = mean(.data[[y]][post==0], na.rm=TRUE),
      postm = mean(.data[[y]][post==1], na.rm=TRUE),
      margin = first(margin),
      .groups = "drop") |>
      mutate(delta = postm - pre) |>
      filter(is.finite(delta), is.finite(margin))

    if(nrow(agg) < 20) {
      rows[[length(rows)+1]] <- tibble(table="T7_rdrobust", filter=thr_label(t), outcome=y,
        estimate=NA_real_, se=NA_real_, pvalue=NA_real_, n_eff=NA_real_, n_events=ne)
      next
    }

    rr <- tryCatch(rdrobust(y=agg$delta, x=agg$margin, c=0), error=function(e)NULL)
    if(is.null(rr)){
      est <- se <- pv <- NA; nb <- NA
    } else {
      est <- rr$coef["Robust", 1]
      se <- rr$se["Robust", 1]
      pv <- rr$pv["Robust", 1]
      nb <- sum(rr$N_h)
    }
    rows[[length(rows)+1]] <- tibble(table="T7_rdrobust", filter=thr_label(t), outcome=y,
      estimate=est, se=se, pvalue=pv, n_eff=nb, n_events=ne)
  }
}
out <- bind_rows(rows)
write_csv(out, fp)
message("saved ", fp, " (", nrow(out), " rows)")
