suppressMessages({library(fixest); library(dplyr); library(tidyr); library(nanoparquet); library(readr); library(purrr)})
OUT <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260622/current_sweep/"
fp <- paste0(OUT, "sweep_T5_bandwidth_FEfix.csv")
source(paste0(OUT, "cur_helpers.R"))
cur <- prep(read_parquet(paste0(OUT, "current_base.parquet")))
message("prepped: ", nrow(cur), " rows")

win_post_row <- function(fit){
  if(is.null(fit) || !("win_post" %in% rownames(coeftable(fit)))) return(c(NA, NA, NA))
  r <- coeftable(fit)["win_post", ]; c(r["Estimate"], r["Std. Error"], r["Pr(>|t|)"])
}

bw_cut <- function(d, bw){
  if(bw == "m20") filter(d, abs(margin) <= .20)
  else if(bw == "m10") filter(d, abs(margin) <= .10)
  else if(bw == "m05") filter(d, abs(margin) <= .05)
  else d
}

rows <- list()
for(t in THR){
  d_all <- get_sub(cur, t)
  for(bw in c("global", "m20", "m10", "m05")){
    d <- bw_cut(d_all, bw); ne <- n_distinct(d$election_id)
    if(ne < 20) next
    message("  ", thr_label(t), " / ", bw, ": ", nrow(d), " reviews, ", ne, " elections")
    f <- tryCatch(feols(v7c("wlb"), d, cluster=CL, warn=FALSE, notes=FALSE), error=function(e) NULL)
    v <- win_post_row(f)
    rows[[length(rows)+1]] <- tibble(filter=thr_label(t), bandwidth=bw,
      estimate=v[1], se=v[2], pvalue=v[3], n_events=ne)
  }
}
t5 <- bind_rows(rows)
t5$sig <- cut(t5$pvalue, c(-Inf,.01,.05,.10,Inf), labels=c("***","**","*",""))
write_csv(t5, fp)
message("saved ", fp, " (", nrow(t5), " rows)")

cat("\n=== T5 WLB (FE-fixed) ===\n")
print(t5 |> select(filter, bandwidth, estimate, se, pvalue, sig, n_events) |> arrange(filter, bandwidth), n=50)
