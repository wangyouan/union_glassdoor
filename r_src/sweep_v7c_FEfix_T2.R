suppressMessages({library(fixest); library(dplyr); library(tidyr); library(nanoparquet); library(readr); library(purrr)})
OUT <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260622/current_sweep/"
fp <- paste0(OUT, "sweep_T2_baseline_FEfix.csv")
source(paste0(OUT, "cur_helpers.R"))
cur <- prep(read_parquet(paste0(OUT, "current_base.parquet")))
message("prepped: ", nrow(cur), " rows")

# Run in batches to avoid timeout
BATCH <- as.integer(Sys.getenv("BATCH_IDX", unset="1"))
BATCH_SIZE <- 3  # thresholds per batch
start_idx <- (BATCH - 1) * BATCH_SIZE + 1
end_idx <- min(BATCH * BATCH_SIZE, length(THR))
message("Batch ", BATCH, ": thresholds ", start_idx, "-", end_idx, " of ", length(THR))

fp_batch <- paste0(OUT, "sweep_T2_FEfix_batch", BATCH, ".csv")

win_post_row <- function(fit){
  if(is.null(fit) || !("win_post" %in% rownames(coeftable(fit)))) return(c(NA, NA, NA))
  r <- coeftable(fit)["win_post", ]; c(r["Estimate"], r["Std. Error"], r["Pr(>|t|)"])
}

rows <- list()
for(idx in start_idx:end_idx){
  t <- THR[[idx]]
  d <- get_sub(cur, t); ne <- n_distinct(d$election_id)
  message("  ", thr_label(t), ": ", nrow(d), " reviews, ", ne, " elections")
  if(ne < 20) next
  for(y in OUTS){
    f <- tryCatch(feols(v7c(y), d, cluster=CL, warn=FALSE, notes=FALSE), error=function(e) NULL)
    v <- win_post_row(f)
    rows[[length(rows)+1]] <- tibble(filter=thr_label(t), outcome=y,
      estimate=v[1], se=v[2], pvalue=v[3], n_events=ne)
  }
}
res <- bind_rows(rows)
res$sig <- cut(res$pvalue, c(-Inf,.01,.05,.10,Inf), labels=c("***","**","*",""))
write_csv(res, fp_batch)
message("saved ", fp_batch, " (", nrow(res), " rows)")

cat("\n=== WLB ===\n")
print(res |> filter(outcome=="wlb") |> select(filter, estimate, se, pvalue, sig, n_events), n=50)
cat("\n=== Comp ===\n")
print(res |> filter(outcome=="comp_benefit") |> select(filter, estimate, pvalue, sig), n=50)
