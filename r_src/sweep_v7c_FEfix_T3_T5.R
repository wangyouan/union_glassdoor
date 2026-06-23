suppressMessages({library(fixest); library(dplyr); library(tidyr); library(nanoparquet); library(readr); library(purrr)})
OUT <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260622/current_sweep/"
BATCH <- as.integer(Sys.getenv("BATCH_IDX", unset="1"))
source(paste0(OUT, "cur_helpers.R"))
cur <- prep(read_parquet(paste0(OUT, "current_base.parquet")))
message("prepped: ", nrow(cur), " rows, batch=", BATCH)

es_all <- cur |> filter(event_q >= -3, event_q <= 3) |>
  mutate(event_q = factor(event_q, levels = as.character(-3:3)))

es_fml <- function(y) as.formula(paste0(y, " ~ i(event_q,win,ref='-1')+win+post:margin+emp_status+seniority_f | gvkey+review_year+state_clean+role_clean+event_q"))

win_post_row <- function(fit){
  if(is.null(fit) || !("win_post" %in% rownames(coeftable(fit)))) return(c(NA, NA, NA))
  r <- coeftable(fit)["win_post", ]; c(r["Estimate"], r["Std. Error"], r["Pr(>|t|)"])
}

# Determine threshold subset
THR_SUB <- if(BATCH == 1) THR[1:3] else if(BATCH == 2) THR[4:6] else THR[7:7]

# ===== T3 =====
fp3 <- paste0(OUT, "sweep_T3_FEfix_batch", BATCH, ".csv")
rows <- list()
for(t in THR_SUB){
  ids <- elig(cur, t$type, t$N)
  d_es <- es_all |> filter(election_id %in% ids)
  d_all <- get_sub(cur, t)
  ne <- n_distinct(d_all$election_id)
  message("  T3 ", thr_label(t), ": ES=", nrow(d_es), " pool=", nrow(d_all), " reviews, ", ne, " elections")
  if(ne < 20) next
  for(y in c("wlb", "comp_benefit", "overall_rating")){
    fe <- tryCatch(feols(es_fml(y), d_es, cluster=CL, warn=FALSE, notes=FALSE), error=function(e) NULL)
    pre <- tryCatch(fixest::wald(fe, "event_q::(-3|-2):win", print=FALSE)$p, error=function(e) NA)
    fp <- tryCatch(feols(v7c(y), d_all, cluster=CL, warn=FALSE, notes=FALSE), error=function(e) NULL)
    v <- win_post_row(fp)
    rows[[length(rows)+1]] <- tibble(filter=thr_label(t), outcome=y,
      pretrend_p=pre, pooled_est=v[1], pooled_se=v[2], pooled_p=v[3])
  }
}
t3 <- bind_rows(rows)
t3$pooled_sig <- cut(t3$pooled_p, c(-Inf,.01,.05,.10,Inf), labels=c("***","**","*",""))
write_csv(t3, fp3)
message("saved ", fp3, " (", nrow(t3), " rows)")

# ===== T5 =====
fp5 <- paste0(OUT, "sweep_T5_FEfix_batch", BATCH, ".csv")
bw_cut <- function(d, bw){
  if(bw == "m20") filter(d, abs(margin) <= .20)
  else if(bw == "m10") filter(d, abs(margin) <= .10)
  else if(bw == "m05") filter(d, abs(margin) <= .05)
  else d
}
rows <- list()
for(t in THR_SUB){
  d_all <- get_sub(cur, t)
  for(bw in c("global", "m20", "m10", "m05")){
    d <- bw_cut(d_all, bw); ne <- n_distinct(d$election_id)
    message("  T5 ", thr_label(t), " / ", bw, ": ", nrow(d), " reviews, ", ne, " elections")
    if(ne < 20) next
    f <- tryCatch(feols(v7c("wlb"), d, cluster=CL, warn=FALSE, notes=FALSE), error=function(e) NULL)
    v <- win_post_row(f)
    rows[[length(rows)+1]] <- tibble(filter=thr_label(t), bandwidth=bw,
      estimate=v[1], se=v[2], pvalue=v[3], n_events=ne)
  }
}
t5 <- bind_rows(rows)
t5$sig <- cut(t5$pvalue, c(-Inf,.01,.05,.10,Inf), labels=c("***","**","*",""))
write_csv(t5, fp5)
message("saved ", fp5, " (", nrow(t5), " rows)")

cat("\n=== T3 WLB ===\n")
print(t3 |> filter(outcome=="wlb") |> select(filter, pretrend_p, pooled_est, pooled_p, pooled_sig), n=50)

cat("\n=== T5 WLB ===\n")
print(t5 |> filter(bandwidth=="global") |> select(filter, estimate, se, pvalue, sig, n_events), n=50)
