suppressMessages({library(nanoparquet); library(dplyr); library(readr); library(purrr)})
library(rdrobust)
OUT <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260622/current_sweep/"
fp <- paste0(OUT, "sweep_T7_rdrobust_FIXED.csv"); if(file.exists(fp)) quit(save="no")

source(paste0(OUT, "cur_helpers.R"))   # THR / elig / thr_label / OUTS
cur <- read_parquet(paste0(OUT, "current_base.parquet"))
OUTS <- c("wlb","comp_benefit","overall_rating","senior_mgmt","culture","career_opp")

safe <- function(m, r, c=1) { out <- tryCatch(m[r,c], error=function(e)NA); if(is.null(out)) NA else out }

run_rd <- function(d, y) {
  agg <- d |> group_by(election_id) |>
    summarise(pre = mean(.data[[y]][post==0], na.rm=TRUE),
              postm = mean(.data[[y]][post==1], na.rm=TRUE),
              margin = first(margin), .groups = "drop") |>
    mutate(delta = postm - pre) |> filter(is.finite(delta), is.finite(margin))

  if(nrow(agg) < 20) return(tibble(tau_conv=NA, se_conv=NA, p_conv=NA, tau_bc=NA, se_rob=NA, p_rob=NA, h=NA, n_eff=NA, n_elec=nrow(agg)))

  rr <- tryCatch(rdrobust(y=agg$delta, x=agg$margin, c=0, kernel="triangular", bwselect="mserd"),
                 error=function(e) NULL)

  if(is.null(rr)) return(tibble(tau_conv=NA, se_conv=NA, p_conv=NA, tau_bc=NA, se_rob=NA, p_rob=NA, h=NA, n_eff=NA, n_elec=nrow(agg)))

  tibble(
    tau_conv = safe(rr$coef, "Conventional"),  se_conv = safe(rr$se, "Conventional"), p_conv = safe(rr$pv, "Conventional"),
    tau_bc   = safe(rr$coef, "Bias-Corrected"), se_rob  = safe(rr$se, "Robust"),      p_rob  = safe(rr$pv, "Robust"),
    h = rr$bws["h","left"], n_eff = sum(rr$N_h), n_elec = nrow(agg))
}

rows <- list()
for(t in THR){
  ids <- elig(cur, t$type, t$N)
  d <- cur |> filter(election_id %in% ids)
  ne <- n_distinct(d$election_id)
  message("  ", thr_label(t), ": ", nrow(d), " reviews, ", ne, " elections")
  if(ne < 20) next
  for(y in OUTS){
    r <- run_rd(d, y)
    r$filter <- thr_label(t); r$outcome <- y
    rows[[length(rows)+1]] <- r
  }
}

res <- bind_rows(rows) |>
  mutate(sig_conv = cut(p_conv, c(-Inf,.01,.05,.10,Inf), labels=c("***","**","*","")),
         sig_rob  = cut(p_rob,  c(-Inf,.01,.05,.10,Inf), labels=c("***","**","*",""))) |>
  select(filter, outcome, tau_conv, se_conv, p_conv, sig_conv, tau_bc, se_rob, p_rob, sig_rob, h, n_eff, n_elec)

write_csv(res, fp)
message("saved ", fp, " (", nrow(res), " rows)")

cat("\n=== WLB across thresholds (FIXED) ===\n")
print(res |> filter(outcome=="wlb") |> select(filter, tau_conv, p_conv, sig_conv, tau_bc, p_rob, sig_rob, n_eff, n_elec), n=50)

cat("\n=== Comp across thresholds (FIXED) ===\n")
print(res |> filter(outcome=="comp_benefit") |> select(filter, tau_conv, p_conv, sig_conv, tau_bc, p_rob, sig_rob, n_elec), n=50)
