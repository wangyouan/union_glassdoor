suppressMessages({library(nanoparquet); library(dplyr); library(readr); library(purrr)})
library(rdrobust)
OUT <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260622/current_sweep/"
source(paste0(OUT, "cur_helpers.R"))
cur <- read_parquet(paste0(OUT, "current_base.parquet"))

safe <- function(m, r, c=1) { out <- tryCatch(m[r,c], error=function(e)NA); if(is.null(out)) NA else out }

KERNELS <- c("triangular", "uniform", "epanechnikov")
PS <- c(1, 2)

rows <- list()
for(t in THR){
  fname <- thr_label(t)
  if(!(fname %in% c("total>=20", "total>=10", "pre&post>=5"))) next

  ids <- elig(cur, t$type, t$N)
  d <- cur |> filter(election_id %in% ids)
  ne <- n_distinct(d$election_id)
  message("  ", fname, ": ", nrow(d), " reviews, ", ne, " elections")

  for(y in c("wlb","comp_benefit")){  # just the two most important
    agg <- d |> group_by(election_id) |>
      summarise(pre = mean(.data[[y]][post==0], na.rm=TRUE),
                postm = mean(.data[[y]][post==1], na.rm=TRUE),
                margin = first(margin), .groups = "drop") |>
      mutate(delta = postm - pre) |> filter(is.finite(delta), is.finite(margin))

    for(k in KERNELS){
      for(p in PS){
        q <- p + 1
        rr <- tryCatch(rdrobust(y=agg$delta, x=agg$margin, c=0, kernel=k, p=p, q=q, bwselect="mserd"),
                       error=function(e) NULL)
        if(is.null(rr)){
          rows[[length(rows)+1]] <- tibble(filter=fname, outcome=y, kernel=k, p=p,
            tau_conv=NA, se_conv=NA, p_conv=NA, tau_bc=NA, se_rob=NA, p_rob=NA, h=NA, n_eff=NA)
        } else {
          rows[[length(rows)+1]] <- tibble(filter=fname, outcome=y, kernel=k, p=p,
            tau_conv = safe(rr$coef,"Conventional"), se_conv = safe(rr$se,"Conventional"), p_conv = safe(rr$pv,"Conventional"),
            tau_bc   = safe(rr$coef,"Bias-Corrected"), se_rob  = safe(rr$se,"Robust"), p_rob  = safe(rr$pv,"Robust"),
            h = rr$bws["h","left"], n_eff = sum(rr$N_h))
        }
      }
    }
  }
}

res <- bind_rows(rows) |>
  mutate(sig_rob = cut(p_rob, c(-Inf,.01,.05,.10,Inf), labels=c("***","**","*","")))

cat("\n========== WLB: p=1 vs p=2, all kernels ==========\n")
print(res |> filter(outcome=="wlb") |>
  select(filter, p, kernel, tau_bc, se_rob, p_rob, sig_rob, h, n_eff) |>
  arrange(filter, p, kernel), n=30)

cat("\n========== Comp: p=1 vs p=2, all kernels ==========\n")
print(res |> filter(outcome=="comp_benefit") |>
  select(filter, p, kernel, tau_bc, p_rob, sig_rob) |>
  arrange(filter, p, kernel), n=30)

# Best WLB combination
cat("\n========== BEST WLB (sorted by p_rob) ==========\n")
print(res |> filter(outcome=="wlb") |>
  select(filter, p, kernel, tau_bc, p_rob, h, n_eff) |>
  arrange(p_rob), n=20)

write_csv(res, paste0(OUT, "rdrobust_poly_kernel_comparison.csv"))
message("\nsaved")
