#!/usr/bin/env Rscript
# T7 aggregate rdrobust + T8 firm-quarter dynamic — 10 DVs, 7 filters
# T7: p=2, q=3, triangular kernel, mserd bandwidth
# T8: election×quarter aggregation, firm FE + event_q FE

suppressMessages({
  library(fixest); library(dplyr); library(tidyr); library(nanoparquet); library(readr); library(purrr)
  library(rdrobust)
})

OUT <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260624/current_former_bargaining_unit/"

DV10 <- c("overall_rating","career_opp","comp_benefit","senior_mgmt","wlb","culture",
          "recommend","business_outlook","ceo_approval","diversity")

df <- read_parquet(paste0(OUT, "enriched_sample.parquet"))
cur <- df[df$is_current_employee == 1, ]

prep <- function(d){
  d |> mutate(
    gvkey=as.character(gvkey), review_year=as.integer(review_year),
    win=as.integer(win), post=as.integer(post), margin=as.numeric(margin), win_post=win*post,
    state_clean=case_when(!is.na(is_us_review)&is_us_review==1~state_y, TRUE~"Non_US") |> replace_na("Non_US"))
}
cur <- prep(cur)
cat(sprintf("Current: %d rows, %d elections\n", nrow(cur), n_distinct(cur$election_id)))

# Filters
THR <- c(lapply(c(1,3,5,10,20), function(N) list(type="each",N=N)),
         lapply(c(10,20),       function(N) list(type="total",N=N)))

elig <- function(d, type, N) {
  if (type == "each")
    d |> group_by(election_id) |> summarise(a=sum(post==0), b=sum(post==1), .groups="drop") |>
      filter(a>=N, b>=N) |> pull(election_id)
  else
    d |> group_by(election_id) |> summarise(n=n(), .groups="drop") |> filter(n>=N) |> pull(election_id)
}
thr_label <- function(t) paste0(ifelse(t$type=="each","pre&post>=","total>="), t$N)

safe <- function(m, r, c=1) { out <- tryCatch(m[r,c], error=function(e) NA); if(is.null(out)) NA else out }

# ═══════════════════════════════════════════════════════════════════════════
# T7: Aggregate rdrobust
# ═══════════════════════════════════════════════════════════════════════════
cat("\n=== T7: rdrobust (p=2, q=3) ===\n")

run_rd <- function(d, dv) {
  agg <- d |> group_by(election_id) |>
    summarise(pre=mean(.data[[dv]][post==0], na.rm=TRUE),
              postm=mean(.data[[dv]][post==1], na.rm=TRUE),
              margin=first(margin), .groups="drop") |>
    mutate(delta=postm-pre) |> filter(is.finite(delta), is.finite(margin))
  if(nrow(agg) < 20) return(tibble(tau_conv=NA, se_conv=NA, p_conv=NA, tau_bc=NA, se_rob=NA, p_rob=NA, h=NA, n_eff=NA, n_elec=nrow(agg)))
  rr <- tryCatch(rdrobust(y=agg$delta, x=agg$margin, c=0, kernel="triangular", p=2, q=3, bwselect="mserd"), error=function(e) NULL)
  if(is.null(rr)) return(tibble(tau_conv=NA, se_conv=NA, p_conv=NA, tau_bc=NA, se_rob=NA, p_rob=NA, h=NA, n_eff=NA, n_elec=nrow(agg)))
  tibble(tau_conv=safe(rr$coef,"Conventional"), se_conv=safe(rr$se,"Conventional"), p_conv=safe(rr$pv,"Conventional"),
         tau_bc=safe(rr$coef,"Bias-Corrected"), se_rob=safe(rr$se,"Robust"), p_rob=safe(rr$pv,"Robust"),
         p=2L, q=3L, h=rr$bws["h","left"], n_eff=sum(rr$N_h), n_elec=nrow(agg))
}

t7_rows <- list()
for(t_iter in THR) {
  cur_wlb <- cur[!is.na(cur$wlb), ]
  eids <- elig(cur_wlb, t_iter$type, t_iter$N)
  ne <- length(eids); if(ne < 20) next
  d <- cur[cur$election_id %in% eids, ]
  for(dv in DV10) {
    d_dv <- d[!is.na(d[[dv]]), ]
    if(n_distinct(d_dv$election_id) < 20) next
    r <- run_rd(d_dv, dv); r$filter <- thr_label(t_iter); r$outcome <- dv
    t7_rows[[length(t7_rows)+1]] <- r
  }
  cat(sprintf("  %s: %d elections\n", thr_label(t_iter), ne))
}

t7_out <- bind_rows(t7_rows) |>
  mutate(sig_conv=cut(p_conv, c(-Inf,.01,.05,.10,Inf), labels=c("***","**","*","")),
         sig_rob=cut(p_rob, c(-Inf,.01,.05,.10,Inf), labels=c("***","**","*",""))) |>
  select(filter, outcome, tau_conv, se_conv, p_conv, sig_conv, tau_bc, se_rob, p_rob, sig_rob, p, q, h, n_eff, n_elec)
write_csv(t7_out, paste0(OUT, "T7_rdrobust_10DV.csv"))
cat(sprintf("Saved T7_rdrobust_10DV.csv (%d rows)\n", nrow(t7_out)))

# WLB summary
cat("\n=== T7 WLB ===\n")
print(t7_out[t7_out$outcome=="wlb", c("filter","tau_conv","p_conv","tau_bc","p_rob","n_eff","n_elec")], n=20)

# ═══════════════════════════════════════════════════════════════════════════
# T8: Firm-quarter dynamic
# ═══════════════════════════════════════════════════════════════════════════
cat("\n=== T8: Firm-quarter dynamic ===\n")

# Map event_time_month to quarters
cur$event_q <- pmax(-3, pmin(3, floor(cur$event_time_month / 3)))
cur_clamped <- cur[cur$event_q >= -3 & cur$event_q <= 3, ]
cur_clamped$event_q <- factor(cur_clamped$event_q, levels=as.character(-3:3))

t8_rows <- list()
for(t_iter in THR) {
  cur_wlb <- cur_clamped[!is.na(cur_clamped$wlb), ]
  eids <- elig(cur_wlb, t_iter$type, t_iter$N)
  ne <- length(eids); if(ne < 20) next
  d <- cur_clamped[cur_clamped$election_id %in% eids, ]

  for(dv in DV10) {
    d_dv <- d[!is.na(d[[dv]]), ]
    agg <- d_dv |> group_by(election_id, gvkey, win, event_q) |>
      summarise(ybar=mean(.data[[dv]], na.rm=TRUE), margin=first(margin), n=n(), .groups="drop") |>
      mutate(post=as.integer(as.integer(as.character(event_q)) >= 0), win_post=win*post)
    if(nrow(agg) < 30) next

    # Pre-trend
    fd <- tryCatch(feols(ybar ~ i(event_q, win, ref='-1') + win + post:margin | gvkey + event_q,
                         data=agg, weights=~n, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e) NULL)
    pre <- tryCatch(fixest::wald(fd, "event_q::(-3|-2):win", print=FALSE)$p, error=function(e) NA)

    # Pooled
    fp2 <- tryCatch(feols(ybar ~ win + post + win_post + post:margin | gvkey + event_q,
                          data=agg, weights=~n, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e) NULL)
    v <- if(is.null(fp2) || !("win_post" %in% rownames(coeftable(fp2)))) c(NA,NA,NA)
         else coeftable(fp2)["win_post", c("Estimate","Std. Error","Pr(>|t|)")]

    t8_rows[[length(t8_rows)+1]] <- tibble(table="T8_fq", filter=thr_label(t_iter), outcome=dv,
      pretrend_p=pre, pooled_est=v[1], pooled_se=v[2], pooled_p=v[3], n_events=ne)
  }
  cat(sprintf("  %s: %d elections\n", thr_label(t_iter), ne))
}

t8_out <- bind_rows(t8_rows)
write_csv(t8_out, paste0(OUT, "T8_firmquarter_dynamic_10DV.csv"))
cat(sprintf("Saved T8_firmquarter_dynamic_10DV.csv (%d rows)\n", nrow(t8_out)))

# WLB summary
cat("\n=== T8 WLB ===\n")
print(t8_out[t8_out$outcome=="wlb", c("filter","pretrend_p","pooled_est","pooled_p","n_events")], n=20)

cat("\nDone.\n")
