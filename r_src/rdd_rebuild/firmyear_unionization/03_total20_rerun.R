#!/usr/bin/env Rscript
# Part B: total>=20 rerun — T3 event study, T5 bandwidth, T7 rdrobust, T8 firm-quarter dynamic
# All 10 DVs, current-only, v7c 4-FE (state_clean+role_clean in FEs)

suppressMessages({
  library(fixest); library(dplyr); library(tidyr); library(nanoparquet); library(readr); library(purrr)
  library(rdrobust)
})

OUT <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260702/firmyear_unionization/"

DV10 <- c("overall_rating","career_opp","comp_benefit","senior_mgmt","wlb","culture",
          "recommend","business_outlook","ceo_approval","diversity")

cat("Loading enriched sample...\n")
df <- read_parquet("outputs/20260624/current_former_bargaining_unit/enriched_sample.parquet")
cur <- df[df$is_current_employee == 1, ]

prep <- function(d){
  d |> mutate(
    gvkey=as.character(gvkey), review_year=as.integer(review_year),
    win=as.integer(win), post=as.integer(post), margin=as.numeric(margin), win_post=win*post,
    emp_status=case_when(
      is.na(reviewer_employment_status)~"unknown",
      reviewer_employment_status=="REGULAR"~"regular",
      reviewer_employment_status=="PART_TIME"~"part_time",
      reviewer_employment_status=="INTERN"~"intern",
      reviewer_employment_status=="CONTRACT"~"contract", TRUE~"other") |>
      factor(levels=c("regular","part_time","intern","contract","other","unknown")),
    seniority_f=factor(ifelse(is.na(seniority),0L,as.integer(seniority))),
    state_clean=case_when(!is.na(is_us_review)&is_us_review==1~state_y, TRUE~"Non_US") |> replace_na("Non_US"))
}

prep2 <- function(d){
  d <- prep(d)
  top50 <- d |> filter(!is.na(role_k1500)) |> count(role_k1500,sort=TRUE) |> slice_head(n=50) |> pull(role_k1500)
  d |> mutate(role_clean=case_when(is.na(role_k1500)~"Missing_role", role_k1500%in%top50~role_k1500, TRUE~"Other_role"))
}

cur <- prep2(cur)
cat(sprintf("Current prepped: %d rows\n", nrow(cur)))

# Total>=20 filter (based on wlb)
cur_wlb <- cur[!is.na(cur$wlb), ]
eids_t20 <- cur_wlb |> group_by(election_id) |> summarise(n=n(), .groups="drop") |> filter(n>=20) |> pull(election_id)
cat(sprintf("total>=20: %d elections\n", length(eids_t20)))

v7c <- function(y) as.formula(paste0(y," ~ win+post+win_post+post:margin+emp_status+seniority_f | gvkey+review_year+state_clean+role_clean"))
CL <- ~gvkey + review_year
safe <- function(m, r, c=1) { out <- tryCatch(m[r,c], error=function(e) NA); if(is.null(out)) NA else out }

# ═══════════════════════════════════════════════════════════════════════════
# T2: Baseline (consistency check) — already have from 06-24, re-run here
# ═══════════════════════════════════════════════════════════════════════════
cat("\n=== T2 Baseline total>=20 ===\n")
t2_rows <- list()
for (dv in DV10) {
  d <- cur[cur$election_id %in% eids_t20 & !is.na(cur[[dv]]), ]
  if (nrow(d) < 100) next
  fit <- tryCatch(feols(v7c(dv), data=d, cluster=CL, warn=FALSE, notes=FALSE), error=function(e) NULL)
  if (is.null(fit) || !("win_post" %in% rownames(coeftable(fit)))) next
  ct <- coeftable(fit)
  pre_mean <- mean(d[[dv]][d$post==0], na.rm=TRUE)
  t2_rows[[length(t2_rows)+1]] <- tibble(table="T2", filter="total>=20", outcome=dv,
    coef=ct["win_post","Estimate"], se=ct["win_post","Std. Error"], p=ct["win_post","Pr(>|t|)"],
    n_reviews=nrow(d), n_elections=length(unique(d$election_id)), pre_mean=pre_mean)
}
t2_out <- bind_rows(t2_rows)

# ─── T5 Bandwidth ──────────────────────────────────────────────────────────
cat("\n=== T5 Bandwidth total>=20 ===\n")
bws <- c(1.0, 0.20, 0.10, 0.05)
t5_rows <- list()
for (dv in DV10) {
  d <- cur[cur$election_id %in% eids_t20 & !is.na(cur[[dv]]), ]
  for (bw in bws) {
    d_bw <- if(bw < 1.0) d[abs(d$margin) <= bw, ] else d
    if (nrow(d_bw) < 50) next
    fit <- tryCatch(feols(v7c(dv), data=d_bw, cluster=CL, warn=FALSE, notes=FALSE), error=function(e) NULL)
    if (is.null(fit) || !("win_post" %in% rownames(coeftable(fit)))) next
    ct <- coeftable(fit)
    t5_rows[[length(t5_rows)+1]] <- tibble(table="T5", filter="total>=20", outcome=dv, bandwidth=bw,
      coef=ct["win_post","Estimate"], se=ct["win_post","Std. Error"], p=ct["win_post","Pr(>|t|)"],
      n_reviews=nrow(d_bw), n_elections=length(unique(d_bw$election_id)))
  }
}
t5_out <- bind_rows(t5_rows)
write_csv(t5_out, paste0(OUT, "t20_T5_bandwidth_10DV.csv"))
cat(sprintf("T5: %d rows\n", nrow(t5_out)))

# ─── T7 rdrobust ───────────────────────────────────────────────────────────
cat("\n=== T7 rdrobust total>=20 ===\n")
run_rd <- function(d, dv) {
  agg <- d |> group_by(election_id) |>
    summarise(pre=mean(.data[[dv]][post==0], na.rm=TRUE), postm=mean(.data[[dv]][post==1], na.rm=TRUE),
              margin=first(margin), .groups="drop") |> mutate(delta=postm-pre) |>
    filter(is.finite(delta), is.finite(margin))
  if(nrow(agg)<20) return(tibble(tau_conv=NA,se_conv=NA,p_conv=NA,tau_bc=NA,se_rob=NA,p_rob=NA,h=NA,n_eff=NA,n_elec=nrow(agg)))
  rr <- tryCatch(rdrobust(y=agg$delta, x=agg$margin, c=0, kernel="triangular", p=2, q=3, bwselect="mserd"), error=function(e)NULL)
  if(is.null(rr)) return(tibble(tau_conv=NA,se_conv=NA,p_conv=NA,tau_bc=NA,se_rob=NA,p_rob=NA,h=NA,n_eff=NA,n_elec=nrow(agg)))
  tibble(tau_conv=safe(rr$coef,"Conventional"), se_conv=safe(rr$se,"Conventional"), p_conv=safe(rr$pv,"Conventional"),
         tau_bc=safe(rr$coef,"Bias-Corrected"), se_rob=safe(rr$se,"Robust"), p_rob=safe(rr$pv,"Robust"),
         p=2L, q=3L, h=rr$bws["h","left"], n_eff=sum(rr$N_h), n_elec=nrow(agg))
}
t7_rows <- list()
for (dv in DV10) {
  d <- cur[cur$election_id %in% eids_t20 & !is.na(cur[[dv]]), ]
  if(n_distinct(d$election_id) < 20) next
  r <- run_rd(d, dv); r$filter <- "total>=20"; r$outcome <- dv
  t7_rows[[length(t7_rows)+1]] <- r
}
t7_out <- bind_rows(t7_rows) |> mutate(sig_conv=cut(p_conv, c(-Inf,.01,.05,.10,Inf), labels=c("***","**","*","")),
  sig_rob=cut(p_rob, c(-Inf,.01,.05,.10,Inf), labels=c("***","**","*",""))) |>
  select(filter,outcome,tau_conv,se_conv,p_conv,sig_conv,tau_bc,se_rob,p_rob,sig_rob,p,q,h,n_eff,n_elec)
write_csv(t7_out, paste0(OUT, "t20_T7_rdrobust_10DV.csv"))
cat(sprintf("T7: %d rows\n", nrow(t7_out)))

# ─── T8 Firm-quarter dynamic ───────────────────────────────────────────────
cat("\n=== T8 Firm-quarter dynamic total>=20 ===\n")
cur$event_q <- pmax(-3, pmin(3, floor(cur$event_time_month / 3)))
cur_clamped <- cur[cur$event_q >= -3 & cur$event_q <= 3, ]
cur_clamped$event_q <- factor(cur_clamped$event_q, levels=as.character(-3:3))
cur_wlb2 <- cur_clamped[!is.na(cur_clamped$wlb), ]
eids_t20q <- cur_wlb2 |> group_by(election_id) |> summarise(n=n(),.groups="drop") |> filter(n>=20) |> pull(election_id)

t8_rows <- list()
for (dv in DV10) {
  d <- cur_clamped[cur_clamped$election_id %in% eids_t20q & !is.na(cur_clamped[[dv]]), ]
  agg <- d |> group_by(election_id, gvkey, win, event_q) |>
    summarise(ybar=mean(.data[[dv]], na.rm=TRUE), margin=first(margin), n=n(), .groups="drop") |>
    mutate(post=as.integer(as.integer(as.character(event_q))>=0), win_post=win*post)
  if(nrow(agg)<30) next

  fd <- tryCatch(feols(ybar ~ i(event_q, win, ref='-1') + win + post:margin | gvkey + event_q,
                       data=agg, weights=~n, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e)NULL)
  pre <- tryCatch(fixest::wald(fd, "event_q::(-3|-2):win", print=FALSE)$p, error=function(e)NA)
  fp2 <- tryCatch(feols(ybar ~ win + post + win_post + post:margin | gvkey + event_q,
                        data=agg, weights=~n, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e)NULL)
  v <- if(is.null(fp2)||!("win_post"%in%rownames(coeftable(fp2)))) c(NA,NA,NA)
       else coeftable(fp2)["win_post", c("Estimate","Std. Error","Pr(>|t|)")]
  t8_rows[[length(t8_rows)+1]] <- tibble(table="T8_fq", filter="total>=20", outcome=dv,
    pretrend_p=pre, pooled_est=v[1], pooled_se=v[2], pooled_p=v[3], n_events=length(unique(agg$election_id)))
}
t8_out <- bind_rows(t8_rows)
write_csv(t8_out, paste0(OUT, "t20_T8_firmquarter_10DV.csv"))
cat(sprintf("T8: %d rows\n", nrow(t8_out)))

# ─── T3 Event study (conservative) ─────────────────────────────────────────
cat("\n=== T3 Event Study total>=20 ===\n")
es_fml <- function(y) as.formula(paste0(y,
  " ~ i(event_q,win,ref='-1') + win + post:margin + emp_status + seniority_f + state_clean + role_clean | gvkey + review_year + event_q"))
t3_rows <- list()
for (dv in DV10) {
  d <- cur_clamped[cur_clamped$election_id %in% eids_t20q & !is.na(cur_clamped[[dv]]), ]
  ne <- n_distinct(d$election_id)
  if(ne < 10) next
  f <- tryCatch(feols(es_fml(dv), d, cluster=CL, warn=FALSE, notes=FALSE), error=function(e)NULL)
  if(is.null(f)) next
  pre <- tryCatch(fixest::wald(f, "event_q::(-3|-2):win", print=FALSE)$p, error=function(e)NA)
  ct <- coeftable(f); rn <- grep("event_q::0:win", rownames(ct), value=TRUE)
  t0_est <- if(length(rn)) ct[rn[1],"Estimate"] else NA; t0_p <- if(length(rn)) ct[rn[1],"Pr(>|t|)"] else NA
  fp2 <- tryCatch(feols(v7c(dv), d, cluster=CL, warn=FALSE, notes=FALSE), error=function(e)NULL)
  v <- if(is.null(fp2)||!("win_post"%in%rownames(coeftable(fp2)))) c(NA,NA,NA)
       else coeftable(fp2)["win_post", c("Estimate","Std. Error","Pr(>|t|)")]
  t3_rows[[length(t3_rows)+1]] <- tibble(table="T3", filter="total>=20", outcome=dv,
    pretrend_p=pre, t0_est=t0_est, t0_p=t0_p,
    pooled_est=v[1], pooled_se=v[2], pooled_p=v[3], n_events=ne, n_reviews=nrow(d))
  cat(sprintf("  %s: pre=%.3f, t0=%.4f, pooled=%.4f(p=%.3f)\n", dv, pre, t0_est, v[1], v[3]))
}
t3_out <- bind_rows(t3_rows)
write_csv(t3_out, paste0(OUT, "t20_T3_eventstudy_10DV.csv"))
cat(sprintf("T3: %d rows\n", nrow(t3_out)))

# ─── STEP 5: 10 vs 20 comparison table ─────────────────────────────────────
cat("\n=== STEP 5: 10 vs 20 comparison ===\n")

# Load total>=10 results from 06-24
t10_t2 <- read_csv("outputs/20260624/current_former_bargaining_unit/current_former_all_outcomes.csv", show_col_types=FALSE) |>
  filter(sample=="current") |> select(outcome=dv, t10_coef=coef, t10_se=se, t10_p=p)
t10_t7 <- read_csv("outputs/20260624/current_former_bargaining_unit/T7_rdrobust_10DV.csv", show_col_types=FALSE) |>
  filter(filter=="total>=10") |> select(outcome, t10_tau_bc=tau_bc, t10_p_rob=p_rob)
t10_t8 <- read_csv("outputs/20260624/current_former_bargaining_unit/T8_firmquarter_dynamic_10DV.csv", show_col_types=FALSE) |>
  filter(filter=="total>=10") |> select(outcome, t10_pooled_est=pooled_est, t10_pooled_p=pooled_p)

t10_t3 <- read_csv("outputs/20260624/current_former_bargaining_unit/T3_eventstudy_10DV.csv", show_col_types=FALSE) |>
  filter(filter=="total>=10") |> select(outcome, t10_pretrend=pretrend_p, t10_pooled_est=pooled_est, t10_pooled_p=pooled_p)

# Merge T10 and T20 for T2
comp_t2 <- t2_out |> select(outcome, t20_coef=coef, t20_se=se, t20_p=p, t20_n=n_reviews, t20_elections=n_elections) |>
  left_join(t10_t2, by="outcome") |>
  mutate(d_coef=t20_coef-t10_coef, d_p=t20_p-t10_p, table="T2")

# T7
comp_t7 <- t7_out |> select(outcome, t20_tau_bc=tau_bc, t20_p_rob=p_rob) |>
  left_join(t10_t7, by="outcome") |> mutate(d_tau=t20_tau_bc-t10_tau_bc, d_p=t20_p_rob-t10_p_rob, table="T7")

# T8
comp_t8 <- t8_out |> select(outcome, t20_pooled_est=pooled_est, t20_pooled_p=pooled_p) |>
  left_join(t10_t8, by="outcome") |>
  mutate(d_est=t20_pooled_est-t10_pooled_est, d_p=t20_pooled_p-t10_pooled_p, table="T8")

# T3
comp_t3 <- t3_out |> select(outcome, t20_pretrend=pretrend_p, t20_pooled_est=pooled_est, t20_pooled_p=pooled_p) |>
  left_join(t10_t3, by="outcome") |>
  mutate(d_pooled=t20_pooled_est-t10_pooled_est, d_p=t20_pooled_p-t10_pooled_p, table="T3")

# Combine T2+T3+T7+T8
comp_all <- bind_rows(
  comp_t2 |> select(outcome, table, t10_c=everything(), t20_c=everything()),
  # Reshape into long format
  comp_t2 |> mutate(row_id=paste0(outcome,"_T2")) |> select(outcome, table, t20_coef, t20_p, t10_coef, t10_p, d_coef, d_p),
  comp_t7 |> mutate(row_id=paste0(outcome,"_T7")) |> select(outcome, table, t20_tau_bc, t20_p_rob, t10_tau_bc, t10_p_rob, d_tau, d_p),
  comp_t8 |> mutate(row_id=paste0(outcome,"_T8")) |> select(outcome, table, t20_pooled_est, t20_pooled_p, t10_pooled_est, t10_pooled_p, d_est, d_p),
  comp_t3 |> mutate(row_id=paste0(outcome,"_T3")) |> select(outcome, table, t20_pooled_est, t20_pooled_p, t10_pooled_est, t10_pooled_p, d_pooled, d_p)
)

# Simple comparison: key metrics per DV per table
comp_simple <- bind_rows(
  comp_t2 |> transmute(outcome, table="T2", t10=round(t10_coef,4), t20=round(t20_coef,4), t10_p=round(t10_p,4), t20_p=round(t20_p,4), diff=round(d_coef,4)),
  comp_t7 |> transmute(outcome, table="T7", t10=round(t10_tau_bc,4), t20=round(t20_tau_bc,4), t10_p=round(t10_p_rob,4), t20_p=round(t20_p_rob,4), diff=round(d_tau,4)),
  comp_t8 |> transmute(outcome, table="T8", t10=round(t10_pooled_est,4), t20=round(t20_pooled_est,4), t10_p=round(t10_pooled_p,4), t20_p=round(t20_pooled_p,4), diff=round(d_est,4)),
  comp_t3 |> transmute(outcome, table="T3", t10=round(t10_pooled_est,4), t20=round(t20_pooled_est,4), t10_p=round(t10_pooled_p,4), t20_p=round(t20_pooled_p,4), diff=round(d_pooled,4))
)
write_csv(comp_simple, paste0(OUT, "total10_vs_total20_comparison.csv"))
cat(sprintf("Saved comparison table (%d rows)\n", nrow(comp_simple)))

# Quick WLB comparison
cat("\n=== WLB: total10 vs total20 ===\n")
print(comp_simple[comp_simple$outcome=="wlb",])

cat("\nDone.\n")
