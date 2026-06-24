#!/usr/bin/env Rscript
# STEP 11: Win×Post×UnitShare interaction regressions
# Outputs: unit_share_regression_results.csv, unit_share_marginal_effects.csv

suppressMessages({library(fixest); library(dplyr); library(tidyr); library(nanoparquet); library(readr)})

OUT <- "/data/disk4/workspace/projects/union_glassdoor/outputs/rdd_rebuild/current_former_bargaining_unit/20260624/"

df <- read_parquet(paste0(OUT, "enriched_sample.parquet"))
df$sample_type <- ifelse(df$is_current_employee == 1, "current",
                  ifelse(df$is_former_employee == 1, "former", "unknown"))

# Load unit_share
ushare <- read_csv(paste0(OUT, "unit_share_election_data.csv"), show_col_types=FALSE)
df <- df |> left_join(ushare |> select(election_id, unit_share_winsor, unit_share_capped, unit_size, emp, emp_actual),
                      by="election_id")

# Use winsorized unit_share, replace NA with 0
df$unit_share <- df$unit_share_winsor
df$unit_share[is.na(df$unit_share)] <- NA

# Create derived variables
df$log_unit_size <- log(1 + df$unit_size)
df$unit_share_above_med <- as.integer(df$unit_share > quantile(df$unit_share, 0.5, na.rm=TRUE))
df$unit_share_tercile <- cut(df$unit_share,
  breaks=quantile(df$unit_share, c(0, 1/3, 2/3, 1), na.rm=TRUE),
  labels=c("low","mid","high"), include.lowest=TRUE)

cat(sprintf("unit_share distribution:\n"))
cat(sprintf("  mean=%.6f, median=%.6f\n", mean(df$unit_share, na.rm=TRUE), median(df$unit_share, na.rm=TRUE)))
cat(sprintf("  P25=%.6f, P75=%.6f, P90=%.6f\n",
            quantile(df$unit_share, 0.25, na.rm=TRUE),
            quantile(df$unit_share, 0.75, na.rm=TRUE),
            quantile(df$unit_share, 0.90, na.rm=TRUE)))
cat(sprintf("  above_med: %d\n", sum(df$unit_share_above_med==1, na.rm=TRUE)))
cat(sprintf("  tercile: low=%d, mid=%d, high=%d\n",
            sum(df$unit_share_tercile=="low", na.rm=TRUE),
            sum(df$unit_share_tercile=="mid", na.rm=TRUE),
            sum(df$unit_share_tercile=="high", na.rm=TRUE)))

DV10 <- c("overall_rating", "career_opp", "comp_benefit", "senior_mgmt", "wlb", "culture",
          "recommend", "business_outlook", "ceo_approval", "diversity")

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

prep2 <- function(d) {
  d <- prep(d)
  top50 <- d |> filter(!is.na(role_k1500)) |> count(role_k1500,sort=TRUE) |> slice_head(n=50) |> pull(role_k1500)
  d |> mutate(role_clean=case_when(is.na(role_k1500)~"Missing_role", role_k1500%in%top50~role_k1500, TRUE~"Other_role"))
}

# ─── A. Continuous unit_share interaction ────────────────────────────────
cat("\n=== A. Continuous unit_share interaction ===\n")

v7c_ushare <- function(y) as.formula(paste0(y,
  " ~ win + post + win_post + unit_share + post:unit_share + win_post:unit_share + post:margin + emp_status + seniority_f | gvkey + review_year + state_clean + role_clean"))

cur <- df[df$sample_type == "current", ]
results_cont <- list()
results_marg <- list()

for (dv in DV10) {
  cat(sprintf("  %s:\n", dv))
  cur_dv <- cur[!is.na(cur[[dv]]) & !is.na(cur$unit_share), ]
  # total>=10 filter
  eids <- cur_dv |> group_by(election_id) |> summarise(n=n(), .groups="drop") |> filter(n>=10) |> pull(election_id)
  sub <- cur_dv[cur_dv$election_id %in% eids, ]
  if (nrow(sub) < 100) next
  sub <- prep2(sub)

  fit <- tryCatch(feols(v7c_ushare(dv), data=sub, cluster=~gvkey+review_year), error=function(e)NULL)
  if (is.null(fit)) {
    results_cont[[length(results_cont)+1]] <- data.frame(dv=dv, note="model_failed")
    next
  }

  ct <- coeftable(fit)
  get_vals <- function(nm) {
    if (nm %in% rownames(ct)) c(ct[nm,"Estimate"], ct[nm,"Std. Error"], ct[nm,"Pr(>|t|)"]) else c(NA,NA,NA)
  }
  wp <- get_vals("win_post")
  wpu <- get_vals("win_post:unit_share")

  results_cont[[length(results_cont)+1]] <- data.frame(
    dv=dv,
    win_post_coef=wp[1], win_post_se=wp[2], win_post_p=wp[3],
    win_post_x_ushare_coef=wpu[1], win_post_x_ushare_se=wpu[2], win_post_x_ushare_p=wpu[3],
    n_reviews=nrow(sub), n_elections=length(unique(sub$election_id)),
    median_ushare=median(sub$unit_share, na.rm=TRUE))

  # Marginal effects at P25/median/P75
  for (pct_name in c("P25","P50","P75")) {
    qval <- quantile(sub$unit_share, switch(pct_name, P25=0.25, P50=0.5, P75=0.75), na.rm=TRUE)
    me <- wp[1] + wpu[1] * qval
    results_marg[[length(results_marg)+1]] <- data.frame(
      dv=dv, percentile=pct_name, unit_share=qval, marginal_effect=me, spec="continuous")
  }
}

cont_df <- bind_rows(results_cont)
write_csv(cont_df, paste0(OUT, "unit_share_regression_continuous.csv"))
cat(sprintf("Saved continuous results (%d rows)\n", nrow(cont_df)))

# ─── B. Above/below median interaction ───────────────────────────────────
cat("\n=== B. Above/below median unit_share interaction ===\n")

v7c_above <- function(y) as.formula(paste0(y,
  " ~ win + post + win_post + unit_share_above_med + post:unit_share_above_med + win_post:unit_share_above_med + post:margin + emp_status + seniority_f | gvkey + review_year + state_clean + role_clean"))

results_above <- list()

for (dv in DV10) {
  cat(sprintf("  %s:", dv))
  cur_dv <- cur[!is.na(cur[[dv]]) & !is.na(cur$unit_share_above_med), ]
  eids <- cur_dv |> group_by(election_id) |> summarise(n=n(), .groups="drop") |> filter(n>=10) |> pull(election_id)
  sub <- cur_dv[cur_dv$election_id %in% eids, ]
  if (nrow(sub) < 100) { cat(" insufficient\n"); next }
  sub <- prep2(sub)

  fit <- tryCatch(feols(v7c_above(dv), data=sub, cluster=~gvkey+review_year), error=function(e)NULL)
  if (is.null(fit)) { cat(" model_failed\n"); next }

  ct <- coeftable(fit)
  get_vals <- function(nm) {
    if (nm %in% rownames(ct)) c(ct[nm,"Estimate"], ct[nm,"Std. Error"], ct[nm,"Pr(>|t|)"]) else c(NA,NA,NA)
  }
  wp <- get_vals("win_post")
  wpa <- get_vals("win_post:unit_share_above_med")

  cat(sprintf(" below=%.4f(p=%.4f), above_diff=%.4f(p=%.4f)\n", wp[1], wp[3], wpa[1], wpa[3]))

  results_above[[length(results_above)+1]] <- data.frame(
    dv=dv, win_post_below_med=wp[1], win_post_diff_above=wpa[1], diff_p=wpa[3],
    n_reviews=nrow(sub), n_elections=length(unique(sub$election_id)))
}

above_df <- bind_rows(results_above)
write_csv(above_df, paste0(OUT, "unit_share_regression_above_median.csv"))
cat(sprintf("Saved above-median results (%d rows)\n", nrow(above_df)))

# ─── C. Log(unit_size) interaction ───────────────────────────────────────
cat("\n=== C. Log(unit_size) interaction ===\n")

v7c_logus <- function(y) as.formula(paste0(y,
  " ~ win + post + win_post + log_unit_size + post:log_unit_size + win_post:log_unit_size + post:margin + emp_status + seniority_f | gvkey + review_year + state_clean + role_clean"))

results_log <- list()

for (dv in DV10) {
  cat(sprintf("  %s:", dv))
  cur_dv <- cur[!is.na(cur[[dv]]) & !is.na(cur$log_unit_size), ]
  eids <- cur_dv |> group_by(election_id) |> summarise(n=n(), .groups="drop") |> filter(n>=10) |> pull(election_id)
  sub <- cur_dv[cur_dv$election_id %in% eids, ]
  if (nrow(sub) < 100) { cat(" insufficient\n"); next }
  sub <- prep2(sub)

  fit <- tryCatch(feols(v7c_logus(dv), data=sub, cluster=~gvkey+review_year), error=function(e)NULL)
  if (is.null(fit)) { cat(" model_failed\n"); next }

  ct <- coeftable(fit)
  get_vals <- function(nm) {
    if (nm %in% rownames(ct)) c(ct[nm,"Estimate"], ct[nm,"Std. Error"], ct[nm,"Pr(>|t|)"]) else c(NA,NA,NA)
  }
  wp <- get_vals("win_post")
  wpl <- get_vals("win_post:log_unit_size")

  cat(sprintf(" wp=%.4f(p=%.4f), wp:log_us=%.4f(p=%.4f)\n", wp[1], wp[3], wpl[1], wpl[3]))

  results_log[[length(results_log)+1]] <- data.frame(
    dv=dv, win_post_coef=wp[1], win_post_se=wp[2], win_post_p=wp[3],
    win_post_x_log_us_coef=wpl[1], win_post_x_log_us_se=wpl[2], win_post_x_log_us_p=wpl[3],
    n_reviews=nrow(sub), n_elections=length(unique(sub$election_id)))
}

log_df <- bind_rows(results_log)
write_csv(log_df, paste0(OUT, "unit_share_regression_log_size.csv"))
cat(sprintf("Saved log(unit_size) results (%d rows)\n", nrow(log_df)))

# ─── Save marginal effects ──────────────────────────────────────────────
marg_df <- bind_rows(results_marg)
write_csv(marg_df, paste0(OUT, "unit_share_marginal_effects.csv"))
cat(sprintf("Saved marginal effects (%d rows)\n", nrow(marg_df)))

# ─── Quick summaries ────────────────────────────────────────────────────
cat("\n=== WLB: Continuous unit_share ===\n")
print(cont_df[cont_df$dv=="wlb",])

cat("\n=== WLB: Above/below median ===\n")
print(above_df[above_df$dv=="wlb",])

cat("\n=== WLB: Log(unit_size) ===\n")
print(log_df[log_df$dv=="wlb",])

cat("\nDone.\n")
