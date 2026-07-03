#!/usr/bin/env Rscript
# Firm-Year Unionization x Glassdoor — FE Ladder Regressions
library(fixest); library(dplyr); library(tidyr); library(readr); library(nanoparquet)
options(fixest_notes=FALSE); setFixest_notes(FALSE)

OUT <- "/data/disk4/workspace/projects/union_glassdoor/outputs/rdd_rebuild/firmyear_unionization/finished_panel/"

cat("Loading merged panel...\n")
df <- nanoparquet::read_parquet(paste0(OUT,"merged_panel_main.parquet"))
cat(sprintf("Rows: %d, gvkeys: %d\n", nrow(df), length(unique(df$gvkey))))

# Prepare
df <- df |> mutate(
  gvkey = as.character(gvkey),
  review_year = as.integer(review_year),
  UNIONIZATION_raw = ifelse(is.na(UNIONIZATION_raw), 0, UNIONIZATION_raw),
  UNIONIZATION = ifelse(is.na(UNIONIZATION), 0, UNIONIZATION),
  UNIONIZATION_cap1 = pmin(UNIONIZATION_raw, 1),
  UNIONIZATION_binary = ifelse(UNIONIZATION > 0, 1, 0),
  sic2 = substr(as.character(sic), 1, 2)
)

# 10 main DVs (all reviews)
DV_ALL <- c("overall_rating","career_opp","comp_benefit","senior_mgmt","wlb","culture",
            "recommend","business_outlook","ceo_approval","diversity")
# 10 current-only DVs
DV_CUR <- c("overall_rating_cur","career_opp_cur","comp_benefit_cur","senior_mgmt_cur",
            "wlb_cur","culture_cur","recommend_cur","business_outlook_cur","ceo_approval_cur","diversity_cur")

# ====== 1. Correlations ======
cat("\n=== Correlations (all reviews, UNIONIZATION_cap1) ===\n")
cor_rows <- list()
for (y in DV_ALL) {
  ok <- !is.na(df[[y]]) & !is.na(df$UNIONIZATION_cap1)
  if (sum(ok) < 20) next
  pear <- cor(df[[y]][ok], df$UNIONIZATION_cap1[ok], method="pearson")
  spear <- cor(df[[y]][ok], df$UNIONIZATION_cap1[ok], method="spearman")
  # Quintile means
  quints <- quantile(df$UNIONIZATION_cap1[ok], probs=seq(0,1,0.2), na.rm=TRUE)
  # >0 vs =0
  gt0 <- mean(df[[y]][ok & df$UNIONIZATION_cap1 > 0], na.rm=TRUE)
  eq0 <- mean(df[[y]][ok & df$UNIONIZATION_cap1 == 0], na.rm=TRUE)
  cor_rows[[length(cor_rows)+1]] <- data.frame(
    outcome=y, pearson=round(pear,4), spearman=round(spear,4),
    mean_gt0=round(gt0,4), mean_eq0=round(eq0,4), n=sum(ok),
    stringsAsFactors=FALSE)
}
cor_df <- bind_rows(cor_rows)
write_csv(cor_df, paste0(OUT,"finished_correlations.csv"))
cat("Correlations saved.\n")
for (i in 1:nrow(cor_df)) cat(sprintf("  %-20s pearson=% 7.4f spearman=% 7.4f gt0=%.4f eq0=%.4f n=%d\n",
  cor_df$outcome[i], cor_df$pearson[i], cor_df$spearman[i], cor_df$mean_gt0[i], cor_df$mean_eq0[i], cor_df$n[i]))

# ====== 2. FE Ladder (main: UNIONIZATION_cap1, t+1 LHS) ======
cat("\n=== FE Ladder Regressions ===\n")

# Create t+1 LHS: lead outcomes by 1 year within gvkey
df <- df |> arrange(gvkey, review_year) |> group_by(gvkey) |>
  mutate(
    across(all_of(c(DV_ALL, DV_CUR)), ~ dplyr::lead(.x, 1), .names = "{.col}_lead1")
  ) |> ungroup()

run_ladder <- function(y_var, d, label) {
  # y_var should be a lead variable
  # L1: gvkey + sic2^year
  f_l1 <- as.formula(paste0(y_var, " ~ UNIONIZATION_cap1 | gvkey + sic2^review_year"))
  # L2: gvkey + year
  f_l2 <- as.formula(paste0(y_var, " ~ UNIONIZATION_cap1 | gvkey + review_year"))
  # L3: sic2^year only (no firm FE)
  f_l3 <- as.formula(paste0(y_var, " ~ UNIONIZATION_cap1 | sic2^review_year"))
  # L4: year only (pooled)
  f_l4 <- as.formula(paste0(y_var, " ~ UNIONIZATION_cap1 | review_year"))

  rows <- list()
  for (li in c("L1","L2","L3","L4")) {
    f <- get(paste0("f_", tolower(li)))
    fit <- tryCatch(feols(f, data=d, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e)NULL)
    if (is.null(fit)) {
      rows[[length(rows)+1]] <- data.frame(model=li, dv=y_var,
        coef=NA, se=NA, pvalue=NA, n_obs=NA, n_firms=NA, dropped=NA, stringsAsFactors=FALSE)
      next
    }
    ct <- coeftable(fit)
    r <- ct["UNIONIZATION_cap1", ]
    # Check if UNIONIZATION was dropped
    dropped_info <- ""
    if (is.na(r["Estimate"])) dropped_info <- "coefficient NA - possibly dropped"
    rows[[length(rows)+1]] <- data.frame(
      model=li, dv=y_var,
      coef=r["Estimate"], se=r["Std. Error"], pvalue=r["Pr(>|t|)"],
      n_obs=nobs(fit), n_firms=length(unique(d$gvkey[!is.na(d[[y_var]])])),
      dropped=dropped_info, stringsAsFactors=FALSE)
  }
  bind_rows(rows)
}

# Run for all DVs with lead
DV_LEAD <- paste0(DV_ALL, "_lead1")
ladder_rows <- list()
for (y in DV_LEAD) {
  cat(sprintf("  %s...\n", y))
  ladder_rows[[length(ladder_rows)+1]] <- run_ladder(y, df, "main")
}
ladder <- bind_rows(ladder_rows)
write_csv(ladder, paste0(OUT,"finished_reg_ladder.csv"))
cat("\n=== FE Ladder Results (key rows) ===\n")
ladder_ok <- ladder[!is.na(ladder$coef), ]
for (i in 1:min(nrow(ladder_ok), 60)) cat(sprintf("  %-4s %-30s coef=% 8.4f se=%7.4f p=%6.4f n=%d\n",
  ladder_ok$model[i], ladder_ok$dv[i], ladder_ok$coef[i], ladder_ok$se[i], ladder_ok$pvalue[i], ladder_ok$n_obs[i]))

# ====== 3. With controls ======
cat("\n=== With Controls (L1-L4) ===\n")
# Controls: SIZE (log at), LEVERAGE ((dlc+dltt)/at), CAPEX (capx/at), EBITDA (ebitda/at), SGA (xsga/sale), NOLCF (tlcf/at > 0), LOSS (ib < 0)
# Winsorize at 1/99
winsor <- function(x, p=0.01) {
  q <- quantile(x, c(p, 1-p), na.rm=TRUE)
  pmax(q[1], pmin(q[2], x))
}

controls_str <- "+ log(SIZE) + LEVERAGE + CAPEX + EBITDA + SGA + NOLCF + LOSS"

# Add controls to data
df <- df |> mutate(
  SIZE = winsor(at, 0.01),
  LEVERAGE = winsor((dlc + dltt) / at, 0.01),
  CAPEX = winsor(capx / at, 0.01),
  EBITDA = winsor(ebitda / at, 0.01),
  SGA = winsor(xsga / sale, 0.01),
  NOLCF = ifelse(tlcf / at > 0, 1, 0),
  LOSS = ifelse(ib < 0, 1, 0)
)

run_with_ctrls <- function(y_var, d) {
  rows <- list()
  for (li in c("L1","L2","L3","L4")) {
    if (li == "L1") {
      f <- as.formula(paste0(y_var, " ~ UNIONIZATION_cap1 + SIZE + LEVERAGE + CAPEX + EBITDA + SGA + NOLCF + LOSS | gvkey + sic2^review_year"))
    } else if (li == "L2") {
      f <- as.formula(paste0(y_var, " ~ UNIONIZATION_cap1 + SIZE + LEVERAGE + CAPEX + EBITDA + SGA + NOLCF + LOSS | gvkey + review_year"))
    } else if (li == "L3") {
      f <- as.formula(paste0(y_var, " ~ UNIONIZATION_cap1 + SIZE + LEVERAGE + CAPEX + EBITDA + SGA + NOLCF + LOSS | sic2^review_year"))
    } else {
      f <- as.formula(paste0(y_var, " ~ UNIONIZATION_cap1 + SIZE + LEVERAGE + CAPEX + EBITDA + SGA + NOLCF + LOSS | review_year"))
    }
    fit <- tryCatch(feols(f, data=d, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e)NULL)
    if (is.null(fit)) {
      rows[[length(rows)+1]] <- data.frame(model=li, dv=y_var, coef=NA, se=NA, pvalue=NA, n_obs=NA, n_firms=NA, dropped=NA, stringsAsFactors=FALSE)
      next
    }
    ct <- coeftable(fit)
    r <- tryCatch(ct["UNIONIZATION_cap1", ], error=function(e)NULL)
    if (is.null(r) || is.na(r["Estimate"])) {
      rows[[length(rows)+1]] <- data.frame(model=li, dv=y_var, coef=NA, se=NA, pvalue=NA, n_obs=nobs(fit), n_firms=length(unique(d$gvkey)), dropped="coefficient dropped/NA", stringsAsFactors=FALSE)
      next
    }
    rows[[length(rows)+1]] <- data.frame(model=li, dv=y_var, coef=r["Estimate"], se=r["Std. Error"], pvalue=r["Pr(>|t|)"], n_obs=nobs(fit), n_firms=length(unique(d$gvkey)), dropped="", stringsAsFactors=FALSE)
  }
  bind_rows(rows)
}

ctrl_rows <- list()
for (y in DV_LEAD[1:5]) {  # Main 5: overall, career, comp, senior, wlb
  cat(sprintf("  %s + controls...\n", y))
  ctrl_rows[[length(ctrl_rows)+1]] <- run_with_ctrls(y, df)
}
ctrls <- bind_rows(ctrl_rows)
write_csv(ctrls, paste0(OUT,"finished_reg_controls.csv"))
cat("\n=== Controls Results ===\n")
for (i in 1:nrow(ctrls)) cat(sprintf("  %-4s %-30s coef=% 8.4f se=%7.4f p=%6.4f n=%d\n",
  ctrls$model[i], ctrls$dv[i], ctrls$coef[i], ctrls$se[i], ctrls$pvalue[i], ctrls$n_obs[i]))

# ====== 4. Robustness ======
cat("\n=== Robustness ===\n")
rob_rows <- list()
# (a) binary UNIONIZATION, L2 (LHS lead1)
for (y in DV_LEAD[1:6]) {
  f <- as.formula(paste0(y, " ~ UNIONIZATION_binary | gvkey + review_year"))
  fit <- tryCatch(feols(f, data=df, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e)NULL)
  if (!is.null(fit)) {
    r <- coeftable(fit)["UNIONIZATION_binary", ]
    rob_rows[[length(rob_rows)+1]] <- data.frame(spec="binary_L2", dv=y, coef=r["Estimate"], se=r["Std. Error"], pvalue=r["Pr(>|t|)"], n_obs=nobs(fit), stringsAsFactors=FALSE)
  }
}
# (b) raw UNIONIZATION (not capped), L2
for (y in DV_LEAD[1:6]) {
  f <- as.formula(paste0(y, " ~ UNIONIZATION_raw | gvkey + review_year"))
  fit <- tryCatch(feols(f, data=df, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e)NULL)
  if (!is.null(fit)) {
    r <- coeftable(fit)["UNIONIZATION_raw", ]
    rob_rows[[length(rob_rows)+1]] <- data.frame(spec="raw_L2", dv=y, coef=r["Estimate"], se=r["Std. Error"], pvalue=r["Pr(>|t|)"], n_obs=nobs(fit), stringsAsFactors=FALSE)
  }
}
# (c) current-only LHS, L2
DV_CUR_LEAD <- paste0(DV_CUR[1:6], "_lead1")
for (y in DV_CUR_LEAD) {
  f <- as.formula(paste0(y, " ~ UNIONIZATION_cap1 | gvkey + review_year"))
  fit <- tryCatch(feols(f, data=df, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e)NULL)
  if (!is.null(fit)) {
    r <- coeftable(fit)["UNIONIZATION_cap1", ]
    rob_rows[[length(rob_rows)+1]] <- data.frame(spec="current_L2", dv=y, coef=r["Estimate"], se=r["Std. Error"], pvalue=r["Pr(>|t|)"], n_obs=nobs(fit), stringsAsFactors=FALSE)
  }
}
# (d) same-year (not lead)
for (y in DV_ALL[1:6]) {
  f <- as.formula(paste0(y, " ~ UNIONIZATION_cap1 | gvkey + review_year"))
  fit <- tryCatch(feols(f, data=df, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e)NULL)
  if (!is.null(fit)) {
    r <- coeftable(fit)["UNIONIZATION_cap1", ]
    rob_rows[[length(rob_rows)+1]] <- data.frame(spec="contemporaneous_L2", dv=y, coef=r["Estimate"], se=r["Std. Error"], pvalue=r["Pr(>|t|)"], n_obs=nobs(fit), stringsAsFactors=FALSE)
  }
}
rob <- bind_rows(rob_rows)
write_csv(rob, paste0(OUT,"finished_reg_robustness.csv"))
cat("\n=== Robustness Results ===\n")
for (i in 1:nrow(rob)) cat(sprintf("  %-25s %-30s coef=% 8.4f se=%7.4f p=%6.4f n=%d\n",
  rob$spec[i], rob$dv[i], rob$coef[i], rob$se[i], rob$pvalue[i], rob$n_obs[i]))

cat("\nSTEP 4 done.\n")
