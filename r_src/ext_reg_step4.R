#!/usr/bin/env Rscript
# Extended panel STEP 4: L2+L4 regressions + controls + robustness
library(fixest); library(dplyr); library(tidyr); library(readr); library(nanoparquet)
options(fixest_notes=FALSE); setFixest_notes(FALSE)

OUT <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/unionization_extension/"

cat("Loading...\n")
df <- nanoparquet::read_parquet(paste0(OUT,"ext_merged_panel.parquet"))

df <- df |> mutate(
  gvkey = as.character(gvkey), review_year = as.integer(review_year),
  UNIONIZATION_cap1 = ifelse(is.na(UNIONIZATION), 0, UNIONIZATION),
  UNIONIZATION_binary = ifelse(UNIONIZATION_cap1 > 0, 1, 0))

winsor <- function(x, p=0.01) {
  q <- quantile(x, c(p, 1-p), na.rm=TRUE)
  pmax(q[1], pmin(q[2], x))
}

df <- df |> mutate(
  SIZE = winsor(at, 0.01), LEVERAGE = winsor((dlc + dltt) / at, 0.01),
  CAPEX = winsor(capx / at, 0.01), EBITDA = winsor(ebitda / at, 0.01),
  SGA = winsor(xsga / sale, 0.01), NOLCF = ifelse(tlcf / at > 0, 1, 0),
  LOSS = ifelse(ib < 0, 1, 0))

DV_ALL <- c("overall_rating","career_opp","comp_benefit","senior_mgmt","wlb","culture",
            "recommend","business_outlook","ceo_approval","diversity")
DV_CUR <- c("overall_rating_cur","career_opp_cur","comp_benefit_cur","senior_mgmt_cur",
            "wlb_cur","culture_cur","recommend_cur","business_outlook_cur","ceo_approval_cur","diversity_cur")

df <- df |> arrange(gvkey, review_year) |> group_by(gvkey) |>
  mutate(across(all_of(c(DV_ALL, DV_CUR)), ~ dplyr::lead(.x, 1), .names = "{.col}_lead1")) |> ungroup()

controls <- "+ SIZE + LEVERAGE + CAPEX + EBITDA + SGA + NOLCF + LOSS"
DV_LEAD <- paste0(DV_ALL, "_lead1")

# ====== L2 + L4 Ladder (skip L1/L3 to avoid segfault) ======
cat("\n=== L2 + L4 Ladder ===\n")
ladder_rows <- list()
for (y in DV_LEAD) {
  cat(sprintf("  %s...\n", y))
  for (li in c("L2","L4")) {
    f <- as.formula(paste0(y, " ~ UNIONIZATION_cap1 | ",
      ifelse(li=="L2", "gvkey + review_year", "review_year")))
    fit <- tryCatch(feols(f, data=df, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e)NULL)
    if (is.null(fit)) {
      ladder_rows[[length(ladder_rows)+1]] <- data.frame(model=li, dv=y, coef=NA, se=NA, pvalue=NA, n_obs=NA, n_firms=NA, dropped="error")
      next
    }
    r <- coeftable(fit)["UNIONIZATION_cap1",]
    ladder_rows[[length(ladder_rows)+1]] <- data.frame(model=li, dv=y, coef=r["Estimate"], se=r["Std. Error"], pvalue=r["Pr(>|t|)"], n_obs=nobs(fit), n_firms=length(unique(df$gvkey)), dropped="")
  }
}
ladder <- bind_rows(ladder_rows)
write_csv(ladder, paste0(OUT,"ext_reg_ladder.csv"))
cat("\nLadder:\n")
for (i in 1:nrow(ladder)) cat(sprintf("  %-4s %-30s coef=% 8.4f se=%7.4f p=%6.4f n=%d\n",
  ladder$model[i], ladder$dv[i], ladder$coef[i], ladder$se[i], ladder$pvalue[i], ladder$n_obs[i]))

# ====== L2 + L4 with controls ======
cat("\n=== Controls ===\n")
ctrl_rows <- list()
for (y in DV_LEAD) {
  cat(sprintf("  %s...\n", y))
  for (li in c("L2","L4")) {
    f <- as.formula(paste0(y, " ~ UNIONIZATION_cap1", controls, " | ",
      ifelse(li=="L2", "gvkey + review_year", "review_year")))
    fit <- tryCatch(feols(f, data=df, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e)NULL)
    if (is.null(fit)) {
      ctrl_rows[[length(ctrl_rows)+1]] <- data.frame(model=li, dv=y, coef=NA, se=NA, pvalue=NA, n_obs=NA, n_firms=NA, dropped="error")
      next
    }
    r <- coeftable(fit)["UNIONIZATION_cap1",]
    ctrl_rows[[length(ctrl_rows)+1]] <- data.frame(model=li, dv=y, coef=r["Estimate"], se=r["Std. Error"], pvalue=r["Pr(>|t|)"], n_obs=nobs(fit), n_firms=length(unique(df$gvkey)), dropped="")
  }
}
ctrls <- bind_rows(ctrl_rows)
write_csv(ctrls, paste0(OUT,"ext_reg_controls.csv"))

# ====== Robustness ======
cat("\n=== Robustness ===\n")
rob_rows <- list()
for (y in DV_LEAD) {
  # binary L2
  f <- as.formula(paste0(y, " ~ UNIONIZATION_binary | gvkey + review_year"))
  fit <- tryCatch(feols(f, data=df, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e)NULL)
  if (!is.null(fit)) {
    r <- coeftable(fit)["UNIONIZATION_binary",]
    rob_rows[[length(rob_rows)+1]] <- data.frame(spec="binary_L2", dv=y, coef=r["Estimate"], se=r["Std. Error"], pvalue=r["Pr(>|t|)"], n_obs=nobs(fit))
  }
  # current L2
  cur_col <- paste0(gsub("_lead1","",y), "_cur_lead1")
  if (cur_col %in% colnames(df)) {
    f <- as.formula(paste0(cur_col, " ~ UNIONIZATION_cap1 | gvkey + review_year"))
    fit <- tryCatch(feols(f, data=df, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e)NULL)
    if (!is.null(fit)) {
      r <- coeftable(fit)["UNIONIZATION_cap1",]
      rob_rows[[length(rob_rows)+1]] <- data.frame(spec="current_L2", dv=cur_col, coef=r["Estimate"], se=r["Std. Error"], pvalue=r["Pr(>|t|)"], n_obs=nobs(fit))
    }
  }
  # contemporaneous
  y_ct <- gsub("_lead1","",y)
  f <- as.formula(paste0(y_ct, " ~ UNIONIZATION_cap1 | gvkey + review_year"))
  fit <- tryCatch(feols(f, data=df, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e)NULL)
  if (!is.null(fit)) {
    r <- coeftable(fit)["UNIONIZATION_cap1",]
    rob_rows[[length(rob_rows)+1]] <- data.frame(spec="contemporaneous_L2", dv=y_ct, coef=r["Estimate"], se=r["Std. Error"], pvalue=r["Pr(>|t|)"], n_obs=nobs(fit))
  }
}
rob <- bind_rows(rob_rows)
write_csv(rob, paste0(OUT,"ext_reg_robustness.csv"))

cat("\n=== Key Results ===\n")
cat("\nL2 WLB:", round(ladder$coef[ladder$model=="L2" & ladder$dv=="wlb_lead1"],4),
    "p:", round(ladder$pvalue[ladder$model=="L2" & ladder$dv=="wlb_lead1"],4))
cat("\nL2 Comp:", round(ladder$coef[ladder$model=="L2" & ladder$dv=="comp_benefit_lead1"],4),
    "p:", round(ladder$pvalue[ladder$model=="L2" & ladder$dv=="comp_benefit_lead1"],4))
cat("\nL2 Overall:", round(ladder$coef[ladder$model=="L2" & ladder$dv=="overall_rating_lead1"],4),
    "p:", round(ladder$pvalue[ladder$model=="L2" & ladder$dv=="overall_rating_lead1"],4))
cat("\n\ndiversity non-null overlap:")
cat(sprintf(" %d fy, %.1f%% unionized\n",
  sum(!is.na(df$diversity_lead1) & !is.na(df$UNIONIZATION_cap1)),
  mean(df$UNIONIZATION_cap1[!is.na(df$diversity_lead1)]>0)*100))
cat("\nDone.\n")
