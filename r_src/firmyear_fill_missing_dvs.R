#!/usr/bin/env Rscript
# Fill missing DVs: controls (5 DVs) + robustness (4 DVs)
library(fixest); library(dplyr); library(tidyr); library(readr); library(nanoparquet)
options(fixest_notes=FALSE); setFixest_notes(FALSE)

OUT_IN <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/firmyear_unionization/finished_panel/"
OUT <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/firmyear_fill_missing/"
dir.create(OUT, showWarnings=FALSE, recursive=TRUE)

cat("Loading merged panel...\n")
df <- nanoparquet::read_parquet(paste0(OUT_IN,"merged_panel_main.parquet"))
cat(sprintf("Rows: %d, gvkeys: %d\n", nrow(df), length(unique(df$gvkey))))

# Data prep (identical to round 1)
df <- df |> mutate(
  gvkey = as.character(gvkey),
  review_year = as.integer(review_year),
  UNIONIZATION_raw = ifelse(is.na(UNIONIZATION_raw), 0, UNIONIZATION_raw),
  UNIONIZATION = ifelse(is.na(UNIONIZATION), 0, UNIONIZATION),
  UNIONIZATION_cap1 = pmin(UNIONIZATION_raw, 1),
  UNIONIZATION_binary = ifelse(UNIONIZATION > 0, 1, 0),
  sic2 = substr(as.character(sic), 1, 2)
)

winsor <- function(x, p=0.01) {
  q <- quantile(x, c(p, 1-p), na.rm=TRUE)
  pmax(q[1], pmin(q[2], x))
}

df <- df |> mutate(
  SIZE = winsor(at, 0.01),
  LEVERAGE = winsor((dlc + dltt) / at, 0.01),
  CAPEX = winsor(capx / at, 0.01),
  EBITDA = winsor(ebitda / at, 0.01),
  SGA = winsor(xsga / sale, 0.01),
  NOLCF = ifelse(tlcf / at > 0, 1, 0),
  LOSS = ifelse(ib < 0, 1, 0)
)

DV_ALL <- c("overall_rating","career_opp","comp_benefit","senior_mgmt","wlb","culture",
            "recommend","business_outlook","ceo_approval","diversity")
DV_CUR <- c("overall_rating_cur","career_opp_cur","comp_benefit_cur","senior_mgmt_cur",
            "wlb_cur","culture_cur","recommend_cur","business_outlook_cur","ceo_approval_cur","diversity_cur")

# Create lead outcomes
df <- df |> arrange(gvkey, review_year) |> group_by(gvkey) |>
  mutate(across(all_of(c(DV_ALL, DV_CUR)), ~ dplyr::lead(.x, 1), .names = "{.col}_lead1")) |> ungroup()

controls_str <- "+ SIZE + LEVERAGE + CAPEX + EBITDA + SGA + NOLCF + LOSS"

# ====== TASK 1: Controls for 5 missing DVs ======
cat("\n=== TASK 1: Controls for missing DVs ===\n")
MISSING_DVS <- c("culture","recommend","business_outlook","ceo_approval","diversity")

ctrl_rows <- list()
for (y_base in MISSING_DVS) {
  y_var <- paste0(y_base, "_lead1")
  cat(sprintf("  %s...\n", y_var))

  # Check data availability
  n_avail <- sum(!is.na(df[[y_var]]))
  if (n_avail < 100) {
    cat(sprintf("    SKIP: only %d non-NA rows\n", n_avail))
    for (li in c("L1","L2","L3","L4")) {
      ctrl_rows[[length(ctrl_rows)+1]] <- data.frame(
        model=li, dv=y_var, coef=NA, se=NA, pvalue=NA, n_obs=n_avail,
        n_firms=NA, dropped=sprintf("only %d non-NA rows", n_avail),
        stringsAsFactors=FALSE)
    }
    next
  }

  for (li in c("L1","L2","L3","L4")) {
    if (li == "L1") {
      f <- as.formula(paste0(y_var, " ~ UNIONIZATION_cap1", controls_str, " | gvkey + sic2^review_year"))
    } else if (li == "L2") {
      f <- as.formula(paste0(y_var, " ~ UNIONIZATION_cap1", controls_str, " | gvkey + review_year"))
    } else if (li == "L3") {
      f <- as.formula(paste0(y_var, " ~ UNIONIZATION_cap1", controls_str, " | sic2^review_year"))
    } else {
      f <- as.formula(paste0(y_var, " ~ UNIONIZATION_cap1", controls_str, " | review_year"))
    }

    fit <- tryCatch(feols(f, data=df, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e)NULL)
    if (is.null(fit)) {
      ctrl_rows[[length(ctrl_rows)+1]] <- data.frame(
        model=li, dv=y_var, coef=NA, se=NA, pvalue=NA, n_obs=NA, n_firms=NA,
        dropped="feols error", stringsAsFactors=FALSE)
      next
    }

    ct <- coeftable(fit)
    r <- tryCatch(ct["UNIONIZATION_cap1", ], error=function(e)NULL)
    if (is.null(r) || is.na(r["Estimate"])) {
      ctrl_rows[[length(ctrl_rows)+1]] <- data.frame(
        model=li, dv=y_var, coef=NA, se=NA, pvalue=NA, n_obs=nobs(fit),
        n_firms=length(unique(df$gvkey)), dropped="coefficient dropped/NA",
        stringsAsFactors=FALSE)
      next
    }
    ctrl_rows[[length(ctrl_rows)+1]] <- data.frame(
      model=li, dv=y_var, coef=r["Estimate"], se=r["Std. Error"], pvalue=r["Pr(>|t|)"],
      n_obs=nobs(fit), n_firms=length(unique(df$gvkey)), dropped="",
      stringsAsFactors=FALSE)
  }
}

fill_ctrl <- bind_rows(ctrl_rows)
write_csv(fill_ctrl, paste0(OUT,"fill_reg_controls.csv"))
cat("\nControls fill saved:\n")
for (i in 1:nrow(fill_ctrl)) cat(sprintf("  %-4s %-30s coef=% 8.4f se=%7.4f p=%6.4f n=%d %s\n",
  fill_ctrl$model[i], fill_ctrl$dv[i], fill_ctrl$coef[i], fill_ctrl$se[i], fill_ctrl$pvalue[i],
  fill_ctrl$n_obs[i], fill_ctrl$dropped[i]))

# ====== TASK 2: Robustness for 4 missing DVs ======
cat("\n=== TASK 2: Robustness for missing DVs ===\n")
MISSING_ROB <- c("recommend","business_outlook","ceo_approval","diversity")

rob_rows <- list()
for (y_base in MISSING_ROB) {
  y_lead <- paste0(y_base, "_lead1")
  y_cur_lead <- paste0(y_base, "_cur_lead1")

  cat(sprintf("  %s...\n", y_base))

  # (a) binary UNIONIZATION, L2
  f <- as.formula(paste0(y_lead, " ~ UNIONIZATION_binary | gvkey + review_year"))
  fit <- tryCatch(feols(f, data=df, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e)NULL)
  if (!is.null(fit)) {
    r <- coeftable(fit)["UNIONIZATION_binary", ]
    rob_rows[[length(rob_rows)+1]] <- data.frame(spec="binary_L2", dv=y_lead,
      coef=r["Estimate"], se=r["Std. Error"], pvalue=r["Pr(>|t|)"], n_obs=nobs(fit), stringsAsFactors=FALSE)
  } else {
    rob_rows[[length(rob_rows)+1]] <- data.frame(spec="binary_L2", dv=y_lead,
      coef=NA, se=NA, pvalue=NA, n_obs=NA, stringsAsFactors=FALSE)
  }

  # (b) raw UNIONIZATION, L2
  f <- as.formula(paste0(y_lead, " ~ UNIONIZATION_raw | gvkey + review_year"))
  fit <- tryCatch(feols(f, data=df, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e)NULL)
  if (!is.null(fit)) {
    r <- coeftable(fit)["UNIONIZATION_raw", ]
    rob_rows[[length(rob_rows)+1]] <- data.frame(spec="raw_L2", dv=y_lead,
      coef=r["Estimate"], se=r["Std. Error"], pvalue=r["Pr(>|t|)"], n_obs=nobs(fit), stringsAsFactors=FALSE)
  } else {
    rob_rows[[length(rob_rows)+1]] <- data.frame(spec="raw_L2", dv=y_lead,
      coef=NA, se=NA, pvalue=NA, n_obs=NA, stringsAsFactors=FALSE)
  }

  # (c) current-only LHS, L2
  cur_col <- paste0(y_base, "_cur_lead1")
  if (cur_col %in% colnames(df)) {
    n_cur <- sum(!is.na(df[[cur_col]]))
    if (n_cur >= 100) {
      f <- as.formula(paste0(cur_col, " ~ UNIONIZATION_cap1 | gvkey + review_year"))
      fit <- tryCatch(feols(f, data=df, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e)NULL)
      if (!is.null(fit)) {
        r <- coeftable(fit)["UNIONIZATION_cap1", ]
        rob_rows[[length(rob_rows)+1]] <- data.frame(spec="current_L2", dv=cur_col,
          coef=r["Estimate"], se=r["Std. Error"], pvalue=r["Pr(>|t|)"], n_obs=nobs(fit), stringsAsFactors=FALSE)
      } else {
        rob_rows[[length(rob_rows)+1]] <- data.frame(spec="current_L2", dv=cur_col,
          coef=NA, se=NA, pvalue=NA, n_obs=NA, stringsAsFactors=FALSE)
      }
    } else {
      rob_rows[[length(rob_rows)+1]] <- data.frame(spec="current_L2", dv=cur_col,
        coef=NA, se=NA, pvalue=NA, n_obs=n_cur,
        stringsAsFactors=FALSE)
    }
  } else {
    rob_rows[[length(rob_rows)+1]] <- data.frame(spec="current_L2", dv="NA",
      coef=NA, se=NA, pvalue=NA, n_obs=NA, stringsAsFactors=FALSE)
    cat(sprintf("    current_L2: column '%s' not found, NA+reason\n", cur_col))
  }

  # (d) contemporaneous, L2
  f <- as.formula(paste0(y_base, " ~ UNIONIZATION_cap1 | gvkey + review_year"))
  fit <- tryCatch(feols(f, data=df, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e)NULL)
  if (!is.null(fit)) {
    r <- coeftable(fit)["UNIONIZATION_cap1", ]
    rob_rows[[length(rob_rows)+1]] <- data.frame(spec="contemporaneous_L2", dv=y_base,
      coef=r["Estimate"], se=r["Std. Error"], pvalue=r["Pr(>|t|)"], n_obs=nobs(fit), stringsAsFactors=FALSE)
  } else {
    rob_rows[[length(rob_rows)+1]] <- data.frame(spec="contemporaneous_L2", dv=y_base,
      coef=NA, se=NA, pvalue=NA, n_obs=NA, stringsAsFactors=FALSE)
  }
}

fill_rob <- bind_rows(rob_rows)
write_csv(fill_rob, paste0(OUT,"fill_reg_robustness.csv"))
cat("\nRobustness fill saved:\n")
for (i in 1:nrow(fill_rob)) cat(sprintf("  %-25s %-35s coef=% 8.4f se=%7.4f p=%6.4f n=%d\n",
  fill_rob$spec[i], fill_rob$dv[i], fill_rob$coef[i], fill_rob$se[i], fill_rob$pvalue[i], fill_rob$n_obs[i]))

cat("\nDone.\n")
