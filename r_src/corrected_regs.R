#!/usr/bin/env Rscript
# Round 13: Corrected panel regressions (Compustat-only + cell n>=10)
library(fixest); library(dplyr); library(tidyr); library(readr); library(nanoparquet)
options(fixest_notes=FALSE); setFixest_notes(FALSE)

OUT <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260704/firmyear_corrected/"
dir.create(OUT, showWarnings=FALSE, recursive=TRUE)

DV <- c("overall_rating","career_opp","comp_benefit","senior_mgmt","wlb","culture",
        "recommend","business_outlook","ceo_approval","diversity")
DV_CUR <- paste0(DV, "_cur")
ctrls <- "+ SIZE + LEVERAGE + CAPEX + EBITDA + SGA + NOLCF + LOSS"

prep <- function(df) {
  df <- df |> mutate(gvkey=as.character(gvkey), review_year=as.integer(review_year),
    sic2=substr(as.character(sic),1,2))
  winsor <- function(x,p=0.01) { q <- quantile(x,c(p,1-p),na.rm=TRUE); pmax(q[1],pmin(q[2],x)) }
  df <- df |> mutate(SIZE=winsor(at,0.01), LEVERAGE=winsor((dlc+dltt)/at,0.01),
    CAPEX=winsor(capx/at,0.01), EBITDA=winsor(ebitda/at,0.01),
    SGA=winsor(xsga/sale,0.01), NOLCF=ifelse(tlcf/at>0,1,0), LOSS=ifelse(ib<0,1,0))
  df |> arrange(gvkey,review_year) |> group_by(gvkey) |>
    mutate(across(all_of(c(DV,DV_CUR)),~dplyr::lead(.x,1),.names="{.col}_lead1")) |> ungroup()
}

run_model <- function(data, y, var_name, fe_spec, model_lbl) {
  f <- as.formula(paste0(y," ~ ",var_name," | ",fe_spec))
  fit <- tryCatch(feols(f, data=data, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e)NULL)
  if (is.null(fit)) return(data.frame(model=model_lbl, dv=y, coef=NA, se=NA, tstat=NA, pvalue=NA, n_obs=NA, n_firms=NA, r2=NA, r2_within=NA, dropped="error"))
  ct <- coeftable(fit)
  if (!var_name %in% rownames(ct)) return(data.frame(model=model_lbl, dv=y, coef=NA, se=NA, tstat=NA, pvalue=NA, n_obs=nobs(fit), n_firms=NA, r2=r2(fit,"r2"), r2_within=r2(fit,"wr2"), dropped="var dropped"))
  r <- ct[var_name,]
  data.frame(model=model_lbl, dv=y, coef=r["Estimate"], se=se(fit)[var_name], tstat=r["Estimate"]/se(fit)[var_name], pvalue=r["Pr(>|t|)"], n_obs=nobs(fit), n_firms=length(unique(data$gvkey)), r2=r2(fit,"r2"), r2_within=r2(fit,"wr2"), dropped="")
}

run_ladder <- function(data, y, var_name, suffix) {
  bind_rows(
    run_model(data, y, var_name, "gvkey + sic2^review_year", paste0("L1",suffix)),
    run_model(data, y, var_name, "gvkey + review_year", paste0("L2",suffix)),
    run_model(data, y, var_name, "sic2^review_year", paste0("L3",suffix)),
    run_model(data, y, var_name, "review_year", paste0("L4",suffix)),
    run_model(data, y, var_name, "sic2 + review_year", paste0("L5",suffix))
  )
}

run_ladder_ctrl <- function(data, y, var_name, suffix) {
  bind_rows(
    run_model(data, y, paste0(var_name,ctrls), "gvkey + sic2^review_year", paste0("L1",suffix)),
    run_model(data, y, paste0(var_name,ctrls), "gvkey + review_year", paste0("L2",suffix)),
    run_model(data, y, paste0(var_name,ctrls), "sic2^review_year", paste0("L3",suffix)),
    run_model(data, y, paste0(var_name,ctrls), "review_year", paste0("L4",suffix)),
    run_model(data, y, paste0(var_name,ctrls), "sic2 + review_year", paste0("L5",suffix))
  )
}

# ============ EST CORRECTED ============
cat("=== EST Corrected Panel ===\n")
df_est <- prep(nanoparquet::read_parquet(paste0(OUT,"corrected_panel.parquet")))
cat(sprintf("EST: %d rows\n", nrow(df_est)))

# Ladder: apply n>=10 filter per DV
cat("EST Ladder...\n")
est_lad <- bind_rows(lapply(DV, function(dv) {
  y <- paste0(dv,"_lead1"); n_col <- paste0("n_",dv,"_all")
  d <- df_est; if (n_col %in% colnames(d)) d <- d |> filter(!is.na(.data[[n_col]]) & .data[[n_col]] >= 10)
  run_ladder(d, y, "UNIONIZATION_cap1", "")
}))
write_csv(est_lad, paste0(OUT,"corr_est_ladder.csv"))
cat(sprintf("  %d rows\n", nrow(est_lad)))

# Controls
cat("EST Controls...\n")
est_ctrl <- bind_rows(lapply(DV, function(dv) {
  y <- paste0(dv,"_lead1"); n_col <- paste0("n_",dv,"_all")
  d <- df_est; if (n_col %in% colnames(d)) d <- d |> filter(!is.na(.data[[n_col]]) & .data[[n_col]] >= 10)
  run_ladder_ctrl(d, y, "UNIONIZATION_cap1", "")
}))
write_csv(est_ctrl, paste0(OUT,"corr_est_controls.csv"))
cat(sprintf("  %d rows\n", nrow(est_ctrl)))

# Robustness: binary/raw/current/contemporaneous (L2, n>=10 filter)
cat("EST Robustness...\n")
rob_rows <- list()
for (dv in DV) {
  y <- paste0(dv,"_lead1"); n_col <- paste0("n_",dv,"_all")
  d <- df_est; if (n_col %in% colnames(d)) d <- d |> filter(!is.na(.data[[n_col]]) & .data[[n_col]] >= 10)

  # binary
  r <- run_model(d, y, "UNIONIZATION_binary", "gvkey + review_year", "binary_L2"); r$dv <- y; rob_rows[[length(rob_rows)+1]] <- r
  # raw
  r <- run_model(d, y, "UNIONIZATION_raw", "gvkey + review_year", "raw_L2"); r$dv <- y; rob_rows[[length(rob_rows)+1]] <- r
  # current
  cur_y <- paste0(dv,"_cur_lead1")
  if (cur_y %in% colnames(d)) {
    r <- run_model(d, cur_y, "UNIONIZATION_cap1", "gvkey + review_year", "current_L2"); r$dv <- cur_y; rob_rows[[length(rob_rows)+1]] <- r
  }
  # contemporaneous
  r <- run_model(d, dv, "UNIONIZATION_cap1", "gvkey + review_year", "contemporaneous_L2"); r$dv <- dv; rob_rows[[length(rob_rows)+1]] <- r
}
est_rob <- bind_rows(rob_rows)
# Fix column order
est_rob <- est_rob |> rename(spec=model) |> select(spec, dv, coef, se, tstat, pvalue, n_obs, n_firms, r2, r2_within, dropped)
write_csv(est_rob, paste0(OUT,"corr_est_robustness.csv"))
cat(sprintf("  %d rows\n", nrow(est_rob)))

# n>=5 and n>=20 robustness (L2 only)
for (ncut in c(5,20)) {
  cat(sprintf("EST n>=%d...\n", ncut))
  rows <- bind_rows(lapply(DV, function(dv) {
    y <- paste0(dv,"_lead1"); n_col <- paste0("n_",dv,"_all")
    d <- df_est; if (n_col %in% colnames(d)) d <- d |> filter(!is.na(.data[[n_col]]) & .data[[n_col]] >= ncut)
    run_model(d, y, "UNIONIZATION_cap1", "gvkey + review_year", "L2")
  }))
  write_csv(rows, paste0(OUT,sprintf("corr_est_n%d.csv", ncut)))
  cat(sprintf("  %d rows\n", nrow(rows)))
}

# ============ EMP EXTENDED ============
cat("\n=== EMP Extended Panel ===\n")
df_emp <- prep(nanoparquet::read_parquet(paste0(OUT,"emp_extended_panel.parquet")))
df_emp <- df_emp |> filter(review_year >= 2005)
cat(sprintf("EMP: %d rows, yrs %d-%d\n", nrow(df_emp), min(df_emp$review_year), max(df_emp$review_year)))

cat("EMP Ladder...\n")
emp_lad <- bind_rows(lapply(DV, function(dv) {
  y <- paste0(dv,"_lead1"); n_col <- paste0("n_",dv,"_all")
  d <- df_emp; if (n_col %in% colnames(d)) d <- d |> filter(!is.na(.data[[n_col]]) & .data[[n_col]] >= 10)
  run_ladder(d, y, "UNIONIZATION_EMP_cap1", "_emp")
}))
write_csv(emp_lad, paste0(OUT,"corr_emp_ladder.csv"))
cat(sprintf("  %d rows\n", nrow(emp_lad)))

cat("EMP Controls...\n")
emp_ctrl <- bind_rows(lapply(DV, function(dv) {
  y <- paste0(dv,"_lead1"); n_col <- paste0("n_",dv,"_all")
  d <- df_emp; if (n_col %in% colnames(d)) d <- d |> filter(!is.na(.data[[n_col]]) & .data[[n_col]] >= 10)
  run_ladder_ctrl(d, y, "UNIONIZATION_EMP_cap1", "_emp")
}))
write_csv(emp_ctrl, paste0(OUT,"corr_emp_controls.csv"))
cat(sprintf("  %d rows\n", nrow(emp_ctrl)))

# EMP A/B test (same sample)
cat("EMP A/B test...\n")
ctrl_cols <- c("at","dltt","dlc","capx","ebitda","sale","tlcf","ib","xsga")
ab_rows <- list()
for (dv in DV) {
  y <- paste0(dv,"_lead1"); n_col <- paste0("n_",dv,"_all")
  d <- df_emp; if (n_col %in% colnames(d)) d <- d |> filter(!is.na(.data[[n_col]]) & .data[[n_col]] >= 10)
  ok <- complete.cases(d[,ctrl_cols]) & !is.na(d[[y]]) & !is.na(d$UNIONIZATION_EMP_cap1)
  dss <- d[ok,]
  if (nrow(dss) < 50) next
  # A: no controls
  r <- run_model(dss, y, "UNIONIZATION_EMP_cap1", "gvkey + review_year", "A_noctrl"); r$dv <- y; ab_rows[[length(ab_rows)+1]] <- r
  # B: with controls
  r <- run_model(dss, y, paste0("UNIONIZATION_EMP_cap1",ctrls), "gvkey + review_year", "B_ctrl"); r$dv <- y; ab_rows[[length(ab_rows)+1]] <- r
}
emp_ab <- bind_rows(ab_rows)
emp_ab <- emp_ab |> rename(spec=model) |> select(spec, dv, coef, se, tstat, pvalue, n_obs, n_firms, r2, r2_within, dropped)
write_csv(emp_ab, paste0(OUT,"corr_emp_ab.csv"))
cat(sprintf("  %d rows\n", nrow(emp_ab)))

# Diversity check
cat("\n=== Diversity ===\n")
div_emp <- df_emp |> filter(!is.na(diversity_lead1) & review_year>=2020 & UNIONIZATION_EMP_cap1>0)
cat(sprintf("EMP diversity (2020-2024, unionized): %d fy\n", nrow(div_emp)))
div_est <- df_est |> filter(!is.na(diversity_lead1) & review_year>=2020 & UNIONIZATION_cap1>0)
cat(sprintf("EST diversity (2020-2022, unionized): %d fy\n", nrow(div_est)))

cat("\nDone\n")
