#!/usr/bin/env Rscript
# STEP 3: Firm-year unionization ratio ~ Glassdoor ratings
#  3.1 Correlations (Pearson + Spearman + binscatter-style quintile table)
#  3.2 Pooled OLS: rating ~ union_ratio + year FE, cluster ~gvkey
#  3.3 Within-firm FE: feols(rating ~ union_ratio | gvkey + year, cluster = ~gvkey)
#  3.4 With firm controls
#  3.5 Robustness: lag, capped, EMP_t-1, current-only, log(1+ratio), binary

suppressMessages({library(fixest); library(dplyr); library(tidyr); library(nanoparquet); library(readr); library(purrr)})

OUT <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260702/firmyear_unionization/"

DV10 <- c("overall_rating","career_opp","comp_benefit","senior_mgmt","wlb","culture",
          "recommend","business_outlook","ceo_approval","diversity")

# ─── Load & merge ───────────────────────────────────────────────────────────
cat("Loading panels...\n")
union_panel <- read_parquet(paste0(OUT, "firmyear_unionization_panel.parquet"))
gd_panel <- read_parquet(paste0(OUT, "firmyear_glassdoor_panel.parquet"))

# Merge: union panel is the master (includes Compustat firms)
# Glassdoor panel provides ratings
panel <- union_panel |>
  left_join(gd_panel |> select(gvkey, review_year, all_of(DV10),
                                paste0(DV10,"_cur"), n_reviews_all, n_reviews_cur),
            by=c("gvkey","fyear"="review_year"))

cat(sprintf("Merged panel: %d rows, %d firms\n", nrow(panel), n_distinct(panel$gvkey)))

# Filter: firm-years with >= 5 reviews (all)
panel$n5_all <- panel$n_reviews_all >= 5
panel$n5_cur <- panel$n_reviews_cur >= 5

panel_main <- panel[panel$n5_all & !is.na(panel$n5_all), ]
cat(sprintf("After n>=5 filter (all): %d rows\n", nrow(panel_main)))

# Log variables
panel_main$log_at <- log(panel_main$at)
panel_main$log_emp_firm <- log(panel_main$emp_actual)

# ═══════════════════════════════════════════════════════════════════════════
# 3.1 Correlations
# ═══════════════════════════════════════════════════════════════════════════
cat("\n=== 3.1 Correlations ===\n")

corr_rows <- list()
for (dv in DV10) {
  sub <- panel_main[!is.na(panel_main[[dv]]) & !is.na(panel_main$union_ratio_winsor), ]
  if (nrow(sub) < 30) next
  r_pearson <- cor(sub[[dv]], sub$union_ratio_winsor, use="complete.obs")
  r_spearman <- cor(sub[[dv]], sub$union_ratio_winsor, method="spearman", use="complete.obs")
  corr_rows[[length(corr_rows)+1]] <- data.frame(
    dv=dv, pearson=r_pearson, spearman=r_spearman,
    N=nrow(sub), firms=n_distinct(sub$gvkey))
  cat(sprintf("  %s: r_pearson=%.4f, r_spearman=%.4f\n", dv, r_pearson, r_spearman))
}
corr_df <- bind_rows(corr_rows)
write_csv(corr_df, paste0(OUT, "firmyear_correlations.csv"))
cat(sprintf("Saved firmyear_correlations.csv\n"))

# Quintile table (binscatter-style) — handle non-unique quantiles
cat("\nQuintile means (zero vs non-zero):\n")
for (dv in DV10) {
  sub <- panel_main[!is.na(panel_main[[dv]]) & !is.na(panel_main$union_ratio_winsor), ]
  if (nrow(sub) < 50) next
  zero_mean <- mean(sub[[dv]][sub$union_ratio_winsor == 0], na.rm=TRUE)
  nonzero_mean <- mean(sub[[dv]][sub$union_ratio_winsor > 0], na.rm=TRUE)
  n_nonzero <- sum(sub$union_ratio_winsor > 0)
  cat(sprintf("  %s: zero=%.3f (n=%d), nonzero=%.3f (n=%d)\n", dv, zero_mean, nrow(sub)-n_nonzero, nonzero_mean, n_nonzero))
}

# ═══════════════════════════════════════════════════════════════════════════
# Helper: run single DV regression and extract results
# ═══════════════════════════════════════════════════════════════════════════
run_and_extract <- function(formula_str, data, dv, spec_name) {
  fml <- as.formula(formula_str)
  d <- data[!is.na(data[[dv]]), ]
  if (nrow(d) < 30) return(data.frame(dv=dv, spec=spec_name, note="insufficient"))
  fit <- tryCatch(feols(fml, data=d, cluster=~gvkey), error=function(e) NULL)
  if (is.null(fit)) return(data.frame(dv=dv, spec=spec_name, note="model_failed"))
  ct <- coeftable(fit)
  ur_row <- grep("union_ratio", rownames(ct), value=TRUE)[1]
  if (is.na(ur_row)) return(data.frame(dv=dv, spec=spec_name, note="no_union_ratio_row"))
  b <- ct[ur_row, "Estimate"]; se <- ct[ur_row, "Std. Error"]; p <- ct[ur_row, "Pr(>|t|)"]
  dv_sd <- sd(d[[dv]], na.rm=TRUE)
  data.frame(dv=dv, spec=spec_name, coef=b, se=se, p=p,
             std_coef=b/dv_sd, N=nrow(d), N_firms=n_distinct(d$gvkey),
             note="")
}

# ═══════════════════════════════════════════════════════════════════════════
# 3.2 Pooled OLS: rating ~ union_ratio_winsor + factor(fyear)
# ═══════════════════════════════════════════════════════════════════════════
cat("\n=== 3.2 Pooled OLS ===\n")
pooled_rows <- list()
for (dv in DV10) {
  r <- run_and_extract(paste0(dv, " ~ union_ratio_winsor + factor(fyear)"), panel_main, dv, "pooled")
  pooled_rows[[length(pooled_rows)+1]] <- r
  if (r$note == "") cat(sprintf("  %s: coef=%.4f, p=%.4f\n", dv, r$coef, r$p))
}
pooled_df <- bind_rows(pooled_rows)
write_csv(pooled_df, paste0(OUT, "firmyear_reg_pooled.csv"))
cat(sprintf("Saved firmyear_reg_pooled.csv (%d rows)\n", nrow(pooled_df)))

# ═══════════════════════════════════════════════════════════════════════════
# 3.3 Within-firm FE
# ═══════════════════════════════════════════════════════════════════════════
cat("\n=== 3.3 Within-firm FE ===\n")
within_rows <- list()
for (dv in DV10) {
  r <- run_and_extract(paste0(dv, " ~ union_ratio_winsor | gvkey + fyear"), panel_main, dv, "within_firm")
  within_rows[[length(within_rows)+1]] <- r
  if (r$note == "") cat(sprintf("  %s: coef=%.4f, p=%.4f\n", dv, r$coef, r$p))
}
within_df <- bind_rows(within_rows)
write_csv(within_df, paste0(OUT, "firmyear_reg_within.csv"))
cat(sprintf("Saved firmyear_reg_within.csv (%d rows)\n", nrow(within_df)))

# ═══════════════════════════════════════════════════════════════════════════
# 3.4 With firm controls: log(AT) + ROA + leverage + log(EMP)
# ═══════════════════════════════════════════════════════════════════════════
cat("\n=== 3.4 With firm controls ===\n")
ctrl_rows <- list()
for (dv in DV10) {
  r <- run_and_extract(
    paste0(dv, " ~ union_ratio_winsor + log_at + L_roa + L_leverage + L_log_emp | gvkey + fyear"),
    panel_main, dv, "fe_with_controls")
  ctrl_rows[[length(ctrl_rows)+1]] <- r
  if (r$note == "") cat(sprintf("  %s: coef=%.4f, p=%.4f, N=%d\n", dv, r$coef, r$p, r$N))
}
ctrl_df <- bind_rows(ctrl_rows)
write_csv(ctrl_df, paste0(OUT, "firmyear_reg_controls.csv"))
cat(sprintf("Saved firmyear_reg_controls.csv (%d rows)\n", nrow(ctrl_df)))

# ═══════════════════════════════════════════════════════════════════════════
# 3.5 Robustness
# ═══════════════════════════════════════════════════════════════════════════
cat("\n=== 3.5 Robustness ===\n")
rob_rows <- list()

for (dv in DV10) {
  # a) Lagged union_ratio
  r <- run_and_extract(paste0(dv, " ~ union_ratio_lag + factor(fyear)"), panel_main, dv, "lagged_pooled")
  rob_rows[[length(rob_rows)+1]] <- r

  # b) Capped@1
  r <- run_and_extract(paste0(dv, " ~ union_ratio_capped | gvkey + fyear"), panel_main, dv, "capped_fe")
  rob_rows[[length(rob_rows)+1]] <- r

  # c) log(1+ratio)
  r <- run_and_extract(paste0(dv, " ~ log1p_union_ratio | gvkey + fyear"), panel_main, dv, "log1p_fe")
  rob_rows[[length(rob_rows)+1]] <- r

  # d) Binary: has_union
  r <- run_and_extract(paste0(dv, " ~ has_union | gvkey + fyear"), panel_main, dv, "binary_fe")
  rob_rows[[length(rob_rows)+1]] <- r
}

# e) Current-only LHS (DV_cur)
for (dv in DV10) {
  dv_cur <- paste0(dv, "_cur")
  if (dv_cur %in% names(panel_main)) {
    r <- run_and_extract(paste0(dv_cur, " ~ union_ratio_winsor | gvkey + fyear"), panel_main, dv, "current_lhs_fe")
    rob_rows[[length(rob_rows)+1]] <- r
  }
}

rob_df <- bind_rows(rob_rows)
write_csv(rob_df, paste0(OUT, "firmyear_reg_robustness.csv"))
cat(sprintf("Saved firmyear_reg_robustness.csv (%d rows)\n", nrow(rob_df)))

# ─── Summary table ──────────────────────────────────────────────────────────
cat("\n=== WLB across all specs ===\n")
wlb_rows <- bind_rows(list(pooled_df, within_df, ctrl_df, rob_df)) |> filter(dv=="wlb")
print(wlb_rows[, c("spec","coef","p","N","N_firms","note")], n=30)

cat("\nDone.\n")
