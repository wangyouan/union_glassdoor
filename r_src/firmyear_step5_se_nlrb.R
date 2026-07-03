#!/usr/bin/env Rscript
# STEP 5: Previous round deliverables
# 5.1: SE comparison (iid vs cluster gvkey vs cluster gvkey+year) for WLB + overall
# 5.2: nlrb_binary_v3.csv — standard 10-row regression table

library(fixest); library(dplyr); library(tidyr); library(readr); library(nanoparquet)
options(fixest_notes=FALSE); setFixest_notes(FALSE)

OUT_IN <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/firmyear_unionization/finished_panel/"
OUT <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/firmyear_fill_missing/"
dir.create(OUT, showWarnings=FALSE, recursive=TRUE)

# ====== 5.1: SE Comparison ======
cat("=== 5.1: SE Comparison ===\n")
df <- nanoparquet::read_parquet(paste0(OUT_IN,"merged_panel_main.parquet"))

df <- df |> mutate(
  gvkey = as.character(gvkey),
  review_year = as.integer(review_year),
  UNIONIZATION_cap1 = pmin(ifelse(is.na(UNIONIZATION_raw), 0, UNIONIZATION_raw), 1)
)

DV_ALL <- c("overall_rating","career_opp","comp_benefit","senior_mgmt","wlb","culture",
            "recommend","business_outlook","ceo_approval","diversity")

df <- df |> arrange(gvkey, review_year) |> group_by(gvkey) |>
  mutate(across(all_of(DV_ALL), ~ dplyr::lead(.x, 1), .names = "{.col}_lead1")) |> ungroup()

# Run models with 3 SE types
se_rows <- list()
for (y in c("wlb_lead1", "overall_rating_lead1")) {
  f <- as.formula(paste0(y, " ~ UNIONIZATION_cap1 | gvkey + review_year"))

  # iid
  fit_iid <- feols(f, data=df, se="standard", warn=FALSE, notes=FALSE)
  r_iid <- coeftable(fit_iid)["UNIONIZATION_cap1", ]

  # cluster gvkey
  fit_cl1 <- feols(f, data=df, cluster=~gvkey, warn=FALSE, notes=FALSE)
  r_cl1 <- coeftable(fit_cl1)["UNIONIZATION_cap1", ]

  # cluster gvkey + review_year (two-way)
  fit_cl2 <- feols(f, data=df, cluster=~gvkey+review_year, warn=FALSE, notes=FALSE)
  r_cl2 <- coeftable(fit_cl2)["UNIONIZATION_cap1", ]

  for (se_type in c("iid", "cluster_gvkey", "cluster_gvkey_year")) {
    r <- switch(se_type, iid=r_iid, cluster_gvkey=r_cl1, cluster_gvkey_year=r_cl2)
    se_rows[[length(se_rows)+1]] <- data.frame(
      dv=y, se_type=se_type,
      coef=r["Estimate"], se=r["Std. Error"], pvalue=r["Pr(>|t|)"],
      stringsAsFactors=FALSE)
  }
}

se_comp <- bind_rows(se_rows)
write_csv(se_comp, paste0(OUT,"diag_se_comparison.csv"))
cat("SE comparison:\n")
for (i in 1:nrow(se_comp)) cat(sprintf("  %-25s %-20s coef=% 8.4f se=%7.4f p=%6.4f\n",
  se_comp$dv[i], se_comp$se_type[i], se_comp$coef[i], se_comp$se[i], se_comp$pvalue[i]))

# Also print the actual fixest call for the report
cat("\n--- Actual fixest call (for report) ---\n")
cat('feols(wlb_lead1 ~ UNIONIZATION_cap1 | gvkey + review_year, data=df, cluster=~gvkey)\n')

# ====== 5.2: NLRB Binary v3 ======
cat("\n=== 5.2: NLRB Binary v3 ===\n")
nlrb <- nanoparquet::read_parquet(paste0(OUT,"nlrb_merged.parquet"))

nlrb <- nlrb |> mutate(
  gvkey = as.character(gvkey),
  review_year = as.integer(review_year),
  has_union_fixed = ifelse(is.na(has_union_fixed), 0, has_union_fixed)
)

nlrb <- nlrb |> arrange(gvkey, review_year) |> group_by(gvkey) |>
  mutate(across(all_of(DV_ALL), ~ dplyr::lead(.x, 1), .names = "{.col}_lead1")) |> ungroup()

nlrb_rows <- list()
DV_LEAD <- paste0(DV_ALL, "_lead1")
for (y in DV_LEAD) {
  n_non_na <- sum(!is.na(nlrb[[y]]))
  if (n_non_na < 50) {
    nlrb_rows[[length(nlrb_rows)+1]] <- data.frame(
      dv=y, coef=NA, se=NA, p=NA, N=n_non_na, N_firms=NA, note=sprintf("only %d non-NA", n_non_na),
      stringsAsFactors=FALSE)
    next
  }
  f <- as.formula(paste0(y, " ~ has_union_fixed | gvkey + review_year"))
  fit <- tryCatch(feols(f, data=nlrb, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e)NULL)
  if (is.null(fit)) {
    nlrb_rows[[length(nlrb_rows)+1]] <- data.frame(
      dv=y, coef=NA, se=NA, p=NA, N=NA, N_firms=NA, note="feols error", stringsAsFactors=FALSE)
    next
  }
  r <- coeftable(fit)["has_union_fixed", ]
  nlrb_rows[[length(nlrb_rows)+1]] <- data.frame(
    dv=y, coef=r["Estimate"], se=r["Std. Error"], p=r["Pr(>|t|)"],
    N=nobs(fit), N_firms=length(unique(nlrb$gvkey)), note="",
    stringsAsFactors=FALSE)
}

nlrb_v3 <- bind_rows(nlrb_rows)
write_csv(nlrb_v3, paste0(OUT,"nlrb_binary_v3.csv"))
cat("\nNLRB binary v3 (10 DVs):\n")
for (i in 1:nrow(nlrb_v3)) cat(sprintf("  %-30s coef=% 8.4f se=%7.4f p=%6.4f N=%d firms=%d %s\n",
  nlrb_v3$dv[i], nlrb_v3$coef[i], nlrb_v3$se[i], nlrb_v3$p[i],
  nlrb_v3$N[i], nlrb_v3$N_firms[i], nlrb_v3$note[i]))

cat("\nSTEP 5 done.\n")
