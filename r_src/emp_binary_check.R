#!/usr/bin/env Rscript
library(fixest); library(dplyr); library(tidyr); library(readr); library(nanoparquet)
options(fixest_notes=FALSE); setFixest_notes(FALSE)
OUT <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/firmyear_emp_denom/"
UNI2 <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/firmyear_unified_v2/"

df <- nanoparquet::read_parquet(paste0(OUT,"emp_panel.parquet"))
df <- df |> mutate(gvkey=as.character(gvkey), review_year=as.integer(review_year))
DV <- c("overall_rating","career_opp","comp_benefit","senior_mgmt","wlb","culture","recommend","business_outlook","ceo_approval","diversity")

df <- df |> arrange(gvkey, review_year) |> group_by(gvkey) |>
  mutate(across(all_of(DV), ~ dplyr::lead(.x,1), .names="{.col}_lead1")) |> ungroup()
df_main <- df |> filter(review_year >= 2005 & review_year <= 2022)

DV_LEAD <- paste0(DV, "_lead1")
est <- read_csv(paste0(UNI2,"unified2_reg_robustness.csv"), show_col_types=FALSE)

cat("Binary consistency check:\n")
all_ok <- TRUE
for (y in DV_LEAD) {
  f <- as.formula(paste0(y," ~ UNIONIZATION_EMP_binary | gvkey + review_year"))
  fit <- feols(f, data=df_main, cluster=~gvkey, warn=FALSE, notes=FALSE)
  r_emp <- coeftable(fit)["UNIONIZATION_EMP_binary",]
  r_est <- est[est$spec=="binary_L2" & est$dv==y,]
  if (nrow(r_est) > 0) {
    diff <- abs(r_emp["Estimate"] - r_est$coef[1])
    ok <- diff < 1e-6
    if (!ok) all_ok <- FALSE
    cat(sprintf("  %-35s EMP=%.6f EST=%.6f diff=%.8f %s\n", y, r_emp["Estimate"], r_est$coef[1], diff, if(ok) "OK" else "FAIL"))
  }
}
cat(sprintf("\nBinary check: %s\n", if(all_ok) "ALL PASS" else "SOME FAILED"))
