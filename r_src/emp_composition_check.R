#!/usr/bin/env Rscript
# Round 11: EMP composition check — covariate absorption vs sample composition
library(fixest); library(dplyr); library(tidyr); library(readr); library(nanoparquet)
options(fixest_notes=FALSE); setFixest_notes(FALSE)

OUT <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/emp_composition_check/"
PREV <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/firmyear_stats_full/"
EMP_DIR <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/firmyear_emp_denom/"
dir.create(OUT, showWarnings=FALSE, recursive=TRUE)

cat("Loading EMP panel...\n")
df <- nanoparquet::read_parquet(paste0(EMP_DIR,"emp_panel.parquet"))

DV_ALL <- c("overall_rating","career_opp","comp_benefit","senior_mgmt","wlb","culture",
            "recommend","business_outlook","ceo_approval","diversity")
DV_LEAD <- paste0(DV_ALL, "_lead1")

# Prep
df <- df |> mutate(gvkey=as.character(gvkey), review_year=as.integer(review_year),
  sic2=substr(as.character(sic),1,2))

winsor <- function(x,p=0.01) { q <- quantile(x,c(p,1-p),na.rm=TRUE); pmax(q[1],pmin(q[2],x)) }
df <- df |> mutate(
  SIZE=winsor(at,0.01), LEVERAGE=winsor((dlc+dltt)/at,0.01),
  CAPEX=winsor(capx/at,0.01), EBITDA=winsor(ebitda/at,0.01),
  SGA=winsor(xsga/sale,0.01), NOLCF=ifelse(tlcf/at>0,1,0), LOSS=ifelse(ib<0,1,0))

df <- df |> arrange(gvkey,review_year) |> group_by(gvkey) |>
  mutate(across(all_of(DV_ALL),~dplyr::lead(.x,1),.names="{.col}_lead1")) |> ungroup()

# ====== STEP 1: Same-sample comparison ======
cat("\n=== STEP 1: Same-sample A vs B ===\n")

# Find the exact sample used in emp_clean_reg_controls (has emp>0 + all 8 controls + in main window)
ctrl_cols <- c("at","dltt","dlc","capx","ebitda","sale","tlcf","ib","xsga")
df_main <- df |> filter(review_year >= 2005 & review_year <= 2022)
df_ss <- df_main |> filter(!is.na(emp) & emp > 0)
df_ss <- df_ss[complete.cases(df_ss[,ctrl_cols]),]
cat(sprintf("Same-sample (emp>0 + all controls): %d rows\n", nrow(df_ss)))

rows <- list()
for (y in DV_LEAD) {
  # A: NO controls, same sample
  fa <- as.formula(paste0(y," ~ UNIONIZATION_EMP_cap1 | gvkey + review_year"))
  fit_a <- feols(fa, data=df_ss, cluster=~gvkey, warn=FALSE, notes=FALSE)
  ra <- coeftable(fit_a)["UNIONIZATION_EMP_cap1",]

  # B: WITH controls, same sample
  fb <- as.formula(paste0(y," ~ UNIONIZATION_EMP_cap1 + SIZE + LEVERAGE + CAPEX + EBITDA + SGA + NOLCF + LOSS | gvkey + review_year"))
  fit_b <- feols(fb, data=df_ss, cluster=~gvkey, warn=FALSE, notes=FALSE)
  rb <- coeftable(fit_b)["UNIONIZATION_EMP_cap1",]

  rows[[length(rows)+1]] <- data.frame(
    dv=y, spec="A_noctrl", coef=ra["Estimate"], se=se(fit_a)["UNIONIZATION_EMP_cap1"],
    tstat=ra["Estimate"]/se(fit_a)["UNIONIZATION_EMP_cap1"], pvalue=ra["Pr(>|t|)"],
    n_obs=nobs(fit_a), r2=r2(fit_a,"r2"), r2_within=r2(fit_a,"wr2"))

  rows[[length(rows)+1]] <- data.frame(
    dv=y, spec="B_ctrl", coef=rb["Estimate"], se=se(fit_b)["UNIONIZATION_EMP_cap1"],
    tstat=rb["Estimate"]/se(fit_b)["UNIONIZATION_EMP_cap1"], pvalue=rb["Pr(>|t|)"],
    n_obs=nobs(fit_b), r2=r2(fit_b,"r2"), r2_within=r2(fit_b,"wr2"))
}

res <- bind_rows(rows)
write_csv(res, paste0(OUT,"same_sample_ab.csv"))

# Verify same sample
for (dv in DV_LEAD) {
  na <- res$n_obs[res$dv==dv & res$spec=="A_noctrl"]
  nb <- res$n_obs[res$dv==dv & res$spec=="B_ctrl"]
  if (na != nb) cat(sprintf("*** SAMPLE MISMATCH: %s A=%d B=%d ***\n", dv, na, nb))
}
cat(sprintf("Same-sample check: %s\n", if(all(res$n_obs[res$spec=="A_noctrl"] == res$n_obs[res$spec=="B_ctrl"])) "ALL MATCH" else "MISMATCH!"))

# Key results
cat("\nKey outcomes (same sample):\n")
for (dv in c("overall_rating_lead1","wlb_lead1","comp_benefit_lead1")) {
  a <- res[res$dv==dv & res$spec=="A_noctrl",]
  b <- res[res$dv==dv & res$spec=="B_ctrl",]
  cat(sprintf("  %-30s A: coef=%.4f p=%.4f  B: coef=%.4f p=%.4f\n",
    dv, a$coef, a$pvalue, b$coef, b$pvalue))
}

# ====== STEP 2: Sample composition ======
cat("\n=== STEP 2: Sample composition ===\n")
df_all <- df_main |> filter(!is.na(emp) & emp > 0)
df_all$in_ctrl_sample <- complete.cases(df_all[,ctrl_cols])

comp_rows <- list()
for (v in c("at","emp","review_year","UNIONIZATION_EMP_cap1")) {
  in_s <- df_all[[v]][df_all$in_ctrl_sample]; out_s <- df_all[[v]][!df_all$in_ctrl_sample]
  comp_rows[[length(comp_rows)+1]] <- data.frame(
    variable=v, in_mean=mean(in_s,na.rm=TRUE), in_median=median(in_s,na.rm=TRUE),
    out_mean=mean(out_s,na.rm=TRUE), out_median=median(out_s,na.rm=TRUE))
}
# Also n_reviews
if ("n_reviews_all" %in% colnames(df_all)) {
  comp_rows[[length(comp_rows)+1]] <- data.frame(
    variable="n_reviews", in_mean=mean(df_all$n_reviews_all[df_all$in_ctrl_sample],na.rm=TRUE),
    in_median=median(df_all$n_reviews_all[df_all$in_ctrl_sample],na.rm=TRUE),
    out_mean=mean(df_all$n_reviews_all[!df_all$in_ctrl_sample],na.rm=TRUE),
    out_median=median(df_all$n_reviews_all[!df_all$in_ctrl_sample],na.rm=TRUE))
}
comp <- bind_rows(comp_rows)
comp$n_in <- sum(df_all$in_ctrl_sample); comp$n_out <- sum(!df_all$in_ctrl_sample)
write_csv(comp, paste0(OUT,"sample_composition.csv"))
print(comp)

# ====== STEP 3: Size split ======
cat("\n=== STEP 3: Size split ===\n")
med_emp <- median(df_ss$emp, na.rm=TRUE)
cat(sprintf("EMP median: %.1f (thousands)\n", med_emp))

split_rows <- list()
for (grp in c("small","large")) {
  dg <- if(grp=="small") df_ss[df_ss$emp <= med_emp,] else df_ss[df_ss$emp > med_emp,]
  cat(sprintf("  %s: %d rows\n", grp, nrow(dg)))
  for (y in c("overall_rating_lead1","wlb_lead1")) {
    f <- as.formula(paste0(y," ~ UNIONIZATION_EMP_cap1 | gvkey + review_year"))
    fit <- feols(f, data=dg, cluster=~gvkey, warn=FALSE, notes=FALSE)
    r <- coeftable(fit)["UNIONIZATION_EMP_cap1",]
    split_rows[[length(split_rows)+1]] <- data.frame(
      group=grp, dv=y, coef=r["Estimate"], se=se(fit)["UNIONIZATION_EMP_cap1"],
      tstat=r["Estimate"]/se(fit)["UNIONIZATION_EMP_cap1"], pvalue=r["Pr(>|t|)"],
      n_obs=nobs(fit), r2=r2(fit,"r2"), r2_within=r2(fit,"wr2"))
  }
}
splits <- bind_rows(split_rows)
write_csv(splits, paste0(OUT,"size_split.csv"))
print(splits)

cat("\nDone.\n")
