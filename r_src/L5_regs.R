#!/usr/bin/env Rscript
# Round 12: L5 = | sic2 + year FE spec
library(fixest); library(dplyr); library(tidyr); library(readr); library(nanoparquet)
options(fixest_notes=FALSE); setFixest_notes(FALSE)

OUT <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260704/firmyear_L5/"
UNI2 <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/firmyear_unified_v2/"
EMP_DIR <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/firmyear_emp_denom/"
PREV <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/firmyear_stats_full/"
dir.create(OUT, showWarnings=FALSE, recursive=TRUE)

DV <- c("overall_rating","career_opp","comp_benefit","senior_mgmt","wlb","culture",
        "recommend","business_outlook","ceo_approval","diversity")
DV_LEAD <- paste0(DV, "_lead1")
ctrls <- "+ SIZE + LEVERAGE + CAPEX + EBITDA + SGA + NOLCF + LOSS"

prep <- function(df) {
  df <- df |> mutate(gvkey=as.character(gvkey), review_year=as.integer(review_year),
                     sic2=substr(as.character(sic),1,2))
  winsor <- function(x,p=0.01) { q <- quantile(x,c(p,1-p),na.rm=TRUE); pmax(q[1],pmin(q[2],x)) }
  df <- df |> mutate(SIZE=winsor(at,0.01), LEVERAGE=winsor((dlc+dltt)/at,0.01),
    CAPEX=winsor(capx/at,0.01), EBITDA=winsor(ebitda/at,0.01),
    SGA=winsor(xsga/sale,0.01), NOLCF=ifelse(tlcf/at>0,1,0), LOSS=ifelse(ib<0,1,0))
  df |> arrange(gvkey,review_year) |> group_by(gvkey) |>
    mutate(across(all_of(DV),~dplyr::lead(.x,1),.names="{.col}_lead1")) |> ungroup()
}

run_L5 <- function(data, y, var_name, with_ctrl) {
  f <- if (with_ctrl) as.formula(paste0(y," ~ ",var_name,ctrls," | sic2 + review_year"))
       else as.formula(paste0(y," ~ ",var_name," | sic2 + review_year"))
  fit <- feols(f, data=data, cluster=~gvkey, warn=FALSE, notes=FALSE)
  r <- coeftable(fit)[var_name,]
  data.frame(model="L5", dv=y, coef=r["Estimate"], se=se(fit)[var_name],
    tstat=r["Estimate"]/se(fit)[var_name], pvalue=r["Pr(>|t|)"],
    n_obs=nobs(fit), n_firms=length(unique(data$gvkey)),
    r2=r2(fit,"r2"), r2_within=r2(fit,"wr2"), dropped="")
}

# ====== Pipeline verification ======
cat("=== Pipeline verification: EST L3 wlb vs round 10 ===\n")
df_est <- prep(nanoparquet::read_parquet(paste0(UNI2,"unified2_panel.parquet")))
df_est_main <- df_est |> filter(review_year>=2005 & review_year<=2022)
fit_l3 <- feols(wlb_lead1 ~ UNIONIZATION_cap1 | sic2^review_year, data=df_est_main, cluster=~gvkey, warn=FALSE, notes=FALSE)
r_l3 <- coeftable(fit_l3)["UNIONIZATION_cap1",]
prev_l3 <- read_csv(paste0(PREV,"unified2_reg_ladder_full.csv"), show_col_types=FALSE)
prev_wlb <- prev_l3[prev_l3$model=="L3" & prev_l3$dv=="wlb_lead1",]
diff_l3 <- abs(r_l3["Estimate"] - prev_wlb$coef)
cat(sprintf("  Current: %.6f, Prev: %.6f, diff=%.2e %s\n",
  r_l3["Estimate"], prev_wlb$coef, diff_l3, if(diff_l3<1e-10) "PASS" else "FAIL"))

# ====== 1. EST L5 baseline ======
cat("\n=== 1. EST L5 baseline ===\n")
est_bl <- bind_rows(lapply(DV_LEAD, function(y) run_L5(df_est_main, y, "UNIONIZATION_cap1", FALSE)))
write_csv(est_bl, paste0(OUT,"est_L5_baseline.csv"))

# ====== 2. EST L5 + controls ======
cat("=== 2. EST L5 + controls ===\n")
est_ctrl <- bind_rows(lapply(DV_LEAD, function(y) run_L5(df_est_main, y, "UNIONIZATION_cap1", TRUE)))
write_csv(est_ctrl, paste0(OUT,"est_L5_controls.csv"))

# ====== 3. EMP clean L5 baseline ======
cat("=== 3. EMP clean L5 baseline ===\n")
df_emp <- prep(nanoparquet::read_parquet(paste0(EMP_DIR,"emp_panel.parquet")))
df_emp_cl <- df_emp |> filter(review_year>=2005 & review_year<=2022 & !is.na(emp) & emp>0)
emp_bl <- bind_rows(lapply(DV_LEAD, function(y) run_L5(df_emp_cl, y, "UNIONIZATION_EMP_cap1", FALSE)))
write_csv(emp_bl, paste0(OUT,"emp_L5_baseline.csv"))

# ====== 4. EMP clean L5 + controls ======
cat("=== 4. EMP clean L5 + controls ===\n")
emp_ctrl <- bind_rows(lapply(DV_LEAD, function(y) run_L5(df_emp_cl, y, "UNIONIZATION_EMP_cap1", TRUE)))
write_csv(emp_ctrl, paste0(OUT,"emp_L5_controls.csv"))

# ====== Key results ======
cat("\n=== L5 Key Results ===\n")
for (label in c("EST baseline","EST +controls","EMP baseline","EMP +controls")) {
  df <- switch(label, "EST baseline"=est_bl, "EST +controls"=est_ctrl, "EMP baseline"=emp_bl, "EMP +controls"=emp_ctrl)
  for (dv in c("overall_rating_lead1","wlb_lead1","comp_benefit_lead1")) {
    r <- df[df$dv==dv,]
    cat(sprintf("  %-20s %-30s coef=% 8.4f p=%.4f n=%d\n", label, dv, r$coef, r$pvalue, r$n_obs))
  }
}

# ====== Self-check ======
cat("\n=== Self-check ===\n")
all_ok <- TRUE
for (f in c("est_L5_baseline.csv","est_L5_controls.csv","emp_L5_baseline.csv","emp_L5_controls.csv")) {
  df <- read_csv(paste0(OUT,f), show_col_types=FALSE)
  ok <- nrow(df)==10 && all(!is.na(df$se[!is.na(df$coef)])) && all(!is.na(df$r2[!is.na(df$coef)]))
  cat(sprintf("  %-30s rows=%d cols_ok=%s\n", f, nrow(df), if(ok) "PASS" else "FAIL"))
  if (!ok) all_ok <- FALSE
}
cat(sprintf("\nAll checks: %s\n", if(all_ok) "PASS" else "FAIL"))
cat("Done\n")
