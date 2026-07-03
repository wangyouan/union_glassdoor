#!/usr/bin/env Rscript
# Round 10: Full stats backfill + EMP clean version
library(fixest); library(dplyr); library(tidyr); library(readr); library(nanoparquet)
options(fixest_notes=FALSE); setFixest_notes(FALSE)

OUT <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/firmyear_stats_full/"
UNI2 <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/firmyear_unified_v2/"
EMP_DIR <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/firmyear_emp_denom/"
dir.create(OUT, showWarnings=FALSE, recursive=TRUE)

# ====== Shared helpers ======
DV_ALL <- c("overall_rating","career_opp","comp_benefit","senior_mgmt","wlb","culture",
            "recommend","business_outlook","ceo_approval","diversity")
DV_CUR <- c("overall_rating_cur","career_opp_cur","comp_benefit_cur","senior_mgmt_cur",
            "wlb_cur","culture_cur","recommend_cur","business_outlook_cur","ceo_approval_cur","diversity_cur")

prep_df <- function(df) {
  df <- df |> mutate(
    gvkey=as.character(gvkey), review_year=as.integer(review_year),
    sic2=substr(as.character(sic),1,2))
  winsor <- function(x,p=0.01) { q <- quantile(x,c(p,1-p),na.rm=TRUE); pmax(q[1],pmin(q[2],x)) }
  df <- df |> mutate(
    SIZE=winsor(at,0.01), LEVERAGE=winsor((dlc+dltt)/at,0.01),
    CAPEX=winsor(capx/at,0.01), EBITDA=winsor(ebitda/at,0.01),
    SGA=winsor(xsga/sale,0.01), NOLCF=ifelse(tlcf/at>0,1,0), LOSS=ifelse(ib<0,1,0))
  df |> arrange(gvkey,review_year) |> group_by(gvkey) |>
    mutate(across(all_of(c(DV_ALL,DV_CUR)),~dplyr::lead(.x,1),.names="{.col}_lead1")) |> ungroup()
}

# Extended output columns: model, dv, coef, se, tstat, pvalue, n_obs, n_firms, r2, r2_within, dropped
run_ladder <- function(data, y, var_name, model_lbl) {
  rows <- list()
  for (li in c("L1","L2","L3","L4")) {
    if (li=="L1") f <- as.formula(paste0(y," ~ ",var_name," | gvkey + sic2^review_year"))
    else if (li=="L2") f <- as.formula(paste0(y," ~ ",var_name," | gvkey + review_year"))
    else if (li=="L3") f <- as.formula(paste0(y," ~ ",var_name," | sic2^review_year"))
    else f <- as.formula(paste0(y," ~ ",var_name," | review_year"))
    fit <- tryCatch(feols(f, data=data, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e)NULL)
    if (is.null(fit)) {
      rows[[length(rows)+1]] <- data.frame(model=paste0(li,model_lbl), dv=y, coef=NA, se=NA, tstat=NA, pvalue=NA, n_obs=NA, n_firms=NA, r2=NA, r2_within=NA, dropped="feols error")
      next
    }
    ct <- coeftable(fit)
    if (!var_name %in% rownames(ct)) {
      rows[[length(rows)+1]] <- data.frame(model=paste0(li,model_lbl), dv=y, coef=NA, se=NA, tstat=NA, pvalue=NA, n_obs=nobs(fit), n_firms=length(unique(data$gvkey)), r2=r2(fit,"r2"), r2_within=r2(fit,"wr2"), dropped=paste("var not in coefs:",paste(rownames(ct)[1:min(3,nrow(ct))],collapse=",")))
      next
    }
    r <- ct[var_name,]
    rows[[length(rows)+1]] <- data.frame(model=paste0(li,model_lbl), dv=y,
      coef=r["Estimate"], se=se(fit)[var_name], tstat=r["Estimate"]/se(fit)[var_name],
      pvalue=r["Pr(>|t|)"], n_obs=nobs(fit), n_firms=length(unique(data$gvkey)),
      r2=r2(fit,"r2"), r2_within=r2(fit,"wr2"), dropped="")
  }
  bind_rows(rows)
}

run_l2 <- function(data, y, var_name, model_lbl) {
  f <- as.formula(paste0(y," ~ ",var_name," | gvkey + review_year"))
  fit <- tryCatch(feols(f, data=data, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e)NULL)
  if (is.null(fit)) return(data.frame(model=model_lbl, dv=y, coef=NA, se=NA, tstat=NA, pvalue=NA, n_obs=NA, n_firms=NA, r2=NA, r2_within=NA, dropped="feols error"))
  ct <- coeftable(fit)
  if (!var_name %in% rownames(ct)) return(data.frame(model=model_lbl, dv=y, coef=NA, se=NA, tstat=NA, pvalue=NA, n_obs=nobs(fit), n_firms=length(unique(data$gvkey)), r2=r2(fit,"r2"), r2_within=r2(fit,"wr2"), dropped="var dropped"))
  r <- ct[var_name,]
  data.frame(model=model_lbl, dv=y, coef=r["Estimate"], se=se(fit)[var_name], tstat=r["Estimate"]/se(fit)[var_name], pvalue=r["Pr(>|t|)"], n_obs=nobs(fit), n_firms=length(unique(data$gvkey)), r2=r2(fit,"r2"), r2_within=r2(fit,"wr2"), dropped="")
}

ctrls <- "+ SIZE + LEVERAGE + CAPEX + EBITDA + SGA + NOLCF + LOSS"
DV_LEAD <- paste0(DV_ALL, "_lead1")

# ==================== STEP 1: EST (round 8) full stats ====================
cat("=== STEP 1: EST full stats ===\n")
df_est <- prep_df(nanoparquet::read_parquet(paste0(UNI2,"unified2_panel.parquet")))
df_est_main <- df_est |> filter(review_year >= 2005 & review_year <= 2022)

# Ladder
cat("  Ladder...\n")
est_ladder <- bind_rows(lapply(DV_LEAD, function(y) run_ladder(df_est_main, y, "UNIONIZATION_cap1", "")))
write_csv(est_ladder, paste0(OUT,"unified2_reg_ladder_full.csv"))

# Controls
cat("  Controls...\n")
est_ctrl <- bind_rows(lapply(DV_LEAD, function(y) {
  bind_rows(lapply(c("L1","L2","L3","L4"), function(li) {
    if (li=="L1") f <- as.formula(paste0(y," ~ UNIONIZATION_cap1",ctrls," | gvkey + sic2^review_year"))
    else if (li=="L2") f <- as.formula(paste0(y," ~ UNIONIZATION_cap1",ctrls," | gvkey + review_year"))
    else if (li=="L3") f <- as.formula(paste0(y," ~ UNIONIZATION_cap1",ctrls," | sic2^review_year"))
    else f <- as.formula(paste0(y," ~ UNIONIZATION_cap1",ctrls," | review_year"))
    fit <- tryCatch(feols(f, data=df_est_main, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e)NULL)
    if (is.null(fit)) return(data.frame(model=li, dv=y, coef=NA, se=NA, tstat=NA, pvalue=NA, n_obs=NA, n_firms=NA, r2=NA, r2_within=NA, dropped="error"))
    ct <- coeftable(fit)
    if (!"UNIONIZATION_cap1" %in% rownames(ct)) return(data.frame(model=li, dv=y, coef=NA, se=NA, tstat=NA, pvalue=NA, n_obs=nobs(fit), n_firms=NA, r2=r2(fit,"r2"), r2_within=r2(fit,"wr2"), dropped="var dropped"))
    r <- ct["UNIONIZATION_cap1",]
    data.frame(model=li, dv=y, coef=r["Estimate"], se=se(fit)["UNIONIZATION_cap1"], tstat=r["Estimate"]/se(fit)["UNIONIZATION_cap1"], pvalue=r["Pr(>|t|)"], n_obs=nobs(fit), n_firms=length(unique(df_est_main$gvkey)), r2=r2(fit,"r2"), r2_within=r2(fit,"wr2"), dropped="")
  }))
}))
write_csv(est_ctrl, paste0(OUT,"unified2_reg_controls_full.csv"))

# Robustness
cat("  Robustness...\n")
rob_rows <- list()
for (y in DV_LEAD) {
  for (spec in c("binary_L2","raw_L2","current_L2","contemporaneous_L2")) {
    if (spec=="binary_L2") { vn <- "UNIONIZATION_binary"; f <- as.formula(paste0(y," ~ UNIONIZATION_binary | gvkey + review_year")) }
    else if (spec=="raw_L2") { vn <- "UNIONIZATION_raw"; f <- as.formula(paste0(y," ~ UNIONIZATION_raw | gvkey + review_year")) }
    else if (spec=="current_L2") {
      cur_col <- paste0(gsub("_lead1","",y),"_cur_lead1")
      if (!cur_col %in% colnames(df_est_main)) next
      vn <- "UNIONIZATION_cap1"; f <- as.formula(paste0(cur_col," ~ UNIONIZATION_cap1 | gvkey + review_year"))
      y_use <- cur_col
    } else { vn <- "UNIONIZATION_cap1"; y_use <- gsub("_lead1","",y); f <- as.formula(paste0(y_use," ~ UNIONIZATION_cap1 | gvkey + review_year")) }
    if (!exists("y_use")) y_use <- y
    fit <- tryCatch(feols(f, data=df_est_main, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e)NULL)
    if (!is.null(fit) && vn %in% rownames(coeftable(fit))) {
      r <- coeftable(fit)[vn,]
      rob_rows[[length(rob_rows)+1]] <- data.frame(spec=spec, dv=y_use, coef=r["Estimate"], se=se(fit)[vn], tstat=r["Estimate"]/se(fit)[vn], pvalue=r["Pr(>|t|)"], n_obs=nobs(fit), n_firms=length(unique(df_est_main$gvkey)), r2=r2(fit,"r2"), r2_within=r2(fit,"wr2"), dropped="")
    }
    rm(y_use)
  }
}
rob_est <- bind_rows(rob_rows)
write_csv(rob_est, paste0(OUT,"unified2_reg_robustness_full.csv"))

# noT3
cat("  noT3...\n")
em <- read_csv("/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/unionization_extension/employer_gvkey_matches.csv", show_col_types=FALSE)
t1t2 <- unique(em$gvkey[em$match_tier %in% c(1,2)]); t3 <- unique(em$gvkey[em$match_tier==3])
t3_only <- setdiff(t3, t1t2)
df_noT3 <- df_est_main
df_noT3$UNIONIZATION_cap1[df_noT3$gvkey %in% as.character(t3_only)] <- NA
est_noT3 <- bind_rows(lapply(DV_LEAD, function(y) run_ladder(df_noT3, y, "UNIONIZATION_cap1", "_noT3")))
write_csv(est_noT3, paste0(OUT,"unified2_reg_ladder_noT3_full.csv"))

# Window A
cat("  Window A...\n")
df_wA <- df_est |> filter(review_year >= 2005 & review_year <= 2017)
est_wA <- bind_rows(lapply(DV_LEAD, function(y) run_ladder(df_wA, y, "UNIONIZATION_cap1", "_wA")))
write_csv(est_wA, paste0(OUT,"unified2_reg_ladder_w2005_2017_full.csv"))

# Carry-forward
cat("  Carry-forward...\n")
df_cf <- df_est |> filter(review_year>=2005 & review_year<=2024) |> arrange(gvkey,review_year) |> group_by(gvkey) |>
  mutate(un2022=ifelse(any(review_year==2022),UNIONIZATION_cap1[review_year==2022][1],0),
         UNIONIZATION_cap1=ifelse(review_year %in% 2023:2024,un2022,UNIONIZATION_cap1)) |> ungroup()
est_cf <- bind_rows(lapply(DV_LEAD, function(y) run_l2(df_cf, y, "UNIONIZATION_cap1", "L2_cf")))
write_csv(est_cf, paste0(OUT,"unified2_reg_carryforward_full.csv"))

# ====== Verify coef/p match ======
cat("\n=== STEP 1 Verification ===\n")
verify <- function(new_path, old_path, label) {
  new <- read_csv(new_path, show_col_types=FALSE)
  old <- read_csv(old_path, show_col_types=FALSE)
  # Match by model+spec and dv
  by_cols <- intersect(names(new), names(old))
  by_cols <- intersect(by_cols, c("model","spec","dv"))
  m <- merge(new, old, by=by_cols, suffixes=c("_new","_old"), all=TRUE)
  if (!"coef_new" %in% names(m) || all(is.na(m$coef_new)) || all(is.na(m$coef_old))) {
    cat(sprintf("  %-50s SKIP (no overlapping coef)\n", label))
    return(0)
  }
  max_d <- max(abs(m$coef_new - m$coef_old), na.rm=TRUE)
  cat(sprintf("  %-50s max|coef diff|=%.2e %s\n", label, max_d, if(max_d<1e-10) "PASS" else "FAIL"))
  max_d
}

diffs <- c()
diffs[1] <- verify(paste0(OUT,"unified2_reg_ladder_full.csv"), paste0(UNI2,"unified2_reg_ladder.csv"), "EST ladder")
diffs[2] <- verify(paste0(OUT,"unified2_reg_controls_full.csv"), paste0(UNI2,"unified2_reg_controls.csv"), "EST controls")
diffs[3] <- verify(paste0(OUT,"unified2_reg_robustness_full.csv"), paste0(UNI2,"unified2_reg_robustness.csv"), "EST robustness")
diffs[4] <- verify(paste0(OUT,"unified2_reg_ladder_noT3_full.csv"), paste0(UNI2,"unified2_reg_ladder_noT3.csv"), "EST noT3")
diffs[5] <- verify(paste0(OUT,"unified2_reg_ladder_w2005_2017_full.csv"), paste0(UNI2,"unified2_reg_ladder_w2005_2017.csv"), "EST windowA")
diffs[6] <- verify(paste0(OUT,"unified2_reg_carryforward_full.csv"), paste0(UNI2,"unified2_reg_carryforward.csv"), "EST carryforward")

# ==================== STEP 2: EMP full stats ====================
cat("\n=== STEP 2: EMP full stats ===\n")
df_emp <- prep_df(nanoparquet::read_parquet(paste0(EMP_DIR,"emp_panel.parquet")))
df_emp_main <- df_emp |> filter(review_year >= 2005 & review_year <= 2022)

cat("  Ladder...\n")
emp_ladder <- bind_rows(lapply(DV_LEAD, function(y) run_ladder(df_emp_main, y, "UNIONIZATION_EMP_cap1", "_emp")))
write_csv(emp_ladder, paste0(OUT,"emp_reg_ladder_full.csv"))

cat("  Controls...\n")
emp_ctrl <- bind_rows(lapply(DV_LEAD, function(y) {
  bind_rows(lapply(c("L1","L2","L3","L4"), function(li) {
    if (li=="L1") f <- as.formula(paste0(y," ~ UNIONIZATION_EMP_cap1",ctrls," | gvkey + sic2^review_year"))
    else if (li=="L2") f <- as.formula(paste0(y," ~ UNIONIZATION_EMP_cap1",ctrls," | gvkey + review_year"))
    else if (li=="L3") f <- as.formula(paste0(y," ~ UNIONIZATION_EMP_cap1",ctrls," | sic2^review_year"))
    else f <- as.formula(paste0(y," ~ UNIONIZATION_EMP_cap1",ctrls," | review_year"))
    fit <- tryCatch(feols(f, data=df_emp_main, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e)NULL)
    if (is.null(fit)) return(data.frame(model=li, dv=y, coef=NA, se=NA, tstat=NA, pvalue=NA, n_obs=NA, n_firms=NA, r2=NA, r2_within=NA, dropped="error"))
    ct <- coeftable(fit)
    if (!"UNIONIZATION_EMP_cap1" %in% rownames(ct)) return(data.frame(model=li, dv=y, coef=NA, se=NA, tstat=NA, pvalue=NA, n_obs=nobs(fit), n_firms=NA, r2=r2(fit,"r2"), r2_within=r2(fit,"wr2"), dropped="var dropped"))
    r <- ct["UNIONIZATION_EMP_cap1",]
    data.frame(model=li, dv=y, coef=r["Estimate"], se=se(fit)["UNIONIZATION_EMP_cap1"], tstat=r["Estimate"]/se(fit)["UNIONIZATION_EMP_cap1"], pvalue=r["Pr(>|t|)"], n_obs=nobs(fit), n_firms=length(unique(df_emp_main$gvkey)), r2=r2(fit,"r2"), r2_within=r2(fit,"wr2"), dropped="")
  }))
}))
write_csv(emp_ctrl, paste0(OUT,"emp_reg_controls_full.csv"))

cat("  Variants...\n")
emp_variants <- bind_rows(lapply(DV_LEAD, function(y) {
  bind_rows(
    run_l2(df_emp_main, y, "UNIONIZATION_EMP_log", "log1p_L2"),
    run_l2(df_emp_main, y, "UNIONIZATION_EMP_lag", "lagEMP_L2"),
    run_l2(df_emp_main |> filter(!exclude_small_emp), y, "UNIONIZATION_EMP_cap1", "drop_small_L2"),
    run_l2(df_emp_main, y, "UNIONIZATION_EMP_raw", "raw_L2")
  )
}))
write_csv(emp_variants, paste0(OUT,"emp_reg_variants_full.csv"))

diffs[7] <- verify(paste0(OUT,"emp_reg_ladder_full.csv"), paste0(EMP_DIR,"emp_reg_ladder.csv"), "EMP ladder")
diffs[8] <- verify(paste0(OUT,"emp_reg_controls_full.csv"), paste0(EMP_DIR,"emp_reg_controls.csv"), "EMP controls")
diffs[9] <- verify(paste0(OUT,"emp_reg_variants_full.csv"), paste0(EMP_DIR,"emp_reg_variants.csv"), "EMP variants")

# ==================== STEP 3: EMP clean ====================
cat("\n=== STEP 3: EMP clean (remove EMP<=0/missing) ===\n")
df_emp_cl <- df_emp |> filter(!is.na(emp) & emp > 0)
cat(sprintf("EMP clean: %d rows (removed %d with emp<=0/missing)\n", nrow(df_emp_cl), nrow(df_emp)-nrow(df_emp_cl)))
gt1_cl <- df_emp_cl |> filter(UNIONIZATION_EMP_raw > 1 & !is.na(UNIONIZATION_EMP_raw))
cat(sprintf(">1 after clean: %d (will cap)\n", nrow(gt1_cl)))
if (nrow(gt1_cl) > 0) {
  df_emp_cl$UNIONIZATION_EMP_raw[df_emp_cl$UNIONIZATION_EMP_raw > 1 & !is.na(df_emp_cl$UNIONIZATION_EMP_raw)] <- 1.0
}

df_emp_cl_main <- df_emp_cl |> filter(review_year >= 2005 & review_year <= 2022)

cat("  Ladder...\n")
emp_cl_ladder <- bind_rows(lapply(DV_LEAD, function(y) run_ladder(df_emp_cl_main, y, "UNIONIZATION_EMP_cap1", "_empcl")))
write_csv(emp_cl_ladder, paste0(OUT,"emp_clean_reg_ladder.csv"))

cat("  Controls...\n")
emp_cl_ctrl <- bind_rows(lapply(DV_LEAD, function(y) {
  bind_rows(lapply(c("L1","L2","L3","L4"), function(li) {
    if (li=="L1") f <- as.formula(paste0(y," ~ UNIONIZATION_EMP_cap1",ctrls," | gvkey + sic2^review_year"))
    else if (li=="L2") f <- as.formula(paste0(y," ~ UNIONIZATION_EMP_cap1",ctrls," | gvkey + review_year"))
    else if (li=="L3") f <- as.formula(paste0(y," ~ UNIONIZATION_EMP_cap1",ctrls," | sic2^review_year"))
    else f <- as.formula(paste0(y," ~ UNIONIZATION_EMP_cap1",ctrls," | review_year"))
    fit <- tryCatch(feols(f, data=df_emp_cl_main, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e)NULL)
    if (is.null(fit)) return(data.frame(model=li, dv=y, coef=NA, se=NA, tstat=NA, pvalue=NA, n_obs=NA, n_firms=NA, r2=NA, r2_within=NA, dropped="error"))
    ct <- coeftable(fit)
    if (!"UNIONIZATION_EMP_cap1" %in% rownames(ct)) return(data.frame(model=li, dv=y, coef=NA, se=NA, tstat=NA, pvalue=NA, n_obs=nobs(fit), n_firms=NA, r2=r2(fit,"r2"), r2_within=r2(fit,"wr2"), dropped="var dropped"))
    r <- ct["UNIONIZATION_EMP_cap1",]
    data.frame(model=li, dv=y, coef=r["Estimate"], se=se(fit)["UNIONIZATION_EMP_cap1"], tstat=r["Estimate"]/se(fit)["UNIONIZATION_EMP_cap1"], pvalue=r["Pr(>|t|)"], n_obs=nobs(fit), n_firms=length(unique(df_emp_cl_main$gvkey)), r2=r2(fit,"r2"), r2_within=r2(fit,"wr2"), dropped="")
  }))
}))
write_csv(emp_cl_ctrl, paste0(OUT,"emp_clean_reg_controls.csv"))

cat("  Variants...\n")
emp_cl_variants <- bind_rows(lapply(DV_LEAD, function(y) {
  bind_rows(
    run_l2(df_emp_cl_main, y, "UNIONIZATION_EMP_log", "log1p_L2"),
    run_l2(df_emp_cl_main, y, "UNIONIZATION_EMP_lag", "lagEMP_L2"),
    run_l2(df_emp_cl_main |> filter(!exclude_small_emp), y, "UNIONIZATION_EMP_cap1", "drop_small_L2"),
    run_l2(df_emp_cl_main, y, "UNIONIZATION_EMP_raw", "raw_L2")
  )
}))
write_csv(emp_cl_variants, paste0(OUT,"emp_clean_reg_variants.csv"))

cat(sprintf("\nAll diffs: %s\n", paste(sprintf("%.2e",diffs), collapse=" ")))
cat(sprintf("Any FAIL (>1e-10): %s\n", if(any(diffs > 1e-10, na.rm=TRUE)) "YES - stop!" else "ALL PASS"))
