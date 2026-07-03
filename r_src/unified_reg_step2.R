#!/usr/bin/env Rscript
# STEP 2: Full regressions on UNIFIED panel (all windows + noT3 sensitivity)
library(fixest); library(dplyr); library(tidyr); library(readr); library(nanoparquet)
options(fixest_notes=FALSE); setFixest_notes(FALSE)

OUT <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/firmyear_unified/"

cat("Loading...\n")
df <- nanoparquet::read_parquet(paste0(OUT,"unified_panel.parquet"))
cat(sprintf("Rows: %d, gvkeys: %d\n", nrow(df), length(unique(df$gvkey))))

df <- df |> mutate(
  gvkey=as.character(gvkey), review_year=as.integer(review_year),
  UNIONIZATION_cap1=ifelse(is.na(UNIONIZATION),0,UNIONIZATION),
  UNIONIZATION_binary=ifelse(UNIONIZATION_cap1>0,1,0),
  UNIONIZATION_raw=ifelse(is.na(UNIONIZATION_raw),0,UNIONIZATION_raw),
  sic2=substr(as.character(sic),1,2))

winsor <- function(x, p=0.01) {
  q <- quantile(x, c(p,1-p), na.rm=TRUE); pmax(q[1], pmin(q[2], x))
}
df <- df |> mutate(
  SIZE=winsor(at,0.01), LEVERAGE=winsor((dlc+dltt)/at,0.01),
  CAPEX=winsor(capx/at,0.01), EBITDA=winsor(ebitda/at,0.01),
  SGA=winsor(xsga/sale,0.01), NOLCF=ifelse(tlcf/at>0,1,0), LOSS=ifelse(ib<0,1,0),
  CHG_NOLCF=0)

DV_ALL <- c("overall_rating","career_opp","comp_benefit","senior_mgmt","wlb","culture",
            "recommend","business_outlook","ceo_approval","diversity")
DV_CUR <- c("overall_rating_cur","career_opp_cur","comp_benefit_cur","senior_mgmt_cur",
            "wlb_cur","culture_cur","recommend_cur","business_outlook_cur","ceo_approval_cur","diversity_cur")

df <- df |> arrange(gvkey, review_year) |> group_by(gvkey) |>
  mutate(across(all_of(c(DV_ALL,DV_CUR)), ~ dplyr::lead(.x,1), .names="{.col}_lead1")) |> ungroup()

ctrls <- "+ SIZE + LEVERAGE + CAPEX + EBITDA + SGA + NOLCF + LOSS"
DV_LEAD <- paste0(DV_ALL, "_lead1")

run_ladder <- function(data, y, prefix) {
  rows <- list()
  for (li in c("L1","L2","L3","L4")) {
    if (li=="L1") f <- as.formula(paste0(y," ~ UNIONIZATION_cap1 | gvkey + sic2^review_year"))
    else if (li=="L2") f <- as.formula(paste0(y," ~ UNIONIZATION_cap1 | gvkey + review_year"))
    else if (li=="L3") f <- as.formula(paste0(y," ~ UNIONIZATION_cap1 | sic2^review_year"))
    else f <- as.formula(paste0(y," ~ UNIONIZATION_cap1 | review_year"))
    fit <- tryCatch(feols(f, data=data, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e)NULL)
    if (is.null(fit)) { rows[[length(rows)+1]] <- data.frame(model=paste0(li,prefix), dv=y, coef=NA, se=NA, pvalue=NA, n_obs=NA, n_firms=NA, dropped="feols error"); next }
    r <- tryCatch(coeftable(fit)["UNIONIZATION_cap1",], error=function(e)NULL)
    if (is.null(r)||is.na(r["Estimate"])) { rows[[length(rows)+1]] <- data.frame(model=paste0(li,prefix), dv=y, coef=NA, se=NA, pvalue=NA, n_obs=nobs(fit), n_firms=NA, dropped="coef NA/dropped"); next }
    rows[[length(rows)+1]] <- data.frame(model=paste0(li,prefix), dv=y, coef=r["Estimate"], se=r["Std.Error"], pvalue=r["Pr(>|t|)"], n_obs=nobs(fit), n_firms=length(unique(data$gvkey)), dropped="")
  }
  bind_rows(rows)
}

# ====== 1. MAIN LADDER (t in [2005,2022]) ======
cat("\n=== 1. Main Ladder ===\n")
df_main <- df |> filter(review_year >= 2005 & review_year <= 2022)
ladder_rows <- list()
for (y in DV_LEAD) { cat(sprintf("  %s\n", y)); ladder_rows[[length(ladder_rows)+1]] <- run_ladder(df_main, y, "") }
ladder <- bind_rows(ladder_rows)
write_csv(ladder, paste0(OUT,"unified_reg_ladder.csv"))
cat(sprintf("Main ladder: %d rows\n", nrow(ladder)))

# ====== 2. CONTROLS ======
cat("\n=== 2. Controls ===\n")
ctrl_rows <- list()
for (y in DV_LEAD) {
  cat(sprintf("  %s\n", y))
  for (li in c("L1","L2","L3","L4")) {
    if (li=="L1") f <- as.formula(paste0(y," ~ UNIONIZATION_cap1",ctrls," | gvkey + sic2^review_year"))
    else if (li=="L2") f <- as.formula(paste0(y," ~ UNIONIZATION_cap1",ctrls," | gvkey + review_year"))
    else if (li=="L3") f <- as.formula(paste0(y," ~ UNIONIZATION_cap1",ctrls," | sic2^review_year"))
    else f <- as.formula(paste0(y," ~ UNIONIZATION_cap1",ctrls," | review_year"))
    fit <- tryCatch(feols(f, data=df_main, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e)NULL)
    if (is.null(fit)) { ctrl_rows[[length(ctrl_rows)+1]] <- data.frame(model=li, dv=y, coef=NA, se=NA, pvalue=NA, n_obs=NA, n_firms=NA, dropped="error"); next }
    r <- tryCatch(coeftable(fit)["UNIONIZATION_cap1",], error=function(e)NULL)
    dropped_info <- if(is.null(r)||is.na(r["Estimate"])) "coef NA/dropped" else ""
    ctrl_rows[[length(ctrl_rows)+1]] <- data.frame(model=li, dv=y, coef=if(is.null(r))NA else r["Estimate"], se=if(is.null(r))NA else r["Std.Error"], pvalue=if(is.null(r))NA else r["Pr(>|t|)"], n_obs=nobs(fit), n_firms=length(unique(df_main$gvkey)), dropped=dropped_info)
  }
}
ctrls_df <- bind_rows(ctrl_rows)
write_csv(ctrls_df, paste0(OUT,"unified_reg_controls.csv"))
cat(sprintf("Controls: %d rows\n", nrow(ctrls_df)))

# ====== 3. ROBUSTNESS ======
cat("\n=== 3. Robustness ===\n")
rob_rows <- list()
for (y in DV_LEAD) {
  # binary L2
  f <- as.formula(paste0(y," ~ UNIONIZATION_binary | gvkey + review_year"))
  fit <- tryCatch(feols(f, data=df_main, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e)NULL)
  if (!is.null(fit)) { r <- coeftable(fit)["UNIONIZATION_binary",]; rob_rows[[length(rob_rows)+1]] <- data.frame(spec="binary_L2", dv=y, coef=r["Estimate"], se=r["Std.Error"], pvalue=r["Pr(>|t|)"], n_obs=nobs(fit)) }
  # raw L2
  f <- as.formula(paste0(y," ~ UNIONIZATION_raw | gvkey + review_year"))
  fit <- tryCatch(feols(f, data=df_main, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e)NULL)
  if (!is.null(fit)) { r <- coeftable(fit)["UNIONIZATION_raw",]; rob_rows[[length(rob_rows)+1]] <- data.frame(spec="raw_L2", dv=y, coef=r["Estimate"], se=r["Std.Error"], pvalue=r["Pr(>|t|)"], n_obs=nobs(fit)) }
  # current L2
  cur_col <- paste0(gsub("_lead1","",y), "_cur_lead1")
  if (cur_col %in% colnames(df_main)) {
    f <- as.formula(paste0(cur_col," ~ UNIONIZATION_cap1 | gvkey + review_year"))
    fit <- tryCatch(feols(f, data=df_main, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e)NULL)
    if (!is.null(fit)) { r <- coeftable(fit)["UNIONIZATION_cap1",]; rob_rows[[length(rob_rows)+1]] <- data.frame(spec="current_L2", dv=cur_col, coef=r["Estimate"], se=r["Std.Error"], pvalue=r["Pr(>|t|)"], n_obs=nobs(fit)) }
  }
  # contemporaneous L2
  y_ct <- gsub("_lead1","",y)
  f <- as.formula(paste0(y_ct," ~ UNIONIZATION_cap1 | gvkey + review_year"))
  fit <- tryCatch(feols(f, data=df_main, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e)NULL)
  if (!is.null(fit)) { r <- coeftable(fit)["UNIONIZATION_cap1",]; rob_rows[[length(rob_rows)+1]] <- data.frame(spec="contemporaneous_L2", dv=y_ct, coef=r["Estimate"], se=r["Std.Error"], pvalue=r["Pr(>|t|)"], n_obs=nobs(fit)) }
}
rob <- bind_rows(rob_rows)
write_csv(rob, paste0(OUT,"unified_reg_robustness.csv"))
cat(sprintf("Robustness: %d rows\n", nrow(rob)))

# ====== 4. noT3 SENSITIVITY ======
cat("\n=== 4. noT3 Sensitivity ===\n")
# Rebuild panel excluding Tier3 gvkeys from notice data
panel_base <- nanoparquet::read_parquet(paste0(OUT,"unified_panel_base.parquet"))
T3_GVK <- c(1004,1006,1010,1015,1018,1021,1025,1033,1035,1037,1040,1045,1050,1055,1061,1063,1067,
            1072,1077,1081,1083,1086,1090,1093,1097,1099,1101,1103,1105,1108,1110,1114,1117,1119,
            1122,1124,1126,1129,1132,1135,1138,1140,1145,1148,1151,1156,1160,1163,1167,1172,1175,
            1178,1180,1182,1185,1189,1193,1196,1199,1202,1205,1208,1211,1213,1218,1220,1224,1228,
            1232,1234,1237,1241,1244,1248,1250,1253,1256,1260,1263,1267,1270,1272,1275,1278,1282,
            1285,1290,1293,1297,1300,1302,1305,1309,1312,1315,1318,1320,1323,1325,1328,1331,1334,
            1337,1340,1342,1345,1348,1351,1354,1358,1360,1363,1366,1369,1372,1374,1377,1380,1383,
            1385,1388,1392,1395,1398,1400,1403,1405,1408,1411,1414,1417,1420,1423,1427,1429,1433,
            1436,1439,1442,1445,1449,1452,1456,1458,1460,1463,1466,1469)

# Actually, load from match file properly
em <- read_csv("/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/unionization_extension/employer_gvkey_matches.csv", show_col_types=FALSE)
t1t2_gvk <- unique(em$gvkey[em$match_tier %in% c(1,2)])
t3_gvk <- unique(em$gvkey[em$match_tier == 3])
t3_only <- setdiff(t3_gvk, t1t2_gvk)
cat(sprintf("Tier3-only gvkeys: %d\n", length(t3_only)))

# Exclude Tier3-only: set UNIONIZATION to NA → rows dropped by fixest
df_noT3 <- df_main
t3_char <- as.character(t3_only)
df_noT3$UNIONIZATION_cap1[df_noT3$gvkey %in% t3_char] <- NA
df_noT3$UNIONIZATION_binary[df_noT3$gvkey %in% t3_char] <- NA
df_noT3$UNIONIZATION_raw[df_noT3$gvkey %in% t3_char] <- NA
cat(sprintf("Tier3 rows set to NA: %d\n", sum(df_main$gvkey %in% t3_char)))

noT3_rows <- list()
for (y in DV_LEAD) {
  cat(sprintf("  %s\n", y))
  noT3_rows[[length(noT3_rows)+1]] <- run_ladder(df_noT3, y, "_noT3")
}
noT3 <- bind_rows(noT3_rows)
write_csv(noT3, paste0(OUT,"unified_reg_ladder_noT3.csv"))
cat(sprintf("noT3: %d rows\n", nrow(noT3)))

# Anti-copy check: n_obs must be strictly smaller
for (y in DV_LEAD) {
  main_n <- ladder$n_obs[ladder$model=="L2" & ladder$dv==y]
  noT3_n <- noT3$n_obs[noT3$model=="L2_noT3" & noT3$dv==y]
  main_coef <- ladder$coef[ladder$model=="L2" & ladder$dv==y]
  noT3_coef <- noT3$coef[noT3$model=="L2_noT3" & noT3$dv==y]
  if (!is.na(main_n) && !is.na(noT3_n) && noT3_n >= main_n) cat(sprintf("  *** ANTI-COPY FAIL: %s noT3 n_obs=%d >= main n_obs=%d ***\n", y, noT3_n, main_n))
  if (!is.na(main_coef) && !is.na(noT3_coef) && abs(main_coef - noT3_coef) < 1e-8) cat(sprintf("  *** ANTI-COPY FAIL: %s coef identical ***\n", y))
}

# ====== 5. WINDOW A: 2005-2017 ======
cat("\n=== 5. Window A (2005-2017) ===\n")
df_wA <- df |> filter(review_year >= 2005 & review_year <= 2017)
wA_rows <- list()
for (y in DV_LEAD) { cat(sprintf("  %s\n", y)); wA_rows[[length(wA_rows)+1]] <- run_ladder(df_wA, y, "_wA") }
wA <- bind_rows(wA_rows)
write_csv(wA, paste0(OUT,"unified_reg_ladder_w2005_2017.csv"))
cat(sprintf("Window A: %d rows\n", nrow(wA)))

# ====== 6. WINDOW B: Carry-forward ======
cat("\n=== 6. Window B (carry-forward) ===\n")
df_cf <- df |> filter(review_year >= 2005 & review_year <= 2024)
# Fill 2023-2024 UNIONIZATION with 2022 values
df_cf <- df_cf |> arrange(gvkey, review_year) |> group_by(gvkey) |>
  mutate(
    un2022 = ifelse(any(review_year==2022), UNIONIZATION_cap1[review_year==2022][1], 0),
    UNIONIZATION_cap1 = ifelse(review_year %in% 2023:2024, un2022, UNIONIZATION_cap1)
  ) |> ungroup()
cf_rows <- list()
for (y in DV_LEAD) {
  f <- as.formula(paste0(y," ~ UNIONIZATION_cap1 | gvkey + review_year"))
  fit <- tryCatch(feols(f, data=df_cf, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e)NULL)
  if (!is.null(fit)) {
    r <- coeftable(fit)["UNIONIZATION_cap1",]
    cf_rows[[length(cf_rows)+1]] <- data.frame(dv=y, coef=r["Estimate"], se=r["Std.Error"], pvalue=r["Pr(>|t|)"], n_obs=nobs(fit))
  }
}
cf <- bind_rows(cf_rows)
write_csv(cf, paste0(OUT,"unified_reg_carryforward.csv"))
cat(sprintf("Carry-forward: %d rows\n", nrow(cf)))

# ====== 7. CORRELATIONS ======
cat("\n=== 7. Correlations ===\n")
cor_rows <- list()
for (y in DV_ALL) {
  ok <- !is.na(df[[y]]) & !is.na(df$UNIONIZATION_cap1)
  if (sum(ok)<20) next
  pear <- cor(df[[y]][ok], df$UNIONIZATION_cap1[ok])
  spear <- cor(df[[y]][ok], df$UNIONIZATION_cap1[ok], method="spearman")
  gt0 <- mean(df[[y]][ok & df$UNIONIZATION_cap1>0], na.rm=TRUE)
  eq0 <- mean(df[[y]][ok & df$UNIONIZATION_cap1==0], na.rm=TRUE)
  cor_rows[[length(cor_rows)+1]] <- data.frame(outcome=y, pearson=round(pear,4), spearman=round(spear,4), mean_gt0=round(gt0,4), mean_eq0=round(eq0,4), n=sum(ok))
}
cor_df <- bind_rows(cor_rows)
write_csv(cor_df, paste0(OUT,"unified_correlations.csv"))
cat(sprintf("Correlations: %d rows\n", nrow(cor_df)))

# ====== 8. DIVERSITY OVERLAP ======
cat("\n=== 8. Diversity overlap ===\n")
div_ok <- !is.na(df$diversity_lead1) & df$review_year >= 2020 & df$review_year <= 2022
n_div <- sum(div_ok)
n_div_union <- sum(div_ok & df$UNIONIZATION_cap1 > 0)
cat(sprintf("diversity (2020-2022): %d fy, %d unionized\n", n_div, n_div_union))
if (n_div_union < 100) cat("*** INSUFFICIENT ***\n")

cat("\nSTEP 2 DONE\n")
