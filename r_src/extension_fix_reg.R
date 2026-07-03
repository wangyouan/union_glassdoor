#!/usr/bin/env Rscript
# STEP 3: Full regressions on fixed UNIONIZATION panel
library(fixest); library(dplyr); library(tidyr); library(readr); library(nanoparquet)
options(fixest_notes=FALSE); setFixest_notes(FALSE)

OUT <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/unionization_extension_fix/"

cat("Loading...\n")
df <- nanoparquet::read_parquet(paste0(OUT,"fix_merged_panel.parquet"))
cat(sprintf("Rows: %d\n", nrow(df)))

df <- df |> mutate(
  gvkey = as.character(gvkey), review_year = as.integer(review_year),
  UNIONIZATION_cap1 = ifelse(is.na(UNIONIZATION), 0, UNIONIZATION),
  UNIONIZATION_binary = ifelse(UNIONIZATION_cap1 > 0, 1, 0),
  UNIONIZATION_raw = ifelse(is.na(UNIONIZATION_raw), 0, UNIONIZATION_raw),
  sic2 = substr(as.character(sic), 1, 2))

winsor <- function(x, p=0.01) {
  q <- quantile(x, c(p, 1-p), na.rm=TRUE); pmax(q[1], pmin(q[2], x))
}

df <- df |> mutate(
  SIZE = winsor(at, 0.01), LEVERAGE = winsor((dlc+dltt)/at, 0.01),
  CAPEX = winsor(capx/at, 0.01), EBITDA = winsor(ebitda/at, 0.01),
  SGA = winsor(xsga/sale, 0.01), NOLCF = ifelse(tlcf/at>0,1,0), LOSS = ifelse(ib<0,1,0),
  CHG_NOLCF = ifelse(is.na(tlcf), 0, 0))  # placeholder

DV_ALL <- c("overall_rating","career_opp","comp_benefit","senior_mgmt","wlb","culture",
            "recommend","business_outlook","ceo_approval","diversity")
DV_CUR <- c("overall_rating_cur","career_opp_cur","comp_benefit_cur","senior_mgmt_cur",
            "wlb_cur","culture_cur","recommend_cur","business_outlook_cur","ceo_approval_cur","diversity_cur")

df <- df |> arrange(gvkey, review_year) |> group_by(gvkey) |>
  mutate(across(all_of(c(DV_ALL, DV_CUR)), ~ dplyr::lead(.x, 1), .names = "{.col}_lead1")) |> ungroup()

controls <- "+ SIZE + LEVERAGE + CAPEX + EBITDA + SGA + NOLCF + LOSS"
DV_LEAD <- paste0(DV_ALL, "_lead1")

run_spec <- function(y, fml_str, model_label) {
  f <- as.formula(fml_str)
  fit <- tryCatch(feols(f, data=df, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e)NULL)
  if (is.null(fit)) return(data.frame(model=model_label, dv=y, coef=NA, se=NA, pvalue=NA, n_obs=NA, n_firms=NA, dropped="feols error"))
  r <- tryCatch(coeftable(fit)["UNIONIZATION_cap1",], error=function(e)NULL)
  if (is.null(r) || is.na(r["Estimate"]))
    return(data.frame(model=model_label, dv=y, coef=NA, se=NA, pvalue=NA, n_obs=nobs(fit), n_firms=NA, dropped="coef NA/dropped"))
  data.frame(model=model_label, dv=y, coef=r["Estimate"], se=r["Std. Error"], pvalue=r["Pr(>|t|)"], n_obs=nobs(fit), n_firms=length(unique(df$gvkey)), dropped="")
}

# ====== Correlations ======
cat("\n=== Correlations ===\n")
cor_rows <- list()
for (y in DV_ALL) {
  ok <- !is.na(df[[y]]) & !is.na(df$UNIONIZATION_cap1)
  if (sum(ok) < 20) next
  pear <- cor(df[[y]][ok], df$UNIONIZATION_cap1[ok])
  spear <- cor(df[[y]][ok], df$UNIONIZATION_cap1[ok], method="spearman")
  gt0 <- mean(df[[y]][ok & df$UNIONIZATION_cap1>0], na.rm=TRUE)
  eq0 <- mean(df[[y]][ok & df$UNIONIZATION_cap1==0], na.rm=TRUE)
  cor_rows[[length(cor_rows)+1]] <- data.frame(outcome=y, pearson=round(pear,4), spearman=round(spear,4), mean_gt0=round(gt0,4), mean_eq0=round(eq0,4), n=sum(ok))
}
cor_df <- bind_rows(cor_rows)
write_csv(cor_df, paste0(OUT,"fix_correlations.csv"))

# ====== L1-L4 Ladder ======
cat("\n=== L1-L4 Ladder ===\n")
ladder_rows <- list()
for (y in DV_LEAD) {
  cat(sprintf("  %s...\n", y))
  ladder_rows[[length(ladder_rows)+1]] <- run_spec(y, paste0(y," ~ UNIONIZATION_cap1 | gvkey + sic2^review_year"), "L1")
  ladder_rows[[length(ladder_rows)+1]] <- run_spec(y, paste0(y," ~ UNIONIZATION_cap1 | gvkey + review_year"), "L2")
  ladder_rows[[length(ladder_rows)+1]] <- run_spec(y, paste0(y," ~ UNIONIZATION_cap1 | sic2^review_year"), "L3")
  ladder_rows[[length(ladder_rows)+1]] <- run_spec(y, paste0(y," ~ UNIONIZATION_cap1 | review_year"), "L4")
}
ladder <- bind_rows(ladder_rows)
write_csv(ladder, paste0(OUT,"fix_reg_ladder.csv"))
cat(sprintf("Ladder: %d rows saved\n", nrow(ladder)))
l2 <- ladder[ladder$model=="L2" & !is.na(ladder$coef),]
for (i in 1:nrow(l2)) cat(sprintf("  L2 %-30s coef=% 8.4f se=%7.4f p=%6.4f\n", l2$dv[i], l2$coef[i], l2$se[i], l2$pvalue[i]))

# ====== Controls ======
cat("\n=== Controls ===\n")
ctrl_rows <- list()
for (y in DV_LEAD) {
  cat(sprintf("  %s...\n", y))
  ctrl_rows[[length(ctrl_rows)+1]] <- run_spec(y, paste0(y," ~ UNIONIZATION_cap1",controls," | gvkey + sic2^review_year"), "L1")
  ctrl_rows[[length(ctrl_rows)+1]] <- run_spec(y, paste0(y," ~ UNIONIZATION_cap1",controls," | gvkey + review_year"), "L2")
  ctrl_rows[[length(ctrl_rows)+1]] <- run_spec(y, paste0(y," ~ UNIONIZATION_cap1",controls," | sic2^review_year"), "L3")
  ctrl_rows[[length(ctrl_rows)+1]] <- run_spec(y, paste0(y," ~ UNIONIZATION_cap1",controls," | review_year"), "L4")
}
ctrls <- bind_rows(ctrl_rows)
write_csv(ctrls, paste0(OUT,"fix_reg_controls.csv"))
cat(sprintf("Controls: %d rows saved\n", nrow(ctrls)))

# ====== Robustness ======
cat("\n=== Robustness ===\n")
rob_rows <- list()
for (y in DV_LEAD[1:6]) {
  # binary L2
  f <- as.formula(paste0(y, " ~ UNIONIZATION_binary | gvkey + review_year"))
  fit <- tryCatch(feols(f, data=df, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e)NULL)
  if (!is.null(fit)) {
    r <- coeftable(fit)["UNIONIZATION_binary",]
    rob_rows[[length(rob_rows)+1]] <- data.frame(spec="binary_L2", dv=y, coef=r["Estimate"], se=r["Std. Error"], pvalue=r["Pr(>|t|)"], n_obs=nobs(fit))
  }
  # raw L2
  f <- as.formula(paste0(y, " ~ UNIONIZATION_raw | gvkey + review_year"))
  fit <- tryCatch(feols(f, data=df, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e)NULL)
  if (!is.null(fit)) {
    r <- coeftable(fit)["UNIONIZATION_raw",]
    rob_rows[[length(rob_rows)+1]] <- data.frame(spec="raw_L2", dv=y, coef=r["Estimate"], se=r["Std. Error"], pvalue=r["Pr(>|t|)"], n_obs=nobs(fit))
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
# Add remaining 4 DVs for robustness
for (y in DV_LEAD[7:10]) {
  for (spec_name in c("binary_L2","raw_L2","contemporaneous_L2")) {
    if (spec_name == "binary_L2") {
      f <- as.formula(paste0(y, " ~ UNIONIZATION_binary | gvkey + review_year"))
    } else if (spec_name == "raw_L2") {
      f <- as.formula(paste0(y, " ~ UNIONIZATION_raw | gvkey + review_year"))
    } else {
      y_ct <- gsub("_lead1","",y)
      f <- as.formula(paste0(y_ct, " ~ UNIONIZATION_cap1 | gvkey + review_year"))
    }
    fit <- tryCatch(feols(f, data=df, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e)NULL)
    if (!is.null(fit)) {
      if (spec_name == "binary_L2") rname <- "UNIONIZATION_binary"
      else if (spec_name == "raw_L2") rname <- "UNIONIZATION_raw"
      else rname <- "UNIONIZATION_cap1"
      r <- coeftable(fit)[rname,]
      rob_rows[[length(rob_rows)+1]] <- data.frame(spec=spec_name, dv=y, coef=r["Estimate"], se=r["Std. Error"], pvalue=r["Pr(>|t|)"], n_obs=nobs(fit))
    }
  }
  # current for remaining
  cur_col <- paste0(gsub("_lead1","",y), "_cur_lead1")
  if (cur_col %in% colnames(df)) {
    f <- as.formula(paste0(cur_col, " ~ UNIONIZATION_cap1 | gvkey + review_year"))
    fit <- tryCatch(feols(f, data=df, cluster=~gvkey, warn=FALSE, notes=FALSE), error=function(e)NULL)
    if (!is.null(fit)) {
      r <- coeftable(fit)["UNIONIZATION_cap1",]
      rob_rows[[length(rob_rows)+1]] <- data.frame(spec="current_L2", dv=cur_col, coef=r["Estimate"], se=r["Std. Error"], pvalue=r["Pr(>|t|)"], n_obs=nobs(fit))
    }
  }
}
rob <- bind_rows(rob_rows)
write_csv(rob, paste0(OUT,"fix_reg_robustness.csv"))
cat(sprintf("Robustness: %d rows saved\n", nrow(rob)))

# ====== No-Tier3 sensitivity ======
cat("\n=== No-Tier3 sensitivity ===\n")
# Load match data and identify Tier3-only gvkeys
em <- read_csv("/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/unionization_extension/employer_gvkey_matches.csv")
t1t2_gvk <- unique(em$gvkey[em$match_tier %in% c(1,2)])
t3_gvk <- unique(em$gvkey[em$match_tier == 3])
t3_only <- setdiff(t3_gvk, t1t2_gvk)
cat(sprintf("Tier3-only gvkeys: %d\n", length(t3_only)))

df_noT3 <- df
# Set UNIONIZATION to 0 for Tier3-only gvkeys
df_noT3$UNIONIZATION_cap1[df_noT3$gvkey %in% as.character(t3_only)] <- 0
df_noT3$UNIONIZATION_binary[df_noT3$gvkey %in% as.character(t3_only)] <- 0

noT3_rows <- list()
for (y in DV_LEAD) {
  cat(sprintf("  %s (noT3)...\n", y))
  noT3_rows[[length(noT3_rows)+1]] <- run_spec(y, paste0(y," ~ UNIONIZATION_cap1 | gvkey + review_year"), "L2_noT3")
  noT3_rows[[length(noT3_rows)+1]] <- run_spec(y, paste0(y," ~ UNIONIZATION_cap1 | review_year"), "L4_noT3")
}
noT3 <- bind_rows(noT3_rows)
write_csv(noT3, paste0(OUT,"fix_reg_ladder_noT3.csv"))
cat(sprintf("No-Tier3: %d rows saved\n", nrow(noT3)))

# ====== V2 version (L1+L2 only) ======
cat("\n=== V2 version ===\n")
v2_panel <- nanoparquet::read_parquet(paste0(OUT,"unionization_panel_v2_fix.parquet"))
v2_panel$gvkey <- as.numeric(v2_panel$gvkey)
v2_panel$Year <- as.integer(v2_panel$Year)
gd <- df |> select(gvkey, review_year, all_of(DV_ALL), all_of(DV_CUR),
                    at, dltt, dlc, capx, ebitda, sale, tlcf, ib, xsga, sic) |>
  distinct(gvkey, review_year, .keep_all=TRUE)
gd$gvkey <- as.numeric(gd$gvkey)
v2m <- gd |> left_join(v2_panel, by=c("gvkey"="gvkey", "review_year"="Year"))
v2m$UNIONIZATION_cap1 <- ifelse(is.na(v2m$UNIONIZATION), 0, v2m$UNIONIZATION)
v2m$gvkey <- as.character(v2m$gvkey)
v2m <- v2m |> arrange(gvkey, review_year) |> group_by(gvkey) |>
  mutate(across(all_of(c(DV_ALL, DV_CUR)), ~ dplyr::lead(.x, 1), .names = "{.col}_lead1")) |> ungroup()

v2_rows <- list()
for (y in DV_LEAD) {
  v2_rows[[length(v2_rows)+1]] <- run_spec(y, paste0(y," ~ UNIONIZATION_cap1 | gvkey + review_year"), "L2_v2")
  v2_rows[[length(v2_rows)+1]] <- run_spec(y, paste0(y," ~ UNIONIZATION_cap1 | review_year"), "L4_v2")
}
v2r <- bind_rows(v2_rows)
write_csv(v2r, paste0(OUT,"fix_reg_ladder_v2.csv"))
cat(sprintf("V2: %d rows saved\n", nrow(v2r)))

# ====== Diversity report ======
cat("\n=== Diversity overlap ===\n")
div_ok <- !is.na(df$diversity_lead1) & !is.na(df$UNIONIZATION_cap1)
n_div <- sum(div_ok)
n_div_union <- sum(div_ok & df$UNIONIZATION_cap1 > 0)
cat(sprintf("diversity non-null: %d fy, %d unionized\n", n_div, n_div_union))
if (n_div_union < 100) cat("*** INSUFFICIENT overlap for diversity ***\n")

cat("\nDone.\n")
