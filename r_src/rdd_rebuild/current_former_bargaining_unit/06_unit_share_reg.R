#!/usr/bin/env Rscript
# STEP 10-11: Unit-share construction + Win×Post×UnitShare interaction
# Outputs: unit_share_election_data.csv, unit_share_regression_results.csv

suppressMessages({library(fixest); library(dplyr); library(tidyr); library(nanoparquet); library(readr)})

OUT <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260624/"
PYTHON_BIN <- "/home/user/anaconda3/envs/union_glassdoor/bin/python"

# ─── STEP 10: Build unit_share via Python ────────────────────────────────
# (Doing this in Python because Compustat merge involves complex year logic)

cat("=== STEP 10: Building unit_share ===\n")

py_script <- sprintf('
import pandas as pd
import numpy as np

OUT = "%s"

# Load NLRB unit data
nlrb = pd.read_parquet("/data/disk4/workspace/projects/union/outputs/preliminary_election_level.parquet",
                       columns=["election_id","unit_size","filing__number_of_eligible_voters"])
nlrb = nlrb.drop_duplicates(subset="election_id")

# Load enriched sample for gvkey and election year
enriched = pd.read_parquet(f"{OUT}/enriched_sample.parquet",
                           columns=["election_id","gvkey","election_year_elec"])
elections = enriched[["election_id","gvkey","election_year_elec"]].drop_duplicates(subset="election_id")

# Merge unit_size
elections = elections.merge(nlrb, on="election_id", how="left")

# Load Compustat
cmp = pd.read_parquet("outputs/compustat_firm_controls.parquet")
# Get EMP per gvkey-year
cmp_emp = cmp[["gvkey","fyear","emp"]].drop_duplicates()
cmp_emp["gvkey"] = cmp_emp["gvkey"].astype(str)

# For each election, find EMP at election_year - 1
elections["fyear_target"] = elections["election_year_elec"] - 1
elections = elections.merge(cmp_emp, left_on=["gvkey","fyear_target"],
                            right_on=["gvkey","fyear"], how="left")
elections.drop(columns=["fyear"], inplace=True)

# If election_year - 1 not found, try election_year - 2
mask_missing = elections["emp"].isna()
elections.loc[mask_missing, "fyear_target"] = elections.loc[mask_missing, "election_year_elec"] - 2
elections2_miss = elections[mask_missing][["election_id","gvkey","fyear_target"]].copy()
elections2_miss = elections2_miss.merge(cmp_emp, left_on=["gvkey","fyear_target"],
                                        right_on=["gvkey","fyear"], how="left")
# Merge back
for idx in elections2_miss.index:
    eid = elections2_miss.loc[idx, "election_id"]
    elections.loc[elections["election_id"]==eid, "emp"] = elections2_miss.loc[idx, "emp"]

# Use eligible voters if unit_size missing, flag it
elections["unit_size_source"] = "unit_size"
mask_no_unit = elections["unit_size"].isna()
elections.loc[mask_no_unit, "unit_size"] = elections.loc[mask_no_unit, "filing__number_of_eligible_voters"]
elections.loc[mask_no_unit, "unit_size_source"] = "eligible_voters"

# Unit share
elections["unit_share_raw"] = elections["unit_size"] / elections["emp"]
elections["unit_share_capped"] = elections["unit_share_raw"].clip(upper=1.0)
elections["unit_share_winsor"] = elections["unit_share_raw"].clip(
    lower=elections["unit_share_raw"].quantile(0.01),
    upper=elections["unit_share_raw"].quantile(0.99)
)

# Stats
print(f"Elections: {len(elections):,}")
print(f"Has unit_size: {elections[\"unit_size\"].notna().sum():,}")
print(f"Has EMP: {elections[\"emp\"].notna().sum():,}")
print(f"Has EMP from year-1: {elections[(elections[\"fyear_target\"]==elections[\"election_year_elec\"]-1)&(elections[\"emp\"].notna())].shape[0]:,}")
print(f"Has unit_share: {elections[\"unit_share_raw\"].notna().sum():,}")
print(f"unit_share_raw: mean={elections[\"unit_share_raw\"].mean():.4f}, med={elections[\"unit_share_raw\"].median():.4f}")
print(f"  P25={elections[\"unit_share_raw\"].quantile(0.25):.4f}, P75={elections[\"unit_share_raw\"].quantile(0.75):.4f}, P90={elections[\"unit_share_raw\"].quantile(0.9):.4f}")
print(f"  >1: {(elections[\"unit_share_raw\"]>1).sum():,}")
print(f"unit_share_capped: mean={elections[\"unit_share_capped\"].mean():.4f}, med={elections[\"unit_share_capped\"].median():.4f}")
print(f"EMP (in thousands?): mean={elections[\"emp\"].mean():.1f}, med={elections[\"emp\"].median():.1f}")
print(f"unit_size: mean={elections[\"unit_size\"].mean():.1f}, med={elections[\"unit_size\"].median():.1f}")

# Save
elections[["election_id","unit_size","emp","unit_share_raw","unit_share_capped","unit_share_winsor","unit_size_source"]].to_csv(
    f"{OUT}/unit_share_election_data.csv", index=False)
print("Saved unit_share_election_data.csv")
' % OUT)

cat(py_script, file="/tmp/build_unit_share.py")
system(paste(PYTHON_BIN, "/tmp/build_unit_share.py"))

# ─── STEP 11: Unit-share interaction regression ─────────────────────────
cat("\n=== STEP 11: Unit-Share Interaction Regression ===\n")

# Load unit_share
ushare <- read_csv(paste0(OUT, "unit_share_election_data.csv"), show_col_types=FALSE)

df <- read_parquet(paste0(OUT, "enriched_sample.parquet"))
df$sample_type <- ifelse(df$is_current_employee == 1, "current",
                  ifelse(df$is_former_employee == 1, "former", "unknown"))

# Merge unit_share
df <- df |> left_join(ushare |> select(election_id, unit_share_raw, unit_share_capped, unit_share_winsor, unit_size, emp),
                      by="election_id")

# Use winsorized (main), with capped as robustness
df$unit_share <- df$unit_share_winsor
df$unit_share[is.na(df$unit_share)] <- 0  # elections without unit_share set to 0

DV10 <- c("overall_rating", "career_opp", "comp_benefit", "senior_mgmt", "wlb", "culture",
          "recommend", "business_outlook", "ceo_approval", "diversity")

prep <- function(d){
  d |> mutate(
    gvkey=as.character(gvkey), review_year=as.integer(review_year),
    win=as.integer(win), post=as.integer(post), margin=as.numeric(margin), win_post=win*post,
    emp_status=case_when(
      is.na(reviewer_employment_status)~"unknown",
      reviewer_employment_status=="REGULAR"~"regular",
      reviewer_employment_status=="PART_TIME"~"part_time",
      reviewer_employment_status=="INTERN"~"intern",
      reviewer_employment_status=="CONTRACT"~"contract", TRUE~"other") |>
      factor(levels=c("regular","part_time","intern","contract","other","unknown")),
    seniority_f=factor(ifelse(is.na(seniority),0L,as.integer(seniority))),
    state_clean=case_when(!is.na(is_us_review)&is_us_review==1~state_y, TRUE~"Non_US") |> replace_na("Non_US"))
}

prep2 <- function(d) {
  d <- prep(d)
  top50 <- d |> filter(!is.na(role_k1500)) |> count(role_k1500,sort=TRUE) |> slice_head(n=50) |> pull(role_k1500)
  d |> mutate(role_clean=case_when(is.na(role_k1500)~"Missing_role", role_k1500%in%top50~role_k1500, TRUE~"Other_role"))
}

# v7c with unit_share interaction
v7c_ushare <- function(y) as.formula(paste0(y,
  " ~ win + post + win_post + unit_share + post:unit_share + win_post:unit_share + post:margin + emp_status + seniority_f | gvkey + review_year + state_clean + role_clean"))

# Run for current sample, total>=10
cur <- df[df$sample_type == "current", ]

ushare_results <- list()
ushare_marginal <- list()

for (dv in DV10) {
  cat(sprintf("  %s:\n", dv))
  cur_dv <- cur[!is.na(cur[[dv]]), ]
  eids <- cur_dv |> group_by(election_id) |> summarise(n=n(), .groups="drop") |> filter(n>=10) |> pull(election_id)
  sub <- cur_dv[cur_dv$election_id %in% eids, ]

  # Only elections with unit_share > 0
  sub2 <- sub[sub$unit_share > 0, ]
  if (nrow(sub2) < 100) {
    ushare_results[[length(ushare_results)+1]] <- data.frame(dv=dv, note="insufficient")
    next
  }

  sub2 <- prep2(sub2)
  fit <- tryCatch(feols(v7c_ushare(dv), data=sub2, cluster=~gvkey+review_year), error=function(e)NULL)
  if (is.null(fit)) {
    ushare_results[[length(ushare_results)+1]] <- data.frame(dv=dv, note="model_failed")
    next
  }

  ct <- coeftable(fit)
  # Extract coefficients
  get_coef <- function(name) {
    if (name %in% rownames(ct)) ct[name,"Estimate"] else NA
  }
  get_se <- function(name) {
    if (name %in% rownames(ct)) ct[name,"Std. Error"] else NA
  }
  get_p <- function(name) {
    if (name %in% rownames(ct)) ct[name,"Pr(>|t|)"] else NA
  }

  ushare_med <- median(sub2$unit_share, na.rm=TRUE)
  cat(sprintf("    median unit_share=%.4f, n=%d\n", ushare_med, nrow(sub2)))

  # Main coefficients
  wp_est <- get_coef("win_post")
  wp_se <- get_se("win_post")
  wp_p <- get_p("win_post")
  wpu_est <- get_coef("win_post:unit_share")
  wpu_se <- get_se("win_post:unit_share")
  wpu_p <- get_p("win_post:unit_share")

  cat(sprintf("    win_post=%.4f (p=%.4f), win_post:unit_share=%.4f (p=%.4f)\n", wp_est, wp_p, wpu_est, wpu_p))

  # Marginal effects at P25, median, P75
  pcts <- quantile(sub2$unit_share, c(0.25, 0.5, 0.75), na.rm=TRUE)
  for (j in seq_along(pcts)) {
    me <- wp_est + wpu_est * pcts[j]
    # SE approximation: SE(β1 + β3*q)
    # We'd need the full VCOV for exact SE. For now, approximate:
    ushare_marginal[[length(ushare_marginal)+1]] <- data.frame(
      dv=dv, percentile=names(pcts)[j], unit_share=pcts[j],
      marginal_effect=me, n=nrow(sub2))
  }

  ushare_results[[length(ushare_results)+1]] <- data.frame(
    dv=dv,
    win_post_coef=wp_est, win_post_se=wp_se, win_post_p=wp_p,
    win_post_x_ushare_coef=wpu_est, win_post_x_ushare_se=wpu_se, win_post_x_ushare_p=wpu_p,
    n_reviews=nrow(sub2), n_elections=length(unique(sub2$election_id)),
    median_ushare=ushare_med)
}

ushare_df <- bind_rows(ushare_results)
write_csv(ushare_df, paste0(OUT, "unit_share_regression_results.csv"))

ushare_marg_df <- bind_rows(ushare_marginal)
write_csv(ushare_marg_df, paste0(OUT, "unit_share_marginal_effects.csv"))

cat(sprintf("\nSaved unit_share_regression_results.csv (%d rows)\n", nrow(ushare_df)))
cat(sprintf("Saved unit_share_marginal_effects.csv (%d rows)\n", nrow(ushare_marg_df)))

# Quick WLB summary
cat("\n=== WLB Unit Share ===\n")
wlb_row <- ushare_df[ushare_df$dv=="wlb",]
print(wlb_row)

cat("\nDone.\n")
