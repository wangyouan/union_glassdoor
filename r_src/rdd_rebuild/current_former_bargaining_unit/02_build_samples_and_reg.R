#!/usr/bin/env Rscript
# STEP 1-2: Sample definition + v7c regressions for current/former/all × 10 DVs
# Outputs: sample_summary.csv, current_former_all_outcomes.csv

suppressMessages({library(fixest); library(dplyr); library(tidyr); library(nanoparquet); library(readr); library(purrr)})

OUT <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260624/current_former_bargaining_unit/"

# ─── Load ──────────────────────────────────────────────────────────────────
df <- read_parquet(paste0(OUT, "enriched_sample.parquet"))
cat("Loaded:", nrow(df), "rows\n")

# ─── 10 DVs ────────────────────────────────────────────────────────────────
DV10 <- c("overall_rating", "career_opp", "comp_benefit", "senior_mgmt", "wlb", "culture",
          "recommend", "business_outlook", "ceo_approval", "diversity")

# ─── Helper: prep (from cur_helpers.R) ────────────────────────────────────
prep <- function(d){
  d <- d |> mutate(
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
  top50 <- d |> filter(!is.na(role_k1500)) |> count(role_k1500,sort=TRUE) |> slice_head(n=50) |> pull(role_k1500)
  d |> mutate(role_clean=case_when(is.na(role_k1500)~"Missing_role", role_k1500%in%top50~role_k1500, TRUE~"Other_role"))
}

v7c <- function(y) as.formula(paste0(y," ~ win+post+win_post+post:margin+emp_status+seniority_f | gvkey+review_year+state_clean+role_clean"))

# ─── STEP 1: Sample definitions ───────────────────────────────────────────
# current = is_current_employee==1
# former  = is_former_employee==1 (explicitly marked, NOT employment-status missing)
# all     = current + former + unknown

df$sample_type <- ifelse(df$is_current_employee == 1, "current",
                  ifelse(df$is_former_employee == 1, "former", "unknown"))

cat("\n=== Sample sizes ===\n")
for (s in c("current","former","unknown")) {
  subs <- df[df$sample_type == s, ]
  cat(sprintf("%s: %d reviews, %d elections, %d firms\n",
              s, nrow(subs), length(unique(subs$election_id)), length(unique(subs$gvkey))))
}

# ─── STEP 1b: Former definition details ───────────────────────────────────
# Check what drives is_former_employee vs is_employment_missing
cat("\n=== Former/missing breakdown ===\n")
cat(sprintf("is_former_employee=1: %d\n", sum(df$is_former_employee == 1)))
cat(sprintf("is_employment_missing=1: %d\n", sum(df$is_employment_missing == 1)))
cat(sprintf("both=0 (current+other): %d\n", sum(df$is_former_employee == 0 & df$is_employment_missing == 0)))

# ─── STEP 1c: Sample characteristics ──────────────────────────────────────
cat("\n=== Sample characteristics by type ===\n")
for (s in c("current","former")) {
  subs <- df[df$sample_type == s, ]
  cat(sprintf("\n--- %s ---\n", s))
  cat(sprintf("  Reviews: %d\n", nrow(subs)))
  cat(sprintf("  Elections: %d\n", length(unique(subs$election_id))))
  cat(sprintf("  Firms (gvkey): %d\n", length(unique(subs$gvkey))))
  cat(sprintf("  Election years: %d–%d\n", min(subs$election_year_elec, na.rm=T), max(subs$election_year_elec, na.rm=T)))
  cat(sprintf("  Avg reviews per election: %.1f\n", nrow(subs)/length(unique(subs$election_id))))
}

# Save sample summary
sample_summary <- data.frame(
  sample = c("current","former","unknown","all"),
  n_reviews = c(sum(df$sample_type=="current"), sum(df$sample_type=="former"),
                sum(df$sample_type=="unknown"), nrow(df)),
  n_elections = c(length(unique(df$election_id[df$sample_type=="current"])),
                  length(unique(df$election_id[df$sample_type=="former"])),
                  length(unique(df$election_id[df$sample_type=="unknown"])),
                  length(unique(df$election_id))),
  n_firms = c(length(unique(df$gvkey[df$sample_type=="current"])),
              length(unique(df$gvkey[df$sample_type=="former"])),
              length(unique(df$gvkey[df$sample_type=="unknown"])),
              length(unique(df$gvkey)))
)
write_csv(sample_summary, paste0(OUT, "sample_summary.csv"))
cat("\nSaved sample_summary.csv\n")

# ─── Helper: election-level filter based on DV non-NA ─────────────────────
# For each DV, count non-NA reviews per election, then keep elections with total>=10
calc_filter <- function(d, dv_col) {
  d_sub <- d[!is.na(d[[dv_col]]), ]
  election_counts <- d_sub |> group_by(election_id) |> summarise(n = n(), .groups = 'drop')
  election_counts$election_id[election_counts$n >= 10]
}

# ─── Helper: run v7c and extract results ──────────────────────────────────
run_v7c <- function(data, dv) {
  keep_eids <- calc_filter(data, dv)
  if (length(keep_eids) < 2) return(NULL)
  d <- data[data$election_id %in% keep_eids, ]
  d <- prep(d)
  n_reviews <- nrow(d)
  n_elections <- length(unique(d$election_id))
  n_firms <- length(unique(d$gvkey))
  pre_mean <- mean(d[[dv]][d$post == 0], na.rm = TRUE)
  pre_sd <- sd(d[[dv]][d$post == 0], na.rm = TRUE)

  fit <- tryCatch(feols(v7c(dv), data = d, cluster = ~gvkey + review_year),
                  error = function(e) NULL)
  if (is.null(fit)) {
    return(data.frame(dv=dv, coef=NA, se=NA, p=NA, n_reviews=n_reviews, n_elections=n_elections,
                       n_firms=n_firms, pre_mean=pre_mean, pre_sd=pre_sd, pct_pre=NA, std_effect=NA,
                       note="model_failed"))
  }
  r <- coeftable(fit)["win_post", ]
  data.frame(
    dv = dv,
    coef = r["Estimate"],
    se = r["Std. Error"],
    p = r["Pr(>|t|)"],
    n_reviews = n_reviews,
    n_elections = n_elections,
    n_firms = n_firms,
    pre_mean = pre_mean,
    pre_sd = pre_sd,
    pct_pre = r["Estimate"] / pre_mean * 100,
    std_effect = r["Estimate"] / pre_sd,
    note = ""
  )
}

# ─── STEP 2: Run all 10 DVs × 3 samples ─────────────────────────────────
cat("\n=== STEP 2: Running v7c for 10 DVs × 3 samples ===\n")

results <- list()
for (s in c("current","former","all")) {
  cat(sprintf("\n--- Sample: %s ---\n", s))
  if (s == "all") {
    data_s <- df
  } else {
    data_s <- df[df$sample_type == s, ]
  }

  for (dv in DV10) {
    cat(sprintf("  %s... ", dv))
    r <- run_v7c(data_s, dv)
    if (!is.null(r)) {
      r$sample <- s
      results[[length(results)+1]] <- r
      if (!is.na(r$coef)) {
        cat(sprintf("coef=%.4f, p=%.4f\n", r$coef, r$p))
      } else {
        cat(sprintf("FAILED: %s\n", r$note))
      }
    } else {
      cat("no_data\n")
    }
  }
}

results_df <- bind_rows(results)
write_csv(results_df, paste0(OUT, "current_former_all_outcomes.csv"))
cat(sprintf("\nSaved current_former_all_outcomes.csv (%d rows)\n", nrow(results_df)))

# Quick summary
cat("\n=== Quick Summary (current, WLB) ===\n")
r <- results_df[results_df$sample == "current" & results_df$dv == "wlb", ]
print(r)

cat("\nDone.\n")
