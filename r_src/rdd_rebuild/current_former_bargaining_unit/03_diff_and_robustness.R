#!/usr/bin/env Rscript
# STEP 3: Current vs Former difference tests (pooled interaction)
# STEP 4: Filter robustness (pre&post>=1/5/10, total>=10/20) + Bandwidth (global/0.20/0.10/0.05)
# Outputs: current_former_difference_tests.csv, filter_bandwidth_robustness.csv

suppressMessages({library(fixest); library(dplyr); library(tidyr); library(nanoparquet); library(readr); library(purrr)})

OUT <- "/data/disk4/workspace/projects/union_glassdoor/outputs/rdd_rebuild/current_former_bargaining_unit/20260624/"

df <- read_parquet(paste0(OUT, "enriched_sample.parquet"))
cat("Loaded:", nrow(df), "rows\n")

# Create sample_type (since it's not in the saved parquet)
df$sample_type <- ifelse(df$is_current_employee == 1, "current",
                  ifelse(df$is_former_employee == 1, "former", "unknown"))

DV10 <- c("overall_rating", "career_opp", "comp_benefit", "senior_mgmt", "wlb", "culture",
          "recommend", "business_outlook", "ceo_approval", "diversity")

# ─── Helper: prep ────────────────────────────────────────────────────────
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

prep_with_role <- function(d) {
  d <- prep(d)
  top50 <- d |> filter(!is.na(role_k1500)) |> count(role_k1500,sort=TRUE) |> slice_head(n=50) |> pull(role_k1500)
  d |> mutate(role_clean=case_when(is.na(role_k1500)~"Missing_role", role_k1500%in%top50~role_k1500, TRUE~"Other_role"))
}

# ─── Pre-compute top50 role globally ─────────────────────────────────────
d_prep <- prep(df)
top50 <- d_prep |> filter(!is.na(role_k1500)) |> count(role_k1500,sort=TRUE) |> slice_head(n=50) |> pull(role_k1500)

prep2 <- function(d) {
  d <- prep(d)
  d |> mutate(role_clean=case_when(is.na(role_k1500)~"Missing_role", role_k1500%in%top50~role_k1500, TRUE~"Other_role"))
}

v7c <- function(y) as.formula(paste0(y," ~ win+post+win_post+post:margin+emp_status+seniority_f | gvkey+review_year+state_clean+role_clean"))

# ─── STEP 3: Pooled interaction (current vs former) ─────────────────────
cat("\n========================================\n")
cat("STEP 3: Current vs Former Difference Tests\n")
cat("========================================\n")

# Pool current+former only
pool <- df[df$sample_type %in% c("current","former"), ]
pool$former <- as.integer(pool$sample_type == "former")

# v7c interaction spec
v7c_int <- function(y) as.formula(paste0(y,
  " ~ win + post + win_post + former + post:former + win_post:former + post:margin + emp_status + seniority_f | gvkey + review_year + state_clean + role_clean"))

diff_results <- list()

for (dv in DV10) {
  cat(sprintf("\n--- %s ---\n", dv))

  # Filter: total>=10 for this DV in the pooled sample
  pool_dv <- pool[!is.na(pool[[dv]]), ]
  eid_counts <- pool_dv |> group_by(election_id) |> summarise(n = n(), .groups = 'drop')
  keep <- eid_counts$election_id[eid_counts$n >= 10]
  pool_dv <- pool_dv[pool_dv$election_id %in% keep, ]
  pool_dv <- prep2(pool_dv)

  n_cur <- sum(pool_dv$former == 0)
  n_for <- sum(pool_dv$former == 1)
  cat(sprintf("  current reviews: %d, former reviews: %d\n", n_cur, n_for))

  if (n_for < 50) {
    diff_results[[length(diff_results)+1]] <- data.frame(
      dv=dv, current_effect=NA, former_effect=NA, diff=NA, diff_se=NA, diff_p=NA,
      n_current=n_cur, n_former=n_for, note="former_too_small")
    next
  }

  fit <- tryCatch(feols(v7c_int(dv), data = pool_dv, cluster = ~gvkey + review_year),
                  error = function(e) NULL)

  if (is.null(fit)) {
    diff_results[[length(diff_results)+1]] <- data.frame(
      dv=dv, current_effect=NA, former_effect=NA, diff=NA, diff_se=NA, diff_p=NA,
      n_current=n_cur, n_former=n_for, note="model_failed")
    next
  }

  ct <- coeftable(fit)
  has_wp <- "win_post" %in% rownames(ct)
  has_wpf <- "win_post:former" %in% rownames(ct)

  current_effect <- if(has_wp) ct["win_post","Estimate"] else NA
  current_se <- if(has_wp) ct["win_post","Std. Error"] else NA

  # former total effect = win_post + win_post:former
  if (has_wp && has_wpf) {
    # Use linearHypothesis-style delta method: test win_post + win_post:former = 0
    # Simple linear combination
    b_wp <- ct["win_post","Estimate"]
    b_wpf <- ct["win_post:former","Estimate"]
    former_effect <- b_wp + b_wpf
    diff_effect <- -b_wpf  # current - former
    diff_se <- ct["win_post:former","Std. Error"]
    diff_p <- ct["win_post:former","Pr(>|t|)"]
  } else {
    former_effect <- NA
    diff_effect <- NA
    diff_se <- NA
    diff_p <- NA
  }

  cat(sprintf("  current: %.4f, former: %.4f, diff: %.4f, diff_p: %.4f\n",
              current_effect, former_effect, diff_effect, diff_p))

  diff_results[[length(diff_results)+1]] <- data.frame(
    dv=dv, current_effect=current_effect, former_effect=former_effect,
    diff=diff_effect, diff_se=diff_se, diff_p=diff_p,
    n_current=n_cur, n_former=n_for, note="")
}

diff_df <- bind_rows(diff_results)
write_csv(diff_df, paste0(OUT, "current_former_difference_tests.csv"))
cat(sprintf("\nSaved current_former_difference_tests.csv (%d rows)\n", nrow(diff_df)))

# ─── STEP 4: Filter & Bandwidth Robustness ──────────────────────────────
cat("\n========================================\n")
cat("STEP 4: Filter & Bandwidth Robustness\n")
cat("========================================\n")

filters <- list(
  list(name="pre&post>=1", type="each", N=1),
  list(name="pre&post>=5", type="each", N=5),
  list(name="pre&post>=10", type="each", N=10),
  list(name="total>=10", type="total", N=10),
  list(name="total>=20", type="total", N=20)
)

bandwidths <- c(1.0, 0.20, 0.10, 0.05)  # 1.0 = global

elig <- function(d, type, N) {
  if (type == "each") {
    d |> group_by(election_id) |> summarise(a=sum(post==0), b=sum(post==1), .groups="drop") |>
      filter(a>=N, b>=N) |> pull(election_id)
  } else {
    d |> group_by(election_id) |> summarise(n=n(), .groups="drop") |> filter(n>=N) |> pull(election_id)
  }
}

robust_results <- list()

for (s in c("current", "former")) {
  cat(sprintf("\n=== Sample: %s ===\n", s))
  data_s <- df[df$sample_type == s, ]

  for (dv in DV10) {
    cat(sprintf("  %s:\n", dv))

    for (flt in filters) {
      data_dv <- data_s[!is.na(data_s[[dv]]), ]
      eids <- elig(data_dv, flt$type, flt$N)
      if (length(eids) < 5) next
      sub <- data_dv[data_dv$election_id %in% eids, ]

      for (bw in bandwidths) {
        if (bw < 1.0) {
          sub_bw <- sub[abs(sub$margin) <= bw, ]
        } else {
          sub_bw <- sub
        }

        if (nrow(sub_bw) < 50) next
        # Check enough win=0 and win=1
        if (sum(sub_bw$win==0) < 3 || sum(sub_bw$win==1) < 3) next

        sub_bw <- prep2(sub_bw)

        fit <- tryCatch(feols(v7c(dv), data = sub_bw, cluster = ~gvkey + review_year),
                        error = function(e) NULL)
        if (is.null(fit)) next

        ct <- coeftable(fit)
        if (!("win_post" %in% rownames(ct))) next

        robust_results[[length(robust_results)+1]] <- data.frame(
          sample = s, dv = dv, filter = flt$name, bandwidth = bw,
          coef = ct["win_post","Estimate"],
          se = ct["win_post","Std. Error"],
          p = ct["win_post","Pr(>|t|)"],
          n_reviews = nrow(sub_bw),
          n_elections = length(unique(sub_bw$election_id)),
          n_firms = length(unique(sub_bw$gvkey))
        )
      }
    }
  }
}

robust_df <- bind_rows(robust_results)
write_csv(robust_df, paste0(OUT, "filter_bandwidth_robustness.csv"))
cat(sprintf("Saved filter_bandwidth_robustness.csv (%d rows)\n", nrow(robust_df)))

# Quick summary
cat("\n=== WLB robustness (current) ===\n")
wlb_cur <- robust_df[robust_df$sample == "current" & robust_df$dv == "wlb", ]
print(wlb_cur[order(wlb_cur$filter, wlb_cur$bandwidth), c("filter","bandwidth","coef","p","n_elections")])

cat("\nDone.\n")
