#!/usr/bin/env Rscript
# Reproducibility check: v7c on current + total>=10
# WLB ≈ +0.082 (p≈0.023), Comp ≈ +0.005 (p≈0.870)
suppressMessages({library(fixest); library(dplyr); library(tidyr); library(nanoparquet); library(readr)})

OUT <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260624/current_former_bargaining_unit/"

df <- read_parquet(paste0(OUT, "enriched_sample.parquet"))
cat("Total rows:", nrow(df), "\n")

# ─── Helper: prep (from cur_helpers.R) ─────────────────────────────────────
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

# ─── Current only ──────────────────────────────────────────────────────────
cur <- df[df$is_current_employee == 1, ]
cur <- prep(cur)
cat("Current rows:", nrow(cur), "\n")

# total>=10 filter
election_counts <- cur |> group_by(election_id) |> summarise(total_reviews = n(), .groups = 'drop')
keep10 <- election_counts$election_id[election_counts$total_reviews >= 10]
cur10 <- cur[cur$election_id %in% keep10, ]
cat(sprintf("Current total>=10: %d rows, %d elections\n", nrow(cur10), length(keep10)))

# ─── Run WLB and Comp ──────────────────────────────────────────────────────
for (dv in c("wlb", "comp_benefit")) {
  cat(sprintf("\n=== %s ===\n", dv))
  m <- feols(v7c(dv), data = cur10, cluster = ~gvkey + review_year)
  print(summary(m))
}

cat("\n=== Expected: WLB ≈ +0.082 (p≈0.023), Comp ≈ +0.005 (p≈0.870) ===\n")
