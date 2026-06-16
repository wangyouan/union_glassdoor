#!/usr/bin/env Rscript
# Check 3: Pre-election rating balance at cutoff
library(fixest); library(rdrobust); library(nanoparquet); library(dplyr); library(readr)
setFixest_notes(FALSE); setwd("/data/disk4/workspace/projects/union_glassdoor")

df <- nanoparquet::read_parquet("outputs/rdd_rebuild/focused_rdd_search_v7/rdd_sample_v7_enriched.parquet")
df <- df |> mutate(
  gvkey=as.character(gvkey), review_year=as.integer(review_year),
  win=as.integer(win), margin=as.numeric(margin), win_margin=as.integer(win)*as.numeric(margin),
  emp_status=factor(case_when(
    is.na(reviewer_employment_status)~"unknown", reviewer_employment_status=="REGULAR"~"regular",
    reviewer_employment_status=="PART_TIME"~"part_time", reviewer_employment_status=="INTERN"~"intern",
    reviewer_employment_status=="CONTRACT"~"contract", TRUE~"other"),
    levels=c("regular","part_time","intern","contract","other","unknown")),
  seniority_f=factor(ifelse(is.na(seniority),0L,as.integer(seniority))),
  state_clean=factor(ifelse(!is.na(is_us_review)&is_us_review==1, state_x, "Non_US")))
top50 <- df |> filter(!is.na(role_k1500)) |> count(role_k1500,sort=TRUE) |> slice_head(n=50) |> pull(role_k1500)
df <- df |> mutate(role_clean=factor(case_when(is.na(role_k1500)~"Missing_role",role_k1500%in%top50~role_k1500,TRUE~"Other_role")))

OUTCOMES <- c("overall_rating","career_opp","comp_benefit","senior_mgmt","wlb","culture")
FE_RHS <- "emp_status + seniority_f + state_clean + role_clean"
ABSORB <- "gvkey + review_year"

cat("=== Check 3: Pre-Rating Balance ===\n")
results <- list()

for (outcome in OUTCOMES) {
  # Approach A: rdrobust on election-level pre means
  ev_pre <- df |> filter(post==0, !is.na(.data[[outcome]])) |>
    group_by(election_id) |> summarise(
      mean_pre=mean(.data[[outcome]], na.rm=TRUE),
      margin=first(margin), n_pre=n(), .groups="drop") |> filter(n_pre >= 3)

  if (nrow(ev_pre) >= 50) {
    rdr_pre <- tryCatch(rdrobust(y=ev_pre$mean_pre, x=ev_pre$margin, c=0, kernel="triangular"),
                         error=function(e) NULL)
    if (!is.null(rdr_pre)) {
      results[[length(results)+1]] <- tibble(outcome=outcome, method="rdrobust_pre",
        estimate=rdr_pre$coef[3], se=rdr_pre$se[3], p_value=rdr_pre$pv[3],
        n_events=nrow(ev_pre), h=rdr_pre$bws[1,1])
      cat(sprintf("  %-20s rdrobust: tau=%+.4f se=%.4f p=%.4f h=%.3f N=%d\n",
        outcome, rdr_pre$coef[3], rdr_pre$se[3], rdr_pre$pv[3], rdr_pre$bws[1,1], nrow(ev_pre)))
    }
  }

  # Approach B: OLS on pre-only reviews (win coefficient = pre-period discontinuity)
  pre_rev <- df |> filter(post==0, !is.na(.data[[outcome]]))
  fml_pre <- as.formula(sprintf("%s ~ win + margin + win:margin + %s | %s", outcome, FE_RHS, ABSORB))
  m_pre <- tryCatch(
    feols(fml_pre, data=pre_rev, cluster=~gvkey+review_year, warn=FALSE, notes=FALSE),
    error=function(e) feols(fml_pre, data=pre_rev, cluster=~gvkey, warn=FALSE, notes=FALSE))
  coef_win <- coef(m_pre)["win"]
  se_win <- se(m_pre)["win"]
  pv_win <- pvalue(m_pre)["win"]
  results[[length(results)+1]] <- tibble(outcome=outcome, method="ols_pre_win",
    estimate=coef_win, se=se_win, p_value=pv_win,
    n_events=n_distinct(pre_rev$election_id), h=NA_real_)
  cat(sprintf("  %-20s OLS win:    tau=%+.4f se=%.4f p=%.4f N=%d\n",
    outcome, coef_win, se_win, pv_win, n_distinct(pre_rev$election_id)))
}

df_out <- bind_rows(results)
write_csv(df_out, "outputs/rdd_rebuild/focused_rdd_search_v9/pre_rating_balance_results.csv")
cat(sprintf("\nSaved %d rows\n", nrow(df_out)))

# Flag any p<0.10
flags <- df_out |> filter(p_value < 0.10)
if (nrow(flags) > 0) {
  cat("\n*** WARNING: Pre-period discontinuity detected at p<0.10: ***\n")
  print(flags)
} else {
  cat("\n*** PASS: No pre-period discontinuity at p<0.10. Design validated. ***\n")
}
