#!/usr/bin/env Rscript
# v7 DiD-RD lean: v7c only, global + |m|<=0.20, all poly variants
# Uses fixest with two-way SE, falls back to gvkey-only
library(fixest); library(nanoparquet); library(dplyr); library(tidyr); library(readr)

setwd("/data/disk4/workspace/projects/union_glassdoor")
OUTDIR <- "outputs/rdd_rebuild/focused_rdd_search_v7"

cat("Loading...\n")
df <- nanoparquet::read_parquet(file.path(OUTDIR, "rdd_sample_v7_enriched.parquet"))
cat(sprintf("%d reviews\n", nrow(df)))

df <- df |> mutate(
  gvkey=as.character(gvkey), review_year=as.integer(review_year),
  win=as.integer(win), post=as.integer(post), margin=as.numeric(margin),
  margin2=margin^2, win_post=win*post,
  emp_status=factor(case_when(
    is.na(reviewer_employment_status) ~ "unknown",
    reviewer_employment_status=="REGULAR" ~ "regular",
    reviewer_employment_status=="PART_TIME" ~ "part_time",
    reviewer_employment_status=="INTERN" ~ "intern",
    reviewer_employment_status=="CONTRACT" ~ "contract", TRUE ~ "other"
  ), levels=c("regular","part_time","intern","contract","other","unknown")),
  seniority_f=factor(ifelse(is.na(seniority),0L,as.integer(seniority))),
  state_val=state_x,
  state_clean=ifelse(!is.na(is_us_review)&is_us_review==1, state_val, "Non_US"),
  state_clean=ifelse(is.na(state_clean),"Non_US",state_clean)
)

top50 <- df |> filter(!is.na(role_k1500)) |> count(role_k1500,sort=TRUE) |> slice_head(n=50) |> pull(role_k1500)
df <- df |> mutate(role_clean=case_when(
  is.na(role_k1500)~"Missing_role", role_k1500%in%top50~role_k1500, TRUE~"Other_role"))

cat(sprintf("Top roles: %d\n", length(top50)))

OUTCOMES <- c("overall_rating","career_opp","comp_benefit","senior_mgmt","wlb","culture")
FILTERS <- list(
  list(type="pre_post",N=1L),list(type="pre_post",N=5L),list(type="pre_post",N=10L),
  list(type="pre_post",N=20L),list(type="pre_post",N=25L),list(type="pre_post",N=50L),
  list(type="total",N=50L),list(type="total",N=100L))

POLY <- list(
  p1_ns=list(rhs="win+post+win_post+post:margin", order=1L, spline=FALSE),
  p1_s =list(rhs="win+post+win_post+post:margin+win_post:margin", order=1L, spline=TRUE),
  p2_ns=list(rhs="win+post+win_post+post:margin+post:margin2", order=2L, spline=FALSE),
  p2_s =list(rhs="win+post+win_post+post:margin+win_post:margin+post:margin2+win_post:margin2", order=2L, spline=TRUE))

output_file <- file.path(OUTDIR, "filter_stability_v7_r_results.csv")
# Fresh start
if(file.exists(output_file)) file.remove(output_file)

total <- 0L
setFixest_notes(FALSE)

for (outcome in OUTCOMES) {
  cat(sprintf("\n=== %s ===\n", outcome))
  for (emp in c("all","current")) {
    df_s <- if(emp=="current") filter(df, is_current_employee==1) else df
    for (bw_name in c("global","m20")) {
      df_bw <- if(bw_name=="m20") filter(df_s, abs(margin)<=0.20) else df_s
      for (fc in FILTERS) {
        if (fc$type=="pre_post") {
          vv <- df_bw |> group_by(election_id) |> summarise(np=sum(post==0), npo=sum(post==1),.groups="drop")
          valid <- vv |> filter(np>=fc$N, npo>=fc$N) |> pull(election_id)
        } else {
          vv <- df_bw |> group_by(election_id) |> summarise(nt=n(),.groups="drop")
          valid <- vv |> filter(nt>=fc$N) |> pull(election_id)
        }
        df_f <- df_bw |> filter(election_id %in% valid, !is.na(.data[[outcome]]))
        nr <- nrow(df_f); ne <- n_distinct(df_f$election_id); ng <- n_distinct(df_f$gvkey)
        if (ne < 30 || nr < 300) next

        for (pn in names(POLY)) {
          pv <- POLY[[pn]]
          fml <- as.formula(sprintf("%s ~ %s + emp_status + seniority_f | gvkey + review_year + state_clean + role_clean", outcome, pv$rhs))

          # Two-way SE; if fails, gvkey-only
          res <- tryCatch({
            m <- feols(fml, data=df_f, cluster=~gvkey+review_year, warn=FALSE, notes=FALSE)
            list(est=coef(m)["win_post"], se=se(m)["win_post"], pv=pvalue(m)["win_post"], st="twoway")
          }, error=function(e1){
            tryCatch({
              m <- feols(fml, data=df_f, cluster=~gvkey, warn=FALSE, notes=FALSE)
              list(est=coef(m)["win_post"], se=se(m)["win_post"], pv=pvalue(m)["win_post"], st="gvkey_only")
            }, error=function(e2){
              list(est=NA_real_, se=NA_real_, pv=NA_real_, st="error")
            })
          })

          write_csv(tibble(outcome=outcome, window_days=365L, employee_sample=emp,
            spec_version="v7c", bandwidth_label=bw_name, poly_variant=pn,
            polynomial_order=pv$order, spline=pv$spline, filter_type=fc$type, filter_N=fc$N,
            estimate=res$est, standard_error=res$se, p_value=res$pv, se_type=res$st,
            n_reviews=nr, n_events=ne, n_gvkeys=ng), output_file, append=TRUE)
          total <- total+1L
        }
        cat(sprintf("  [%s|%s|%s|%s>=%-3d] E=%d\n", outcome, emp, bw_name, fc$type, fc$N, ne))
      }
    }
  }
}
cat(sprintf("\nDONE: %d rows\n", total))
