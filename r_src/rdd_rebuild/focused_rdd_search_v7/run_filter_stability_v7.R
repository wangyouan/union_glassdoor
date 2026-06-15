#!/usr/bin/env Rscript
# v7 DiD-RD filter stability with fixest: firm FE + two-way clustering
library(fixest); library(nanoparquet); library(dplyr); library(tidyr); library(purrr); library(readr); library(stringr); library(data.table)

setwd("/data/disk4/workspace/projects/union_glassdoor")
OUTDIR <- "outputs/rdd_rebuild/focused_rdd_search_v7"
dir.create(OUTDIR, showWarnings=FALSE, recursive=TRUE)

cat("Loading data...\n")
df <- nanoparquet::read_parquet(file.path(OUTDIR, "rdd_sample_v7_enriched.parquet"))
cat(sprintf("  %d reviews loaded\n", nrow(df)))

# Prep
df <- df |> mutate(
  gvkey = as.character(gvkey), review_year = as.integer(review_year),
  win = as.integer(win), post = as.integer(post), margin = as.numeric(margin),
  margin2 = margin^2, win_post = win * post,
  # employment status
  emp_status = factor(case_when(
    is.na(reviewer_employment_status) ~ "unknown",
    reviewer_employment_status == "REGULAR"  ~ "regular",
    reviewer_employment_status == "PART_TIME" ~ "part_time",
    reviewer_employment_status == "INTERN"    ~ "intern",
    reviewer_employment_status == "CONTRACT"  ~ "contract",
    TRUE ~ "other"
  ), levels=c("regular","part_time","intern","contract","other","unknown")),
  # seniority
  seniority_f = factor(ifelse(is.na(seniority), 0L, as.integer(seniority))),
  # state (state_x from RDD sample, state_y from full GD — use state_x)
  state_val = state_x,
  state_clean = ifelse(!is.na(is_us_review) & is_us_review==1, state_val, "Non_US"),
  state_clean = ifelse(is.na(state_clean), "Non_US", state_clean)
)

# Top-50 roles
top50 <- df |> filter(!is.na(role_k1500)) |> count(role_k1500, sort=TRUE) |> slice_head(n=50) |> pull(role_k1500)
df <- df |> mutate(
  role_clean = case_when(
    is.na(role_k1500) ~ "Missing_role",
    role_k1500 %in% top50 ~ role_k1500,
    TRUE ~ "Other_role"
  )
)

cat(sprintf("  Prepared. Top role categories: %d\n", length(top50)))

# Constants
OUTCOMES <- c("overall_rating","career_opp","comp_benefit","senior_mgmt","wlb","culture")
EMPLOYEE_SAMPLES <- c("all","current")
WINDOW_DAYS <- 365L
MIN_EVENTS <- 30L; MIN_REVIEWS <- 300L

POLY_VARIANTS <- list(
  poly1_non_spline = list(rhs="win + post + win_post + post:margin", order=1L, spline=FALSE),
  poly1_spline     = list(rhs="win + post + win_post + post:margin + win_post:margin", order=1L, spline=TRUE),
  poly2_non_spline = list(rhs="win + post + win_post + post:margin + post:margin2", order=2L, spline=FALSE),
  poly2_spline     = list(rhs="win + post + win_post + post:margin + win_post:margin + post:margin2 + win_post:margin2", order=2L, spline=TRUE)
)

SPECS <- list(
  v7a = list(fe_rhs="emp_status + seniority_f", absorb="gvkey + review_year"),
  v7b = list(fe_rhs="emp_status + seniority_f", absorb="gvkey + review_year + state_clean"),
  v7c = list(fe_rhs="emp_status + seniority_f", absorb="gvkey + review_year + state_clean + role_clean")
)

FILTER_COMBOS <- list(
  list(filter_type="pre_post", filter_N=1L),  list(filter_type="pre_post", filter_N=5L),
  list(filter_type="pre_post", filter_N=10L), list(filter_type="pre_post", filter_N=20L),
  list(filter_type="pre_post", filter_N=25L), list(filter_type="pre_post", filter_N=50L),
  list(filter_type="total", filter_N=50L),   list(filter_type="total", filter_N=100L)
)

BANDWIDTHS <- list(
  global = list(label="global", fn=function(d) d),
  m20    = list(label="|m|<=0.20", fn=function(d) filter(d, abs(margin) <= 0.20))
)

apply_filter <- function(data, filter_type, filter_N) {
  if (filter_type == "pre_post") {
    vv <- data |> group_by(election_id) |> summarise(n_pre=sum(post==0), n_post=sum(post==1), .groups="drop")
    valid <- vv |> filter(n_pre >= filter_N, n_post >= filter_N) |> pull(election_id)
  } else {
    vv <- data |> group_by(election_id) |> summarise(n_total=n(), .groups="drop")
    valid <- vv |> filter(n_total >= filter_N) |> pull(election_id)
  }
  data |> filter(election_id %in% valid)
}

write_empty <- function() {
  tibble(outcome=character(), window_days=integer(), employee_sample=character(),
    spec_version=character(), bandwidth_label=character(), poly_variant=character(),
    polynomial_order=integer(), spline=logical(), filter_type=character(),
    filter_N=integer(), estimate=numeric(), standard_error=numeric(),
    p_value=numeric(), se_type=character(), n_reviews=integer(),
    n_events=integer(), n_gvkeys=integer())
}

output_file <- file.path(OUTDIR, "filter_stability_v7_r_results.csv")
write_csv(write_empty(), output_file); total_rows <- 0L

for (outcome in OUTCOMES) {
  cat(sprintf("\n=== %s ===\n", outcome))
  for (emp_sample in EMPLOYEE_SAMPLES) {
    df_samp <- if (emp_sample=="current") filter(df, is_current_employee==1) else df
    for (bw_name in names(BANDWIDTHS)) {
      bw_def <- BANDWIDTHS[[bw_name]]; df_bw <- bw_def$fn(df_samp)
      for (fc in FILTER_COMBOS) {
        df_filt <- apply_filter(df_bw, fc$filter_type, fc$filter_N)
        # Must filter NA outcome BEFORE counting
        df_filt <- df_filt |> filter(!is.na(.data[[outcome]]))
        n_reviews <- nrow(df_filt); n_events <- n_distinct(df_filt$election_id); n_gvkeys <- n_distinct(df_filt$gvkey)

        if (n_events < MIN_EVENTS || n_reviews < MIN_REVIEWS) {
          for (spec_name in names(SPECS)) {
            pv <- POLY_VARIANTS[[1]]; sv <- SPECS[[spec_name]]
            row <- tibble(outcome=outcome, window_days=WINDOW_DAYS, employee_sample=emp_sample,
              spec_version=spec_name, bandwidth_label=bw_def$label, poly_variant="insufficient", polynomial_order=NA_integer_,
              spline=NA, filter_type=fc$filter_type, filter_N=fc$filter_N,
              estimate=NA_real_, standard_error=NA_real_, p_value=NA_real_,
              se_type="insufficient_data", n_reviews=n_reviews, n_events=n_events, n_gvkeys=n_gvkeys)
            write_csv(row, output_file, append=TRUE); total_rows <- total_rows + 1L
          }
          next
        }

        for (spec_name in names(SPECS)) {
          spec_def <- SPECS[[spec_name]]
          for (poly_name in names(POLY_VARIANTS)) {
            pv <- POLY_VARIANTS[[poly_name]]
            fml_str <- sprintf("%s ~ %s + %s | %s", outcome, pv$rhs, spec_def$fe_rhs, spec_def$absorb)
            fml <- as.formula(fml_str)

            # Try two-way, fall back to gvkey-only
            res <- tryCatch({
              m <- feols(fml, data=df_filt, cluster=~gvkey+review_year, warn=FALSE, notes=FALSE)
              list(est=coef(m)["win_post"], se=se(m)["win_post"], pv=pvalue(m)["win_post"], st="twoway")
            }, error=function(e1) {
              tryCatch({
                m <- feols(fml, data=df_filt, cluster=~gvkey, warn=FALSE, notes=FALSE)
                list(est=coef(m)["win_post"], se=se(m)["win_post"], pv=pvalue(m)["win_post"], st="gvkey_only")
              }, error=function(e2) {
                list(est=NA_real_, se=NA_real_, pv=NA_real_, st="error")
              })
            })

            row <- tibble(outcome=outcome, window_days=WINDOW_DAYS, employee_sample=emp_sample,
              spec_version=spec_name, bandwidth_label=bw_def$label, poly_variant=poly_name,
              polynomial_order=pv$order, spline=pv$spline, filter_type=fc$filter_type, filter_N=fc$filter_N,
              estimate=res$est, standard_error=res$se, p_value=res$pv, se_type=res$st,
              n_reviews=n_reviews, n_events=n_events, n_gvkeys=n_gvkeys)
            write_csv(row, output_file, append=TRUE); total_rows <- total_rows + 1L
          }
        }
        cat(sprintf("  [%s|%s|%s|%s>=%-3d] E=%d R=%d\n", outcome, emp_sample, bw_def$label, fc$filter_type, fc$filter_N, n_events, n_reviews))
      }
    }
  }
}

cat(sprintf("\nDONE. Total rows: %d\n", total_rows))
