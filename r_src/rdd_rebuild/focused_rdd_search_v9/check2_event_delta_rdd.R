#!/usr/bin/env Rscript
# Check 2: Event-level delta RDD via rdrobust
library(rdrobust); library(nanoparquet); library(dplyr); library(readr)
setwd("/data/disk4/workspace/projects/union_glassdoor")

df <- nanoparquet::read_parquet("outputs/rdd_rebuild/focused_rdd_search_v7/rdd_sample_v7_enriched.parquet")
df <- df |> mutate(margin=as.numeric(margin), win=as.integer(win))

OUTCOMES <- c("overall_rating","career_opp","comp_benefit","senior_mgmt","wlb","culture")

cat("=== Check 2: Event-level Delta RDD ===\n")
results <- list()

for (outcome in OUTCOMES) {
  # Build election-level delta
  ev <- df |> filter(!is.na(.data[[outcome]])) |>
    group_by(election_id) |>
    summarise(
      mean_pre=mean(.data[[outcome]][post==0], na.rm=TRUE),
      mean_post=mean(.data[[outcome]][post==1], na.rm=TRUE),
      n_pre=sum(post==0 & !is.na(.data[[outcome]])),
      n_post=sum(post==1 & !is.na(.data[[outcome]])),
      margin=first(margin), win=first(win), gvkey=first(as.character(gvkey)),
      .groups="drop") |>
    filter(n_pre >= 5, n_post >= 5)
  ev$delta <- ev$mean_post - ev$mean_pre

  if (nrow(ev) < 50) { cat(sprintf("  %s: insufficient (%d elections)\n", outcome, nrow(ev))); next }

  # Optimal bandwidth
  rdr <- tryCatch(rdrobust(y=ev$delta, x=ev$margin, c=0, kernel="triangular", bwselect="mserd"),
                   error=function(e) NULL)
  if (!is.null(rdr)) {
    results[[length(results)+1]] <- list(
      outcome=outcome, bandwidth_type="opt",
      h_left=rdr$bws[1,1], h_right=rdr$bws[1,2],
      n_left=rdr$N[1], n_right=rdr$N[2],
      estimate=rdr$coef[3], se=rdr$se[3], p_value=rdr$pv[3],
      ci_lower=rdr$ci[3,1], ci_upper=rdr$ci[3,2])
    cat(sprintf("  %-20s opt: h=[%.3f,%.3f] N=[%d,%d] tau=%+.4f se=%.4f p=%.4f\n",
      outcome, rdr$bws[1,1], rdr$bws[1,2], rdr$N[1], rdr$N[2], rdr$coef[3], rdr$se[3], rdr$pv[3]))
  }

  # Fixed bandwidths
  for (h in c(0.20, 0.10, 0.05)) {
    rdr_f <- tryCatch(rdrobust(y=ev$delta, x=ev$margin, c=0, h=h, kernel="uniform"),
                       error=function(e) NULL)
    if (!is.null(rdr_f)) {
      results[[length(results)+1]] <- list(
        outcome=outcome, bandwidth_type=sprintf("fixed_%.2f",h),
        h_left=h, h_right=h, n_left=rdr_f$N[1], n_right=rdr_f$N[2],
        estimate=rdr_f$coef[3], se=rdr_f$se[3], p_value=rdr_f$pv[3],
        ci_lower=rdr_f$ci[3,1], ci_upper=rdr_f$ci[3,2])
    }
  }
}

df_out <- bind_rows(results)
write_csv(df_out, "outputs/rdd_rebuild/focused_rdd_search_v9/event_delta_rdd_results.csv")
cat(sprintf("Saved %d rows\n", nrow(df_out)))
