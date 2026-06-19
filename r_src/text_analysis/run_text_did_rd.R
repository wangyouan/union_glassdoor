#!/usr/bin/env Rscript
# DiD-RD on text outcomes — BERT WLB + LLM annotation dimensions
library(fixest); library(nanoparquet); library(dplyr); library(readr)
setFixest_notes(FALSE); setwd("/data/disk4/workspace/projects/union_glassdoor")

OUTDIR <- "outputs/20260618/text_analysis"

cat("=== Loading data ===\n")
df <- nanoparquet::read_parquet(file.path(OUTDIR, "full_sample_bert_predictions.parquet")) %>% as.data.frame()

# Prep covariates
df <- df %>% mutate(
  gvkey=as.character(gvkey), review_year=as.integer(review_year),
  win=as.integer(win), post=as.integer(post), margin=as.numeric(margin), win_post=win*post,
  emp_status=factor(case_when(
    is.na(reviewer_employment_status)~"unknown", reviewer_employment_status=="REGULAR"~"regular",
    reviewer_employment_status=="PART_TIME"~"part_time", reviewer_employment_status=="INTERN"~"intern",
    reviewer_employment_status=="CONTRACT"~"contract", TRUE~"other"),
    levels=c("regular","part_time","intern","contract","other","unknown")),
  seniority_f=factor(ifelse(is.na(seniority),0L,as.integer(seniority))),
  state_clean=factor(ifelse(!is.na(is_us_review)&is_us_review==1, state_x, "Non_US")))

# Text outcomes
df$wlb_mention_num <- as.numeric(df$wlb_mention_bert)
# Standardize net sentiment
df$wlb_net_std <- scale(df$wlb_net_text_bert)

# Filter function
apply_filter <- function(data, N, outcome) {
  data %>% filter(!is.na(.data[[outcome]])) %>%
    group_by(election_id) %>%
    filter(sum(post==0)>=N, sum(post==1)>=N) %>% ungroup()
}

v7c_fml <- function(y) as.formula(sprintf("%s ~ win + post + win_post + post:margin + emp_status + seniority_f | gvkey + review_year", y))

cat("\n=== DiD-RD: WLB text outcomes ===\n")
results <- list()
for (outcome_var in c("wlb_mention_num", "wlb_net_std")) {
  for (filter_N in c(1,5,10)) {
    for (bw_name in c("global","m20","m10")) {
      d <- df
      if (bw_name=="m20") d <- filter(d, abs(margin)<=0.20)
      else if (bw_name=="m10") d <- filter(d, abs(margin)<=0.10)
      d <- apply_filter(d, filter_N, outcome_var)
      nr <- nrow(d); ne <- n_distinct(d$election_id)
      if (ne < 30 || nr < 500) next
      fml <- v7c_fml(outcome_var)
      fit <- tryCatch(feols(fml, data=d, cluster=~gvkey+review_year, warn=FALSE, notes=FALSE),
                      error=function(e) feols(fml, data=d, cluster=~gvkey, warn=FALSE, notes=FALSE))
      ct <- coeftable(fit); r <- ct["win_post",,drop=FALSE]
      results[[length(results)+1]] <- tibble(
        outcome=outcome_var, filter=paste0("n>=",filter_N), bandwidth=bw_name,
        estimate=r[1,"Estimate"], se=r[1,"Std. Error"], pvalue=r[1,"Pr(>|t|)"],
        n_events=ne, n_reviews=nr)
    }
  }
}
df_r <- bind_rows(results)

cat("\n=== WLB Mention DiD-RD (n>=5, global) ===\n")
for (bw in c("global","m20","m10")) {
  r <- df_r %>% filter(outcome=="wlb_mention_num", filter=="n>=5", bandwidth==bw)
  if (nrow(r)>0) {
    sig <- if(r$pvalue<0.01) "***" else if(r$pvalue<0.05) "**" else if(r$pvalue<0.10) "*" else ""
    cat(sprintf("  %-10s: tau=%+.5f se=%.5f p=%.4f%s E=%d\n", bw, r$estimate, r$se, r$pvalue, sig, r$n_events))
  }
}

cat("\n=== WLB Net Sentiment DiD-RD (n>=5, global) ===\n")
for (bw in c("global","m20","m10")) {
  r <- df_r %>% filter(outcome=="wlb_net_std", filter=="n>=5", bandwidth==bw)
  if (nrow(r)>0) {
    sig <- if(r$pvalue<0.01) "***" else if(r$pvalue<0.05) "**" else if(r$pvalue<0.10) "*" else ""
    cat(sprintf("  %-10s: tau=%+.5f se=%.5f p=%.4f%s E=%d\n", bw, r$estimate, r$se, r$pvalue, sig, r$n_events))
  }
}

# Compare with rating WLB
cat("\n=== Rating WLB DiD-RD (n>=5, global) for comparison ===\n")
for (bw in c("global","m20","m10")) {
  d <- df; if (bw=="m20") d <- filter(d, abs(margin)<=0.20) else if (bw=="m10") d <- filter(d, abs(margin)<=0.10)
  d <- apply_filter(d, 5, "wlb")
  d$wlb_std <- scale(d$wlb)
  fml <- v7c_fml("wlb_std")
  fit <- tryCatch(feols(fml, data=d, cluster=~gvkey+review_year, warn=FALSE, notes=FALSE),
                  error=function(e) feols(fml, data=d, cluster=~gvkey, warn=FALSE, notes=FALSE))
  ct <- coeftable(fit); r <- ct["win_post",,drop=FALSE]
  sig <- if(r[1,"Pr(>|t|)"]<0.01) "***" else if(r[1,"Pr(>|t|)"]<0.05) "**" else if(r[1,"Pr(>|t|)"]<0.10) "*" else ""
  cat(sprintf("  %-10s: tau=%+.4f se=%.4f p=%.4f%s E=%d\n", bw, r[1,"Estimate"], r[1,"Std. Error"], r[1,"Pr(>|t|)"], sig, n_distinct(d$election_id)))
}

# Save
write_csv(df_r, file.path(OUTDIR, "text_did_rd_results.csv"))
cat("\nSaved text_did_rd_results.csv\nDone.\n")
