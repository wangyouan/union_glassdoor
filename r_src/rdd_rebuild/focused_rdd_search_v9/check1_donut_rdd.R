#!/usr/bin/env Rscript
# Check 1: Donut RD — exclude |margin| < 0.01
library(fixest); library(nanoparquet); library(dplyr); library(readr)
setFixest_notes(FALSE); setwd("/data/disk4/workspace/projects/union_glassdoor")

df <- nanoparquet::read_parquet("outputs/rdd_rebuild/focused_rdd_search_v7/rdd_sample_v7_enriched.parquet")
df <- df |> mutate(
  gvkey=as.character(gvkey), review_year=as.integer(review_year),
  win=as.integer(win), post=as.integer(post), margin=as.numeric(margin),
  margin2=margin^2, win_post=win*post,
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
FILTERS <- list(list(type="pre_post",N=1L),list(type="pre_post",N=10L),list(type="pre_post",N=25L))
BANDWIDTHS <- list(
  global_donut=list(label="global_donut", fn=function(d) filter(d, abs(margin)>=0.01)),
  m20_donut=list(label="m20_donut", fn=function(d) filter(d, abs(margin)<=0.20, abs(margin)>=0.01)),
  m05_donut=list(label="m05_donut", fn=function(d) filter(d, abs(margin)<=0.05, abs(margin)>=0.01)))

# Load v7 results for comparison
v7 <- read.csv("outputs/rdd_rebuild/focused_rdd_search_v7/filter_stability_v7_r_results.csv")

OUTFILE <- "outputs/rdd_rebuild/focused_rdd_search_v9/donut_rdd_results.csv"
write_csv(tibble(outcome=character(),bandwidth_label=character(),filter_type=character(),
  filter_N=integer(),n_events=integer(),n_reviews=integer(),
  estimate=numeric(),standard_error=numeric(),p_value=numeric(),se_type=character()), OUTFILE)

cat("=== Check 1: Donut RD ===\n")
total <- 0L
for (outcome in OUTCOMES) {
  for (bw_name in names(BANDWIDTHS)) {
    bw_def <- BANDWIDTHS[[bw_name]]; df_bw <- bw_def$fn(df)
    for (fc in FILTERS) {
      if (fc$type=="pre_post") {
        vv <- df_bw |> group_by(election_id) |> summarise(np=sum(post==0),npo=sum(post==1),.groups="drop")
        valid <- vv |> filter(np>=fc$N,npo>=fc$N) |> pull(election_id)
      } else { next }
      df_f <- df_bw |> filter(election_id %in% valid, !is.na(.data[[outcome]]))
      nr <- nrow(df_f); ne <- n_distinct(df_f$election_id)
      if (ne < 20 || nr < 200) next
      fml <- as.formula(sprintf("%s ~ win + post + win_post + post:margin + win_post:margin + %s | %s", outcome, FE_RHS, ABSORB))
      res <- tryCatch({
        m <- feols(fml, data=df_f, cluster=~gvkey+review_year, warn=FALSE, notes=FALSE)
        list(est=coef(m)["win_post"], se=se(m)["win_post"], pv=pvalue(m)["win_post"], st="twoway")
      }, error=function(e){
        tryCatch({
          m <- feols(fml, data=df_f, cluster=~gvkey, warn=FALSE, notes=FALSE)
          list(est=coef(m)["win_post"], se=se(m)["win_post"], pv=pvalue(m)["win_post"], st="gvkey_only")
        }, error=function(e2){ list(est=NA_real_, se=NA_real_, pv=NA_real_, st="error") })
      })
      write_csv(tibble(outcome=outcome,bandwidth_label=bw_def$label,filter_type=fc$type,
        filter_N=fc$N,n_events=ne,n_reviews=nr,estimate=res$est,standard_error=res$se,
        p_value=res$pv,se_type=res$st), OUTFILE, append=TRUE)
      total <- total+1L
      cat(sprintf("  [%s|%s|pre>=%d] E=%d\n", outcome, bw_def$label, fc$N, ne))
    }
  }
}

cat(sprintf("DONE: %d rows\n", total))

# Comparison with v7
cat("\n=== Donut vs Original for WLB (all, pre>=1) ===\n")
for (bw_donut in c("global_donut","m20_donut","m05_donut")) {
  bw_orig <- sub("_donut","",bw_donut)
  if (bw_orig=="global") bw_orig_label <- "global" else if (bw_orig=="m20") bw_orig_label <- "m20" else bw_orig_label <- "m05"
  donut_row <- read.csv(OUTFILE) |> filter(outcome=="wlb", bandwidth_label==bw_donut, filter_type=="pre_post", filter_N==1)
  if (nrow(donut_row)>0) {
    cat(sprintf("  %s: donut tau=%+.4f p=%.4f E=%d\n", bw_donut, donut_row$estimate[1], donut_row$p_value[1], donut_row$n_events[1]))
  }
}
