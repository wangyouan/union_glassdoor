#!/usr/bin/env Rscript
# Subgroup DiD-RD: union/non-union and management/non-management splits
library(fixest); library(nanoparquet); library(dplyr); library(tidyr); library(purrr); library(readr); library(data.table)
setFixest_notes(FALSE)
setwd("/data/disk4/workspace/projects/union_glassdoor")
OUTDIR <- "outputs/rdd_rebuild/subgroup_splits"; dir.create(OUTDIR, recursive=TRUE, showWarnings=FALSE)

cat("=== 1. DATA SETUP ===\n")
df <- nanoparquet::read_parquet("outputs/rdd_rebuild/focused_rdd_search_v7/rdd_sample_v7_enriched.parquet")
clf <- fread("outputs/union_classified_title_universe_step1d.csv") %>%
  select(title_standardized, union_classification, union_confidence,
         oc_likely, oc_management, oc_technical_engineering, oc_creative_product, oc_ambiguous) %>%
  mutate(title_key = tolower(trimws(title_standardized)))

df <- df %>% mutate(title_key = tolower(trimws(job_title_clean))) %>%
  left_join(clf, by = "title_key")

# Coverage
n_all <- nrow(df); n_matched <- sum(!is.na(df$union_classification))
cat(sprintf("Total: %d reviews\nMatched: %d (%.1f%%)\n", n_all, n_matched, n_matched/n_all*100))
cat("By union_classification:\n"); print(table(df$union_classification, useNA="ifany"))
cat("By oc_management:\n"); print(table(df$oc_management, useNA="ifany"))
cat("Top 10 unmatched:\n")
unmatched <- df %>% filter(is.na(union_classification)) %>% count(title_key, sort=TRUE) %>% head(10)
print(unmatched)

# Prepare covariates
df <- df %>% mutate(
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

top50 <- df %>% filter(!is.na(role_k1500)) %>% count(role_k1500,sort=TRUE) %>% slice_head(n=50) %>% pull(role_k1500)
df <- df %>% mutate(role_clean=factor(case_when(is.na(role_k1500)~"Missing_role",role_k1500%in%top50~role_k1500,TRUE~"Other_role")))

cat(sprintf("Prepared. Top roles: %d\n", length(top50)))

# Subgroups
SUBGROUPS <- list(
  full=list(label="Full sample", fn=function(d) d),
  matched=list(label="Matched titles", fn=function(d) filter(d, !is.na(union_classification))),
  unionizable=list(label="Likely unionizable", fn=function(d) filter(d, union_classification=="likely_unionizable")),
  excluded=list(label="Likely excluded (mgmt/supervisory)", fn=function(d) filter(d, union_classification=="likely_excluded")),
  ambiguous=list(label="Ambiguous (unclassified)", fn=function(d) filter(d, union_classification=="ambiguous")),
  oc_management=list(label="OC Management roles", fn=function(d) filter(d, !is.na(oc_management)&oc_management==1)),
  non_oc=list(label="Non-OC / non-management workers", fn=function(d) filter(d, (is.na(oc_management)|oc_management==0)&(is.na(union_classification)|union_classification!="likely_excluded")))
)

OUTCOMES <- c("overall_rating","career_opp","comp_benefit","senior_mgmt","wlb","culture")
OC_LABELS <- c("overall_rating"="Overall","career_opp"="Career Opp","comp_benefit"="Comp","senior_mgmt"="Senior","wlb"="WLB","culture"="Culture")

BANDWIDTHS <- list(
  global=list(label="global", fn=function(d) d),
  m20=list(label="|m|<=0.20", fn=function(d) filter(d, abs(margin)<=0.20)),
  m10=list(label="|m|<=0.10", fn=function(d) filter(d, abs(margin)<=0.10)),
  m05=list(label="|m|<=0.05", fn=function(d) filter(d, abs(margin)<=0.05)))

FILTERS <- list(n1=list(label="n>=1",type="pre_post",N=1), n5=list(label="n>=5",type="pre_post",N=5), n10=list(label="n>=10",type="pre_post",N=10))
MIN_OBS <- 200; MIN_EVENTS <- 20

apply_filter <- function(data, type, N, outcome) {
  data <- data %>% filter(!is.na(.data[[outcome]]))
  valid <- data %>% group_by(election_id) %>% summarise(n_pre=sum(post==0), n_post=sum(post==1), .groups="drop") %>% filter(n_pre>=N, n_post>=N) %>% pull(election_id)
  data %>% filter(election_id %in% valid)
}

run_one <- function(sg_key, outcome, filter_key, bw_key, emp_filter=NULL) {
  sg <- SUBGROUPS[[sg_key]]; flt <- FILTERS[[filter_key]]; bw <- BANDWIDTHS[[bw_key]]
  d <- sg$fn(df); if (!is.null(emp_filter)) d <- filter(d, is_current_employee==1)
  d <- bw$fn(d); d <- apply_filter(d, flt$type, flt$N, outcome)
  nr <- nrow(d); ne <- n_distinct(d$election_id); ng <- n_distinct(d$gvkey)
  if (nr < MIN_OBS || ne < MIN_EVENTS) {
    return(tibble(subgroup=sg_key,outcome=outcome,filter=flt$label,bandwidth=bw$label,
      estimate=NA,se=NA,pvalue=NA,n_obs=nr,n_events=ne,n_gvkeys=ng,converged=FALSE,skip="insufficient"))
  }
  fml <- as.formula(sprintf("%s ~ win + post + win_post + post:margin + win_post:margin + emp_status + seniority_f + state_clean + role_clean | gvkey + review_year", outcome))
  fit <- tryCatch(feols(fml, data=d, cluster=~gvkey+review_year, warn=FALSE, notes=FALSE),
                  error=function(e) tryCatch(feols(fml, data=d, cluster=~gvkey, warn=FALSE, notes=FALSE),
                                            error=function(e2) NULL))
  if (is.null(fit)) return(tibble(subgroup=sg_key,outcome=outcome,filter=flt$label,bandwidth=bw$label,
    estimate=NA,se=NA,pvalue=NA,n_obs=nr,n_events=ne,n_gvkeys=ng,converged=FALSE,skip="feols_error"))
  ct <- coeftable(fit)
  r <- ct["win_post",,drop=FALSE]
  tibble(subgroup=sg_key,subgroup_label=sg$label,outcome=outcome,filter=flt$label,bandwidth=bw$label,
    estimate=r[1,"Estimate"],se=r[1,"Std. Error"],pvalue=r[1,"Pr(>|t|)"],
    n_obs=nr,n_events=ne,n_gvkeys=ng,converged=TRUE,skip=NA_character_)
}

# ── Main Grid ──
cat("\n=== MAIN GRID ===\n")
all_keys <- expand.grid(subgroup=names(SUBGROUPS), outcome=OUTCOMES, filter_key=names(FILTERS), bw_key=names(BANDWIDTHS)[1:3], stringsAsFactors=FALSE)
results <- pmap_dfr(list(all_keys$subgroup, all_keys$outcome, all_keys$filter_key, all_keys$bw_key), run_one)
write_csv(results, file.path(OUTDIR, "subgroup_did_rd_results.csv"))
cat(sprintf("Main grid: %d rows\n", nrow(results)))

# ── 5A: Current employees only ──
cat("\n=== 5A: Current employees ===\n")
cur_keys <- expand.grid(subgroup=c("full","unionizable","excluded","oc_management"), outcome=OUTCOMES, filter_key="n5", bw_key=c("global","m10"), stringsAsFactors=FALSE)
cur_res <- pmap_dfr(list(cur_keys$subgroup, cur_keys$outcome, cur_keys$filter_key, cur_keys$bw_key), function(sg,oc,fk,bk) run_one(sg,oc,fk,bk,emp_filter=TRUE))
write_csv(cur_res, file.path(OUTDIR, "subgroup_current_only_results.csv"))
cat(sprintf("Current-only: %d rows\n", nrow(cur_res)))

# ── 5B: WLB bandwidth stability ──
cat("\n=== 5B: WLB bandwidth ===\n")
wlb_keys <- expand.grid(subgroup=names(SUBGROUPS), filter_key=names(FILTERS), bw_key=names(BANDWIDTHS), stringsAsFactors=FALSE)
wlb_res <- pmap_dfr(list(wlb_keys$subgroup, wlb_keys$filter_key, wlb_keys$bw_key), function(sg,fk,bk) run_one(sg,"wlb",fk,bk))
write_csv(wlb_res, file.path(OUTDIR, "wlb_bandwidth_stability_by_subgroup.csv"))
cat(sprintf("WLB: %d rows\n", nrow(wlb_res)))

# ── 5C: comp_benefit ──
cat("\n=== 5C: comp_benefit ===\n")
comp_keys <- expand.grid(subgroup=names(SUBGROUPS), filter_key=names(FILTERS), bw_key=names(BANDWIDTHS), stringsAsFactors=FALSE)
comp_res <- pmap_dfr(list(comp_keys$subgroup, comp_keys$filter_key, comp_keys$bw_key), function(sg,fk,bk) run_one(sg,"comp_benefit",fk,bk))
write_csv(comp_res, file.path(OUTDIR, "comp_benefit_by_subgroup.csv"))
cat(sprintf("Comp: %d rows\n", nrow(comp_res)))

# ── 5D: senior_mgmt ──
cat("\n=== 5D: senior_mgmt ===\n")
sen_keys <- expand.grid(subgroup=names(SUBGROUPS), filter_key=names(FILTERS), bw_key=names(BANDWIDTHS), stringsAsFactors=FALSE)
sen_res <- pmap_dfr(list(sen_keys$subgroup, sen_keys$filter_key, sen_keys$bw_key), function(sg,fk,bk) run_one(sg,"senior_mgmt",fk,bk))
write_csv(sen_res, file.path(OUTDIR, "senior_mgmt_mechanism_by_subgroup.csv"))
cat(sprintf("Senior Mgmt: %d rows\n", nrow(sen_res)))

# ── REPORT ──
cat("\n=== WLB Main Table (n5, poly1 spline) ===\n")
m <- results %>% filter(outcome=="wlb", filter=="n>=5")
for (sg in names(SUBGROUPS)) {
  r <- m %>% filter(subgroup==sg)
  vals <- sapply(c("global","|m|<=0.20","|m|<=0.10"), function(bw) {
    rr <- r %>% filter(bandwidth==bw)
    if (nrow(rr)>0 && !is.na(rr$estimate[1])) sprintf("%+.3f (%.3f) p=%.3f E=%d", rr$estimate[1], rr$se[1], rr$pvalue[1], rr$n_events[1]) else "—"
  })
  cat(sprintf("  %-30s | %s | %s | %s\n", SUBGROUPS[[sg]]$label, vals[1], vals[2], vals[3]))
}

cat("\n=== All Outcomes (global, n5) ===\n")
ao <- results %>% filter(bandwidth=="global", filter=="n>=5")
for (oc in OUTCOMES) {
  r <- ao %>% filter(outcome==oc, subgroup=="full")
  ru <- ao %>% filter(outcome==oc, subgroup=="unionizable")
  re <- ao %>% filter(outcome==oc, subgroup=="excluded")
  cat(sprintf("  %-12s full=%+.3f(p=%.3f) u=%+.3f(p=%.3f) e=%+.3f(p=%.3f)\n",
    OC_LABELS[oc],
    if(nrow(r)>0) r$estimate[1] else NA, if(nrow(r)>0) r$pvalue[1] else NA,
    if(nrow(ru)>0) ru$estimate[1] else NA, if(nrow(ru)>0) ru$pvalue[1] else NA,
    if(nrow(re)>0) re$estimate[1] else NA, if(nrow(re)>0) re$pvalue[1] else NA))
}

cat("\n=== Sample Size Warnings ===\n")
small <- results %>% filter(n_events < 50 | converged==FALSE)
cat(sprintf("%d rows with n_events<50 or failed\n", nrow(small)))
if (nrow(small)>0) print(small %>% select(subgroup,outcome,filter,bandwidth,n_events,converged) %>% head(20))

cat("\nDone.\n")
