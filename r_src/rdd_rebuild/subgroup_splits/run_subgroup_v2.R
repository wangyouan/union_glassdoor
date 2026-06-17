#!/usr/bin/env Rscript
# Subgroup DiD-RD v2: corrected v7c spec + management/technical/OC splits
library(fixest); library(nanoparquet); library(dplyr); library(purrr); library(readr); library(data.table)
setFixest_notes(FALSE); setwd("/data/disk4/workspace/projects/union_glassdoor")
OUTDIR <- "outputs/20260617/subgroup_v2"; dir.create(OUTDIR, recursive=TRUE, showWarnings=FALSE)

cat("=== DATA SETUP ===\n")
df <- nanoparquet::read_parquet("outputs/rdd_rebuild/focused_rdd_search_v7/rdd_sample_v7_enriched.parquet") |> as.data.frame()
clf <- fread("outputs/20260617/union_classified_title_universe_step1d.csv") |> as.data.frame() |>
  select(title_standardized, union_classification, oc_likely, oc_management, oc_technical_engineering, oc_creative_product) |>
  mutate(title_key=tolower(trimws(title_standardized)))
df <- df |> mutate(title_key=tolower(trimws(job_title_clean))) |> left_join(clf, by="title_key")

cat(sprintf("Total: %d reviews, Matched: %d (%.1f%%)\n", nrow(df), sum(!is.na(df$union_classification)), mean(!is.na(df$union_classification))*100))

# Covariates
df <- df |> mutate(
  gvkey=as.character(gvkey), review_year=as.integer(review_year),
  win=as.integer(win), post=as.integer(post), margin=as.numeric(margin), win_post=win*post,
  emp_status=case_when(is.na(reviewer_employment_status)~"unknown", reviewer_employment_status=="REGULAR"~"regular",
    reviewer_employment_status=="PART_TIME"~"part_time", reviewer_employment_status=="INTERN"~"intern",
    reviewer_employment_status=="CONTRACT"~"contract", TRUE~"other") |>
    factor(levels=c("regular","part_time","intern","contract","other","unknown")),
  seniority_f=factor(ifelse(is.na(seniority),0L,as.integer(seniority))),
  state_clean=ifelse(!is.na(is_us_review)&is_us_review==1, state_x, "Non_US") |> tidyr::replace_na("Non_US"))
top50 <- df |> filter(!is.na(role_k1500)) |> count(role_k1500,sort=TRUE) |> slice_head(n=50) |> pull(role_k1500)
df <- df |> mutate(role_clean=case_when(is.na(role_k1500)~"Missing_role", role_k1500%in%top50~role_k1500, TRUE~"Other_role"))

# ── SPEC: v7c exact ──
OUTCOMES <- c("overall_rating","career_opp","comp_benefit","senior_mgmt","wlb","culture")
OC_LABELS <- c("overall_rating"="Overall","career_opp"="Career","comp_benefit"="Comp","senior_mgmt"="Senior","wlb"="WLB","culture"="Culture")

make_formula <- function(outcome) as.formula(paste0(outcome, " ~ win + post + win_post + post:margin + emp_status + seniority_f | gvkey + review_year"))

CLUSTER_FML <- ~gvkey + review_year; MIN_OBS <- 200; MIN_EVENTS <- 20

# ── SUBGROUPS ──
SUBGROUPS <- list(
  full=list(label="Full sample (baseline)", fn=function(d) d),
  oc_mgmt=list(label="OC Management", fn=function(d) filter(d, !is.na(oc_management)&oc_management==1)),
  non_mgmt=list(label="Non-management", fn=function(d) filter(d, is.na(oc_management)|oc_management==0)),
  oc_tech=list(label="OC Technical/R&D", fn=function(d) filter(d, !is.na(oc_technical_engineering)&oc_technical_engineering==1)),
  non_tech=list(label="Non-technical", fn=function(d) filter(d, is.na(oc_technical_engineering)|oc_technical_engineering==0)),
  oc_any=list(label="Any OC role", fn=function(d) filter(d, !is.na(oc_likely)&oc_likely==1)),
  non_oc_pure=list(label="Pure non-OC", fn=function(d) filter(d, is.na(oc_likely)|oc_likely==0)),
  unionizable=list(label="Unionizable (NLRA eligible)", fn=function(d) filter(d, union_classification=="likely_unionizable")),
  excluded=list(label="Excluded (NLRA mgmt)", fn=function(d) filter(d, union_classification=="likely_excluded")))

# ── FILTERS & BANDWIDTHS ──
FILTERS <- list(n1=list(label="n>=1",N=1), n5=list(label="n>=5",N=5), n10=list(label="n>=10",N=10))
BANDWIDTHS <- list(global=list(label="global",fn=function(d) d), m20=list(label="|m|<=0.20",fn=function(d) filter(d,abs(margin)<=0.20)), m10=list(label="|m|<=0.10",fn=function(d) filter(d,abs(margin)<=0.10)), m05=list(label="|m|<=0.05",fn=function(d) filter(d,abs(margin)<=0.05)))

apply_filter <- function(data, N, outcome) {
  data <- data |> filter(!is.na(.data[[outcome]]))
  valid <- data |> group_by(election_id) |> summarise(n_pre=sum(post==0), n_post=sum(post==1), .groups="drop") |> filter(n_pre>=N, n_post>=N) |> pull(election_id)
  data |> filter(election_id %in% valid)
}

# ── RUN ──
run_one <- function(sg_key, outcome, flt_key, bw_key) {
  sg <- SUBGROUPS[[sg_key]]; flt <- FILTERS[[flt_key]]; bw <- BANDWIDTHS[[bw_key]]
  d <- sg$fn(df) |> bw$fn() |> apply_filter(flt$N, outcome)
  nr <- nrow(d); ne <- n_distinct(d$election_id); nw <- n_distinct(d$election_id[d$win==1]); nl <- n_distinct(d$election_id[d$win==0])
  base <- data.frame(subgroup=sg_key,subgroup_label=sg$label,outcome=outcome,filter=flt$label,bandwidth=bw$label,
    estimate=NA_real_,se=NA_real_,tstat=NA_real_,pvalue=NA_real_,n_obs=nr,n_events=ne,n_win=nw,n_loss=nl,converged=FALSE,skip=NA_character_)
  if (nr<MIN_OBS||ne<MIN_EVENTS) { base$skip<-"insufficient"; return(base) }
  fit <- tryCatch(feols(make_formula(outcome), data=d, cluster=CLUSTER_FML, warn=FALSE, notes=FALSE), error=function(e) NULL)
  if (is.null(fit)) { base$skip<-"feols_error"; return(base) }
  ct <- coeftable(fit); r <- ct["win_post",,drop=FALSE]
  base$estimate<-r[1,"Estimate"]; base$se<-r[1,"Std. Error"]; base$tstat<-r[1,"t value"]; base$pvalue<-r[1,"Pr(>|t|)"]; base$converged<-TRUE; base
}

cat("\n=== RUNNING 648 regressions ===\n")
all_combos <- expand.grid(sg_key=names(SUBGROUPS), outcome=OUTCOMES, flt_key=names(FILTERS), bw_key=names(BANDWIDTHS), stringsAsFactors=FALSE)
results <- pmap_dfr(list(all_combos$sg_key, all_combos$outcome, all_combos$flt_key, all_combos$bw_key), run_one)
write_csv(results, file.path(OUTDIR, "subgroup_v2_results.csv"))
cat(sprintf("Saved: %d rows\n", nrow(results)))

# ── VERIFICATION ──
cat("\n=== VERIFICATION ===\n")
check <- results |> filter(subgroup=="full", outcome=="wlb", filter=="n>=1", bandwidth=="global")
cat(sprintf("Full WLB (n>=1, global): tau=%+.4f p=%.4f (expected: ~0.062, p~0.010)\n", check$estimate, check$pvalue))

# ── Main contrast table: WLB (n5, global + m05) ──
cat("\n=== WLB Contrasts (n>=5) ===\n")
pairs <- list(c("oc_mgmt","non_mgmt"), c("oc_tech","non_tech"), c("oc_any","non_oc_pure"), c("unionizable","excluded"))
for (bw in c("global","m05")) {
  cat(sprintf("\n  Bandwidth: %s\n", bw))
  for (p in pairs) {
    rA <- results |> filter(outcome=="wlb",filter=="n>=5",bandwidth==bw,subgroup==p[1])
    rB <- results |> filter(outcome=="wlb",filter=="n>=5",bandwidth==bw,subgroup==p[2])
    a <- if(nrow(rA)>0) sprintf("%+.3f(%.3f) p=%.3f E=%d", rA$estimate[1], rA$se[1], rA$pvalue[1], rA$n_events[1]) else "—"
    b <- if(nrow(rB)>0) sprintf("%+.3f(%.3f) p=%.3f E=%d", rB$estimate[1], rB$se[1], rB$pvalue[1], rB$n_events[1]) else "—"
    cat(sprintf("  %-30s | %-45s | %s\n", SUBGROUPS[[p[1]]]$label, a, b))
  }
}

cat("\nDone.\n")
