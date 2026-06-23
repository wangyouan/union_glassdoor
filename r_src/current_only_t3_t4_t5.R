#!/usr/bin/env Rscript
# T3 Event Study + Pooled Post + T4 Subgroups + T5 Bandwidth (current only)
library(fixest); library(dplyr); library(tidyr); library(readr); library(purrr); library(nanoparquet)
options(fixest_notes=FALSE)
OUT <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260622/current_only/"

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
CL <- ~gvkey+review_year
v7c <- function(y) as.formula(paste0(y," ~ win+post+win_post+post:margin+emp_status+seniority_f | gvkey+review_year+state_clean+role_clean"))

fit_one <- function(y, d, label){
  f <- tryCatch(feols(v7c(y), data=d, cluster=CL, warn=FALSE, notes=FALSE), error=function(e)NULL)
  if(is.null(f)) return(tibble(sample=label, outcome=y, estimate=NA, se=NA, pvalue=NA, n_obs=nrow(d), n_events=n_distinct(d$election_id)))
  r <- coeftable(f)["win_post", , drop=FALSE]
  tibble(sample=label, outcome=y, estimate=r[1,"Estimate"], se=r[1,"Std. Error"], pvalue=r[1,"Pr(>|t|)"],
         n_obs=nrow(d), n_events=n_distinct(d$election_id))
}

OUTCOMES <- c("wlb","comp_benefit","overall_rating","senior_mgmt","culture")

# ====== T3: Event Study with manual pre-trend ======
cat("=== T3: Event Study (current only) ===\n")
es <- prep(nanoparquet::read_parquet(paste0(OUT,"sample_current_eventstudy.parquet")))
es <- es |> mutate(event_q=factor(event_q, levels=as.character(-3:3)))
cat(sprintf("ES rows: %d, elections: %d\n", nrow(es), n_distinct(es$election_id)))

all_rows <- list()
pt_rows <- list()

for (y in OUTCOMES) {
  cat(sprintf("  %s...\n", y))
  f <- as.formula(paste0(y, " ~ i(event_q, win, ref=-1) + win + post:margin + emp_status + seniority_f | gvkey+review_year+state_clean+role_clean+event_q"))
  fit <- feols(f, data=es, cluster=CL, warn=FALSE, notes=FALSE)

  ct <- as.data.frame(coeftable(fit))
  ct$term <- rownames(ct)

  # Extract event_q interaction terms
  idx <- grepl("event_q::", ct$term, fixed=TRUE)
  event_terms <- ct$term[idx]

  eq_str <- gsub("event_q::", "", event_terms, fixed=TRUE)
  eq_str <- gsub(":win", "", eq_str, fixed=TRUE)
  eq_vals <- as.integer(eq_str)

  for (i in seq_along(event_terms)) {
    r <- ct[ct$term == event_terms[i], ]
    all_rows[[length(all_rows) + 1]] <- data.frame(
      outcome = y, event_q = eq_vals[i],
      estimate = r[1, "Estimate"], se = r[1, "Std. Error"],
      pvalue = r[1, "Pr(>|t|)"], stringsAsFactors = FALSE)
  }

  # Manual pre-trend: b_{-3} + b_{-2} = 0 ?
  # Extract vcov, find indices for event_q::-3:win and event_q::-2:win
  V <- vcov(fit)
  b <- coef(fit)
  i3 <- which(names(b) == "event_q::-3:win")
  i2 <- which(names(b) == "event_q::-2:win")

  pretrend_p <- NA
  if (length(i3) == 1 && length(i2) == 1) {
    # H0: b3 + b2 = 0  (joint pre-trend)
    # Actually test separately: is each individually ~0 and is joint = 0?
    # Simple approach: test H0: b_{-3} = b_{-2} = 0
    # But we want "are pre-period coeffs jointly zero?" -> F-test on both

    # Use linearHypothesis-style test via vcov
    R <- matrix(0, nrow=2, ncol=length(b))
    colnames(R) <- names(b)
    R[1, i3] <- 1
    R[2, i2] <- 1
    # Wald test: (R*b)' * (R*V*R')^{-1} * (R*b) ~ chi2(2)
    Rb <- R %*% b
    RVR <- R %*% V %*% t(R)
    # Use generalized inverse for numerical stability
    stat <- tryCatch(as.numeric(t(Rb) %*% MASS::ginv(RVR) %*% Rb), error=function(e)NA)
    pretrend_p <- tryCatch(pchisq(stat, df=2, lower.tail=FALSE), error=function(e)NA)
  }
  cat(sprintf("    coefs=%d, pre-trend F-stat=%.2f p=%.4f\n", length(eq_vals),
              if(is.na(pretrend_p)) NA else stat, if(is.na(pretrend_p)) NA else pretrend_p))
  pt_rows[[y]] <- data.frame(outcome=y, pretrend_p=pretrend_p, stringsAsFactors=FALSE)
}

es_res <- bind_rows(all_rows)
# Merge pre-trend p
pt_df <- bind_rows(pt_rows)
es_res <- es_res |> left_join(pt_df, by="outcome")

write_csv(es_res, paste0(OUT,"T3_eventstudy_current.csv"))
cat("\nT3 event study saved.\n")

# ====== T3: Pooled Post ======
cat("\n=== T3: Pooled Post (current) ===\n")
pooled <- map_dfr(OUTCOMES, fit_one, d=es, label="current_pooled_post")
write_csv(pooled, paste0(OUT,"T3_pooled_post_current.csv"))
cat("Saved:\n")
print(as.data.frame(pooled), n=100)

# ====== T4: Subgroups (current) ======
cat("\n=== T4: Subgroups (current) ===\n")
curd <- prep(nanoparquet::read_parquet(paste0(OUT,"sample_current_n5.parquet")))
cat(sprintf("Current n5: %d rows, %d elections\n", nrow(curd), n_distinct(curd$election_id)))

# Check union_classification values
cat("union_classification values:\n")
print(table(curd$union_classification, useNA="ifany"))

sub <- function(d, grp) filter(d, union_classification==grp)
t4 <- bind_rows(
  map_dfr(OUTCOMES, fit_one, d=sub(curd, "likely_excluded"),    label="excluded_current"),
  map_dfr(OUTCOMES, fit_one, d=sub(curd, "likely_unionizable"), label="unionizable_current"))
write_csv(t4, paste0(OUT,"T4_current.csv"))
cat("\nT4 saved:\n")
print(as.data.frame(t4), n=100)

# ====== T5: WLB Bandwidth (current) ======
cat("\n=== T5: WLB Bandwidth (current) ===\n")
bw_fit <- function(bw){
  d <- curd
  lbl <- bw
  if(bw=="m20") { d <- filter(d, abs(margin)<=0.20); lbl <- "current_m20" }
  if(bw=="m10") { d <- filter(d, abs(margin)<=0.10); lbl <- "current_m10" }
  if(bw=="m05") { d <- filter(d, abs(margin)<=0.05); lbl <- "current_m05" }
  if(bw=="global") lbl <- "current_global"
  r <- fit_one("wlb", d, lbl); r$bandwidth <- bw; r
}
t5 <- bind_rows(lapply(c("global","m20","m10","m05"), bw_fit))
write_csv(t5, paste0(OUT,"T5_wlb_bandwidth_current.csv"))
cat("\nT5 saved:\n")
print(as.data.frame(t5), n=100)

cat("\n=== Pre-trend summary ===\n")
print(as.data.frame(pt_df), n=100)
cat("\nAll steps done.\n")
