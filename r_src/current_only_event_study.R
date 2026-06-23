#!/usr/bin/env Rscript
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

# ====== T3: Event Study ======
cat("Reading event study data...\n")
es <- prep(nanoparquet::read_parquet(paste0(OUT,"sample_current_eventstudy.parquet")))
es <- es |> mutate(event_q=factor(event_q, levels=as.character(-3:3)))
cat(sprintf("ES rows: %d, elections: %d\n", nrow(es), n_distinct(es$election_id)))

OUTCOMES <- c("wlb","comp_benefit","overall_rating","senior_mgmt","culture")
all_rows <- list()

for (y in OUTCOMES) {
  cat(sprintf("  Event study: %s\n", y))
  f <- as.formula(paste0(y, " ~ i(event_q, win, ref=-1) + win + post:margin + emp_status + seniority_f | gvkey+review_year+state_clean+role_clean+event_q"))
  fit <- feols(f, data=es, cluster=CL, warn=FALSE, notes=FALSE)

  ct <- as.data.frame(coeftable(fit))
  ct$term <- rownames(ct)

  # Extract event_q interaction terms: format "event_q::-3:win"
  idx <- grepl("event_q::", ct$term, fixed=TRUE)
  event_terms <- ct$term[idx]

  if (length(event_terms) == 0) {
    cat("    WARNING: no event_q terms found!\n")
    next
  }

  # Extract quarter number: "event_q::-3:win" -> "-3", "event_q::0:win" -> "0"
  eq_str <- gsub("event_q::", "", event_terms, fixed=TRUE)
  eq_str <- gsub(":win", "", eq_str, fixed=TRUE)
  eq_vals <- as.integer(eq_str)

  for (i in seq_along(event_terms)) {
    r <- ct[ct$term == event_terms[i], ]
    all_rows[[length(all_rows) + 1]] <- data.frame(
      outcome = y,
      event_q = eq_vals[i],
      estimate = r[1, "Estimate"],
      se = r[1, "Std. Error"],
      pvalue = r[1, "Pr(>|t|)"],
      stringsAsFactors = FALSE
    )
  }

  # Pre-trend: joint test event_q::-3:win + event_q::-2:win = 0
  pre <- tryCatch({
    w <- fixest::wald(fit, "event_q::-3:win + event_q::-2:win = 0", print=FALSE)
    w$p
  }, error=function(e)NA)
  cat(sprintf("    Coefs: %d, Pre-trend p: %s\n", length(eq_vals), if(is.na(pre)) "NA" else sprintf("%.4f", pre)))

  # Store pre-trend p for each row
  for (j in seq_along(event_terms)) {
    all_rows[[length(all_rows) - length(event_terms) + j]]$pretrend_p <- pre
  }
}

es_res <- bind_rows(all_rows)
write_csv(es_res, paste0(OUT,"T3_eventstudy_current.csv"))
cat("\n=== T3 current event study ===\n")
print(as.data.frame(es_res), n=100)

# ====== T3: Pooled Post ======
cat("\n--- Pooled post (current) ---\n")
pooled <- map_dfr(OUTCOMES, fit_one, d=es, label="current_pooled_post")
write_csv(pooled, paste0(OUT,"T3_pooled_post_current.csv"))
cat("\n=== T3 pooled post current ===\n")
print(as.data.frame(pooled), n=100)

cat("\n=== Pre-trend tests ===\n")
pt <- es_res |> distinct(outcome, pretrend_p)
print(as.data.frame(pt), n=100)
cat("\nT3 done.\n")
