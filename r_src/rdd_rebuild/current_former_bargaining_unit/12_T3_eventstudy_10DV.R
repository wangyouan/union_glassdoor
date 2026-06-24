#!/usr/bin/env Rscript
# T3 Event Study — 10 DVs, main filter total>=10 only (5-FE on 260k rows too slow for full sweep)
# Spec: event_q×win interactions + state/role on RHS, gvkey+review_year+event_q FEs

suppressMessages({library(fixest); library(dplyr); library(tidyr); library(nanoparquet); library(readr); library(purrr)})

OUT <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260624/current_former_bargaining_unit/"

DV10 <- c("overall_rating","career_opp","comp_benefit","senior_mgmt","wlb","culture",
          "recommend","business_outlook","ceo_approval","diversity")

df <- read_parquet(paste0(OUT, "enriched_sample.parquet"))
cur <- df[df$is_current_employee == 1, ]

cat(sprintf("Current: %d rows\n", nrow(cur)))

prep <- function(d){
  d |> mutate(
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
}

prep2 <- function(d) {
  d <- prep(d)
  top50 <- d |> filter(!is.na(role_k1500)) |> count(role_k1500,sort=TRUE) |> slice_head(n=50) |> pull(role_k1500)
  d |> mutate(role_clean=case_when(is.na(role_k1500)~"Missing_role", role_k1500%in%top50~role_k1500, TRUE~"Other_role"))
}

cur <- prep2(cur)

# Map event_time_month to quarters
cur$event_q_raw <- pmax(-3, pmin(3, floor(cur$event_time_month / 3)))
cur <- cur[cur$event_q_raw >= -3 & cur$event_q_raw <= 3, ]
cur$event_q <- factor(cur$event_q_raw, levels=as.character(-3:3))
cat(sprintf("After clamp to Q[-3,3]: %d rows\n", nrow(cur)))

# Total>=10 filter (on wlb)
cur_wlb <- cur[!is.na(cur$wlb), ]
eids_total10 <- cur_wlb |> group_by(election_id) |> summarise(n=n(), .groups="drop") |> filter(n>=10) |> pull(election_id)
cat(sprintf("total>=10: %d elections\n", length(eids_total10)))

# Event study spec: 3 FEs (gvkey+review_year+event_q), state/role on RHS
# This is the "保守版" from CLAUDE.md
es_fml <- function(y) as.formula(paste0(y,
  " ~ i(event_q,win,ref='-1') + win + post:margin + emp_status + seniority_f + state_clean + role_clean | gvkey + review_year + event_q"))

CL <- ~gvkey + review_year

t3_rows <- list()

for (dv in DV10) {
  cat(sprintf("  %s: ", dv))
  d_dv <- cur[cur$election_id %in% eids_total10 & !is.na(cur[[dv]]), ]
  ne <- n_distinct(d_dv$election_id)
  cat(sprintf("%d reviews, %d elections... ", nrow(d_dv), ne))

  if (ne < 10) { cat("insufficient\n"); next }

  f <- tryCatch(feols(es_fml(dv), d_dv, cluster=CL, warn=FALSE, notes=FALSE), error=function(e) NULL)
  if (is.null(f)) { cat("model_failed\n"); next }

  # Pre-trend: quarters -3 and -2 jointly = 0
  pre <- tryCatch(fixest::wald(f, "event_q::(-3|-2):win", print=FALSE)$p, error=function(e) NA)

  # t=0 effect
  t0 <- NA; t0_est <- NA; t0_se <- NA
  ct <- coeftable(f)
  rn <- grep("event_q::0:win", rownames(ct), value=TRUE)
  if (length(rn)) {
    t0_est <- ct[rn[1], "Estimate"]
    t0_se <- ct[rn[1], "Std. Error"]
    t0 <- ct[rn[1], "Pr(>|t|)"]
  }

  # Pooled post effect
  fp2 <- tryCatch(feols(as.formula(paste0(dv,
    " ~ win+post+win_post+post:margin+emp_status+seniority_f+state_clean+role_clean | gvkey+review_year")),
    d_dv, cluster=CL, warn=FALSE, notes=FALSE), error=function(e) NULL)

  if (is.null(fp2) || !("win_post" %in% rownames(coeftable(fp2)))) {
    pooled_est <- NA; pooled_se <- NA; pooled_p <- NA
  } else {
    pooled_est <- coeftable(fp2)["win_post","Estimate"]
    pooled_se <- coeftable(fp2)["win_post","Std. Error"]
    pooled_p <- coeftable(fp2)["win_post","Pr(>|t|)"]
  }

  cat(sprintf("pre=%.3f, t0_est=%.4f, pooled=%.4f(p=%.3f)\n", pre, t0_est, pooled_est, pooled_p))

  t3_rows[[length(t3_rows)+1]] <- tibble(
    table="T3", filter="total>=10", outcome=dv,
    pretrend_p=pre, t0_est=t0_est, t0_se=t0_se, t0_p=t0,
    pooled_est=pooled_est, pooled_se=pooled_se, pooled_p=pooled_p,
    n_reviews=nrow(d_dv), n_events=ne)
}

t3_out <- bind_rows(t3_rows)
write_csv(t3_out, paste0(OUT, "T3_eventstudy_10DV.csv"))
cat(sprintf("\nSaved T3_eventstudy_10DV.csv (%d rows)\n", nrow(t3_out)))

print(t3_out[, c("outcome","pretrend_p","t0_est","t0_p","pooled_est","pooled_p","n_events")], n=20)

cat("\nDone.\n")
