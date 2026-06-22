#!/usr/bin/env Rscript
library(fixest); library(nanoparquet); library(dplyr); library(tidyr); library(readr)
setFixest_notes(FALSE)
OUT <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260622/event_study/"
df <- nanoparquet::read_parquet(paste0(OUT,"event_study_data.parquet")) |> as.data.frame()

df <- df |> mutate(
  gvkey=as.character(gvkey), review_year=as.integer(review_year),
  win=as.integer(win), margin=as.numeric(margin),
  event_q=factor(event_q, levels=as.character(-4:4)), post=as.integer(post),
  emp_status=factor(case_when(
    is.na(reviewer_employment_status)~"unknown", reviewer_employment_status=="REGULAR"~"regular",
    reviewer_employment_status=="PART_TIME"~"part_time", reviewer_employment_status=="INTERN"~"intern",
    reviewer_employment_status=="CONTRACT"~"contract", TRUE~"other"),
    levels=c("regular","part_time","intern","contract","other","unknown")),
  seniority_f=factor(ifelse(is.na(seniority),0L,as.integer(seniority))),
  state_clean=case_when(!is.na(is_us_review)&is_us_review==1~state_y,TRUE~"Non_US") |> replace_na("Non_US"))
top50 <- df |> filter(!is.na(role_k1500)) |> count(role_k1500,sort=TRUE) |> slice_head(n=50) |> pull(role_k1500)
df <- df |> mutate(role_clean=case_when(is.na(role_k1500)~"Missing_role",role_k1500%in%top50~role_k1500,TRUE~"Other_role"))

CL <- ~gvkey+review_year
OUTCOMES <- c("wlb","comp_benefit","overall_rating","culture","senior_mgmt")

run_es <- function(y){
  f <- as.formula(paste0(y," ~ i(event_q, win, ref=-1) + win + post:margin + emp_status + seniority_f | gvkey + review_year + state_clean + role_clean + event_q"))
  fit <- feols(f, data=df, cluster=CL, warn=FALSE, notes=FALSE)
  ct <- as.data.frame(coeftable(fit)); ct$term <- rownames(ct)
  dyn <- ct[grepl("event_q::.*:win", ct$term), ]
  dyn$event_q <- as.integer(gsub(".*::(-?[0-9]+):win","\\1", dyn$term))
  dyn$outcome <- y
  pre <- tryCatch(fixest::wald(fit, "event_q::(-4|-3|-2):win", print=FALSE), error=function(e)NULL)
  dyn$pretrend_p <- if(!is.null(pre)) pre$p else NA
  dyn[,c("outcome","event_q","Estimate","Std. Error","Pr(>|t|)","pretrend_p")]
}
res <- bind_rows(lapply(OUTCOMES, run_es))
names(res) <- c("outcome","event_q","estimate","se","pvalue","pretrend_p")
res <- res |> mutate(ci_lo=estimate-1.96*se, ci_hi=estimate+1.96*se) |> arrange(outcome,event_q)
write_csv(res, paste0(OUT,"event_study_coefs.csv"))
print(res, n=100)
cat("\n=== Pre-trend tests ===\n")
print(res |> distinct(outcome, pretrend_p))

# Aggregate version
cat("\n=== Aggregate event study ===\n")
agg <- df |> group_by(election_id, gvkey, win, event_q) |>
  summarise(across(all_of(OUTCOMES), ~mean(.x, na.rm=TRUE)),
            margin=first(margin), n=n(), review_year=as.integer(round(mean(review_year))), .groups="drop") |>
  mutate(gvkey=as.character(gvkey), event_q=factor(event_q, levels=as.character(-4:4)),
         post=as.integer(as.integer(as.character(event_q))>=0))

run_agg <- function(y){
  f <- as.formula(paste0(y," ~ i(event_q, win, ref=-1) + win + post:margin | gvkey + event_q"))
  fit <- feols(f, data=agg, weights=~n, cluster=~gvkey, warn=FALSE, notes=FALSE)
  ct <- as.data.frame(coeftable(fit)); ct$term <- rownames(ct)
  dyn <- ct[grepl("event_q::.*:win", ct$term), ]
  dyn$event_q <- as.integer(gsub(".*::(-?[0-9]+):win","\\1", dyn$term)); dyn$outcome <- y
  dyn[,c("outcome","event_q","Estimate","Std. Error","Pr(>|t|)")]
}
agg_res <- bind_rows(lapply(OUTCOMES, run_agg))
names(agg_res) <- c("outcome","event_q","estimate","se","pvalue")
write_csv(agg_res, paste0(OUT,"event_study_agg_coefs.csv"))
print(agg_res, n=100)
