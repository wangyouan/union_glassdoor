#!/usr/bin/env Rscript
# Full STEP 4: Filter & Bandwidth Robustness (current + former, 10 DVs, 5 filters, 4 bandwidths)
# Optimized: pre-compute filters, avoid redundant prep calls

suppressMessages({library(fixest); library(dplyr); library(tidyr); library(nanoparquet); library(readr)})

OUT <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260624/current_former_bargaining_unit/"

df <- read_parquet(paste0(OUT, "enriched_sample.parquet"))
df$sample_type <- ifelse(df$is_current_employee == 1, "current",
                  ifelse(df$is_former_employee == 1, "former", "unknown"))

DV10 <- c("overall_rating", "career_opp", "comp_benefit", "senior_mgmt", "wlb", "culture",
          "recommend", "business_outlook", "ceo_approval", "diversity")

# Pre-compute global top50 role
d_prep <- df |> mutate(
    gvkey=as.character(gvkey), review_year=as.integer(review_year),
    win=as.integer(win), post=as.integer(post), margin=as.numeric(margin), win_post=win*post,
    state_clean=case_when(!is.na(is_us_review)&is_us_review==1~state_y, TRUE~"Non_US") |> replace_na("Non_US"))
top50 <- d_prep |> filter(!is.na(role_k1500)) |> count(role_k1500,sort=TRUE) |> slice_head(n=50) |> pull(role_k1500)

prep <- function(d) {
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
    state_clean=case_when(!is.na(is_us_review)&is_us_review==1~state_y, TRUE~"Non_US") |> replace_na("Non_US"),
    role_clean=case_when(is.na(role_k1500)~"Missing_role", role_k1500%in%top50~role_k1500, TRUE~"Other_role"))
}

v7c <- function(y) as.formula(paste0(y," ~ win+post+win_post+post:margin+emp_status+seniority_f | gvkey+review_year+state_clean+role_clean"))

filters <- list(
  list(name="pre&post>=1", type="each", N=1),
  list(name="pre&post>=5", type="each", N=5),
  list(name="pre&post>=10", type="each", N=10),
  list(name="total>=10", type="total", N=10),
  list(name="total>=20", type="total", N=20)
)
bandwidths <- c("global", "0.20", "0.10", "0.05")

elig <- function(d, type, N) {
  if (type == "each") {
    d |> group_by(election_id) |> summarise(a=sum(post==0), b=sum(post==1), .groups="drop") |>
      filter(a>=N, b>=N) |> pull(election_id)
  } else {
    d |> group_by(election_id) |> summarise(n=n(), .groups="drop") |> filter(n>=N) |> pull(election_id)
  }
}

cat("=== Full Robustness Sweep ===\n")
cat(sprintf("10 DVs × 5 filters × 4 bandwidths × 2 samples = %d regressions\n", 10*5*4*2))

results <- list()
total <- 10 * 5 * 4 * 2
cnt <- 0

for (s in c("current", "former")) {
  data_s <- df[df$sample_type == s, ]
  cat(sprintf("\n--- %s ---\n", s))

  for (dv in DV10) {
    data_dv <- data_s[!is.na(data_s[[dv]]), ]

    for (flt in filters) {
      eids <- elig(data_dv, flt$type, flt$N)
      if (length(eids) < 5) next
      sub <- data_dv[data_dv$election_id %in% eids, ]

      for (bw_name in bandwidths) {
        cnt <- cnt + 1
        if (bw_name == "global") {
          sub_bw <- sub
        } else {
          sub_bw <- sub[abs(sub$margin) <= as.numeric(bw_name), ]
        }

        if (nrow(sub_bw) < 50) next
        if (sum(sub_bw$win==0) < 3 || sum(sub_bw$win==1) < 3) next

        sub_bw <- prep(sub_bw)
        fit <- tryCatch(
          feols(v7c(dv), data=sub_bw, cluster=~gvkey+review_year),
          error=function(e) NULL
        )
        if (is.null(fit)) next

        ct <- coeftable(fit)
        if (!("win_post" %in% rownames(ct))) next

        pre_mean <- mean(sub_bw[[dv]][sub_bw$post==0], na.rm=TRUE)

        results[[length(results)+1]] <- data.frame(
          sample=s, dv=dv, filter=flt$name, bandwidth=bw_name,
          coef=ct["win_post","Estimate"],
          se=ct["win_post","Std. Error"],
          p=ct["win_post","Pr(>|t|)"],
          n_reviews=nrow(sub_bw),
          n_elections=length(unique(sub_bw$election_id)),
          n_firms=length(unique(sub_bw$gvkey)),
          pre_mean=pre_mean,
          stringsAsFactors=FALSE
        )
      }
    }
    # Progress
    if (cnt %% 20 == 0) cat(sprintf("  [%d/%d] %s done\n", cnt, total, dv))
  }
}

rob_df <- bind_rows(results)
write_csv(rob_df, paste0(OUT, "filter_bandwidth_robustness_full.csv"))
cat(sprintf("\nSaved: %d rows\n", nrow(rob_df)))

# Quick WLB summary
cat("\n=== WLB current robustness ===\n")
wlb_cur <- rob_df[rob_df$sample=="current" & rob_df$dv=="wlb",]
print(wlb_cur[order(wlb_cur$filter, wlb_cur$bandwidth), c("filter","bandwidth","coef","p","n_elections")])

cat("\nDone.\n")
