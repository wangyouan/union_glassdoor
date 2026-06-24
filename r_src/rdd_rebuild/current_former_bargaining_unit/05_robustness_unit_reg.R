#!/usr/bin/env Rscript
# Focused robustness (STEP 4) + unit-member regression (STEP 9)
# Key filters: total>=10, total>=20, pre&post>=5
# Key bandwidths: global, 0.10
# Current sample only, 10 DVs

suppressMessages({library(fixest); library(dplyr); library(tidyr); library(nanoparquet); library(readr)})

OUT <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260624/"

df <- read_parquet(paste0(OUT, "enriched_sample.parquet"))
df$sample_type <- ifelse(df$is_current_employee == 1, "current",
                  ifelse(df$is_former_employee == 1, "former", "unknown"))

DV10 <- c("overall_rating", "career_opp", "comp_benefit", "senior_mgmt", "wlb", "culture",
          "recommend", "business_outlook", "ceo_approval", "diversity")

# Helpers
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

v7c <- function(y) as.formula(paste0(y," ~ win+post+win_post+post:margin+emp_status+seniority_f | gvkey+review_year+state_clean+role_clean"))

# ─── STEP 4 (focused): Filter + Bandwidth Robustness ────────────────────
cat("=== STEP 4: Focused Robustness ===\n")

filters <- list(
  list(name="total>=10", type="total", N=10),
  list(name="total>=20", type="total", N=20),
  list(name="pre&post>=5", type="each", N=5)
)
bandwidths <- c("global", "0.10")
cur <- df[df$sample_type == "current", ]

results <- list()
for (dv in DV10) {
  cat(sprintf("  %s:\n", dv))
  cur_dv <- cur[!is.na(cur[[dv]]), ]
  for (flt in filters) {
    if (flt$type == "total") {
      eid_n <- cur_dv |> group_by(election_id) |> summarise(n=n(), .groups="drop")
      eids <- eid_n$election_id[eid_n$n >= flt$N]
    } else {
      eids <- cur_dv |> group_by(election_id) |>
        summarise(a=sum(post==0), b=sum(post==1), .groups="drop") |>
        filter(a>=flt$N, b>=flt$N) |> pull(election_id)
    }
    sub <- cur_dv[cur_dv$election_id %in% eids, ]
    for (bw_name in bandwidths) {
      if (bw_name == "global") { sub_bw <- sub }
      else { sub_bw <- sub[abs(sub$margin) <= as.numeric(bw_name), ] }
      if (nrow(sub_bw) < 100) next
      sub_bw <- prep2(sub_bw)
      fit <- tryCatch(feols(v7c(dv), data=sub_bw, cluster=~gvkey+review_year), error=function(e)NULL)
      if (is.null(fit)) next
      ct <- coeftable(fit)
      if (!("win_post" %in% rownames(ct))) next
      results[[length(results)+1]] <- data.frame(
        dv=dv, filter=flt$name, bandwidth=bw_name,
        coef=ct["win_post","Estimate"], se=ct["win_post","Std. Error"],
        p=ct["win_post","Pr(>|t|)"],
        n_reviews=nrow(sub_bw), n_elections=length(unique(sub_bw$election_id)))
    }
  }
}
rob_df <- bind_rows(results)
write_csv(rob_df, paste0(OUT, "filter_bandwidth_robustness.csv"))
cat(sprintf("Saved: %d rows\n", nrow(rob_df)))

# Quick summary
cat("\nWLB robustness:\n")
print(rob_df[rob_df$dv=="wlb", c("filter","bandwidth","coef","p","n_elections")])

# ─── STEP 9: Unit-member vs non-unit regression ─────────────────────────
cat("\n=== STEP 9: Unit-Member vs Non-Unit Regression ===\n")

matches <- read_parquet(paste0(OUT, "review_title_unit_matches.parquet"))
df_m <- df |> left_join(matches |> select(review_id, unit_match, unit_match_confidence), by="review_id")

# Focus on current only, with unit_match not ambiguous
cur_m <- df_m[df_m$sample_type == "current" & !is.na(df_m$unit_match), ]
# Create unit_member indicator
cur_m$unit_member <- as.integer(cur_m$unit_match == "member")
cur_m$non_member <- as.integer(cur_m$unit_match == "not_member")

cat(sprintf("Current, matched: %d reviews\n", nrow(cur_m)))
cat(sprintf("  Members: %d, Non-members: %d, Ambiguous: %d\n",
            sum(cur_m$unit_match=="member"), sum(cur_m$unit_match=="not_member"),
            sum(cur_m$unit_match=="ambiguous")))

# For regression: use non-member vs member, exclude ambiguous
cur_reg <- cur_m[cur_m$unit_match != "ambiguous", ]

# group indicator: 1 = member, 0 = non_member (reference)
cur_reg$is_member <- as.integer(cur_reg$unit_match == "member")

# Interaction spec
v7c_member <- function(y) as.formula(paste0(y,
  " ~ win + post + win_post + is_member + post:is_member + win_post:is_member + post:margin + emp_status + seniority_f | gvkey + review_year + state_clean + role_clean"))

member_results <- list()

for (dv in DV10) {
  cat(sprintf("  %s... ", dv))
  cur_dv <- cur_reg[!is.na(cur_reg[[dv]]), ]
  # total>=10 filter
  eids <- cur_dv |> group_by(election_id) |> summarise(n=n(), .groups="drop") |> filter(n>=10) |> pull(election_id)
  sub <- cur_dv[cur_dv$election_id %in% eids, ]

  n_mem <- sum(sub$is_member == 1)
  n_non <- sum(sub$is_member == 0)
  if (n_mem < 30 || n_non < 30) {
    member_results[[length(member_results)+1]] <- data.frame(
      dv=dv, non_member_effect=NA, member_effect=NA, diff=NA, diff_p=NA,
      n_non_member=n_non, n_member=n_mem, note="insufficient_sample")
    cat(sprintf("insufficient (mem=%d, non=%d)\n", n_mem, n_non))
    next
  }

  sub <- prep2(sub)
  fit <- tryCatch(feols(v7c_member(dv), data=sub, cluster=~gvkey+review_year), error=function(e)NULL)
  if (is.null(fit)) {
    member_results[[length(member_results)+1]] <- data.frame(
      dv=dv, non_member_effect=NA, member_effect=NA, diff=NA, diff_p=NA,
      n_non_member=n_non, n_member=n_mem, note="model_failed")
    cat("model_failed\n")
    next
  }

  ct <- coeftable(fit)
  wp_est <- if("win_post" %in% rownames(ct)) ct["win_post","Estimate"] else NA
  wpm_est <- if("win_post:is_member" %in% rownames(ct)) ct["win_post:is_member","Estimate"] else NA
  wpm_p <- if("win_post:is_member" %in% rownames(ct)) ct["win_post:is_member","Pr(>|t|)"] else NA

  non_member_effect <- wp_est
  member_effect <- wp_est + wpm_est
  diff <- wpm_est

  cat(sprintf("non=%.4f, mem=%.4f, diff=%.4f, p=%.4f\n", non_member_effect, member_effect, diff, wpm_p))

  member_results[[length(member_results)+1]] <- data.frame(
    dv=dv, non_member_effect=non_member_effect, member_effect=member_effect,
    diff=diff, diff_p=wpm_p, n_non_member=n_non, n_member=n_mem,
    n_elections=length(unique(sub$election_id)), note="")
}

mem_df <- bind_rows(member_results)
write_csv(mem_df, paste0(OUT, "unit_member_regression_results.csv"))
cat(sprintf("\nSaved unit_member_regression_results.csv (%d rows)\n", nrow(mem_df)))

cat("\nDone.\n")
