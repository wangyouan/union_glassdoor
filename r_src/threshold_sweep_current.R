library(fixest); library(dplyr); library(tidyr); library(nanoparquet); library(readr); library(purrr)

OUT <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260622/current_thresholds/"
cur <- read_parquet(paste0(OUT, "current_base.parquet"))

prep <- function(d) {
  d <- d |>
    mutate(
      gvkey = as.character(gvkey),
      review_year = as.integer(review_year),
      win = as.integer(win),
      post = as.integer(post),
      margin = as.numeric(margin),
      win_post = win * post,
      emp_status = case_when(
        is.na(reviewer_employment_status) ~ "unknown",
        reviewer_employment_status == "REGULAR" ~ "regular",
        reviewer_employment_status == "PART_TIME" ~ "part_time",
        reviewer_employment_status == "INTERN" ~ "intern",
        reviewer_employment_status == "CONTRACT" ~ "contract",
        TRUE ~ "other"
      ) |> factor(levels = c("regular", "part_time", "intern", "contract", "other", "unknown")),
      seniority_f = factor(ifelse(is.na(seniority), 0L, as.integer(seniority))),
      state_clean = case_when(
        !is.na(is_us_review) & is_us_review == 1 ~ state_y,
        TRUE ~ "Non_US"
      ) |> replace_na("Non_US")
    )
  top50 <- d |>
    filter(!is.na(role_k1500)) |>
    count(role_k1500, sort = TRUE) |>
    slice_head(n = 50) |>
    pull(role_k1500)
  d |>
    mutate(role_clean = case_when(
      is.na(role_k1500) ~ "Missing_role",
      role_k1500 %in% top50 ~ role_k1500,
      TRUE ~ "Other_role"
    ))
}

cur <- prep(cur)
message("Prep done: ", nrow(cur), " rows, ", n_distinct(cur$election_id), " elections")

CL <- ~ gvkey + review_year

v7c <- function(y) {
  as.formula(paste0(y, " ~ win + post + win_post + post:margin + emp_status + seniority_f | gvkey + review_year + state_clean + role_clean"))
}

elig_each <- function(d, N) {
  d |>
    group_by(election_id) |>
    summarise(a = sum(post == 0), b = sum(post == 1), .groups = "drop") |>
    filter(a >= N, b >= N) |>
    pull(election_id)
}

elig_total <- function(d, N) {
  d |>
    group_by(election_id) |>
    summarise(n = n(), .groups = "drop") |>
    filter(n >= N) |>
    pull(election_id)
}

fit_one <- function(y, d, ftype, thr) {
  f <- tryCatch(
    feols(v7c(y), data = d, cluster = CL, warn = FALSE, notes = FALSE),
    error = function(e) NULL
  )
  base <- tibble(
    filter_type = ftype,
    threshold = thr,
    outcome = y,
    n_obs = nrow(d),
    n_events = n_distinct(d$election_id)
  )
  if (is.null(f) || !("win_post" %in% rownames(coeftable(f)))) {
    return(bind_cols(base, tibble(estimate = NA_real_, se = NA_real_, pvalue = NA_real_)))
  }
  r <- coeftable(f)["win_post", , drop = FALSE]
  bind_cols(base, tibble(
    estimate = r[1, "Estimate"],
    se = r[1, "Std. Error"],
    pvalue = r[1, "Pr(>|t|)"]
  ))
}

OUTS <- c("wlb", "comp_benefit", "overall_rating", "senior_mgmt", "culture", "career_opp")

res <- list()

# pre&post >= N
for (N in c(1, 3, 5, 10, 20, 25, 50)) {
  ids <- elig_each(cur, N)
  d <- cur |> filter(election_id %in% ids)
  n_ev <- n_distinct(d$election_id)
  message("pre&post>=", N, ": ", nrow(d), " reviews, ", n_ev, " elections")
  if (n_ev >= 20) {
    res[[paste0("each", N)]] <- map_dfr(OUTS, fit_one, d = d, ftype = "pre&post>=N", thr = N)
  } else {
    message("  -> skipped (<20 elections)")
  }
}

# total >= N
for (N in c(10, 20, 50)) {
  ids <- elig_total(cur, N)
  d <- cur |> filter(election_id %in% ids)
  n_ev <- n_distinct(d$election_id)
  message("total>=", N, ": ", nrow(d), " reviews, ", n_ev, " elections")
  if (n_ev >= 20) {
    res[[paste0("tot", N)]] <- map_dfr(OUTS, fit_one, d = d, ftype = "total>=N", thr = N)
  } else {
    message("  -> skipped (<20 elections)")
  }
}

out <- bind_rows(res)
out$sig <- cut(out$pvalue, c(-Inf, 0.01, 0.05, 0.10, Inf), labels = c("***", "**", "*", ""))
write_csv(out, paste0(OUT, "current_threshold_sweep.csv"))
message("CSV written: ", nrow(out), " rows")

cat("\n=== WLB across thresholds (current) ===\n")
print(out |> filter(outcome == "wlb") |> select(filter_type, threshold, estimate, se, pvalue, sig, n_events), n = 50)

cat("\n=== Comp across thresholds (current) ===\n")
print(out |> filter(outcome == "comp_benefit") |> select(filter_type, threshold, estimate, pvalue, sig, n_events), n = 50)
