#!/usr/bin/env Rscript
# T3 Event Study + T7 rdrobust + T8 Firm-quarter dynamic — all 10 DVs
# Extends existing sweep scripts from 6 rating DVs to full 10 DV set.
# Uses enriched_sample.parquet (current-only for T7/T8, current for T3).

suppressMessages({
  library(fixest); library(dplyr); library(tidyr); library(nanoparquet); library(readr); library(purrr)
  library(rdrobust)
})

OUT <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260624/current_former_bargaining_unit/"
set.seed(42)

# ─── 10 DVs ────────────────────────────────────────────────────────────────
DV10 <- c("overall_rating","career_opp","comp_benefit","senior_mgmt","wlb","culture",
          "recommend","business_outlook","ceo_approval","diversity")

# ─── Load & prep ───────────────────────────────────────────────────────────
df <- read_parquet(paste0(OUT, "enriched_sample.parquet"))
cur <- df[df$is_current_employee == 1, ]

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
cat(sprintf("Current prepped: %d rows, %d elections\n", nrow(cur), n_distinct(cur$election_id)))

# ─── Filters ──────────────────────────────────────────────────────────────
THR <- c(lapply(c(1,3,5,10,20), function(N) list(type="each",N=N)),
         lapply(c(10,20),       function(N) list(type="total",N=N)))

elig <- function(d, type, N) {
  if (type == "each") {
    d |> group_by(election_id) |> summarise(a=sum(post==0), b=sum(post==1), .groups="drop") |>
      filter(a>=N, b>=N) |> pull(election_id)
  } else {
    d |> group_by(election_id) |> summarise(n=n(), .groups="drop") |> filter(n>=N) |> pull(election_id)
  }
}

thr_label <- function(t) paste0(ifelse(t$type=="each","pre&post>=","total>="), t$N)

CL <- ~gvkey + review_year

cat("\nFilters: ", length(THR), " thresholds\n")

# ═══════════════════════════════════════════════════════════════════════════
# T3: Event Study (review-level, 10 DVs)
# ═══════════════════════════════════════════════════════════════════════════
cat("\n========================================\n")
cat("T3: Event Study (10 DVs)\n")
cat("========================================\n")

# T3 spec: event_q FE + gvkey + review_year absorbed; state_clean+role_clean on RHS
# To avoid 5-FE R GC crash, use 2-FE RHS approach (state_clean+role_clean as controls)
es_fml <- function(y) as.formula(paste0(y,
  " ~ i(event_q,win,ref='-1') + win + post:margin + emp_status + seniority_f + state_clean + role_clean | gvkey + review_year + event_q"))

t3_rows <- list()
for (t_iter in THR) {
  # DV-agnostic filter: use wlb to determine election set (consistent with original)
  cur_wlb <- cur[!is.na(cur$wlb), ]
  eids <- elig(cur_wlb, t_iter$type, t_iter$N)
  if (length(eids) < 20) next

  # For each DV, filter to non-NA and clamp event_q
  for (dv in DV10) {
    d_dv <- cur[cur$election_id %in% eids & !is.na(cur[[dv]]), ]
    d_dv <- d_dv[d_dv$event_time_month >= -9 & d_dv$event_time_month <= 9, ]  # approx quarterly
    # Convert event_time_month to quarters
    # Map event_time_month to quarters: floor(month/3), clamped to [-3,3]
    d_dv$event_q <- pmax(-3, pmin(3, floor(d_dv$event_time_month / 3)))
    d_dv$event_q <- factor(d_dv$event_q, levels=as.character(-3:3))
    ne <- n_distinct(d_dv$election_id)
    if (ne < 10) next

    f <- tryCatch(feols(es_fml(dv), d_dv, cluster=CL, warn=FALSE, notes=FALSE), error=function(e) NULL)
    pre <- tryCatch(fixest::wald(f, "event_q::(-3|-2):win", print=FALSE)$p, error=function(e) NA)
    t0 <- NA
    if (!is.null(f)) {
      ct <- coeftable(f)
      rn <- grep("event_q::0:win", rownames(ct), value=TRUE)
      if (length(rn)) t0 <- ct[rn[1], "Pr(>|t|)"]
    }

    # Pooled post effect
    fp2 <- tryCatch(feols(as.formula(paste0(dv,
      " ~ win+post+win_post+post:margin+emp_status+seniority_f+state_clean+role_clean | gvkey+review_year")),
      d_dv, cluster=CL, warn=FALSE, notes=FALSE), error=function(e) NULL)

    v <- if(is.null(fp2) || !("win_post" %in% rownames(coeftable(fp2)))) {
      c(NA, NA, NA)
    } else {
      coeftable(fp2)["win_post", c("Estimate","Std. Error","Pr(>|t|)")]
    }

    t3_rows[[length(t3_rows)+1]] <- tibble(
      table="T3", filter=thr_label(t_iter), outcome=dv,
      pretrend_p=pre, t0_p=t0, pooled_est=v[1], pooled_se=v[2], pooled_p=v[3], n_events=ne)
  }
  cat(sprintf("  %s: %d elections\n", thr_label(t_iter), length(eids)))
}

t3_out <- bind_rows(t3_rows)
write_csv(t3_out, paste0(OUT, "T3_eventstudy_10DV.csv"))
cat(sprintf("Saved T3_eventstudy_10DV.csv (%d rows)\n", nrow(t3_out)))

# ═══════════════════════════════════════════════════════════════════════════
# T7: Aggregate rdrobust (election-level delta, 10 DVs, p=2, q=3)
# ═══════════════════════════════════════════════════════════════════════════
cat("\n========================================\n")
cat("T7: Aggregate rdrobust (10 DVs, p=2, q=3)\n")
cat("========================================\n")

safe <- function(m, r, c=1) { out <- tryCatch(m[r,c], error=function(e) NA); if(is.null(out)) NA else out }

run_rd <- function(d, dv) {
  # Aggregate: election-level pre/post means, then delta
  agg <- d |> group_by(election_id) |>
    summarise(
      pre = mean(.data[[dv]][post==0], na.rm=TRUE),
      postm = mean(.data[[dv]][post==1], na.rm=TRUE),
      margin = first(margin),
      .groups = "drop") |>
    mutate(delta = postm - pre) |>
    filter(is.finite(delta), is.finite(margin))

  if (nrow(agg) < 20) return(tibble(tau_conv=NA, se_conv=NA, p_conv=NA,
    tau_bc=NA, se_rob=NA, p_rob=NA, h=NA, n_eff=NA, n_elec=nrow(agg)))

  rr <- tryCatch(rdrobust(y=agg$delta, x=agg$margin, c=0, kernel="triangular",
                          p=2, q=3, bwselect="mserd"), error=function(e) NULL)

  if (is.null(rr)) return(tibble(tau_conv=NA, se_conv=NA, p_conv=NA,
    tau_bc=NA, se_rob=NA, p_rob=NA, h=NA, n_eff=NA, n_elec=nrow(agg)))

  tibble(
    tau_conv = safe(rr$coef, "Conventional"), se_conv = safe(rr$se, "Conventional"),
    p_conv = safe(rr$pv, "Conventional"),
    tau_bc = safe(rr$coef, "Bias-Corrected"), se_rob = safe(rr$se, "Robust"),
    p_rob = safe(rr$pv, "Robust"),
    p = 2L, q = 3L,
    h = rr$bws["h","left"], n_eff = sum(rr$N_h), n_elec = nrow(agg))
}

t7_rows <- list()
for (t_iter in THR) {
  # DV-agnostic filter on wlb
  cur_wlb <- cur[!is.na(cur$wlb), ]
  eids <- elig(cur_wlb, t_iter$type, t_iter$N)
  ne <- length(eids)
  if (ne < 20) next
  d <- cur[cur$election_id %in% eids, ]

  for (dv in DV10) {
    d_dv <- d[!is.na(d[[dv]]), ]
    if (n_distinct(d_dv$election_id) < 20) next
    r <- run_rd(d_dv, dv)
    r$filter <- thr_label(t_iter); r$outcome <- dv
    t7_rows[[length(t7_rows)+1]] <- r
  }
  cat(sprintf("  %s: %d elections\n", thr_label(t_iter), ne))
}

t7_out <- bind_rows(t7_rows) |>
  mutate(
    sig_conv = cut(p_conv, c(-Inf,.01,.05,.10,Inf), labels=c("***","**","*","")),
    sig_rob  = cut(p_rob,  c(-Inf,.01,.05,.10,Inf), labels=c("***","**","*",""))) |>
  select(filter, outcome, tau_conv, se_conv, p_conv, sig_conv, tau_bc, se_rob, p_rob, sig_rob, p, q, h, n_eff, n_elec)

write_csv(t7_out, paste0(OUT, "T7_rdrobust_10DV.csv"))
cat(sprintf("Saved T7_rdrobust_10DV.csv (%d rows)\n", nrow(t7_out)))

# ═══════════════════════════════════════════════════════════════════════════
# T8: Firm-quarter dynamic (aggregate, 10 DVs)
# ═══════════════════════════════════════════════════════════════════════════
cat("\n========================================\n")
cat("T8: Firm-quarter dynamic (10 DVs)\n")
cat("========================================\n")

# Clamp event_q to [-3, 3] for T8
cur_clamped <- cur[cur$event_time_month >= -9 & cur$event_time_month <= 9, ]
cur_clamped$event_q <- pmax(-3, pmin(3, floor(cur_clamped$event_time_month / 3)))
cur_clamped$event_q <- factor(cur_clamped$event_q, levels=as.character(-3:3))

t8_rows <- list()
for (t_iter in THR) {
  cur_wlb <- cur_clamped[!is.na(cur_clamped$wlb), ]
  eids <- elig(cur_wlb, t_iter$type, t_iter$N)
  ne <- length(eids)
  if (ne < 20) next
  d <- cur_clamped[cur_clamped$election_id %in% eids, ]

  for (dv in DV10) {
    d_dv <- d[!is.na(d[[dv]]), ]

    # Aggregate to election×quarter
    agg <- d_dv |> group_by(election_id, gvkey, win, event_q) |>
      summarise(ybar = mean(.data[[dv]], na.rm=TRUE), margin = first(margin),
                n = n(), .groups = "drop") |>
      mutate(post = as.integer(as.integer(as.character(event_q)) >= 0),
             win_post = win * post)

    if (nrow(agg) < 30) next

    # Pre-trend test
    fd <- tryCatch(
      feols(ybar ~ i(event_q, win, ref='-1') + win + post:margin | gvkey + event_q,
            data=agg, weights=~n, cluster=~gvkey, warn=FALSE, notes=FALSE),
      error=function(e) NULL)
    pre <- tryCatch(fixest::wald(fd, "event_q::(-3|-2):win", print=FALSE)$p,
                    error=function(e) NA)

    # Pooled post
    fp2 <- tryCatch(
      feols(ybar ~ win + post + win_post + post:margin | gvkey + event_q,
            data=agg, weights=~n, cluster=~gvkey, warn=FALSE, notes=FALSE),
      error=function(e) NULL)

    v <- if(is.null(fp2) || !("win_post" %in% rownames(coeftable(fp2)))) {
      c(NA, NA, NA)
    } else {
      coeftable(fp2)["win_post", c("Estimate","Std. Error","Pr(>|t|)")]
    }

    t8_rows[[length(t8_rows)+1]] <- tibble(
      table="T8_fq_dynamic", filter=thr_label(t_iter), outcome=dv,
      pretrend_p=pre, pooled_est=v[1], pooled_se=v[2], pooled_p=v[3],
      n_events=ne)
  }
  cat(sprintf("  %s: %d elections\n", thr_label(t_iter), ne))
}

t8_out <- bind_rows(t8_rows)
write_csv(t8_out, paste0(OUT, "T8_firmquarter_dynamic_10DV.csv"))
cat(sprintf("Saved T8_firmquarter_dynamic_10DV.csv (%d rows)\n", nrow(t8_out)))

# ─── Quick summary ─────────────────────────────────────────────────────────
cat("\n=== T3 WLB across filters ===\n")
print(t3_out[t3_out$outcome=="wlb", c("filter","pretrend_p","t0_p","pooled_est","pooled_p","n_events")], n=20)

cat("\n=== T7 WLB across filters ===\n")
print(t7_out[t7_out$outcome=="wlb", c("filter","tau_conv","p_conv","tau_bc","p_rob","n_eff","n_elec")], n=20)

cat("\n=== T8 WLB across filters ===\n")
print(t8_out[t8_out$outcome=="wlb", c("filter","pretrend_p","pooled_est","pooled_p","n_events")], n=20)

cat("\nDone.\n")
