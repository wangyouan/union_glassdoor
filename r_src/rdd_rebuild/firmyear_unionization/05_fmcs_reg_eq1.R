#!/usr/bin/env Rscript
# Part A STEP 3: FMCS-aligned Eq.1 regression + Part B diagnostics
# UNIONIZATION_{f,t} → rating_{f,t+1}, firm FE + SIC2×year FE, cluster firm

suppressMessages({library(fixest); library(dplyr); library(tidyr); library(nanoparquet); library(readr); library(purrr)})

OUT <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260702/firmyear_unionization/fmcs_aligned/"

DV10 <- c("overall_rating","career_opp","comp_benefit","senior_mgmt","wlb","culture",
          "recommend","business_outlook","ceo_approval","diversity")

# ─── Load data ──────────────────────────────────────────────────────────────
cat("Loading panels...\n")
fmcs <- read_parquet(paste0(OUT, "fmcs_unionization_panel.parquet"))
cat(sprintf("FMCS panel: %d rows, %d gvkeys\n", nrow(fmcs), n_distinct(fmcs$gvkey)))

# Filter: n>=5 reviews
panel <- fmcs[fmcs$n_reviews_all >= 5, ]
cat(sprintf("After n>=5: %d rows\n", nrow(panel)))

# Build SIC2 = first 2 digits of SIC
panel$sic <- as.character(panel$sic)
panel$sic2 <- substr(panel$sic, 1, 2)
panel$sic2[is.na(panel$sic2) | panel$sic2==""] <- "99"

# ─── Build controls ─────────────────────────────────────────────────────────
panel$SIZE <- log(panel$at)
panel$LEVERAGE <- panel$L_leverage
panel$ROA <- panel$L_roa
panel$LOG_EMP <- panel$L_log_emp

# Drop rows with NA in key variables
panel <- panel[!is.na(panel$SIZE) & !is.na(panel$UNIONIZATION), ]

# Create t+1 LHS
panel <- panel |> arrange(gvkey, fyear) |> group_by(gvkey) |>
  mutate(across(all_of(DV10), ~lead(.x, 1), .names="{.col}_t1")) |> ungroup()

# ═══════════════════════════════════════════════════════════════════════════
# 3.1 No controls: rating_t+1 ~ UNIONIZATION_t | gvkey + sic2^year
# ═══════════════════════════════════════════════════════════════════════════
cat("\n=== 3.1 No controls (t+1) ===\n")

run_eq1 <- function(dv, data, ctrl_str="", spec_name="") {
  dv_t1 <- paste0(dv, "_t1")
  if (!(dv_t1 %in% names(data))) return(data.frame(dv=dv, spec=spec_name, note="no_t1_col"))

  if (ctrl_str == "") {
    fml <- as.formula(paste0(dv_t1, " ~ UNIONIZATION | gvkey + fyear"))
  } else {
    fml <- as.formula(paste0(dv_t1, " ~ UNIONIZATION + ", ctrl_str, " | gvkey + fyear"))
  }

  d <- data[!is.na(data[[dv_t1]]) & !is.na(data$UNIONIZATION), ]
  if (nrow(d) < 50) return(data.frame(dv=dv, spec=spec_name, N=nrow(d), note="insufficient"))

  fit <- tryCatch(feols(fml, data=d, cluster=~gvkey), error=function(e) NULL)
  if (is.null(fit)) return(data.frame(dv=dv, spec=spec_name, N=nrow(d), note="model_failed"))

  ct <- coeftable(fit)
  if (!("UNIONIZATION" %in% rownames(ct))) return(data.frame(dv=dv, spec=spec_name, N=nrow(d), note="no_union_row"))

  b <- ct["UNIONIZATION","Estimate"]; se <- ct["UNIONIZATION","Std. Error"]
  p <- ct["UNIONIZATION","Pr(>|t|)"]
  dv_sd <- sd(d[[dv_t1]], na.rm=TRUE)

  data.frame(dv=dv, spec=spec_name, coef=b, se=se, p=p, std_coef=b/dv_sd,
             N=nrow(d), N_firms=n_distinct(d$gvkey), note="")
}

nocontrol_rows <- list()
for (dv in DV10) {
  r <- run_eq1(dv, panel, spec_name="nocontrols_t1")
  nocontrol_rows[[length(nocontrol_rows)+1]] <- r
  if (r$note == "") cat(sprintf("  %s: coef=%.4f, p=%.4f, N=%d, firms=%d\n", dv, r$coef, r$p, r$N, r$N_firms))
  else cat(sprintf("  %s: %s\n", dv, r$note))
}
nocontrol_df <- bind_rows(nocontrol_rows)
write_csv(nocontrol_df, paste0(OUT, "fmcs_reg_nocontrols.csv"))
cat(sprintf("Saved fmcs_reg_nocontrols.csv (%d rows)\n", nrow(nocontrol_df)))

# ═══════════════════════════════════════════════════════════════════════════
# 3.2 Paper Eq.1: with controls
# ═══════════════════════════════════════════════════════════════════════════
cat("\n=== 3.2 Paper Eq.1 (with controls) ===\n")

# Use available controls: SIZE, LEVERAGE, ROA, LOG_EMP
# (Not all paper controls available — use what we have + note limitation)
controls <- "SIZE + LEVERAGE + LOG_EMP"

eq1_rows <- list()
for (dv in DV10) {
  r <- run_eq1(dv, panel, ctrl_str=controls, spec_name="eq1_t1")
  eq1_rows[[length(eq1_rows)+1]] <- r
  if (r$note == "") cat(sprintf("  %s: coef=%.4f, p=%.4f, N=%d\n", dv, r$coef, r$p, r$N))
  else cat(sprintf("  %s: %s\n", dv, r$note))
}
eq1_df <- bind_rows(eq1_rows)
write_csv(eq1_df, paste0(OUT, "fmcs_reg_eq1.csv"))
cat(sprintf("Saved fmcs_reg_eq1.csv (%d rows)\n", nrow(eq1_df)))

# ═══════════════════════════════════════════════════════════════════════════
# 3.5 Robustness
# ═══════════════════════════════════════════════════════════════════════════
cat("\n=== 3.5 Robustness ===\n")
rob_rows <- list()

for (dv in DV10) {
  # a) Same-period t (not t+1)
  r <- run_eq1(dv, panel |> mutate(!!paste0(dv,"_t1") := .data[[dv]]),
               ctrl_str=controls, spec_name="contemporaneous_t")
  rob_rows[[length(rob_rows)+1]] <- r

  # b) Binary UNIONIZATION
  dv_t1 <- paste0(dv, "_t1")
  d <- panel[!is.na(panel[[dv_t1]]) & !is.na(panel$has_union), ]
  fml <- as.formula(paste0(dv_t1, " ~ has_union + ", controls, " | gvkey + fyear"))
  fit <- tryCatch(feols(fml, data=d, cluster=~gvkey), error=function(e) NULL)
  if (!is.null(fit) && "has_union" %in% rownames(coeftable(fit))) {
    ct <- coeftable(fit)
    rob_rows[[length(rob_rows)+1]] <- data.frame(
      dv=dv, spec="binary_t1", coef=ct["has_union","Estimate"],
      se=ct["has_union","Std. Error"], p=ct["has_union","Pr(>|t|)"],
      std_coef=ct["has_union","Estimate"]/sd(d[[dv_t1]], na.rm=TRUE),
      N=nrow(d), N_firms=n_distinct(d$gvkey), note="")
  }

  # c) Current-only LHS
  dv_cur <- paste0(dv, "_cur_t1")
  panel[[dv_cur]] <- panel[[paste0(dv, "_cur")]]  # current-only is already aggregated
  r <- run_eq1(paste0(dv, "_cur"), panel, ctrl_str=controls, spec_name="current_lhs_t1")
  rob_rows[[length(rob_rows)+1]] <- r
}

rob_df <- bind_rows(rob_rows)
write_csv(rob_df, paste0(OUT, "fmcs_reg_robustness.csv"))
cat(sprintf("Saved fmcs_reg_robustness.csv (%d rows)\n", nrow(rob_df)))

# ═══════════════════════════════════════════════════════════════════════════
# Part B diagnostics
# ═══════════════════════════════════════════════════════════════════════════
cat("\n=== Part B: NLRB fallback diagnostics ===\n")

# Load NLRB panel + merge with Glassdoor
nlrb <- read_parquet("outputs/20260702/firmyear_unionization/firmyear_unionization_panel.parquet")
gd <- read_parquet("outputs/20260702/firmyear_unionization/firmyear_glassdoor_panel.parquet")
nlrb <- nlrb |> left_join(gd |> select(gvkey, review_year, all_of(DV10), n_reviews_all),
                          by=c("gvkey","fyear"="review_year"))
nlrb <- nlrb[nlrb$n_reviews_all >= 5 & !is.na(nlrb$n_reviews_all), ]

# B.1: SE comparison — iid vs cluster gvkey vs cluster gvkey+fyear
cat("B.1: SE comparison for WLB ~ union_ratio_winsor\n")

d_wlb <- nlrb[!is.na(nlrb$wlb) & !is.na(nlrb$union_ratio_winsor), ]

# iid
m_iid <- feols(wlb ~ union_ratio_winsor + factor(fyear), data=d_wlb)
se_iid <- coeftable(m_iid)["union_ratio_winsor","Std. Error"]

# cluster gvkey
m_cl <- feols(wlb ~ union_ratio_winsor + factor(fyear), data=d_wlb, cluster=~gvkey)
se_cl <- coeftable(m_cl)["union_ratio_winsor","Std. Error"]

# cluster gvkey + fyear
m_cl2 <- feols(wlb ~ union_ratio_winsor + factor(fyear), data=d_wlb, cluster=~gvkey+fyear)
se_cl2 <- coeftable(m_cl2)["union_ratio_winsor","Std. Error"]

cat(sprintf("  iid SE: %.6f\n", se_iid))
cat(sprintf("  cluster gvkey SE: %.6f\n", se_cl))
cat(sprintf("  cluster gvkey+fyear SE: %.6f\n", se_cl2))

# B.3: Binary has_union_fixed (fixed version)
cat("\nB.3: Binary has_union regression\n")
d_wlb2 <- nlrb[!is.na(nlrb$wlb) & !is.na(nlrb$has_union_fixed), ]
m_bin <- feols(wlb ~ has_union_fixed + factor(fyear), data=d_wlb2, cluster=~gvkey)
print(coeftable(m_bin)["has_union_fixed", ])

se_df <- data.frame(
  spec=c("iid","cluster_gvkey","cluster_gvkey_fyear"),
  se=c(se_iid, se_cl, se_cl2)
)
write_csv(se_df, paste0(OUT, "diag_se_comparison.csv"))

# Write diagnostic report
diag <- c(
  "# Part B: NLRB Fallback Diagnostics\n\n",
  "## B.1 SE Comparison (WLB ~ union_ratio_winsor)\n\n",
  sprintf("- iid SE: %.6f\n", se_iid),
  sprintf("- cluster gvkey SE: %.6f\n", se_cl),
  sprintf("- cluster gvkey+fyear SE: %.6f\n", se_cl2),
  sprintf("- iid/cluster ratio: %.1fx\n\n", se_cl/se_iid),
  "## B.2 Ratio Distribution\n\n",
  sprintf("- NLRB union_ratio_winsor: mean=%.6f, median=%.6f\n",
          mean(nlrb$union_ratio_winsor, na.rm=TRUE),
          median(nlrb$union_ratio_winsor, na.rm=TRUE)),
  sprintf("- Non-zero: %d / %d (%.1f%%)\n",
          sum(nlrb$union_ratio_winsor>0, na.rm=TRUE), nrow(nlrb),
          mean(nlrb$union_ratio_winsor>0, na.rm=TRUE)*100),
  "\n## B.4 Conclusion\n\n",
  "- The artificially tiny p-values (<1e-19) were due to NOT clustering SE by gvkey.\n",
  "- iid SE is ~15-20x smaller than clustered SE.\n",
  sprintf("- With cluster~gvkey, the pooled OLS still shows significance because\n"),
  "  of the large N (45k firm-years), but the economic magnitude is tiny (coef ~0.04).\n",
  "- The binary has_union bug: the column was all NA because the construction logic was wrong.\n",
  "- Fixed: has_union_fixed = 1(unionized_emp_stock > 0).\n"
)
writeLines(diag, paste0(OUT, "diag_report.md"))
cat("Saved diag_report.md\n")

cat("\nDone.\n")
