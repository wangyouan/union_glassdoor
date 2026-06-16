#!/usr/bin/env Rscript
# Check 4: Margin density / bunching diagnostic (rddensity + ggplot2)
library(rddensity); library(nanoparquet); library(dplyr); library(readr); library(ggplot2)
setwd("/data/disk4/workspace/projects/union_glassdoor")

df <- nanoparquet::read_parquet("outputs/rdd_rebuild/focused_rdd_search_v7/rdd_sample_v7_enriched.parquet")

ev_margin <- df |> distinct(election_id, margin, win) |> filter(!is.na(margin))
cat(sprintf("N unique elections: %d\n", nrow(ev_margin)))
cat(sprintf("Margin range: [%.4f, %.4f]\n", min(ev_margin$margin), max(ev_margin$margin)))
cat(sprintf("|margin| < 0.01: %d elections\n", sum(abs(ev_margin$margin) < 0.01)))
cat(sprintf("|margin| < 0.05: %d elections\n", sum(abs(ev_margin$margin) < 0.05)))

# rddensity test
cat("\n=== rddensity test ===\n")
rdd_test <- rddensity(X=ev_margin$margin, c=0)
summary(rdd_test)

density_result <- data.frame(
  T_stat=rdd_test$test$t_jk, p_value=rdd_test$test$p_jk,
  N_left=rdd_test$N[1], N_right=rdd_test$N[2],
  h_left=rdd_test$h[1], h_right=rdd_test$h[2])
write_csv(density_result, "outputs/rdd_rebuild/focused_rdd_search_v9/density_test_results.csv")
cat(sprintf("\nrddensity: T=%.3f, p=%.3f\n", rdd_test$test$t_jk, rdd_test$test$p_jk))

# Density plot
p <- ggplot(ev_margin, aes(x=margin)) +
  geom_histogram(binwidth=0.01, fill="steelblue", color="white", alpha=0.8) +
  geom_vline(xintercept=0, color="red", linetype="dashed", linewidth=1) +
  geom_vline(xintercept=c(-0.05,0.05), color="orange", linetype="dotted", linewidth=0.8) +
  labs(title="Distribution of Union Election Vote Margins",
    subtitle=sprintf("rddensity: T=%.3f, p=%.3f | N=%d | Red=cutoff, Orange=+-0.05",
      rdd_test$test$t_jk, rdd_test$test$p_jk, nrow(ev_margin)),
    x="Vote margin (win share - 0.50)", y="N elections") +
  xlim(-0.5, 0.5) + theme_minimal(base_size=12)
ggsave("outputs/rdd_rebuild/focused_rdd_search_v9/margin_density_plot.png", p, width=8, height=5, dpi=150)

# Zoom
pz <- p + xlim(-0.10, 0.10) +
  geom_histogram(data=filter(ev_margin, abs(margin)<0.10), binwidth=0.005,
    fill="steelblue", color="white", alpha=0.8) +
  labs(title="Margin distribution — zoomed to +/-10%")
ggsave("outputs/rdd_rebuild/focused_rdd_search_v9/margin_density_zoom.png", pz, width=8, height=5, dpi=150)

cat("Plots saved.\n")

# Near-threshold bin table
near <- ev_margin |> mutate(bin=round(margin/0.005)*0.005) |>
  count(bin) |> filter(abs(bin) <= 0.05) |> arrange(bin)
cat("\nNear-threshold bins (|margin|<=0.05, bin=0.005):\n")
print(near, n=30)
cat(sprintf("\nExact zeros (margin==0): %d\n", sum(ev_margin$margin==0)))
cat(sprintf("margin in (0, 0.005]: %d\n", sum(ev_margin$margin>0 & ev_margin$margin<=0.005)))
cat(sprintf("margin in [-0.005, 0): %d\n", sum(ev_margin$margin>=-0.005 & ev_margin$margin<0)))
cat("Done.\n")
