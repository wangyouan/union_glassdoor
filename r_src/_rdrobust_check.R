suppressMessages({library(nanoparquet); library(dplyr)})
if(!requireNamespace("rdrobust",quietly=TRUE)) install.packages("rdrobust",repos="https://cloud.r-project.org")
library(rdrobust)
OUT <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260622/current_sweep/"
cur <- read_parquet(paste0(OUT,"current_base.parquet"))

# full sample WLB delta
agg <- cur |> group_by(election_id) |>
  summarise(pre=mean(wlb[post==0],na.rm=TRUE), postm=mean(wlb[post==1],na.rm=TRUE),
            margin=first(margin), .groups="drop") |>
  mutate(delta=postm-pre) |> filter(is.finite(delta), is.finite(margin))

cat("Agg rows:", nrow(agg), "\n")
cat("margin range:", range(agg$margin), "\n")
cat("delta range:", range(agg$delta, na.rm=TRUE), "\n")

rr <- rdrobust(y=agg$delta, x=agg$margin, c=0, kernel="triangular", bwselect="mserd")

cat("\ncoef rownames:", rownames(rr$coef), "\n")
cat("se   rownames:", rownames(rr$se), "\n")
cat("pv   rownames:", rownames(rr$pv), "\n")
cat("\n--- full summary ---\n")
print(summary(rr))
cat("\n--- coef matrix ---\n")
print(rr$coef)
cat("\n--- se matrix ---\n")
print(rr$se)
cat("\n--- pv matrix ---\n")
print(rr$pv)
cat("\n--- bws ---\n")
print(rr$bws)
cat("\n--- N_h ---\n")
print(rr$N_h)
