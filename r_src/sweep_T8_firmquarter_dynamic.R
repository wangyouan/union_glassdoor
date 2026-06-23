OUT <- "/data/disk4/workspace/projects/union_glassdoor/outputs/20260622/current_sweep/"
source(paste0(OUT,"cur_helpers.R"))
fp <- paste0(OUT,"sweep_T8_firmquarter_dynamic.csv"); if(file.exists(fp)) quit(save="no")

cur <- read_parquet(paste0(OUT,"current_base.parquet")) |> filter(event_q >= -3, event_q <= 3)
message("loaded (q clamped): ", nrow(cur), " rows")

rows <- list()
for(t in THR){
  ids <- elig(cur, t$type, t$N)
  d <- cur |> filter(election_id %in% ids)
  ne <- n_distinct(d$election_id)
  message("  ", thr_label(t), ": ", nrow(d), " reviews, ", ne, " elections")
  if(ne < 20) next

  for(y in KEY){
    agg <- d |> group_by(election_id, gvkey, win, event_q) |>
      summarise(ybar = mean(.data[[y]], na.rm=TRUE), margin = first(margin), n = n(), .groups = "drop") |>
      mutate(event_q = factor(event_q, levels = as.character(-3:3)),
             post = as.integer(as.integer(as.character(event_q)) >= 0),
             win_post = win * post)

    # dynamic pre-trend + pooled post (weighted by election×quarter N)
    fd <- tryCatch(
      feols(ybar ~ i(event_q, win, ref='-1') + win + post:margin | gvkey + event_q,
            data = agg, weights = ~n, cluster = ~gvkey, warn = FALSE, notes = FALSE),
      error = function(e) NULL)

    pre <- tryCatch(fixest::wald(fd, "event_q::(-3|-2):win", print=FALSE)$p, error=function(e) NA)

    fp2 <- tryCatch(
      feols(ybar ~ win + post + win_post + post:margin | gvkey + event_q,
            data = agg, weights = ~n, cluster = ~gvkey, warn = FALSE, notes = FALSE),
      error = function(e) NULL)

    v <- if(is.null(fp2) || !("win_post" %in% rownames(coeftable(fp2)))) {
      c(NA, NA, NA)
    } else {
      coeftable(fp2)["win_post", c("Estimate", "Std. Error", "Pr(>|t|)")]
    }

    rows[[length(rows)+1]] <- tibble(table="T8_fq_dynamic", filter=thr_label(t), outcome=y,
      pretrend_p = pre, pooled_est = v[1], pooled_se = v[2], pooled_p = v[3], n_events = ne)
  }
}
out <- bind_rows(rows)
write_csv(out, fp)
message("saved ", fp, " (", nrow(out), " rows)")
