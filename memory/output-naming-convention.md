---
name: output-naming-convention
description: Outputs go under outputs/YYYYMMDD/ not nested subdirectories
metadata:
  type: project
---

All output files go to `projects/union_glassdoor/outputs/YYYYMMDD/` where YYYYMMDD is the task start date. Do NOT nest outputs under descriptive subdirectories like `outputs/rdd_rebuild/current_former_bargaining_unit/20260624/`. Scripts can live in nested directories under `r_src/`, but outputs always at `outputs/YYYYMMDD/`.

**Why:** Consistent with workspace CLAUDE.md convention: "项目正式输出 → projects/<项目>/outputs/YYYYMMDD/（每日新建日期子文件夹）"

**How to apply:** Always set `OUT <- "outputs/YYYYMMDD/"` in R scripts or `OUT = "outputs/YYYYMMDD"` in Python scripts. Use today's date for the folder name.
