---
name: output-naming-convention
description: Outputs go under outputs/YYYYMMDD/, nesting allowed below date
metadata:
  type: project
---

Output files go under `projects/union_glassdoor/outputs/YYYYMMDD/` where YYYYMMDD is the task start date. Nesting under the date folder is fine (e.g., `outputs/20260624/current_former/`). The key rule: **date comes first**, not `outputs/some_project/YYYYMMDD/`.

**Why:** Consistent with workspace CLAUDE.md convention: "项目正式输出 → projects/<项目>/outputs/YYYYMMDD/（每日新建日期子文件夹）"

**How to apply:** Always use `outputs/YYYYMMDD/` as the root, optionally with a subdirectory for the specific analysis.
