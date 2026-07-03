# union_glassdoor 项目指南（projects/union_glassdoor/CLAUDE.md）

## 当前目标：寻找稳健、恒定、可复现的 Union Election × Glassdoor 结果

本阶段的核心任务是用现有数据系统评估：**union election 是否影响员工在 Glassdoor 上的评分/评价**，并找出最值得推进的一组 outcome、样本口径和回归规格。

所有分析都应能被重复运行，并输出清晰的诊断表、结果表和图。

---

## 当前进度（截至 2026-06-11）

### 已完成的分析

| 脚本 | 状态 | 产出 |
|------|------|------|
| `src/analysis/00_inventory_union_glassdoor.py` | ✅ | 变量盘点 |
| `src/analysis/01_review_level_regressions.py` | ✅ | Review-level DiD (R1–R4) |
| `src/analysis/02_firm_year_regressions.py` | ✅ | Firm-year DiD (FY1–FY4) |
| `src/analysis/02b_firm_year_rdd_regressions.py` | ✅ | Firm-year RDD (全局多项式) |
| `src/analysis/03_event_study.py` | ✅ | 事件研究图 + 系数 |
| `src/analysis/04_05_stability_analysis_and_report.py` | ✅ | 稳定性汇总 + 报告 |
| `src/analysis/build_sample_attrition_table.py` | ✅ | 样本归因诊断 |

### 初步结论（七个判断问题）

1. **哪个 outcome 最值得推进？** → **Overall Rating**。覆盖面最广（68k 条评论，192 个 gvkey），RDD5 (ANCOVA) 下显著（β=−0.36 SD, p=0.003）。Diversity & Inclusion 虽然在 review-level DiD 中显著，但仅集中在 26 家企业（Top 5 占 90%），不可作为主结果。

2. **Review-level 和 firm-year 方向是否一致？** → **是，均为负**。Review-level DiD: −0.01 到 −0.08 SD。Firm-year RDD (cubic): −0.27 到 −0.43 SD。方向一致但量级不同。

3. **Current vs non-current 哪个更强？** → **Former 略强但差异不大**（Diversity: former −0.091 vs current −0.067 SD）。主分析建议用 all employees。

4. **Job-category 分组是否有解释力？** → **尚未评估**。岗位分类文件的 merge key 需要额外处理（`title_standardized` ↔ `GD_JobTitle`）。

5. **Min review threshold 是否改变结论？** → **会**。Firm-year DiD 系数随门槛提高而衰减（+0.060 → −0.018），说明低门槛结果被小样本企业驱动。

6. **Event-study 是否支持因果解释？** → **部分支持**。整体来看系数小且 bouncing around zero。Diversity 显示 pre-positive → post-negative 模式但样本过于集中。GD_Management 有 pre-trend 问题。

7. **是否值得继续写成论文结果？** → **可以，但有条件**。推荐以 Overall Rating 为主 outcome，RDD (ANCOVA) 为主设定。需透明报告全部搜索范围、pre-trend 问题、以及 Diversity 的集中度问题。

---

## 关键数据发现

### 样本归因（全量 Glassdoor → 事件窗口）

| 步骤 | N Reviews | N gvkey | 占比 |
|------|-----------|---------|------|
| A. 全量 Glassdoor | 13,854,743 | 34,110 | 100% |
| B. 工会选举企业 | 1,918,990 | 798 | 13.9% |
| C. ±365d 事件窗口（fresh merge） | 490,815 | 607 | 3.5% |
| D. window365.parquet（现有文件） | 68,201 | 192 | 0.5% |

⚠️ **Fresh merge 找到 490k 条评论（607 gvkey），但现有 window365 只有 68k 条（192 gvkey）。** 丢失了 ~86% 的可匹配评论。原因待查——可能是 merge 逻辑过于严格或去重规则不同。**建议优先修复此问题**，修复后预期样本量可扩大 7 倍。

### Diversity & Inclusion 集中度 ⚠️

- 仅 26 家企业、24,324 条评论
- Top 5 企业占 90%，Top 10 占 97.1%
- **D&I 结果几乎完全由个别企业驱动，不可作为主结果**

---

## 数据路径

```text
项目根目录: /data/disk4/workspace/projects/union_glassdoor/

主要输入:
  outputs/union_glassdoor_firm_year_regression.parquet     # 2059 elections × 1994 cols
  outputs/union_glassdoor_comment_level_window365.parquet  # 68,201 reviews × 71 cols
  outputs/compustat_firm_controls.parquet                  # 598,127 firm-years × 52 cols
  /data/disk4/workspace/projects/union/outputs/union_election_rc_votes_gvkey_only.parquet
  /data/disk4/workspace/projects/glassdoor/outputs/sentiment_individual_reviews_with_gvkey.parquet  # 13.85M reviews

岗位分类:
  outputs/union_title_universe_normalized.csv      # ✓ 存在
  outputs/union_classified_title_universe.csv      # ✓ 存在（但 merge key 不匹配）
  outputs/union_classified_title_universe_final.csv # ✗ 不存在
```

---

## 分析脚本与运行命令

```bash
cd /data/disk4/workspace
conda activate union_glassdoor

# 0. 变量盘点
python projects/union_glassdoor/src/analysis/00_inventory_union_glassdoor.py

# 1. Review-level DiD（Li & Pinto 风格）
python projects/union_glassdoor/src/analysis/01_review_level_regressions.py

# 2a. Firm-year DiD（原设计）
python projects/union_glassdoor/src/analysis/02_firm_year_regressions.py

# 2b. Firm-year RDD（全局多项式，⚠️ 推荐取代 2a）
python projects/union_glassdoor/src/analysis/02b_firm_year_rdd_regressions.py

# 3. 事件研究
python projects/union_glassdoor/src/analysis/03_event_study.py

# 4+5. 稳定性汇总 + 报告
python projects/union_glassdoor/src/analysis/04_05_stability_analysis_and_report.py

# 补充：样本归因诊断
python projects/union_glassdoor/src/analysis/build_sample_attrition_table.py
```

### RDD 结果复现

当前 RDD 核心结果：

**RDD5 (ANCOVA-RDD, 线性全局多项式) — 推荐主设定：**
- GD_rating: β = −0.360 SD, HC2 se = 0.122, **p = 0.003**
- GD_senior_mgmt: β = −0.218 SD, p = 0.078
- GD_culture: β = −0.241 SD, p = 0.052

**RDD1 (基准 RDD, 三次全局多项式) — 作为稳健性检验：**
- GD_rating: β = −0.268 SD, se = 0.319, p = 0.401（量级相似但方差大）
- GD_comp_benefit: β = −0.433 SD, p = 0.170
- GD_senior_mgmt: β = −0.353 SD, p = 0.288

---

## 输出目录

```
outputs/analysis_stability/
├── variable_inventory_ratings.csv           # 评分变量盘点
├── review_level_variable_inventory.csv      # Review-level 全部变量
├── subsample_outcome_inventory.csv          # Firm-year 子群体映射
├── review_regression_results.csv            # R1–R5 (DiD) 结果
├── review_eventstudy_coefficients.csv       # 月度事件研究系数
├── firm_year_regression_results.csv         # FY1–FY4 (DiD) 结果
├── firm_year_rdd_results.csv               # RDD 全部结果 (75 specs)
├── firm_year_rdd_summary.csv               # RDD 最佳设定汇总
├── firm_year_rdd_poly_comparison.csv       # RDD 多项式阶数比较
├── firm_year_eventstudy_coefficients.csv    # 年度事件研究系数
├── stability_grid_results.csv              # 稳定性网格 (240 rows)
├── stability_summary_by_outcome.csv        # 稳定性评分
├── sample_attrition_table.csv              # 样本归因漏斗
├── sample_attrition_by_outcome.csv         # 各 outcome 样本量
├── sample_attrition_current_vs_all.csv     # 现任 vs 全部员工
├── sample_attrition_by_window.csv          # 窗口比较
├── diversity_sample_diagnostics.csv        # D&I 集中度诊断
├── union_glassdoor_stability_report.md     # 稳定性分析报告
├── sample_attrition_report.md              # 样本归因报告
└── figures/
    ├── review_eventstudy_*.png             # 各 outcome 月度事件图
    ├── firm_year_eventstudy_*.png          # 年度事件图
    ├── rdd_*.png                            # RDD 散点+拟合图
    ├── outcome_stability_heatmap.png
    ├── min_review_threshold_sensitivity.png
    └── current_vs_noncurrent_comparison.png
```

---

## 重要约束

1. 不要覆盖旧 pipeline 输出。
2. 不要修改 `/data/disk4/workspace/projects/glassdoor/` 或 `/data/disk4/workspace/projects/union/`。
3. 不要把探索性显著结果直接写成结论。必须报告所有 outcome 的搜索范围。
4. 不要只报告显著结果。必须保存完整 grid。
5. 如果某个 outcome 稳定但经济意义很小，也要明确说明。
6. 如果所有 outcome 都不稳定，也要如实报告，不要强行找故事。
7. 代码必须可重复运行。
8. 图表和表格必须写入 `outputs/analysis_stability/`。

---

## 已知问题 & 下一步

### 高优先级

1. **修复 merge 逻辑**：`build_sample_attrition_table.py` 的 fresh merge 找到 490k 条评论（607 gvkey），但现有 window365 仅 68k（192 gvkey）。检查 `build_union_glassdoor_comment_level.py` 的匹配逻辑，恢复丢失的 86% 评论。
2. **RDD 主设定确认**：当前 RDD5 (ANCOVA, p=1) 的 p=0.003 很强。建议做以下稳健性检验：
   - Donut-hole RDD（排除 margin=0 附近的 election）
   - 安慰剂 cutoff（在 margin=±0.1/±0.2 处重跑）
   - McCrary density test（检查 running variable 的连续性）

### 中优先级

3. **岗位分类 merge**：`union_classified_title_universe.csv` 中的 `title_standardized` 需要与 review-level 的 `GD_JobTitle` 做 fuzzy match，才能跑 R3（job title FE）和 R5（job category subsamples）。
4. **扩大事件窗口**：尝试 ±730 天以增加样本。
5. **Multiple elections per firm**：检查同一 firm 多个 election 时的 review 归属逻辑。

### 低优先级

6. **文本情绪变量**：全量 GD 有 `GD_Pros`/`GD_Cons` 文本列，可跑 FinBERT/VADER 生成情绪分数作为补充 outcome。
7. **CEO Approval / Recommend / Outlook**：当前为分类变量 (o/v/r/x)，review-level DiD 已转为数值并标准化，结果均不显著。可作为附录。

---

## 设计参考

- Li & Pinto (2025, Management Science): Glassdoor review-level event study（IPO setting）
- 标准 union election RDD: DiNardo & Lee (2004), Frandsen (2017), Wang & Young (2022)
- 本项目的 firm-year RDD 遵循标准设计：running variable = vote margin, cutoff = 0, treatment = win_union
- Review-level DiD 借鉴 Li & Pinto 的评论级粒度 + firm FE + year FE

---

## 推荐的下一步执行顺序

1. **修复 merge** → 扩大 sample（预期 490k reviews）
2. **RDD 稳健性检验** → donut-hole, placebo cutoff, McCrary test
3. **岗位分类 merge** → 解锁 R3/R5
4. **更新报告** → 基于修复后的 sample 重跑全部分析
5. **论文初稿** → Overall Rating 为主，RDD (ANCOVA) 为主要 specification
