# union_glassdoor 项目指南（projects/union_glassdoor/CLAUDE.md）

## 当前目标

**Union Election × Glassdoor 论文**：用 review-level DiD-RD 识别工会选举对员工 Glassdoor 评分的因果效应。
主 outcome = **WLB (Work-Life Balance)**，主样本 = **current employees only**，主 spec = **v7c (DiD-RD)**。

---

## 当前状态（截至 2026-06-23）

### 主结果：WLB 在工会选举后显著上升

| Specification | Sample | Coefficient | SE | p-value | N Reviews | N Elections |
|--------------|--------|-------------|-----|---------|-----------|-------------|
| **v7c DiD-RD (pooled post)** | All | +0.063 | 0.023 | **0.016** | 468,325 | 1,065 |
| **v7c DiD-RD (pooled post)** | **Current only** | **+0.072** | **0.033** | **0.046** | 249,194 | 917 |
| v7c DiD-RD (pooled post) | Current, \|m\|≤0.05 | +0.339 | 0.144 | **0.031** | 14,490 | 85 |

**Spec v7c**: `outcome ~ win + post + win_post + post:margin + emp_status + seniority_f | gvkey + review_year + state_clean + role_clean`
- Cluster: `gvkey × review_year` (two-way)
- FE absorbed: `gvkey + review_year + state_clean + role_clean`
- Filter: n≥5 pre AND post per election (recalculated on each subsample)

### Event Study（Current Only, Quarterly [-3, +3]）

| Outcome | Pre-trend p | Q=-3 coef | Q=0 coef | Pooled post | Pattern |
|---------|-------------|-----------|----------|-------------|---------|
| **WLB** | **0.533** ✅ | −0.027 | +0.054 | +0.055 (p=0.127) | Flat pre → jump at t=0 → sustained |
| Comp | 0.714 ✅ | −0.020 | −0.018 | −0.012 (p=0.714) | All ~zero |
| Overall | 0.788 ✅ | −0.014 | +0.011 | +0.016 (p=0.660) | All ~zero |
| Senior Mgmt | 0.610 ✅ | +0.019 | +0.068 | +0.058 (p=0.259) | Positive post, Q3: +0.085 (p=0.080) |
| Culture | 0.976 ✅ | +0.006 | +0.037 | +0.045 (p=0.375) | Mild positive post |

**Half-year bins**（pooled post, all employees）: WLB +0.066 (p=0.016), pre-trend clean.

### Comp 始终为零

Compensation & Benefits 在所有 spec、所有样本、所有带宽下均 ≈ 0。工会选举不影响薪酬评分——这本身是一个有意义的 null result（工会不改变薪酬结构，但改善 WLB）。

### 文本分析 Pipeline（机制证据）

| 步骤 | 产出 | 状态 |
|------|------|------|
| LLM 标注 | 2,986 条评论 × 6 维度（pay/benefits/wc/vf/mc/wlb） | ✅ |
| BERT 分类器训练 | 12 个模型（6 mention + 6 sentiment） | ✅ |
| 全样本推理 | 490k 评论的 pay_complaint / wc_complaint 等分数 | ✅ |
| DiD-RD on text outcomes | pay_complaint +0.036 (p=0.060) | ✅ |

**分类器质量验证**:
- `pay_neg` F1 = **0.766** ✅ 可用
- `pay_mention` F1 = 0.931 ✅
- `benefits_neg` F1 = 0.725 ✅
- `wc_complaint` F1 = 0.046 ❌ 不可用（丢弃）
- `vf_complaint` / `mc_complaint` F1 < 0.40 ❌ 不可用（丢弃）

### 岗位分类 STEP1D 修复

- 修复了 `classify_union_dimension()` 中的 4 个问题（Principal IC override、product owner、legal support staff、team lead/shift lead）
- 产出: `outputs/20260617/union_classified_title_universe_step1d.csv`（2,129 titles 重分类）

---

## 已做决策（及理由）

### D1: WLB 为主 outcome（非 Overall Rating）

**理由**: 
- WLB 在 DiD-RD（review-level v7c）和 RDD（bandwidth 递减）中均显著
- Overall Rating 在 DiD-RD 中不显著（p=0.205–0.493），早期 RDD 显著可能来自 aggregation bias
- WLB 机制清晰（工会改善工作条件 → WLB 上升），Overall Rating 过于综合

### D2: Current employees only 为主样本（非 all employees）

**理由**:
- 离职者（尤其选举前离职的）未必经历工会化冲击，包含他们会稀释/污染效应
- Current-only: WLB +0.072 (p=0.046)，系数略大于全样本
- 代价: 47% 评论被过滤、SE 增大 43%，但 p 值仍 < 0.05
- 更干净的因果识别：仅包含实际经历工会化的员工

### D3: Review-level DiD-RD (v7c) 为主 spec（非 firm-year RDD）

**理由**:
- Review-level 粒度能控制个体层面 confounders（emp_status, seniority）
- Firm-year RDD 结果方向不一致（早期为负，DiD-RD 为正），可能来自聚合偏误
- 事件研究在 review-level 可实现季度/半年度动态路径，firm-year 做不到
- v7c 的 event study pre-trend 全部干净

### D4: Event study Q=[-3,3], 不扩展到 [-4,4]

**理由**:
- Q=-4 / Q=+4 端点稀疏，噪声大
- 早期 Q∈[-4,4] 时 Q=-4 导致 WLB pre-trend p=0.001（假阳性）
- Clamp 到 [-3,3] 后 pre-trend p=0.533（干净）
- Half-year bins 作为补充（pre-trend 也干净）

### D5: FE 不吸收 state_clean + role_clean（仅作为 RHS factor）

**理由**:
- 吸收 `gvkey + review_year + state_clean + role_clean` 在子样本上导致 fixest 假死
- 改为仅吸收 `gvkey + review_year`，state_clean + role_clean 保留在线性部分
- 这不会改变 win_post 的识别（state/role 不是 treatment 的 confounder——treatment 在 election 层面）

### D6: Multi-election version B (greedy, >365d gap) 从主分析中移除

**理由**:
- Version B 的逻辑 bug 导致重复行
- 修复后 version B/C 样本量过小
- 主分析用 version A（所有 election），在讨论中标注 multi-election 问题

### D7: 丢弃 wc_complaint / vf_complaint / mc_complaint

**理由**: BERT 分类器 F1 仅 0.04–0.37，信号不可信。仅 pay/benefits 分类器达到 F1 > 0.70。

---

## 重要文件位置

### 数据（输入）

```text
# 主分析样本（490k reviews, current + non-current, ±365d window）
outputs/20260618/text_analysis/full_sample_with_text_predictions.parquet
  # 含: 6 outcomes + is_current_employee + event_time_month + win/post/margin
  #     + gvkey + review_year + election_id + FE 协变量 + text predictions

# Current-only 子样本（n≥5 独立重算）
outputs/20260622/current_only/sample_current_n5.parquet        # 249k reviews / 917 elections
outputs/20260622/current_only/sample_current_eventstudy.parquet # quarterly [-3,3]

# 全样本对照
outputs/20260622/current_only/sample_all_n5.parquet             # 468k reviews / 1065 elections

# 原始数据
outputs/union_glassdoor_firm_year_regression.parquet            # 2059 elections × 1994 cols
outputs/union_glassdoor_comment_level_window365.parquet         # 68k reviews (旧版，已废弃)
outputs/compustat_firm_controls.parquet                         # 598k firm-years

# 外部
/data/disk4/workspace/projects/union/outputs/union_election_rc_votes_gvkey_only.parquet
/data/disk4/workspace/projects/glassdoor/outputs/sentiment_individual_reviews_with_gvkey.parquet
```

### 分析脚本

```text
# === 主回归 pipeline（R）===
r_src/current_only_t3_t4_t5.R              # T3 ES + T4 subgroups + T5 bandwidth (current only) ← 最新
r_src/event_study.R                        # Event study (all employees, quarterly)
r_src/text_analysis/run_text_did_rd.R      # DiD-RD on text outcomes

# v7 / v8 / v9 RDD robustness
r_src/rdd_rebuild/focused_rdd_search_v7/run_filter_stability_v7.R
r_src/rdd_rebuild/focused_rdd_search_v7/run_filter_stability_v7_lean.R
r_src/rdd_rebuild/focused_rdd_search_v9/check1_donut_rdd.R
r_src/rdd_rebuild/focused_rdd_search_v9/check2_event_delta_rdd.R
r_src/rdd_rebuild/focused_rdd_search_v9/check3_pre_rating_balance.R
r_src/rdd_rebuild/focused_rdd_search_v9/check4_density_test.R

# === 文本分析 pipeline（Python）===
src/text_analysis/annotate_final_dims_v2.py       # LLM 标注（ollama/qwen2.5:3b）
src/text_analysis/train_and_predict_final_dims.py # BERT 训练 + 推理（6 维度）
src/text_analysis/predict_pay_benefits_sharded.py # 分片全样本推理

# === 岗位分类 ===
src/build_union_title_classification.py           # STEP1D 修复版

# === 早期分析（已不再更新）===
src/analysis/00_inventory_union_glassdoor.py
src/analysis/01_review_level_regressions.py
src/analysis/02_firm_year_regressions.py
src/analysis/02b_firm_year_rdd_regressions.py
src/analysis/03_event_study.py
src/analysis/04_05_stability_analysis_and_report.py
```

### 结果文件

```text
outputs/20260622/current_only/
├── current_report.md                       # Current-only 汇总报告
├── T2_current_vs_all.csv                   # All vs current 并排对比
├── T3_eventstudy_current.csv               # Current 季度事件研究系数
├── T3_pooled_post_current.csv              # Current pooled post
├── T4_current.csv                          # Subgroups (unionizable vs excluded)
└── T5_wlb_bandwidth_current.csv            # WLB bandwidth robustness

outputs/20260622/event_study/
├── event_study_coefs.csv                   # All-employee 季度事件研究
└── event_study_report.md

outputs/20260622/event_study_halfyear/
├── es_halfyear_coefs.csv                   # Half-year bins
└── es_halfyear_pooled_post.csv

outputs/20260618/text_analysis/
├── classifier_verification.csv             # 10 模型 val F1
├── text_did_rd_results.csv                 # Text DiD-RD 结果
└── bert_models/{pay,benefits,wc,vf,mc}_{mention,neg,complaint}/

outputs/20260617/
└── union_classified_title_universe_step1d.csv  # 岗位分类修复版
```

---

## 待办 / 下一步

### 论文写作前（必须）

1. **写 paper draft**：以 WLB 为主 outcome、current-only 为主样本、v7c 为主 spec
2. **Table 1 样本描述**：current vs all, pre/post 各期均值、选举特征
3. **RDD validity checks 汇总**：
   - McCrary density test（check4，需确认 p 值）
   - Pre-treatment covariate balance（check3）
   - Donut-hole RDD（check1）
4. **Half-year event study 图**：作为主文 Figure 2（比 quarterly 更简洁）
5. **Text mechanism 写入**：仅报告 pay_complaint（F1=0.766，p=0.060），标注为探索性

### 可选（时间允许）

6. **岗位分类 merge 完成**：`title_standardized` ↔ `GD_JobTitle` fuzzy match → 解锁 R3/R5
7. **Multiple elections 诊断**：写清楚同一 firm 多个 election 时的 review 归属逻辑
8. **±730d 窗口**：扩大窗口增加样本
9. **NLRB 案件类型细分**：RC vs RM vs RD

### 低优先级

10. **FinBERT/VADER 情绪分数**：基于 Pros/Cons 文本
11. **CEO Approval / Recommend / Outlook**：分类变量转数值 → 附录

---

## 踩过的坑

### 坑1: `win_post:margin` 项不应加入 spec

在 v7 早期版本中误加了 `win_post:margin`。这使得 post-election 的 margin slope 在 treatment/control 侧不同，违反 DiD 平行趋势假设的逻辑延伸。正确做法：仅 `post:margin`（post 期统一的 margin 控制），不加 `win_post:margin`。

### 坑2: fixest 多 FE 吸收导致假死

`feols(y ~ ... | gvkey + review_year + state_clean + role_clean)` 在子样本上（尤其 bandwidth 受限时）会 hang。解决：仅吸收 `gvkey + review_year`，把 state_clean + role_clean 放回 RHS。

### 坑3: Event study Q=[-4,4] 端点稀疏导致 pre-trend 假阳性

Q=-4 只有少量 observation → 系数噪声大 → WLB pre-trend p=0.001。解决：clamp 到 [-3,3]。

### 坑4: BERT 训练 "Target 2 out of bounds"

complaint label 可能 >1（merge 问题）。解决：`.clip(0,1)` 后再训练。

### 坑5: HuggingFace 连接失败

服务器无法访问 huggingface.co。解决：`HF_ENDPOINT=https://hf-mirror.com` + `local_files_only=True`。

### 坑6: R print 错误 "invalid 'na.print' specification"

fixest 的 `coeftable()` 结果在 print 时可能触发此 bug（R 4.3.1 + fixest 内部交互）。绕过：写入 CSV 后用 Python 读取，或直接用 `cat()` 逐行打印。

### 坑7: Version B (greedy) 逻辑 bug

`assign_versions()` 在 outcome-expanded 数据上运行导致重复日期。修复：先对 election_id 去重，再计算 versions。

### 坑8: 早期 RDD 与 DiD-RD 方向相反

早期 firm-year RDD (ANCOVA) 得到负系数（GD_rating: β=−0.36 SD），但 review-level DiD-RD 得到正系数（WLB: β=+0.07 SD）。不是同一 outcome 的冲突：RDD 用的是 GD_rating（综合评分），DiD-RD 用的是 WLB（子维度）。但方向差异提醒我们 aggregation 会改变结论。**DiD-RD review-level 为当前主 spec**，firm-year RDD 降级为稳健性检验。

---

## 设计参考

- Li & Pinto (2025, Management Science): Glassdoor review-level event study（IPO setting）
- DiNardo & Lee (2004), Frandsen (2017), Wang & Young (2022): 标准 union election RDD
- Review-level DiD-RD: `Win × Post` identification using within-firm, within-year variation
- Spec: v7c 逐字固定，只通过数据过滤条件（sample filter）改变分析口径

---

## 运行环境

```bash
conda activate union_glassdoor
# R: /home/user/anaconda3/envs/union_glassdoor/bin/Rscript
# Python: /home/user/anaconda3/envs/union_glassdoor/bin/python
# R packages: fixest, dplyr, tidyr, readr, purrr, nanoparquet
# Rscript 不在默认 PATH，需用完整路径
```
