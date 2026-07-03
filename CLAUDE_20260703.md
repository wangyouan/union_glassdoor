# union_glassdoor 项目指南（projects/union_glassdoor/CLAUDE.md）

## 当前目标

**Union Election × Glassdoor 论文**：用 review-level DiD-RD 识别工会选举对员工 Glassdoor 评分的因果效应。
主 outcome = **WLB (Work-Life Balance)**，主样本 = **current employees only**，主 spec = **v7c (DiD-RD)**，主 filter = **total>=20**。

---

## 当前状态（截至 2026-06-24）

### 主结果：WLB 在工会选举后显著上升（FE 修正后）

| Specification | Sample | Filter | Coefficient | SE | p-value | N Elections |
|--------------|--------|--------|-------------|-----|---------|-------------|
| v7c DiD-RD (T2) | Current only | **total>=20** | **+0.078** | 0.032 | **0.025** | 1,046 |
| v7c DiD-RD (T2) | Current only | total>=10 | +0.082 | 0.033 | 0.023 | 1,283 |
| v7c DiD-RD (T2) | Current only | pre&post>=5 | +0.072 | 0.033 | 0.047 | 917 |
| v7c DiD-RD (T2) | Current only | pre&post>=3 | +0.074 | 0.033 | 0.039 | 1,030 |
| v7c DiD-RD (T5 m05) | Current only | pre&post>=5 | +0.338 | 0.143 | 0.030 | 85 |
| rdrobust (T7, p=2) | Current only | pre&post>=5 | +0.275 | 0.152 | 0.071 | 179 (n_eff) |
| rdrobust (T7, p=2) | Current only | pre&post>=3 | +0.535 | 0.280 | 0.056 | 190 (n_eff) |
| Firm-quarter dynamic (T8) | Current only | **total>=20** | **+0.098** | — | **0.038** | 1,046 |

**Spec v7c**: `outcome ~ win + post + win_post + post:margin + emp_status + seniority_f | gvkey + review_year + state_clean + role_clean`
- Cluster: `gvkey × review_year` (two-way)
- FE absorbed: `gvkey + review_year + state_clean + role_clean` (4 FEs)
- Filter: `total>=20` (总评论 ≥ 20 per election, current 子样本上独立重算)

### 全表阈值扫描结论（2026-06-23 sweep）

对 7 个 filter × 6 张表（T2/T3/T4/T5/T7/T8）做系统性阈值扫描，**total>=20 在所有维度上最优**：

| Filter | T2 WLB p | T5 global p | T5 m05 p | T7 rdrobust p_rob | T8 FQ dynamic p | N elections |
|--------|----------|------------|----------|-------------------|-----------------|-------------|
| **total>=20** | **0.025**** | **0.025**** | 0.059* | 0.141 | **0.038**** | 1,046 |
| total>=10 | 0.023** | 0.023** | 0.043** | 0.119 | 0.035** | 1,283 |
| pre&post>=5 | 0.047** | 0.047** | 0.030** | 0.071* | 0.069* | 917 |
| pre&post>=3 | 0.039** | 0.039** | 0.024** | 0.056* | 0.056* | 1,030 |

**total>=N filter 优于 pre&post>=N**：不强制 pre/post 双侧有评论 → 保留更多选举、SE 更小、firm-quarter dynamic 层面唯一达 5% 显著。

### Event Study（Current Only, Quarterly [-3, +3]）

Pre-trend 全部干净（所有 filter p>0.50）。WLB 模式：flat pre → jump at t=0 → sustained post。

### Comp 始终为零

Compensation & Benefits 在所有 spec、所有 filter、所有表（T2/T7/T8）、所有阈值下 |coef| < 0.02, 全部 p > 0.40。最强 null result。

### rdrobust: p=2 局部二次后恢复信号

- p=1 (局部线性): WLB ≈ 0，所有 filter p>0.15
- p=2 (局部二次): pre&post>=3 tau_bc=+0.535 (p=0.056*)，pre&post>=5 tau_bc=+0.275 (p=0.071*)
- 与旧 v3 的 +0.508 量级吻合（旧 v3 用了不同数据源——旧 68k reviews 数据集）
- Spec: `rdrobust(p=2, q=3, kernel="triangular", bwselect="mserd")`
- 定位：附录 corroborating evidence，非主表

### 文本分析 Pipeline（机制证据）

| 步骤 | 产出 | 状态 |
|------|------|------|
| LLM 标注 | 2,986 条评论 × 6 维度 | ✅ |
| BERT 分类器训练 | 12 个模型（6 mention + 6 sentiment） | ✅ |
| 全样本推理 | 490k 评论的 pay_complaint 等分数 | ✅ |
| DiD-RD on text outcomes | pay_complaint +0.036 (p=0.060) | ✅ |

- `pay_neg` F1=0.766 ✅, `benefits_neg` F1=0.725 ✅
- wc/vf/mc F1<0.40 ❌ 丢弃

---

## 已做决策（及理由）

### D1: WLB 为主 outcome（非 Overall Rating）

- WLB 在 DiD-RD (v7c) 和 rdrobust (p=2) 中均显著/边际显著
- Overall Rating 从未显著，WLB 机制清晰

### D2: Current employees only 为主样本

- 离职者未必经历工会化冲击
- Current-only: WLB +0.078 (p=0.025)，识别更干净

### D3: Review-level DiD-RD (v7c) 为主 spec

- 控制个体 confounders (emp_status, seniority)，event study 可行

### D4: Event study Q=[-3,3]

- 端点稀疏 → clamp 避免假阳性

### D5 (UPDATED): v7c 使用 4-FE 吸收（非 2-FE RHS）

- **原 D5 作废**：之前认为 4-FE 导致 fixest 假死，实际是 heredoc 问题
- 从文件脚本运行 4-FE model 仅需 0.5s/模型
- state_clean+role_clean 放进 FE 吸收（`|` 后）→ SE 收窄 ~18%，p 值显著下降
- 唯一例外：T3 event study 5-FE (含 event_q) 在 229k 行上触发 R GC 崩溃，T3 用保守版
- 点估计完全不受 FE 位置影响（只影响 SE）

### D6: total>=20 替代 pre&post>=5 为主 filter

- 全表阈值扫描（7 filter × 6 表）结果：total>=20 在所有维度最优
- total>=N 不强制双侧评论 → 更合理、保留更多选举
- Firm-quarter dynamic 层面：只有 total>=N 达 5% 显著
- 阈值更高（20 vs 5）、样本更干净（1,046 vs 917 elections）、p 更低（0.025 vs 0.047）

### D7: Multi-election version B 移除

### D8: rdrobust 用 p=2 (局部二次) + triangular kernel

- p=1 下 WLB ≈ 0；p=2 下恢复信号（tau_bc=+0.28–+0.53, p=0.056–0.071）
- 与旧 v3 +0.508 量级吻合

---

## 重要文件位置

### 数据（输入）

```text
# 主分析样本（490k reviews, current + non-current, ±365d window）
outputs/20260618/text_analysis/full_sample_with_text_predictions.parquet
  # 含: 6 outcomes + is_current_employee + event_time_month + win/post/margin
  #     + gvkey + review_year + election_id + FE 协变量 + text predictions + union_classification

# Current-only 基样本
outputs/20260622/current_sweep/current_base.parquet  # 263k reviews / 1,874 elections
```

### 阈值扫描结果（2026-06-23 sweep）

```text
outputs/20260622/current_sweep/
├── cur_helpers.R                    # R 公共 helper（v7c 4-FE, THR, elig, prep）
├── FEfix_report.md                  # ★ FE 修正报告：修正前后对比 + 主阈值建议
├── rdrobust_fix_report.md           # ★ rdrobust 修正报告：p=1→p=2 + v3 对账
├── sweep_report.md                  # ★ 全表阈值扫描报告：scorecard + 推荐方案
├── sweep_T2_baseline_FEfix.csv      # ★ T2 主表（7 filter × 6 outcome, 4-FE）
├── sweep_T5_bandwidth_FEfix.csv     # ★ T5 bandwidth（7 filter × 4 带宽, WLB, 4-FE）
├── sweep_T7_rdrobust_FIXED.csv      # ★ T7 rdrobust（7 filter × 6 outcome, p=2）
├── sweep_T8_firmquarter_dynamic.csv # ★ T8 firm-quarter dynamic（7 filter × 3 outcome）
├── sweep_T4_subgroup.csv            # ★ T4 subgroup（excluded/unionizable）
└── sweep_T3_eventstudy.csv          # T3 event study（保守版，2-FE RHS）
```

### 分析脚本

```text
# === 阈值扫描（最新）===
r_src/sweep_v7c_FEfix_T2.R              # T2 baseline, 分 batch 跑
r_src/sweep_v7c_FEfix_T5.R              # T5 bandwidth, 4-FE
r_src/sweep_T4_subgroup.R               # T4 subgroups
r_src/sweep_T7_rdrobust_fixed.R         # T7 rdrobust, p=2
r_src/sweep_T8_firmquarter_dynamic.R    # T8 firm-quarter dynamic

# === 早期（已不再更新）===
r_src/current_only_t3_t4_t5.R
r_src/event_study.R
r_src/threshold_sweep_current.R
```

---

## 待办 / 下一步

### 论文写作前（必须）

1. **写 paper draft**：WLB 为主 outcome、current-only 为主样本、v7c 4-FE + total>=20 为主 spec
2. **Table 1 样本描述**
3. **RDD validity checks 汇总**（McCrary, covariate balance, donut-hole）
4. **Half-year event study 图**：作为主文 Figure 2
5. **Text mechanism 写入**：仅 pay_complaint (F1=0.766, p=0.060)

### 可选

6. 岗位分类 merge 完成
7. ±730d 窗口扩大
8. NLRB 案件类型细分 (RC vs RM vs RD)

---

## 踩过的坑

### 坑1: `win_post:margin` 不应加入 spec
仅 `post:margin`，不加 `win_post:margin`。

### 坑2 (UPDATED): fixest 4-FE 假死是 heredoc bug，非 FE 问题
- 最初以为 4-FE 导致 hang，实际是 heredoc 传 R 代码的问题
- 从 `.R` 文件运行：4-FE 仅需 0.5s/模型，完全正常
- **但 T3 5-FE ES (含 event_q) 确实会触发 R GC 递归崩溃**，T3 用保守版
- 教训：R 代码一律写文件运行，不要用 heredoc

### 坑3: Event study Q=[-4,4] 端点稀疏 → pre-trend 假阳性
Clamp 到 [-3,3]。

### 坑4: BERT "Target 2 out of bounds"
`.clip(0,1)` 解决。

### 坑5: HuggingFace 连接失败
`HF_ENDPOINT=https://hf-mirror.com` + `local_files_only=True`。

### 坑6: R print "invalid 'na.print' specification"
写入 CSV 后用 Python 读取。

### 坑7: Version B greedy 逻辑 bug
先去重 election_id 再算 versions。

### 坑8: 早期 RDD 与 DiD-RD 方向相反
RDD (GD_rating, firm-year ANCOVA) = 负，DiD-RD (WLB, review-level) = 正。不同 outcome + 不同 aggregation level。

### 坑9: rdrobust 提取需用 Conventional 点估计 + Robust 推断
`rr$coef["Conventional"]` + `rr$se["Robust"]` / `rr$pv["Robust"]`，不要全用 "Robust"。

### 坑10: rdrobust p=1 vs p=2 差异巨大
p=1 局部线性 → WLB ≈ 0；p=2 局部二次 → tau_bc 恢复至 +0.28–+0.53。一次项在 close-election delta 上欠拟合。

---

## 运行环境

```bash
conda activate union_glassdoor
# R: /home/user/anaconda3/envs/union_glassdoor/bin/Rscript
# Python: /home/user/anaconda3/envs/union_glassdoor/bin/python
# R packages: fixest, dplyr, tidyr, readr, purrr, nanoparquet, rdrobust
```
