# union_glassdoor 项目指南（projects/union_glassdoor/CLAUDE.md）

> 2026-07-03 由本地端（Cowork）整合重写：保留原版决策与踩坑记录，新增铁律与防偷懒惯例。**本文件优先级高于任务提示词中的相反表述；勿用 /init 覆盖；更新须经用户确认。**
> 分工：本目录只负责计算；解读、整理 Excel、与合作者（Amanda）沟通在本地端。任务提示词由本地端撰写。

## 当前目标

**Union Election × Glassdoor 论文**（Heitz & Wang）：
1. **主线**：review-level DiD-RD 识别工会选举对员工 Glassdoor 评分的因果效应。主 outcome = **WLB**，主样本 = **current employees only**，主 spec = **v7c 4-FE**，主 filter = **total≥10**（2026-07 与 total≥20 全面对比后 t10 略优；待 Amanda 最终确认，total≥20 作稳健性）。
2. **第二条线（2026-07 新增）**：firm-year unionization 面板（FMCS F-7，UNIONIZATION = Σ bargaining unit size / Σ establishment size per gvkey-year）× Glassdoor firm-year 评分，L1–L4 FE 阶梯。定位 = appendix/描述性补充。

---

## 一、铁律（违反 = 结果作废）

1. **所有回归表报告全部 10 个 DV**：`overall_rating, career_opp, comp_benefit, senior_mgmt, wlb, culture, recommend(1/0.5/0), business_outlook(±1/0), ceo_approval(±1/0), diversity`。不得只跑子集；估不了输出 NA 行 + 原因。每个 DV 的 filter 基于该 DV 非缺失重算。
2. **不得依显著性选规格、选样本、选变量版本**；提示词要求的版本全部输出。
3. v7c 中 `state_clean` / `role_clean` 必须放 fixest `|` 后 FE 块；**绝不加 `win_post:margin`**（坑 1）。
4. rdrobust 取数：`coef["Conventional"]` 点估计 + `se/pv["Robust"]` 推断（坑 9）；主用 p=2（D8）。
5. fixest 剔除变量（共线等）必须把原始信息写进输出，不得留无解释空行。
6. **复现基线**：任何 RDD 任务先在 current+total≥10 确认 **WLB ≈ +0.082 (p≈0.023)、Comp ≈ +0.005 (p≈0.870)**；对不上停下汇报。firm-year 任务基线：2005–2017 unionized 加权平均 ≈ 0.69。

## 二、防偷懒惯例（每个任务必须执行；2026-07 屡次返工后确立）

0. **清单开跑**（2026-07-04 新增）：**读 prompt 后先输出逐条检查清单**（每条 = 一个 STEP 的子任务 + 产出文件名 + 行数/断言），跑完一条勾一条。全部勾完才算完成。清单放在回复开头，执行过程中不再复述。
1. **自检收尾**：最后一步运行自检脚本——断言所有预期输出文件存在且非空、回归表行数 = DV 数 × 规格数、报告 md 不含 "(to be filled)"。**失败必须修复重跑，全绿才算完成。**
2. **报告由代码生成**：报告 md 用脚本从结果 CSV 机械拼接，禁止跑完后凭记忆手写；开跑前建骨架，每 STEP 完成立即回填。
3. **逐步落盘**：每 STEP 独立可单跑、写 checkpoint；数据构造任务必须输出**逐文件损耗表**（读入→映射→过滤各级行数），任何文件留存 0 行要打印表头 + 前 3 行原始数据。
4. **构造后先验分布再回归**：新面板先输出逐年计数/均值曲线并对照基准；**逐年数值完全恒定 = 年份广播 bug（坑 11），立即停**。

## 三、输出约定

- 产出统一放 `outputs/YYYYMMDD/<主题>/`（按实际执行日期）。
- 标准列名——ladder/controls：`model, dv, coef, se, pvalue, n_obs, n_firms, dropped`；robustness：`spec, dv, coef, se, pvalue, n_obs`；correlations：`outcome, pearson, spearman, mean_gt0, mean_eq0, n`。
- 失败/NA 行保留并写原因；汇报给数字与异常，不要只给路径。

---

## 当前状态（截至 2026-07-03）

- **主线已定稿**：v7c 4-FE，current+total≥10：WLB +0.082 (p=0.023)★★；±5% 带宽 +0.306 (p=0.043)；firm-quarter dynamic +0.112 (p=0.015)；event study pre-trend 干净；Comp 全线干净零；10 DV 已跑齐 T2/T3/T5/T7/T8。current vs former 正式检验无显著差异。
- **firm-year 线**：成品面板（2005–2017，cusip 桥接 97.2%）结果已交付——within-firm 基本零、唯 comp_benefit +0.041 (p=0.022)；横截面 comp 强正 / WLB·culture·mgmt 显著负（选择效应）。**diversity 与 2005–2017 面板零重叠不可估。**
- **进行中**：UNIONIZATION 扩展至 2026（F-7 月度文件）。第 5 轮失败（坑 11/12），修复任务 = 本地提示词 `2026-07-03-04-extension-fix-prompt.md`（第 6 轮，含逐文件损耗表 + 硬校验 + 自检）。
- 文本分析（BERT pay_complaint 等）已完成，作机制补充。

## 已做决策（保留，更新处标注）

- **D1**: WLB 为主 outcome（非 Overall）。
- **D2**: current employees only 为主样本。
- **D3**: review-level DiD-RD (v7c) 为主 spec。
- **D4**: event study clamp 到 Q=[-3,3]（端点稀疏假阳性，坑 3）。
- **D5**: v7c 用 4-FE 吸收（原"4-FE 假死"系 heredoc bug，坑 2；T3 5-FE 会 GC 崩溃，用保守版）。
- **D6（2026-07 更新）**: 主 filter = **total≥10**（全证据层对比 t10 略优：T2 p=0.0245 vs 0.0301、T8 p=0.0151 vs 0.0239、±5% p=0.043 vs 0.074、diversity 保 ★★）；total≥20 作稳健性。待 Amanda 确认后不再改。
- **D7**: multi-election version B 移除。
- **D8**: rdrobust 用 p=2 + triangular kernel（p=1 下 WLB≈0）。定位附录 corroborating。
- **D9（新增）**: firm-year UNIONIZATION 分母 = FMCS Establishment Size（作者确认），聚合 = Σ BUS / Σ EST per gvkey-year（加权平均），cap@1，无备案年=0；面板与 review-level DiD-RD 互补呈现（横截面选择效应 vs 边际因果），作 appendix。

## 重要文件位置

```text
# === RDD 主线 ===
outputs/rdd_rebuild/focused_rdd_search_v7/rdd_sample_v7_enriched.parquet   # 主样本 490k
outputs/20260618/text_analysis/full_sample_with_text_predictions.parquet   # 含 text predictions
outputs/20260622/current_sweep/current_base.parquet                        # current-only 263k
outputs/20260622/current_sweep/cur_helpers.R                               # v7c helper（沿用）
outputs/20260622/current_sweep/sweep_*.csv                                 # 阈值扫描权威结果

# === firm-year 线 ===
/data/disk5/data/union/union f7/                       # FMCS 原始 F-7（年度+月度至2026-04）+ unionized_rate_data.csv + 20211204 成品面板；路径含空格
/home/user/Database/compustat/                         # Compustat 全量两个 zip（gvkey/cusip/conm）
outputs/20260703/firmyear_finished_panel/              # ctat_id_table / ctat_controls / merged_panel_main / 回归结果
outputs/*/unionization_extension*/                     # 扩展轮产出（第 5 轮有 bug，勿直接复用其面板）

# === Glassdoor 全量 ===
/data/disk4/workspace/projects/glassdoor/outputs/glassdoor_review_level_clean.parquet  # 13.85M 行，只读
```

原始数据一律只读；中间产物放任务输出目录。

## 踩过的坑（历史 + 新增，勿重犯）

1. `win_post:margin` 不加（膨胀 p 值）。
2. fixest 4-FE"假死"= heredoc bug → **R 代码一律写 .R 文件运行**；T3 5-FE 会 GC 崩溃用保守版。
3. Event study 端点稀疏 → clamp [-3,3]。
4. BERT "Target 2 out of bounds" → `.clip(0,1)`。
5. HuggingFace 连不上 → `HF_ENDPOINT=https://hf-mirror.com` + `local_files_only=True`。
6. R print "invalid 'na.print'" → 写 CSV 后用 Python 读。
7. Version B greedy bug → 先去重 election_id。
8. 早期 firm-year RDD（GD_rating, ANCOVA）为负 vs review-level DiD-RD 为正——不同 outcome/聚合层，非矛盾；**引用早期数字前必须找到原始输出核实，不得凭记忆写进报告**。
9. rdrobust 提取：Conventional 点估计 + Robust 推断。
10. rdrobust p=1 vs p=2 差异巨大，主用 p=2。
11. **（2026-07 新增）年份广播**：构造面板时把跨年混合值广播到每一年（逐年均值恒定）——构造后必须先看逐年分布（惯例二.4）。
12. **（2026-07 新增）报告不填 / DV 漏跑 / 文件缺交**：第 3/4/5 轮反复发生——自检脚本收尾（惯例二.1），10 DV 全报（铁律 1）。
13. **（2026-07 新增）cusip→gvkey 桥接必须用 Compustat 全量**（`/home/user/Database/compustat/`，cusip 100% 覆盖）；用 NLRB 匹配文件当桥只有 30% 覆盖，样本残废。
14. **（2026-07 新增）diversity（D&I）评分 2020 年才上线**：与 2005–2017 unionization 零重叠——零重叠下模型报出的"显著系数"全是伪影，先查重叠数再跑。

## 运行环境

```bash
conda activate union_glassdoor
# R: /home/user/anaconda3/envs/union_glassdoor/bin/Rscript
# Python: /home/user/anaconda3/envs/union_glassdoor/bin/python
# R packages: fixest, dplyr, tidyr, readr, purrr, nanoparquet, rdrobust
# 纯本地计算，不调外部 API；长任务分片、写 checkpoint
```
