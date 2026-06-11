# union_glassdoor 项目指南（projects/union_glassdoor/CLAUDE.md）

## 当前目标：寻找稳健、恒定、可复现的 Union Election × Glassdoor 结果

本阶段的核心任务不是继续重构岗位分类，也不是一次性追求复杂机制，而是用现有数据系统评估：**union election 是否影响员工在 Glassdoor 上的评分/评价**，并找出最值得推进的一组 outcome、样本口径和回归规格。

请优先形成一组稳定、可解释、可复现的结果。所有分析都应能被重复运行，并输出清晰的诊断表、结果表和图。

---

## 研究重点

### 1. 不预设唯一主 outcome

不要只以 overall rating 为主，也不要只比较 work-life balance 和 senior management。

请系统尝试所有可用 Glassdoor rating / subrating outcomes，并从中判断哪一个子指标最值得继续推进。

至少包括以下类型的结果变量（实际变量名以数据为准）：

```text
overall rating
work-life balance
senior management
culture and values
career opportunities
compensation and benefits
CEO approval / recommend / outlook（如果数据中存在）
文本情绪变量 / pros-cons sentiment（如果数据中存在）
```

请先自动识别所有 rating/subrating/sentiment 相关变量，并输出：

```text
outputs/analysis_stability/variable_inventory_ratings.csv
```

字段包括：

```text
variable_name
variable_type  # rating / subrating / sentiment / text-count / other
nonmissing_n
nonmissing_share
mean
sd
min
p1
p25
median
p75
p99
max
```

### 2. 目标不是找显著性，而是找稳定模式

请比较每个 outcome 在不同样本、不同评论数门槛、不同固定效应、不同事件窗口下的结果是否方向一致、量级稳定、样本量足够、图形不违背 pre-trend。

最终报告需要明确判断：

```text
最值得推进的 outcome 是什么？
是否有稳定 negative / positive effect？
结果主要来自哪些样本或员工类型？
是否只是某个 specification 偶然显著？
是否值得继续写成论文结果？
```

---

## 数据路径

项目根目录：

```text
/data/disk4/workspace/projects/union_glassdoor/
```

主要输入文件：

```text
/data/disk4/workspace/projects/union_glassdoor/outputs/union_glassdoor_firm_year_regression.parquet
/data/disk4/workspace/projects/union_glassdoor/outputs/union_glassdoor_comment_level_window365.parquet
/data/disk4/workspace/projects/union_glassdoor/outputs/compustat_firm_controls.parquet
/data/disk4/workspace/projects/union/outputs/union_election_rc_votes_gvkey_only.parquet
/data/disk4/workspace/projects/glassdoor/outputs/sentiment_individual_reviews_with_gvkey.parquet
```

岗位/员工类型相关文件：

```text
/data/disk4/workspace/projects/union_glassdoor/outputs/union_title_universe_normalized.csv
/data/disk4/workspace/projects/union_glassdoor/outputs/union_classified_title_universe.csv
/data/disk4/workspace/projects/union_glassdoor/outputs/union_classified_title_universe_final.csv  # 如果存在则优先使用
```

如果 final title classification 不存在，请不要临时大规模重构岗位分类。先使用现有分类变量，并在报告中说明使用了哪个版本。

---

## 输出目录与代码组织

所有新增分析脚本放在：

```text
/data/disk4/workspace/projects/union_glassdoor/src/analysis/
```

所有新增结果放在：

```text
/data/disk4/workspace/projects/union_glassdoor/outputs/analysis_stability/
```

图放在：

```text
/data/disk4/workspace/projects/union_glassdoor/outputs/analysis_stability/figures/
```

请不要覆盖旧的 pipeline 输出。旧数据只读。

建议新增脚本：

```text
src/analysis/00_inventory_union_glassdoor.py
src/analysis/01_review_level_regressions.py
src/analysis/02_firm_year_regressions.py
src/analysis/03_event_study_review_level.py
src/analysis/04_threshold_and_subsample_stability.py
src/analysis/05_make_stability_report.py
```

如果你认为用更少脚本更高效，可以合并，但必须保证逻辑清楚、可复现。

---

## 参考文献设计：Li and Pinto (2025, Management Science)

请参考附件中的 Li and Pinto (2025) 的 Glassdoor review-level 设计，但不要机械照搬 IPO 研究。

可借鉴的核心点：

1. 使用 review-level regression，而不是只做 firm-year aggregation。
2. 充分利用 Glassdoor 的评论级粒度。
3. 对 rating 进行标准化，构造 `Sd_Outcome = (Outcome - mean) / sd`。
4. 控制 firm fixed effects、year fixed effects，并在可行时加入 state / job title / employment length fixed effects。
5. 可对所有 Glassdoor component ratings 做系统比较，而不是只看 overall rating。
6. 可对 current employees 与 former/non-current employees 分开检验。

Union setting 下不一定有 state、employment length、gender、education 等变量。如果不存在，不要强行构造；请先 inventory 并报告可用性。

---

## 核心分析一：Review-level regressions

### 1. 数据准备

优先使用：

```text
union_glassdoor_comment_level_window365.parquet
```

如果该文件变量不够，请从全量 Glassdoor review 文件和 union election 文件重新构造 review-level sample，但不要覆盖旧文件。新文件可输出为：

```text
outputs/analysis_stability/review_level_analysis_sample.parquet
```

请识别并输出以下变量是否存在：

```text
gvkey
review_id
review_date / date / year / month
company / employer
job_title / title_standardized / title_canonical_en
current_employee / employee_status
rating variables
sentiment variables
union election date
union win indicator
vote share / margin
relative day / relative month / relative year to election
post-election indicator
state / location
employment length / tenure
job classification variables
```

输出：

```text
outputs/analysis_stability/review_level_variable_inventory.csv
```

### 2. Review-level baseline specifications

对每个 outcome 循环估计以下模型。

#### Model R1: Simple before-after within event window

```text
Sd_Outcome_{i,f,t} = beta * PostElection_{f,t} + firm FE + year FE + error
```

#### Model R2: Add calendar time and event controls

```text
Sd_Outcome_{i,f,t} = beta * PostElection_{f,t} + firm FE + year-month FE + error
```

如果 year-month FE 太重或样本不足，则使用 year FE + month FE。

#### Model R3: Add job title fixed effects

```text
Sd_Outcome_{i,f,t} = beta * PostElection_{f,t} + firm FE + year FE + job_title FE + error
```

job title FE 可以优先使用：

```text
title_canonical_en
```

如果不存在，则使用最稳定的 title field。

#### Model R4: Add employee-status subsamples

分样本估计：

```text
current employees only
former / non-current employees only
all employees
```

如果 current/non-current 变量不存在，请报告。

#### Model R5: Job-category subsamples

如果岗位分类变量存在，请分组估计：

```text
likely_unionizable
likely_excluded
ambiguous
OC likely
non-OC
management / technical / frontline / customer-service / operations 等已有分类
```

不要因为分类不完美而停止分析。目标是探索哪个员工组最有信号。

### 3. Review-level event-study

构造以 union election date 为中心的事件时间变量：

```text
relative_month = floor((review_date - election_date) / 30)
```

主要窗口：

```text
[-12, +12] months
[-6, +6] months
[-3, +3] months
```

至少输出：

```text
Post bins: -12 to +12 monthly bins, omitted bin = -1 or 0
Alternative bins: <=-6, -5 to -3, -2 to -1, 0 to 1, 2 to 3, 4 to 6, >=7
```

对每个核心 outcome 画事件图。不要只画显著的 outcome；至少画所有主要 rating/subrating 的图。

输出：

```text
outputs/analysis_stability/review_eventstudy_coefficients.csv
outputs/analysis_stability/figures/review_eventstudy_<outcome>.png
```

### 4. 标准误

优先使用：

```text
cluster by gvkey
```

如果可行，另做：

```text
cluster by election/event id
cluster by gvkey and year（如果工具支持）
```

报告不同 clustering 下结果是否改变。

---

## 核心分析二：Firm-year regressions

继续使用：

```text
union_glassdoor_firm_year_regression.parquet
```

但不要只看一个 outcome。请自动识别所有 outcome，并按以下维度系统跑表。

### 1. 评论数门槛

对每个 outcome，分别用以下 firm-year 最小评论数门槛：

```text
no threshold
>= 1 review
>= 3 reviews
>= 5 reviews
>= 10 reviews
```

如果不同 outcome 有不同 `n_reviews` 变量，请自动识别对应评论数变量；否则使用总评论数。

### 2. Firm-year baseline models

#### Model FY1

```text
Outcome_{f,t} = beta * PostElection_{f,t} + firm FE + year FE + error
```

#### Model FY2

```text
Outcome_{f,t} = beta * UnionWin_{f} * PostElection_{f,t} + firm FE + year FE + error
```

如果 treatment 已经定义为 union win，则用现有变量，但请解释。

#### Model FY3

加入控制变量：

```text
size
leverage
ROA
market-to-book
cash
sales growth
industry-year controls
```

只使用实际存在的控制变量。不要因为缺控制变量而中断。

#### Model FY4

尝试事件时间：

```text
relative_year bins around election
```

输出 event-study 图。

### 3. outcome families

请尝试：

```text
overall rating
work-life balance
senior management
culture and values
career opportunities
compensation and benefits
text sentiment / pros / cons / net sentiment
review volume / review length / pros length / cons length
```

如果 firm-year 文件里已经有 current / non-current 版本、job-category 版本、unionizable/excluded/OC 版本，请全部纳入搜索，但最终报告只保留最有价值、最稳定的若干结果。

---

## 核心分析三：current / non-current 与员工类型

项目里可能已有 current 与 non-current employee 的拆分，也可能已有根据职位分类生成的一堆子结果。请系统识别这些变量。

请创建变量 inventory：

```text
outputs/analysis_stability/subsample_outcome_inventory.csv
```

按变量名推断其所属维度：

```text
all reviews
current employee
former / non-current employee
likely_unionizable
likely_excluded
ambiguous
OC likely
management
technical / engineering
frontline / customer service / operations
```

如果已有多个 outcome suffix/prefix，请建立映射表，不要手工随意挑选。

分析时至少比较：

1. all employees
2. current employees
3. non-current/former employees
4. likely_unionizable employees
5. likely_excluded / management / OC employees
6. ambiguous employees（作为参考，不作为主结论）

目标是判断 union election 的影响是否更集中在某类员工评论中。

---

## 核心分析四：稳定性筛选表

请生成一个总结果表：

```text
outputs/analysis_stability/stability_grid_results.csv
```

每一行是一组 specification：

```text
analysis_level          # review-level / firm-year / firm-month if constructed
outcome
outcome_family
sample_group            # all/current/non-current/unionizable/excluded/OC/etc.
window                  # full, [-365,+365], [-180,+180], [-90,+90], etc.
min_reviews_threshold
model_id
fixed_effects
controls
cluster_level
N
N_firms
N_events
coef
se
t_stat
p_value
ci_low
ci_high
mean_y
sd_y
economic_magnitude_sd
sign
significant_10
significant_5
significant_1
pretrend_flag
notes
```

`economic_magnitude_sd` 应尽量表示 coefficient 占 outcome 标准差的比例。如果 outcome 已标准化，则 coefficient 本身就是标准差单位。

### 稳定性评分

请构造一个简单的 stability score，不用于 p-hacking，只用于组织结果：

```text
same_sign_count
significant_count
median_coef
coef_iqr
sample_size_median
pretrend_pass_count
stability_score
```

按 outcome × sample_group 汇总，输出：

```text
outputs/analysis_stability/stability_summary_by_outcome.csv
```

---

## 核心分析五：图和报告

输出以下图：

```text
figures/outcome_stability_heatmap.png
figures/top_outcome_eventstudy_review_level.png
figures/top_outcome_eventstudy_firm_year.png
figures/min_review_threshold_sensitivity_<top_outcome>.png
figures/current_vs_noncurrent_<top_outcome>.png
figures/job_group_heterogeneity_<top_outcome>.png
```

如果某些图无法生成，请在报告中说明原因。

最终报告：

```text
outputs/analysis_stability/union_glassdoor_stability_report.md
```

报告结构：

1. Executive summary：是否发现值得推进的结果。
2. Data and sample：review-level 和 firm-year 样本量、覆盖年份、gvkey、events。
3. Outcome inventory：所有 rating/subrating/sentiment 变量。
4. Main stability grid：哪些 outcome 最稳定。
5. Review-level evidence：参考 Li and Pinto 的评论级回归结果。
6. Firm-year evidence：不同 min review threshold 下的结果。
7. Current vs non-current：是否集中在现任员工或离职员工。
8. Job-category evidence：unionizable/excluded/OC/management/frontline 等结果。
9. Event-study/pretrend：是否存在明显 pretrend 或动态模式。
10. Recommended baseline：建议下一步主表使用哪一个 outcome、哪一个样本、哪一个 specification。
11. Caveats：样本量、匹配、评论选择偏误、岗位分类误差、multiple testing 风险。
12. Next steps：是否需要重构数据、扩大窗口、人工复核岗位分类、或放弃某条方向。

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

## 推荐执行顺序

第一步：inventory

```bash
python src/analysis/00_inventory_union_glassdoor.py
```

第二步：review-level regressions

```bash
python src/analysis/01_review_level_regressions.py
```

第三步：firm-year regressions

```bash
python src/analysis/02_firm_year_regressions.py
```

第四步：event-study and heterogeneity

```bash
python src/analysis/03_event_study_review_level.py
python src/analysis/04_threshold_and_subsample_stability.py
```

第五步：report

```bash
python src/analysis/05_make_stability_report.py
```

如果执行时间太长，请先用小样本 dry run，然后完整运行。

---

## 当前阶段的判断标准

Claude 完成后，请不要只给一个“跑完了”的回复。请明确回答：

```text
1. 哪个 outcome 最值得推进？
2. review-level 和 firm-year 是否方向一致？
3. current vs non-current 哪个更强？
4. job-category 分组是否有解释力？
5. min review threshold 是否改变结论？
6. event-study 是否支持因果解释？
7. 是否值得继续写成论文结果？
```
