# Workspace 全局指南（/data/disk4/workspace/CLAUDE.md）

> 厦门大学服务器 wyaamd-server001 · 172.16.10.68 · AMD EPYC 9654（384核）/ 125GB 内存

## 硬性规则（必须遵守）

1. **禁止写入数据库盘**：/data/disk1 ~ disk5 下的 database/、data/、wrds 目录是原始数据，只读，任何脚本不得向其中写入或修改。
2. **disk2 仅剩约 300GB（96%）**：不要在 /data/disk2 产生任何新文件，读 optionmetrics/markit 时中间结果一律写到本 workspace。
3. **输出位置约定**：
   - 项目正式输出 → `projects/<项目>/outputs/YYYYMMDD/`（每日新建日期子文件夹）
   - 临时/中间文件 → `scratch/`（可随时清理）
   - 日志 → `logs/`
   - **以任务开始日期为准**：每个 prompt 执行时，根据该次任务的启动日期创建 `outputs/YYYYMMDD/` 子文件夹。即使任务跨天完成，所有输出仍放在启动日文件夹下，避免长任务结果分散到多个目录。
4. **Git 操作**：commit 可以自主进行（信息用英文，格式 `type: summary`），但 **push 前先向我确认**。
5. 删除任何大于 100MB 的文件前先向我确认。

## 运行环境

- 分析统一使用 conda 环境 `union_glassdoor`：
  `source /home/user/anaconda3/bin/activate union_glassdoor`
- 机器内存 125GB：单个 parquet 全量载入没问题（最大 2.5GB），但 1385 万行评论数据做 groupby/merge 时注意峰值内存，优先用 pyarrow/polars 或分块。
- 384 核：耗时任务可以放心并行（joblib / multiprocessing），但 n_jobs 建议 ≤ 64 起步。
- 长任务用 `nohup ... > logs/xxx.log 2>&1 &` 后台运行并告知我 PID。

## 项目总览

| 目录 | 内容 | Git 仓库 |
|------|------|----------|
| projects/union | NLRB 工会选举数据处理 | wangyouan/union |
| projects/glassdoor | Glassdoor 评论与情绪数据 | wangyouan/glassdoor-server |
| projects/union_glassdoor | 两者合并的回归/事件研究 | wangyouan/union_glassdoor |
| projects/BoardGenderDiversity | 董事会性别多样性（暂不接手） | wangyouan/BoardGenderDiversity |
| vibe_notes | 开发笔记，git 管理 | （私有仓库） |

## 关键数据速查

| 数据 | 路径（相对 workspace） | 规模 |
|------|------|------|
| 评论级情绪（主） | projects/glassdoor/outputs/sentiment_individual_reviews_with_gvkey.parquet | 1385万行 / 34,110 gvkey / 2008-2025 |
| 评论清洗版 | projects/glassdoor/outputs/glassdoor_review_level_clean.parquet | 同上 |
| 选举匹配（主） | projects/union/outputs/union_election_rc_votes_gvkey_only.parquet | 4,906行 / 1,635 gvkey / 1999-2026 |
| 回归面板 | projects/union_glassdoor/outputs/union_glassdoor_firm_year_regression.parquet | 2,059行 / 1,218 gvkey / 1994列 |
| 事件窗口 | projects/union_glassdoor/outputs/union_glassdoor_comment_level_window365.parquet | 68,201行 / 192 gvkey |
| Compustat 控制变量 | projects/union_glassdoor/outputs/compustat_firm_controls.parquet | 598,127行 / 46,732 gvkey |

WRDS 数据库入口（只读软链接）：`wrds_registry/high_confidence/`
（crsp、compustat、dealscan、ibes、optionmetrics、markit、boardex、sec_edgar）

## 工作习惯

- 每次会话开始可先跑健康检查：`df -h | grep '^/dev' && free -h | head -2`
- 重要发现、决策、踩坑记录追加到 `vibe_notes/` 对应笔记中
- 修改脚本前先 `git status` 确认工作区干净；新分析另写脚本，不直接覆盖已有流水线脚本

### vibe_notes prompt 执行规则

执行 `vibe_notes/` 下的 prompt 时，遵循三步流程：

1. **跑前 commit prompt**：`cd /data/disk4/workspace/vibe_notes && git add -A && git commit -m "notes: <prompt主题>"`
2. **执行 prompt**：按 prompt 要求运行分析或生成代码
3. **跑后 commit 代码**：`cd /data/disk4/workspace/projects/<项目> && git add <新文件> && git commit -m "<type>: <描述>"`

如果 prompt 没被 commit（`git status` 显示未跟踪），先 commit 再开始跑。

