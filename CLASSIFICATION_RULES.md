# Union Glassdoor — Title Classification Rules

> 基于 `src/build_union_title_translation_map.py` + `src/build_union_title_classification.py`，整理自 2026 年 5 月版本的 STEP1C 最终版。

---

## 目录

1. [总体流程](#1-总体流程)
2. [维度一：非英语标题翻译](#2-维度一非英语标题翻译)
3. [维度二：工会谈判单元归属分类](#3-维度二工会谈判单元归属分类)
4. [维度三：组织资本分类](#4-维度三组织资本分类)
5. [STEP1C 修订：规则变更摘要](#5-step1c-修订规则变更摘要)
6. [产出变量一览](#6-产出变量一览)
7. [附录：关键词速查表](#7-附录关键词速查表)

---

## 1. 总体流程

```
Glassdoor 标准化标题（612,422 unique titles，来自 glassdoor 项目）
    │
    ▼
[Step A] 非英语标题翻译 → title_canonical_en（葡萄牙语/西语/法语/德语 → 英语）
    │
    ▼
[Step B] 标题标准化（同样的 normalize_text 处理：小写、去重音、去特殊字符）
    │
    ▼
[Step C] 工会谈判单元归属分类 → union_classification ∈ {likely_unionizable, likely_excluded, ambiguous}
    │    + 组织资本分类 → oc_likely / oc_management / oc_technical_engineering / oc_creative_product
    │
    ▼
[Step D] STEP1C 修订 → 收紧 unionizable 判定，弱监管岗降为 ambiguous
    │
    ▼
union_classified_title_universe.csv（产出到 union_glassdoor/outputs/）
```

**当前数据规模**（STEP1C 之后）：

| 分类 | 标题数 | 占比 | 评论加权占比 |
|------|--------|------|------------|
| `likely_unionizable` | 48,188 | 7.9% | 13.8% |
| `likely_excluded` | 39,196 | 6.4% | 4.7% |
| `ambiguous` | 525,038 | 85.7% | 81.5% |
| **合计** | **612,422** | 100% | 100% |

**OC 维度**：125,928 个标题被标记为 OC-likely（20.6%，评论加权 28.3%）。

---

## 2. 维度一：非英语标题翻译

**脚本**：`src/build_union_title_translation_map.py`  
**输入**：Glassdoor 项目的 `job_title_standardized_universe.csv`  
**输出**：`union_title_translation_map.csv` + `union_title_universe_normalized.csv`

### 2.1 语言检测

基于硬编码的四种语言信号集：

| 语言 | 信号词数量 | 示例 |
|------|-----------|------|
| Portuguese | 28 个 | `gerente`, `analista`, `vendedor`, `engenheiro`, `estagiario` |
| Spanish | 12 个 | `practicante`, `cajero`, `vendedor`, `enfermera`, `ingeniero` |
| French | 15 个 | `stagiaire`, `caissier`, `vendeur`, `technicien`, `serveur` |
| German | 1 个 | `werkstudent` |

检测逻辑：统计每种语言的信号词命中数，取命中最多的语言。若全部为零且标题仅含 ASCII → 标记为 `english_or_unknown`。

### 2.2 翻译策略（两级）

#### 第一级：精确短语映射（EXACT_PHRASE_MAP，~200 条规则）

完整短语精确匹配 → 直接替换为英文。示例：

| 源标题 | 翻译 |
|--------|------|
| `gerente comercial` | `sales manager` |
| `analista de sistemas senior` | `senior systems analyst` |
| `jovem aprendiz` | `apprentice` |
| `chef de projet` | `project manager` |
| `operateur de production` | `production operator` |
| `recursos humanos` | `human resources` |
| `coordinador de recursos humanos` | `human resources coordinator` |

#### 第二级：Token 级映射（TOKEN_MAP，~50 条规则）

短语未命中时，逐 token 替换：

| 源 token | 英文 | 源 token | 英文 |
|----------|------|----------|------|
| `gerente` | `manager` | `coordenador` | `coordinator` |
| `analista` | `analyst` | `vendedor` | `salesperson` |
| `cajero` / `caissier` | `cashier` | `atendente` | `attendant` |
| `operador` | `operator` | `motorista` | `driver` |
| `tecnico` / `technicien` | `technician` | `engenheiro` / `ingenieur` / `ingeniero` | `engineer` |
| `desenvolvedor` | `developer` | `enfermeiro` / `enfermera` | `nurse` |
| `estagiario` / `stagiaire` / `becario` | `intern` | `auxiliar` / `assistente` | `assistant` |
| `producao` | `production` | `logistica` | `logistics` |

**停用词**（翻译时跳过）：`de`, `do`, `da`, `del`, `di`, `du`, `des`, `la`, `le`, `el`, `y`, `e`

**修饰词**（翻译后重新排序到前面）：`senior`, `sr`, `junior`, `jr`, `pleno`, `trainee`, `lead`, `principal`

### 2.3 置信度评级

| 条件 | 置信度 |
|------|--------|
| 精确短语匹配 + 标题变化 | `high` |
| Token 级映射 + 标题变化 | `medium` |
| 有非英语信号但无映射 | `low` |
| 已是英语/无变化 | `high`（标记为 `already_english_or_unchanged`） |
| 空标题 | `low` |

---

## 3. 维度二：工会谈判单元归属分类

**脚本**：`src/build_union_title_classification.py`  
**分类标签**：`likely_unionizable` / `likely_excluded` / `ambiguous`  
**设计原则**：保守的规则化分类，ambiguous 涵盖不明确的情况而非强行分类。

### 3.1 决策流程

```
输入：标准化标题
    │
    ├─ 低信息/可疑标题？ → ambiguous（low confidence）
    │
    ├─ 命中 EXCLUDED_STRONG？ → likely_excluded（high confidence）
    │
    ├─ 主要为弱监管标题（WEAK_SUPERVISORY_AMBIGUOUS）？ → ambiguous（low confidence）
    │
    ├─ 无 unionizable + 无 excluded + 命中 AMBIGUOUS_ROLE_PHRASES？ → ambiguous（low confidence）
    │
    ├─ 同时命中 unionizable 和 excluded 关键词（冲突）：
    │   ├─ 含弱监管/冲突管理关键词（如 production manager） → ambiguous
    │   ├─ excluded ≥ 3 且 unionizable ≤ 1 → likely_excluded（medium confidence）
    │   └─ 其他冲突 → ambiguous
    │
    ├─ 命中 CONFLICT_MANAGER_AMBIGUOUS_PHRASES（如 production manager、customer service manager）→ ambiguous
    │
    ├─ 仅命中 excluded → likely_excluded（high confidence）
    │
    ├─ 仅命中 unionizable → likely_unionizable（high confidence）
    │
    └─ 均未命中 → ambiguous（low confidence）
```

### 3.2 候选工会化岗位关键词（UNIONIZABLE_KEYWORDS）

按行业分组：

#### 零售/门店（Retail / Store Workers）
```
retail sales associate, sales floor associate, seasonal sales associate,
part time sales associate, part-time sales associate, stock associate,
stocker, overnight stocker, shelf stocker, store associate, store clerk,
grocery clerk, retail assistant, shop assistant, beauty advisor,
key holder, keyholder, team member, cashier
```

#### 餐饮服务（Food Service）
```
barista, crew member, crew, sandwich artist, line cook, prep cook,
cook, dishwasher, server, waiter, waitress, host, hostess, bartender,
baker, food service worker, busser, kitchen staff
```

#### 物流/运输（Logistics / Transportation）
```
package handler, part time package handler, part-time package handler,
material handler, picker, packer, picker packer, order picker,
warehouse worker, warehouse associate, fulfillment associate,
sortation associate, dock worker, loader, forklift operator,
delivery driver, driver, truck driver, courier, ramp agent, postman
```

#### 制造业（Manufacturing）
```
machine operator, operator, production, production worker, assembler,
laborer, general laborer, welder, machinist, mechanic, maintenance worker,
maintenance technician, field technician, installer, electrician
```

#### 医疗辅助（Healthcare Support）
```
medical assistant, certified nursing assistant, nursing assistant,
cna, caregiver, home health aide, patient care technician, phlebotomist
```

#### 航空/酒店（Aviation / Hospitality）
```
housekeeper, room attendant, front desk agent, front desk receptionist,
receptionist, concierge
```

#### 银行一线（Banking）
```
bank teller, teller
```

#### 一线/通用客服（Frontline / Generalist）
```
security guard, security officer, customer service representative,
customer service associate, customer service agent, customer care representative,
technical support representative, call center representative,
contact center representative, service representative, clerk,
janitor, cleaner
```

#### 多语言（Spanish / Portuguese / French）
```
vendedor, cajero, operador, operador de caixa, operador de producao,
operador de maquinas, motorista, recepcionista, auxiliar de producao,
auxiliar de logistica, tecnico,
atendente, operador, motorista, tecnico,
caissier, caissiere, technicien
```

### 3.3 排除岗位关键词（EXCLUDED_STRONG）

**一级排除**（触发即 `likely_excluded`，high confidence）：

| 类别 | 关键词 |
|------|--------|
| **C-Suite / 高管** | `ceo`, `cfo`, `coo`, `cto`, `cio`, `chief`, `vice president`, `vp`, `head of`, `managing director`, `general manager` |
| **总监** | `director`, `regional manager`, `district manager` |
| **法律** | `attorney`, `lawyer`, `legal`, `counsel` |
| **人事/劳资关系** | `human resources`, `labor relations`, `employee relations` |
| **战略/企业** | `strategy`, `corporate development` |
| **创始人/所有者** | `founder`, `owner`, `partner`, `principal` |

**二级排除**（额外关键词，不一定达到 STRONG 级别）：
```
human resources, hr, people operations, people partner, hrbp,
recruiter, recruiting, talent acquisition, employee relations,
labor relations, industrial relations, compensation,
benefits manager, payroll manager,
strategy, corporate development, business strategy,
management consultant, internal consultant, transformation,
strategic initiatives, corporate planning,
founder, co founder, owner, partner, principal
```

### 3.4 冲突处理规则

以下标题同时命中 unionizable 和 excluded 时 → **ambiguous**（弱监管/模糊管理语境）：

**弱监管/领导力（WEAK_SUPERVISORY_AMBIGUOUS）**：
```
assistant manager, shift supervisor, shift leader, team lead, team leader, lead
```

**冲突管理短语（CONFLICT_MANAGER_AMBIGUOUS_PHRASES）**：
```
production manager, customer service manager, service delivery manager, delivery manager
```

**通用模糊角色（AMBIGUOUS_ROLE_PHRASES）**：
```
sales associate, sales, sales representative, sales consultant, sales assistant,
sales advisor, inside sales, outside sales representative, sales specialist,
business development, business development representative, account executive,
account manager, consultant, analyst, associate, specialist, assistant,
coordinator, advisor, agent, officer, representative
```

**数量主导规则**：如果 excluded 命中 ≥ 3 且 unionizable 命中 ≤ 1 → 判为 `likely_excluded`（medium confidence）。

### 3.5 低信息/可疑标题 → 强制 ambiguous

继承自 Glassdoor 标准化的标记：
- `low_information_title == 1`
- `is_suspicious == 1`

额外低信息 token（Union 项目特有）：
```
"", anonymous, employee, unemployed, none, na, n/a, unknown, test,
other, non, dy, spring, material, student
```

---

## 4. 维度三：组织资本分类

**定义**（与 Li & Pinto 2025 一致）：OC-likely 角色主要涉及：
1. 人员/产品/运营管理
2. 技术/工程工作
3. 创意/产品设计
4. 研究/科学工作

### 4.1 分类逻辑

```
输入：标准化标题
    │
    ├─ 低信息/可疑？ → oc_likely=0, oc_ambiguous=1
    │
    ├─ 在 OC_AMBIGUOUS_KWS 中（analyst, associate, consultant, specialist, assistant, support）？
    │   ├─ 含 data/business/product/research/science 上下文 → oc_likely=1, oc_technical=1
    │   └─ 否则 → oc_likely=0, oc_ambiguous=1
    │
    ├─ 命中管理关键词？ → oc_management=1
    ├─ 命中技术/工程关键词？ → oc_technical_engineering=1
    ├─ 命中创意/设计关键词？ → oc_creative_product=1
    │
    ├─ 任何 OC 子维度命中？ → oc_likely=1
    │
    ├─ oc_likely=0 且命中非 OC 一线词（warehouse, driver, cashier, barista, cook 等）？
    │   → oc_likely=0, oc_ambiguous=0  （明确标记为非 OC）
    │
    └─ 默认 → oc_likely=0, oc_ambiguous=1
```

### 4.2 OC 子维度关键词

#### 管理层（OC_MANAGEMENT_KWS）
```
manager, director, vice president, vp, president, chief, head of,
supervisor, team lead, shift lead, operations manager,
project manager, program manager, product manager
```

#### 技术/工程（OC_TECH_KWS）
```
software engineer, senior software engineer, software developer,
senior software developer, developer, programmer,
data scientist, data engineer, machine learning engineer,
systems engineer, systems analyst, system administrator,
systems administrator, database administrator, network engineer,
cybersecurity analyst, information security analyst,
it analyst, it specialist, information technology,
business analyst, data analyst, financial analyst,
quantitative analyst, research analyst, research scientist,
scientist, researcher, economist, statistician,
technical writer, devops,
desenvolvedor, engenheiro, ingenieur, ingeniero,
business intelligence, analytics,
research scientist
```

#### 产品/设计（OC_PRODUCT_DESIGN_KWS）
```
product manager, product designer, ux designer, ui designer,
graphic designer, designer
```

### 4.3 OC_AMBIGUOUS_KWS（独立判断）

```
analyst, associate, consultant, specialist, assistant, support
```

这些词单独出现时无法判断是否 OC，需要上下文。如果标题中包含 `data` / `business` / `product` / `research` / `science` 任一词 → 判为 OC-likely（技术/工程方向）。

### 4.4 非 OC 一线排除词

命中以下任一且未命中 OC 关键词 → 标记为非 OC（`oc_ambiguous=0`）：

```
warehouse, package handler, dock worker, loader, forklift,
material handler, driver, delivery, cashier, barista,
call center, contact center, cleaner, janitor, housekeeper,
room attendant, server, cook, dishwasher, kitchen,
food service, receptionist, customer service, customer support,
store associate, retail associate, bank teller,
flight attendant, front desk
```

---

## 5. STEP1C 修订：规则变更摘要

**脚本**：`build_union_title_classification.py`（STEP1C 为代码内的最新版本）  
**前置版本**：`outputs/union_classified_title_universe_pre_step1c.csv`

### 5.1 变更内容

| # | 变更 | 原因 |
|---|------|------|
| 1 | 移除通用 sales 和 technical/developer 的 broad unionizable 信号 | 避免过度分类：sales associate、account manager 等回到 ambiguous |
| 2 | 新增细颗粒度一线岗位规则 | 针对零售、餐饮、物流、技工、医疗辅助、酒店、客服的精确短语规则 |
| 3 | 新增多语言一线规则 | 葡萄牙语/西班牙语技术专业术语不再默认为 unionizable |
| 4 | 弱监管标题保持 ambiguous | assistant manager、shift supervisor、team lead 等不再强行判 excluded |
| 5 | 冲突处理改为保守 | 不再按 token 数量判定；strong excluded 可胜出，否则冲突保持 ambiguous |

### 5.2 变更影响

| 指标 | 修订前 | 修订后 |
|------|--------|--------|
| `likely_unionizable` 标题数 | 69,018 | **48,188** (-30%) |
| `likely_excluded` 标题数 | 112,554 | **39,196** (-65%) |
| `ambiguous` 标题数 | 430,850 | **525,038** (+22%) |
| 评论加权：unionizable | 17.7% | 13.8% |
| 评论加权：excluded | 20.8% | 4.7% |
| 评论加权：ambiguous | 61.6% | 81.5% |

### 5.3 重点重新分类案例

**Ambiguous → Likely Unionizable**（一线岗位被精确识别）：
`cook`, `hostess`, `stocker`, `bartender`, `beauty advisor`, `security officer`,
`team member`, `overnight stocker`, `retail assistant`, `teller`, `key holder`,
`customer care representative`, `technical support representative`, `shop assistant`,
`postman`, `baker`, `electrician`, `cna`, `concierge`, `ramp agent`, `courier`, `phlebotomist`

**Likely Excluded → Ambiguous**（管理/监督岗位回到模糊状态）：
`manager`, `project manager`, `assistant manager`, `account manager`, `store manager`,
`account executive`, `senior manager`, `supervisor`, `product manager`, `sales manager`,
`operations manager`, `team lead`, `program manager`, `recruiter`, `shift supervisor`

---

## 6. 产出变量一览

### 6.1 翻译映射文件（`union_title_translation_map.csv`）

| 变量 | 含义 |
|------|------|
| `title_standardized` | Glassdoor 标准化标题 |
| `title_normalized` | 本地标准化后的标题 |
| `title_canonical_en` | 翻译/规范化后的英文标题 |
| `detected_language_or_origin` | 检测到的语言：portuguese / spanish / french / german / english_or_unknown |
| `translation_source` | 翻译来源：dictionary / untranslated / already_english_or_unchanged |
| `translation_confidence` | 翻译置信度：high / medium / low |
| `translation_note` | 翻译备注 |

### 6.2 分类文件（`union_classified_title_universe.csv`）

| 变量组 | 变量 | 含义 |
|--------|------|------|
| **Union** | `union_likely_unionizable` | 0/1 — 是否可能属于谈判单元 |
| | `union_likely_excluded` | 0/1 — 是否被排除（管理/法律/人事等）|
| | `union_ambiguous` | 0/1 — 是否无法判断 |
| | `union_classification` | likely_unionizable / likely_excluded / ambiguous |
| | `union_confidence` | high / medium / low |
| | `union_reason` | 分类理由 |
| **OC** | `oc_likely` | 0/1 — 是否组织资本角色 |
| | `oc_management` | 0/1 — 管理层 |
| | `oc_technical_engineering` | 0/1 — 技术/工程 |
| | `oc_creative_product` | 0/1 — 创意/产品设计 |
| | `oc_ambiguous` | 0/1 — OC 维度无法判断 |
| | `oc_reason` | OC 分类理由 |

### 6.3 辅助诊断文件

| 文件 | 内容 |
|------|------|
| `union_title_classification_diagnostics.json` | 各分类数量/份额/加权份额/Top 30 |
| `union_ambiguous_title_examples.csv` | Top 500 ambiguous 标题（按评论量）|
| `union_low_information_title_examples.csv` | 低信息/可疑标题 |
| `union_top_titles_by_reviews.csv` | Top 1000 标题的完整分类 |
| `union_title_classification_protocol.md` | 自动生成的协议摘要 |
| `STEP1C_RECLASSIFIED_EXAMPLES.csv` | STEP1C 重新分类的案例 |
| `STEP1C_SUMMARY.md` | STEP1C 变更摘要 |

---

## 7. 附录：关键词速查表

### A. Unionizable 关键词（按行业）

| 行业 | 典型关键词 |
|------|-----------|
| 零售 | `retail sales associate`, `cashier`, `stocker`, `store associate`, `key holder`, `beauty advisor`, `team member` |
| 餐饮 | `barista`, `server`, `cook`, `line cook`, `dishwasher`, `bartender`, `crew member`, `sandwich artist` |
| 物流 | `package handler`, `warehouse worker`, `delivery driver`, `truck driver`, `forklift operator`, `order picker` |
| 制造 | `machine operator`, `assembler`, `welder`, `machinist`, `mechanic`, `electrician` |
| 医疗辅助 | `cna`, `medical assistant`, `caregiver`, `home health aide`, `phlebotomist` |
| 酒店 | `housekeeper`, `room attendant`, `front desk agent`, `concierge` |
| 银行 | `bank teller`, `teller` |
| 客服 | `customer service representative`, `call center representative`, `technical support representative` |
| 安保/清洁 | `security guard`, `security officer`, `janitor`, `cleaner` |

### B. Excluded 关键词（Strong 级别 — 触发即排除）

| 类别 | 关键词 |
|------|--------|
| 高管 | `ceo`, `cfo`, `coo`, `cto`, `cio`, `chief`, `president`, `vice president`, `vp`, `head of`, `managing director`, `general manager` |
| 总监/区域 | `director`, `regional manager`, `district manager` |
| 法律 | `attorney`, `lawyer`, `legal`, `counsel`, `general counsel`, `corporate counsel` |
| 人事/劳资 | `human resources`, `labor relations`, `employee relations` |
| 战略 | `strategy`, `corporate development` |
| 创始人/合伙人 | `founder`, `co founder`, `owner`, `partner`, `principal` |

### C. 保持 Ambiguous 的关键词

| 类别 | 关键词 |
|------|--------|
| 弱监管 | `assistant manager`, `shift supervisor`, `shift leader`, `team lead`, `team leader`, `lead` |
| 冲突管理 | `production manager`, `customer service manager`, `service delivery manager`, `delivery manager` |
| 通用角色 | `analyst`, `associate`, `consultant`, `specialist`, `assistant`, `coordinator`, `advisor`, `agent`, `officer`, `representative` |
| 通用销售 | `sales associate`, `sales`, `sales representative`, `sales consultant`, `sales assistant`, `sales advisor`, `inside sales`, `sales specialist` |
| 通用客户 | `account executive`, `account manager`, `business development`, `business development representative` |

### D. OC 排除一线词（non-OC frontline）

```
warehouse, package handler, dock worker, loader, forklift, material handler,
driver, delivery, cashier, barista, call center, contact center, cleaner,
janitor, housekeeper, room attendant, server, cook, dishwasher, kitchen,
food service, receptionist, customer service, customer support,
store associate, retail associate, bank teller, flight attendant, front desk
```

---

*最后更新：2026-06-14*  
*基于脚本版本：`build_union_title_translation_map.py` + `build_union_title_classification.py`（STEP1C，2026 年 5 月版本）*  
*关联文档：`projects/glassdoor/CLASSIFICATION_RULES.md`（Glassdoor 项目的职能/层级/暴露分类）*
