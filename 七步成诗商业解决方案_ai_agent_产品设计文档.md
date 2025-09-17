# 七步成诗商业解决方案 AI Agent — 产品设计文档

## 概要
目标：将麦肯锡七步问题解决法（Define, Break down, Prioritize, Plan, Analyze, Synthesize, Communicate）映射为可编排、可观测、可落地的AI Agent系统。产出能够自动或半自动交付的商业解决方案包（问题定义、MECE问题树、优先级、分析计划、数据分析、结构化结论、PPT/视频/音频汇报、可执行行动清单）。

核心价值：保证方法论严谨性、结果可验证性、沟通可执行性。

范围：MVP 目标是对输入的商业问题生成完整可执行方案并提供可下载报告与PPT。后续扩展到自动化数据接入、A/B 执行跟踪和闭环。

---

## 设计原则
- 假设驱动。每一步以可验证假设为单元。
- MECE。问题树与分析项互斥且穷尽。
- 金字塔原理。结论先行，证据支持。
- 可解释与可审计。每个结论必须关联数据或人工核验记录。
- 最小惊喜原则。默认保守建议，风险与置信度并列给出。

---

## 目标用户与角色
- 提问者（Business Owner/PM）
- 分析师（Data Analyst）
- 执行者（Operations/PMO）
- 决策者（C-level）
- 系统管理员（Data Engineer, SRE）

---

## 典型用例（MVP 三场景）
1. SaaS: 新增客户渠道有效性，目标降低 CAC 20%。
2. 零售：门店利润率下降，目标恢复至历史基线。
3. 市场进入：是否进入 Country X，需 6 个月内完成可行性评估。

每个用例最终产出：问题定义、问题树、优先级、数据/实验计划、关键分析、三条可执行建议、PPT/视频/音频汇报、行动跟踪任务。

---

# 七步对齐到 Agent（职责、输入、输出）

每一步由一个或一组 Agent 负责。Orchestrator 控制执行顺序与失败回滚。

1. **Problem Framer Agent**
   - 职责：澄清问题，定义目标、范围、KPIs、时间窗、约束、初步假设。
   - 输入：raw_text, 上传文件（pdf,csv,xlsx）,元数据。
   - 输出：ProblemFrame JSON `{goal, scope, kpis, stakeholders, constraints, assumptions, desired_outcome}`。

2. **Issue Tree Agent**
   - 职责：基于 ProblemFrame 生成 MECE 问题树，每一叶子结点包含可验证假设。
   - 输出：IssueTree JSON（节点 id,title,hypotheses,required_data,priority_hint）和可视化 JSON 用于 d3 渲染。

3. **Prioritization Agent**
   - 职责：对叶子节点进行影响×可行性评分，输出推荐优先级与打分理由。
   - 方法：影响度(1-10),可行性(1-10),置信度,成本估算。

4. **Planner Agent**
   - 职责：为优先问题生成分析计划（数据需求、方法、模型、估时、负责人、里程碑）。
   - 输出：AnalysisPlan 列表。

5. **Data Analysis Agent**
   - 职责：执行分析计划（ETL、统计、回归、cohort、segmentation、causal inference 等），产出 artifacts（图表、模型、notebook、CSV）。
   - 运行环境：受控 Python sandbox + SQL runner + Notebook。

6. **Synthesizer Agent**
   - 职责：整合分析结果，生成金字塔式结论与证据链，给出 3 条可操作建议（证据、步骤、估算影响、风险与缓解）。
   - 输出：Recommendation package + confidence scores + action list (OKR/任务分解)。

7. **Presentation Agent**
   - 职责：把 Recommendation 包转成可交付物：PPT、PDF、视频脚本、执行清单、音频朗读（TTS）与一页摘要。
   - 输出：PPT 文件、视频 storyboard、音频文件（或 TTS 指令）、Executive Summary 长短版本。

---

# 端到端工作流（序列）
1. 提问者在 UI 输入问题并上传附件。
2. Problem Framer 运行并回问必要澄清问题（交互式）。
3. Issue Tree 生成并展示图形化树。
4. Prioritizer 评分并由用户确认或调整。
5. Planner 生成分析任务并分配资源。
6. Data Analysis 执行并把 artifacts 存入 Memory DB。
7. Synthesizer 读取 artifacts 生成建议草案。
8. Presentation 生成多媒体交付品。
9. 用户审阅并签署执行计划，触发任务到 PMO/执行系统。

---

# API 设计样例

**POST /v1/problems**
请求体：
```json
{ "raw_text":"string", "attachments":[{"name":"file.pdf","type":"pdf"}], "submitter":"user@corp" }
```

响应示例：
```json
{ "id":"uuid", "status":"framing_pending", "created_at":"ISO8601" }
```

**GET /v1/problems/{id}/frame** 返回 ProblemFrame JSON。

**POST /v1/problems/{id}/run-step**
```json
{ "step":"tree|prioritize|plan|analyze|synthesize|present" }
```

**GET /v1/problems/{id}/results** 返回最终报告与下载链接。

---

# 关键数据模型（JSON schema 摘要）

**ProblemRequest**
```json
{ "id":"uuid","raw_text":"string","attachments":[{"type":"pdf|csv","id":"..."}],"submitter":"string","timestamp":"ISO8601" }
```

**ProblemFrame**
```json
{ "goal":"string","scope":"include/exclude","kpis":[{"name":"string","baseline":number,"target":number,"window":"30d"}],"assumptions":["..."],"constraints":["..."],"stakeholders":["..."],"confidence":"low|med|high" }
```

**IssueTree**
```json
{"root":{"id":"","title":"","children":[{"id":"","title":"","hypotheses":["..."],"required_data":["logs","crm"],"notes":""}]}}
```

**AnalysisPlan**
```json
{"task_id":"uuid","question_id":"...","methods":["cohort","regression"],"data_requirements":[{"source":"redshift","table":"events"}],"est_hours":16,"owner":"analyst@corp"}
```

**AnalysisResult**
```json
{"task_id":"","artifacts":[{"type":"chart","url":"/files/.."},{"type":"csv","url":".."}],"summary":"string","metrics":{"p_value":0.01}}```

**Recommendation**
```json
{"id":"","title":"","summary":"","actions":[{"step":"...","owner":"...","due":"ISO8601"}],"expected_impact":{"kpi":"churn","delta_pct":-2},"confidence":0.7}
```

---

# Prompt 设计（每 Agent 最小可用模板）

**Problem Framer - System Prompt**
```
你是严谨的商业分析助手。输入原始问题和附件摘要。返回严格的 JSON：{goal,scope,kpis,assumptions,stakeholders,constraints,confidence}. 必须说明哪些问题需要澄清，并生成交互式澄清问题列表（最多10个）。
```

**Issue Tree**
```
根据 ProblemFrame 生成 MECE 问题树。每个叶子节点给出可验证假设、需要的数据、潜在分析方法。输出 JSON。
```

**Prioritizer**
```
为每个叶子节点计算:影响(1-10),可行性(1-10),成本估计,推荐优先级。说明评分依据。
```

**Planner**
```
为 top-N 问题生成可执行分析计划。包含数据源、SQL/分析脚本骨架、估时、负责人和验收标准。
```

**Data Analysis**
```
执行 Plan 中的方法。输出结构化结果和 artifacts。每个分析步骤输出:操作记录,代码片段,图表,结论和置信度。
```

**Synthesizer**
```
对所有分析结果进行证据链综合。输出:三条建议(每条:背景,证据,执行步骤3步,预估影响,风险与缓解,置信度)。
```

**Presentation**
```
根据 Recommendation 生成 PPT 大纲和每页文本。生成 2 分钟视频脚本和 60 秒音频摘要的 TTS 文本。
```

---

# 数据接入与技术栈
- LLM 层：生产可替换（GPT-family 或类似）并使用 RAG 做事实检索。
- Orchestrator：轻量任务队列（Celery / Temporal / Airflow）或 agent framework（LangChain/CrewAI/Temporal + custom）。
- Vector DB：Pinecone/Weaviate/RedisVector（存 embeddings 及 memory）。
- DB：Postgres（metadata），Data Warehouse（Snowflake/BigQuery/Redshift）。
- Notebook/Runtime：Jupyter/Papermill 或受控 Python sandbox（限制网络/包）。
- 可视化：d3.js（问题树）、Plotly/matplotlib for artifacts。
- 前端：Next.js + Typescript。
- 导出：python-pptx 或 deckgen, FFmpeg 用于视频合成，TTS 引擎用于音频。

---

# UI 设计要点
- 输入页：自然语言输入 + 文件拖拽 + 模板选择。
- 澄清对话：弹性问答插槽，支持快速选择和补充文本。
- 问题树页：交互式树，节点可注释/拆分/合并。
- 优先级页：影响×可行性矩阵，支持手动调整并记录理由。
- 分析任务页：任务队列，实时日志，Artifacts 浏览。
- 报告页：PPT/视频/音频下载及在线播放。
- 审计页：每个结论的证据链与责任人。

---

# 多媒体交付规范（模板）

**PPT 模板结构（每个报告）**
1. 封面：问题、提交者、日期、目标KPI。
2. 摘要页：3 条建议 + 预估影响 + 推荐动作。
3. 问题定义：范围与假设。
4. 问题树：图示。
5. 关键分析：图表与主要结论（每张图给出方法/数据源/置信度）。
6. 建议与执行清单：每条建议 3 个关键步骤。
7. 风险与缓解。
8. 里程碑与责任人。
9. 附件：数据表、Notebook 链接、原始文件。

**2分钟视频脚本（high-level）**
- 0:00-0:10 封面+一句问题陈述。
- 0:10-0:40 关键发现（视觉化图表同步旁白）。
- 0:40-1:20 三条建议逐条说明并展示行动步骤。
- 1:20-1:50 风险与缓解。
- 1:50-2:00 结尾：下一步与联系人。

**60s 音频摘要（TTS）**
- 15s 背景与目标
- 30s 关键建议
- 15s 下一步呼吁

---

# 发现性问题清单（用于获取实际情况）

**业务与目标**
- 这个问题为什么现在重要？
- 目标 KPI 是什么？baseline 和目标值？时间窗？
- 哪些利益相关者必须被包含或批准？

**现状与历史**
- 最近 12 个月关键趋势有哪些？是否有季节性？
- 有没有历史尝试过的措施？效果如何？

**数据可得性**
- 可提供的数据源清单（系统名、表名、接入方式、样本量）。
- 数据所有权与访问权限是谁？是否有 PII 或敏感数据？
- 典型的事件/行为定义（例如 churn 定义）。

**用户与客户**
- 目标用户画像是什么？分层逻辑？
- 是否有已做过的客户调研/访谈/CSAT 数据？

**技术与限制**
- 现有数据仓/ETL 所在位置与 SLA。
- 允许使用第三方服务吗？（TTS、视频合成、外部 LLM）

**政策与合规**
- 是否存在合规约束（GDPR、行业规范、本地法律）？

**执行与预算**
- 可投入的工程与分析小时预算。
- 可接受的实施风险和失败容忍度。

**验收标准**
- 成功看什么？如何量化？执行后评估周期？

---

# 可量化 KPI（平台）
- 业务：建议采纳率、建议实施后 KPI 改进（绝对/相对）、建议导致的 ROI 估算。
- 平台：请求成功率、平均处理时长、Agent失败率、人工干预率。

---

# 可观测性与审计
- 每个结论必须记录来源 artifacts id、运行脚本 hash、运行时间与执行者。
- 改动日志与审批链。
- 证据可回放（notebook + 隔离数据副本）。

---

# 安全与合规
- RBAC、OAuth2 身份认证。
- Data Masking 对 PII 字段。
- 审计日志与数据删除 API。
- 第三方 LLM 使用需记录 prompt 与返回以便审计。

---

# 测试与验收标准（MVP）
- E2E 场景测试（3个行业场景）。
- 对每个场景系统能自动产生：ProblemFrame, IssueTree, Top-3 AnalysisPlan, 至少 1 个 AnalysisResult artifact, Synthesized Recommendations, PPT（可下载）。
- 每个自动结论的置信度需标注。
- 任务成功率 ≥ 90%，单个用户请求端到端响应（不含人工跑脚本）≤ 10 分钟。

---

# 开发计划（建议）
**Sprint0（2 周）**：需求细化、架构定义、CI/CD、基础 API、Auth。
**Sprint1（3 周）**：Problem Framer、IssueTree、输入 UI、Memory DB。
**Sprint2（3 周）**：Prioritizer、Planner、基础 Data Analysis runtime、样例脚本。
**Sprint3（3 周）**：Synthesizer、Presentation、PPT 导出、视频/音频模板、E2E 测试与验收。

---

# 团队与估算
- PM 0.5 FTE
- Backend 2 FTE
- Frontend 1 FTE
- ML/Analyst 1 FTE
- SRE 0.5 FTE
- QA 0.5 FTE

MVP 人月估算：~5 人 × 2 个月 = 10 人月。

---

# 运营与闭环
- 建议建立客户反馈回路。实施后 30/90 天回收 KPI 并进行因果验证。
- 维护模型/analysis 脚本仓库并固定审查周期（每季度）。

---

# 交付物示例（清单）
- API 文档（OpenAPI）
- Prompt 集合与示例输入输出
- 三个演示场景数据与 Notebook
- PPT 模板与示例报告
- 视频 storyboard 与 TTS 文本
- 安全与审计手册

---

# 附录：示例 Prompt 与输出片段
**示例 Problem Framer Prompt**
```
用户问题: "过去三个月 SaaS 客户流失率上升，如何降低流失率 5%？"
请返回 JSON: {goal,scope,kpis,assumptions,constraints,clarifying_questions}
```

**示例 ProblemFrame 输出（简化）**
```json
{ "goal":"降低 churn 率 5% 于 90 天内","scope":"北美付费用户","kpis":[{"name":"monthly_churn","baseline":0.06,"target":0.01}],"assumptions":["流失主要由价格敏感导致"],"clarifying_questions":["是否可访问近期 6 个月的 churn cohort 数据?","是否允许对部分用户执行优惠实验? "] }
```

---

## 结语
该文档提供从产品到工程到交付的端到端设计。下一步建议：
1. 立即运行发现问题清单与三个典型场景的数据收集。
2. 我可导出完整 OpenAPI contract、Postman collection、以及 3 个 prompt 的精细版本并生成示例 PPT。输入“生成 OpenAPI + demo PPT”开始。

