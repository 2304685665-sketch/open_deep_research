# Evidence Ledger（证据台账）

每一条对外说法都对应：实现位置、测试、已验证的环境、尚未验证的部分与限制。
“验证状态”只写实际跑过的东西；没跑过的明确写“未验证”。上游已有功能不在本台账内（见 README 的对照表）。

约定：测试数量按**模块分别列出，不相加**；“离线”表示不调用任何真实模型/Embedding API；
“沙箱”指 Linux 沙箱（Python 3.10；`test_sql_executor` 另在 3.12 下跑过）。

---

## 1. 工具调用可靠性
- **说法**：对瞬时性故障做有限次指数退避重试；连续失败熔断；外层超时；按真实异常类型把失败分为 `tool_input_error` / `infrastructure_error`，后者才计入熔断。
- **实现**：`src/open_deep_research/tool_reliability.py`、`deep_researcher.py` 中 `execute_tool_safely` 与 `researcher_tools`。
- **测试**：`test_tool_reliability.py`（18，需 Docker）、`test_sql_reliability.py`；`test_degradation_disclosure.py`（27，另含 19 个确定性场景）。
- **合成故障注入**：`sim_fault_injection.py`（`test_sim_fault_injection.py`，6）。对真实 `execute_tool_safely` 注入独立同分布的瞬时故障，成功率与 `1 − p^(r+1)` 吻合；结果在 `eval_results/sim_fault_injection.json`，**属合成数据**。
- **验证状态**：离线测试沙箱通过；Docker 相关测试此前在 Mac 通过（Docker 运行时）。
- **限制**：真实故障常成片出现，重试收益会低于独立同分布的模拟；熔断未纳入模拟。

## 2. 模型降级
- **说法**：主模型在连接失败、超时、限流、服务端 5xx（且 `with_retry` 耗尽）后改用备用模型；认证失败、参数错误等请求类错误不降级。
- **实现**：`model_fallback.py`；配置项 `research_model_fallback`（默认不启用）。
- **测试**：`test_model_fallback.py`（8，假模型）。
- **限制**：备用模型输出质量无保证，只保证可用性。

## 3. 降级披露与引用核验
- **说法**：工具降级事件在报告中披露；报告引用的 URL 必须出现在本次抓取记录中；引用的内部文档 doc_id 必须出现在本次 `internal_doc_search` 返回中。
- **实现**：`citation_verification.py`、`deep_researcher.py`（`fetched_sources` / `fetched_doc_ids` 状态通道）。
- **测试**：`test_degradation_disclosure.py`（27）、`test_internal_citations.py`（23）、`eval_degradation_t3.py`。
- **限制**：核验只证明“这个来源在本次检索中出现过”，**不证明报告对其内容的转述正确**；不得表述为“消除幻觉”。

## 4. 运行追踪、循环检测、失败归因
- **实现**：`run_trace.py`（默认关闭）；循环检测（重复/ABAB，默认关闭）；`my_extensions/summarize_trace.py`；`attribute_failures.py`。
- **测试**：`test_run_trace.py`（16）、`test_loop_detection.py`（22）、`test_attribute_failures.py`（6）。
- **限制**：失败归因目前只在合成轨迹上验证，尚无真实运行轨迹的分析结果。

## 5. 文档摄取
- **说法**：Markdown/PDF 解析、页眉页脚去除、清洗、分块。
- **实现**：`doc_ingest.py`。**测试**：`test_doc_ingest.py`（10；含与现有 25 块索引一致性检查）。
- **限制**：不含 OCR；PDF 标题识别是启发式。

## 6. 混合检索
- **说法**：可选 BM25（CJK 二元分词）与向量检索经 RRF 融合。
- **实现**：`hybrid_retrieval.py`、`internal_doc_search.py`（`INTERNAL_DOC_HYBRID_SEARCH=1` 启用，默认关闭）。
- **测试**：`test_hybrid_retrieval.py`（14）、`test_hybrid_integration.py`（6，含关闭时与原路径逐字节一致）。
- **限制**：BM25 只在向量候选池（≤50）内重排，**不是完整双路召回**；**无任何检索效果数据**。

## 7. Rerank
- **说法**：可选 LLM 相关性重排，失败（异常/超时/输出无法解析/模型构造失败）回退原顺序。
- **实现**：`rerank.py`（`INTERNAL_DOC_RERANK=1`，默认关闭）。**测试**：`test_rerank.py`（8，假打分器）。
- **限制**：未用真实模型评估过效果与延迟；每次启用多一次 LLM 调用。

## 8. 检索评测（草稿）
- **说法**：已有评测脚本与 33 条冻结查询集 v1。
- **实现**：`eval_retrieval.py`、`eval_retrieval_queries.json`；**测试**：`test_eval_retrieval.py`（8）。
- **验证状态**：查询集**未经审核、未运行**；没有任何指标结果。
- **限制**：25 个 chunk 的小语料存在天花板效应；不得在审核运行之前报告任何提升。

## 9. Agent 评测用例与判分器
- **说法**：27 条用例，期望值由 `demo.db` 现算；判分器含金额容差、日期归一化、实体首现、缺失数据拒答。
- **实现**：`agent_eval_cases.py`、`agent_eval_score.py`；**测试**：`test_agent_eval.py`（8，含“标准答案全过、篡改答案全不过”的判分器审计）。
- **验证状态**：**尚未用真实 Agent 跑过**，没有通过率数据。
- **限制**：数据库仅 4 家公司/5 轮融资，是覆盖多种查询形态的回归集，不是统计意义上的大样本。

## 10. SQL 安全
- **说法**：只读连接（`mode=ro`）加语句白名单加超时；SQL 攻击/正常/资源矩阵对新旧实现对比。
- **实现**：`sql_executor.py`；`sql_guard_cases.py`、`eval_sql_guard.py`（37 条：23 攻击、12 正常、2 资源）。
- **测试**：`test_sql_executor.py`（20，3.12 下通过）。结果见 `eval_results/sql_guard_*_AUDIT.json`。
- **限制**：部分用例标记为“测试未能验证”（见脚本文档字符串）；不是形式化安全证明。

## 11. Docker Python 沙箱
- **实现**：`python_executor.py`、`my_extensions/docker/`。**测试**：`test_python_executor.py`（23，需 Docker）。
- **验证状态**：需 Docker 的测试在 Mac 上运行；本台账的沙箱环境无 Docker，未重新验证。
- **限制**：不得描述为生产级沙箱安全。

## 12. MCP 服务端
- **说法**：以 MCP 协议对外暴露只读 `sql_executor` 与 `internal_doc_search`（stdio / 仅本机 streamable-http）；刻意不暴露 `python_executor`。
- **实现**：`mcp_server.py`。**测试**：`test_mcp_server.py`（6）。
- **限制**：未用外部 MCP 客户端做过联调。

## 13. Skill
- **实现**：`skills.py`、`skills/*/SKILL.md`（3 个）；`DEEP_RESEARCH_SKILLS=1` 启用，默认关闭。**测试**：`test_skills.py`（7，含接入 researcher 节点的假模型回归）。
- **限制**：触发词关键词匹配，不是语义匹配；没命中不注入。

## 14. 记忆准入过滤
- **说法**：写入前拒绝指令式文本/密钥口令/证件号/Luhn 通过的卡号/超长内容；读取时丢弃不合格存量条目并截断。
- **实现**：`memory_guard.py`、`memory_manager.py`（`MEMORY_GUARD=1`，默认关闭）。**测试**：`test_memory_guard.py`（8；18 条攻击、12 条正常用例）。
- **限制**：正则规则，拦不住换了说法的注入，也会误拦讨论注入的正常提问；**不得表述为“防住了提示词注入”**。

## 15. 人工审批
- **说法**：工具**执行前**的审批闸门；被拒绝/超时/无审批代理时工具不会被调用。
- **实现**：`approval.py`、`deep_researcher.py` 的 `execute_tool_safely` 开头（`human_approval_enabled`，默认关闭）。
- **测试**：`test_approval.py`（10）：拒绝时调用次数为 0，批准时恰好 1 次，无代理时不执行。
- **限制**：这不是 LangGraph `interrupt()`；等待期间占用并发位；进程重启丢失等待中的协程；策略不是安全边界（如 `FROM a, b` 隐式连接不被识别）。

## 16. 服务层与 Web 控制台
- **说法**：FastAPI + Redis 异步研究任务，幂等（`Idempotency-Key` + 请求体哈希）、限流（固定窗口）、并发上限、状态查询、SSE 状态流、指标端点、React 控制台。
- **实现**：`api_service.py`、`web/`、`deploy/`。
- **测试**：`test_api_service.py`（11）、`test_api_metrics_web.py`（8）。使用 fakeredis 与假 runner。
- **验证状态**：真实 Redis、Docker 构建、服务启动、浏览器渲染均**未验证**。
- **限制**：任务为进程内 asyncio 任务，**重启不恢复**；SSE 是状态变化而非 token 流；指标是服务自身计数，无真实用户数据，不得写留存率/激活率数字。

## 17. CI
- **实现**：`.github/workflows/tests.yml`（Python 3.12，运行上述离线测试脚本）。
- **验证状态**：**尚未在真实 GitHub Actions 上运行**。沙箱中按同样列表在临时克隆里跑过：22 项中 `test_sql_executor` 在 3.10 失败（已知的 `sqlite3.Warning`），在 3.12 通过。
