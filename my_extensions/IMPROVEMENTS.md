# 项目改进清单（已发现、待办、已修复）

最后更新：2026-09-18

## 已完成

### DataAnalyst 角色路由（Prompt 层面）
- **状态**：已完成，已提交（commit `0c8c6c4`），已用真实测试验证
- **实现方式**：这套架构下 `ConductResearch` 只有一个字符串字段 `research_topic`，没有结构化的角色类型字段，因此"角色路由"只能通过 Prompt 措辞体现，不是代码结构层面的独立子图
- **具体改动**：`research_system_prompt` 加入 sql_executor/python_executor 的工具说明；`lead_researcher_prompt` 加入"内部数据类任务需在 research_topic 中显式点名工具"的指令
- **验证证据**：真实运行中，Supervisor 的 think_tool 反思文字明确写出"requiring use of sql_executor and python_executor"，并且能看到完整的 python_executor 执行代码原文（`statistics.pstdev(...)`）
- **证据强度的边界**：反思文字证明的是"计划"，代码原文证明的是"内容"，但两者与"真实执行结果"之间尚未做结构化关联（比如把某次 tool_calls 和对应的 ToolMessage 直接对应起来）——如果被追问"你怎么证明这份代码真的被执行了"，诚实的回答是：间接证据（最终数字精确匹配、Sources 列表点名 pandas/statistics）+ 反思文字的过程描述叠加支撑，不是结构化断言

### P1：async def 包裹同步阻塞调用
- **状态**：已修复，已用 13 条现有测试回归验证通过
- **问题**：`sql_executor`/`python_executor` 用 `async def` 声明，但内部调用 `sqlite3`/`exec()` 是同步阻塞的，会占用主事件循环、影响其他并发请求
- **发现方式**：Claude Code 独立静态代码审查发现（不是自己写代码时发现的）
- **修复方式**：用 `asyncio.to_thread()` 把同步逻辑包进独立线程执行

## 已发现，待验证/待修复

### 报告中的统计定义与代码不一致
- **代码证据**：`statistics.pstdev(...)`，采用总体标准差（分母为 N）
- **报告问题**：报告文字描述的公式是样本标准差（分母为 N-1）
- **影响**：读者可能无法判断报告里的文字描述和实际计算是否一致
- **建议改进**：报告生成时应明确标注用的是总体还是样本标准差；增加"实际计算方法"与"报告文字描述"的一致性校验
- **注意**：只能确认"报告文字"和"代码"这两处不一致，不能断言这个不一致具体发生在模型的哪个内部环节

### Supervisor 派发时的停止原因丢失
- **发现来源**：精读 `researcher_tools`/`compress_research` 源码确认
- **问题**：`tool_call_iterations`（记录 Researcher 是否因撞上限被迫中断）从未被读取、也从未传给 Supervisor，`compressed_research` 只包含摘要文字
- **影响**：Supervisor 无法区分"Researcher 主动完成"还是"被迫中断"，只能凭摘要内容长短判断信息够不够
- **建议改进**：给 `compress_research` 返回值增加结构化字段 `stop_reason`（如 `completed` / `iteration_limit_reached`）

### Supervisor 并行判断门槛偏保守
- **发现来源**：真实运行观察——明显可拆分的多维度问题，Supervisor 仍只派发单个 Researcher
- **原因**：`lead_researcher_prompt` 里 "Bias towards single agent" 这条策略门槛较高
- **建议改进**：调低判断门槛，做一次并行 vs 串行的对照实验（Token/耗时/准确率）

### 最终报告语言不受控（观察到一次，未复现验证）
- **现象**：英文提问，最终报告是纯中文生成的
- **根本原因**：未确认，怀疑是 `final_report_generation_prompt` 没有强制约束输出语言
- **建议改进**：Prompt 里加一句明确要求语言与提问一致

### python_executor 安全边界（P2，Claude Code 已提示）
- **已覆盖**：`__import__` 白名单限制（pandas/math/statistics），已用测试验证 os 访问被拦住
- **未覆盖**：模块白名单不等于安全沙箱——没有内存/CPU/执行时间限制，没有真正的进程隔离；`asyncio.to_thread()` 只解决了阻塞事件循环的问题，不提供资源隔离
- **建议改进**：生产化前需要接入真正的沙箱（Docker容器、E2B等）

### sql_executor 安全边界（P2，Claude Code 已提示）
- **已覆盖**：只允许 SELECT/PRAGMA 开头，已用测试验证 DELETE 被拒绝（双重验证：拒绝消息 + 数据确实还在）
- **未覆盖**：危险 PRAGMA（部分 PRAGMA 语句能修改数据库运行时配置，不是绝对只读）、注释前缀绕过检测、数据库文件缺失时的静默失败、并发访问同一 SQLite 文件
- **建议改进**：如需更保守，可将允许前缀从 select/pragma 收紧为只允许 select，或对 pragma 后续内容做二次白名单匹配

### 端到端测试的可观测性限制
- **问题**：`get_state()` 顶层状态只保留两条消息（提问+报告），子图内部细节不可见；流式接口 `stream_subgraphs` 默认为 False；即使打开也只能看到节点跳转结构，具体 `tool_calls` 字段拿不到
- **根本原因**：Researcher 子图是在 `supervisor_tools` 函数内部用 `.ainvoke()` 程序化调用的，不是正式挂接的图节点，这层细节默认不会被转发
- **当前应对方式**：退回到间接证据交叉验证（最终数字精确匹配 + compress_research 过程描述 + Sources 引用列表点名工具/库名）
- **未验证**：是否能通过自定义 instrumentation（比如在 `.ainvoke()` 调用处手动加日志/回调）实现结构化断言，这条路径还没有尝试

## 待办（尚未开始）

- [ ] mem0 跨会话记忆接入图（已验证 add/search 可用，未接入 deep_researcher 图）
- [ ] RAG（企业内部非结构化文档检索——注意这是与 sql_executor "故意不做 RAG" 完全不同的应用场景，前者是文档语义检索，后者是结构化 schema 探索，面试时需要分清楚讲，不要说成矛盾）
- [ ] MCP Server 封装（把已稳定的 SQL/RAG 能力包装成 MCP 工具）
- [ ] 重复公司名测试（需要先往 demo.db 插入测试数据）
- [ ] 系统化可靠性测试集的 python_executor 部分（目前只有 sql_executor 有 3 条可靠性测试）
