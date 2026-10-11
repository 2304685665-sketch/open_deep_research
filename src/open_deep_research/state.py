"""Graph state definitions and data structures for the Deep Research agent."""

import operator
from typing import Annotated, Optional

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


###################
# Structured Outputs
###################
class ConductResearch(BaseModel):
    """Call this tool to conduct research on a specific topic."""
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )

class ResearchComplete(BaseModel):
    """Call this tool to indicate that the research is complete."""

class Summary(BaseModel):
    """Research summary with key findings."""
    
    summary: str
    key_excerpts: str

class ClarifyWithUser(BaseModel):
    """Model for user clarification requests."""
    
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )

class ResearchQuestion(BaseModel):
    """Research question and brief for guiding research."""
    
    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )


###################
# State Definitions
###################

def override_reducer(current_value, new_value):
    """Reducer function that allows overriding values in state."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)
    
class AgentInputState(MessagesState):
    """InputState is only 'messages'."""

DEGRADATION_EVENT_DOC = """degradation_events：工具在这次研究过程中真正"不可用"的事件记录
（2026-10 T3新增）。每条事件是一个dict：
    {type, source, reason, failure_kind, attempts, research_topic, timestamp}
- type：目前只有 "source_unavailable"（一个工具/数据源在本次任务里确实
  不可用），为未来 loop_terminated/budget_exhausted 这类非工具层面的降级
  预留，当前不产生其他type值。
- source：工具名（比如 "internal_doc_search"）。
- reason："retries_exhausted"（execute_tool_safely 重试/外层超时都用尽了，
  仍是 infrastructure_error）或 "circuit_breaker_open"（熔断已经打开，
  这次调用被直接拦截，工具这次根本没被真的调用）。
- failure_kind：和产生这条事件的那次 _run_one 结果一致（目前恒为
  "infrastructure_error"，因为只有这种情况才会生成事件，但保留这个字段
  是为了不丢失原始分类信息，不是因为它会取别的值）。
- attempts：真实尝试次数；熔断拦截时固定是0（工具没被真的调用）；外层
  安全超时打断时是None（不知道内部具体执行到第几次重试，如实标注未知，
  不编造一个数字）。
- research_topic：产生这条事件时，所在的 researcher 子图正在调查的主题——
  这是并行多个 researcher 时区分"哪个研究方向遇到了问题"的关键字段，
  丢了这个字段整条记录就没法在最终报告里讲清楚是哪部分结论受影响。
- timestamp：UTC ISO格式，仅用于排查/调试，不进最终报告的披露小节
  （披露小节按 source+research_topic+reason 去重，timestamp 不参与去重
  也不展示）。

reducer 用 operator.add（简单list拼接，和 researcher_messages 同一种
做法）——这是事件日志，天然该累积，不需要 override_reducer 的"整体替换"
语义。原始事件全部保留，不在state层面去重；去重是最终报告披露小节自己的
职责（按 source+research_topic+reason 三元组），不是这里的责任。
"""

FETCHED_SOURCES_DOC = """fetched_sources：本次研究过程中真实从 ToolMessage
里抽取出的URL记录（2026-10 T3新增，供 citation_verification.py 用）。
每条是一个dict：{source, url, research_topic}。

刻意**不从 raw_notes 解析**——raw_notes 混了 AI 消息和 ToolMessage
（见 compress_research 的 filter_messages(..., include_types=["tool","ai"])），
AI 消息里可能出现模型自己写的、并非真的抓取过的URL，混进来会让
citation_verification 的"这个引用是不是真的检索到过"这个判断失去意义。
只在 researcher_tools 里对真实 ToolMessage 内容做结构化解析（匹配
tavily_search 自己固定输出的 "URL: {url}" 这一行格式，不是关键词猜测）。

reducer 同样用 operator.add。
"""


class AgentState(MessagesState):
    """Main agent state containing messages and research data."""

    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    final_report: str
    degradation_events: Annotated[list[dict], operator.add] = []  # 见 DEGRADATION_EVENT_DOC
    fetched_sources: Annotated[list[dict], operator.add] = []  # 见 FETCHED_SOURCES_DOC

class SupervisorState(TypedDict):
    """State for the supervisor that manages research tasks."""

    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0
    raw_notes: Annotated[list[str], override_reducer] = []
    degradation_events: Annotated[list[dict], operator.add] = []  # 见 DEGRADATION_EVENT_DOC
    fetched_sources: Annotated[list[dict], operator.add] = []  # 见 FETCHED_SOURCES_DOC

class ResearcherState(TypedDict):
    """State for individual researchers conducting research."""

    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []
    # 每个工具连续失败次数的快照，key 是工具名。没有用 Annotated 包 reducer——
    # 和 tool_call_iterations 一样采用 LangGraph 对普通字段的默认行为（整体替换），
    # 因为 researcher_tools 每次都会算出"更新后的完整计数字典"再整体写回，不需要
    # 任何合并语义。这个字段只存在于单次 researcher 子图调用的 state 里，天然随每次
    # researcher_subgraph.ainvoke(...) 调用重新开始，不会被不同的研究任务/并发运行共享。
    tool_failure_counts: dict[str, int] = {}
    degradation_events: Annotated[list[dict], operator.add] = []  # 见 DEGRADATION_EVENT_DOC
    fetched_sources: Annotated[list[dict], operator.add] = []  # 见 FETCHED_SOURCES_DOC

class ResearcherOutputState(BaseModel):
    """Output state from individual researchers."""

    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []
    # 这两个字段必须在这里重复声明一遍——researcher_subgraph 编译时用
    # output=ResearcherOutputState 收窄了输出，没有在这里声明的
    # ResearcherState 字段（包括之前的 tool_failure_counts）会在子图边界
    # 被直接丢弃，调用方（supervisor_tools）完全看不到。这是T3诊断阶段
    # 读代码时确认过的真实边界，不是假设。
    degradation_events: Annotated[list[dict], operator.add] = []  # 见 DEGRADATION_EVENT_DOC
    fetched_sources: Annotated[list[dict], operator.add] = []  # 见 FETCHED_SOURCES_DOC