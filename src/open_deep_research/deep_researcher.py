"""Main LangGraph implementation for the Deep Research agent."""

import asyncio
import logging
import time
from typing import Callable, Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from open_deep_research.memory_manager import save_memory, search_memory, DEMO_USER_ID
from open_deep_research.configuration import (
    Configuration,
)
from open_deep_research.python_executor import python_executor_tool_call
from open_deep_research.internal_doc_search import internal_doc_search_tool_call
from open_deep_research.tool_reliability import (
    TransientToolError,
    TRANSIENT_NETWORK_EXCEPTION_TYPES,
    NonRetryableInfrastructureError,
)
from open_deep_research.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
)
from open_deep_research.state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
    SupervisorState,
)
from open_deep_research.utils import (
    anthropic_websearch_called,
    get_all_tools,
    get_api_key_for_model,
    get_model_token_limit,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    openai_websearch_called,
    remove_up_to_last_ai_message,
    think_tool,
)

# Initialize a configurable model that we will use throughout the agent
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)

async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "__end__"]]:
    """Analyze user messages and ask clarifying questions if the research scope is unclear.
    
    This function determines whether the user's request needs clarification before proceeding
    with research. If clarification is disabled or not needed, it proceeds directly to research.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings and preferences
        
    Returns:
        Command to either end with a clarifying question or proceed to research brief
    """
    # Step 1: Check if clarification is enabled in configuration
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        # Skip clarification step and proceed directly to research
        return Command(goto="write_research_brief")
    
    # Step 2: Prepare the model for structured clarification analysis
    messages = state["messages"]
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Configure model with structured output and retry logic
    clarification_model = (
        configurable_model
        .with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )
    
    # Step 3: Analyze whether clarification is needed
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages), 
        date=get_today_str()
    )
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])
    
    # Step 4: Route based on clarification analysis
    if response.need_clarification:
        # End with clarifying question for user
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        # Proceed to research with verification message
        return Command(
            goto="write_research_brief", 
            update={"messages": [AIMessage(content=response.verification)]}
        )


async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """Transform user messages into a structured research brief and initialize supervisor.
    
    This function analyzes the user's messages and generates a focused research brief
    that will guide the research supervisor. It also sets up the initial supervisor
    context with appropriate prompts and instructions.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to research supervisor with initialized context
    """
    # Step 1: Set up the research model for structured output
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Configure model for structured research question generation
    research_model = (
        configurable_model
        .with_structured_output(ResearchQuestion)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 2: Generate structured research brief from user messages
    # 先检索这个用户的历史记忆（跨对话的偏好/约束），拼进生成简报的上下文
    user_messages_text = get_buffer_string(state.get("messages", []))
    memory_search_result = await search_memory(user_messages_text, user_id=DEMO_USER_ID)
    remembered_context = ""
    if memory_search_result["success"] and memory_search_result["memories"]:
        remembered_items = [m.get("memory", "") for m in memory_search_result["memories"]]
        remembered_context = (
            "\n\n[以下是这个用户之前对话中设定过的、可能仍然有效的偏好或约束，"
            "仅供参考，如果本次提问明确给出了不同的信息，以本次为准]\n"
            + "\n".join(f"- {item}" for item in remembered_items)
        )

    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=user_messages_text + remembered_context,
        date=get_today_str()
    )
    response = await research_model.ainvoke([HumanMessage(content=prompt_content)])

    # 写记忆：只存这次提问本身的原始文本，不存 research_brief（模型生成的、
    # 已经转写过的内容）——避免把模型的转述/推断当成"用户说的事实"存下来，
    # 这次提问文本本身可能包含偏好，也可能不包含，mem0 内部的提炼机制会
    # 自己判断值不值得存，这里只负责"喂给它这次用户说了什么"这个原始输入，
    # 不做额外筛选（筛选逻辑交给 mem0 自己的实现，我们不重复造轮子）。
    await save_memory(user_messages_text, user_id=DEMO_USER_ID)
    
    # Step 3: Initialize supervisor with research brief and instructions
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations
    )
    
    return Command(
        goto="research_supervisor", 
        update={
            "research_brief": response.research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.research_brief)
                ]
            }
        }
    )


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """Lead research supervisor that plans research strategy and delegates to researchers.
    
    The supervisor analyzes the research brief and decides how to break down the research
    into manageable tasks. It can use think_tool for strategic planning, ConductResearch
    to delegate tasks to sub-researchers, or ResearchComplete when satisfied with findings.
    
    Args:
        state: Current supervisor state with messages and research context
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to supervisor_tools for tool execution
    """
    # Step 1: Configure the supervisor model with available tools
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Available tools: research delegation, completion signaling, and strategic thinking
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]
    
    # Configure model with tools, retry logic, and model settings
    research_model = (
        configurable_model
        .bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 2: Generate supervisor response based on current context
    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)
    
    # Step 3: Update state and proceed to tool execution
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )

async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """Execute tools called by the supervisor, including research delegation and strategic thinking.
    
    This function handles three types of supervisor tool calls:
    1. think_tool - Strategic reflection that continues the conversation
    2. ConductResearch - Delegates research tasks to sub-researchers
    3. ResearchComplete - Signals completion of research phase
    
    Args:
        state: Current supervisor state with messages and iteration count
        config: Runtime configuration with research limits and model settings
        
    Returns:
        Command to either continue supervision loop or end research phase
    """
    # Step 1: Extract current state and check exit conditions
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]
    
    # Define exit criteria for research phase
    exceeded_allowed_iterations = research_iterations > configurable.max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    # Exit if any termination condition is met
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )
    
    # Step 2: Process all tool calls together (both think_tool and ConductResearch)
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}
    
    # Handle think_tool calls (strategic reflection)
    think_tool_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "think_tool"
    ]
    
    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(ToolMessage(
            content=f"Reflection recorded: {reflection_content}",
            name="think_tool",
            tool_call_id=tool_call["id"]
        ))
    
    # Handle ConductResearch calls (research delegation)
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "ConductResearch"
    ]
    
    if conduct_research_calls:
        try:
            # Limit concurrent research units to prevent resource exhaustion
            allowed_conduct_research_calls = conduct_research_calls[:configurable.max_concurrent_research_units]
            overflow_conduct_research_calls = conduct_research_calls[configurable.max_concurrent_research_units:]
            
            # Execute research tasks in parallel
            research_tasks = [
                researcher_subgraph.ainvoke({
                    "researcher_messages": [
                        HumanMessage(content=tool_call["args"]["research_topic"])
                    ],
                    "research_topic": tool_call["args"]["research_topic"]
                }, config) 
                for tool_call in allowed_conduct_research_calls
            ]
            
            tool_results = await asyncio.gather(*research_tasks)
            
            # Create tool messages with research results
            for observation, tool_call in zip(tool_results, allowed_conduct_research_calls):
                all_tool_messages.append(ToolMessage(
                    content=observation.get("compressed_research", "Error synthesizing research report: Maximum retries exceeded"),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ))
            
            # Handle overflow research calls with error messages
            for overflow_call in overflow_conduct_research_calls:
                all_tool_messages.append(ToolMessage(
                    content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {configurable.max_concurrent_research_units} or fewer research units.",
                    name="ConductResearch",
                    tool_call_id=overflow_call["id"]
                ))
            
            # Aggregate raw notes from all research results
            raw_notes_concat = "\n".join([
                "\n".join(observation.get("raw_notes", [])) 
                for observation in tool_results
            ])
            
            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]
                
        except Exception as e:
            # Handle research execution errors
            if is_token_limit_exceeded(e, configurable.research_model) or True:
                # Token limit exceeded or other error - end research phase
                return Command(
                    goto=END,
                    update={
                        "notes": get_notes_from_tool_calls(supervisor_messages),
                        "research_brief": state.get("research_brief", "")
                    }
                )
    
    # Step 3: Return command with all tool results
    update_payload["supervisor_messages"] = all_tool_messages
    return Command(
        goto="supervisor",
        update=update_payload
    ) 

# Supervisor Subgraph Construction
# Creates the supervisor workflow that manages research delegation and coordination
supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)

# Add supervisor nodes for research management
supervisor_builder.add_node("supervisor", supervisor)           # Main supervisor logic
supervisor_builder.add_node("supervisor_tools", supervisor_tools)  # Tool execution handler

# Define supervisor workflow edges
supervisor_builder.add_edge(START, "supervisor")  # Entry point to supervisor

# Compile supervisor subgraph for use in main workflow
supervisor_subgraph = supervisor_builder.compile()

async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    """Individual researcher that conducts focused research on specific topics.
    
    This researcher is given a specific research topic by the supervisor and uses
    available tools (search, think_tool, MCP tools) to gather comprehensive information.
    It can use think_tool for strategic planning between searches.
    
    Args:
        state: Current researcher state with messages and topic context
        config: Runtime configuration with model settings and tool availability
        
    Returns:
        Command to proceed to researcher_tools for tool execution
    """
    # Step 1: Load configuration and validate tool availability
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    
    # Get all available research tools (search, MCP, think_tool)
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError(
            "No tools found to conduct research: Please configure either your "
            "search API or add MCP tools to your configuration."
        )
    
    # Step 2: Configure the researcher model with tools
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Prepare system prompt with MCP context if available
    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "", 
        date=get_today_str()
    )
    
    # Configure model with tools, retry logic, and settings
    research_model = (
        configurable_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 3: Generate researcher response with system context
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)
    
    # Step 4: Update state and proceed to tool execution
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
        }
    )

# Tool Execution Helper Function

# 哪些工具有"抛类型化异常"的入口可以用来做重试判断（而不是只能走会吞掉一切
# 异常、返回字符串的公开 .ainvoke() 接口）。没有在这里注册的工具（搜索工具、
# MCP 工具、think_tool、ResearchComplete）仍然走原来的 tool.ainvoke() 路径，
# 行为和加这层重试之前完全一致。sql_executor 没有出现在这里：它所有失败路径
# 都已经在工具内部被转换成字符串返回、从不抛异常，对 execute_tool_safely 而言
# 天然就是"不重试"，不需要额外包装（对应 2026-10 诊断报告里"sql_executor：
# 所有错误都不重试"这条分类——不是遗漏，是本来就已经满足）。
_RETRY_AWARE_TOOL_CALLS: dict[str, Callable] = {
    "python_executor": python_executor_tool_call,
    "internal_doc_search": internal_doc_search_tool_call,
}

# 已知局限（2026-10 确认，不是推测）：对注册在 _RETRY_AWARE_TOOL_CALLS 里的
# 工具，execute_tool_safely 调用的是裸的异步函数，不是 tool.ainvoke(args,
# config)。实测验证过——用一个挂了 on_tool_start/on_tool_end 的回调处理器，
# 走 tool.ainvoke() 这条路径（sql_executor 和所有未注册的工具）会正常触发
# 这两个回调；直接调用 python_executor_tool_call(...) 则完全不会触发任何
# 回调事件。这意味着 python_executor / internal_doc_search 经由重试逻辑
# 执行时，不会在 LangSmith/LangGraph 的 tracing 里单独出现为一条工具调用
# span——它们的执行会被"吞"进 researcher_tools 这个节点自身的 span 里，
# 看不到独立的输入/输出/耗时记录。sql_executor 和其他未注册的工具不受影响，
# 仍然走 tool.ainvoke()，tracing 完整。
#
# 为什么不直接修：要保留 tracing，最直接的办法是让 execute_tool_safely 始终
# 调用 tool.ainvoke(args, config)，并让 TransientToolError 真的从公开的
# python_executor / internal_doc_search 工具函数里逃出来（而不是像现在这样
# 被公开函数自己的 except TransientToolError 分支转换回字符串）。但这样会
# 改变这两个工具"无论如何都返回字符串、从不抛异常"这条从加固阶段就定下来的
# 设计原则本身——不只是不破坏现有测试的问题，是任何人以后直接
# `await python_executor.ainvoke(...)` 调用它时的行为预期都会变。这是一次
# 需要明确讨论的设计取舍，不是这一轮顺手能定的事，所以暂时保留现状、如实
# 记录这个局限。


_tool_reliability_logger = logging.getLogger("open_deep_research.tool_reliability")


def _log_tool_attempt(
    tool_name: str, attempt: int, failure_kind: str, duration_s: float, will_retry: bool
) -> None:
    """每次工具尝试（不管成功失败）都记一条结构化日志，作为 2026-10 确认过的
    tracing 丢失（python_executor/internal_doc_search 经由重试逻辑执行时不会
    出现在 LangSmith/LangGraph 的工具调用 span 里）的补偿——至少本地日志里能
    看到"这次 researcher_tools 调用里，哪个工具试了几次、每次什么结果、花了
    多久、有没有接着重试"，也供 T2 第四步的 E2E 故障注入实验分析用。
    用 logger.info 而不是 print，方便按需过滤/重定向，不强制写文件。
    """
    _tool_reliability_logger.info(
        "tool_attempt tool=%s attempt=%d failure_kind=%s duration_s=%.3f will_retry=%s",
        tool_name, attempt, failure_kind, duration_s, will_retry,
    )


async def execute_tool_safely(
    tool, args, config, configurable: "Configuration | None" = None
) -> tuple[str, str, str]:
    """执行单个工具调用，对分类为瞬时性的失败做有限次数的指数退避重试。

    返回 (content, status, failure_kind) 三元组。这是 2026-10 的一次修正：
    之前只返回 content 字符串，熔断逻辑被迫靠"看起来像不像错误"（status）去
    判断要不要计入熔断——这是错的：sql_executor 把原始报错返回给模型自我
    纠错是设计本身的一部分，模型探索表结构时连续写错几条 SQL 非常正常，不该
    被当成"这个工具坏了"而熔断掉。status 和 failure_kind 是两个独立的维度：

    - status（"success" / "error"）：纯展示性的，靠各工具已有的错误字符串
      前缀判断（见 _looks_like_tool_error），用于设置 ToolMessage.status，
      判断错了的代价只是展示不准，不影响任何重试/熔断逻辑。
    - failure_kind（"none" / "tool_input_error" / "infrastructure_error"）：
      决定要不要计入熔断，**只能**由"有没有真的抛出异常、异常是什么类型"
      这个结构性事实决定，绝不根据字符串内容猜测：
        * 工具自己把失败处理成一个字符串正常返回（没有异常逃出来）——无论
          这个字符串是不是看起来像错误（SQL语法错、Python代码跑错、
          python_executor 自己的"代码跑太久"超时）——都是 tool_input_error，
          这是"模型这次的输入/代码有问题"，不是环境的问题，不计入熔断。
        * 有异常真的从工具实现里逃出来，被这里的 except 捕获——无论是
          TransientToolError 重试耗尽、还是未分类的其他异常、还是没注册
          "抛异常版"入口的工具自己 ainvoke() 时抛的异常——都是
          infrastructure_error，计入熔断。

    分类依据真实抛出的异常类型，不解析返回字符串的前缀或内容：
    - 工具在 _RETRY_AWARE_TOOL_CALLS 里注册了"抛异常版"入口时，调用那个入口，
      按 TransientToolError 或已知网络异常类型（TRANSIENT_NETWORK_EXCEPTION_TYPES，
      目前是 openai.APIConnectionError / APITimeoutError）判断是否重试；重试
      次数耗尽后，把异常的原始消息原样返回——和直接调用对应公开工具时会得到
      的字符串完全一致，不额外加前缀。
    - 没有注册"抛异常版"入口的工具，走原来的 tool.ainvoke()，行为和加这层
      重试之前完全一致：任何异常都不重试，直接格式化成
      "Error executing tool: {消息}" 返回，归为 infrastructure_error。

    重试安全性前提：当前已知的全部工具都是只读或无副作用操作，重复执行不会
    产生"副作用被多执行一次"的风险（详见 tool_reliability.py 模块文档字符串）。
    """
    max_retries = configurable.tool_retry_max_attempts if configurable else 2
    base_delay = configurable.tool_retry_base_delay_seconds if configurable else 0.5

    tool_name = tool.name if hasattr(tool, "name") else "<unknown>"
    raising_call = _RETRY_AWARE_TOOL_CALLS.get(tool_name) if tool_name else None

    if raising_call is None:
        t0 = time.monotonic()
        try:
            result = await tool.ainvoke(args, config)
        except Exception as e:
            _log_tool_attempt(tool_name, attempt=1, failure_kind="infrastructure_error",
                               duration_s=time.monotonic() - t0, will_retry=False)
            return f"Error executing tool: {str(e)}", "error", "infrastructure_error"
        content, status, failure_kind = _classify_clean_return(result)
        _log_tool_attempt(tool_name, attempt=1, failure_kind=failure_kind,
                           duration_s=time.monotonic() - t0, will_retry=False)
        return content, status, failure_kind

    attempt = 0
    while True:
        attempt += 1
        t0 = time.monotonic()
        try:
            result = await raising_call(**args)
        except NonRetryableInfrastructureError as e:
            # 分类为基础设施问题（计入熔断），但刻意不重试——见
            # NonRetryableInfrastructureError 的文档字符串（目前唯一场景是
            # internal_doc_search 超时：被放弃的后台线程仍在运行，重试会让
            # 这类线程堆积）。
            _log_tool_attempt(tool_name, attempt=attempt, failure_kind="infrastructure_error",
                               duration_s=time.monotonic() - t0, will_retry=False)
            return str(e), "error", "infrastructure_error"
        except (TransientToolError,) + TRANSIENT_NETWORK_EXCEPTION_TYPES as e:
            will_retry = attempt <= max_retries
            _log_tool_attempt(tool_name, attempt=attempt, failure_kind="infrastructure_error",
                               duration_s=time.monotonic() - t0, will_retry=will_retry)
            if not will_retry:
                return str(e), "error", "infrastructure_error"
            await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
        except Exception as e:
            # 未被分类成瞬时性错误的其他异常——不重试，直接返回，算
            # infrastructure_error（这是一个我们没预料到的失败形态，不是
            # 工具按自己的逻辑产出的、可以理解的"输入有问题"）。
            _log_tool_attempt(tool_name, attempt=attempt, failure_kind="infrastructure_error",
                               duration_s=time.monotonic() - t0, will_retry=False)
            return f"Error executing tool: {str(e)}", "error", "infrastructure_error"
        else:
            content, status, failure_kind = _classify_clean_return(result)
            _log_tool_attempt(tool_name, attempt=attempt, failure_kind=failure_kind,
                               duration_s=time.monotonic() - t0, will_retry=False)
            return content, status, failure_kind


# 各工具已有的错误字符串前缀——仅用于给 ToolMessage 设置展示性的 status 字段
# 和（在"工具自己正常返回字符串，没有异常逃出来"的情况下）区分
# tool_input_error 还是 none。**不用于判断 infrastructure_error**——那个
# 判断只看有没有异常真的逃出来（见 execute_tool_safely 的文档字符串）。
# 这些前缀本来就是各工具一直在用的既有约定，不是本轮新发明的。
_ERROR_STRING_PREFIXES = (
    "SQL Error:",
    "Python Error:",
    "Document search error:",
    "Error executing tool:",
)


def _looks_like_tool_error(result) -> bool:
    return isinstance(result, str) and result.startswith(_ERROR_STRING_PREFIXES)


def _classify_clean_return(result) -> tuple[str, str, str]:
    """工具没有抛异常、正常返回了 result 时的分类：status 靠字符串前缀判断
    （纯展示性，见上）；failure_kind 要么是 "none"（看起来是成功），要么是
    "tool_input_error"（看起来是失败，但既然是正常返回而不是抛异常，就说明
    是工具自己识别并处理掉的、关于这次输入/代码的问题，不是环境问题）——
    这里**不可能**产出 "infrastructure_error"，那个标签只在有异常真的逃出来
    时才会被打上。
    """
    if _looks_like_tool_error(result):
        return result, "error", "tool_input_error"
    return result, "success", "none"


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    """Execute tools called by the researcher, including search tools and strategic thinking.
    
    This function handles various types of researcher tool calls:
    1. think_tool - Strategic reflection that continues the research conversation
    2. Search tools (tavily_search, web_search) - Information gathering
    3. MCP tools - External tool integrations
    4. ResearchComplete - Signals completion of individual research task
    
    Args:
        state: Current researcher state with messages and iteration count
        config: Runtime configuration with research limits and tool settings
        
    Returns:
        Command to either continue research loop or proceed to compression
    """
    # Step 1: Extract current state and check early exit conditions
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]
    
    # Early exit if no tool calls were made (including native web search)
    has_tool_calls = bool(most_recent_message.tool_calls)
    has_native_search = (
        openai_websearch_called(most_recent_message) or 
        anthropic_websearch_called(most_recent_message)
    )
    
    if not has_tool_calls and not has_native_search:
        return Command(goto="compress_research")
    
    # Step 2: Handle other tool calls (search, MCP tools, etc.)
    tools = await get_all_tools(config)
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool
        for tool in tools
    }

    # Execute all tool calls in parallel
    tool_calls = most_recent_message.tool_calls

    # 熔断计数快照：来自本次 researcher 子图运行的 state，每次调用都会算出
    # 更新后的完整字典再整体写回（见下面 Command 的 update），天然随每次
    # researcher_subgraph.ainvoke(...) 调用重新开始，不会跨运行/跨并发污染
    # ——不需要额外加锁或命名空间隔离，LangGraph 的 state 传递机制本身就
    # 保证了这一点（和 tool_call_iterations 走的是同一条路）。
    tool_failure_counts = dict(state.get("tool_failure_counts", {}))

    async def _run_one(tool_call: dict) -> tuple[str, str, str]:
        """返回 (content, status, failure_kind)。

        status 只取 "success" / "error"，纯展示性。
        failure_kind 取 "none" / "tool_input_error" / "infrastructure_error"，
        决定要不要计入熔断（见下面的计数更新和 execute_tool_safely 的文档
        字符串）——未知工具名和"已被熔断"这两种情况下 failure_kind 填什么
        无所谓，因为紧接着的计数更新循环会用 `name not in tools_by_name`
        跳过未知工具名，不会让它污染任何真实工具的计数；"已被熔断"的情况下
        继续累加 infrastructure_error 计数也无害（只会让计数继续增长，不影响
        "是否 >= 阈值"这个判断）。
        """
        name = tool_call["name"]

        # 2026-10 修复：之前这里是 tools_by_name[tool_call["name"]] 的裸索引，
        # 模型一旦产生幻觉调用了不存在的工具名就会抛未捕获的 KeyError，
        # 把整个 researcher_tools 节点崩掉（已经在诊断阶段用真实的
        # researcher_tools(state, config) 调用复现过这个崩溃）。
        if name not in tools_by_name:
            available = ", ".join(sorted(tools_by_name.keys()))
            return (
                f"Error executing tool: unknown tool '{name}'. "
                f"Available tools: {available}",
                "error",
                "infrastructure_error",
            )

        if tool_failure_counts.get(name, 0) >= configurable.tool_circuit_breaker_threshold:
            return (
                f"Error executing tool: '{name}' has failed "
                f"{tool_failure_counts[name]} times in a row during this "
                "research task and is temporarily marked unavailable for the "
                "rest of it.",
                "error",
                "infrastructure_error",
            )

        try:
            content, status, failure_kind = await asyncio.wait_for(
                execute_tool_safely(
                    tools_by_name[name], tool_call["args"], config, configurable
                ),
                timeout=configurable.tool_call_timeout_seconds,
            )
        except asyncio.TimeoutError:
            # 没有自带超时保护的工具的安全网，覆盖的是 execute_tool_safely
            # 整个调用（包括它内部的重试循环和退避等待），不只是单次尝试。
            # 对已有内部超时的三个自建工具（sql_executor 5s / python_executor
            # 最坏约23s含清理 / internal_doc_search 15s）而言，
            # tool_call_timeout_seconds 的默认值（30s）刻意设得比它们各自
            # 最坏情况更长，所以这条分支正常不会抢在工具自己的内部超时之前
            # 触发——它们会先用自己的机制（比如 python_executor 的 docker
            # kill + 轮询确认）干净地返回，不依赖这里的 cancel 去做清理。
            # 这是"执行环境异常导致的超时"，不是代码本身的问题，算
            # infrastructure_error。
            return (
                f"Error executing tool: '{name}' did not complete within "
                f"{configurable.tool_call_timeout_seconds}s (outer safety "
                "timeout).",
                "error",
                "infrastructure_error",
            )

        return content, status, failure_kind

    results = await asyncio.gather(*(_run_one(tc) for tc in tool_calls))

    # 更新熔断计数：只看 failure_kind，不看 status（这是本轮修复的核心）。
    #   infrastructure_error -> 计数 +1
    #   tool_input_error     -> 计数不变（既不清零也不增加——SQL写错/代码
    #                            报错是模型自我纠错流程的正常部分，不该让它
    #                            悄悄侵蚀熔断计数，但也不该因为"看起来像
    #                            成功"而被当成"这个工具现在没问题了"）
    #   none（真正成功）      -> 计数清零
    # 幻觉工具名不计入任何真实工具的失败计数（见 _run_one 的说明）。
    for (_, _, failure_kind), tool_call in zip(results, tool_calls):
        name = tool_call["name"]
        if name not in tools_by_name:
            continue
        if failure_kind == "infrastructure_error":
            tool_failure_counts[name] = tool_failure_counts.get(name, 0) + 1
        elif failure_kind == "none":
            tool_failure_counts[name] = 0
        # failure_kind == "tool_input_error"：有意不做任何改动。

    # Create tool messages from execution results
    tool_outputs = [
        ToolMessage(
            content=content,
            name=tool_call["name"],
            tool_call_id=tool_call["id"],
            status=status,
        )
        for (content, status, _), tool_call in zip(results, tool_calls)
    ]

    # Step 3: Check late exit conditions (after processing tools)
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    if exceeded_iterations or research_complete_called:
        # End research and proceed to compression
        return Command(
            goto="compress_research",
            update={
                "researcher_messages": tool_outputs,
                "tool_failure_counts": tool_failure_counts,
            }
        )

    # Continue research loop with tool results
    return Command(
        goto="researcher",
        update={
            "researcher_messages": tool_outputs,
            "tool_failure_counts": tool_failure_counts,
        }
    )

async def compress_research(state: ResearcherState, config: RunnableConfig):
    """Compress and synthesize research findings into a concise, structured summary.
    
    This function takes all the research findings, tool outputs, and AI messages from
    a researcher's work and distills them into a clean, comprehensive summary while
    preserving all important information and findings.
    
    Args:
        state: Current researcher state with accumulated research messages
        config: Runtime configuration with compression model settings
        
    Returns:
        Dictionary containing compressed research summary and raw notes
    """
    # Step 1: Configure the compression model
    configurable = Configuration.from_runnable_config(config)
    synthesizer_model = configurable_model.with_config({
        "model": configurable.compression_model,
        "max_tokens": configurable.compression_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.compression_model, config),
        "tags": ["langsmith:nostream"]
    })
    
    # Step 2: Prepare messages for compression
    researcher_messages = state.get("researcher_messages", [])
    
    # Add instruction to switch from research mode to compression mode
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))
    
    # Step 3: Attempt compression with retry logic for token limit issues
    synthesis_attempts = 0
    max_attempts = 3
    
    while synthesis_attempts < max_attempts:
        try:
            # Create system prompt focused on compression task
            compression_prompt = compress_research_system_prompt.format(date=get_today_str())
            messages = [SystemMessage(content=compression_prompt)] + researcher_messages
            
            # Execute compression
            response = await synthesizer_model.ainvoke(messages)
            
            # Extract raw notes from all tool and AI messages
            raw_notes_content = "\n".join([
                str(message.content) 
                for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
            ])
            
            # Return successful compression result
            return {
                "compressed_research": str(response.content),
                "raw_notes": [raw_notes_content]
            }
            
        except Exception as e:
            synthesis_attempts += 1
            
            # Handle token limit exceeded by removing older messages
            if is_token_limit_exceeded(e, configurable.research_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                continue
            
            # For other errors, continue retrying
            continue
    
    # Step 4: Return error result if all attempts failed
    raw_notes_content = "\n".join([
        str(message.content) 
        for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
    ])
    
    return {
        "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
        "raw_notes": [raw_notes_content]
    }

# Researcher Subgraph Construction
# Creates individual researcher workflow for conducting focused research on specific topics
researcher_builder = StateGraph(
    ResearcherState, 
    output=ResearcherOutputState, 
    config_schema=Configuration
)

# Add researcher nodes for research execution and compression
researcher_builder.add_node("researcher", researcher)                 # Main researcher logic
researcher_builder.add_node("researcher_tools", researcher_tools)     # Tool execution handler
researcher_builder.add_node("compress_research", compress_research)   # Research compression

# Define researcher workflow edges
researcher_builder.add_edge(START, "researcher")           # Entry point to researcher
researcher_builder.add_edge("compress_research", END)      # Exit point after compression

# Compile researcher subgraph for parallel execution by supervisor
researcher_subgraph = researcher_builder.compile()

async def final_report_generation(state: AgentState, config: RunnableConfig):
    """Generate the final comprehensive research report with retry logic for token limits.
    
    This function takes all collected research findings and synthesizes them into a 
    well-structured, comprehensive final report using the configured report generation model.
    
    Args:
        state: Agent state containing research findings and context
        config: Runtime configuration with model settings and API keys
        
    Returns:
        Dictionary containing the final report and cleared state
    """
    # Step 1: Extract research findings and prepare state cleanup
    notes = state.get("notes", [])
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)
    
    # Step 2: Configure the final report generation model
    configurable = Configuration.from_runnable_config(config)
    writer_model_config = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Step 3: Attempt report generation with token limit retry logic
    max_retries = 3
    current_retry = 0
    findings_token_limit = None
    
    while current_retry <= max_retries:
        try:
            # Create comprehensive prompt with all research context
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str()
            )
            
            # Generate the final report
            final_report = await configurable_model.with_config(writer_model_config).ainvoke([
                HumanMessage(content=final_report_prompt)
            ])
            
            # Return successful report generation
            return {
                "final_report": final_report.content, 
                "messages": [final_report],
                **cleared_state
            }
            
        except Exception as e:
            # Handle token limit exceeded errors with progressive truncation
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1
                
                if current_retry == 1:
                    # First retry: determine initial truncation limit
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                            "messages": [AIMessage(content="Report generation failed due to token limits")],
                            **cleared_state
                        }
                    # Use 4x token limit as character approximation for truncation
                    findings_token_limit = model_token_limit * 4
                else:
                    # Subsequent retries: reduce by 10% each time
                    findings_token_limit = int(findings_token_limit * 0.9)
                
                # Truncate findings and retry
                findings = findings[:findings_token_limit]
                continue
            else:
                # Non-token-limit error: return error immediately
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [AIMessage(content="Report generation failed due to an error")],
                    **cleared_state
                }
    
    # Step 4: Return failure result if all retries exhausted
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [AIMessage(content="Report generation failed after maximum retries")],
        **cleared_state
    }

# Main Deep Researcher Graph Construction
# Creates the complete deep research workflow from user input to final report
deep_researcher_builder = StateGraph(
    AgentState, 
    input=AgentInputState, 
    config_schema=Configuration
)

# Add main workflow nodes for the complete research process
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)           # User clarification phase
deep_researcher_builder.add_node("write_research_brief", write_research_brief)     # Research planning phase
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)       # Research execution phase
deep_researcher_builder.add_node("final_report_generation", final_report_generation)  # Report generation phase

# Define main workflow edges for sequential execution
deep_researcher_builder.add_edge(START, "clarify_with_user")                       # Entry point
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation") # Research to report
deep_researcher_builder.add_edge("final_report_generation", END)                   # Final exit point

# Compile the complete deep researcher workflow
deep_researcher = deep_researcher_builder.compile()