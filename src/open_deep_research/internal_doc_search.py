"""internal_doc_search 工具 —— 检索企业内部非结构化文档（RAG）。

设计理念：跟 sql_executor 的"不做RAG"刚好相反——sql_executor 面对的是结构化的
schema，信息密度高，模型自己探索就够了；这里面对的是非结构化文档（公司介绍、
会议纪要、行业报告），需要语义检索才能找到相关片段，这是完全不同的场景，
两者不矛盾（详见 IMPROVEMENTS.md 的角色划分记录）。

已知限制（经过三轮对照实验确认）：
- 使用 OpenAI text-embedding-3-small，每次查询产生少量 API 费用
- 只在小型模拟知识库（10篇文档）上验证过基本检索能力，未做大规模评测
- 无答案场景的拒答能力尚未系统性验证
"""

import asyncio
import os
import chromadb
from dotenv import load_dotenv
load_dotenv()
from chromadb.utils import embedding_functions
from langchain_core.tools import tool

from open_deep_research.tool_reliability import (
    TRANSIENT_NETWORK_EXCEPTION_TYPES,
    TransientToolError,
    NonRetryableInfrastructureError,
)

CHROMA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "my_extensions", "chroma_db"
)

# 2026-10 加入：之前这个工具完全没有超时保护——Chroma 查询卡住时没有任何一层
# 会打断它。15s 是刻意选的"够一次正常 Embedding API 调用用，但不会让一次卡住
# 的调用无限期占着"的折衷值，和 sql_executor 的 5s、python_executor 的 15s
# 属于同一量级，不是精确调优过的数字。
INTERNAL_DOC_SEARCH_TIMEOUT_SECONDS = 15

INTERNAL_DOC_SEARCH_DESCRIPTION = (
    "Search the company's internal document knowledge base (company introductions, "
    "product descriptions, industry reports, meeting notes, investment memos). "
    "Use this INSTEAD of web search when the question is about internal company "
    "opinions, internal analysis, or company/product background that would not be "
    "publicly available online. Returns relevant text snippets with their source "
    "document. If no clearly relevant results are found, say so honestly rather "
    "than guessing."
)


async def _search_internal_docs(query: str, n_results: int) -> str:
    """核心实现：执行检索并格式化结果。

    网络层异常（openai 连接/超时类）不在这里被吞掉转成字符串，而是转换成
    TransientToolError 向上抛——是否转换回字符串由下面两个调用入口之一决定：
    公开的 internal_doc_search 工具会转换（保留原有的直接调用行为），
    execute_tool_safely 用的 internal_doc_search_tool_call 不转换（让重试
    判断能看到真实异常类型，而不是解析字符串前缀）。
    """
    def _run_search():
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model_name="text-embedding-3-small",
        )
        collection = client.get_collection(
            "company_knowledge_base", embedding_function=openai_ef
        )
        return collection.query(query_texts=[query], n_results=n_results)

    try:
        results = await asyncio.wait_for(
            asyncio.to_thread(_run_search), timeout=INTERNAL_DOC_SEARCH_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        # 已知局限，诚实记录而非假装已解决：asyncio.wait_for 包一个用
        # asyncio.to_thread 跑的阻塞调用时，超时只会让我们这边提前返回，
        # 后台线程里真正的 Embedding API 请求不会被真正打断，会继续跑到它
        # 自己的网络超时/完成为止，结果只是被丢弃——这是"放弃等待"，不是
        # python_executor 那种用 docker kill 真正终止执行的"确认清理"。
        #
        # 2026-10 分类修正：这不是"这次查询的内容有问题"（tool_input_error），
        # 是网络/Embedding 服务层面的故障，应该归为 infrastructure_error、
        # 计入熔断——但不能重试（见 NonRetryableInfrastructureError 的文档
        # 字符串：被放弃的后台线程仍在运行，重试会让这类线程持续堆积）。
        # 改成 raise 而不是直接 return 字符串，这样 execute_tool_safely 才能
        # 按真实的异常类型分类，不用猜字符串。
        raise NonRetryableInfrastructureError(
            f"Document search error: query did not complete within "
            f"{INTERNAL_DOC_SEARCH_TIMEOUT_SECONDS}s and was abandoned "
            "(the underlying request may still be running in the background)."
        )
    except TRANSIENT_NETWORK_EXCEPTION_TYPES as e:
        raise TransientToolError(f"Document search error: {type(e).__name__}: {str(e)}")

    if not results["ids"][0]:
        return "No relevant documents found in the internal knowledge base."

    output_lines = []
    for i in range(len(results["ids"][0])):
        doc_id = results["metadatas"][0][i]["doc_id"]
        section = results["metadatas"][0][i]["section"]
        distance = results["distances"][0][i]
        text = results["documents"][0][i]
        output_lines.append(
            f"[Source: {doc_id}, section: {section}, relevance_distance: {distance:.3f}]\n{text}"
        )

    return "\n\n---\n\n".join(output_lines)


@tool(description=INTERNAL_DOC_SEARCH_DESCRIPTION)
async def internal_doc_search(query: str, n_results: int = 3) -> str:
    """Search internal documents and return relevant snippets with sources.

    Args:
        query: Natural language search query.
        n_results: How many top results to return (default 3).

    Returns:
        Formatted text with each result's source document and content,
        or an error message if the search failed.
    """
    try:
        return await _search_internal_docs(query, n_results)
    except (TransientToolError, NonRetryableInfrastructureError) as e:
        return str(e)
    except Exception as e:
        return f"Document search error: {type(e).__name__}: {str(e)}"


async def internal_doc_search_tool_call(query: str, n_results: int = 3) -> str:
    """execute_tool_safely 实际调用的入口，而不是上面的公开工具。

    行为和公开工具完全一致，唯一区别是不吞掉 TransientToolError——
    execute_tool_safely 需要看到真实异常类型才能判断是否值得重试。
    LangChain 通过 `.ainvoke()` 直接调用公开工具时（包括现有测试）走的是
    上面那条会把 TransientToolError 转换回字符串的路径，不受这个函数影响。
    """
    return await _search_internal_docs(query, n_results)
