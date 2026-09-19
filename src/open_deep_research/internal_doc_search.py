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

import os
import chromadb
from dotenv import load_dotenv
load_dotenv()
from chromadb.utils import embedding_functions
from langchain_core.tools import tool

CHROMA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "my_extensions", "chroma_db"
)

INTERNAL_DOC_SEARCH_DESCRIPTION = (
    "Search the company's internal document knowledge base (company introductions, "
    "product descriptions, industry reports, meeting notes, investment memos). "
    "Use this INSTEAD of web search when the question is about internal company "
    "opinions, internal analysis, or company/product background that would not be "
    "publicly available online. Returns relevant text snippets with their source "
    "document. If no clearly relevant results are found, say so honestly rather "
    "than guessing."
)


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
        import asyncio
        results = await asyncio.to_thread(_run_search)

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

    except Exception as e:
        return f"Document search error: {type(e).__name__}: {str(e)}"
