"""sql_executor 工具 —— DataAnalyst 专精工具之一。

设计理念：不做 RAG、不建语义层。
不提前告诉模型有哪些表、哪些字段是什么意思，模型必须自己先探索
schema（比如查 sqlite_master），再写查询；报错信息原样返回给模型，
不做任何"友好化"处理，让模型自己看懂错误、自己改。

安全限制：只允许 SELECT / PRAGMA 这类只读查询，拒绝任何会修改
或删除数据的语句（INSERT/UPDATE/DELETE/DROP 等），防止模型生成
的 SQL 意外破坏数据。
"""

import sqlite3
import asyncio
from langchain_core.tools import tool

SQL_EXECUTOR_DESCRIPTION = (
    "Execute a READ-ONLY SQL query against the internal demo database (SQLite). "
    "Only SELECT and PRAGMA statements are allowed — write operations "
    "(INSERT/UPDATE/DELETE/DROP/etc.) will be rejected. "
    "You do NOT know the schema in advance — if you need to know what tables "
    "or columns exist, query `sqlite_master` first (e.g. "
    "\"SELECT name, sql FROM sqlite_master WHERE type='table'\"). "
    "If your query fails, the raw database error will be returned to you — "
    "read it carefully and correct your next query accordingly."
)

from pathlib import Path
DB_PATH = str(Path(__file__).resolve().parents[2] / "my_extensions" / "demo.db")


@tool(description=SQL_EXECUTOR_DESCRIPTION)
async def sql_executor(query: str) -> str:
    """Execute a read-only SQL query and return the raw result or raw error.

    Args:
        query: A single SELECT or PRAGMA statement to execute.

    Returns:
        The query result formatted as rows of text, or an error message
        if execution failed or the query was not a read-only statement.
    """
    stripped = query.strip().lower()
    if not (stripped.startswith("select") or stripped.startswith("pragma")):
        return (
            "SQL Error: only SELECT and PRAGMA statements are permitted. "
            "Write operations are not allowed through this tool."
        )

    def _run_query():
        conn = sqlite3.connect(DB_PATH)
        try:
            cur = conn.cursor()
            cur.execute(query)
            columns = [desc[0] for desc in cur.description] if cur.description else []
            rows = cur.fetchall()
            if not rows:
                return "Query executed successfully but returned no rows."
            header = " | ".join(columns)
            body = "\n".join(" | ".join(str(v) for v in row) for row in rows)
            return f"{header}\n{body}"
        finally:
            conn.close()

    try:
        # 用 asyncio.to_thread 把同步的 sqlite3 调用丢到独立线程跑，
        # 不阻塞主事件循环——之前 langgraph dev 启动时反复警告的
        # "blocking I/O" 问题，这里正式修掉
        return await asyncio.to_thread(_run_query)
    except sqlite3.Error as e:
        return f"SQL Error: {str(e)}"
