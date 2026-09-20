"""memory_manager.py —— 跨会话记忆的读写封装。

设计原则（跟 sql_executor/internal_doc_search 保持一致的独立可测试性）：
- 不直接嵌入图节点内部写死逻辑，单独封装成两个函数，方便脱离图独立测试
- user_id 用于隔离不同用户的记忆，不是可选参数
- 读写失败不应该让整个研究流程崩溃——记忆是辅助能力，不是核心链路，
  失败时应该降级（不影响主流程），不能因为记忆模块出问题导致整个任务失败
"""

import os
from dotenv import load_dotenv
load_dotenv()

from mem0 import Memory

_config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini",
            "temperature": 0.1,
        },
    },
}

_memory_instance = None


def _get_memory() -> Memory:
    """懒加载单例，避免每次调用都重新初始化 Memory 对象（涉及连接本地向量库）。"""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = Memory.from_config(_config)
    return _memory_instance


async def save_memory(text: str, user_id: str) -> dict:
    """存一条记忆。失败时不抛异常，返回一个标记失败的字典，让调用方决定
    要不要继续主流程（设计原则：记忆写入失败不应该阻塞研究任务本身）。
    """
    import asyncio

    def _run():
        m = _get_memory()
        return m.add(text, user_id=user_id)

    try:
        result = await asyncio.to_thread(_run)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {str(e)}"}


async def search_memory(query: str, user_id: str, limit: int = 3) -> dict:
    """检索跟当前问题相关的历史记忆。失败时返回空列表 + 失败标记，
    不阻塞主流程——检索不到记忆，应该表现得跟"这个用户没有历史记忆"一样，
    而不是让整个任务报错。
    """
    import asyncio

    def _run():
        m = _get_memory()
        return m.search(query, filters={"user_id": user_id}, limit=limit)

    try:
        result = await asyncio.to_thread(_run)
        memories = result.get("results", []) if isinstance(result, dict) else result
        return {"success": True, "memories": memories}
    except Exception as e:
        return {"success": False, "memories": [], "error": f"{type(e).__name__}: {str(e)}"}
