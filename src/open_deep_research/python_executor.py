"""python_executor 工具 —— DataAnalyst 专精工具之二。

设计理念：让模型自己写 pandas 代码去分析数据（比如算平均值、算增长率），
而不是提前写死计算逻辑。

安全边界（已知简化，非生产级）：
- 用 exec() 执行，没有真正的进程/容器级隔离
- 限制暴露给代码的命名空间，只给 pandas，不给 os/subprocess 等系统访问能力
- 生产环境应换成真正的沙箱（Docker容器、E2B等），这是明确的已知待办
"""

import io
import asyncio
import contextlib
import pandas as pd
from langchain_core.tools import tool

PYTHON_EXECUTOR_DESCRIPTION = (
    "Execute Python (pandas) code for data analysis. "
    "The variable pd (pandas) is available. "
    "If you need data to analyze, first query it via sql_executor and "
    "construct a DataFrame from the result yourself. "
    "Use print() to output what you want to see -- only printed output "
    "is returned to you. "
    "Only pandas is available; no file system or network access."
)


@tool(description=PYTHON_EXECUTOR_DESCRIPTION)
async def python_executor(code: str) -> str:
    """Execute Python/pandas code and return captured print output.

    Args:
        code: Python code using pandas (as pd). Must use print() to
              produce visible output.

    Returns:
        Captured stdout, or the raw error message if execution failed.
    """
    # 加固：允许 import，但限制只能导入白名单里的模块（比如 pandas），
    # 拦住 __import__("os") 这类试图访问系统模块的调用。
    # 仍不构成可靠沙箱——只是比之前收紧一点，这是应用层限制，不是真正隔离。
    ALLOWED_MODULES = {"pandas", "math", "statistics"}

    def restricted_import(name, *args, **kwargs):
        if name not in ALLOWED_MODULES:
            raise ImportError(f"Importing '{name}' is not allowed. Only {ALLOWED_MODULES} are permitted.")
        return __import__(name, *args, **kwargs)

    safe_builtins = {
        "print": print, "len": len, "range": range, "str": str, "int": int,
        "float": float, "list": list, "dict": dict, "__import__": restricted_import,
    }
    safe_globals = {"pd": pd, "__builtins__": safe_builtins}

    def _run_code():
        output_buffer = io.StringIO()
        with contextlib.redirect_stdout(output_buffer):
            exec(code, safe_globals)
        result = output_buffer.getvalue()
        if not result.strip():
            return "Code executed successfully but produced no printed output. Use print() to see results."
        return result

    try:
        # 同样的原因：exec() 是同步阻塞调用，丢到独立线程跑，
        # 避免占用主事件循环
        return await asyncio.to_thread(_run_code)
    except Exception as e:
        return f"Python Error: {type(e).__name__}: {str(e)}"
