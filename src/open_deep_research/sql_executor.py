"""sql_executor 工具 —— DataAnalyst 专精工具之一。

设计理念：不做 RAG、不建语义层。
不提前告诉模型有哪些表、哪些字段是什么意思，模型必须自己先探索
schema（比如查 sqlite_master），再写查询；报错信息原样返回给模型，
不做任何"友好化"处理，让模型自己看懂错误、自己改。

安全边界（2026-10-09 加固版本，2026-10-09 第二次修复：ATTACH 漏洞）：
- 真正的只读边界是数据库连接本身，不是 SQL 字符串前缀判断，但这个
  "连接本身的只读边界"由两层组成，缺一不可——这是一次真实审查中
  漏判后修复才确认下来的，不是一次性设计对的：
    1. `mode=ro` 打开连接（`sqlite3.connect(uri, uri=True)`）+
       `PRAGMA query_only = ON`：拦住对主数据库、以及任何被 ATTACH
       进来的数据库的写 SQL（已实测：
       `sqlite3.OperationalError: attempt to write a readonly database`）。
    2. `conn.set_authorizer(...)` 对 `SQLITE_ATTACH`/`SQLITE_DETACH`
       显式返回 `SQLITE_DENY`：**这一层是必须的**。之前以为
       `mode=ro` 就足够，但实测发现 `ATTACH DATABASE 'evil.db' AS e`
       能绕过 `mode=ro`（甚至 `PRAGMA query_only = ON`）在磁盘上真的
       创建出一个新的数据库文件——因为 ATTACH 打开的是一个全新的、
       独立的数据库文件/连接，根本不受主连接这个 `mode=ro` URI 参数
       的约束。只有 authorizer 显式拒绝 ATTACH/DETACH 这两个动作本身，
       才能真正挡住这条路径（已实测：单独开 query_only 不装
       authorizer，ATTACH 依然成功创建文件；只装 authorizer 不开
       query_only，ATTACH 被正确拒绝）。细节见 `_deny_attach_and_detach`
       函数的 docstring。
  字符串检查只是第一道、更友好的过滤，不是唯一的安全边界——这条原则
  没变，但"数据库层"本身的真正含义被这次修复重新校准过。
- 只允许 SELECT 语句。之前的版本在工具描述里写着"PRAGMA 也允许"，
  但实际代码早就只放行 select 开头的语句——这个措辞不一致已经修正，
  PRAGMA 继续被拒绝（部分 PRAGMA 能持久化修改数据库文件/设置，
  不是真正的只读操作）。
- 暂不支持 WITH（CTE）。原因：SQLite 允许 `WITH ... AS (...) DELETE/
  INSERT/UPDATE ...`——WITH 子句可以出现在写操作前面，不只能出现在
  SELECT 前面（已实测确认：`WITH c AS (SELECT 1) DELETE FROM t` 在
  可写连接上真的会删除数据）。要准确区分"安全的 WITH...SELECT"和
  "伪装成 CTE 的写操作"，光靠字符串/正则无法可靠做到（需要正确处理
  嵌套括号、多个 CTE、字符串字面量里的关键字等），这正是需要一个真正
  SQL 解析器才能可靠解决的问题。本次没有引入新的 SQL 解析依赖，所以
  选择暂不支持 WITH，而不是用不可靠的正则去"大概"区分——这是刻意的
  安全取舍，不是遗漏。即使字符串检查这里出于某种原因失效，mode=ro
  连接仍然会在真正执行写操作时报错，但工具描述和校验逻辑应该准确
  反映"目前不支持"这一事实，而不是假装支持。
- 查询类型判断会先剥掉开头的空白、`--` 行注释、`/* */` 块注释，
  再看真正的第一个关键字是什么——这是为了避免"注释前缀"把判断搞错
  （不管是误判成允许，还是误判成拒绝），不是只做最简单的
  `startswith("select")`。剥离只用于分类判断，真正执行时永远用
  原始、未经修改的 query 字符串。
- 查询执行时间通过 `sqlite3.Connection.set_progress_handler()` 做
  best-effort 超时控制：每隔约 1000 条虚拟机指令检查一次是否已经超过
  配置的秒数，超过则中断查询。这不是硬实时保证——两次检查之间的那一段
  查询仍然会执行完，所以实际中断时间点可能比设定的秒数略晚，但已实测
  确认能在一个耗时数秒的查询执行到一半时就把它打断，不会让查询无限跑
  下去。
- 返回行数通过 `cur.fetchmany(MAX_ROWS + 1)` 限制，不再无上限调用
  `fetchall()`。超过上限时明确提示"结果已截断"。注意：这只限制了
  行数，不限制单行里某个字段本身的字节大小——如果某一行里塞进一个
  巨大的字符串/BLOB，这里不会特别处理，这是已知的、没有在本轮解决的
  缺口，不要误以为行数限制等于响应体积限制。
- 数据库文件不存在时，会先显式检查并返回明确错误，不会创建新文件；
  即使跳过这个检查，`mode=ro` 本身在文件不存在时也会直接报错而不是
  创建空数据库（已实测确认）——两层都不依赖对方单独生效。
"""

import asyncio
import re
import sqlite3
import time
from pathlib import Path

from langchain_core.tools import tool

DB_PATH = Path(__file__).resolve().parents[2] / "my_extensions" / "demo.db"

QUERY_TIMEOUT_SECONDS = 5  # best-effort，不是硬实时保证，见模块docstring
MAX_ROWS = 500
PROGRESS_HANDLER_N_INSTRUCTIONS = 1000  # 每隔这么多条虚拟机指令检查一次超时

# 用于分类查询类型：先剥掉开头的空白/注释，再判断真正的第一个关键字
_LEADING_NOISE_RE = re.compile(r"\A(\s+|--[^\n]*\n?|/\*.*?\*/)+", re.DOTALL)
_SELECT_RE = re.compile(r"select\b", re.IGNORECASE)
_WITH_RE = re.compile(r"with\b", re.IGNORECASE)

SQL_EXECUTOR_DESCRIPTION = (
    "Execute a READ-ONLY SQL query against the internal demo database (SQLite). "
    "Only SELECT statements are allowed -- PRAGMA, WITH/CTE queries, and any "
    "write operation (INSERT/UPDATE/DELETE/DROP/etc.) are all rejected. "
    "The underlying database connection itself is opened read-only (SQLite "
    "URI mode=ro), so this is enforced at the database layer, not merely by "
    "inspecting your query text. "
    f"Results are capped at {MAX_ROWS} rows, and a query will be interrupted "
    f"if it runs for roughly more than {QUERY_TIMEOUT_SECONDS} seconds "
    "(best-effort, not a hard guarantee). "
    "You do NOT know the schema in advance -- if you need to know what tables "
    "or columns exist, query `sqlite_master` first (e.g. "
    "\"SELECT name, sql FROM sqlite_master WHERE type='table'\"). "
    "If your query fails, the raw database error will be returned to you -- "
    "read it carefully and correct your next query accordingly."
)


def _classify_query(query: str) -> str:
    """返回 'select' / 'with' / 'other'。

    只用来做分类判断（剥掉开头注释和空白看真正的第一个关键字），
    不会修改、也不会被用来替代真正传给 sqlite3 执行的原始 query。
    """
    core = _LEADING_NOISE_RE.sub("", query, count=1)
    if _SELECT_RE.match(core):
        return "select"
    if _WITH_RE.match(core):
        return "with"
    return "other"


def _build_readonly_uri(path: Path) -> str:
    return f"{path.as_uri()}?mode=ro"


def _deny_attach_and_detach(action, arg1, arg2, dbname, source):
    """SQLite authorizer callback，只拒绝 ATTACH/DETACH，其余一切放行。

    2026-10-09 安全修复：实测发现 `mode=ro` 只保护"主数据库连接"本身，
    并不能阻止 `ATTACH DATABASE 'evil.db' AS e` —— ATTACH 打开的是一个
    全新的、独立的数据库文件/连接，不受主连接 mode=ro 这个 URI 参数的
    约束。复现过的真实后果：在一个临时目录里直接调用
    `_run_query_sync("ATTACH DATABASE 'evil.db' AS e")`，返回
    "Query executed successfully"，且磁盘上真的创建出了 evil.db。

    `PRAGMA query_only = ON` 同样不够：实测确认它只拦截针对
    主/已挂载数据库的写 SQL（INSERT/UPDATE/DELETE等），不拦截 ATTACH/
    DETACH 这两个命令本身——单独开启 query_only、不装这个 authorizer，
    ATTACH 依然会成功并创建文件。

    唯一经实测确认能挡住 ATTACH 创建文件的，是在 SQLite 的 authorizer
    层显式对 SQLITE_ATTACH/SQLITE_DETACH 返回 SQLITE_DENY（单独测试：
    只装 authorizer、不开 query_only，ATTACH 同样会被挡住，文件不会被
    创建）。所以这里的 authorizer 是真正解决这个漏洞的那一层，
    query_only 是额外保留的纵深防御，不是同一件事的两种写法。
    """
    if action in (sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH):
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def _run_query_sync(query: str, timeout_seconds: float | None = None) -> str:
    """真正执行查询的同步函数（在独立线程里跑，见下方 asyncio.to_thread）。

    数据库层面的只读保护在这里生效，是真正的安全边界——这个函数本身
    不会、也不能对数据库做任何写操作，不依赖调用方有没有做过字符串
    检查。这个"只读保护"目前由两层组成，缺一不可：
      1. 连接本身用 mode=ro 打开 + `PRAGMA query_only = ON`——拦住对
         主数据库（以及任何被 ATTACH 进来的数据库）的写 SQL。
      2. `conn.set_authorizer(...)` 对 SQLITE_ATTACH/SQLITE_DETACH 显式
         返回 SQLITE_DENY——拦住 ATTACH/DETACH 这两个命令本身。
         这一层是必须的：mode=ro 和 query_only 都不会阻止 ATTACH 打开
         或创建一个全新的、独立的数据库文件（已实测确认，见
         `_deny_attach_and_detach` 的 docstring）。

    authorizer 必须在 `cur.execute(query)` 之前完成注册，否则对这次
    查询不生效。

    `timeout_seconds` 留空时使用模块级常量 QUERY_TIMEOUT_SECONDS（正常
    调用路径，即 sql_executor 工具本身，永远不传这个参数）。之所以做成
    显式参数而不是让测试去 monkeypatch 模块级的 QUERY_TIMEOUT_SECONDS，
    是为了彻底避免"共享可变状态"——如果测试靠改全局变量来模拟短超时，
    理论上会跟其他并发跑着的真实调用互相干扰（即便当前测试都是顺序执行，
    没有真的触发这个问题，但这个模式本身就是隐患，不该依赖"恰好没撞上"）。
    每次调用的 deadline 都是这个函数内部的局部变量，不同调用之间互不影响。
    """
    db_path = DB_PATH
    if not db_path.exists():
        return (
            f"SQL Error: database file not found at {db_path}. "
            "Refusing to connect -- a read-only connection to a missing file "
            "would either fail or (without mode=ro) could create an empty "
            "database file, which this tool must not do."
        )

    effective_timeout = QUERY_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds

    conn = None
    timed_out = False
    try:
        conn = sqlite3.connect(_build_readonly_uri(db_path), uri=True)
        conn.execute("PRAGMA query_only = ON")
        conn.set_authorizer(_deny_attach_and_detach)

        deadline = time.monotonic() + effective_timeout

        def _progress_handler():
            nonlocal timed_out
            if time.monotonic() > deadline:
                timed_out = True
                return 1  # 非 0 返回值会让 SQLite 中断当前正在执行的语句
            return 0

        conn.set_progress_handler(_progress_handler, PROGRESS_HANDLER_N_INSTRUCTIONS)

        cur = conn.cursor()
        cur.execute(query)
        columns = [desc[0] for desc in cur.description] if cur.description else []
        rows = cur.fetchmany(MAX_ROWS + 1)
        truncated = len(rows) > MAX_ROWS
        if truncated:
            rows = rows[:MAX_ROWS]

        if not rows:
            return "Query executed successfully but returned no rows."

        header = " | ".join(columns)
        body = "\n".join(" | ".join(str(v) for v in row) for row in rows)
        result = f"{header}\n{body}"
        if truncated:
            result += (
                f"\n... [result truncated at {MAX_ROWS} rows; the query "
                "matched more rows than this limit]"
            )
        return result
    except sqlite3.Error as e:
        if timed_out or "interrupted" in str(e).lower():
            return (
                "SQL Error: query timed out after approximately "
                f"{effective_timeout} seconds and was interrupted. "
                "(best-effort cutoff via SQLite's progress handler -- not a "
                "hard real-time guarantee, but it does reliably stop "
                "runaway queries rather than letting them run forever)"
            )
        return f"SQL Error: {e}"
    finally:
        if conn is not None:
            conn.close()


@tool(description=SQL_EXECUTOR_DESCRIPTION)
async def sql_executor(query: str) -> str:
    """Execute a read-only SQL query and return the raw result or raw error.

    Args:
        query: A single SELECT statement to execute.

    Returns:
        The query result formatted as rows of text, or an error message
        if execution failed or the query was not a read-only SELECT.
    """
    if not isinstance(query, str) or not query.strip():
        return "SQL Error: query must be a non-empty string."

    classification = _classify_query(query)
    if classification == "with":
        return (
            "SQL Error: WITH (CTE) queries are not currently supported by "
            "this tool. SQLite allows a WITH clause to prefix INSERT/UPDATE/"
            "DELETE statements, not only SELECT, and this tool cannot yet "
            "reliably distinguish a safe 'WITH ... SELECT' from a disguised "
            "write without a full SQL parser -- rewrite your query as a "
            "plain SELECT (e.g. inline the CTE as a subquery)."
        )
    if classification != "select":
        return (
            "SQL Error: only SELECT statements are permitted. "
            "Write operations are not allowed through this tool."
        )

    try:
        # 用 asyncio.to_thread 把同步的 sqlite3 调用丢到独立线程跑，
        # 不阻塞主事件循环。
        return await asyncio.to_thread(_run_query_sync, query)
    except sqlite3.Error as e:
        return f"SQL Error: {str(e)}"
