"""sql_executor 工具的单元测试 —— 用 assert 自动判断结果，不靠肉眼看输出。"""

import asyncio
import sys
sys.path.insert(0, 'src/open_deep_research')
from sql_executor import sql_executor


async def test_error_on_wrong_table():
    """查一个不存在的表，应该拿到原始报错，而不是崩溃或者返回空结果。"""
    result = await sql_executor.ainvoke({'query': 'SELECT * FROM company'})
    assert "SQL Error" in result
    assert "no such table" in result
    print("✅ test_error_on_wrong_table 通过")


async def test_schema_discovery():
    """查 sqlite_master，应该能看到 companies 和 funding_rounds 这两张表。"""
    result = await sql_executor.ainvoke({
        'query': "SELECT name, sql FROM sqlite_master WHERE type='table'"
    })
    assert "companies" in result
    assert "funding_rounds" in result
    print("✅ test_schema_discovery 通过")


async def test_correct_join_query():
    """基于真实 schema 写出的连表查询，应该能正确返回融资金额前三的公司。"""
    result = await sql_executor.ainvoke({'query': '''
        SELECT c.name, f.round_type, f.amount_usd
        FROM companies c JOIN funding_rounds f ON c.id = f.company_id
        ORDER BY f.amount_usd DESC LIMIT 3
    '''})
    assert "SynthMotion" in result
    assert "80000000" in result
    print("✅ test_correct_join_query 通过")


async def test_write_operation_rejected():
    """故意传一条 DELETE 语句，应该被拒绝，数据不应该被真的删除。"""
    result = await sql_executor.ainvoke({'query': 'DELETE FROM companies WHERE id = 1'})
    assert "only SELECT" in result

    check = await sql_executor.ainvoke({'query': 'SELECT name FROM companies WHERE id = 1'})
    assert "RoboWorks" in check
    print("✅ test_write_operation_rejected 通过（DELETE 被拒绝，数据完好）")


async def test_pragma_now_rejected():
    """加固后追加：PRAGMA 语句应该被拒绝，因为部分 PRAGMA（如 user_version）
    能对数据库文件做持久性写入，不是真正的只读操作。"""
    result = await sql_executor.ainvoke({'query': "PRAGMA user_version = 999"})
    assert "SQL Error" in result
    assert "only SELECT" in result
    print("✅ test_pragma_now_rejected 通过（PRAGMA 被正确拒绝）")


async def main():
    await test_error_on_wrong_table()
    await test_schema_discovery()
    await test_correct_join_query()
    await test_write_operation_rejected()
    await test_pragma_now_rejected()
    print("\n全部测试通过！")


asyncio.run(main())
