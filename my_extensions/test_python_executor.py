"""python_executor 工具的单元测试。"""

import asyncio
import sys
sys.path.insert(0, "src/open_deep_research")
from python_executor import python_executor


async def test_basic_calculation():
    """基本的 pandas 计算能力，比如算一组数字的平均值。"""
    code = """
import pandas as pd
data = {"amount": [45000000, 15000000, 12000000, 80000000, 5000000]}
df = pd.DataFrame(data)
print(df["amount"].mean())
"""
    result = await python_executor.ainvoke({"code": code})
    assert "31400000" in result
    print("test_basic_calculation 通过")


async def test_no_print_gives_hint():
    """代码跑了但没有 print，应该提示用户要用 print 才能看到结果，而不是返回空字符串。"""
    code = "x = 1 + 1"
    result = await python_executor.ainvoke({"code": code})
    assert "print" in result.lower()
    print("test_no_print_gives_hint 通过")


async def test_syntax_error_returned_raw():
    """代码写错了，报错应该原样返回，不是吞掉或者假装成功。"""
    code = "print(1 +"
    result = await python_executor.ainvoke({"code": code})
    assert "Python Error" in result
    print("test_syntax_error_returned_raw 通过")


async def test_os_import_now_blocked():
    """验证加固后：之前能用 __import__("os") 绕过的路径，现在被白名单拦住了。"""
    code = """
os_module = __import__("os")
print(os_module.getcwd())
"""
    result = await python_executor.ainvoke({"code": code})
    assert "Python Error" in result
    assert "not allowed" in result
    print("test_os_import_now_blocked 通过（之前能绕过的路径，现在被正确拦住了）")


async def test_allowed_module_still_works():
    """确认加固没有误伤正常需要的模块，比如 math。"""
    code = """
import math
print(math.sqrt(16))
"""
    result = await python_executor.ainvoke({"code": code})
    assert "4.0" in result
    print("test_allowed_module_still_works 通过（白名单里的模块依然可用）")


async def test_security_boundary_still_incomplete_in_other_ways():
    """诚实测试：白名单挡住了 os，但没有做资源限制（内存、CPU、执行时间），
    也没有做真正的进程隔离——这条记录这个仍然存在的局限，不代表当前风险已解决。"""
    code = """
# 没有超时限制，理论上一个死循环会一直占用资源，这里不真的跑死循环，
# 只是记录这个局限存在，作为已知待办
print("no timeout / resource limit enforced -- known gap")
"""
    result = await python_executor.ainvoke({"code": code})
    assert "known gap" in result
    print("test_security_boundary_still_incomplete_in_other_ways 通过（这是个记录性质的测试，提醒还有资源限制没做）")


async def main():
    await test_basic_calculation()
    await test_no_print_gives_hint()
    await test_syntax_error_returned_raw()
    await test_os_import_now_blocked()
    await test_allowed_module_still_works()
    await test_security_boundary_still_incomplete_in_other_ways()
    print("\n全部测试通过。os 访问已被拦住，但资源/时间限制仍是已知缺口。")


asyncio.run(main())
