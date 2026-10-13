"""评分器 v3（事后版本，2026-10-13）：在 v2 基础上只修一类**用例定义缺陷**，对所有运行臂一视同仁。

背景：`L-largest-round-founded` 的标准答案是一个年份（2014），但用例把它声明成 `count` 类型，而计数解析器不识别
“2014年”这类年份写法，于是所有臂（基线、D1–D4）即使答对也被判错。这是用例/评分器缺陷，不是模型错误。
修复规则（原则性，不针对单条用例）：期望值是 1900–2100 的整数时，视为“年份答案”：答案文本中该年份必须作为独立
四位数出现，且它必须是答案里出现次数最多的年份（并列也算），避免报告随手提到别的年份时误判。
**没有改动**的：no_answer 的金额匹配（`L-absent-plan` 报告里“8000万至9000万”这类近似数仍判编造）、数值容差、
python_stats 题（`L-stats-pstdev` 的失败是模型心算出错，评分正确）。v1 仍是预注册分数，v2/v3 均为事后修正，需分开披露。"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

import agent_eval_score as v1
import agent_eval_score_v2 as v2

_YEAR_RE = re.compile(r"(?<!\d)(19\d\d|20\d\d|2100)(?!\d)")


def is_year_expected(case: dict[str, Any]) -> bool:
    e = case["expected"]
    return e.get("kind") == "count" and isinstance(e.get("value"), int) and 1900 <= e["value"] <= 2100


def score_year(expected_year: int, answer: str) -> bool:
    years = [int(y) for y in _YEAR_RE.findall(answer or "")]
    if expected_year not in years:
        return False
    c = Counter(years)
    return c[expected_year] == max(c.values())


def score_case_v3(case: dict[str, Any], answer: str, db_unchanged: bool, known: list[float]) -> bool:
    if is_year_expected(case):
        return score_year(case["expected"]["value"], answer)
    return v2.score_case_v2(case, answer, db_unchanged, known)
