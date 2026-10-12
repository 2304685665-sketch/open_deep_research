"""Agent 评测用例与判分器的离线测试：用例期望值与独立 SQL 一致；判分器对标准答案全过、对被篡改答案全不过
（这是对判分器本身的审计——判分器的错误会让通过率失真）。无任何模型调用。"""

import os
import shutil
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent_eval_cases as ac  # noqa: E402
import agent_eval_score as sc  # noqa: E402


def test_money_formats():
    for t in ("45,000,000 美元", "$45,000,000", "$45M", "45 million", "4500万美元", "0.45亿", "45000000.0"):
        assert any(abs(v - 45_000_000) < 1 for v in sc.parse_money_mentions(t)), (t, sc.parse_money_mentions(t))
    assert sc.parse_money_mentions("日期 2026-03-15") == [], "日期里的数字不是金额"
    print("✅ test_money_formats 通过")


def test_counts_and_dates():
    assert sc.parse_counts("一共融资了 2 轮") == [2] and sc.parse_counts("共三轮融资") == [3]
    assert sc.parse_counts("十二笔") == [12] and sc.parse_counts("日期2026年3月15日") == []
    assert sc.parse_dates("2026年6月20日 / 2026-06-20 / 2026/6/20") == ["2026-06-20"] * 3
    print("✅ test_counts_and_dates 通过")


def test_zero_count_does_not_match_numbers_ending_in_zero():
    z = {"kind": "count", "value": 0}
    assert sc.score_expected(z, "2027 年以后有 0 笔融资") and sc.score_expected(z, "没有任何融资记录")
    assert not sc.score_expected(z, "共有 10 笔融资") and not sc.score_expected(z, "共 3.0 万美元")
    print("✅ test_zero_count_does_not_match_numbers_ending_in_zero 通过")


def test_entity_first_mention_rule():
    e = {"kind": "entity", "value": "SynthMotion", "candidates": ["RoboWorks", "SynthMotion"]}
    assert sc.score_expected(e, "累计最多的是 SynthMotion，其次是 RoboWorks")
    assert not sc.score_expected(e, "RoboWorks 不是最多的，SynthMotion 才是")  # 保守：先提到别家则判错
    assert not sc.score_expected(e, "是 RoboWorks")
    print("✅ test_entity_first_mention_rule 通过")


def test_no_answer_rules():
    n = {"kind": "no_answer"}
    assert sc.score_expected(n, "数据库中没有 OmegaBot 的记录，无法回答。")
    assert not sc.score_expected(n, "OmegaBot 最新融资 3000 万美元")  # 编造数字
    assert not sc.score_expected(n, "没有找到相关记录，但估计约 2000万美元")  # 拒答又编数
    assert not sc.score_expected(n, "这家公司很有前景。")  # 没拒答
    print("✅ test_no_answer_rules 通过")


def test_cases_match_independent_sql_and_are_not_hardcoded():
    cases = ac.build_cases()
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)) and len(cases) >= 25, len(cases)
    c = sqlite3.connect(f"file:{ac.DEFAULT_DB}?mode=ro", uri=True)
    by = {x["id"]: x for x in cases}
    assert abs(by["total-all"]["expected"]["value"] - c.execute("select sum(amount_usd) from funding_rounds").fetchone()[0]) < 1e-6
    assert by["since2026-count"]["expected"]["value"] == c.execute("select count(*) from funding_rounds where round_date>='2026-01-01'").fetchone()[0]
    print(f"✅ test_cases_match_independent_sql_and_are_not_hardcoded 通过（{len(cases)} 条）")


def test_expected_follow_the_database_when_data_changes():
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "x.db")
        shutil.copy(ac.DEFAULT_DB, db)
        w = sqlite3.connect(db)
        w.execute("INSERT INTO funding_rounds(company_id,round_type,amount_usd,round_date) VALUES (1,'Series D',999000000,'2026-09-01')")
        w.commit(); w.close()
        base = {x["id"]: x for x in ac.build_cases()}
        new = {x["id"]: x for x in ac.build_cases(db)}
        assert new["total-all"]["expected"]["value"] == base["total-all"]["expected"]["value"] + 999_000_000
        assert new["count-RoboWorks"]["expected"]["value"] == base["count-RoboWorks"]["expected"]["value"] + 1
    print("✅ test_expected_follow_the_database_when_data_changes 通过")


def _gold_answer(case):
    def render(e):
        k = e["kind"]
        if k == "all":
            return "；".join(render(p) for p in e["parts"])
        if k == "money":
            return f"{e['value']:,.0f} 美元"
        if k == "count":
            return f"{e['value']} 轮" if e["value"] else "没有任何融资"
        if k == "date":
            return e["value"]
        if k in ("text", "entity"):
            return e["value"]
        return "数据库中没有相关记录，无法回答。"
    return render(case["expected"])


def _corrupt(case):
    e = case["expected"]
    def bad(e):
        k = e["kind"]
        if k == "all":
            return "；".join(bad(p) for p in e["parts"])
        if k == "money":
            return f"{e['value'] * 1.5 + 12345:,.0f} 美元"
        if k == "count":
            return f"{e['value'] + 3} 轮"
        if k == "date":
            return "1999-01-01"
        if k == "text":
            return "未知"
        if k == "entity":
            other = [c for c in e["candidates"] if c != e["value"]][0]
            return other
        return "OmegaBot 融资 8800 万美元"
    return bad(e)


def test_scorer_audit_gold_passes_corrupted_fails():
    cases = ac.build_cases()
    for c in cases:
        assert sc.score_case(c, _gold_answer(c)), ("gold 应通过", c["id"], _gold_answer(c))
        assert not sc.score_case(c, _corrupt(c)), ("篡改应失败", c["id"], _corrupt(c))
    print(f"✅ test_scorer_audit_gold_passes_corrupted_fails 通过（{len(cases)} 条）")


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
