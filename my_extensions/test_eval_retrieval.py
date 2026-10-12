"""eval_retrieval 的离线测试：指标计算是否正确，以及查询集标注本身是否有效。
不调用任何 API，也不检查任何检索效果。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_retrieval as ev  # noqa: E402

CHUNKS = ev.load_kb_chunks()
QUERIES = ev.load_queries()


def test_real_kb_has_expected_shape():
    assert len(CHUNKS) == 25, len(CHUNKS)
    assert len({c["id"] for c in CHUNKS}) == 25
    assert all(c["section"] != "开头" for c in CHUNKS)
    print("✅ test_real_kb_has_expected_shape 通过（25 个 chunk）")


def test_every_target_exists_in_the_index():
    existing = {(c["doc_id"], c["section"]) for c in CHUNKS}
    for q in QUERIES:
        for d, s in q["targets"]:
            assert (d, s) in existing, f"{q['id']} 的标注目标不在索引里：{d} / {s}"
    print("✅ test_every_target_exists_in_the_index 通过（所有标注目标都能在 25 个 chunk 里找到）")


def test_query_set_structure_and_reasons():
    ids = [q["id"] for q in QUERIES]
    assert len(ids) == len(set(ids)), "查询 id 必须唯一"
    allowed = {"entity", "semantic", "keyword", "no_answer"}
    for q in QUERIES:
        assert q["type"] in allowed, q["id"]
        assert q["query"].strip() and q["reason"].strip(), f"{q['id']} 缺少查询文本或标注理由"
        if q["type"] == "no_answer":
            assert q["targets"] == [], f"{q['id']} 无答案类不应有目标"
        else:
            assert q["targets"], f"{q['id']} 有答案类必须有目标"
    types = {t: sum(1 for q in QUERIES if q["type"] == t) for t in allowed}
    assert all(types[t] >= 5 for t in allowed), types
    print(f"✅ test_query_set_structure_and_reasons 通过（{types}）")


def test_first_hit_rank_and_metrics_hand_computed():
    qs = [
        {"id": "a", "type": "entity", "query": "x", "targets": [["doc01_synthmotion_intro", "融资历程"]], "reason": "r"},
        {"id": "b", "type": "semantic", "query": "x", "targets": [["doc05_industry_report", "地域分布"]], "reason": "r"},
        {"id": "c", "type": "keyword", "query": "x", "targets": [["doc03_roboworks_intro", "技术路线"]], "reason": "r"},
        {"id": "n", "type": "no_answer", "query": "x", "targets": [], "reason": "r"},
    ]
    t = {q["id"]: ev.target_chunk_ids(q, CHUNKS) for q in qs}
    assert all(len(t[k]) == 1 for k in "abc")
    a, b, c = (next(iter(t[k])) for k in "abc")
    other = next(x["id"] for x in CHUNKS if x["id"] not in {a, b, c})
    rankings = {"a": [a, other], "b": [other, b], "c": [other, c.replace("chunk", "chunkX")], "n": [other]}
    res = ev.evaluate(rankings, qs, CHUNKS)
    # a: 第1名 -> rr=1；b: 第2名 -> rr=0.5；c: 未命中 -> 0；无答案不计入
    assert res["overall"]["n"] == 3
    assert abs(res["overall"]["hit1"] - 1 / 3) < 1e-9
    assert abs(res["overall"]["mrr"] - (1 + 0.5 + 0) / 3) < 1e-9
    assert res["by_type"]["entity"] == {"n": 1, "hit1": 1.0, "mrr": 1.0}
    assert res["by_type"]["semantic"]["mrr"] == 0.5
    assert "no_answer" not in res["by_type"]
    print("✅ test_first_hit_rank_and_metrics_hand_computed 通过")


def test_multi_target_query_hits_on_any_target():
    q = {"id": "m", "type": "semantic", "query": "x",
         "targets": [["doc02_synthmotion_product", "技术布局"], ["doc08_meeting_notes_synthmotion", "待观察风险"]],
         "reason": "r"}
    ids = ev.target_chunk_ids(q, CHUNKS)
    assert len(ids) == 2
    assert ev.first_hit_rank(["zzz", sorted(ids)[1], sorted(ids)[0]], ids) == 2
    print("✅ test_multi_target_query_hits_on_any_target 通过")


class _FakeCollection:
    def __init__(self, ids):
        self._ids = ids
        self.calls = []

    def count(self):
        return len(self._ids)

    def query(self, query_texts, n_results):
        self.calls.append((query_texts, n_results))
        return {"ids": [self._ids[:n_results]]}


def test_vector_candidate_pool_is_capped_and_never_exceeds_collection_size():
    small = _FakeCollection([f"c{i}" for i in range(25)])
    assert len(ev.vector_rank_ids(small, "q")) == 25 and small.calls[0][1] == 25
    big = _FakeCollection([f"c{i}" for i in range(80)])
    assert len(ev.vector_rank_ids(big, "q")) == 50 and big.calls[0][1] == 50
    print("✅ test_vector_candidate_pool_is_capped_and_never_exceeds_collection_size 通过")


def test_hybrid_only_reorders_the_vector_candidate_pool():
    q = [{"id": "q1", "type": "keyword", "query": "伺服电机", "targets": [], "reason": "r"}]
    vec_ids = [c["id"] for c in CHUNKS][:10]  # 向量候选池只有前 10 个
    out = ev.hybrid_rankings(q, CHUNKS, {"q1": vec_ids})["q1"]
    assert set(out) == set(vec_ids), "混合排序结果必须正好是向量候选池的重排，不能引入池外 chunk"
    assert len(out) == len(vec_ids)
    print("✅ test_hybrid_only_reorders_the_vector_candidate_pool 通过")


def test_api_modes_refuse_without_explicit_approval():
    import subprocess
    env = {k: v for k, v in os.environ.items() if k != "EVAL_RETRIEVAL_ALLOW_API"}
    for mode in ("vector", "hybrid"):
        p = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_retrieval.py"),
                            "--mode", mode], capture_output=True, text=True, env=env, timeout=30)
        assert p.returncode != 0 and "--allow-api" in (p.stderr + p.stdout), (mode, p.stderr[-300:])
        p2 = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_retrieval.py"),
                             "--mode", mode, "--allow-api"], capture_output=True, text=True, env=env, timeout=30)
        assert p2.returncode != 0, "只有 --allow-api 而没有环境变量也必须拒绝"
    print("✅ test_api_modes_refuse_without_explicit_approval 通过（未批准时 vector/hybrid 均拒绝运行）")


def main():
    test_real_kb_has_expected_shape()
    test_every_target_exists_in_the_index()
    test_query_set_structure_and_reasons()
    test_first_hit_rank_and_metrics_hand_computed()
    test_multi_target_query_hits_on_any_target()
    test_vector_candidate_pool_is_capped_and_never_exceeds_collection_size()
    test_hybrid_only_reorders_the_vector_candidate_pool()
    test_api_modes_refuse_without_explicit_approval()
    print("\n全部测试通过！")


if __name__ == "__main__":
    main()
