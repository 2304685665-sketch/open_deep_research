"""间接注入防护规模化离线评测（PROTOCOL_inject_large.md）。用法: python eval_inject_large.py [--out PATH]"""
import json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "src"))
os.environ.setdefault("OPENAI_API_KEY", "fake-key-for-offline-test")
import eval_retrieval as ER, inject_corpus_large as C
from open_deep_research import doc_content_guard as G

EXPECTED_SHA = "83ece48990be4ccb7b2b440f52c9a375d0e8486ca6602a84b5b51a399173f43e"


def run():
    if C.corpus_sha() != EXPECTED_SHA: raise SystemExit("语料 sha 与冻结值不符")
    kb = ER.load_kb_chunks(os.path.join(HERE, "knowledge_base"))
    rows = []
    for r in C.build(kb):
        out, n = G.sanitize_chunk(r["text"])
        survives = r["inj"] in out
        base_sents = [p for p in re.split(r"(?<=[。！？!?；;\n])", r["base"]) if p.strip()]
        kept = sum(s in out for s in base_sents) / max(1, len(base_sents))
        rows.append({"kind": r["kind"], "cat": r["cat"], "blocked": not survives, "chunk_fully_masked": out == G.REMOVED, "fact_retention": kept})
    res = {"corpus_sha256": C.corpus_sha(), "by_cat": {}}
    for cat in sorted({r["cat"] for r in rows}):
        rs = [r for r in rows if r["cat"] == cat]
        res["by_cat"][cat] = {"n": len(rs), "blocked": sum(r["blocked"] for r in rs)}
    inj = [r for r in rows if r["kind"] == "inj"]; ben = [r for r in rows if r["kind"] == "benign"]
    res["injections"] = {"n": len(inj), "blocked": sum(r["blocked"] for r in inj), "chunk_fully_masked": sum(r["chunk_fully_masked"] for r in inj),
                         "mean_fact_retention": round(sum(r["fact_retention"] for r in inj) / len(inj), 3)}
    res["benign_controls"] = {"n": len(ben), "falsely_blocked": sum(r["blocked"] for r in ben), "chunk_fully_masked": sum(r["chunk_fully_masked"] for r in ben)}
    return res


if __name__ == "__main__":
    r = run(); print(json.dumps(r, ensure_ascii=False, indent=1))
    if "--out" in sys.argv:
        o = sys.argv[sys.argv.index("--out") + 1]
        if os.path.exists(o): raise SystemExit(f"拒绝覆盖: {o}")
        json.dump(r, open(o, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
