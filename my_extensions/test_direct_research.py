import asyncio, os, sys
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from open_deep_research import deep_researcher as dr


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv("ODR_DIRECT_RESEARCH", raising=False)
    assert dr._direct_research_enabled() is False
    monkeypatch.setenv("ODR_DIRECT_RESEARCH", "1")
    assert dr._direct_research_enabled() is True


def test_graph_has_direct_node_and_edge():
    g = dr.deep_researcher.get_graph()
    names = set(g.nodes)
    assert {"direct_research", "research_supervisor", "final_report_generation"} <= names
    edges = {(e.source, e.target) for e in g.edges}
    assert ("direct_research", "final_report_generation") in edges


def test_direct_research_maps_state_like_supervisor(monkeypatch):
    seen = {}

    async def fake_ainvoke(inp, cfg):
        seen["topic"] = inp["research_topic"]
        return {"compressed_research": "ANSWER", "raw_notes": ["a", "b"],
                "degradation_events": [{"k": 1}], "fetched_sources": [{"u": 1}], "fetched_doc_ids": [{"d": 1}]}

    monkeypatch.setattr(dr.researcher_subgraph, "ainvoke", fake_ainvoke)
    out = asyncio.run(dr.direct_research({"research_brief": "BRIEF"}, {"configurable": {}}))
    assert seen["topic"] == "BRIEF"
    assert out["notes"] == ["ANSWER"] and out["research_brief"] == "BRIEF"
    assert out["raw_notes"] == ["a\nb"] and out["degradation_events"] == [{"k": 1}]
    assert out["fetched_sources"] == [{"u": 1}] and out["fetched_doc_ids"] == [{"d": 1}]


def test_direct_research_failure_degrades(monkeypatch):
    async def boom(inp, cfg):
        raise RuntimeError("x")
    monkeypatch.setattr(dr.researcher_subgraph, "ainvoke", boom)
    out = asyncio.run(dr.direct_research({"research_brief": "B"}, {"configurable": {}}))
    assert "Error synthesizing" in out["notes"][0]
