"""指标端点与前端静态页的离线测试：fakeredis + 假 runner + 注入时钟，不碰真实服务/浏览器/模型。"""

import asyncio
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import fakeredis.aioredis as fr  # noqa: E402
import httpx  # noqa: E402

from open_deep_research.api_service import Settings, create_app  # noqa: E402

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "open_deep_research", "web")
DAY = 86400


def run(c):
    return asyncio.run(c)


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def make(runner=None, **kw):
    redis = fr.FakeRedis()
    clock = kw.pop("clock", Clock())

    async def ok(q):
        if q.startswith("boom"):
            raise RuntimeError("x")
        await asyncio.sleep(0.01)
        return "ok"

    app = create_app(redis, runner or ok, Settings(clock=clock, rate_limit_per_window=1000, **kw))
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")
    return redis, app, client, clock


async def drain(app):
    while app.state.tasks:
        await asyncio.gather(*list(app.state.tasks), return_exceptions=True)


def test_metrics_counts_success_rate_and_durations():
    async def go():
        redis, app, c, _ = make()
        for q in ("a", "b", "boom1"):
            await c.post("/research", json={"question": q})
        await drain(app)
        m = (await c.get("/metrics/summary")).json()
        assert (m["tasks_created"], m["tasks_succeeded"], m["tasks_failed"]) == (3, 2, 1), m
        assert abs(m["success_rate"] - 2 / 3) < 1e-9 and m["duration_ms"]["samples"] == 3
        assert m["duration_ms"]["p50"] is not None and m["duration_ms"]["p95"] >= m["duration_ms"]["p50"]
        await c.aclose()
    run(go())
    print("✅ test_metrics_counts_success_rate_and_durations 通过")


def test_metrics_empty_state_has_null_rates():
    async def go():
        _, _, c, _ = make()
        m = (await c.get("/metrics/summary")).json()
        assert m["tasks_created"] == 0 and m["success_rate"] is None and m["duration_ms"]["p50"] is None
        assert m["returning_clients_today"] == 0
        await c.aclose()
    run(go())
    print("✅ test_metrics_empty_state_has_null_rates 通过")


def test_idempotent_replays_and_rate_limited_are_counted():
    async def go():
        redis = fr.FakeRedis()

        async def r(q):
            return "ok"

        app = create_app(redis, r, Settings(rate_limit_per_window=3, rate_limit_window_seconds=60))
        c = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")
        h = {"Idempotency-Key": "k"}
        await c.post("/research", json={"question": "q"}, headers=h)
        await c.post("/research", json={"question": "q"}, headers=h)  # 命中幂等
        await c.post("/research", json={"question": "q2"})
        await c.post("/research", json={"question": "q3"})  # 第 4 次 -> 429
        await drain(app)
        m = (await c.get("/metrics/summary")).json()
        assert m["idempotent_replays"] == 1 and m["rate_limited"] == 1 and m["tasks_created"] == 2, m
        await c.aclose()
    run(go())
    print("✅ test_idempotent_replays_and_rate_limited_are_counted 通过")


def test_daily_active_and_returning_clients_with_injected_clock():
    async def go():
        redis, app, c, clock = make()
        for key in ("alice", "bob"):  # 第 0 天
            await c.post("/research", json={"question": "q"}, headers={"X-API-Key": key})
        clock.t += DAY  # 第 1 天：alice 回访，carol 新增
        for key in ("alice", "carol"):
            await c.post("/research", json={"question": "q"}, headers={"X-API-Key": key})
        await drain(app)
        m = (await c.get("/metrics/summary", headers={"X-API-Key": "alice"})).json()
        vals = list(m["daily_active_clients"].values())
        assert vals[0] == 2 and vals[1] == 2 and sum(vals) == 4, m["daily_active_clients"]
        assert m["returning_clients_today"] == 1  # 只有 alice 在今天之前出现过
        await c.aclose()
    run(go())
    print("✅ test_daily_active_and_returning_clients_with_injected_clock 通过")


def test_raw_api_keys_and_ips_never_stored_in_redis():
    async def go():
        redis, app, c, _ = make()
        await c.post("/research", json={"question": "q"}, headers={"X-API-Key": "super-secret-key-123"})
        await drain(app)
        dump = []
        async for k in redis.scan_iter("metrics:*"):
            t = await redis.type(k)
            t = t.decode() if isinstance(t, bytes) else t
            dump.append(k.decode() if isinstance(k, bytes) else k)
            if t == "set":
                dump += [m.decode() for m in await redis.smembers(k)]
        assert dump and not any("super-secret-key-123" in x for x in dump), dump
        await c.aclose()
    run(go())
    print("✅ test_raw_api_keys_and_ips_never_stored_in_redis 通过")


def test_metrics_requires_auth_when_keys_configured():
    async def go():
        _, _, c, _ = make(api_keys=frozenset({"k"}))
        assert (await c.get("/metrics/summary")).status_code == 401
        assert (await c.get("/metrics/summary", headers={"X-API-Key": "k"})).status_code == 200
        await c.aclose()
    run(go())
    print("✅ test_metrics_requires_auth_when_keys_configured 通过")


def test_web_page_and_script_are_served():
    async def go():
        _, _, c, _ = make()
        r = await c.get("/")
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
        assert 'id="root"' in r.text and "/static/app.js" in r.text and "react.production.min.js" in r.text
        j = await c.get("/static/app.js")
        assert j.status_code == 200 and "javascript" in j.headers["content-type"] and "ReactDOM.createRoot" in j.text
        await c.aclose()
    run(go())
    print("✅ test_web_page_and_script_are_served 通过")


def test_frontend_script_syntax_and_safety_rules():
    path = os.path.join(WEB, "app.js")
    if shutil.which("node"):
        r = subprocess.run(["node", "--check", path], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    code = [ln for ln in open(path, encoding="utf-8").read().splitlines() if not ln.strip().startswith("//")]
    joined = "\n".join(code)
    for banned in ("innerHTML", "dangerouslySetInnerHTML", "localStorage", "sessionStorage", "eval("):
        assert banned not in joined, banned
    print("✅ test_frontend_script_syntax_and_safety_rules 通过")


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
