"""Tests for the two pieces of TA feedback."""
import json

import pytest
from fakes import FakeResponse, FakeSession, ScriptedChooser

from wikiagent import golden
from wikiagent.agent import Agent, Config, GreedyBaseline, LLMChooser
from wikiagent.cache import PageCache
from wikiagent.ranking import Ranker
from wikiagent.wikipedia import WikipediaGraph


def page_json(title, links):
    return {"query": {"pages": [{"title": title, "lastrevid": 1,
                                 "links": [{"title": t} for t in links]}]}}


class WideWikipedia:
    """Every page has 40 unique links, none reaching the target. Worst case for a cache."""
    headers = {}

    def __init__(self):
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        title = params.get("titles", params.get("bltitle", "x"))
        if params.get("list") == "backlinks":
            return FakeResponse({"query": {"backlinks": []}})
        if "extracts" in params.get("prop", ""):
            return FakeResponse({"query": {"pages": [{"extract": "target text"}]}})
        if "info" in params.get("prop", ""):
            return FakeResponse({"query": {"pages": [{"title": title, "lastrevid": 7}]}})
        return FakeResponse(page_json(title, [f"{title}/{i}" for i in range(40)]))


def test_cache_never_exceeds_cap_and_evicts_oldest():
    cache = PageCache(":memory:", max_bytes=3000)
    import random
    rng = random.Random(0)
    for i in range(200):
        cache.put("links", f"p{i}", [str(rng.random()) for _ in range(50)])
    assert cache.total_bytes() <= 3000 and cache.stats()["evictions"] > 0
    assert cache.get("links", "p0") is None and cache.get("links", "p199") is not None


def test_recently_read_page_survives_eviction():
    import random
    rng = random.Random(1)
    blob = lambda: [str(rng.random()) for _ in range(50)]
    cache = PageCache(":memory:", max_bytes=3000)
    cache.put("links", "keep", blob())
    for i in range(3):
        cache.put("links", f"a{i}", blob())
        assert cache.get("links", "keep") is not None
    assert cache.get("links", "keep") is not None


def test_pinned_pages_survive_a_flood():
    import random
    rng = random.Random(2)
    cache = PageCache(":memory:", max_bytes=3000)
    cache.put("links", "eval-page", ["x"] * 20)
    cache.pin("links", "eval-page")
    for i in range(200):
        cache.put("links", f"p{i}", [str(rng.random()) for _ in range(50)])
    assert cache.get("links", "eval-page") == ["x"] * 20


def test_cache_persists_across_restarts(tmp_path):
    path = str(tmp_path / "c.sqlite")
    PageCache(path).put("links", "A", ["B"], revid=9)
    assert PageCache(path).get("links", "A") == ["B"]


def test_budget_stops_wide_search_even_with_huge_click_limit():
    session = WideWikipedia()
    graph = WikipediaGraph(PageCache(":memory:"), session=session, max_requests_per_run=30,
                           min_interval=0, sleep=lambda s: None)
    agent = Agent(graph, Ranker(), GreedyBaseline(), Config(click_limit=10_000))
    r = agent.run("Root", "Elsewhere")
    assert r.status == "budget_hit" and session.calls <= 30


def test_second_run_is_served_from_cache_and_offline_replay_works():
    world = {"Start": ["Target"], "Target": []}

    class Small(WideWikipedia):
        def get(self, url, params=None, timeout=None):
            self.calls += 1
            if params.get("prop", "").startswith("links") or params.get("prop") == "links":
                t = params["titles"]
                return FakeResponse(page_json(t, world[t]))
            return super().get(url, params, timeout)

    session, cache = Small(), PageCache(":memory:")
    kw = dict(min_interval=0, sleep=lambda s: None)
    agent = Agent(WikipediaGraph(cache, session=session, **kw), Ranker(), GreedyBaseline())
    first = agent.run("Start", "Target")
    used = session.calls
    assert first.status == "reached" and used > 0
    assert agent.run("Start", "Target").path == first.path and session.calls == used
    offline = WikipediaGraph(cache, session=session, offline=True, **kw)
    assert Agent(offline, Ranker(), GreedyBaseline()).run("Start", "Target").path == first.path
    assert session.calls == used


def samples_file(tmp_path):
    path = tmp_path / "g.json"
    path.write_text(json.dumps([
        {"start": "A", "target": "C", "verified": False, "hops": [
            {"page": "B", "reason": "hub"}, {"page": "C", "reason": "direct"}]},
        {"start": "A", "target": "Z", "verified": False, "hops": [
            {"page": "Z", "reason": "this link is fake"}]},
    ]))
    return path


def test_verify_keeps_real_paths_and_drops_fake_ones(tmp_path):
    from fakes import FakeGraph
    path = samples_file(tmp_path)
    assert golden.load(path) == []
    failures = golden.verify(FakeGraph({"A": ["B"], "B": ["C"], "C": []}), path)
    assert [s["target"] for s in golden.load(path)] == ["C"]
    assert any("A -> Z" in f for f in failures)


def test_rate_limit_is_not_reported_as_missing_link(tmp_path):
    from fakes import FakeGraph
    path = samples_file(tmp_path)
    failures = golden.verify(FakeGraph({"A": ["B"], "B": ["C"], "C": []}, fail=["A"]), path)
    assert all("not checked" in f for f in failures) and len(failures) == 2


def test_rate_limit_stops_the_run_instead_of_backing_up():
    from fakes import FakeGraph
    from wikiagent.graph import RateLimited

    class Limited(FakeGraph):
        def links(self, title):
            raise RateLimited("wait")

    r = Agent(Limited({"A": [], "B": []}), Ranker(), GreedyBaseline()).run("A", "B")
    assert r.status == "rate_limited"


def test_golden_samples_reach_the_prompt():
    class LLM:
        calls = 0

        def choose_json(self, system, user):
            self.system = system
            return {"link": "B", "reason": "x"}

    llm = LLM()
    text = golden.format_samples([{"start": "A", "target": "C", "hops": [
        {"page": "C", "reason": "direct"}]}])
    LLMChooser(llm, text).choose({
        "target": "T", "summary": "", "current": "A", "path": ["A"], "clicks_left": 3,
        "candidates": [{"title": "B", "links_to_target": False}], "error": None})
    assert "Example 1: A to C" in llm.system


def test_golden_overlap_with_eval_case_raises():
    with pytest.raises(ValueError):
        golden.assert_no_leak([{"start": "A", "target": "B"}], [{"start": "A", "target": "B"}])


def test_shipped_golden_file_does_not_overlap_dev_cases():
    from pathlib import Path
    root = Path(golden.__file__).parent.parent
    cases = [json.loads(x) for x in (root / "cases/dev.jsonl").read_text().splitlines()]
    assert len(cases) == 3
    golden.assert_no_leak(golden.load(verified_only=False), cases)
