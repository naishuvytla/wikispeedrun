import pytest
import requests
from fakes import FakeResponse, FakeSession

from wikiagent.cache import PageCache
from wikiagent.graph import BudgetExceeded, RateLimited, ToolError
from wikiagent.llm import Gemini
from wikiagent.ranking import Ranker
from wikiagent.wikipedia import WikipediaGraph


def wiki(replies, **kw):
    session = FakeSession(replies)
    kw = {"min_interval": 0, "sleep": lambda s: None, **kw}
    return WikipediaGraph(PageCache(":memory:"), session=session, **kw), session


def test_links_follow_continuation():
    g, s = wiki([
        FakeResponse({"continue": {"plcontinue": "x"}, "query": {"pages": [
            {"links": [{"title": "A"}]}]}}),
        FakeResponse({"query": {"pages": [{"links": [{"title": "B"}]}]}}),
    ])
    assert g.links("P") == ["A", "B"] and s.calls[1]["plcontinue"] == "x"


def test_resolve_reads_redirect_and_disambiguation():
    g, _ = wiki([FakeResponse({"query": {"pages": [{
        "title": "Mercury", "lastrevid": 5, "pageprops": {"disambiguation": ""},
        "redirects": [{"title": "Mercury (disambiguation)"}]}]}})])
    page = g.resolve("mercury")
    assert page.title == "Mercury" and page.disambiguation and page.revid == 5
    assert page.aliases == ["Mercury (disambiguation)"]


def test_missing_page_detected():
    g, _ = wiki([FakeResponse({"query": {"pages": [{"title": "Nope", "missing": True}]}})])
    assert g.resolve("Nope").exists is False


def test_timeout_retries_then_succeeds():
    g, s = wiki([requests.Timeout(), FakeResponse(status=503),
                 FakeResponse({"query": {"pages": [{"links": [{"title": "A"}]}]}})])
    assert g.links("P") == ["A"] and len(s.calls) == 3


def test_gives_up_with_tool_error_not_a_crash():
    g, _ = wiki([requests.Timeout() for _ in range(5)])
    with pytest.raises(ToolError):
        g.links("P")


def test_budget_blocks_requests():
    g, s = wiki([], max_requests_per_run=0)
    with pytest.raises(BudgetExceeded):
        g.links("P")
    assert s.calls == []


def test_short_retry_after_is_honored():
    waits = []
    g, _ = wiki([FakeResponse(status=429, headers={"Retry-After": "7"}),
                 FakeResponse({"query": {"pages": [{"links": []}]}})],
                sleep=waits.append)
    assert g.links("P") == [] and waits == [7.0]


def test_no_header_waits_at_least_five_seconds():
    waits = []
    g, _ = wiki([FakeResponse(status=429),
                 FakeResponse({"query": {"pages": [{"links": []}]}})],
                sleep=waits.append)
    g.links("P")
    assert waits == [5.0]


def test_long_retry_after_stops_without_more_requests():
    g, s = wiki([FakeResponse(status=429, headers={"Retry-After": "1800"})])
    with pytest.raises(RateLimited, match="30 min"):
        g.links("P")
    assert len(s.calls) == 1


def test_contact_and_token_are_sent(monkeypatch):
    monkeypatch.setenv("WIKI_CONTACT", "me@example.org")
    monkeypatch.setenv("WIKI_TOKEN", "abc")
    _, s = wiki([])
    assert "me@example.org" in s.headers["User-Agent"]
    assert s.headers["Authorization"] == "Bearer abc"


def gemini(replies):
    session = FakeSession(replies)
    return Gemini(api_key="k", model="m", embed_model="e", session=session,
                  min_interval=0, sleep=lambda s: None), session


def test_gemini_parses_json_and_logs_model_and_tokens():
    llm, s = gemini([FakeResponse({
        "candidates": [{"content": {"parts": [{"text": '{"link": "A", "reason": "r"}'}]}}],
        "modelVersion": "flash-001",
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 2}})])
    assert llm.choose_json("sys", "user")["link"] == "A"
    assert llm.model_version == "flash-001" and llm.input_tokens == 10
    assert s.calls[0]["generationConfig"]["temperature"] == 0


def test_gemini_retries_429_then_succeeds():
    llm, s = gemini([FakeResponse(status=429), FakeResponse({
        "candidates": [{"content": {"parts": [{"text": "{}"}]}}]})])
    assert llm.choose_json("s", "u") == {} and len(s.calls) == 2


def test_bad_model_output_is_tool_error():
    llm, _ = gemini([FakeResponse({"candidates": [{"content": {"parts": [{"text": "hi"}]}}]})])
    with pytest.raises(ToolError):
        llm.choose_json("s", "u")


def test_embed_batches_at_100():
    llm, s = gemini([FakeResponse({"embeddings": [{"values": [1.0]}] * 100}),
                     FakeResponse({"embeddings": [{"values": [1.0]}] * 50})])
    assert len(llm.embed(["x"] * 150)) == 150 and len(s.calls) == 2


class FakeEmbedder:
    vectors = {"Quantum": [1, 0], "Banana": [0, 1], "T: s": [1, 0.1]}

    def __init__(self):
        self.asked = []

    def embed(self, texts):
        self.asked += texts
        return [self.vectors[t] for t in texts]


def test_embedding_ranking_and_vector_cache():
    embedder = FakeEmbedder()
    ranker = Ranker(embedder, PageCache(":memory:"))
    assert ranker.rank(["Banana", "Quantum"], "T", "s")[0]["title"] == "Quantum"
    ranker.rank(["Banana", "Quantum"], "T", "s")
    assert embedder.asked.count("Banana") == 1


def test_links_to_target_rank_first():
    rows = Ranker().rank(["Physics of target", "Plain"], "target", "", backlinks=["Plain"])
    assert rows[0]["title"] == "Plain" and rows[0]["links_to_target"]


def test_ranker_falls_back_when_embeddings_fail():
    class Broken:
        def embed(self, texts):
            raise ToolError("quota")

    ranker = Ranker(Broken())
    assert ranker.rank(["Black hole", "Cat"], "Black hole", "")[0]["title"] == "Black hole"
    assert ranker.fallbacks == 1
