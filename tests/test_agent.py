from fakes import FakeGraph, ScriptedChooser
from wikiagent.agent import Agent, Config, GreedyBaseline
from wikiagent.checker import check, score_case
from wikiagent.ranking import Ranker

WORLD = {
    "Start": ["Hub", "Dead", "Noise"],
    "Hub": ["Start", "Near"],
    "Near": ["Target"],
    "Dead": ["Start"],
    "Noise": ["Start"],
    "Target": [],
}


def run(graph, picks, **cfg):
    chooser = ScriptedChooser(picks)
    agent = Agent(graph, Ranker(), chooser, Config(**cfg))
    return agent.run("Start", "Target"), chooser


def test_direct_link_needs_no_model_call():
    g = FakeGraph({"Start": ["Target"], "Target": []})
    r, chooser = run(g, [])
    assert r.status == "reached" and chooser.calls == 0 and r.path == ["Start", "Target"]
    assert check(r.to_dict(), g) == (True, "ok")


def test_solves_multi_hop_and_checker_passes():
    g = FakeGraph(WORLD)
    r, _ = run(g, ["Hub", "Near"])
    assert r.path == ["Start", "Hub", "Near", "Target"]
    assert check(r.to_dict(), g) == (True, "ok")


def test_made_up_link_is_rejected_then_reprompted():
    g = FakeGraph(WORLD)
    r, chooser = run(g, ["Target", "Hub", "Near"])
    assert r.rejected == ["Target"]
    assert "not an available link" in chooser.seen[1]["error"]
    assert r.status == "reached"


def test_repeated_invalid_picks_fall_back_to_a_real_link():
    g = FakeGraph({"Start": ["Near", "Other"], "Near": ["Target"], "Other": [], "Target": []})
    r, _ = run(g, ["Fake1", "Fake2"])
    assert "fallback" in r.hops[0]["reason"] and check(r.to_dict(), g)[0]


def test_dead_end_backs_up_and_abandoned_clicks_count():
    world = {"Start": ["Dead", "Hub"], "Dead": ["Start"], "Hub": ["Target"], "Target": []}
    g = FakeGraph(world)
    r, _ = run(g, ["Dead", "Hub"])
    assert r.status == "reached" and r.backtracks == 1 and r.clicks == 3
    assert r.path == ["Start", "Hub", "Target"]


def test_click_limit_stops_run():
    world = {f"P{i}": [f"P{i+1}"] for i in range(30)}
    world.update({"Start": ["P0"], "Target": [], "P30": []})
    r, _ = run(FakeGraph(world), [f"P{i}" for i in range(31)], click_limit=5)
    assert r.status == "limit_hit" and r.clicks == 5


def test_model_call_limit_stops_run():
    world = {f"P{i}": [f"P{i+1}"] for i in range(30)}
    world.update({"Start": ["P0"], "Target": [], "P30": []})
    r, chooser = run(FakeGraph(world), [f"P{i}" for i in range(31)],
                     click_limit=100, max_model_calls=3)
    assert r.status == "model_limit" and chooser.calls == 3


def test_disambiguation_target_is_flagged():
    g = FakeGraph({"Start": [], "Target": ["Planet", "Element"]}, disambiguation=["Target"])
    r, _ = run(g, [])
    assert r.status == "needs_disambiguation" and r.options == ["Planet", "Element"]
    assert score_case({"expected_status": "needs_disambiguation"}, r.to_dict(), g)[0]


def test_redirect_to_target_counts_as_reached():
    g = FakeGraph({"Start": ["Targets"], "Target": []}, redirects={"Targets": "Target"})
    r, _ = run(g, [])
    assert r.status == "reached" and r.path[-1] == "Target"
    assert check(r.to_dict(), g) == (True, "ok")


def test_tool_failure_backs_up_instead_of_crashing():
    world = {"Start": ["Bad", "Hub"], "Bad": [], "Hub": ["Target"], "Target": []}
    r, _ = run(FakeGraph(world, fail=["Bad"]), ["Bad", "Hub"])
    assert r.status == "reached" and r.backtracks == 1


def test_tool_failure_on_start_returns_error_status():
    r, _ = run(FakeGraph(WORLD, fail=["Start"]), [])
    assert r.status == "error"


def test_red_link_is_skipped():
    world = {"Start": ["Ghost", "Hub"], "Ghost": [], "Hub": ["Target"], "Target": []}
    r, _ = run(FakeGraph(world, missing=["Ghost"]), ["Ghost", "Hub"])
    assert r.status == "reached" and "Ghost" not in r.path


def test_baseline_runs_without_a_model():
    g = FakeGraph(WORLD)
    r = Agent(g, Ranker(), GreedyBaseline()).run("Start", "Target")
    assert r.agent == "baseline" and r.status == "reached" and check(r.to_dict(), g)[0]


def test_checker_catches_each_failure_type():
    g = FakeGraph(WORLD)
    good, _ = run(g, ["Hub", "Near"])
    d = good.to_dict()
    assert check({**d, "status": "limit_hit"}, g)[0] is False
    assert check({**d, "clicks": 13}, g)[1] == "over click limit"
    assert check({**d, "path": d["path"][:-1]}, g)[1] == "wrong end"
    fake_hop = [{"from": "Start", "clicked": "Target", "landed": "Target", "reason": ""}]
    made_up = {**d, "path": ["Start", "Target"], "hops": fake_hop}
    assert "made-up" in check(made_up, g)[1]
    assert check({**d, "path": ["Start", "Hub", "Start", "Target"]}, g)[1] == "loop in path"
