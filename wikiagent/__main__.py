import argparse
import json
from datetime import datetime
from pathlib import Path

from . import golden
from .agent import Agent, Config, GreedyBaseline, LLMChooser
from .cache import PageCache
from .checker import score_case
from .llm import Gemini
from .ranking import Ranker
from .wikipedia import WikipediaGraph


def build(args):
    cache = PageCache(args.cache, max_bytes=int(args.cache_mb * 1_000_000))
    graph = WikipediaGraph(cache, offline=args.offline,
                           max_requests_per_run=args.max_requests)
    llm = None
    if args.agent == "llm":
        llm = Gemini()
        samples = golden.load()
        if not samples:
            print("Warning: no verified golden samples. Run verify-golden first.")
        chooser = LLMChooser(llm, golden.format_samples(samples))
    else:
        chooser = GreedyBaseline()
    embedder = None if args.no_embed or llm is None else llm
    agent = Agent(graph, Ranker(embedder, cache), chooser, Config(click_limit=args.limit))
    return agent, graph, cache, llm


def finish(result, graph, cache, llm):
    if llm:
        result.model_id = llm.model_version
        result.input_tokens, result.output_tokens = llm.input_tokens, llm.output_tokens
    result.wiki_requests = graph.requests_this_run
    result.cache = cache.stats()
    out = result.to_dict()
    Path("runs").mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    Path(f"runs/{stamp}.json").write_text(json.dumps(out, indent=2))
    return out


def show(out):
    print(f"\n{out['start']} -> {out['target']}: {out['status']} "
          f"({out['clicks']} clicks, {out['backtracks']} backtracks, "
          f"{out['model_calls']} model calls)")
    for h in out["hops"]:
        print(f"  {h['from']} -> {h['landed']}  | {h['reason']}")
    if out["rejected"]:
        print(f"  rejected picks: {out['rejected']}")
    if out["options"]:
        print(f"  options: {out['options']}")
    for e in out["events"]:
        print(f"  note: {e}")
    print(f"  model: {out['model_id']} | run: {out['run_date']}")


def cmd_run(args):
    agent, graph, cache, llm = build(args)
    out = finish(agent.run(args.start, args.target), graph, cache, llm)
    show(out)


def cmd_eval(args):
    cases = [json.loads(x) for x in Path(args.cases).read_text().splitlines() if x.strip()]
    golden.assert_no_leak(golden.load(verified_only=False), cases)
    agent, graph, cache, llm = build(args)
    rows = []
    for case in cases:
        out = finish(agent.run(case["start"], case["target"]), graph, cache, llm)
        graph.pin_touched()  # keep eval pages replayable
        show(out)
        correct, reason = score_case(case, out, graph, args.limit)
        rows.append({**case, "correct": correct, "reason": reason, "status": out["status"],
                     "hops": len(out["hops"]), "path": out["path"]})
    print("\ncase | correct | hops | why")
    for row in rows:
        print(f"{row['id']} | {row['correct']} | {row['hops']} | {row['reason']}")
    passed = sum(r["correct"] for r in rows)
    print(f"\n{passed}/{len(rows)} correct ({args.agent})")
    Path("runs").mkdir(exist_ok=True)
    summary = Path(f"runs/eval_{Path(args.cases).stem}_{args.agent}.json")
    summary.write_text(json.dumps(rows, indent=2))
    print(f"Saved {summary}")


def cmd_verify_golden(args):
    cache = PageCache(args.cache, max_bytes=int(args.cache_mb * 1_000_000))
    failures = golden.verify(WikipediaGraph(cache, max_requests_per_run=1000))
    print("\n".join(failures) or "All hops verified.")
    print(f"{len(golden.load())} verified samples ready")


def cmd_cache_stats(args):
    print(json.dumps(PageCache(args.cache, int(args.cache_mb * 1_000_000)).stats(), indent=2))


def main():
    p = argparse.ArgumentParser(prog="wikiagent")
    p.add_argument("--cache", default="data/cache.sqlite")
    p.add_argument("--cache-mb", type=float, default=200)
    sub = p.add_subparsers(required=True)

    def shared(s):
        s.add_argument("--agent", choices=["llm", "baseline"], default="llm")
        s.add_argument("--limit", type=int, default=12)
        s.add_argument("--max-requests", type=int, default=120)
        s.add_argument("--offline", action="store_true", help="cache only, no network")
        s.add_argument("--no-embed", action="store_true", help="rank by word overlap")

    s = sub.add_parser("run")
    s.add_argument("start")
    s.add_argument("target")
    shared(s)
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("eval")
    s.add_argument("cases")
    shared(s)
    s.set_defaults(fn=cmd_eval)
    sub.add_parser("verify-golden").set_defaults(fn=cmd_verify_golden)
    sub.add_parser("cache-stats").set_defaults(fn=cmd_cache_stats)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
