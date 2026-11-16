"""Few-shot "golden samples" of good searches.

Only samples checked against the graph are used, and sample pairs may never
overlap an eval case. Otherwise the examples would leak answers into results.
"""
import json
from pathlib import Path

from .graph import ToolError

GOLDEN = Path(__file__).parent.parent / "golden" / "golden_samples.json"


def load(path=GOLDEN, verified_only=True):
    samples = json.loads(Path(path).read_text())
    return [s for s in samples if s.get("verified")] if verified_only else samples


def format_samples(samples):
    if not samples:
        return ""
    blocks = []
    for i, s in enumerate(samples, 1):
        hops = "\n".join(f"  -> {h['page']}: {h['reason']}" for h in s["hops"])
        blocks.append(f"Example {i}: {s['start']} to {s['target']}\n  {s['start']}\n{hops}")
    return "Examples of efficient searches:\n" + "\n".join(blocks) + "\n"


def verify(graph, path=GOLDEN):
    """Check every hop against the graph, write the result back, return failures."""
    samples = load(path, verified_only=False)
    failures = []
    for s in samples:
        cur, ok, checked = s["start"], True, True
        for hop in s["hops"]:
            try:
                real = hop["page"] in graph.links(cur)
            except ToolError as e:
                failures.append(f"{s['start']} to {s['target']}: not checked ({e})")
                checked = False
                break
            if not real:
                failures.append(f"{s['start']} to {s['target']}: no link {cur} -> {hop['page']}")
                ok = False
                break
            cur = hop["page"]
        if checked:  # a failed request says nothing about the path
            s["verified"] = ok and cur == s["target"]
    Path(path).write_text(json.dumps(samples, indent=2) + "\n")
    return failures


def assert_no_leak(samples, cases):
    pairs = {(s["start"], s["target"]) for s in samples}
    leaked = [c for c in cases if (c["start"], c["target"]) in pairs]
    if leaked:
        raise ValueError(f"golden sample overlaps eval case: {leaked}")
