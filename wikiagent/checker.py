"""Code-only scorer. Trusts nothing in a result without checking the graph."""


def check(result, graph, click_limit=12):
    """Return (correct, reason). `result` is a Result.to_dict() dictionary."""
    path, hops = result["path"], result["hops"]
    if result["status"] != "reached":
        return False, f"status {result['status']}"
    if not path or path[0] != result["start"]:
        return False, "wrong start"
    if path[-1] != result["target"]:
        return False, "wrong end"
    if len(set(path)) != len(path):
        return False, "loop in path"
    if result["clicks"] > click_limit:
        return False, "over click limit"
    if len(hops) != len(path) - 1:
        return False, "hops do not match path"
    for i, hop in enumerate(hops):
        if hop["from"] != path[i] or hop["landed"] != path[i + 1]:
            return False, f"hop {i} does not match path"
        if hop["clicked"] not in graph.links(hop["from"]):
            return False, f"made-up link: {hop['from']} -> {hop['clicked']}"
        if graph.resolve(hop["clicked"]).title != hop["landed"]:
            return False, f"wrong landing page for {hop['clicked']}"
    return True, "ok"


def score_case(case, result, graph, click_limit=12):
    """Cases can expect a stop status instead of a path, e.g. Mercury."""
    expected = case.get("expected_status", "reached")
    if expected != "reached":
        ok = result["status"] == expected
        return ok, "ok" if ok else f"expected {expected}, got {result['status']}"
    correct, reason = check(result, graph, click_limit)
    max_hops = case.get("max_hops")
    if correct and max_hops and len(result["hops"]) > max_hops:
        return False, f"took {len(result['hops'])} hops, expected at most {max_hops}"
    return correct, reason
