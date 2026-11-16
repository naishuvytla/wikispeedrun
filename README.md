# Wikipedia Speedrun Agent

CIS 1990 final project. Finds a path of real links from a start article to a target article.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY=your_key
export WIKI_CONTACT=your_email_or_repo_url
pytest -q
python -m wikiagent verify-golden
```

## Use

| Goal | Command |
|---|---|
| One run | `python -m wikiagent run "Philadelphia" "Benjamin Franklin"` |
| Dev cases, agent | `python -m wikiagent eval cases/dev.jsonl` |
| Dev cases, baseline | `python -m wikiagent eval cases/dev.jsonl --agent baseline` |
| Faster run, no embeddings | add `--no-embed` after the command |
| Replay from cache only | add `--offline` after the command |
| Cache size and pins | `python -m wikiagent cache-stats` |

Each run writes a JSON log to `runs/` with the model ID, run date, tokens and cache stats.
Set `WIKI_CONTACT` to an email or repo URL, since Wikimedia asks for contact details.
Optional settings: `WIKI_TOKEN`, `GEMINI_MODEL`, `GEMINI_EMBED_MODEL`, `GEMINI_MIN_INTERVAL`.

## Feedback changes

| Feedback | Change | Where |
|---|---|---|
| Cache could oversaturate under BFS or poor search | The model picks one link per step, and pages load only when clicked | `agent.py` |
| | Each run has a request budget, default 120 | `wikipedia.py` |
| | Cache has a size cap with LRU eviction, default 200 MB | `cache.py` |
| | Eval pages are pinned, so eviction cannot break replay | `cache.py`, `__main__.py` |
| Few-shot golden samples | Hop-verified samples are added to the prompt | `golden.py`, `golden/golden_samples.json` |
| | Samples may not overlap eval cases | `golden.assert_no_leak` |
| Research paper extension | The agent depends only on a `Graph` interface, so a citation graph can plug in later | `graph.py` |

## Scope by milestone

| Milestone | Status |
|---|---|
| Tools, cache, path checker, baseline | Built |
| Agent solves easy pairs, 3 dev cases | Built |
| Dead-end backup and API retries | Basic version built |
| Waypoint planning, full tracing, 15 dev cases | Nov 11 |
| Route memory, fault injection, safety suite | Nov 18 to 21 |

## Check-in video (about 2 minutes)

| Step | Show | Say |
|---|---|---|
| 1 | The proposal's input and output | What the agent does and how success is checked in code |
| 2 | `python -m wikiagent eval cases/dev.jsonl` | Direct link, multi-hop run with reasons, Mercury flagged as ambiguous |
| 3 | The result table and a JSON log in `runs/` | Paths are verified against real links, with model ID and date logged |
| 4 | `pytest -q` and `cache-stats` | TA feedback: bounded cache, run budget, verified golden samples |
| 5 | The milestone table above | Next: planning, backtracking and tracing, then 15 dev cases |
