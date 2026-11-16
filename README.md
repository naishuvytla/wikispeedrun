# Wikipedia Speedrun Agent

An agent that plays the Wikipedia Game. Given a start article and a target article, it clicks through real links until it reaches the target, within a click limit (12 by default). Built for CIS 1990.

## How it works

1. Resolve both titles, following redirects and flagging disambiguation pages.
2. On each page, rank the links by how close they are to the target.
3. Gemini picks one link and gives a short reason.
4. Code checks that the link really exists on the page before clicking it.
5. At a dead end, back up. API errors are retried with backoff.
6. Stop at the target, the click limit, or 40 model calls.

Pages are cached with their revision ID, so reruns don't hit Wikipedia again and results stay the same.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY=your_key
export WIKI_CONTACT=your_email   # Wikimedia asks API users for contact info
```

## Running

```bash
python -m wikiagent run "Philadelphia" "Benjamin Franklin"
python -m wikiagent eval cases/dev.jsonl
python -m wikiagent eval cases/dev.jsonl --agent baseline
pytest -q
```

`--no-embed` ranks links by word overlap instead of embeddings, which uses fewer API calls. `--offline` runs from the cache only.

Each run is saved to `runs/` with the path, the reason for each hop, the model ID, the date, and token counts.

## Layout

```
wikiagent/   agent loop, Wikipedia tools, cache, ranking, path checker
cases/       dev cases and their expected results
golden/      example searches used as few-shot prompts
tests/       offline tests, no network needed
```

## Status

Working: tools, cache, ranking, agent loop, baseline, path checker, and 3 dev cases. Next: waypoint planning, tracing, and the first dev evaluation.
