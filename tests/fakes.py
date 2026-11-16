from wikiagent.graph import Page, ToolError


class FakeGraph:
    """Small in-memory graph with the same interface as WikipediaGraph."""

    def __init__(self, links, disambiguation=(), redirects=None, missing=(), fail=()):
        self._links, self.disambiguation = links, set(disambiguation)
        self.redirects, self.missing = redirects or {}, set(missing)
        self.fail = set(fail)
        self.fetched = []

    def resolve(self, title):
        real = self.redirects.get(title, title)
        return Page(real, exists=real in self._links and real not in self.missing,
                    disambiguation=real in self.disambiguation,
                    aliases=[k for k, v in self.redirects.items() if v == real])

    def links(self, title):
        self.fetched.append(title)
        if title in self.fail:
            raise ToolError(f"timeout on {title}")
        return self._links[title]

    def summary(self, title):
        return f"{title} summary"

    def backlinks(self, title):
        return [p for p, ls in self._links.items() if title in ls]


class ScriptedChooser:
    name = "llm"

    def __init__(self, picks):
        self.picks, self.calls, self.seen = list(picks), 0, []

    def choose(self, ctx):
        self.calls += 1
        self.seen.append(ctx)
        return self.picks.pop(0), "scripted"


class FakeResponse:
    def __init__(self, payload=None, status=200, text="", headers=None):
        self.payload, self.status_code, self.text = payload, status, text
        self.headers = headers or {}

    def json(self):
        return self.payload

    def raise_for_status(self):
        import requests
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class FakeSession:
    """Replays queued responses. An Exception in the queue is raised."""

    def __init__(self, replies):
        self.replies, self.calls, self.headers = list(replies), [], {}

    def _next(self, payload):
        self.calls.append(payload)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def get(self, url, params=None, timeout=None):
        return self._next(params)

    def post(self, url, json=None, timeout=None, headers=None):
        return self._next(json)
