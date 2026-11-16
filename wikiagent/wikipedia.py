"""Read-only MediaWiki tools with caching, retries and a per-run budget."""
import os
import time

import requests

from .graph import BudgetExceeded, Page, RateLimited, ToolError

API = "https://en.wikipedia.org/w/api.php"
RETRYABLE = {429, 500, 502, 503, 504}


class WikipediaGraph:
    def __init__(self, cache, session=None, offline=False, max_requests_per_run=120,
                 min_interval=1.0, retries=3, backoff=5.0, max_wait=60.0,
                 sleep=time.sleep):
        self.cache = cache
        self.session = session or requests.Session()
        # Wikimedia's User-Agent policy asks for contact details. Clients it cannot
        # identify get its lowest rate limit.
        contact = os.environ.get("WIKI_CONTACT")
        if not contact and session is None and not offline:
            print("Warning: set WIKI_CONTACT or Wikipedia may rate limit you sooner.")
        self.session.headers.update({"User-Agent": "CIS1990-WikiSpeedrun/0.1"
                                     + (f" ({contact})" if contact else "")})
        token = os.environ.get("WIKI_TOKEN")  # optional OAuth 2 token, higher limit
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
        self.offline = offline  # True = cache only, a miss is an error
        self.max_requests_per_run = max_requests_per_run
        self.min_interval = min_interval
        self.retries = retries
        self.backoff = backoff
        self.max_wait = max_wait
        self.sleep = sleep
        self.requests_this_run = 0
        self.total_requests = 0
        self._last = 0.0
        self.touched = set()  # (kind, key) used this run, for pinning

    def start_run(self):
        self.requests_this_run = 0
        self.touched = set()

    def pin_touched(self):
        for kind, key in self.touched:
            self.cache.pin(kind, key)

    def _get(self, params):
        if self.offline:
            raise ToolError(f"offline cache miss: {params}")
        if self.requests_this_run >= self.max_requests_per_run:
            raise BudgetExceeded(f"over {self.max_requests_per_run} requests this run")
        params = {"format": "json", "formatversion": 2, "maxlag": 5, **params}
        self.requests_this_run += 1
        self.total_requests += 1
        last_error, delay = "unknown", None
        for attempt in range(self.retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                self.sleep(wait)
            self._last = time.monotonic()
            delay = None
            try:
                r = self.session.get(API, params=params, timeout=20)
                if r.status_code in RETRYABLE:
                    last_error = f"HTTP {r.status_code}"
                    delay = self._retry_after(r)
                else:
                    r.raise_for_status()
                    data = r.json()
                    if data.get("error", {}).get("code") == "maxlag":
                        last_error = "maxlag"
                    elif "error" in data:
                        raise ToolError(str(data["error"]))
                    else:
                        return data
            except (requests.Timeout, requests.ConnectionError) as e:
                last_error = type(e).__name__
            except (requests.HTTPError, ValueError) as e:
                raise ToolError(str(e)) from e
            wait = self.backoff * 2**attempt if delay is None else max(delay, 1.0)
            if wait > self.max_wait and delay is not None:
                raise RateLimited(f"{last_error}: Wikipedia asked to wait {wait / 60:.0f} min")
            if attempt < self.retries:
                self.sleep(wait)
        raise ToolError(f"gave up after retries: {last_error}")

    @staticmethod
    def _retry_after(response):
        value = getattr(response, "headers", {}).get("Retry-After", "")
        return float(value) if value.isdigit() else None

    def _cached(self, kind, key, fetch):
        self.touched.add((kind, key))
        value = self.cache.get(kind, key)
        if value is not None:
            return value
        value = fetch()
        revid = value.get("revid") if isinstance(value, dict) else None
        self.cache.put(kind, key, value, revid=revid)
        return value

    def _continued(self, params, section):
        """Follow API continuation and collect one list."""
        out, cont = [], {}
        while True:
            data = self._get({**params, **cont})
            query = data.get("query", {})
            if section == "links":
                for page in query.get("pages", []):
                    out += [x["title"] for x in page.get("links", [])]
            else:
                out += [x["title"] for x in query.get(section, [])]
            if "continue" not in data:
                return out
            cont = data["continue"]

    def resolve(self, title):
        def fetch():
            data = self._get({
                "action": "query", "titles": title, "redirects": 1,
                "prop": "info|pageprops|redirects", "ppprop": "disambiguation",
                "rdnamespace": 0, "rdlimit": "max",
            })
            page = data["query"]["pages"][0]
            return {
                "title": page["title"],
                "exists": not page.get("missing", False) and "invalid" not in page,
                "disambiguation": "disambiguation" in page.get("pageprops", {}),
                "revid": page.get("lastrevid"),
                "aliases": [r["title"] for r in page.get("redirects", [])],
            }
        return Page(**self._cached("resolve", title, fetch))

    def links(self, title):
        return self._cached("links", title, lambda: self._continued(
            {"action": "query", "titles": title, "prop": "links",
             "plnamespace": 0, "pllimit": "max"}, "links"))

    def summary(self, title):
        def fetch():
            data = self._get({
                "action": "query", "titles": title, "prop": "extracts",
                "exintro": 1, "explaintext": 1, "exsentences": 3,
            })
            return data["query"]["pages"][0].get("extract", "")
        return self._cached("summary", title, fetch)

    def backlinks(self, title):
        # One page of results on purpose: a hint for likely last hops, not a crawl.
        def fetch():
            data = self._get({
                "action": "query", "list": "backlinks", "bltitle": title,
                "blnamespace": 0, "bllimit": 500, "blfilterredir": "nonredirects",
            })
            return [x["title"] for x in data["query"]["backlinks"]]
        return self._cached("backlinks", title, fetch)
