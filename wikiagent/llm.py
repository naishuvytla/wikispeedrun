"""Thin Gemini REST client: JSON chat, embeddings, retries, usage logging."""
import json
import os
import time

import requests

from .graph import ToolError

BASE = "https://generativelanguage.googleapis.com/v1beta/models"
RETRYABLE = {429, 500, 502, 503, 504}


class Gemini:
    def __init__(self, api_key=None, model=None, embed_model=None, session=None,
                 min_interval=None, retries=4, backoff=2.0, sleep=time.sleep):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise SystemExit("Set GEMINI_API_KEY first, or use --agent baseline.")
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
        self.embed_model = embed_model or os.environ.get(
            "GEMINI_EMBED_MODEL", "gemini-embedding-001")
        self.session = session or requests.Session()
        # Pacing for free-tier per-minute limits. Lower it if your quota allows.
        self.min_interval = float(os.environ.get("GEMINI_MIN_INTERVAL", 4.0)) \
            if min_interval is None else min_interval
        self.retries, self.backoff, self.sleep = retries, backoff, sleep
        self.calls = 0
        self.embed_calls = 0
        self.input_tokens = self.output_tokens = 0
        self.model_version = None
        self._last = 0.0

    def _post(self, url, body):
        last_error = "unknown"
        for attempt in range(self.retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                self.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self.session.post(url, json=body, timeout=60,
                                      headers={"x-goog-api-key": self.api_key})
                if r.status_code not in RETRYABLE:
                    if r.status_code >= 400:
                        raise ToolError(f"Gemini HTTP {r.status_code}: {r.text[:200]}")
                    return r.json()
                last_error = f"HTTP {r.status_code}"
            except (requests.Timeout, requests.ConnectionError) as e:
                last_error = type(e).__name__
            if attempt < self.retries:
                self.sleep(self.backoff * 2**attempt)
        raise ToolError(f"Gemini gave up after retries: {last_error}")

    def choose_json(self, system, user):
        self.calls += 1
        data = self._post(f"{BASE}/{self.model}:generateContent", {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0,
                                 "responseMimeType": "application/json"},
        })
        self.model_version = data.get("modelVersion", self.model)
        usage = data.get("usageMetadata", {})
        self.input_tokens += usage.get("promptTokenCount", 0)
        self.output_tokens += usage.get("candidatesTokenCount", 0)
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text)
        except (KeyError, IndexError, ValueError) as e:
            raise ToolError(f"bad model output: {e}") from e

    def embed(self, texts):
        vectors = []
        for i in range(0, len(texts), 100):  # API batch limit
            self.embed_calls += 1
            data = self._post(f"{BASE}/{self.embed_model}:batchEmbedContents", {
                "requests": [{
                    "model": f"models/{self.embed_model}",
                    "content": {"parts": [{"text": t}]},
                    "outputDimensionality": 256,  # small vectors keep the cache small
                } for t in texts[i:i + 100]]
            })
            vectors += [e["values"] for e in data["embeddings"]]
        return vectors
