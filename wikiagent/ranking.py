"""Rank links by closeness to the target. No model choice happens here."""
import re

import numpy as np

from .graph import ToolError

STOP = set("a an and as at by for from in is it of on or that the to was were with".split())


def tokens(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP}


class Ranker:
    def __init__(self, embedder=None, cache=None, max_embed=300):
        self.embedder, self.cache, self.max_embed = embedder, cache, max_embed
        self.fallbacks = 0

    def rank(self, links, target, target_summary, backlinks=()):
        """Return [{title, score, links_to_target}], best first."""
        wanted = tokens(f"{target} {target_summary}")
        back = set(backlinks)
        lexical = {t: len(tokens(t) & wanted) / (len(tokens(t)) or 1) for t in links}
        order = sorted(links, key=lambda t: (t not in back, -lexical[t], t))
        scores = self._embedding_scores(order[: self.max_embed], target, target_summary)
        if scores is None:
            scores = lexical
        rows = [{"title": t, "score": round(float(scores.get(t, -1.0)), 4),
                 "links_to_target": t in back} for t in links]
        return sorted(rows, key=lambda r: (not r["links_to_target"], -r["score"], r["title"]))

    def _vectors(self, titles):
        found, missing = {}, []
        for t in titles:
            v = self.cache.get("embed", t) if self.cache else None
            if v is None:
                missing.append(t)
            else:
                found[t] = v
        if missing:
            for t, v in zip(missing, self.embedder.embed(missing)):
                found[t] = v
                if self.cache:
                    self.cache.put("embed", t, v)
        return found

    def _embedding_scores(self, titles, target, summary):
        if self.embedder is None or not titles:
            return None
        try:
            query = self.embedder.embed([f"{target}: {summary}"[:1500]])[0]
            vectors = self._vectors(titles)
        except (ToolError, KeyError) as e:
            self.fallbacks += 1  # ranking is a hint, so a failure must not end the run
            print(f"  [ranker] embeddings failed, using word overlap: {e}")
            return None
        names = list(vectors)
        m = np.array([vectors[n] for n in names], dtype=float)
        q = np.array(query, dtype=float)
        m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
        q /= np.linalg.norm(q) + 1e-9
        return dict(zip(names, m @ q))
