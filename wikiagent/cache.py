"""Bounded page cache in one SQLite file.

The size cap and LRU eviction stop a wide or BFS-style search from filling the
cache. Pinned entries are never evicted, so eval pages stay replayable.
"""
import json
import sqlite3
import time
import zlib
from pathlib import Path


class PageCache:
    def __init__(self, path="data/cache.sqlite", max_bytes=200_000_000):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.max_bytes = max_bytes
        self.hits = self.misses = self.evictions = 0
        self._tick = 0
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS entries(
                kind TEXT, key TEXT, revid INTEGER, fetched_at REAL,
                last_used REAL, pinned INTEGER DEFAULT 0, size INTEGER,
                payload BLOB, PRIMARY KEY(kind, key))"""
        )
        self.db.commit()

    def _now(self):
        # Strictly increasing, so LRU order holds within one clock tick.
        self._tick += 1
        return time.time() + self._tick * 1e-6

    def get(self, kind, key):
        row = self.db.execute(
            "SELECT payload FROM entries WHERE kind=? AND key=?", (kind, key)
        ).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        self.db.execute(
            "UPDATE entries SET last_used=? WHERE kind=? AND key=?",
            (self._now(), kind, key),
        )
        self.db.commit()
        return json.loads(zlib.decompress(row[0]))

    def put(self, kind, key, value, revid=None):
        blob = zlib.compress(json.dumps(value).encode())
        now = self._now()
        self.db.execute(
            """INSERT INTO entries(kind,key,revid,fetched_at,last_used,size,payload)
               VALUES(?,?,?,?,?,?,?)
               ON CONFLICT(kind,key) DO UPDATE SET revid=excluded.revid,
               fetched_at=excluded.fetched_at, last_used=excluded.last_used,
               size=excluded.size, payload=excluded.payload""",
            (kind, key, revid, now, now, len(blob), blob),
        )
        self.db.commit()
        self._evict(keep=(kind, key))

    def _evict(self, keep):
        total = self.total_bytes()
        if total <= self.max_bytes:
            return
        rows = self.db.execute(
            "SELECT kind,key,size FROM entries WHERE pinned=0 ORDER BY last_used"
        ).fetchall()
        for kind, key, size in rows:
            if total <= self.max_bytes:
                break
            if (kind, key) == keep:
                continue
            self.db.execute("DELETE FROM entries WHERE kind=? AND key=?", (kind, key))
            total -= size
            self.evictions += 1
        self.db.commit()

    def pin(self, kind, key):
        self.db.execute(
            "UPDATE entries SET pinned=1 WHERE kind=? AND key=?", (kind, key)
        )
        self.db.commit()

    def total_bytes(self):
        return self.db.execute("SELECT COALESCE(SUM(size),0) FROM entries").fetchone()[0]

    def stats(self):
        entries, pinned = self.db.execute(
            "SELECT COUNT(*), COALESCE(SUM(pinned),0) FROM entries"
        ).fetchone()
        return {
            "entries": entries, "pinned": pinned, "bytes": self.total_bytes(),
            "max_bytes": self.max_bytes, "hits": self.hits, "misses": self.misses,
            "evictions": self.evictions,
        }
