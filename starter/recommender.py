"""Baseline recommender: the all-time best sellers that are in stock.

    python starter/recommender.py        # reads TEAM_ID and RAWTREE_API_KEY from .env or the environment

pull → predict → push: it polls shop.requests, answers every new request and inserts the
answers in batches into team_NN.submissions. recommend() is the part to replace: it is
exactly what the leaderboard calls "baseline", and what shoppers see when an answer is
missing, late or invalid (the fallback).
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import requests

for line in Path(".env").read_text().splitlines() if Path(".env").exists() else []:
    key, _, value = line.partition("=")
    os.environ.setdefault(key.strip(), value.strip())
API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
TEAM = os.environ["TEAM_ID"]
http = requests.Session()
http.headers["Authorization"] = f"Bearer {os.environ['RAWTREE_API_KEY']}"

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    from model.predictor import Predictor
    _pred = Predictor()
    MODEL_VERSION = _pred.VERSION
except Exception as e:
    _pred = None
    MODEL_VERSION = "baseline"
    print(f"model not loaded ({e}); answering with the best-seller fallback")

SALES = """
SELECT product_id, count() AS n
FROM shop.history_events
WHERE event_type = 'purchase'{sales}
GROUP BY product_id"""
"""How many times each product was bought in the historical events."""

NEW_REQUESTS = """
SELECT request_id, ts, user_id, is_new_user, context, recent_views
FROM shop.requests
WHERE ts > now() - INTERVAL 2 MINUTE
ORDER BY ts DESC
LIMIT 200"""
"""Recent unanswered candidates. The 2-minute window keeps the scan cheap all
weekend; 200 rows absorb arrival bursts that a 50-row window would miss."""

OUT_OF_STOCK = """
SELECT product_id::String AS product_id
FROM shop.catalog_updates
WHERE change::String IN ('out_of_stock', 'back_in_stock')
GROUP BY product_id
HAVING argMax(change::String, ts::DateTime64(3)) = 'out_of_stock'"""
"""Products whose latest stock change is a stock-out. Used once at startup
and as fallback; afterwards deltas keep the set current (see StockTracker)."""

STOCK_DELTA = """
SELECT product_id::String AS product_id, change::String AS change, toString(ts) AS ts
FROM shop.catalog_updates
WHERE ts > toDateTime64('{since}', 3) AND change::String IN ('out_of_stock', 'back_in_stock')
ORDER BY ts"""
"""Stock changes since a watermark timestamp. Applying a handful of rows is
much cheaper than re-scanning the whole update log every second, and the set
reacts to 'back_in_stock' within a second (fewer invalid answers)."""

Record = dict[str, Any]
"""RawTree returns rows as JSON objects: {"column": value, ...}"""


def query(sql: str) -> list[Record]:
    r = http.post(f"{API}/v1/query", json={"sql": sql}, timeout=10)
    r.raise_for_status()
    return r.json()["data"]


def best_sellers(until: str | None = None) -> list[str]:
    """Every product ever bought, best-selling first (ties: higher product_id first).

    `until` ('YYYY-MM-DD') restricts the history the ranking is built from, for offline
    evaluation. Live it is computed once, at start: the ranking never refreshes.
    """
    sales_cut = f" AND ts < '{until}'" if until else ""
    sales = sorted(((int(r["n"]), r["product_id"]) for r in query(SALES.format(sales=sales_cut))), reverse=True)
    return [product for _, product in sales]


def out_of_stock() -> set[str]:
    """Products out of stock right now. Recommending one makes the whole answer invalid."""
    return {r["product_id"] for r in query(OUT_OF_STOCK)}


class StockTracker:
    """Keeps a shared dict's 'unavailable' set current via watermark deltas.

    Seeds once with a full scan, then applies only the new stock changes each
    second; any failure falls back to a full re-seed, so correctness never
    depends on the delta path."""

    def __init__(self, state: dict) -> None:
        """One full scan + remember the newest update timestamp as watermark."""
        self.state = state
        self.reseed()

    def reseed(self) -> None:
        """Full scan (also the fallback after any delta error)."""
        self.state["unavailable"] = out_of_stock()
        self.since = query("SELECT toString(max(ts)) AS wm FROM shop.catalog_updates")[0]["wm"]

    def update(self) -> None:
        """Apply the stock changes since the last watermark."""
        try:
            rows = query(STOCK_DELTA.format(since=self.since))
        except Exception as e:
            print(f"stock delta failed ({e}); full re-seed")
            try:
                self.reseed()
            except Exception as e2:
                print(f"stock re-seed failed: {e2}")
            return
        if rows:
            u = set(self.state["unavailable"])
            for r in rows:
                (u.add if r["change"] == "out_of_stock" else u.discard)(r["product_id"])
            self.state["unavailable"] = u
            self.since = rows[-1]["ts"]


def pull(seen: set[str]) -> list[Record]:
    """New shop.requests records not answered yet."""
    fresh = []
    for req in query(NEW_REQUESTS):
        if req["request_id"] not in seen:
            seen.add(req["request_id"])
            fresh.append(req)
    return fresh


def predict(requests: list[Record], best: list[str], unavailable: set[str]) -> list[Record]:
    """One submission per request."""
    answers = []
    model_items: dict[str, list[str]] = {}
    if _pred is not None and requests:
        try:
            model_items = _pred.recommend_batch(requests, unavailable)
        except Exception as e:
            print(f"model batch failed ({e}); falling back to best sellers")
            model_items = {}
    for req in requests:
        items = model_items.get(req["request_id"])
        version = MODEL_VERSION
        if not items or len(items) != 10:
            items = recommend(req.get("recent_views") or [], best, unavailable)
            version = "baseline"
        answers.append({"request_id": req["request_id"], "items": items, "model_version": version})
    return answers


def recommend(recent_views: list[Record], best: list[str], unavailable: set[str] = frozenset()) -> list[str]:
    """10 product IDs for one request: the part teams replace. Here: the first 10 best sellers in stock."""
    return [product for product in best if product not in unavailable][:10]


def push(answers: list[Record]) -> None:
    """Batch insert into team_NN.submissions."""
    http.post(f"{API}/v1/tables/submissions", params={"database": TEAM}, json=answers, timeout=10).raise_for_status()
    print(f"answered {len(answers)}")


CHUNK = 10
"""Answers are predicted and pushed in chunks of this size, so the first
requests of a poll never wait for the featurizing of the last ones."""


def maintainer(state: dict) -> None:
    """Background thread for the queries that do not have to block answering:
    the stock set (watermark deltas, every second) and the model's catalogue /
    trend signals (every minute). Failures keep the last known values; the
    answering loop only ever reads the shared dict."""
    signal_t = 0.0
    stock = StockTracker(state)
    while True:
        now = time.time()
        stock.update()
        try:
            if _pred is not None and now - signal_t >= 60:
                signal_t = now
                _pred.refresh()
        except Exception as e:
            print(f"signal refresh failed ({e}); keeping last known signals")
        time.sleep(0.2)


def main() -> None:
    """pull → predict → push forever. All RawTree maintenance (stock, catalogue,
    trends) runs in a background thread, so the only thing the answering loop
    ever waits for is predicting and pushing the chunk it is holding."""
    best: list[str] = []
    seen = set[str]()
    state = {"unavailable": set[str]()}
    threading.Thread(target=maintainer, args=(state,), daemon=True).start()
    while True:
        try:
            if not best:
                best = best_sellers()
            reqs = pull(seen)
            for i in range(0, len(reqs), CHUNK):
                answers = predict(reqs[i:i + CHUNK], best, state["unavailable"])
                if answers:
                    push(answers)
        except Exception as e:  # keep running: shoppers keep arriving while you sleep
            print(f"error: {e}")
            time.sleep(2)
        time.sleep(0.1)


if __name__ == "__main__":
    main()
