"""Serving: scores candidate products per request with the trained model.

    from model.predictor import Predictor
    p = Predictor()                       # loads model/out/* + features/out/*
    p.recommend(request_record, unavailable)   -> 10 product ids

Candidate generation = co-view neighbours of the last viewed product + stars
of the session's categories + live trending + a few random explorers + best
sellers as filler. Feature assembly goes through model/featurize.py, the same
code path used for training, so live rows look exactly like training rows.
The catalogue is refreshed from RawTree at startup (catalog + catalog_updates)
so new products and price changes are visible; refresh() redoes it together
with the live trend list, call it periodically during long runs.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _line in (ROOT / ".env").read_text().splitlines() if (ROOT / ".env").exists() else []:
    _k, _, _v = _line.partition("=")
    if _k.strip() and not _k.strip().startswith("#"):
        os.environ.setdefault(_k.strip(), _v.strip())

from model.featurize import FeatureStore, attach_candidates, feature_matrix, session_base

CATALOG_SQL = """
SELECT product_id::String AS product_id,
       argMax(category::String, v)    AS category,
       argMax(subcategory::String, v) AS subcategory,
       argMax(brand::String, v)       AS brand,
       argMax(price_eur::Float64, v)  AS price,
       argMax(is_active::Bool, v)     AS is_active
FROM (
    SELECT product_id::String AS product_id, category::String AS category, subcategory::String AS subcategory,
           brand::String AS brand, price_eur::Float64 AS price_eur, is_active::Bool AS is_active, '' AS v
    FROM shop.catalog
    UNION ALL
    SELECT product_id::String, category::String, subcategory::String, brand::String,
           price_eur::Float64, is_active::Bool, toString(ts)
    FROM shop.catalog_updates
)
GROUP BY product_id"""

CATALOG_WM_SQL = """
SELECT toString(max(ts)) AS wm
FROM shop.catalog_updates
WHERE ts > toDateTime64('{since}', 3)"""
"""Newest catalogue update after a watermark: one tiny query that tells
refresh() whether the full catalogue re-fetch is worth doing at all."""

TRENDS_SQL = """
SELECT product_id::String AS product_id,
       countIf(event_type::String='view') AS views_24h,
       countIf(event_type::String='purchase') AS purchases_24h
FROM shop.events
WHERE ts > now() - INTERVAL 24 HOUR
GROUP BY product_id"""
"""Last-24h view/purchase counts, the live half of the trend-lift ratio."""


class Predictor:
    VERSION = "gbdt-v1"

    def __init__(self, model_dir: Path | None = None, api: str | None = None):
        d = Path(model_dir or ROOT / "model" / "out")
        self.meta = json.loads((d / "meta.json").read_text())
        self.cats = self.meta["categories"]
        lgb_path = d / self.meta.get("model_file", "model_lgbm.txt")
        if lgb_path.exists():
            import lightgbm as lgb
            self.model = lgb.Booster(model_file=str(lgb_path))
            self.backend = "lightgbm"
            self.VERSION = "pchavarria-v2"
        else:
            self.model = joblib.load(d / "model.joblib")
            self.backend = "sklearn-histgb"
            self.VERSION = "gbdt-v1"
        self.fs = FeatureStore.load()
        self._api = api or os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
        self._http = requests.Session()
        self._http.headers["Authorization"] = f"Bearer {os.environ['RAWTREE_API_KEY']}"
        self.rng = np.random.default_rng(0)
        self._live_catalog()
        print(f"predictor ready: {self.VERSION} · {len(self.fs.products):,} products "
              f"({int(self.fs.products.is_active.sum()):,} active)")

    def _live_catalog(self) -> None:
        r = self._http.post(f"{self._api}/v1/query", json={"sql": CATALOG_SQL}, timeout=60)
        r.raise_for_status()
        live = pd.DataFrame(r.json()["data"])
        stats = self.fs.products.set_index("product_id")
        m = live.set_index("product_id").join(
            stats[["purchase_rate", "cart_rate", "buyers", "subcat_sales_rank",
                   "cat_sales_share", "price_vs_subcat"]], how="left")
        self.fs.products = m.reset_index()
        self.fs.pidx = self.fs.products.set_index("product_id")
        active = self.fs.products[self.fs.products.is_active == True]
        self.fs.best = active.sort_values("buyers", ascending=False).product_id.tolist()
        self.fs.cat_stars = {c: g[g.is_active == True].sort_values(
            "buyers", ascending=False).product_id.head(40).tolist()
            for c, g in self.fs.products.groupby("category")}
        self.active = set(active.product_id)
        self._active_list = active.product_id.tolist()
        self._cat_wm = self._max_update_ts()

    def _max_update_ts(self) -> str:
        """Newest shop.catalog_updates timestamp, as watermark for refresh().
        Returns the epoch on failure, which just forces a rebuild next time."""
        try:
            r = self._http.post(f"{self._api}/v1/query", json={"sql": "SELECT toString(max(ts)) AS wm FROM shop.catalog_updates"}, timeout=10)
            r.raise_for_status()
            return r.json()["data"][0]["wm"]
        except Exception:
            return "1970-01-01 00:00:00.000"

    def refresh(self) -> None:
        """Cheap periodic refresh for long runs: the trend list is rebuilt from
        the last 24 h of events, but the heavy catalogue re-fetch only happens
        when catalog_updates actually has new rows (one watermark query). Call
        it every minute or so; catch exceptions in the caller."""
        try:
            r = self._http.post(f"{self._api}/v1/query",
                                json={"sql": CATALOG_WM_SQL.format(since=self._cat_wm)}, timeout=10)
            r.raise_for_status()
            wm = r.json()["data"][0]["wm"]
        except Exception:
            wm = None  # watermark lost: be safe and rebuild the catalogue
        if wm is None or wm > self._cat_wm:
            self._live_catalog()
            self._cat_wm = wm if wm is not None else self._max_update_ts()
        r = self._http.post(f"{self._api}/v1/query", json={"sql": TRENDS_SQL}, timeout=60)
        r.raise_for_status()
        live = pd.DataFrame(r.json()["data"])
        if live.empty or self.fs.trends.empty:
            return
        m = self.fs.trends.drop(columns=["views_24h", "purchases_24h", "view_lift", "purchase_lift"],
                                errors="ignore").merge(live, on="product_id", how="left")
        m[["views_24h", "purchases_24h"]] = m[["views_24h", "purchases_24h"]].fillna(0)
        m["view_lift"] = m.views_24h / (m.views_hist / 60).replace(0, np.nan)
        m["purchase_lift"] = m.purchases_24h / (m.purchases_hist / 60).replace(0, np.nan)
        self.fs.trends = m.sort_values("purchase_lift", ascending=False)
        self.fs.trend = self.fs.trends[self.fs.trends.views_24h >= 20].product_id.head(120).tolist()

    def _base_row(self, req: dict) -> dict:
        """Session features straight from the raw request record."""
        ctx = req.get("context") or {}
        rv = [v for v in (req.get("recent_views") or []) if isinstance(v, dict)]
        ids = [v["product_id"] for v in rv]
        return {"request_id": req["request_id"],
                "ts": req.get("ts"),
                "user_id": req.get("user_id"),
                "page": ctx.get("page", "home"),
                "device": ctx.get("device", "web"),
                "region": ctx.get("region", ""),
                "n_views": len(ids),
                "last_view": max(rv, key=lambda x: x.get("ts", ""))["product_id"] if rv else "",
                "recent_ids": ids}

    def _candidates(self, row: dict, unavailable: set) -> list:
        """Candidate pool for one request: co-view neighbours of every viewed
        product (all-time and 14-day windows), star products of the viewed and
        the shopper's favourite categories, trending, best sellers and a few
        random explorers. Only active, in-stock candidates are accepted, so
        nothing reaches the scorer just to be dropped later."""
        fs = self.fs
        cands, seen = [], set()

        def add(pids):
            for p in pids:
                if p not in seen and p not in unavailable and p in self.active:
                    seen.add(p)
                    cands.append(p)

        cats = fs.pidx["category"]
        for pid in dict.fromkeys([row["last_view"], *row["recent_ids"][:3]]):
            if not pid:
                continue
            d1, d2 = fs.cov.get(pid), fs.cov14.get(pid)
            if d1:
                add(sorted(d1, key=d1.get, reverse=True)[:50])
            if d2:
                add(sorted(d2, key=d2.get, reverse=True)[:50])
            cat = cats.get(pid)
            if isinstance(cat, str):
                add(fs.cat_stars.get(cat, []))
        u = fs.user_by_id.get(row["user_id"])
        if u is not None and isinstance(getattr(u, "top_category", None), str):
            add(fs.cat_stars.get(u.top_category, []))
        add(fs.trend[:150])
        add(fs.best[:80])
        add(list(self.rng.choice(self._active_list, size=20, replace=False)))
        return cands[:400]

    def _score(self, X) -> np.ndarray:
        """Raw model score per candidate row (higher = more likely bought)."""
        if self.backend == "lightgbm":
            return self.model.predict(X)
        return self.model.predict_proba(X)[:, 1]

    def recommend_batch(self, reqs: list[dict], unavailable: set) -> dict:
        """Score every request's candidates and return the 10 best product ids
        per request_id. When the pool leaves the list short, next-best scored
        candidates fill it before best sellers do."""
        rows = [self._base_row(r) for r in reqs]
        ids = [r["request_id"] for r in rows]
        reqs_df = pd.DataFrame(rows)
        base = session_base(reqs_df, self.fs).set_axis(pd.Index(ids, name="request_id"))
        cands = {rid: self._candidates(row, unavailable) for row, rid in zip(rows, ids)}
        slots = pd.DataFrame([(rid, p) for rid in ids for p in cands[rid]],
                             columns=["request_id", "pid"])
        out = {rid: [] for rid in ids}
        if len(slots):
            j = attach_candidates(base, slots, self.fs)
            X = feature_matrix(j, self.cats)
            scores = self._score(X)
            j["score"] = scores
            for rid, g in j.groupby("request_id", sort=False):
                order = g.sort_values("score", ascending=False, na_position="last")
                ranked = [str(p) for p in order.pid.head(10).tolist()]
                for p in order.pid.iloc[10:]:  # scored extras beat best-seller filler
                    p = str(p)
                    if len(ranked) >= 10:
                        break
                    if p not in ranked and p not in unavailable:
                        ranked.append(p)
                for p in self.fs.best:  # safety net if the pool is tiny
                    if len(ranked) >= 10:
                        break
                    if p not in ranked and p not in unavailable:
                        ranked.append(p)
                out[rid] = ranked[:10]
        return out

    def recommend(self, req: dict, unavailable: set = frozenset()) -> list:
        """Top 10 product ids for a single request record."""
        return self.recommend_batch([req], set(unavailable)).get(req["request_id"], [])


if __name__ == "__main__":
    t0 = time.time()
    p = Predictor()
    print(f"init in {time.time() - t0:.1f}s")
    fake = {"request_id": "r_test1",
            "ts": pd.Timestamp.now(tz="UTC").isoformat(),
            "user_id": list(p.fs.user_by_id)[0],
            "context": {"page": "product", "device": "ios", "region": "Catalunya"},
            "recent_views": [{"product_id": p.fs.trend[0], "ts": "2026-11-14T12:00:00Z"}]}
    t0 = time.time()
    items = p.recommend(fake)
    print(f"1 request in {(time.time() - t0) * 1000:.0f} ms")
    cats = p.fs.products.set_index("product_id")["category"]
    print("items:", [(i, cats.get(i)) for i in items])
