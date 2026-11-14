"""Serving: scores candidate products per request with the trained model.

    from model.predictor import Predictor
    p = Predictor()                       # loads model/out/* + features/out/*
    p.recommend(request_record, unavailable)   -> 10 product ids

Candidate generation = co-view neighbours of the last viewed product + stars
of the session's categories + live trending + a few random explorers + best
sellers as filler. Feature assembly goes through model/featurize.py, the same
code path used for training, so live rows look exactly like training rows.

Four background loops (call them from the serving thread, catch exceptions):
  refresh()          catalogue + trends, gated by a watermark
  refresh_cov()      today's co-visitation from live shop.events
  refresh_feedback() empirical per-product conversion from YOUR OWN served
                     traffic (team feedback), applied as a score multiplier
  add_purchases()    keeps the already-bought exclusion current; shoppers
                     who own a product rarely buy it again, so those slots
                     are freed for buyable items.
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

TEAM = os.environ.get("TEAM_ID", "team_NN")

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
SELECT count() AS n, max(ts::UInt64) AS mx
FROM shop.catalog_updates
WHERE ts::UInt64 > {since}"""
"""Whether any catalogue update exists after a watermark (epoch seconds;
RawTree's ts is a String of unix seconds, so compare as integers): one tiny
query that tells refresh() whether the full catalogue re-fetch is worth it."""

TRENDS_SQL = """
SELECT product_id::String AS product_id,
       countIf(event_type::String='view') AS views_24h,
       countIf(event_type::String='purchase') AS purchases_24h
FROM shop.events
WHERE ts > now() - INTERVAL 24 HOUR
GROUP BY product_id"""
"""Last-24h view/purchase counts, the live half of the trend-lift ratio."""

BOUGHT_SQL = """
SELECT user_id::String AS user_id, groupUniqArray(product_id::String) AS pids
FROM shop.history_events
WHERE event_type::String = 'purchase'
GROUP BY user_id"""
"""Everything every shopper ever bought: the already-bought exclusion map."""

LIVE_PURCHASES_SQL = """
SELECT user_id::String AS user_id, product_id::String AS product_id, ts::UInt64 AS ts
FROM shop.events
WHERE ts::UInt64 >= {since} AND event_type::String = 'purchase'
ORDER BY ts"""
"""Live purchases since a watermark (epoch seconds), folded into the bought
map; the >= re-delivery of the boundary second is harmless (set.add)."""

CVR_SQL = """
SELECT p::String AS product_id, count() AS shown, countIf(purchased != '') AS bought
FROM (
    SELECT arrayJoin(shown::Array(String)) AS p, purchased::String AS purchased
    FROM {team}.feedback
    WHERE request_id != '__init__' AND served_by::String = 'team'
) GROUP BY product_id"""
"""Exposure and purchases per product from OUR served traffic: the empirical
side of the score multiplier."""

LIVE_COV_SQL = """
SELECT p::String AS p, q::String AS q, count() AS n
FROM (
    SELECT session_id::String AS sid, product_id::String AS p, ts::DateTime64(3) AS ts
    FROM shop.events
    WHERE event_type::String = 'view' AND ts::DateTime64(3) >= now() - INTERVAL 24 HOUR
) a
INNER JOIN (
    SELECT session_id::String AS sid, product_id::String AS q, ts::DateTime64(3) AS ts
    FROM shop.events
    WHERE event_type::String = 'view' AND ts::DateTime64(3) >= now() - INTERVAL 24 HOUR
) b ON a.sid = b.sid AND a.p != b.q
WHERE a.ts <= b.ts
GROUP BY p, q
ORDER BY n DESC
LIMIT 30 BY p"""
"""Today's co-visitation, same directional pattern as the offline build
(a viewed before b) but over the live stream, so neighbourhoods follow
today's taste without retraining."""


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
            self.VERSION = "pchavarria-v4"
        else:
            self.model = joblib.load(d / "model.joblib")
            self.backend = "sklearn-histgb"
            self.VERSION = "gbdt-v1"
        self.fs = FeatureStore.load()
        self._api = api or os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
        self._http = requests.Session()
        self._http.headers["Authorization"] = f"Bearer {os.environ['RAWTREE_API_KEY']}"
        self.rng = np.random.default_rng(0)
        self.bought: dict[str, set[str]] | None = None
        self.cvr_mult: dict[str, float] = {}
        self._live_catalog()
        self.load_bought()
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

    def _max_update_ts(self) -> int:
        """Newest shop.catalog_updates timestamp (unix seconds), watermark for
        refresh(). Returns 0 on failure, which just forces a rebuild."""
        try:
            r = self._http.post(f"{self._api}/v1/query", json={"sql": "SELECT max(ts::UInt64) AS wm FROM shop.catalog_updates"}, timeout=10)
            r.raise_for_status()
            return int(r.json()["data"][0]["wm"])
        except Exception:
            return 0

    def refresh(self) -> None:
        """Cheap periodic refresh for long runs: the trend list is rebuilt from
        the last 24 h of events, but the heavy catalogue re-fetch only happens
        when catalog_updates actually has new rows (one watermark query). Call
        it every minute or so; catch exceptions in the caller."""
        try:
            r = self._http.post(f"{self._api}/v1/query",
                                json={"sql": CATALOG_WM_SQL.format(since=self._cat_wm)}, timeout=10)
            r.raise_for_status()
            row = r.json()["data"][0]
            if int(row["n"]) > 0:
                self._live_catalog()
                self._cat_wm = int(row["mx"])
        except Exception:
            self._live_catalog()  # watermark lost: be safe and rebuild
            self._cat_wm = self._max_update_ts()
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

    def _max_purchase_ts(self) -> int:
        """Newest live purchase timestamp (unix seconds), watermark for
        add_purchases(). Returns 0 on failure, which just re-reads more."""
        try:
            r = self._http.post(f"{self._api}/v1/query",
                                json={"sql": "SELECT max(ts::UInt64) AS wm FROM shop.events WHERE event_type::String = 'purchase'"},
                                timeout=10)
            r.raise_for_status()
            return int(r.json()["data"][0]["wm"])
        except Exception:
            return 0

    def load_bought(self) -> None:
        """One-time map of every shopper's already-bought products (history).
        The simulator makes repeat purchases unlikely, so a recommended
        product the shopper owns wastes a slot. On failure the map stays None
        (no exclusion) and the next add_purchases() retries the full load."""
        try:
            r = self._http.post(f"{self._api}/v1/query", json={"sql": BOUGHT_SQL}, timeout=120)
            r.raise_for_status()
            self.bought = {row["user_id"]: set(row["pids"]) for row in r.json()["data"]}
            self._purch_wm = self._max_purchase_ts()
        except Exception as e:
            self.bought = None
            print(f"bought-map load failed ({e}); will retry on the next tick")

    def add_purchases(self) -> None:
        """Fold live purchases (shop.events watermark deltas) into the bought
        map, so a shopper who just bought something stops being shown it."""
        if self.bought is None:
            self.load_bought()
            if self.bought is None:
                return
        r = self._http.post(f"{self._api}/v1/query",
                            json={"sql": LIVE_PURCHASES_SQL.format(since=self._purch_wm)}, timeout=30)
        r.raise_for_status()
        rows = r.json()["data"]
        for row in rows:
            self.bought.setdefault(row["user_id"], set()).add(row["product_id"])
        if rows:
            self._purch_wm = max(int(row["ts"]) for row in rows)

    CVR_SHRINK = 20
    """Pseudo-count pulling small-sample conversion towards the overall rate."""
    CVR_ALPHA = 1.0
    """Strength of the empirical-vs-model correction (1 = gentle)."""

    def refresh_feedback(self) -> None:
        """Empirical per-product conversion from our own served traffic
        (team feedback), shrunk towards the overall rate and turned into a
        multiplicative score factor. Products the simulator likes more than
        the model predicts get shown more, which also sharpens the estimate:
        a soft bandit over the recommendation policy."""
        r = self._http.post(f"{self._api}/v1/query", json={"sql": CVR_SQL.format(team=TEAM)}, timeout=60)
        r.raise_for_status()
        rows = r.json()["data"]
        shown = sum(row["shown"] for row in rows)
        bought = sum(row["bought"] for row in rows)
        if shown < 500 or bought <= 0:
            return  # not enough signal yet
        overall = bought / shown
        mult = {}
        for row in rows:
            s, b = row["shown"], row["bought"]
            cvr = (b + self.CVR_SHRINK * overall) / (s + self.CVR_SHRINK)
            mult[row["product_id"]] = float(
                np.clip(1 + self.CVR_ALPHA * (cvr - overall) / overall, 0.6, 1.6))
        self.cvr_mult = mult

    def refresh_cov(self) -> None:
        """Rebuild the live co-visitation dict (last 24 h of shop.events) used
        as an extra candidate source in _candidates."""
        r = self._http.post(f"{self._api}/v1/query", json={"sql": LIVE_COV_SQL}, timeout=120)
        r.raise_for_status()
        cov: dict[str, dict[str, int]] = {}
        for row in r.json()["data"]:
            cov.setdefault(row["p"], {})[row["q"]] = int(row["n"])
        self.fs.cov_live = cov

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
        product (all-time, 14-day and today's live windows), star products of
        the viewed and the shopper's favourite categories, trending, best
        sellers and a few random explorers. Only active, in-stock candidates
        are accepted, so nothing reaches the scorer just to be dropped later."""
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
            d1, d2, d3 = fs.cov.get(pid), fs.cov14.get(pid), fs.cov_live.get(pid)
            if d1:
                add(sorted(d1, key=d1.get, reverse=True)[:50])
            if d2:
                add(sorted(d2, key=d2.get, reverse=True)[:50])
            if d3:
                add(sorted(d3, key=d3.get, reverse=True)[:30])
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
        per request_id. Model scores are multiplied by the empirical-feedback
        factor (cvr_mult), products the shopper already bought are skipped,
        and the list fills greedily from the best score down, then from best
        sellers if the pool runs short."""
        rows = [self._base_row(r) for r in reqs]
        ids = [r["request_id"] for r in rows]
        uid = {r["request_id"]: r["user_id"] for r in rows}
        reqs_df = pd.DataFrame(rows)
        base = session_base(reqs_df, self.fs).set_axis(pd.Index(ids, name="request_id"))
        cands = {rid: self._candidates(row, unavailable) for row, rid in zip(rows, ids)}
        slots = pd.DataFrame([(rid, p) for rid in ids for p in cands[rid]],
                             columns=["request_id", "pid"])
        out = {rid: [] for rid in ids}
        bought = self.bought or {}
        if len(slots):
            j = attach_candidates(base, slots, self.fs)
            X = feature_matrix(j, self.cats)
            j["score"] = self._score(X) * j.pid.map(self.cvr_mult).fillna(1.0)
            for rid, g in j.groupby("request_id", sort=False):
                order = list(g.sort_values("score", ascending=False, na_position="last").pid)
                ranked, taken = [], set()
                mine = bought.get(uid[rid], ())
                for p in order + self.fs.best:  # scored extras, then best-seller filler
                    p = str(p)
                    if p in taken or p in unavailable or p in mine:
                        continue
                    taken.add(p)
                    ranked.append(p)
                    if len(ranked) == 10:
                        break
                out[rid] = ranked
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
