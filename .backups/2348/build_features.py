"""Feature engineering pipeline: reference tables the recommender scores from.

    python features/build_features.py              # all sections
    python features/build_features.py coviews      # one section only

All heavy aggregation runs in RawTree SQL; this script only assembles, derives
a few pandas features and saves. Output: features/out/ (csv.gz + npz).
"""
from __future__ import annotations

import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "out"

for line in (ROOT / ".env").read_text().splitlines() if (ROOT / ".env").exists() else []:
    key, _, value = line.partition("=")
    if key.strip() and not key.strip().startswith("#"):
        os.environ.setdefault(key.strip(), value.strip())

API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
http = requests.Session()
http.headers["Authorization"] = f"Bearer {os.environ['RAWTREE_API_KEY']}"


def q(sql: str) -> pd.DataFrame:
    r = http.post(f"{API}/v1/query", json={"sql": sql}, timeout=600)
    r.raise_for_status()
    return pd.DataFrame(r.json()["data"])


def save(df: pd.DataFrame, name: str) -> None:
    with gzip.open(OUT / f"{name}.csv.gz", "wt", newline="") as f:
        df.to_csv(f, index=False)
    print(f"  saved {name}.csv.gz  ({len(df):,} rows, {df.shape[1]} cols)")


def catalog() -> pd.DataFrame:
    cat = q("""SELECT product_id::String AS product_id, category::String AS category,
                      subcategory::String AS subcategory, brand::String AS brand,
                      price_eur::Float64 AS price, is_active::Bool AS is_active,
                      created_at::String AS created_at, description AS description
               FROM shop.catalog""")
    return cat


def product_stats() -> None:
    print("  fetching catalogue + per-product event counts...")
    cat = catalog()
    ev = q("""SELECT product_id::String AS product_id,
                     countIf(event_type::String='view')        AS views,
                     countIf(event_type::String='add_to_cart') AS carts,
                     countIf(event_type::String='purchase')    AS purchases,
                     uniqExact(user_id::String)                AS buyers,
                     max(ts::DateTime64(3))                    AS last_event_ts
              FROM shop.history_events GROUP BY product_id""")
    cat_demand = q("""SELECT c.category::String AS category,
                             countIf(e.event_type::String='purchase') AS cat_purchases,
                             countIf(e.event_type::String='view')     AS cat_views
                      FROM shop.history_events e
                      INNER JOIN shop.catalog c ON c.product_id::String = e.product_id::String
                      GROUP BY category""")
    df = (cat.merge(ev, on="product_id", how="left")
             .merge(cat_demand, on="category", how="left"))
    for c in ["views", "carts", "purchases", "buyers", "cat_purchases", "cat_views"]:
        df[c] = df[c].fillna(0).astype(int)
    df["purchase_rate"] = df.purchases / (df.views + 1)
    df["cart_rate"] = df.carts / (df.views + 1)
    df["price_vs_subcat"] = df.groupby("subcategory").price.transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
    df["subcat_sales_rank"] = df.groupby("subcategory").purchases.rank(pct=True)
    df["cat_sales_share"] = df.purchases / (df.cat_purchases + 1)
    save(df, "product_stats")
    print(f"  sample features: {list(df.columns)}")


def session_context() -> None:
    print("  conversion by page × device × last-viewed category...")
    pd_dev = q("""SELECT context.page::String AS page, context.device::String AS device,
                         count() AS n, round(countIf(purchased != '')/count()*100, 3) AS conv_pct
                  FROM shop.history_sessions GROUP BY page, device""")
    last_view = q("""
        SELECT page, category AS last_view_category, count() AS n,
               round(countIf(purchased != '') / count() * 100, 3) AS conv_pct
        FROM (
            SELECT request_id, any(page) AS page, any(purchased) AS purchased,
                   argMax(pid, vts) AS last_view
            FROM (
                SELECT request_id, context.page::String AS page, purchased,
                       v.product_id::String AS pid, v.ts::DateTime64(3) AS vts
                FROM shop.history_sessions ARRAY JOIN recent_views AS v
            ) GROUP BY request_id
        ) AS s
        INNER JOIN shop.catalog c ON c.product_id::String = s.last_view
        GROUP BY page, last_view_category""")
    save(pd_dev, "ctx_page_device")
    save(last_view, "ctx_last_view_category")
    wide = last_view.pivot(index="last_view_category", columns="page", values="conv_pct")
    wide["overall"] = last_view.groupby("last_view_category").apply(
        lambda g: (g.conv_pct * g.n).sum() / g.n.sum(), include_groups=False)
    print("  conversion % by last-viewed category (top 5 by n):")
    top = last_view.groupby("last_view_category").n.idxmax()
    print(wide.loc[last_view.loc[top, "last_view_category"]].head(5).round(2).to_string())


def coviews() -> None:
    cut = q("SELECT toString(toDate(max(ts::DateTime64(3)) - INTERVAL 60 DAY)) AS cut FROM shop.history_events").cut[0]
    print(f"  session co-visitation since {cut} (60 days)...")
    cov = q(f"""
        SELECT p, q, count() AS n
        FROM (
            SELECT session_id::String AS sid, product_id::String AS p, ts::DateTime64(3) AS ts
            FROM shop.history_events WHERE event_type::String='view' AND ts::DateTime64(3) >= '{cut}'
        ) a
        INNER JOIN (
            SELECT session_id::String AS sid, product_id::String AS q, ts::DateTime64(3) AS ts
            FROM shop.history_events WHERE event_type::String='view' AND ts::DateTime64(3) >= '{cut}'
        ) b ON a.sid = b.sid AND a.p != b.q
        WHERE a.ts <= b.ts
        GROUP BY p, q
        ORDER BY n DESC
        LIMIT 50 BY p""")
    save(cov, "coviews")
    cov14 = q(f"""
        SELECT p, q, count() AS n
        FROM (
            SELECT session_id::String AS sid, product_id::String AS p, ts::DateTime64(3) AS ts
            FROM shop.history_events WHERE event_type::String='view' AND ts::DateTime64(3) >= '{cut}'
        ) a
        INNER JOIN (
            SELECT session_id::String AS sid, product_id::String AS q, ts::DateTime64(3) AS ts
            FROM shop.history_events WHERE event_type::String='view' AND ts::DateTime64(3) >= toDate('{cut}') + 46
        ) b ON a.sid = b.sid AND a.p != b.q
        WHERE a.ts <= b.ts
        GROUP BY p, q
        ORDER BY n DESC
        LIMIT 50 BY p""")
    save(cov14, "coviews_14d")


def user_stats() -> None:
    prof = q("""SELECT user_id::String AS user_id, count() AS buys,
                       uniqExact(product_id::String) AS distinct_products,
                       round(avg(price_eur::Float64), 2) AS avg_price_bought
                FROM shop.history_events WHERE event_type::String='purchase' GROUP BY user_id""")
    bycat = q("""SELECT e.user_id::String AS user_id, c.category::String AS category, count() AS n
                 FROM shop.history_events e
                 INNER JOIN shop.catalog c ON c.product_id::String = e.product_id::String
                 WHERE e.event_type::String='purchase' GROUP BY user_id, category""")
    top = bycat.sort_values(["user_id", "n"], ascending=[True, False]).groupby("user_id").head(1)
    totals = bycat.groupby("user_id").n.sum()
    top = top.set_index("user_id")
    top["top_cat_share"] = (top.n / totals).round(3)
    df = prof.merge(top[["category", "top_cat_share"]].rename(columns={"category": "top_category"}),
                    on="user_id", how="left")
    save(df, "user_stats")
    print(f"  {len(df):,} users with purchase history (features for returning users only)")


def trends() -> None:
    live = q("""SELECT product_id::String AS product_id,
                       countIf(event_type::String='view') AS views_24h,
                       countIf(event_type::String='purchase') AS purchases_24h
                FROM shop.events WHERE ts > now() - INTERVAL 24 HOUR GROUP BY product_id""")
    hist = q("""SELECT product_id::String AS product_id,
                       countIf(event_type::String='view') AS views_hist,
                       countIf(event_type::String='purchase') AS purchases_hist
                FROM shop.history_events GROUP BY product_id""")
    df = live.merge(hist, on="product_id", how="outer").fillna(0)
    days = 60
    df["hist_views_daily"] = df.views_hist / days
    df["hist_purchases_daily"] = df.purchases_hist / days
    df["view_lift"] = df.views_24h / df.hist_views_daily.replace(0, np.nan)
    df["purchase_lift"] = df.purchases_24h / df.hist_purchases_daily.replace(0, np.nan)
    df = df.sort_values("purchase_lift", ascending=False)
    save(df, "trends_product")
    sub = q("""SELECT c.subcategory::String AS subcategory,
                      countIf(e.event_type::String='view')     AS views_hist,
                      countIf(e.event_type::String='purchase') AS purchases_hist
               FROM shop.history_events e
               INNER JOIN shop.catalog c ON c.product_id::String = e.product_id::String
               GROUP BY subcategory""")
    live_sub = q("""SELECT c.subcategory::String AS subcategory,
                           countIf(e.event_type::String='view')     AS views_24h,
                           countIf(e.event_type::String='purchase') AS purchases_24h
                    FROM shop.events e
                    INNER JOIN shop.catalog c ON c.product_id::String = e.product_id::String
                    GROUP BY subcategory""")
    sub = sub.merge(live_sub, on="subcategory", how="outer").fillna(0)
    sub["hist_views_daily"] = sub.views_hist / 60
    sub["hist_purchases_daily"] = sub.purchases_hist / 60
    sub["view_lift"] = sub.views_24h / sub.hist_views_daily.replace(0, np.nan)
    sub["purchase_lift"] = sub.purchases_24h / sub.hist_purchases_daily.replace(0, np.nan)
    sub = sub.sort_values("purchase_lift", ascending=False)
    save(sub, "trends_subcategory")
    print("  top-5 subcategories by purchase lift right now:")
    print(sub.head(5).round(2).to_string(index=False))


def text_tfidf() -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    cat = catalog()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english", max_features=30000)
    x = vec.fit_transform(cat.description.fillna(""))
    x = normalize(x)
    import scipy.sparse as sp
    sp.save_npz(OUT / "text_tfidf.npz", x)
    cat[["product_id"]].to_csv(OUT / "text_tfidf_ids.csv", index=False)
    with open(OUT / "text_tfidf_vocab.json", "w") as f:
        json.dump(sorted(vec.vocabulary_, key=vec.vocabulary_.get), f)
    print(f"  saved text_tfidf.npz  ({x.shape[0]:,} products × {x.shape[1]:,} tf-idf features)")


SECTIONS = [product_stats, session_context, coviews, user_stats, trends, text_tfidf]


def main() -> None:
    args = sys.argv[1:]
    only = [a for a in args if not a.startswith("--")]
    OUT.mkdir(exist_ok=True)
    for fn in SECTIONS:
        if only and fn.__name__ not in only:
            continue
        print(f"[{fn.__name__}]")
        t = time.time()
        try:
            fn()
            print(f"  done in {time.time() - t:.1f}s")
        except Exception as e:
            print(f"  SKIPPED ({e})")


if __name__ == "__main__":
    main()
