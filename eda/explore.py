"""Datathon FME 2026 · EDA: generates CSVs and charts from RawTree.

    python eda/explore.py            # all sections
    python eda/explore.py --no-live  # skip live tables
    python eda/explore.py features   # visualize the engineered feature tables

Read-only: every aggregation runs as SQL in RawTree, Python only plots.
Outputs: CSVs in eda/out/, one interactive figure per section (nothing stored).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
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
TEAM = os.environ["TEAM_ID"]
http = requests.Session()
http.headers["Authorization"] = f"Bearer {os.environ['RAWTREE_API_KEY']}"

plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "figure.autolayout": True})


def q(sql: str) -> pd.DataFrame:
    r = http.post(f"{API}/v1/query", json={"sql": sql}, timeout=180)
    r.raise_for_status()
    return pd.DataFrame(r.json()["data"])


def save(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT / f"{name}.csv", index=False)
    print(f"  csv: out/{name}.csv  ({len(df)} rows)")


def chart(title: str, panels: int = 4):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    return fig, [axes[0][0], axes[0][1], axes[1][0], axes[1][1]][:panels]


def annotate_bars(ax, fmt="{:,.0f}"):
    for b in ax.patches:
        if b.get_width() > 0:
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2, fmt.format(b.get_width()), va="center", fontsize=8)


def catalogue() -> None:
    fig, ax = chart("1 · Catalogue: 5,000 products in 10 categories")

    cats = q("""SELECT category::String AS category, count() AS products,
                       countIf(is_active::Bool = true) AS active, round(avg(price_eur::Float64), 2) AS avg_price
                FROM shop.catalog GROUP BY category ORDER BY products DESC""")
    save(cats, "catalog_categories")
    a = ax[0]
    cats[::-1].plot.barh(x="category", y="products", legend=False, ax=a, color="#4a72b0")
    a.set_title("A · Products per category")
    annotate_bars(a)

    a = ax[1]
    cats[::-1].plot.barh(x="category", y="avg_price", legend=False, ax=a, color="#c44e52")
    a.set_title("B · Average price per category (EUR)")
    annotate_bars(a, fmt="{:.0f}€")

    a = ax[2]
    prices = q("""SELECT floor(price_eur::Float64 / 5) * 5 AS bucket, count() AS n
                  FROM shop.catalog GROUP BY bucket ORDER BY bucket""")
    save(prices, "catalog_prices")
    a.bar(prices.bucket, prices.n, width=4)
    a.axvline(prices.bucket.iloc[(prices.n * prices.bucket).cumsum().searchsorted(prices.n.sum() / 2)],
              ls="--", c="k", lw=1)
    a.set_title("C · Price distribution (median = dashed line)")
    a.set_xlabel("price (EUR, 5€ buckets)")
    a.set_xlim(-5, min(300, prices.bucket.max()))

    a = ax[3]
    brands = q("""SELECT brand::String AS brand, count() AS n FROM shop.catalog
                  GROUP BY brand ORDER BY n DESC LIMIT 15""")
    save(brands, "catalog_brands")
    a.text(0.5, 0.5, f"{brands.n.iloc[0]} products / top brand\n"
                     f"{brands.brand.nunique()} brands in top 15 of {q('SELECT uniqExact(brand::String) AS n FROM shop.catalog').n[0]} total\n\n"
                     f"descriptions: one or two sentences (raw text for embeddings)\ncreated_at spans 2025-01 .. 2026-11",
           ha="center", va="center", fontsize=12)
    a.set_title("D · top-15 brands by catalogue size")
    b = brands[::-1].plot.barh(x="brand", y="n", legend=False, ax=a.twinx(), color="#55a868")
    b.get_legend().remove() if b.get_legend() else None
    a.axis("off")

# interactive figure: shown by plt.show() at the end of the run


def sessions() -> None:
    t0 = q("""SELECT count() AS n, countIf(clicked != '') AS clicks, countIf(purchased != '') AS purchases,
                     round(countIf(clicked != '')/count()*100, 2) AS ctr,
                     round(countIf(purchased != '')/count()*100, 2) AS conv
              FROM shop.history_sessions""")
    fig, ax = chart("2 · history_sessions: 2M impressions from the OLD recommender "
                    f"(CTR {t0.ctr[0]}% · conv {t0.conv[0]}%)")

    a = ax[0]
    funnel = pd.DataFrame({"stage": ["impressions", "clicks", "purchases"],
                           "n": [t0.n[0], t0.clicks[0], t0.purchases[0]]})
    save(funnel, "sess_funnel")
    a.bar(funnel.stage, funnel.n, color=["#4a72b0", "#55a868", "#c44e52"])
    for i, (s, n) in enumerate(zip(funnel.stage, funnel.n)):
        a.text(i, n, f"{n:,}", ha="center", va="bottom")
    a.set_yscale("log")
    a.set_title(f"A · Funnel: CTR {t0.ctr[0]}% · conversion {t0.conv[0]}%")

    a = ax[1]
    slots = q("""SELECT indexOf(shown, clicked) AS slot, count() AS clicks
                 FROM shop.history_sessions WHERE clicked != '' GROUP BY slot ORDER BY slot""")
    sample = q("SELECT shown, propensity FROM shop.history_sessions ORDER BY rand() LIMIT 20000")
    prop = pd.DataFrame([{"slot": i + 1, "avg_propensity": float(p)}
                         for _, r in sample.iterrows() if isinstance(r["propensity"], list)
                         for i, p in enumerate(r["propensity"])])
    slots = slots.merge(prop.groupby("slot").avg_propensity.mean().reset_index(), on="slot", how="left")
    save(slots, "sess_slots")
    a.bar(slots.slot, slots.clicks, color="#4a72b0", label="clicks in slot")
    a2 = a.twinx()
    a2.plot(slots.slot, slots.avg_propensity, "o-", c="#c44e52", label="avg propensity of old policy")
    a2.set_ylim(0, 0.25)
    a2.grid(False)
    a.set_title("B · Position bias: clicks by slot vs old policy propensity")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=8)

    a = ax[2]
    pd_ = q("""SELECT context.page::String AS page, context.device::String AS device, count() AS n,
                       round(countIf(purchased != '')/count()*100, 3) AS conv_pct
                FROM shop.history_sessions GROUP BY page, device""")
    save(pd_, "sess_page_device")
    wide = pd_.pivot(index="page", columns="device", values="conv_pct").reindex(["home", "category", "product"])
    wide.plot.bar(ax=a, color={"web": "#4a72b0", "ios": "#55a868", "android": "#c44e52"})
    a.set_title(f"C · Conversion % by page × device — home page is dead ({pd_[pd_.page == 'home'].n.sum():,} impressions)")
    a.set_ylabel("conversion %")

    a = ax[3]
    rv = q("""SELECT length(recent_views) AS n_views, count() AS n,
                     round(countIf(purchased != '')/count()*100, 2) AS conv_pct
              FROM shop.history_sessions GROUP BY n_views ORDER BY n_views""")
    save(rv, "sess_recent_views")
    a.bar(rv.n_views.astype(int), rv.conv_pct, color="#55a868")
    for x, (n, c) in enumerate(zip(rv.n, rv.conv_pct)):
        a.text(x, c + 0.03, f"{n / 1e3:.0f}k", ha="center", fontsize=8)
    a.set_xlabel("products already viewed this session")
    a.set_title("D · Conversion % by session depth (labels: #requests)")

# interactive figure: shown by plt.show() at the end of the run


def events() -> None:
    t0 = q("""SELECT count() AS n, countIf(event_type::String='purchase') AS purchases FROM shop.history_events""")
    fig, ax = chart(f"3 · history_events: {t0.n[0]/1e6:.1f}M organic events "
                    f"({t0.purchases[0]/1e3:.0f}K purchases)")

    a = ax[0]
    hour = q("""SELECT toHour(ts::DateTime64(3)) AS h, event_type::String AS t, count() AS n
                FROM shop.history_events GROUP BY h, t ORDER BY h""")
    save(hour, "ev_hourly")
    tot = hour.groupby("t").n.sum()
    for t, c in zip(["view", "add_to_cart", "purchase"], ["#4a72b0", "#55a868", "#c44e52"]):
        sub = hour[hour.t == t].set_index("h").reindex(range(24), fill_value=0)
        a.plot(sub.index, sub.n / tot[t] * 100, label=f"{t}", c=c)
    a.set_xticks(range(0, 24, 2))
    a.legend()
    a.set_title("A · Diurnal pattern: % of each event type per hour (UTC)\npeak ~02-04h, trough ~08-12h — ~10x swing")

    a = ax[1]
    cat = q("""SELECT c.category::String AS category, count() AS purchases
               FROM shop.history_events e INNER JOIN shop.catalog c ON c.product_id::String = e.product_id::String
               WHERE e.event_type::String='purchase' GROUP BY category ORDER BY purchases DESC""")
    save(cat, "ev_category")
    cat[::-1].plot.barh(x="category", y="purchases", legend=False, ax=a, color="#4a72b0")
    a.set_title(f"B · Purchases by category: {cat.category[0]} has "
                f"{cat.purchases[0] / cat.purchases.iloc[-1]:.1f}x the demand of {cat.category.iloc[-1]}")
    annotate_bars(a)

    a = ax[2]
    prod = q("""SELECT product_id::String AS product_id, count() AS purchases
                FROM shop.history_events WHERE event_type::String='purchase' GROUP BY product_id""")
    save(prod, "ev_product_sales")
    p = prod.sort_values("purchases", ascending=False)
    cum = p.purchases.cumsum() / p.purchases.sum()
    a.plot(np.linspace(0, 100, len(cum)), cum * 100)
    a.plot([0, 100], [0, 100], "k--", lw=1)
    top100 = float(cum.iloc[99] * 100)
    a.axvline(2, ls=":", c="r")
    a.text(2.5, 5, f"top 2% of products ({int(len(p)*0.02)}) → {top100:.0f}% of purchases", fontsize=9, color="r")
    a.set_xlabel("products, ranked by sales (most popular first), %")
    a.set_ylabel("cumulative % of purchases")
    a.set_title("C · Lorenz curve: popularity is diffuse → personalization can win")

    a = ax[3]
    h = q("""SELECT event_type::String AS t, floor(price_eur::Float64 / 10) * 10 AS bucket, count() AS n
             FROM shop.history_events GROUP BY t, bucket ORDER BY bucket""")
    save(h.rename(columns={"bucket": "price_bucket"}), "ev_price_hist")
    for t, c in zip(["view", "add_to_cart", "purchase"], ["#4a72b0", "#55a868", "#c44e52"]):
        sub = h[(h.t == t) & (h.bucket <= 200)]
        a.plot(sub.bucket, sub.n / sub.n.sum(), label=t, c=c, drawstyle="steps-mid")
    a.legend(title="event type")
    a.set_xlabel("price (EUR, 10€ buckets)")
    a.set_title("D · Price distribution per event: buyers skew chea­per\n(p50 view €28.99 → purchase €23.99)")

# interactive figure: shown by plt.show() at the end of the run


def live() -> None:
    stream = q("SELECT count() AS n, min(ts::DateTime64(3)) AS first_ts, max(ts::DateTime64(3)) AS last_ts FROM shop.requests")
    fig, ax = chart(f"4 · LIVE stream (since {stream.first_ts[0]},{stream.n[0]:,} requests so far)")

    a = ax[0]
    rpm = q("""SELECT toStartOfMinute(ts::DateTime64(3)) AS minute, count() AS n
               FROM shop.requests GROUP BY minute ORDER BY minute""")
    save(rpm, "live_requests")
    a.plot(pd.to_datetime(rpm.minute), rpm.n)
    a.set_title("A · Recommendation requests per minute")

    a = ax[1]
    fb = q(f"""SELECT served_by::String AS served_by, fallback_reason::String AS reason, count() AS n,
                      round(countIf(purchased != '')/count()*100, 2) AS conv_pct,
                      round(avg(latency_ms::Float64)) AS avg_ms
               FROM {TEAM}.feedback WHERE request_id != '__init__' GROUP BY served_by, reason""")
    save(fb, "live_feedback")
    b = fb[fb.served_by == "fallback"]
    a.bar(b.reason + " (fallback)", b.n, color="#c44e52", label="fallback")
    a.bar(["team"], [fb[fb.served_by == "team"].n.sum()], color="#55a868", label="team")
    a.set_yscale("log")
    a.legend()
    a.set_title("B · Who served each shopper (log scale) — 'missing' = we are not running!")

    a = ax[2]
    fb2 = q(f"""SELECT served_by::String AS served_by, count() AS n,
                       round(countIf(clicked != '')/count()*100, 2) AS ctr_pct,
                       round(countIf(purchased != '')/count()*100, 2) AS conv_pct
                FROM {TEAM}.feedback WHERE request_id != '__init__' AND served_by::String = 'team'
                GROUP BY served_by""")
    save(fb2, "live_served")
    ww = fb2.melt(id_vars="served_by", value_vars=["ctr_pct", "conv_pct"], var_name="metric")
    a.bar(ww.metric, ww.value, color=["#4a72b0", "#c44e52"])
    for i, v in enumerate(ww.value):
        a.text(i, v + 0.05, f"{v}%", ha="center")
    a.set_title(f"C · OUR baseline so far: CTR & conversion (avg latency {fb[fb.served_by=='team'].avg_ms.iloc[0]} ms)")

    a = ax[3]
    trend = q("""SELECT c.subcategory::String AS subcategory,
                        countIf(e.ts > now() - INTERVAL 30 MINUTE)                              AS v30,
                        countIf(e.ts <= now() - INTERVAL 30 MINUTE) / 6                         AS v_before,
                        v30 / greatest(v_before, 1)                                             AS lift
                 FROM shop.events e INNER JOIN shop.catalog c ON c.product_id::String = e.product_id::String
                 WHERE e.event_type::String = 'view' AND e.ts > now() - INTERVAL 3 HOUR
                 GROUP BY subcategory ORDER BY lift DESC LIMIT 10""")
    save(trend, "live_trends")
    trend[::-1].plot.barh(x="subcategory", y="lift", legend=False, ax=a, color="#dd8452")
    a.set_xlabel("views last 30 min ÷ avg views per 30 min (previous 3 h)")
    a.set_title("D · Trending subcategories RIGHT NOW — chase these in the final round")

# interactive figure: shown by plt.show() at the end of the run


def features() -> None:
    import scipy.sparse as sp

    FEAT = ROOT / "features" / "out"
    needed = ["product_stats", "ctx_last_view_category", "coviews", "coviews_14d",
              "user_stats", "trends_subcategory"]
    missing = [n for n in needed if not (FEAT / f"{n}.csv.gz").exists()]
    if missing:
        print(f"  run features/build_features.py first — missing {missing}")
        return
    ps = pd.read_csv(FEAT / "product_stats.csv.gz")
    lastcat = pd.read_csv(FEAT / "ctx_last_view_category.csv.gz")
    cov = pd.read_csv(FEAT / "coviews.csv.gz")
    cov14 = pd.read_csv(FEAT / "coviews_14d.csv.gz")
    users = pd.read_csv(FEAT / "user_stats.csv.gz")
    trend = pd.read_csv(FEAT / "trends_subcategory.csv.gz")

    fig, A = plt.subplots(2, 3, figsize=(18, 9))
    (a1, a2, a3), (a4, a5, a6) = A
    fig.suptitle("5 · Engineered features (features/out) — what the model will see",
                 fontsize=14, fontweight="bold")

    wide = lastcat.pivot(index="last_view_category", columns="page", values="conv_pct")
    wide = wide.reindex(wide["product"].sort_values().index)
    wide.plot.bar(ax=a1, width=0.75)
    a1.axhline(1.4, ls="--", c="k", lw=1)
    a1.text(8.3, 1.5, "store average 1.4%", fontsize=8)
    a1.set_ylabel("conversion %")
    a1.set_title("A · Intent feature: conversion by LAST-VIEWED product's category\n(from 0.15% 'browsing' to 4.2% 'Buying mood')")
    a1.tick_params(axis="x", labelsize=8, rotation=45)

    best = ps.loc[ps.purchases.idxmax()]
    c60 = cov[cov.p == best.product_id].nlargest(10, "n") \
        .merge(ps[["product_id", "subcategory"]], left_on="q", right_on="product_id", how="left")
    c14 = cov14[cov14.p == best.product_id].set_index("q")
    pair = pd.DataFrame({"60 days": c60.n.values,
                         "last 14 days": c60.q.map(c14.n).fillna(0).values},
                        index=[f"{s} · {q}" for s, q in zip(c60.subcategory.fillna('?'), c60.q)])
    pair[::-1].plot.barh(ax=a2, color={"60 days": "#4a72b0", "last 14 days": "#dd8452"})
    a2.legend(fontsize=8)
    a2.set_title(f"B · Retrieval feature: 'also viewed' for the top seller\n{best.product_id} ({best.subcategory}, {best.purchases:,} purchases)")

    t10 = trend.nlargest(10, "purchase_lift")[::-1]
    a3.barh(t10.subcategory, t10.purchase_lift, color="#c44e52")
    a3.set_xlabel("lift = live 24h rate ÷ 60-day daily average")
    a3.set_title("C · Dynamics feature: subcategories trending RIGHT NOW")
    for y, v in enumerate(t10.purchase_lift):
        a3.text(v + 0.1, y, f"{v:.1f}x", va="center", fontsize=8)

    a4.scatter(ps.views, ps.purchases, s=5, alpha=0.35, color="#4a72b0")
    hot = ps.nlargest(20, "purchases")
    a4.scatter(hot.views, hot.purchases, s=12, color="#c44e52", label="top-20 sellers")
    a4.set_xscale("log")
    a4.set_yscale("log")
    a4.set_xlabel("views (60d)")
    a4.set_ylabel("purchases (60d)")
    a4.legend(fontsize=8)
    a4.set_title("D · Attractiveness feature: product funnel spread\n(same views can convert very differently — that's the model's room)")

    a5.hist(users.buys, bins=60, color="#55a868")
    a5.set_yscale("log")
    a5.set_xlabel("purchases per user (60 days)")
    a5.set_ylabel("users (log)")
    a5.set_title(f"E · User feature: {len(users):,} returning shoppers\nmedian {users.buys.median():.0f} buys · blank for the 30% new users → NaN-tolerant")

    x = sp.load_npz(FEAT / "text_tfidf.npz")
    ids = pd.read_csv(FEAT / "text_tfidf_ids.csv").product_id.values
    rng = np.random.default_rng(0)
    lines = []
    for i in sorted(rng.choice(len(ids), 3, replace=False)):
        sims = (x[i] @ x.T).toarray().ravel()
        sims[i] = -1
        top = sims.argsort()[::-1][:3]
        lines.append(f"{ids[i]} [{ps.subcategory.iloc[i]}]")
        lines += [f"   → {ids[j]} [{ps.subcategory.iloc[j]}]  cos={sims[j]:.2f}" for j in top]
    a6.axis("off")
    a6.text(0.02, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=9)
    a6.set_title("F · Cold-start feature: description tf-idf nearest neighbours\n(new products rank by text similarity — sanity check above)")

    fig.tight_layout()


SECTIONS = [catalogue, sessions, events, live, features]


def main() -> None:
    args = sys.argv[1:]
    only = [a for a in args if not a.startswith("--")]
    OUT.mkdir(exist_ok=True)
    for fn in SECTIONS:
        if fn is live and "--no-live" in args:
            continue
        if only and fn.__name__ not in only:
            continue
        t = time.time()
        print(f"[{fn.__name__}]")
        try:
            fn()
            print(f"  done in {time.time() - t:.1f}s")
        except Exception as e:
            print(f"  SKIPPED ({e})")
    plt.show()


if __name__ == "__main__":
    main()
