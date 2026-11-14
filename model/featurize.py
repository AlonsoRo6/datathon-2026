"""Shared feature assembly: one code path used by both training and serving.

    from model.featurize import FeatureStore, session_base, attach_candidates, feature_matrix

FeatureStore loads the precomputed tables from features/out/ (built by
features/build_features.py). session_base() computes one row of context
features per request; attach_candidates() joins the per-(request, product)
features on top. The same functions build the multi-million-row training
table and the few-hundred-row serving batch, so train and live never drift.

Feature columns: NUMERIC + CATEGORICAL (categorical columns are pandas
`category` dtype; the allowed categories for each are stored in meta.json at
training time and reused at serving time).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

FEATURES_OUT = Path(__file__).resolve().parent.parent / "features" / "out"

NUMERIC = ["n_views", "hour", "has_user", "user_buys", "user_avg_price", "user_top_share",
           "ctx_page_device", "ctx_page_lastcat", "price", "purchase_rate", "cart_rate",
           "buyers", "subcat_sales_rank", "cat_sales_share", "price_vs_subcat",
           "cov_n", "cov_n14", "text_cos", "price_ratio_user",
           "same_cat_last", "same_sub_last", "seen_before", "user_top_match"]

CATEGORICAL = ["page", "device", "region", "last_view_cat", "cand_cat", "user_top_cat"]


@dataclass
class FeatureStore:
    products: pd.DataFrame
    ctx_pd: dict
    ctx_plc: dict
    cov: dict
    cov14: dict
    users: pd.DataFrame
    user_by_id: dict = field(default_factory=dict, init=False)
    tfidf: sp.csr_matrix | None = None
    tfidf_ids: np.ndarray | None = None
    pid_row: dict = field(default_factory=dict, init=False)
    best: list = field(default_factory=list, init=False)
    cat_stars: dict = field(default_factory=dict, init=False)
    trend: list = field(default_factory=list, init=False)
    pidx: pd.DataFrame = field(default_factory=pd.DataFrame, init=False)
    """products indexed by product_id: built once at load time and reused by
    every scoring component instead of rebuilding the index per request."""
    trends: pd.DataFrame = field(default_factory=pd.DataFrame, init=False)
    """Full trends_product table, kept around so the live trend list can be
    rebuilt periodically against the historical baseline.""" 

    @classmethod
    def load(cls) -> FeatureStore:
        rd = lambda n: pd.read_csv(FEATURES_OUT / f"{n}.csv.gz")
        ps = rd("product_stats")
        ctx_pd = {(r.page, r.device): r.conv_pct for r in rd("ctx_page_device").itertuples()}
        plc = rd("ctx_last_view_category")
        ctx_plc = {(r.page, r.last_view_category): r.conv_pct for r in plc.itertuples()}
        cov, cov14 = rd("coviews"), rd("coviews_14d")
        c1 = {p: dict(zip(g.q, g.n)) for p, g in cov.groupby("p")}
        c2 = {p: dict(zip(g.q, g.n)) for p, g in cov14.groupby("p")}
        fs = cls(products=ps, ctx_pd=ctx_pd, ctx_plc=ctx_plc, cov=c1, cov14=c2,
                 users=rd("user_stats"))
        fs.user_by_id = {r.user_id: r for r in fs.users.itertuples()}
        tf_path = FEATURES_OUT / "text_tfidf.npz"
        if tf_path.exists():
            fs.tfidf = sp.load_npz(tf_path)
            fs.tfidf_ids = pd.read_csv(FEATURES_OUT / "text_tfidf_ids.csv").product_id.values
            fs.pid_row = {p: i for i, p in enumerate(fs.tfidf_ids)}
        fs.best = ps.sort_values("purchases", ascending=False).product_id.tolist()
        fs.cat_stars = {c: g.sort_values("purchases", ascending=False).product_id.head(40).tolist()
                        for c, g in ps.groupby("category")}
        fs.trends = rd("trends_product")
        t = fs.trends.sort_values("purchase_lift", ascending=False)
        fs.trend = t[t.views_24h >= 20].product_id.head(120).tolist()
        fs.pidx = fs.products.set_index("product_id")
        return fs


def session_base(reqs: pd.DataFrame, fs: FeatureStore) -> pd.DataFrame:
    """One row of per-request context features (pages, hour, user stats, ...)."""
    s = pd.DataFrame(index=reqs.index)
    s["page"] = reqs["page"].astype(str)
    s["device"] = reqs["device"].astype(str)
    s["region"] = reqs["region"].astype(str)
    s["n_views"] = reqs["n_views"].astype("float32")
    s["hour"] = pd.to_datetime(reqs["ts"]).dt.hour.astype("float32")
    last = reqs["last_view"].fillna("").astype(str).mask(reqs["n_views"].fillna(0) == 0, "")
    s["last_view"] = last
    ps = fs.products.set_index("product_id")
    cat_names = sorted(ps["category"].dropna().unique())
    s["last_view_cat"] = pd.Categorical(last.map(ps["category"]), categories=cat_names)
    u = reqs["user_id"].map(fs.user_by_id)
    s["has_user"] = u.notna().astype("int8")

    def uf(attr):
        def get(r):
            try:
                return getattr(r, attr)
            except AttributeError:
                return np.nan
        return pd.Series(u.map(get), index=s.index, dtype="float32")

    s["user_buys"] = uf("buys")
    s["user_avg_price"] = uf("avg_price_bought")
    s["user_top_cat"] = pd.Categorical(u.map(lambda r: getattr(r, "top_category", None)),
                                       categories=cat_names)
    s["user_top_share"] = uf("top_cat_share")
    s["ctx_page_device"] = pd.Series([fs.ctx_pd.get((p, d), np.nan) for p, d in zip(s.page, s.device)],
                                     index=s.index, dtype="float32")
    s["ctx_page_lastcat"] = pd.Series([fs.ctx_plc.get((p, str(c))) for p, c in zip(s.page, s.last_view_cat)],
                                     index=s.index, dtype="float32")
    s["recent_ids"] = reqs["recent_ids"]
    return s


def attach_candidates(base: pd.DataFrame, slots: pd.DataFrame, fs: FeatureStore) -> pd.DataFrame:
    """Join candidate-level features (product stats, co-views, text cosine)
    onto every (request, candidate) slot."""
    ps = fs.pidx
    df = slots.reset_index(drop=True).merge(base.reset_index(names="rid"),
                                            left_on="request_id", right_on="rid", how="left")
    j = df.merge(ps[["category", "subcategory", "brand", "price", "purchase_rate", "cart_rate",
                     "buyers", "subcat_sales_rank", "cat_sales_share", "price_vs_subcat"]],
                 left_on="pid", right_index=True, how="left")
    j["cand_cat"] = pd.Categorical(j["category"], categories=sorted(ps["category"].dropna().unique()))
    j.drop(columns=["category"], inplace=True)

    cov_n, cov_n14 = np.zeros(len(j), "float32"), np.zeros(len(j), "float32")
    lv = j.last_view.to_numpy()
    pid = j.pid.to_numpy()
    for i in range(len(j)):
        d1, d2 = fs.cov.get(lv[i]), fs.cov14.get(lv[i])
        if d1 is not None:
            cov_n[i] = d1.get(pid[i], 0)
        if d2 is not None:
            cov_n14[i] = d2.get(pid[i], 0)
    j["cov_n"] = np.log1p(cov_n)
    j["cov_n14"] = np.log1p(cov_n14)

    lv_cat = pd.Series(lv).map(ps["category"])
    lv_sub = pd.Series(lv).map(ps["subcategory"])
    j["same_cat_last"] = np.where(lv == "", np.nan, (np.asarray(lv_cat) == np.asarray(j.cand_cat)).astype("float32"))
    j["same_sub_last"] = np.where(lv == "", np.nan, (np.asarray(lv_sub) == np.asarray(j.subcategory)).astype("float32"))
    if "recent_ids" in j:
        rsets = j.drop_duplicates("request_id").set_index("request_id").recent_ids.map(set)
        j["seen_before"] = np.array([pid[i] in rsets[j.request_id.iloc[i]] for i in range(len(j))],
                                    dtype="float32")
    else:
        j["seen_before"] = 0.0
    j["user_top_match"] = np.where(j.user_top_cat.isna() | j.cand_cat.isna(), np.nan,
                                   (j.user_top_cat.astype(str) == j.cand_cat.astype(str)).astype("float32"))
    j["price_ratio_user"] = j.price / j.user_avg_price
    j["text_cos"] = _text_cos(j, fs) if fs.tfidf is not None else np.nan
    if "slot" not in j:
        j["slot"] = 1
    return j


def _text_cos(j: pd.DataFrame, fs: FeatureStore) -> np.ndarray:
    """TF-IDF cosine similarity between each candidate and the last viewed
    product, computed per unique last-view to reuse sparse products."""
    out = np.full(len(j), np.nan, dtype="float32")
    cand_rows = j.pid.map(fs.pid_row).to_numpy(dtype="float64")
    lv_rows = pd.Series(j.last_view.to_numpy()).map(fs.pid_row).to_numpy(dtype="float64")
    has_lv = ~np.isnan(lv_rows)
    if not has_lv.any():
        return out
    order = np.argsort(np.where(has_lv, lv_rows, np.inf), kind="stable")
    lv_s = lv_rows[order]
    idx_s = order[has_lv[order]]
    lv_valid = lv_s[has_lv[order]]
    uniq = np.unique(lv_valid)
    lo = np.searchsorted(lv_valid, uniq, side="left")
    hi = np.searchsorted(lv_valid, uniq, side="right")
    for l, h in zip(lo, hi):
        idx = idx_s[l:h]
        cand_idx = idx[~np.isnan(cand_rows[idx])]
        if len(cand_idx) == 0:
            continue
        sims = (fs.tfidf[int(lv_valid[l])] @ fs.tfidf[cand_rows[cand_idx].astype(int)].T).toarray().ravel()
        out[cand_idx] = sims
    return out


def feature_matrix(j: pd.DataFrame, cats: dict[str, list[str]]) -> pd.DataFrame:
    """Model input: numeric columns as float32, categoricals restricted to the
    training-time category lists from meta.json."""
    out = j[NUMERIC].astype("float32").copy()
    for c in CATEGORICAL:
        v = pd.Series(j[c].to_numpy(), dtype=object)
        v = v.where(v.notna() & ~v.astype(str).str.lower().isin(["nan", "none", ""]), None)
        out[c] = pd.Categorical(v, categories=cats[c])
    return out[NUMERIC + CATEGORICAL]
