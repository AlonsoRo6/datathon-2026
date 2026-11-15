"""Build the training table from shop.history_sessions and train the v1 model.

    python model/build_training.py

Training rows: one per (session, shown product) pair of a sampled subset of
sessions — every session with a click or purchase, plus 7% of pure negatives.
Label = the product that was purchased; weight = 1/propensity (IPS, to cancel
the old policy's biased exposure). Time-ordered holdout: last 10 days.

Outputs (model/out/): model.joblib, meta.json and a printed report, including
a full-catalogue hit@10 head-to-head vs the best-sellers baseline on the
holdout.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.featurize import CATEGORICAL, NUMERIC, FeatureStore, attach_candidates, feature_matrix, session_base

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
NEG_RATE_TAG = "neg7pct"
HOLDOUT_DAYS = 10
N_EVAL = 800
CHUNK = 100

for line in (HERE.parent / ".env").read_text().splitlines() if (HERE.parent / ".env").exists() else []:
    key, _, value = line.partition("=")
    if key.strip() and not key.strip().startswith("#"):
        os.environ.setdefault(key.strip(), value.strip())
API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
http = requests.Session()
http.headers["Authorization"] = f"Bearer {os.environ['RAWTREE_API_KEY']}"


def q(sql: str) -> pd.DataFrame:
    r = http.post(f"{API}/v1/query", json={"sql": sql}, timeout=600)
    if not r.ok:
        raise RuntimeError(f"query failed [{r.status_code}]: {r.text[:300]}\nSQL: {sql[:200]}")
    return pd.DataFrame(r.json()["data"])


def fetch_sessions() -> pd.DataFrame:
    cols = """request_id, ts::String AS ts, user_id::String AS user_id,
              context.page::String AS page, context.device::String AS device,
              context.region::String AS region, recent_views, shown, propensity,
              coalesce(clicked::String, '') AS clicked, coalesce(purchased::String, '') AS purchased"""
    sel = "(clicked != '' OR purchased != '' OR cityHash64(request_id::String) % 100 < 7)"
    parts, last = [], "2000-01-01 00:00:00.000"
    while True:
        chunk = q(f"""SELECT {cols} FROM shop.history_sessions
                      WHERE {sel} AND ts::DateTime64(3) > toDateTime64('{last}', 3)
                      ORDER BY ts LIMIT 40000""")
        parts.append(chunk)
        if len(chunk) < 40000:
            break
        last = chunk.ts.iloc[-1]
        print(f"  ... {sum(len(p) for p in parts):,} sessions")
    df = pd.concat(parts, ignore_index=True)
    df["recent_ids"] = df.recent_views.map(lambda v: [x["product_id"] for x in v] if isinstance(v, list) else [])
    df["n_views"] = df.recent_ids.map(len)
    last_view = []
    for v in df.recent_views:
        last_view.append(max(v, key=lambda x: x.get("ts", ""))["product_id"] if isinstance(v, list) and v else "")
    df["last_view"] = last_view
    return df


def fetch_feedback_contexts(rids: list[str]) -> pd.DataFrame:
    """Context + recent views for our served requests (IN-batches by rid)."""
    parts = []
    for i in range(0, len(rids), 2000):
        ids = ",".join(f"'{r}'" for r in rids[i:i + 2000])
        parts.append(q(f"""SELECT request_id::String AS request_id, ts::String AS ts,
                                  user_id::String AS user_id,
                                  context.page::String AS page, context.device::String AS device,
                                  context.region::String AS region, recent_views
                           FROM shop.requests WHERE request_id::String IN ({ids})"""))
        if i % 40000 == 0 and i:
            print(f"  ... {i:,}/{len(rids):,} contexts")
    return pd.concat(parts, ignore_index=True)


def fetch_feedback() -> pd.DataFrame:
    """Our own served impressions (on-policy data, no propensity correction needed)."""
    cols = "request_id::String AS request_id, shown, coalesce(clicked::String,'') AS clicked, coalesce(purchased::String,'') AS purchased"
    parts, last = [], "2000-01-01 00:00:00.000"
    while True:
        chunk = q(f"""SELECT ts::String AS ts, {cols} FROM {os.environ['TEAM_ID']}.feedback
                      WHERE served_by::String = 'team' AND request_id != '__init__'
                        AND ts::DateTime64(3) > toDateTime64('{last}', 3)
                      ORDER BY ts LIMIT 40000""")
        parts.append(chunk)
        if len(chunk) < 40000:
            break
        last = chunk.ts.iloc[-1]
        print(f"  ... {sum(len(p) for p in parts):,} feedback rows")
    fb = pd.concat(parts, ignore_index=True).drop_duplicates("request_id")
    print(f"  {len(fb):,} served feedback impressions")
    ctx = fetch_feedback_contexts(fb.request_id.tolist())
    fb = fb.merge(ctx, on="request_id", how="inner", suffixes=("", "_req"))
    fb["ts"] = fb["ts_req"]
    fb.drop(columns=["ts_req"], inplace=True)
    fb["propensity"] = fb.shown.map(lambda s: [1.0] * len(s) if isinstance(s, list) else [])
    fb["recent_ids"] = fb.recent_views.map(lambda v: [x["product_id"] for x in v] if isinstance(v, list) else [])
    fb["n_views"] = fb.recent_ids.map(len)
    fb["last_view"] = [max(v, key=lambda x: x.get("ts", ""))["product_id"] if isinstance(v, list) and v else ""
                       for v in fb.recent_views]
    return fb


def to_slots(df: pd.DataFrame) -> pd.DataFrame:
    def pad(p, n):
        p = list(p) if isinstance(p, list) else []
        return (p + [1.0] * n)[:n]

    props = [pad(p, len(s) if isinstance(s, list) else 0) for s, p in zip(df.shown, df.propensity)]
    lens = np.array([len(p) for p in props])
    rep = df.loc[df.index.repeat(lens)].copy()
    rep["pid"] = np.concatenate([list(s) if isinstance(s, list) else [] for s in df.shown])
    rep["prop"] = np.concatenate(props) if lens.sum() else np.array([])
    rep["slot"] = rep.groupby("request_id", sort=False).cumcount() + 1
    return rep[["request_id", "pid", "prop", "slot"]], lens


def main() -> None:
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    fs = FeatureStore.load()
    print(f"feature store loaded in {time.time() - t0:.1f}s")

    print("fetching sampled sessions...")
    sess = fetch_sessions()
    pos = (sess.clicked != "") | (sess.purchased != "")
    print(f"  {len(sess):,} sessions ({pos.sum():,} with click/purchase)")

    print("fetching our own feedback (on-policy data)...")
    fb = fetch_feedback()
    sess_all = pd.concat([sess, fb[sess.columns.tolist()]], ignore_index=True)
    base = session_base(sess_all, fs).set_axis(pd.Index(sess_all.request_id, name="request_id"))
    slots, lens = to_slots(sess_all)
    purch = sess_all.set_index("request_id").purchased.reindex(slots.request_id).fillna("").to_numpy()
    y = (slots.pid.to_numpy() == purch).astype("int8")
    w = 1.0 / np.clip(slots.prop.to_numpy(dtype="float64"), 0.01, 1.0)
    is_fb = slots.request_id.isin(set(fb.request_id)).to_numpy()
    print(f"  on-policy rows: {is_fb.sum():,}")
    feats = attach_candidates(base, slots, fs)
    print(f"  {len(feats):,} training rows, positive rate {y.mean() * 100:.2f}%")

    ts_s = pd.to_datetime(sess.ts, format="mixed")
    cut = ts_s.max() - pd.Timedelta(days=HOLDOUT_DAYS)
    va_s = (ts_s >= cut).to_numpy()
    va = np.concatenate([np.repeat(va_s, lens[:len(sess)]),
                         np.zeros(int(lens[len(sess):].sum()), dtype=bool)])
    cats = {
        "page": ["home", "category", "product"],
        "device": ["web", "ios", "android"],
        "region": sorted(feats.region.dropna().astype(str).unique()),
        "last_view_cat": sorted(fs.products.category.dropna().astype(str).unique()),
        "cand_cat": sorted(fs.products.category.dropna().astype(str).unique()),
        "user_top_cat": sorted(fs.products.category.dropna().astype(str).unique()),
    }
    X = feature_matrix(feats, cats)
    X_tr, y_tr, w_tr = X[~va], y[~va], w[~va]
    X_va, y_va, w_va = X[va], y[va], w[va]
    print(f"  train {len(X_tr):,} rows / holdout {len(X_va):,} rows (cut {cut.date()})")

    import lightgbm as lgb
    from sklearn.metrics import log_loss, roc_auc_score
    model = lgb.LGBMClassifier(
        objective="binary", learning_rate=0.05, n_estimators=600, num_leaves=63,
        min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=5.0, random_state=0, verbosity=-1)
    model.fit(X_tr, y_tr, sample_weight=w_tr,
              eval_set=[(X_va, y_va)], eval_sample_weight=[w_va],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    print(f"  lgbm best iteration: {model.best_iteration_}")
    p_va = model.predict_proba(X_va)[:, 1]

    import xgboost as xgb
    xmodel = xgb.XGBClassifier(
        n_estimators=600, learning_rate=0.05, max_depth=8, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=5.0, tree_method="hist",
        enable_categorical=True, eval_metric="logloss",
        early_stopping_rounds=50, random_state=0, verbosity=0)
    xmodel.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)], verbose=False)
    p_xva = xmodel.predict_proba(X_va)[:, 1]
    p_ens = (p_va + p_xva) / 2
    metrics = {
        "auc_valid_lgbm": round(float(roc_auc_score(y_va, p_va, sample_weight=w_va)), 4),
        "auc_valid_xgb": round(float(roc_auc_score(y_va, p_xva, sample_weight=w_va)), 4),
        "auc_valid_ensemble": round(float(roc_auc_score(y_va, p_ens, sample_weight=w_va)), 4),
        "logloss_valid": round(float(log_loss(y_va, p_ens, sample_weight=w_va)), 5),
        "positive_rate": round(float(y.mean()), 5),
        "n_rows": int(len(y)), "n_history_sessions": int(len(sess)), "n_feedback_rows": int(is_fb.sum()),
        "cut": str(cut.date()), "trained_at": pd.Timestamp.now().isoformat(),
        "lgbm_best_iteration": int(model.best_iteration_),
        "xgb_best_iteration": int(getattr(xmodel, "best_iteration", 0) or 0),
    }
    print(f"  {metrics}")

    print("full-catalogue hit@10 vs baseline on holdout clicked sessions...")
    metrics |= _hit_experiment(fs, sess, va_s, cats, model, xmodel)
    print(f"  hit10 lgbm {metrics['hit10_model']} · ensemble {metrics['hit10_ensemble']} vs "
          f"baseline {metrics['hit10_baseline']} (top1 {metrics['top1_ensemble']}, n={metrics['n_eval']})")

    model.booster_.save_model(str(OUT / "model_lgbm_v3.txt"))
    xmodel.save_model(str(OUT / "model_xgb_v3.json"))
    meta = {"features": NUMERIC + CATEGORICAL, "numeric": NUMERIC, "categorical": CATEGORICAL,
            "categories": cats, "model_class": "LGBMClassifier+XGBClassifier", "backend": "lightgbm",
            "model_file": "model_lgbm_v3.txt", "xgb_file": "model_xgb_v3.json",
            "neg_sampling": NEG_RATE_TAG, "on_policy_feedback": True, "metrics": metrics}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {meta['model_file']} + {meta['xgb_file']} + meta.json · total {time.time() - t0:.1f}s")


def _hit_experiment(fs, sess, va_s, cats, model, xmodel=None) -> dict:
    val = sess[va_s & (sess.clicked != "")]
    val = val.sample(min(N_EVAL, len(val)), random_state=0)
    pids = fs.products.product_id.tolist()
    k = len(pids)
    best10 = set(fs.best[:10])
    h10 = h1 = h10x = b10 = n = 0
    for chunk in np.array_split(val.index.to_numpy(), max(1, int(np.ceil(len(val) / CHUNK)))):
        sub = val.loc[chunk]
        base = session_base(sub, fs).set_axis(pd.Index(sub.request_id, name="request_id"))
        slots = pd.DataFrame({"request_id": np.repeat(sub.request_id.to_numpy(), k),
                              "pid": np.tile(pids, len(sub))})
        X = feature_matrix(attach_candidates(base, slots, fs), cats)
        s_lgb = model.predict_proba(X)[:, 1]
        s_xgb = xmodel.predict_proba(X)[:, 1] if xmodel is not None else 0.0
        s_ens = (s_lgb + s_xgb) / 2 if xmodel is not None else s_lgb
        top = s_ens.reshape(len(sub), k).argsort(axis=1)[:, ::-1][:, :10]
        topl = s_lgb.reshape(len(sub), k).argsort(axis=1)[:, ::-1][:, :10]
        for r, (_, row) in enumerate(sub.iterrows()):
            names = {pids[c] for c in top[r]}
            namesl = {pids[c] for c in topl[r]}
            h10 += row.clicked in names
            h10x += row.clicked in namesl
            h1 += row.clicked == pids[top[r][0]]
            b10 += row.clicked in best10
            n += 1
    return {"hit10_ensemble": round(h10 / n, 4), "hit10_model": round(h10x / n, 4),
            "hit10_baseline": round(b10 / n, 4),
            "top1_ensemble": round(h1 / n, 4), "n_eval": int(n)}


if __name__ == "__main__":
    main()
