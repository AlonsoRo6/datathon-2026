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
    base = session_base(sess, fs).set_axis(pd.Index(sess.request_id, name="request_id"))
    slots, lens = to_slots(sess)
    purch = sess.set_index("request_id").purchased.reindex(slots.request_id).fillna("").to_numpy()
    y = (slots.pid.to_numpy() == purch).astype("int8")
    w = 1.0 / np.clip(slots.prop.to_numpy(dtype="float64"), 0.01, 1.0)
    feats = attach_candidates(base, slots, fs)
    print(f"  {len(feats):,} training rows, positive rate {y.mean() * 100:.2f}%")

    ts_s = pd.to_datetime(sess.ts)
    cut = ts_s.max() - pd.Timedelta(days=HOLDOUT_DAYS)
    va_s = (ts_s >= cut).to_numpy()
    va = np.repeat(va_s, lens)
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
    print(f"  best iteration: {model.best_iteration_}")
    p_va = model.predict_proba(X_va)[:, 1]
    metrics = {
        "auc_valid": round(float(roc_auc_score(y_va, p_va, sample_weight=w_va)), 4),
        "logloss_valid": round(float(log_loss(y_va, p_va, sample_weight=w_va)), 5),
        "positive_rate": round(float(y.mean()), 5),
        "n_rows": int(len(y)), "n_sessions": int(len(sess)),
        "cut": str(cut.date()), "trained_at": pd.Timestamp.now().isoformat(),
        "best_iteration": int(model.best_iteration_),
    }
    print(f"  {metrics}")

    print(f"full-catalogue hit@10 vs baseline on holdout clicked sessions...")
    metrics |= _hit_experiment(fs, sess, va_s, cats, model)
    print(f"  hit10 model {metrics['hit10_model']} vs baseline {metrics['hit10_baseline']} "
          f"(top1 {metrics['top1_model']}, n={metrics['n_eval']})")

    model.booster_.save_model(str(OUT / "model_lgbm.txt"))
    meta = {"features": NUMERIC + CATEGORICAL, "numeric": NUMERIC, "categorical": CATEGORICAL,
            "categories": cats, "model_class": "LGBMClassifier", "backend": "lightgbm",
            "model_file": "model_lgbm.txt", "neg_sampling": NEG_RATE_TAG, "metrics": metrics}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {meta['model_file']} + meta.json · total {time.time() - t0:.1f}s")


def _hit_experiment(fs, sess, va_s, cats, model) -> dict:
    val = sess[va_s & (sess.clicked != "")]
    val = val.sample(min(N_EVAL, len(val)), random_state=0)
    pids = fs.products.product_id.tolist()
    k = len(pids)
    best10 = set(fs.best[:10])
    h10 = h1 = b10 = n = 0
    for chunk in np.array_split(val.index.to_numpy(), max(1, int(np.ceil(len(val) / CHUNK)))):
        sub = val.loc[chunk]
        base = session_base(sub, fs).set_axis(pd.Index(sub.request_id, name="request_id"))
        slots = pd.DataFrame({"request_id": np.repeat(sub.request_id.to_numpy(), k),
                              "pid": np.tile(pids, len(sub))})
        X = feature_matrix(attach_candidates(base, slots, fs), cats)
        scores = model.predict_proba(X)[:, 1].reshape(len(sub), k)
        top = scores.argsort(axis=1)[:, ::-1][:, :10]
        for r, (_, row) in enumerate(sub.iterrows()):
            names = {pids[c] for c in top[r]}
            h10 += row.clicked in names
            h1 += row.clicked == pids[top[r][0]]
            b10 += row.clicked in best10
            n += 1
    return {"hit10_model": round(h10 / n, 4), "hit10_baseline": round(b10 / n, 4),
            "top1_model": round(h1 / n, 4), "n_eval": int(n)}


if __name__ == "__main__":
    main()
