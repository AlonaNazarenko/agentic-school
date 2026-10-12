# Auto-generated from 08_telegram_features_v2.ipynb by export_scripts.py — edit the notebook, then re-export.
# Run:  .venv/bin/python scripts/08_telegram_features_v2.py
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[1])   # always run from the project root (data/, src/)
import matplotlib as _mpl
_mpl.use("Agg")
import matplotlib.pyplot as _plt
_FIG_DIR = _Path("figures"); _FIG_DIR.mkdir(exist_ok=True)
_fig_n = 0


def _save_show(*args, **kwargs):
    global _fig_n
    for num in _plt.get_fignums():
        _fig_n += 1
        _plt.figure(num).savefig(_FIG_DIR / ("08_telegram_features_v2_%02d.png" % _fig_n), dpi=120, bbox_inches="tight")
    _plt.close("all")


_plt.show = _save_show



# # 08 — Telegram features v2: gazetteer, direction, counts, upstream, relative rates
# Improvements over notebook 07 (Telegram v1 gave +0.005 at 15 min, ≈0 at 30/60 min):
# 1. **Gazetteer** — 424 town/raion stems from the official dataset's Ukrainian raion/hromada names → oblast.
# 2. **Location vs destination** — oblasts before «курсом на / у напрямку» = where the threat is, after = where it is heading; plus object counts («5 БпЛА», «групи»).
# 3. **Upstream** — posts about border neighbours and lift-weighted upstream oblasts, 15/60/180 min.
# 4. **Relative rates** — counts divided by the channel's own posting rate over the previous 7 days (posting volume grew 2.5×).
#
# **Original question:** do official Air Force threat posts (threat type, current oblast, direction — e.g. «БпЛА на Сумщині курсом на Полтавщину»)
# improve the best alert-history model (v4, mean per-oblast PR-AUC 0.227 ± 0.021 at 30 min)?
#
# **Data:** `data/telegram_raw.parquet` from `scrape_telegram.py` (channel `kpszsu`).
# **Leakage rules** (`src/telegram.py`, tested in `tests/test_telegram.py`):
# - a post is used only from the end of the 5-min bin it was **published** in;
# - posts edited > 5 min after publication are dropped (edited text may contain later information);
# - forwarded posts are dropped.
#
# **Protocol:** identical to notebooks 05–06 (6 monthly walk-forward folds, 60-min embargo).
#
# *Note:* Telegram API Terms §1.5 restrict using platform data to train ML models — this is an academic, non-commercial experiment; confirm with the supervisor before publishing.


import os, sys
from pathlib import Path
if not Path("src").is_dir() and Path("../src").is_dir():
    os.chdir("..")                      # notebooks/ → project root, where data/ and src/ live
sys.path.insert(0, str(Path.cwd()))
import json, sys, time, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import average_precision_score, brier_score_loss

sys.path.insert(0, str(Path.cwd()))
from src.data import load_clean, sub_grid, raion_share_grid, regime_switch_date
from src.features import (build_features_v4, build_targets, compute_lift, HORIZONS, STEP,
                          OWN, CAL, SPATIAL, WAVE, ADJ_F, SUB, RAION, REGIME, nbr_cols)
from src.telegram import clean, tg_features, WEAPON_PATTERNS, build_gazetteer, Parser, clean_v2, tg_features_v2
from src.features import adjacency

sns.set_theme(style="whitegrid")
DATA_DIR = Path("data")
tg_raw = pd.read_parquet(DATA_DIR / "telegram_raw.parquet")
msgs = clean(tg_raw)                                         # v1 parsing
GAZ = build_gazetteer(DATA_DIR / "official_data_uk.csv", DATA_DIR / "official_data_en.csv")
msgs2 = clean_v2(tg_raw, Parser(GAZ))                         # v2 parsing
print(f"gazetteer stems: {len(GAZ)}; posts with ≥1 oblast v1 {msgs.oblasts.map(len).gt(0).mean():.3f} → v2 {msgs2.oblasts.map(len).gt(0).mean():.3f}")
print(f"raw posts: {len(tg_raw)}, kept after edit/forward filter: {len(msgs)} ({len(msgs)/len(tg_raw):.0%})")
print("period:", msgs.date.min(), "→", msgs.date.max())


# ## 1. Sanity check: is this the official Air Force channel, and what does it post?


for t in msgs.sample(12, random_state=0).text:
    print("•", t[:200].replace("\n", " "))


fig, ax = plt.subplots(1, 2, figsize=(15, 4))
msgs.set_index("date").resample("W").size().plot(ax=ax[0], title="Posts per week")
pd.Series({w: msgs.weapons.map(lambda x: w in x).mean() for w in WEAPON_PATTERNS}).plot.bar(ax=ax[1], rot=0,
          title="Share of posts mentioning each threat type")
plt.show()
print("posts mentioning ≥1 oblast:", round(msgs.oblasts.map(len).gt(0).mean(), 3),
      "| with a direction phrase:", round(msgs.direction.mean(), 3))
print(msgs.explode("oblasts").oblasts.value_counts().head(24))


# ## 2. Lead time (v2 parsing): how early do posts mention an oblast before its alert starts?
# For every alert start, look back 3 h for the latest post that mentions that oblast.


state_all  = pd.read_parquet(DATA_DIR / "state_5min.parquet")
starts_all = pd.read_parquet(DATA_DIR / "starts_5min.parquet")
ep = pd.read_parquet(DATA_DIR / "episodes.parquet")
ep = ep[(ep.start >= msgs.date.min()) & (ep.start <= msgs.date.max())]

men = msgs2.explode("oblasts").dropna(subset=["oblasts"])[["date", "oblasts"]].rename(columns={"oblasts": "oblast"})
rows = []
for o, g in ep.groupby("oblast"):
    m = men[men.oblast == o].sort_values("date")
    if m.empty: continue
    j = pd.merge_asof(g.sort_values("start")[["start"]], m[["date"]], left_on="start", right_on="date",
                      direction="backward", tolerance=pd.Timedelta("3h"))
    j["oblast"] = o
    rows.append(j)
lead = pd.concat(rows)
lead["lead_min"] = (lead.start - lead.date).dt.total_seconds() / 60
print("alert starts with a mention of the oblast in the previous 3 h:", round(lead.lead_min.notna().mean(), 3))
print("… within 60 min:", round((lead.lead_min <= 60).mean(), 3), "| median lead:", lead.lead_min.median(), "min")
print(lead.groupby("oblast").lead_min.agg(["count", lambda s: s.notna().mean(), "median"]).set_axis(
    ["alerts", "share mentioned ≤3h", "median lead min"], axis=1).sort_values("share mentioned ≤3h").round(2))


# ## 3. Walk-forward experiment: v4 vs v4 + Telegram


raw = load_clean(DATA_DIR / "official_data_en.csv")
REGIME_DATE = regime_switch_date(raw)
END = min(state_all.index[-1], msgs.date.max().floor("5min"))
N_FOLDS, TEST_DAYS, ES_DAYS, TRAIN_DAYS = 6, 30, 30, 365
EMBARGO, TRAIN_EVERY = pd.Timedelta("60min"), 2
folds = []
for k in range(N_FOLDS):
    te = (END - pd.Timedelta(days=TEST_DAYS * k)).floor("5min"); ts = te - pd.Timedelta(days=TEST_DAYS)
    es = ts - pd.Timedelta(days=ES_DAYS)
    folds.append(dict(fold=N_FOLDS - k, train_start=es - pd.Timedelta(days=TRAIN_DAYS), es_start=es, test_start=ts, test_end=te))
folds = pd.DataFrame(folds).sort_values("fold").reset_index(drop=True)
# Telegram must cover the whole training span, otherwise early train rows see "no posts" by construction
if msgs.date.min() > folds.train_start.min():
    print(f"WARNING: posts start {msgs.date.min().date()}, after first train start {folds.train_start.min().date()} — "
          "train rows before that are dropped")

lo = folds.train_start.min() - pd.Timedelta(days=10)
state_all, starts_all = state_all.loc[lo:], starts_all.loc[lo:]
oblasts = list(state_all.columns)
A_adj = adjacency(oblasts)
sub_all, full_all = sub_grid(raw, state_all.index, oblasts)
rs_all = raion_share_grid(raw, state_all.index, oblasts)

V4 = OWN + CAL + SPATIAL + nbr_cols() + WAVE + ADJ_F + SUB + RAION + REGIME
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=63, min_data_in_leaf=500,
              feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=0)

def recency_weight(X, ref, hl=180):
    age = (ref - X.index.get_level_values("time")).total_seconds() / 86400
    return np.asarray(0.5 ** (age / hl), dtype=float)
print(folds)


all_preds = []
for f in folds.itertuples():
    t0 = time.time()
    lift  = compute_lift(starts_all, f.es_start)
    front = state_all.loc[f.train_start:f.es_start].mean().pipe(lambda s: s[s > .25].index.tolist())
    idx = state_all.index
    tr_lo = max(f.train_start, msgs.date.min().ceil("5min"))
    T = {"train": idx[(idx >= tr_lo) & (idx < f.es_start - EMBARGO)][::TRAIN_EVERY],
         "es":    idx[(idx >= f.es_start) & (idx < f.test_start - EMBARGO)],
         "test":  idx[(idx >= f.test_start) & (idx < min(f.test_end, idx[-1]) - EMBARGO)]}
    D = {}
    for part, t in T.items():
        X = build_features_v4(state_all, starts_all, lift, t, sub_all, full_all, rs_all, REGIME_DATE)
        X = X.join(tg_features(msgs, state_all.index, oblasts, t))
        X = X.join(tg_features_v2(msgs2, state_all.index, oblasts, t, A_adj, lift.loc[oblasts, oblasts].values)).astype(np.float32)
        X["oblast"] = pd.Categorical(X.index.get_level_values("oblast"), categories=oblasts)
        D[part] = (X, build_targets(starts_all, t))
    TG1 = [c for c in D["train"][0].columns if c.startswith("tg_")]
    TG2 = [c for c in D["train"][0].columns if c.startswith("tg2_")]
    CONFIGS = {"v4": V4, "v4 + Telegram v1": V4 + TG1, "v4 + Telegram v2": V4 + TG2}
    Xtr, Ytr = D["train"]; Xes, Yes = D["es"]; Xte, Yte = D["test"]
    w = recency_weight(Xtr, f.es_start)
    for h in HORIZONS:
        P = pd.DataFrame({"y": Yte[f"y_{h}"].values}, index=Xte.index)
        for name, feats in CONFIGS.items():
            dtr = lgb.Dataset(Xtr[feats + ["oblast"]], Ytr[f"y_{h}"], weight=w)
            des = lgb.Dataset(Xes[feats + ["oblast"]], Yes[f"y_{h}"], reference=dtr)
            mdl = lgb.train(PARAMS, dtr, 4000, valid_sets=[des], callbacks=[lgb.early_stopping(200, verbose=False)])
            P[name] = mdl.predict(Xte[feats + ["oblast"]])
            if f.fold == folds.fold.max() and h == "30min" and name == "v4 + Telegram v2":
                last_model = mdl
        P["fold"], P["horizon"] = f.fold, h
        P["front"] = P.index.get_level_values("oblast").isin(front)
        all_preds.append(P)
    print(f"fold {f.fold}: test {f.test_start.date()}→{f.test_end.date()}  ({time.time()-t0:.0f}s)")
    del D, Xtr, Xes, Xte; gc.collect()

preds = pd.concat(all_preds)
preds.to_parquet(DATA_DIR / "preds_08_telegram_v2.parquet")
MODELS = ["v4", "v4 + Telegram v1", "v4 + Telegram v2"]


def mean_ap(P, m):
    return np.mean([average_precision_score(g.y, g[m]) for _, g in P.groupby(level="oblast") if g.y.sum() >= 5])

rows = []
for (fold, h), P in preds.groupby(["fold", "horizon"]):
    for grp, sel in {"all": P, "rear": P[~P.front], "frontline": P[P.front]}.items():
        sel = sel.dropna(subset=MODELS + ["y"])
        for m in MODELS:
            rows.append({"fold": fold, "horizon": h, "group": grp, "model": m, "mean AP": mean_ap(sel, m)})
res = pd.DataFrame(rows)
res.to_csv(DATA_DIR / "results_08_telegram_v2.csv", index=False)

for grp in ["all", "rear", "frontline"]:
    t = res.query("group == @grp").groupby(["model", "horizon"])["mean AP"].agg(["mean", "std"])
    print(f"\n=== mean per-oblast PR-AUC — {grp} ===")
    print((t["mean"].round(3).astype(str) + " ± " + t["std"].round(3).astype(str)).unstack().loc[MODELS, list(HORIZONS)])

cmp = []
for h in HORIZONS:
    for grp in ["all", "rear", "frontline"]:
        t = res.query("horizon == @h and group == @grp").pivot(index="fold", columns="model", values="mean AP")
        for a, b in [("v4 + Telegram v2", "v4"), ("v4 + Telegram v2", "v4 + Telegram v1"), ("v4 + Telegram v1", "v4")]:
            d = t[a] - t[b]
            cmp.append({"horizon": h, "group": grp, "comparison": f"{a} − {b}", "mean Δ": d.mean(), "min Δ": d.min(),
                        "max Δ": d.max(), "folds won": f"{(d > 0).sum()}/{len(d)}"})
print(pd.DataFrame(cmp).round(4))


imp = pd.Series(last_model.feature_importance("gain"), index=last_model.feature_name()); imp /= imp.sum()
print("gain share of Telegram v2 features:", round(imp[[c for c in imp.index if c.startswith("tg2_")]].sum(), 3))
imp[[c for c in imp.index if c.startswith("tg2_")]].sort_values().tail(15).plot.barh(figsize=(7, 6),
    title="Top Telegram features (v4 + Telegram v2, h=30 min, latest fold)"); plt.show()


# ## Conclusions (fill in after running)
# - Coverage / lead time of posts: …
# - v4 + Telegram vs v4: …
# - Which threat types / oblast mentions matter: …
