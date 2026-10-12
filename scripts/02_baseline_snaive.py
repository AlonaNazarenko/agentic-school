# Auto-generated from 02_baseline_snaive.ipynb by export_scripts.py — edit the notebook, then re-export.
# Run:  .venv/bin/python scripts/02_baseline_snaive.py
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
        _plt.figure(num).savefig(_FIG_DIR / ("02_baseline_snaive_%02d.png" % _fig_n), dpi=120, bbox_inches="tight")
    _plt.close("all")


_plt.show = _save_show



# # 02 — Baseline: Seasonal Naive (sNaive)
# **Task:** at time *t*, for every oblast, predict **P(an alert starts in (t, t+h])**, h ∈ {15, 30, 60} min.
#
# **Seasonal naive** = "the future looks like the same time window in the previous season":
# - **sNaive-1d:** prediction = did an alert start in the same window *yesterday*?
# - **sNaive-7d:** … in the same window *one week ago*?
# - **sNaive-mean (probabilistic):** share of the last *K* days/weeks where an alert started in that window.
#
# For reference we also include **climatology** (P(start) by oblast × hour, from train).
#
# **No leakage:** the lagged window (t−L, t−L+h] ends at t−L+h ≤ t because h ≤ 60 min < L = 1 day.


import os, sys
from pathlib import Path
if not Path("src").is_dir() and Path("../src").is_dir():
    os.chdir("..")                      # notebooks/ → project root, where data/ and src/ live
sys.path.insert(0, str(Path.cwd()))
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, precision_recall_curve

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (13, 4)
DATA_DIR = Path("data")

starts = pd.read_parquet(DATA_DIR / "starts_5min.parquet")
state  = pd.read_parquet(DATA_DIR / "state_5min.parquet")
split  = {k: pd.Timestamp(v) for k, v in json.loads((DATA_DIR / "split.json").read_text()).items()}
STEP = starts.index[1] - starts.index[0]
DAY  = int(pd.Timedelta("1D") / STEP)
HORIZONS = {"15min": 3, "30min": 6, "60min": 12}
oblasts = list(starts.columns)
print(starts.shape, STEP, split)


# ## 1. Target: alert start within (t, t+h]
# Computed with cumulative sums: `y[t] = cs[t+k] − cs[t]` = number of starts in steps t+1 … t+k.


def future_starts(S: pd.DataFrame, k: int) -> pd.DataFrame:
    cs = np.vstack([np.zeros((1, S.shape[1])), np.cumsum(S.values, axis=0)])   # cs[i] = starts in steps < i
    n = len(S)
    y = np.full(S.shape, np.nan)
    y[: n - k] = cs[k + 1 : n + 1] - cs[1 : n - k + 1]                          # steps t+1 … t+k
    return pd.DataFrame((y > 0).astype(float), index=S.index, columns=S.columns).where(~np.isnan(y))

Y = {h: future_starts(starts, k) for h, k in HORIZONS.items()}
print(Y["30min"].loc[split["test_start"]:].mean().sort_values(ascending=False).round(3).head())


# ## 2. Oblast groups
# Frontline oblasts are under alert most of the time, which makes "a new alert starts" a different problem there.
# Group defined **on train data only**: share of time on alert > 25 % → frontline.


train_state = state.loc[split["val_start"] - pd.Timedelta(days=365): split["val_start"]]
share = train_state.mean().sort_values(ascending=False)
FRONT = share[share > 0.25].index.tolist()
REAR  = [o for o in oblasts if o not in FRONT]
print("frontline:", FRONT)
print("rear     :", REAR)


# ## 3. Baseline forecasts


def snaive(Yh: pd.DataFrame, lag_days: int) -> pd.DataFrame:
    return Yh.shift(lag_days * DAY)

def snaive_mean(Yh: pd.DataFrame, lag_days: int, K: int) -> pd.DataFrame:
    return sum(Yh.shift(m * lag_days * DAY) for m in range(1, K + 1)) / K

def climatology(Yh: pd.DataFrame, train_end, days=365) -> pd.DataFrame:
    tr = Yh.loc[train_end - pd.Timedelta(days=days): train_end - pd.Timedelta("1h")]
    prof = tr.groupby(tr.index.hour).mean()                     # hour × oblast, train only
    return pd.DataFrame(prof.loc[Yh.index.hour].values, index=Yh.index, columns=Yh.columns)

def make_forecasts(Yh, train_end):
    return {
        "sNaive-1d":        snaive(Yh, 1),
        "sNaive-7d":        snaive(Yh, 7),
        "sNaive-mean 7×1d": snaive_mean(Yh, 1, 7),
        "sNaive-mean 4×7d": snaive_mean(Yh, 7, 4),
        "Climatology":      climatology(Yh, train_end),
    }

# Leakage sanity check: the lag-1d window must end before t
for h, k in HORIZONS.items():
    assert k < DAY, "horizon must be shorter than the seasonal lag"
print("ok")


# ## 4. Evaluation


def evaluate(F: dict, Yh: pd.DataFrame, start, end, cols) -> pd.DataFrame:
    rows = []
    y = Yh.loc[start:end, cols].values.ravel()
    for name, P in F.items():
        p = P.loc[start:end, cols].values.ravel()
        m = ~np.isnan(y) & ~np.isnan(p)
        yy, pp = y[m], np.clip(p[m], 0, 1)
        rows.append({"model": name,
                     "Brier": brier_score_loss(yy, pp),
                     "PR-AUC": average_precision_score(yy, pp),
                     "ROC-AUC": roc_auc_score(yy, pp),
                     "base rate": yy.mean()})
    return pd.DataFrame(rows).set_index("model")

results = []
for h in HORIZONS:
    F = make_forecasts(Y[h], split["val_start"])
    for period, (a, b) in {"val": (split["val_start"], split["test_start"]),
                           "test": (split["test_start"], None)}.items():
        for grp, cols in {"all": oblasts, "rear": REAR, "frontline": FRONT}.items():
            r = evaluate(F, Y[h], a, b, cols)
            r["horizon"], r["period"], r["group"] = h, period, grp
            results.append(r.reset_index())
res = pd.concat(results)
print(res.query("period == 'test' and group == 'rear'").round(4))


# Summary: PR-AUC on test, horizon × model, per group
for grp in ["rear", "frontline", "all"]:
    t = res.query("period == 'test' and group == @grp").pivot(index="model", columns="horizon", values="PR-AUC")
    print(f"\n=== PR-AUC, test, {grp} ===")
    print(t[list(HORIZONS)].round(3))


fig, ax = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
for a, grp in zip(ax, ["rear", "frontline", "all"]):
    t = res.query("period == 'test' and group == @grp").pivot(index="horizon", columns="model", values="PR-AUC").loc[list(HORIZONS)]
    t.plot(kind="bar", ax=a, legend=(grp == "rear"), rot=0, title=f"PR-AUC test — {grp}")
    base = res.query("period == 'test' and group == @grp").groupby("horizon")["base rate"].first().loc[list(HORIZONS)]
    a.plot(range(3), base.values, "k--", label="random (base rate)")
plt.tight_layout(); plt.show()


# ## 5. Precision–recall curves (h = 30 min, test, rear oblasts)


h = "30min"
F = make_forecasts(Y[h], split["val_start"])
y = Y[h].loc[split["test_start"]:, REAR].values.ravel()
plt.figure(figsize=(7, 6))
for name, P in F.items():
    p = P.loc[split["test_start"]:, REAR].values.ravel()
    m = ~np.isnan(y) & ~np.isnan(p)
    pr, rc, _ = precision_recall_curve(y[m], p[m])
    plt.plot(rc, pr, label=f"{name} (AP={average_precision_score(y[m], p[m]):.3f})")
plt.axhline(np.nanmean(y), ls="--", c="k", label="base rate")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.title("PR curve — h=30 min, rear, test")
plt.show()


# ## 6. Per-oblast results (h = 30 min, test, best sNaive)


best = res.query("period=='val' and group=='rear' and horizon=='30min' and model.str.startswith('sNaive')", engine="python") \
          .sort_values("PR-AUC").iloc[-1]["model"]
print("best sNaive on validation:", best)

per = []
for o in oblasts:
    yy = Y[h].loc[split["test_start"]:, o]; pp = F[best].loc[split["test_start"]:, o]
    m = yy.notna() & pp.notna()
    if yy[m].sum() < 5: continue
    per.append({"oblast": o, "group": "front" if o in FRONT else "rear",
                "PR-AUC": average_precision_score(yy[m], pp[m]),
                "base rate": yy[m].mean(), "n starts": int(starts.loc[split['test_start']:, o].sum())})
per = pd.DataFrame(per).set_index("oblast").sort_values("PR-AUC")
per["lift over random"] = per["PR-AUC"] / per["base rate"]
print(per.round(3))


# ## 7. Example: Kyiv City, a week in the test period


o = "Kyiv City"
w0 = split["test_start"]; w1 = w0 + pd.Timedelta(days=7)
fig, ax = plt.subplots(figsize=(15, 4))
ax.fill_between(Y[h].loc[w0:w1].index, Y[h].loc[w0:w1, o], step="post", alpha=.3, label="actual: start within 30 min")
F[best].loc[w0:w1, o].plot(ax=ax, label=best, c="C3")
F["Climatology"].loc[w0:w1, o].plot(ax=ax, label="Climatology", c="C2")
ax.legend(); ax.set_title(f"{o}: sNaive vs actual")
plt.show()


res.to_csv(DATA_DIR / "results_02_snaive.csv", index=False)
print("saved")


# ## Conclusions (fill in after running)
# - Best sNaive variant: …
# - sNaive vs climatology: …
# - Rear vs frontline: …
# - **This is the bar** for the next models (own-history logistic regression, LightGBM with spatial features).
