# Experiment Results Log — Air Raid Alert Start Forecasting

**Task:** at decision time τ, for each of 24 oblasts, predict P(an air raid alert **starts** in the oblast within the next 15 / 30 / 60 min). 5-min grid.
**Data:** official alerts (Vadimkin/ukrainian-air-raid-sirens-dataset), 15 Mar 2022 → 7 Sep 2026.
**Leakage control:** features use only bins ≤ t, the target uses bins t+1…t+k, the neighbour graph and frontline/rear split are fitted on train only, and there's a 60-min embargo between periods. Checked by `tests/` (11 tests, all passing).

Experiments are listed in the order they were run. **Two evaluation protocols were used, so their numbers aren't directly comparable:**
- **Protocol A (notebooks 02–04):** one test month, 8 Aug – 7 Sep 2026. Validation 9 Jul – 8 Aug. Train is the preceding 365 days.
- **Protocol B (notebooks 05–06):** 6 monthly walk-forward folds (test months Mar → Sep 2026). Each fold uses 365 days train → 30 days early-stopping → 30 days test, with 60-min embargoes. **This is the main, more reliable protocol.**

Metrics:
- **Pooled PR-AUC:** all oblast-rows together.
- **Mean per-oblast PR-AUC:** average over oblasts with ≥ 5 positives. **This is the main metric from notebook 04 onwards.**
- **BSS:** Brier skill score vs climatology.

Groups: **rear** = oblasts on alert < 25 % of train time (16 oblasts); **frontline** = the other 8.

---

## 1. Exploratory analysis — `01_eda_air_raid_alerts.ipynb`
Not a model. Key facts that shaped the modelling:
- 299k raw alerts → 84k merged oblast-level episodes. Median duration 45 min, p90 ≈ 4 h.
- Strong daily seasonality, and the hourly profile changes from year to year.
- Lead–lag (last 12 months): a Kyivska start raises the chance of a Kyiv City start within 30 min **4.6×**; Rivne, Cherkasy and Zhytomyr raise it ≈ 2×.
- 742 large-scale days (≥ 15 oblasts on alert at once).
- Frontline oblasts are on alert most of the time (Dnipro 76 %, Donetsk 56 %).

---

## 2. Seasonal naive baselines — `02_baseline_snaive.ipynb` (Protocol A)
Pooled PR-AUC, test month:

| Model | Rear 15 m | Rear 30 m | Rear 60 m | All 15 m | All 30 m | All 60 m |
|---|---|---|---|---|---|---|
| Random (base rate) | 0.025 | 0.049 | 0.094 | – | – | – |
| sNaive-1d | 0.029 | 0.064 | 0.141 | 0.034 | 0.073 | 0.155 |
| sNaive-7d | 0.027 | 0.059 | 0.128 | 0.032 | 0.068 | 0.141 |
| sNaive-mean 4×7d | 0.034 | 0.081 | 0.189 | 0.038 | 0.085 | 0.189 |
| sNaive-mean 7×1d | 0.040 | 0.094 | 0.210 | 0.043 | 0.098 | 0.211 |
| **Climatology** (oblast × hour) | **0.073** | **0.145** | **0.274** | **0.060** | **0.118** | **0.223** |

**Conclusion:** attacks don't repeat at the same time each day or week. Within a single oblast, sNaive PR-AUC ≈ base rate (lift ≈ 1.0). Climatology is the reference baseline.

---

## 3. Own-history vs spatial models — `03_models_lightgbm.ipynb` (Protocol A)
Pooled PR-AUC, test month. Train sampled every 15 min.

| Model | All 15 m | All 30 m | All 60 m | Rear 30 m | Frontline 30 m |
|---|---|---|---|---|---|
| Climatology | 0.060 | 0.118 | 0.223 | 0.144 | 0.116 |
| sNaive-mean 7×1d | 0.043 | 0.098 | 0.211 | 0.095 | 0.107 |
| LogReg-own | 0.193 | 0.295 | 0.418 | 0.233 | 0.354 |
| LightGBM-own | 0.198 | 0.302 | 0.416 | 0.235 | 0.370 |
| **LightGBM-spatial v1** | **0.209** | **0.312** | **0.433** | **0.240** | **0.375** |

- Rear ROC-AUC at 30 min: climatology 0.805 → spatial v1 **0.884**.
- Spatial − own, rear, block bootstrap: 15 m **+0.016** [+0.004, +0.030]; 30 m +0.005 [−0.008, +0.019]; 60 m **−0.017** [−0.034, −0.003].
- Per oblast, spatial features improved 17 of 21 oblasts. Kyiv City 0.17 → **0.33**; Ternopil, Cherkasy and Volyn each gain about +0.10.
- Early warning (global threshold at precision 0.5, 60 m): own 23 % / spatial 20 % of rear alerts warned, median lead 50 / 35 min.

**Conclusion:** recent own history makes models ≈ 3× better than seasonal baselines. Spatial information helps most per oblast, but the pooled metric hides this, which led to switching the main metric to the per-oblast mean.

---

## 4. Improved features and evaluation — `04_improved_models.ipynb` (Protocol A)
New features in v2:
- attack-wave direction (share of oblasts on alert per macro-region)
- border adjacency
- raion-level alert counts
- burst signals

Training changes: rows every 5 min, recency weights (half-life 180 d).

**Mean per-oblast PR-AUC** (test month):

| Model | All 15 m | All 30 m | All 60 m | Rear 15 m | Rear 30 m | Rear 60 m | Front 30 m |
|---|---|---|---|---|---|---|---|
| Climatology | 0.041 | 0.082 | 0.155 | 0.032 | 0.064 | 0.123 | 0.110 |
| LightGBM-own | 0.132 | 0.201 | 0.280 | 0.064 | 0.119 | 0.200 | 0.335 |
| LightGBM-spatial v1 | 0.165 | 0.239 | 0.321 | 0.115 | 0.174 | 0.247 | 0.344 |
| LightGBM-spatial v2 | 0.184 | 0.255 | 0.332 | 0.134 | 0.191 | 0.261 | 0.359 |
| **v2 + recency weights** | **0.188** | **0.255** | **0.335** | **0.140** | **0.196** | **0.262** | 0.351 |
| v2 + rw, isotonic-calibrated | 0.179 | 0.247 | 0.324 | 0.133 | 0.191 | 0.255 | 0.339 |

Pooled PR-AUC, all: v2 + rw 0.222 / 0.322 / 0.445 (15 / 30 / 60 m).

- Brier, all, 30 m: climatology 0.0552 → v2 + rw 0.0481 → calibrated **0.0477**.
- v2 + rw − v1, block bootstrap: **+0.025** [+0.017, +0.036] / **+0.018** [+0.004, +0.032] / **+0.012** [+0.006, +0.019].
- v2 + rw − own: about **+0.06** at all horizons (CI excludes 0).
- Largest per-oblast gains vs v1 (30 m): Kyivska +0.055, Chernivtsi +0.054, Dnipro +0.040, Volyn +0.035, Kyiv City +0.033 (→ 0.355).
- Feature importance (gain share): own 35 %, oblast ID 23 %, spatial v1 16 %, adjacency 9 %, calendar 7 %, raion 6 %, wave 5 %.
- Early warning, 60 m (all / rear warned, median lead):

  | Model | Global threshold | Per-oblast threshold |
  |---|---|---|
  | own | 51 % / 34 %, 45 min | 42 % / 27 % |
  | v1 | 56 % / 41 %, 40 min | 45 % / 30 % |
  | v2 + rw | **58 % / 45 %, 40 min** | 46 % / 28 % |

**Conclusion:** v2 features give a significant gain at all horizons. Calibration improves Brier but lowers PR-AUC. Per-oblast thresholds make early warning worse, so keep a global threshold.

---

## 5. Walk-forward evaluation + Hawkes-proxy decay features — `05_rolling_cv_decay.ipynb` (Protocol B)
v3 = v2 + exponentially decayed start counts (half-lives 15 m / 1 h / 6 h / 24 h; own, neighbours, region, national).
**Mean ± sd over 6 folds:**

| Mean per-oblast PR-AUC | All 15 m | All 30 m | All 60 m | Rear 30 m |
|---|---|---|---|---|
| Climatology | 0.034 ± 0.005 | 0.069 ± 0.009 | 0.130 ± 0.017 | 0.052 ± 0.010 |
| LightGBM-own | 0.110 ± 0.015 | 0.168 ± 0.022 | 0.232 ± 0.028 | 0.075 ± 0.024 |
| **LightGBM v2** | **0.152 ± 0.018** | **0.222 ± 0.020** | **0.295 ± 0.026** | **0.147 ± 0.031** |
| LightGBM v3 (+decay) | 0.153 ± 0.022 | 0.221 ± 0.021 | 0.291 ± 0.026 | 0.147 ± 0.031 |

| Other metrics, all oblasts, 30 m | Climatology | Own | v2 | v3 |
|---|---|---|---|---|
| BSS vs climatology | 0 | 0.095 | **0.111** | 0.110 |
| Log loss | 0.200 | 0.177 | **0.169** | 0.170 |
| Mean PR-AUC / prevalence (× random) | 1.3 | 4.7 | 11.8 ± 6.5 | 11.9 ± 6.6 |

- v2 − own: wins **6/6 folds** at every horizon (all and rear). Rear 30 m **+0.072** (≈ 2×).
- v3 − v2: −0.001 at 30 m (won 1/6). Decay features have 25 % of gain importance but duplicate the rolling counts, so they add **no** skill.

**Conclusion:** the honest benchmark is **0.222 ± 0.020** (30 m). The single test month from notebooks 02–04 was easier than average. A full Hawkes model is unlikely to add accuracy; its value is the interpretable propagation map.

---

## 6. Research round 2 experiments — `06_experiments_round2.ipynb` (Protocol B)
- **Regime:** raion-level alerting switch detected in the data at **15 Nov 2025** (official nationwide rollout: 15 Dec 2025).
- **v4** = v2 + raion-share features (own, neighbours now and −15…−120 min, lift-weighted upstream, national) + regime flag.

| Mean per-oblast PR-AUC, 6 folds | All 15 m | All 30 m | All 60 m | Rear 30 m | Front 30 m |
|---|---|---|---|---|---|
| Climatology | 0.034 | 0.069 | 0.130 | 0.052 | 0.106 |
| v2 | 0.152 ± 0.018 | 0.222 ± 0.020 | 0.295 ± 0.026 | 0.147 | 0.397 |
| v2 + regime flag | 0.152 ± 0.017 | 0.220 ± 0.019 | 0.297 ± 0.027 | 0.144 | 0.396 |
| v2 post-switch only | 0.153 ± 0.023 | 0.220 ± 0.022 | 0.291 ± 0.031 | 0.147 | 0.388 |
| **v4 (+raion share)** | **0.156 ± 0.015** | **0.227 ± 0.021** | **0.298 ± 0.027** | **0.153** | **0.398** |
| v4 post-switch only | 0.155 ± 0.020 | 0.220 ± 0.022 | 0.292 ± 0.029 | 0.148 | 0.386 |

v4 − v2 (folds won):

| | 15 m | 30 m | 60 m |
|---|---|---|---|
| All | +0.004 (4/6) | **+0.004 (6/6)** | **+0.003 (6/6)** |
| Rear | **+0.007 (5/6)** | **+0.005 (6/6)** | **+0.004 (6/6)** |
| Frontline | −0.005 (1/6) | +0.002 (4/6) | +0.003 (4/6) |

Raion-share features have 17 % of gain importance.

- **Regime flag:** no effect.
- **Training only on post-switch data:** worse, especially for frontline (−0.008 to −0.012), because it throws away training data.

**Simultaneous-start diagnostic** (≥ 8 oblasts starting in the same 5-min bin ≈ MiG-31K or ballistic, near-zero lead time):
- 10 % of starts and 6 % of positive targets are of this kind.

| Mean per-oblast PR-AUC | All events 15 / 30 / 60 m | Excluding simultaneous-only positives |
|---|---|---|
| Climatology | 0.034 / 0.069 / 0.130 | 0.034 / 0.066 / 0.125 |
| v2 | 0.152 / 0.222 / 0.295 | 0.166 / 0.249 / 0.321 |
| v4 | 0.156 / 0.227 / 0.298 | **0.171 / 0.255 / 0.323** |

**Conclusion:** v4 is the new best model, with a small but consistent gain (6/6 folds at 30 and 60 min). The raion-alerting switch needs no special handling beyond recency weights. About 0.03 of the PR-AUC gap comes from alerts that can't be forecast from alert history.

---

## General evaluation

### Progress of the best model (30-min horizon, mean per-oblast PR-AUC)
| Step | Model | Protocol A (1 test month) | Protocol B (6 folds) |
|---|---|---|---|
| Baseline | Climatology | 0.082 | 0.069 ± 0.009 |
| + own history | LightGBM-own | 0.201 | 0.168 ± 0.022 |
| + neighbours (lift, top-k) | Spatial v1 | 0.239 | – |
| + wave / adjacency / raion counts / recency | Spatial v2 | 0.255 | 0.222 ± 0.020 |
| + decay (Hawkes proxy) | v3 | – | 0.221 ± 0.021 (no gain) |
| + raion share | **v4 (final)** | – | **0.227 ± 0.021** |

### Final model: LightGBM v4
| Horizon | Mean per-oblast PR-AUC | vs climatology | Rear | Frontline | Excl. simultaneous |
|---|---|---|---|---|---|
| 15 min | 0.156 ± 0.015 | 4.6× | 0.104 | 0.275 | 0.171 |
| 30 min | **0.227 ± 0.021** | **3.3×** | 0.153 | 0.398 | 0.255 |
| 60 min | 0.298 ± 0.027 | 2.3× | 0.213 | 0.495 | 0.323 |

### What worked / what didn't
| Worked | Didn't help |
|---|---|
| Own recent history (≈ 3× over seasonal baselines) | Seasonal naive (≈ random within an oblast) |
| Spatial features, especially for rear oblasts (≈ 2× over own-only) | Hawkes-proxy decay counts (redundant) |
| v2 wave / adjacency / raion features (significant, all horizons) | Regime flag; training only after the raion switch |
| Raion-share features (6/6 folds at 30/60 min) | Isotonic calibration for ranking (good for Brier only) |
| Recency weights (small gain at 15 min) | Per-oblast early-warning thresholds |

### Limitations and next steps
- About 6 % of positives (simultaneous nationwide starts) are essentially unforecastable from alert history.
- Gains from alert history have levelled off: +0.054 → +0.017 → +0.005 → 0.
- Further progress needs **new information**:
  - Kaggle attack-intensity data, lagged to its publication time
  - Telegram, pending legal and terms-of-service review
- Also planned: a two-stage "national wave × oblast joins" model, corridor-lag features, and a Hawkes propagation map for interpretation.
- The official data ends on 7 Sep 2026. Re-run the evaluation when newer data is published.
