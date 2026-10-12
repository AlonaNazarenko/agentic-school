# Project Plan: Forecasting the Spatio-Temporal Spread of Air Raid Alerts in Ukraine

## 1. Problem statement
At forecast time **t**, for every oblast **i**, predict

> **P(an alert starts in oblast i within (t, t + h])**, for h ∈ {15, 30, 60} min

using only information available at **t**: the alert history of all oblasts and, optionally, Telegram text published before **t**.

**Main question:** does knowing what's happening in *other* oblasts (the spatial graph) improve on a forecast that uses only the oblast's *own* history?

## 2. What the EDA tells us (notebook 01)
| Finding | Implication for modelling |
|---|---|
| 84k oblast-level episodes, Mar 2022 → 7 Sep 2026, 24 oblasts | Enough data for ML; use the last ~12–18 months for training because tactics change |
| Median alert lasts 45 min, heavy tail (p90 ≈ 4 h) | Predict **alert starts** (events), not the on/off state, which is too easy to predict with persistence |
| Strong daily seasonality, and the hourly profile changes by year | Hour/day-of-week features; use a recent training window, or weight recent data more |
| Kyivska → Kyiv City lift 4.6×; Rivne, Cherkasy, Zhytomyr ≈ 2× | Neighbour features carry real signal, which supports the graph/Hawkes approach |
| Frontline oblasts (Dnipro 76%, Donetsk 56% time on alert) | Model **rear oblasts** as the main target and report frontline ones separately (they're nearly always on alert) |
| 742 large-scale days (≥15 oblasts on alert at once) | Feature for "attack size right now" = number of oblasts on alert; evaluate separately on mass-attack nights |

## 3. Data and leakage rules
- **Source:** the Vadimkin official dataset. Re-download right before the final run to get the newest data.
- **Split (time-based only):**
  - train: up to 9 Jul 2026
  - validation: 9 Jul → 8 Aug 2026 (hyperparameters, threshold)
  - test: 8 Aug → 7 Sep 2026 (touched **once**, at the end)
- **Walk-forward CV** inside train: 4 folds of 1 month each, expanding window.
- **Leakage checklist:**
  - Compute every feature at t from data with timestamp **< t**. In code, the shift happens *before* rolling.
  - The target window (t, t+h] never overlaps the inputs.
  - Fit scalers, encoders and graph weights (lift matrix, Hawkes parameters) on **train only**.
  - An alert that is still active at t: we know its start but **not its end**, so no "duration" feature for ongoing alerts.
  - Telegram: use the message's `date`, not the edit date, and only messages with `date < t`.
  - A unit test that randomly samples t, rebuilds the features from data truncated at t, and checks they're identical.

## 4. Notebooks / modules
```
data/                      raw + processed parquet
src/
  data.py                  download, clean, episodes, 5-min grid  (from notebook 01)
  features.py              leakage-safe feature builder
  evaluate.py              metrics, walk-forward splitter
  telegram.py              (optional) scraping + keyword counts
notebooks/
  01_eda.ipynb             ✅ done
  02_baselines.ipynb
  03_features_gbm.ipynb
  04_hawkes.ipynb
  05_graph_nn.ipynb        (stretch)
  06_telegram.ipynb        (optional)
  07_final_evaluation.ipynb
```

## 5. Modelling steps

### Step 1. Baselines (notebook 02)
1. **Climatology:** P(start) by oblast × hour of day, from train.
2. **Seasonal naive:** whether an alert started at the same time yesterday or last week.
3. **Own-history logistic regression:** the oblast's own lags only (time since last start, number of starts in the last 1/6/24 h, hour, day of week).

These set the bar. Everything below must beat **#3**.

### Step 2. Spatio-temporal features + LightGBM (notebook 03), main model
One row per (t, oblast), with t every 5 minutes (or subsampled to every 15 minutes).
- **Own:** on alert now (0/1), minutes since last start and last end, starts in the last 1/3/6/24 h.
- **Spatial:**
  - number of oblasts on alert now, and the change in that number over the last 15/30/60 min (the attack wave)
  - for each neighbour (from the train-only lift matrix, top-k): neighbour on alert, minutes since the neighbour's alert started
  - lift-weighted sum of recent neighbour starts
- **Calendar:** hour (sin/cos), day of week, night flag.
- One pooled model across all oblasts with an `oblast` categorical feature, plus a per-oblast comparison.
- **Ablation:** own-only vs own + spatial. This is the headline result.

### Step 3. Multivariate Hawkes process (notebook 04), interpretable model
- Intensity λᵢ(t) = μᵢ(hour) + Σⱼ Σ_{tₖʲ<t} αⱼᵢ · β·e^{−β(t − tₖʲ)}
- Fit with the `tick` library (or a custom implementation in PyTorch/NumPy if `tick` won't install) on train starts.
- P(start in (t, t+h]) ≈ 1 − exp(−∫λ).
- Plus: the **α matrix is an interpretable propagation graph**. Draw it on a map of Ukraine.

### Step 4. Graph neural network (notebook 05), stretch goal
- STGCN or GConvGRU from `torch-geometric-temporal`; nodes = oblasts, edges = geographic adjacency + train lift.
- Only if time allows. Compare honestly with LightGBM.

### Step 5. Telegram text (notebook 06), optional
- Channels: the official Air Force channel plus 1–2 public monitoring channels.
- Scrape with `Telethon` over the same period, and store `date` and `text`.
- Features at t: counts in the last 15/60 min of keywords (Шахед/БпЛА, Ту-95, МіГ-31К, балістика, пуски, курсом на) and mentions of each oblast name.
- **Ablation:** LightGBM with vs without the text features, especially on mass-attack nights.

## 6. Evaluation (notebook 07)
- **Metrics per horizon (15/30/60 min):** Brier score, log loss, PR-AUC (events are rare), ROC-AUC, plus a reliability (calibration) diagram.
- **Operational metric:** at a fixed false-alarm rate, how many minutes of early warning do we give before the official alert?
- **Breakdowns:** rear vs frontline oblasts; normal vs mass-attack nights; Kyiv City as the case study.
- **Significance:** block bootstrap by day for the confidence intervals of the PR-AUC difference between the model and the own-only baseline.
- **Final table:** baselines → LightGBM own → LightGBM + spatial → Hawkes → (GNN) → (+Telegram).

## 7. Risks
| Risk | Mitigation |
|---|---|
| Frontline oblasts are always on alert | Report rear oblasts as the main result |
| Concept drift (new tactics) | Recent training window and walk-forward CV; show performance over time |
| Rare events → misleading accuracy | PR-AUC, Brier score and calibration instead of accuracy |
| `tick` doesn't install on Python 3.12 | Custom exponential-kernel Hawkes model (maximum likelihood in PyTorch) |
| Telegram access/rate limits | Optional part; scrape once and cache |

## 8. Timeline (suggested)
| Week | Deliverable |
|---|---|
| 1 | ✅ EDA; `src/data.py`, `src/features.py` + leakage test |
| 2 | Baselines + LightGBM with ablation |
| 3 | Hawkes model + propagation map |
| 4 | Telegram features (optional) / GNN (stretch) |
| 5 | Final evaluation on test, report, slides |

## 9. Final deliverables
- Reproducible repo (`requirements.txt`, `.venv`, notebooks run top to bottom).
- Report: problem → data → leakage protocol → models → results → limitations.
- Key figures: propagation map (Hawkes α), PR curves per horizon, calibration plot, Kyiv case study on a recent mass-attack night.
