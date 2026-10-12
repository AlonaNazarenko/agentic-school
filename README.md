# Air Raid Alert Start Forecasting — Ukraine

Forecasts, for each of 24 oblasts, the probability that an air raid alert **starts** within the next 15 / 30 / 60 minutes, using only information available at forecast time (no data leakage). The best model is LightGBM v4, with a mean per-oblast PR-AUC of **0.227 ± 0.021 at 30 min** over 6 monthly walk-forward folds (Mar–Sep 2026), about 3.3× climatology. All results are in [docs/RESULTS.md](docs/RESULTS.md).

## Project structure
```
├── notebooks/            analysis & experiments, run in order (01 → 08)
│   ├── 01_eda_air_raid_alerts.ipynb       exploratory analysis, 5-min grid, train/val/test split
│   ├── 02_baseline_snaive.ipynb           seasonal naive + climatology baselines
│   ├── 03_models_lightgbm.ipynb           own-history vs spatial (LogReg, LightGBM v1)
│   ├── 04_improved_models.ipynb           v2 features, calibration, early-warning metrics
│   ├── 05_rolling_cv_decay.ipynb          6-fold walk-forward evaluation, Hawkes-proxy features
│   ├── 06_experiments_round2.ipynb        raion-alerting regime, raion-share features (v4)
│   ├── 07_telegram_features.ipynb         Air Force Telegram posts, v1
│   └── 08_telegram_features_v2.ipynb      Telegram v2: gazetteer, direction, upstream
├── scripts/              runnable .py versions of the notebooks + tools
│   ├── 01_… – 08_….py                     generated from notebooks/ (figures → figures/)
│   ├── export_scripts.py                  regenerate scripts/*.py from notebooks/
│   └── scrape_telegram.py                 download Air Force channel posts (Telethon)
├── src/                  reusable library code
│   ├── data.py                            download, clean, episodes, 5-min grids, raion shares
│   ├── features.py                        leakage-safe features (v1–v4), targets, lift graph
│   └── telegram.py                        Telegram scraping, parsing, gazetteer, TG features
├── tests/                leakage and parser tests (pytest)
├── dashboard/            interactive Ukraine map of forecast probabilities
│   ├── make_dashboard.py                  builds index.html from saved predictions
│   ├── template.html                      HTML / CSS / JS template
│   └── index.html                         the built page (open in a browser)
├── docs/
│   ├── PLAN.md                            project plan
│   ├── RESULTS.md                         every experiment's scores, in order
│   ├── reports/                           research reports (sources linked)
│   └── research_notes/                    notes behind the reports
├── data/                 raw & processed data, predictions (git-ignored)
├── figures/              charts saved by scripts/ (git-ignored)
└── requirements.txt
```

## Setup
```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name air-raid --display-name "Python (air-raid)"
```

## Run
```bash
# notebooks (kernel "Python (air-raid)") — or the equivalent scripts:
.venv/bin/python scripts/01_eda_air_raid_alerts.py      # downloads data, builds data/*.parquet
.venv/bin/python scripts/02_baseline_snaive.py
# … 03 → 06 (05 and 06 take ~30–60 min each)

# optional: Telegram features (needs TG_API_ID / TG_API_HASH in .env, see .env.example)
.venv/bin/python scripts/scrape_telegram.py --since 2025-01-25
.venv/bin/python scripts/07_telegram_features.py

# leakage tests
.venv/bin/python -m pytest -q tests

# dashboard
.venv/bin/python dashboard/make_dashboard.py && open dashboard/index.html
```
Notebooks and scripts always run from the project root (they switch to it automatically), so `data/` and `src/` are found from any folder.

## Data & leakage
- **Alerts:** [Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset) (official data, MIT licence).
- **Telegram:** official Air Force channel posts, used only from the end of the 5-min bin they were published in. Posts edited more than 5 min later are dropped. Note that Telegram's API Terms §1.5 restrict ML training on platform data; this is a non-commercial academic experiment.
- **Leakage controls:** every feature is trailing, the neighbour graph is fitted on train only, and there's a 60-min embargo between train, validation and test. Verified by `tests/`.

*Research prototype — not a warning system. Always follow official alerts.*
