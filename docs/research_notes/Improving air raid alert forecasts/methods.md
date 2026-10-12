# Modelling methods for spatio-temporal forecasting of rare binary events (air raid alert starts, 24 regions, 15/30/60-min horizons)

## Multivariate Hawkes / self-exciting point processes: applications, comparison with ML, Python libraries

### Takeaway
Self-exciting (Hawkes/ETAS-style) models have strong, peer-reviewed track records for clustered events (crime, terrorism) and beat human/hotspot baselines, but direct head-to-head evidence vs gradient boosting is thin. tick is the canonical Hawkes library but its Python 3.12 support could not be confirmed and the repo shows signs of low maintenance.

### Cited Findings
- Mohler, Short, Brantingham, Schoenberg & Tita, "Self-Exciting Point Process Modeling of Crime", JASA 106(493), 2011, pp.100–108: intensity = spatially varying background + self-exciting (near-repeat) component, borrowed from seismology; fitted nonparametrically on 5,376 LAPD residential burglaries (subsampling needed for compute) — [Georgia Tech PDF](https://mshort9.math.gatech.edu/papers/crime3.pdf); [Review, Reinhart 2017](https://arxiv.org/pdf/1708.02647)
- A Bogotá replication found a simplified Mohler-2011 SEPP predicted crime better than KDE or ellipse hotspot models — [search summary of SEPP literature; review](https://arxiv.org/pdf/1708.02647) (secondary; original Bogotá study not fetched)
- Spatial-covariate extension: if leading indicators, near-repeats and spatial features are not modelled jointly their effects are confounded — [Reinhart & Greenhouse, arXiv 1708.03579](https://ar5iv.arxiv.org/html/1708.03579)
- Mohler et al. 2015, JASA 110(512):1399–1411, randomized controlled field trials (LAPD, Kent UK): ETAS forecasts predicted 1.4–2.2x as much crime as dedicated crime analysts; ETAS-guided patrols gave avg 7.4% crime reduction per unit patrol time — [preprint PDF](https://mshort9.math.gatech.edu/papers/MohlerEtAl-2015-JASA-Predictive-InPress.pdf); [CEBCP summary](https://cebcp.org/evidence-based-policing/the-matrix/micro-places/micro-places-mohler-et-al-2015/)
- Porter & White, "Self-exciting hurdle models for terrorist activity", Annals of Applied Statistics 6(1), 2012: daily attack counts in Indonesia 1994–2007; hurdle model separates P(zero) from count on attack days; shot-noise self-excitation; power-law count with shot-noise-driven parameter fit best. Hurdle (not zero-inflated) chosen because zeros come from a separate process — directly analogous to rare-binary alert onsets — [arXiv 1203.3680](https://arxiv.org/pdf/1203.3680); [Project Euclid](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-6/issue-1/Self-exciting-hurdle-models-for-terrorist/10.1214/11-AOAS513.pdf)
- White, Porter & Mazerolle 2013 (J. Quant. Criminology) used SEPPs for Indonesia/Philippines/Thailand 2000–2010 to define risk, resilience, volatility metrics — [OJP abstract](https://ojp.gov/ncjrs/virtual-library/abstracts/terrorism-risk-resilience-and-volatility-comparison-terrorism)
- tick (JMLR 18, 2018): simulation and fitting of Hawkes with exponential / sum-of-exponential kernels, basis-kernel combos, sparse interactions — [JMLR](https://jmlr.org/papers/v18/17-381.html)
- tick GitHub README says "requires Python 3.5 or newer", no mention of 3.11/3.12; Travis CI disabled, 78 open issues, Windows support "experimental" — [GitHub](https://github.com/X-DataInitiative/tick)

### Inferences
- For 24 regions, a 24-dim multivariate exponential-kernel Hawkes (576 excitation params + 24 baselines) is tractable and gives an interpretable cross-region excitation matrix; add time-of-day modulated baseline since alerts follow strike-wave patterns.
- Porter & White's hurdle framing suggests modelling "onset" and "co-onsets across regions" separately (alerts often trigger many regions at once — mass simultaneous marks break simple Hawkes assumptions).
- tick on 3.12: treat as risky; test `pip install tick` in a 3.12 venv; fallback is a custom PyTorch/NumPy exponential-kernel MLE (closed-form recursive likelihood, ~100 lines) or EasyTPP's classical baselines.

### Gaps
- Could not confirm tick's latest PyPI version / 3.12 wheels (PyPI page blocked by client challenge). hawkeslib status not checked.
- No peer-reviewed direct comparison of Hawkes vs gradient boosting on conflict/crime found.

## Neural temporal point processes (RMTPP, Neural Hawkes, THP, SAHP): benchmarks, do they beat classic Hawkes?

### Takeaway
EasyTPP (ICLR 2024) is the standard open benchmark/library; an independent large-scale study (Bosser & Ben Taieb, TMLR 2023) found results are highly setup-dependent and neural TPPs have calibration problems — gains over classic Hawkes are not uniform.

### Cited Findings
- EasyTPP (Ant Group et al., ICLR 2024): first central repository for TPP research — unified data interface, evaluation programs, implementations of popular neural TPPs; repo ant-research/EasyTemporalPointProcess — [arXiv 2307.08097](https://arxiv.org/abs/2307.08097); [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/e66233a208ef32f56df6312263239fa0-Abstract-Conference.html)
- EasyTPP authors note the prior toolbox (Poppy) unmaintained since 2021 — [arXiv PDF](https://arxiv.org/pdf/2307.08097)
- Bosser & Ben Taieb (TMLR 2023): prior neural TPP papers used inconsistent baselines/datasets; their unified study on 15 real datasets plus classical Hawkes/Poisson baselines examined history encoder, event encoding, decoder; found neural TPP mark distributions are miscalibrated and drew conclusions on importance of history size — [arXiv 2306.17066](https://arxiv.org/abs/2306.17066v2); [TMLR](https://mlanthology.org/tmlr/2023/bosser2023tmlr-predictive)

### Inferences
- With only 24 marks and long histories, classic multivariate Hawkes is a strong baseline; neural TPPs (THP/SAHP) are worth trying only after a well-tuned Hawkes, and must be judged on calibrated probability of onset within the horizon window (integrate intensity), not on log-likelihood alone.

### Gaps
- Did not retrieve EasyTPP's numerical tables; cannot state which model wins on which dataset.

## Spatio-temporal GNNs (STGCN, GraphWaveNet, DCRNN, torch-geometric-temporal) vs gradient boosting

### Takeaway
STGNN superiority is largely reported on dense traffic benchmarks; later work shows simpler models match them, and boosting won a large traffic competition. No evidence found that STGNNs beat GBDT on small graphs with rare binary targets.

### Cited Findings
- STGCN (IJCAI 2018) reported outperforming baselines on real traffic datasets — [IJCAI](https://www.ijcai.org/proceedings/2018/0505)
- SimST ("Do We Really Need GNNs for Traffic Forecasting?", 2023): non-message-passing model with comparable accuracy and up to 39x throughput across five benchmarks — [arXiv 2301.12603](https://arxiv.org/pdf/2301.12603)
- STG4Traffic: fair comparison of STGNN baselines was an open problem; builds standardized benchmark — [arXiv 2307.00495](https://arxiv.org/pdf/2307.00495)
- Traffic4cast 2022 gradient boosting solution argues GBDT is strong on structured traffic data — [arXiv 2211.00157](https://arxiv.org/pdf/2211.00157)

### Inferences
- With 24 nodes, LightGBM with neighbour-lag features already encodes the graph; a GNN's main potential gain is learning the adjacency. Heavy class imbalance and few positives argue for GBDT; try a GNN only as a stacked/ensemble member.

### Gaps
- No direct GBDT-vs-STGNN study on rare-event binary classification found; torch-geometric-temporal version compatibility not checked.

## Conflict forecasting systems: ViEWS, ACLED CAST, Conflict Forecast

### Takeaway
All operational systems use tree ensembles/ensembles of models with lagged violence and spatial spillover, evaluated on true-future windows; ensembles beat individual models; onsets in previously calm units are the hardest case.

### Cited Findings
- ViEWS (Hegre et al., JPR 2019): monthly country and sub-national (PRIO-GRID) forecasts up to 36 months, ensemble of constituent models — [PRIO](https://www.prio.org/publications/11231)
- ViEWS 2020 (JPR 2021): revised training/evaluation/ensemble weighting; models specialised for near vs distant horizons; retrospective out-of-sample evaluation Jul 2018–Dec 2019 — [PRIO](https://www.prio.org/publications/13335); [SAGE](https://journals.sagepub.com/doi/full/10.1177/0022343320962157)
- ViEWS escalation competition (International Interactions 2022): simple weighted ensemble of contributions beat the average individual model — [PRIO](https://www.prio.org/publications/13337)
- VIEWS Prediction Challenge 2023/24: probabilistic (distributional) fatality forecasts, 13 institutions, 23+ models, true-future window Jul 2024–Jun 2025 — [PRIO project](https://www.prio.org/projects/2002)
- Mueller & Rauh, "The Hard Problem of Prediction for Conflict Prevention" (JEEA 2022): onsets in peaceful countries are the hard problem; topic model on 4M+ news articles -> random forest; separate within- vs between-country variation; runs conflictforecast.org — [Cambridge](https://www.econ.cam.ac.uk/research/pub-abstracts?pub=1579); [VoxEU](https://cepr.org/voxeu/columns/why-we-fail-prevent-civil-wars-forecasters-perspective)
- ACLED CAST: forecasts event counts per rolling 4-week period, 6 periods ahead, admin-1 level; inputs include recent violence trends, cross-border spillover, WorldPop, V-Dem; weekly refresh with published accuracy — [CAST methodology](https://acleddata.com/methodology/cast-methodology)
- CAST self-reported misses are largest during sudden escalations (Oct 2023: +24%, driven by explosions/remote violence) — [CAST Oct report](https://acleddata.com/wp-content/data-viz/cast-report/october/report.html)

### Inferences
- Lessons: evaluate on rolling true-future splits; separate metrics for "onset after quiet" vs "continuation"; ensembles of diverse models; report calibrated probabilities (Brier/log-loss, PR-AUC) rather than accuracy.

### Gaps
- No independent accuracy evaluation of CAST found; ViEWS challenge scores not retrieved.

## ETAS aftershock forecasting as analogy; combining ETAS with ML

### Takeaway
Recent work finds ML matches or slightly beats ETAS for aftershock counts and is far faster; hybrids (ETAS + deep learning) address ETAS underestimation.

### Cited Findings
- Edinburgh/BGS (Earth, Planets and Space): ML models trained on CA, NZ, Italy, Japan, Greece forecast 24h aftershock counts after M4+ with similar performance to ETAS, in seconds vs hours/days — [BGS](https://www.bgs.ac.uk/news/new-research-shows-ai-earthquake-tools-forecast-aftershock-risk-in-seconds/)
- RECAST (UCSC): deep model slightly beat ETAS on Southern California, especially with more data; ETAS described as "brittle" to fit at scale — [UCSC news](https://news.ucsc.edu/?p=56772)
- CL-ETAS preprint: ConvLSTM+ETAS hybrid; ETAS underestimates event counts for Hector Mine, Baja, Ridgecrest sequences — [arXiv 2310.02574](https://arxiv.org/pdf/2310.02574)

### Inferences
- ETAS fitting fragility is a practical warning for Hawkes on alert data: use constrained parameters (branching ratio < 1), multiple restarts.

### Gaps
- No peer-reviewed paper found that uses ETAS intensity explicitly as a GBDT feature in seismology.

## Point-process intensity as a feature in gradient boosting (stacking)

### Takeaway
No published study found that feeds fitted Hawkes intensity into LightGBM/XGBoost; related hybrids (boosted intensity, LSTM-Hawkes) exist and show gains, so stacking is plausible but unproven.

### Cited Findings
- Boost-R: gradient-boosted additive trees estimate cumulative intensity of recurrent events with static/dynamic features — [IISc ePrints](https://eprints.iisc.ac.in/84732)
- LSTM-Hawkes hybrid for ad clicks reduced MSE by 27.1–83.8% vs Hawkes Intensity Process — [PLOS ONE via RePEc](https://ideas.repec.org/a/plo/pone00/0232887.html)
- Hawkes with feature-dependent base intensity (text) — [arXiv 2010.11851](https://arxiv.org/pdf/2010.11851)

### Inferences
- Practical recipe: fit multivariate Hawkes on train window only; compute per-region λ(t) at prediction time (and excitation vs baseline components, and integrated intensity over 15/30/60-min horizon -> P = 1-exp(-∫λ)); add as LightGBM features. Leakage risk: intensity must use only events strictly before t and parameters fit out-of-fold. Cheap proxy: exponentially decayed counts of past alerts per region with several decay rates (these are Hawkes sufficient statistics) — likely captures most of the gain.

### Gaps
- No empirical evidence of stacking gains for this exact setup; must be validated on the project's data.
