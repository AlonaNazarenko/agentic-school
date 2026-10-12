# Sharpening air raid alert start forecasts across 24 oblasts

The evidence points to three main improvements. First, keep LightGBM as the core model and make it better informed rather than replacing it. Second, add leakage-safe "strike-regime" features: days since the last mass salvo, lagged Air Force daily totals, and precursors such as strategic-bomber takeoffs that are timestamped at publication. Third, fix the evaluation: rolling-origin folds, base-rate-normalized PR-AUC, proper scoring rules, and event-level lead-time metrics. Your 0.255 mean per-oblast PR-AUC at 30 minutes has **no published comparator**. The only directly comparable ML paper predicts alert *state* ("alert active in next 5 min"), not alert onset, and it reports no numeric metrics in its text ([arXiv 2411.14625](https://arxiv.org/html/2411.14625v1)). Your start-event benchmark is therefore new ground, and persistence-inflated numbers from elsewhere should not be used as targets. For the planned upgrades: Hawkes processes have strong records for clustered events ([Mohler et al. 2015](https://mshort9.math.gatech.edu/papers/MohlerEtAl-2015-JASA-Predictive-InPress.pdf)), but nobody has published a test that feeds Hawkes intensity into gradient boosting. Spatio-temporal GNNs show no evidence of beating boosted trees on small, rare-event graphs ([SimST](https://arxiv.org/pdf/2301.12603)). The biggest non-technical risk is the Telegram kpszsu plan: Telegram's API terms prohibit using platform data to train ML models ([Telegram API Terms](https://core.telegram.org/api/terms)), and Ukrainian law criminalizes spreading certain military movement information under martial law ([DonNU](https://r.donnu.edu.ua/handle/123456789/2595?locale=en)).

*Verification note:* claims marked **[partial]** come from search snippets, secondary summaries, or sources whose primary page could not be fetched.

## Prior work has forecast alert state, never alert starts

Demian and Bohdan Pavlyshenko (Nov 2024) is the closest study ([arXiv](https://arxiv.org/html/2411.14625v1)):

- **Data and setup:** Klymenko's dataset on a 1-minute grid from March 2022 to November 2024, with **one Random Forest per region** for only four oblasts.
- **Features:** lagged alert states of all oblasts, current alert duration, calendar features, and a trend term.
- **Validation:** a time split at July 2024. Accuracy and ROC-AUC are shown only in figures, and some figure labels appear mislabelled.
- **Findings:** neighbouring-region features matter most, seasonality matters, and drift is real.
- **Telegram channel:** the authors also ran a channel that predicted alert starts from monitoring-channel text using NLP heuristics, with no published accuracy **[partial: from search summary]** ([arXiv](https://arxiv.org/abs/2411.14625)).

Their findings confirm what your LightGBM already encodes. They add nothing on external precursors, which is where the novel gains are.

Other prior work is behavioural or descriptive. Van Dijcke, Wright and Polyak (PNAS 2023) used more than 3,000 alerts and pings from 17M devices. They documented **alert fatigue** and estimated that **35–45% of civilian casualties were averted** ([PNAS PDF](https://harris.uchicago.edu/files/pnas.2220160120.pdf)). An unverified LinkedIn claim describes a production LightGBM producing 24-hour alert probabilities for 25 regions **[partial]** ([LinkedIn](https://ua.linkedin.com/in/yevhenii-opria)).

**Data sources:**

- **Historical labels:** Vadimkin's dataset (MIT licence, updated daily) holds official and volunteer eTryvoga alerts ([GitHub](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset)).
- **Live data from alerts.in.ua:**
  - oblast, raion and hromada status (A = full, P = partial, N = none)
  - **threat tags** such as strategic aviation, MiG-31K, ballistic, cruise, drones and guided bombs
  - history limited to `month_ago`, and the service says it is not for critical use ([alerts.in.ua](https://devs.alerts.in.ua/))
- **Consequence:** threat tags must be **logged prospectively**. Partial (P) status also means your oblast "start" label depends on how raion alerts are aggregated, so document that rule for any comparison.

## Your planned upgrades have uneven evidence

**Hawkes processes.** Self-exciting point processes have the strongest pedigree:

- The 2011 JASA crime model combines a spatial background with near-repeat excitation ([Georgia Tech PDF](https://mshort9.math.gatech.edu/papers/crime3.pdf)).
- In randomized field trials, ETAS forecasts predicted **1.4–2.2x as much crime as human analysts** and cut crime 7.4% per unit of patrol time ([preprint](https://mshort9.math.gatech.edu/papers/MohlerEtAl-2015-JASA-Predictive-InPress.pdf)).
- Porter and White's self-exciting **hurdle** model for terrorism treats P(zero) separately from activity on attack days, because the zeros come from a different process ([arXiv 1203.3680](https://arxiv.org/pdf/1203.3680)).

The hurdle idea maps well onto alert starts, where many oblasts often trigger at once. Simultaneous starts break simple Hawkes assumptions, so model "wave onset" separately from "which oblasts join". This is my inference.

**Hawkes-to-LightGBM stacking is unproven.** No published study feeds fitted Hawkes intensity into LightGBM. Related hybrids exist:

- gradient-boosted intensity (Boost-R) ([IISc](https://eprints.iisc.ac.in/84732))
- an LSTM-Hawkes model that cut MSE by 27–84% for ad clicks ([RePEc](https://ideas.repec.org/a/plo/pone00/0232887.html))

The cheap proxy is exponentially decayed past-start counts per oblast at several decay rates. These are the Hawkes sufficient statistics, and they likely capture most of the gain (inference).

**Hawkes tooling is fragile.** The `tick` library documents only "Python 3.5 or newer", has CI disabled and 78 open issues ([GitHub](https://github.com/X-DataInitiative/tick)). Seismology also warns that fitting is brittle:

- ETAS is described as "brittle" at scale ([UCSC](https://news.ucsc.edu/?p=56772)) and underestimates counts in large sequences ([arXiv 2310.02574](https://arxiv.org/pdf/2310.02574)).
- ML matches ETAS on 24-hour aftershock counts in seconds rather than hours ([BGS](https://www.bgs.ac.uk/news/new-research-shows-ai-earthquake-tools-forecast-aftershock-risk-in-seconds/)).

Constrain the branching ratio below 1 and use multiple restarts.

**Neural point processes have weaker evidence.** EasyTPP (ICLR 2024) is the standard benchmark library ([arXiv 2307.08097](https://arxiv.org/abs/2307.08097)). An independent study across 15 datasets found results depend heavily on setup and that neural TPP mark distributions are **miscalibrated** ([TMLR 2023](https://arxiv.org/abs/2306.17066v2)).

**GNNs have the weakest case.** STGCN's gains came from dense traffic benchmarks ([IJCAI](https://www.ijcai.org/proceedings/2018/0505)). A model without message passing matched STGNNs at up to 39x throughput ([SimST](https://arxiv.org/pdf/2301.12603)), and gradient boosting was competitive in Traffic4cast 2022 ([arXiv 2211.00157](https://arxiv.org/pdf/2211.00157)). With 24 nodes and adjacency features already present, a GNN at most learns the adjacency, so use it only as an ensemble member.

**Operational conflict forecasters point the same way.** ViEWS, ACLED CAST and conflictforecast.org all use tree ensembles with lagged violence and spillover, evaluated on true-future windows:

- Simple weighted ensembles beat the average single model ([PRIO](https://www.prio.org/publications/13337)).
- Onsets after calm periods are "the hard problem" ([Mueller & Rauh](https://www.econ.cam.ac.uk/research/pub-abstracts?pub=1579)).
- CAST misses most during sudden escalations ([ACLED](https://acleddata.com/wp-content/data-viz/cast-report/october/report.html)).

## Strike-cycle and precursor data carry real but leak-prone signal

**Mass-strike rhythm is the most defensible new feature family.**

- CSIS reports an average **8.2-day gap between major salvos in 2025**, ranging from 2 to 38 days.
- Average salvo size grew from about 100 weapons in 2022 to about 370 in 2025 ([CSIS](https://www.csis.org/analysis/russias-massed-strikes-strategy-coercion-salvo)).
- The Air Force said the stockpile supports roughly weekly mass strikes ([NV](https://english.nv.ua/russian-war/russia-can-launch-massive-strikes-weekly-despite-factory-problems-air-force-says-50564585.html)).
- Institute for Science and International Security reports describe **24–32-hour multi-wave cycles** running into daylight in 2026, and daily Shahed averages swinging between about 260 and 151 within June 2026 ([ISIS-online](https://isis-online.org/uploads/isis-reports/documents/June-2026-Updated-Analysis-of-Russian-Shahed-type-UAVs-Deployment-Against-Ukraine-up-to-July-6.pdf)).

"Days since last salvo above N", rolling launch counts, and wave-phase indicators are hazard features that a 5-minute model cannot learn from alert history alone.

**Air Force messages come in two kinds, with different leakage rules.**

- **Morning summaries** give drones and missiles launched, downed and "lost" ([MarketScreener/Reuters](https://ca.marketscreener.com/quote/currency/US-DOLLAR-RUSSIAN-ROUBLE--2370597/news/Ukraine-says-it-destroys-28-Russian-drones-overnight-48546658/)). They describe the night that has just ended and get revised ([ISIS-online](https://isis-online.org/uploads/isis-reports/documents/June-2026-Updated-Analysis-of-Russian-Shahed-type-UAVs-Deployment-Against-Ukraine-up-to-July-6.pdf)). Join them by message publication time, never by event date.
- **Tu-95MS takeoff notices** come with launch-window estimates hours ahead ([UP](https://www.pravda.com.ua/eng/news/2024/01/8/7436250/)). Some flights simulate launches or return to base ([Censor.net](https://censor.net/en/news/3520072/aktyvnist_rosiyiskoyi_aviatsiyi)). These are genuine 60-minute-horizon precursors.
- **MiG-31K takeoffs** trigger nationwide alerts immediately ([UP](https://www.pravda.com.ua/eng/news/2023/11/06/7427447/)). They are part of the label and have zero lead time.
- **Reporting itself drifts.** In August 2026 the Air Force stopped reporting ballistic missile numbers in morning reports ([Ukranews](https://ukranews.com/en/news/1168459-ihnat-confirms-that-ukrainian-air-forces-change-format-of-reports-on-air-attacks-by-occupation)). That falls in your test window, so any ballistic-count feature would break silently.
- **Channel handle unverified:** t.me/kpszsu could not be confirmed as the official Air Force channel from a primary page **[partial]**.

**Volunteer monitoring channels** give launch-line and arrival-window estimates ([Censor.net](https://censor.net/en/news/3501616/strategic_bombers_take_off_in_russia)). AFP describes them as giving "less certainty, shorter warning" ([AFP](https://english.aawsat.com/node/5307591)). They became vaguer after official warnings ([Kyiv Post](https://kyivpost.com/post/14817)), so text features are non-stationary.

**Slow sources only work as lagged regime covariates:**

| Source | Cadence and timing | Best use |
|---|---|---|
| FIRMS fire hotspots | about 3 h after satellite pass ([Earthdata](https://www.earthdata.nasa.gov/learn/find-data/near-real-time/firms?page=1)) | mostly damage after an attack, not a precursor |
| ACLED | weekly, on Wednesdays ([ACLED](https://acleddata.com/use-access/when-are-acled-data-updated)) | weekly lagged intensity; keep snapshot versions |
| ISW daily assessment | publication time unverified **[partial]** | days after publication only |

**Weather** is cited by Defence Intelligence as a factor in choosing attack dates ([ThePage](https://thepage.ua/news/raketnye-obstrely-ukrainy-kogda-zhdat-novuyu-ataku-raket-i-shahed-rf)), but there is no quantitative study. Open-Meteo provides hourly cloud, wind and precipitation, with ERA5 arriving on a 5-day delay ([Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api)). Reanalysis weather is an oracle feature: realistic use needs forecasts archived before the cutoff.

**Moon phase** has no evidence for or against it. It is free and cannot leak, so test it.

## Evaluation must reward calibration and lead time, not one test month

**Imbalance and calibration.** LightGBM's docs say `is_unbalance` and `scale_pos_weight` can raise the performance metric but give "**poor estimates of the individual class probabilities**" ([LightGBM](https://lightgbm.readthedocs.io/en/latest/Parameters.html)). Undersampling biases posteriors without changing ranking, and the correction is p = βpₛ/(βpₛ − pₛ + 1) ([Dal Pozzolo et al.](https://ieeexplore.ieee.org/document/7376606)). Because PR-AUC only measures ranking, reweighting will barely move it. To publish probabilities:

- recalibrate on a later time block
- use **sigmoid** calibration, which suits small samples, rather than isotonic, which needs about 1,000+ samples ([scikit-learn](https://scikit-learn.org/stable/modules/calibration.html))
- pool calibrators across oblasts, because per-oblast positives are scarce

**Metrics.** PR-AUC is the right ranking metric under imbalance, but it shifts with prevalence ([Saito & Rehmsmeier](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432)). A "mean per-oblast PR-AUC of 0.255" therefore mixes oblasts with very different base rates. Report each oblast's AP/prevalence lift alongside the macro mean. Add log loss and Brier skill against climatology, with the reliability decomposition, because a lower Brier score alone does not prove calibration ([scikit-learn](https://scikit-learn.org/stable/modules/calibration.html)).

**Validation design.**

- Use **rolling-origin** folds, not a single Aug–Sep holdout ([fpp3](https://otexts.com/fpp3/tscv.html)). Blocked CV gives more robust model selection than a last-block holdout ([Bergmeir & Benítez](https://research.monash.edu/en/publications/on-the-use-of-cross-validation-for-time-series-predictor-evaluati/)).
- Add an embargo at least as long as the feature lookback plus the horizon. The purging/embargo citation to López de Prado was not verified **[partial]**.
- Test model differences with **Diebold-Mariano plus the HLN correction** on daily loss differences ([statsmodels](https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.diebold_mariano_test.html)). DM loses power under unstable environments ([Nottingham](https://nottingham-repository.worktribe.com/output/40565499/testing-for-equal-average-forecast-accuracy-in-possibly-unstable-environments)), so also use a day-block bootstrap for PR-AUC deltas.

**Drift.** Gama et al. recommend windowing and forgetting ([ACM CSUR 2014](https://research.tue.nl/en/publications/a-survey-on-concept-drift-adaptation/)). Your recency weights already do this. Tune the decay inside the walk-forward loop and monitor feature PSI.

**Operational metrics from earthquake early warning.**

- Score forecasts as **matched, missed, false or late** ([SCEC](https://central.scec.org/node/5061)).
- Count a late warning as a miss, and avoid metrics that depend on true negatives ([arXiv 2009.06316](https://arxiv.org/pdf/2009.06316)).
- ShakeAlert's historical false/missed rates for M5+ were 8%/16% ([Caltech](https://authors.library.caltech.edu/records/dz4m3-02r27)).
- Accepting more false alerts optimizes overall performance ([USGS](https://pubs.usgs.gov/publication/70215102); journal and DOI metadata inconsistent **[partial]**).

For your system, report hit rate within the horizon, median lead time, and false alarms per oblast per week at fixed thresholds.

## Telegram training and public forecasts face legal limits

**Telegram API terms.** §1.5 prohibits "using, accessing or aggregating data obtained from the Telegram platform to train" AI/ML models. Violators lose API access after 10 days ([Telegram API Terms](https://core.telegram.org/api/terms)). A Telethon scraper feeding LightGBM falls squarely within that clause.

**Ukrainian criminal law.** Article 114-2 was added by Law 2160-IX in March 2022. It criminalizes unauthorized dissemination, under martial law, of information on weapon movements and Armed Forces deployment ([DonNU](https://r.donnu.edu.ua/handle/123456789/2595?locale=en)). A study reviewed 112 guilty verdicts from 2022–2023 ([VNU](https://evnuir.vnu.edu.ua/bitstream/123456789/25712/1/2788-6018.2024.05.115.pdf)). The Rada text returned 403, so the exact wording and penalties are **[partial]**.

**Official guidance on tracking channels.** The Air Force's 2023 concern was real-time radar screenshots and flight paths that reveal **air-defence positions**. The spokesman later clarified there was no ban on established channels ([Ukrainska Pravda](https://pravda.com.ua/eng/news/2023/03/22/7394545); [Kyiv Post](https://kyivpost.com/post/14817)).

**What this means for your forecaster (inference, not legal advice):**

- A start-probability forecaster built from public alert history carries low risk.
- Explanations or features that reveal where threats disappear carry more risk.
- Never frame the forecast as a substitute for official alerts. The PNAS fatigue finding ([PNAS](https://harris.uchicago.edu/files/pnas.2220160120.pdf)) suggests noisy public forecasts could erode compliance.
- No official guidance on publishing probabilistic alert forecasts was found.

## Prioritized improvements

Impact estimates are my judgement, not measured results.

| # | Improvement | Expected impact | Effort | Leakage notes |
|---|---|---|---|---|
| 1 | Rolling-origin CV (e.g. 6+ monthly folds) with embargo; report AP/prevalence per oblast, Brier skill, log loss, DM-HLN plus block bootstrap | High (trustworthy model selection; stops overfitting to Aug–Sep) | Low | Embargo ≥ lookback + horizon; calibrators fit on folds after training |
| 2 | Multi-scale exponentially decayed start counts (own, neighbours, macro-region, national) as Hawkes-proxy features | Medium–high | Low | Use only events strictly before t |
| 3 | Strike-regime features: days since last mass salvo, rolling 1/3/7-day launch totals from Air Force morning summaries | Medium–high, especially for 60 min and quiet-period onsets | Medium (parse summaries) | Join by message publish time, not event date; store edit_date; ballistic counts vanish from Aug 2026 |
| 4 | Sigmoid calibration (pooled, with an oblast offset); ablate scale_pos_weight ∈ {1, √ratio, ratio} | Large gain in log loss and Brier; small change in PR-AUC | Low | Calibrate on a time block after training |
| 5 | Event-level metrics: hit rate, median lead time, false alarms per oblast per week | High for usability | Low | Count late warnings as misses |
| 6 | Prospectively log alerts.in.ua threat tags and raion P/A status; add strategic-aviation-takeoff flags | Medium–high for 30/60 min | Medium (logging pipeline; months of data before use) | History is only one month, so no retroactive backfill; MiG-31K is part of the label |
| 7 | Multivariate Hawkes (24-dim, exponential kernel, time-of-day baseline, branching < 1) fit on train only; feed λ(t) and the integrated horizon probability into LightGBM | Uncertain; likely incremental over #2 | Medium–high (`tick` on Python 3.12 risky; custom MLE fallback) | Fit parameters out-of-fold; intensity uses only past events |
| 8 | Hurdle / two-stage model: P(national wave starts), then P(oblast joins the wave) | Medium | Medium | Wave definition must use only past data at t |
| 9 | Weather from archived forecasts and moon illumination | Low–uncertain | Low | ERA5 reanalysis is an oracle; use forecasts issued before the cutoff |
| 10 | Simple weighted ensemble (LightGBM + Hawkes + optionally a GNN) | Low–medium | Medium | Learn weights on a later fold |
| 11 | GNN as a standalone replacement | Low (no supporting evidence) | High | Fixed vs learned adjacency must not use future co-occurrence |
| 12 | Telegram kpszsu / monitoring-channel text features | Potentially high for nowcasting | High | **Blocked pending legal review** (Telegram API ToS §1.5; Art. 114-2 sensitivities); text non-stationary; exclude posts that echo the alert |

## Conclusion

The research shifts the question from "which architecture" to "which information arrives before the alert, and when was it knowable". The model already captures historical alert dynamics well. The remaining signal sits in two places. One is the multi-day salvo rhythm, which explains *whether* tonight is a wave night. The other is real-time precursors, which explain *when* the wave reaches an oblast. Both require timestamp discipline that matters more than any modelling choice, especially because Ukraine changed its reporting format inside your test window.

Two consequences follow. First, the 0.255 headline should be restated as per-oblast lift over prevalence across several folds, before any upgrade is judged. Second, the most promising data source (Telegram) is also the one most constrained by its terms of service and by martial-law sensitivities. Official, licensed alert feeds logged going forward are the safer route to precursor features.
